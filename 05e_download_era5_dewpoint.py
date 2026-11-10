"""
05e_download_era5_dewpoint.py
==============================
Download 2m dewpoint temperature (d2m) from ERA5 derived daily statistics
for the WMO baseline period (1991-2020) and the study period (2019-2021),
then extend the pipeline with wet-bulb temperature event thresholds.

Scientific motivation
---------------------
Dry-bulb temperature alone cannot distinguish hot-dry from hot-humid
conditions. Two cities at identical Tmax can face very different physiological
heat loads and cooling-energy demands depending on atmospheric humidity.
Wet-bulb temperature (Tw) integrates heat and humidity into a single metric
and is the key variable for "humid heat" research (Sherwood & Huber 2010;
Raymond et al. 2020; Im et al. 2017).

Two new daily wet-bulb variables per city
------------------------------------------
  tw_max  — wet-bulb temperature paired with Tmax (daytime humid heat load)
             uses (Tmax, d2m_min) because both daily maxima typically occur
             in the afternoon when dewpoint is at its diurnal minimum
  tw_min  — wet-bulb temperature paired with Tmin (overnight recovery)
             uses (Tmin, d2m_max) because both daily minima / maxima of
             their respective variables occur in early morning

Wet-bulb formula (Stull 2011)
------------------------------
Valid for 5% ≤ RH ≤ 99%, −20°C ≤ T ≤ 50°C (±0.35°C accuracy):

  Tw = T · atan(0.151977 · √(RH + 8.313659))
       + atan(T + RH)
       − atan(RH − 1.676331)
       + 0.00391838 · RH^1.5 · atan(0.023101 · RH)
       − 4.686035

Relative humidity from August-Roche-Magnus (Bolton 1980):
  RH = 100 · exp(17.625 · Td / (243.04 + Td)) / exp(17.625 · T / (243.04 + T))

New threshold columns added to threshold_table_doy.parquet
-----------------------------------------------------------
  tw_max_p90(city, DOY) — 90th percentile of tw_max, ±15-day DOY window,
                           1991-2020 baseline (same method as tmax_p90)
  tw_min_p90(city, DOY) — 90th percentile of tw_min (warm-night threshold)

New event types enabled in 04_define_events.py after this script runs
----------------------------------------------------------------------
  humid_heat_event — tw_max > tw_max_p90 for ≥3 consecutive days
  warm_night_event — tw_min > tw_min_p90 for ≥2 consecutive days
  hot_dry_event    — heat_event AND NOT humid_heat_event
  hot_humid_event  — heat_event AND humid_heat_event

CDS requests
-------------
  Dataset  : derived-era5-single-levels-daily-statistics
  Variable : 2m_dewpoint_temperature
  Stats    : daily_max + daily_min (one request per statistic per chunk)
  Baseline : 30 years × 2 statistics = 60 annual CDS requests  (~2 days)
  Study    :  3 years × 2 statistics =  6 CDS requests         (~3–6 hours)

  Same adaptive chunking as 05c (tries 12→6→3→1 months if CDS rejects).

Outputs
-------
  ERA5/dewpoint/d2m_daily_{year}.nc          (2019–2021, consumed by 05d)
      vars: d2m_max, d2m_min  (°C)
  ERA5/baseline/threshold_table_doy.parquet  (PATCHED: adds tw_max_p90,
                                              tw_min_p90 columns)
  ERA5/baseline/threshold_table_doy.backup_pre_d2m.parquet  (safety backup)

Run
---
  python 05e_download_era5_dewpoint.py               # baseline + study period
  python 05e_download_era5_dewpoint.py --only-study  # 6 CDS requests, ~3h
  python 05e_download_era5_dewpoint.py --only-base   # 60 CDS requests, ~2 days
  python 05e_download_era5_dewpoint.py --patch-only  # recompute thresholds
                                                      # from existing NetCDFs
"""

import argparse
import calendar
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
ERA5_DIR     = BASE / "ERA5"
BASELINE_DIR = ERA5_DIR / "baseline"
DEWPOINT_DIR = ERA5_DIR / "dewpoint"
SHPFILE      = BASE / "ShapefileData" / "FUA_cities_final.shp"

BASELINE_YEARS = list(range(1991, 2021))   # 30-year WMO reference period
STUDY_YEARS    = [2019, 2020, 2021]
MONTHS         = [f"{m:02d}" for m in range(1, 13)]

# CDS settings — mirrors 05c exactly
_DATASET_DERIVED = "derived-era5-single-levels-daily-statistics"
CDS_VARIABLE     = "2m_dewpoint_temperature"
DERIVED_STATISTICS = {
    "daily_max": "d2m_max",
    "daily_min": "d2m_min",
}
GRID_DEG = 0.25   # must match 05c / 05d

# Threshold parameters — identical to 05c
DOY_WINDOW   = 15   # ±days for pooling observations
HUMID_HT_PCT = 90   # percentile for tw_max_p90
WARM_NT_PCT  = 90   # percentile for tw_min_p90


# ─────────────────────────────────────────────────────────────────────────────
# WET-BULB PHYSICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """
    Relative humidity (%) from dry-bulb T and dewpoint Td (both °C).
    August-Roche-Magnus approximation (Bolton 1980).
    Returned values are clipped to [1, 100] % to guard against
    numerical artefacts where d2m slightly exceeds t2m.
    """
    gamma_t  = 17.625 * t_c  / (243.04 + t_c)
    gamma_td = 17.625 * td_c / (243.04 + td_c)
    rh = 100.0 * np.exp(gamma_td - gamma_t)
    return np.clip(rh, 1.0, 100.0).astype(np.float32)


def stull_wetbulb(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Wet-bulb temperature (°C) via Stull (2011).
    Valid for 5% ≤ RH ≤ 99%, −20°C ≤ T ≤ 50°C.
    RH is internally clipped to [5, 99] %.
    Reference: Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity
               and Air Temperature. J. Appl. Meteor. Climatol., 50, 2267-2269.
    """
    rh_c = np.clip(rh, 5.0, 99.0)
    tw = (
        t_c * np.arctan(0.151977 * np.sqrt(rh_c + 8.313659))
        + np.arctan(t_c + rh_c)
        - np.arctan(rh_c - 1.676331)
        + 0.00391838 * rh_c ** 1.5 * np.arctan(0.023101 * rh_c)
        - 4.686035
    )
    return tw.astype(np.float32)


def compute_wetbulb_daily(
    tmax_c:   np.ndarray,
    tmin_c:   np.ndarray,
    d2m_max:  np.ndarray,
    d2m_min:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute daytime (tw_max) and nighttime (tw_min) wet-bulb temperatures.

    Pairing rationale for daily-aggregated data:
      tw_max: Tmax and d2m_min both typically occur in the afternoon
              → (Tmax, d2m_min) represents afternoon humid-heat conditions
      tw_min: Tmin and d2m_max both typically occur in early morning
              → (Tmin, d2m_max) represents overnight recovery conditions

    Returns:
        tw_max  — daytime wet-bulb (°C), shape = tmax_c.shape
        tw_min  — overnight wet-bulb (°C), shape = tmin_c.shape
    """
    rh_day   = compute_rh(tmax_c, d2m_min)   # afternoon pairing
    rh_night = compute_rh(tmin_c, d2m_max)   # morning pairing
    tw_max   = stull_wetbulb(tmax_c, rh_day)
    tw_min   = stull_wetbulb(tmin_c, rh_night)
    return tw_max, tw_min


# ─────────────────────────────────────────────────────────────────────────────
# CDS UTILITIES  (replicated from 05c; parameterised for d2m)
# ─────────────────────────────────────────────────────────────────────────────
def get_cds_client():
    try:
        import cdsapi
        return cdsapi.Client()
    except ImportError:
        sys.exit('pip install "cdsapi>=0.7.7"')
    except Exception as e:
        sys.exit(f"CDS initialisation error: {e}\nCheck ~/.cdsapirc")


def retrieve_and_unzip(client, request: dict, out: Path):
    """Atomically download one CDS request; unzip archive if necessary."""
    download = out.with_suffix(".download")
    partial  = out.with_suffix(".part.nc")
    download.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    try:
        client.retrieve(_DATASET_DERIVED, request, str(download))
        if not download.exists() or download.stat().st_size == 0:
            raise RuntimeError(f"CDS returned an empty file for {out.name}")
        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as archive:
                members = [m for m in archive.namelist()
                           if m.lower().endswith((".nc", ".nc4", ".netcdf"))]
                if len(members) != 1:
                    raise RuntimeError(
                        f"Expected 1 NetCDF in archive, got {len(members)}: {members}"
                    )
                with archive.open(members[0]) as src, partial.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            download.replace(partial)
        partial.replace(out)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        download.unlink(missing_ok=True)


def is_cost_limit_error(exc: Exception) -> bool:
    """Detect CDS rejections for requests that exceed the cost/size envelope."""
    text = str(exc).lower()
    return "cost limits exceeded" in text or "request is too large" in text


def month_groups(group_size: int) -> list[list[str]]:
    """Split MONTHS into consecutive chunks (12, 6, 3, or 1 months per group)."""
    return [MONTHS[i:i + group_size] for i in range(0, len(MONTHS), group_size)]


def derived_request(year: int, months: list[str], statistic: str) -> dict:
    """Build a CDS derived-daily-statistics request for 2m_dewpoint_temperature."""
    return {
        "product_type":    "reanalysis",
        "variable":        [CDS_VARIABLE],          # 2m_dewpoint_temperature
        "year":            str(year),
        "month":           months,
        "day":             [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": statistic,
        "time_zone":       "utc+00:00",
        "frequency":       "1_hourly",
        "grid":            [GRID_DEG, GRID_DEG],
        "data_format":     "netcdf",
        "download_format": "unarchived",
    }


def find_time_name(obj):
    for name in ("valid_time", "time", "date"):
        if name in obj.coords:
            return name
    raise RuntimeError(
        f"No time coordinate found. Coords: {list(obj.coords)}; dims: {list(obj.dims)}"
    )


def load_daily_stat_dataset(path: Path, output_name: str, statistic: str):
    """Load a derived-stats NetCDF and rename its single data variable."""
    import xarray as xr
    with xr.open_dataset(path) as src:
        time_name = find_time_name(src)
        candidates = [v for v, da in src.data_vars.items()
                      if time_name in da.dims and da.dtype.kind in "iuf"]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot identify variable in {path.name}; "
                f"data_vars: {list(src.data_vars)}"
            )
        ds = src[[candidates[0]]].rename({candidates[0]: output_name}).load()
        ds[output_name].attrs["daily_statistic"] = statistic
        return ds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load city coordinates
# ─────────────────────────────────────────────────────────────────────────────
def load_city_coords() -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("pip install geopandas")
    print(f"  Loading {SHPFILE.name} …")
    gdf = gpd.read_file(SHPFILE)
    gdf["city_id"] = gdf.index.astype(str).str.zfill(4)   # must match 05c
    gdf["lon"]     = gdf.geometry.centroid.x
    gdf["lat"]     = gdf.geometry.centroid.y
    keep = ["city_id", "city", "country", "lon", "lat"]
    coords = gdf[[c for c in keep if c in gdf.columns]].copy()
    print(f"  {len(coords)} cities loaded")
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Download one year of d2m (adaptive chunking: 12→6→3→1 months)
# ─────────────────────────────────────────────────────────────────────────────
def download_d2m_year(client, year: int) -> Path:
    """
    Download annual d2m_max + d2m_min using the same adaptive chunking strategy
    as 05c: try a full-year request first (2 CDS calls), fall back to 6-month,
    3-month, and 1-month chunks if CDS rejects for exceeding cost limits.

    Returns the merged annual NetCDF path in ERA5/dewpoint/.
    """
    import xarray as xr

    out = DEWPOINT_DIR / f"d2m_daily_{year}.nc"
    if out.exists() and out.stat().st_size > 0:
        print(f"    {year}: already downloaded ({out.stat().st_size / 1e6:.0f} MB) — skip")
        return out

    for group_size, label in [(12, "annual"), (6, "half-year"),
                               (3, "quarterly"), (1, "monthly")]:
        groups     = month_groups(group_size)
        temp_files = []
        loaded     = []
        combined   = None

        try:
            # ── Download all chunks for both statistics ───────────────────────
            for statistic, output_name in DERIVED_STATISTICS.items():
                for months in groups:
                    month_label = f"{months[0]}-{months[-1]}"
                    chunk = DEWPOINT_DIR / f"_temp_{year}_{output_name}_{month_label}.nc"
                    temp_files.append(chunk)
                    if not chunk.exists() or chunk.stat().st_size == 0:
                        print(f"    {year} {output_name} {month_label}: "
                              f"submitting {label} CDS request …")
                        t0 = time.time()
                        retrieve_and_unzip(client, derived_request(year, months, statistic), chunk)
                        elapsed = time.time() - t0
                        print(f"    {year} {output_name} {month_label}: "
                              f"{chunk.stat().st_size / 1e6:.0f} MB ({elapsed / 60:.0f} min)")

            # ── Merge chunks into one annual dataset per statistic ────────────
            stat_datasets = []
            for statistic, output_name in DERIVED_STATISTICS.items():
                parts = []
                for months in groups:
                    month_label = f"{months[0]}-{months[-1]}"
                    chunk = DEWPOINT_DIR / f"_temp_{year}_{output_name}_{month_label}.nc"
                    part = load_daily_stat_dataset(chunk, output_name, statistic)
                    parts.append(part)
                    loaded.append(part)
                time_dim = find_time_name(parts[0])
                stat_ds  = xr.concat(parts, dim=time_dim).sortby(time_dim)
                stat_datasets.append(stat_ds)
                loaded.append(stat_ds)

            combined = xr.merge(stat_datasets)

            # ── Write merged annual NetCDF ─────────────────────────────────────
            partial = out.with_suffix(".part.nc")
            partial.unlink(missing_ok=True)
            combined.to_netcdf(
                partial,
                engine="netcdf4",
                encoding={
                    name: {"zlib": True, "complevel": 4, "dtype": "float32"}
                    for name in combined.data_vars
                },
            )
            partial.replace(out)

            # Clean up chunk files
            for chunk in temp_files:
                chunk.unlink(missing_ok=True)

            size_mb = out.stat().st_size / 1e6
            print(f"    {year}: merged → {out.name}  ({size_mb:.0f} MB, {label} chunks)")
            return out

        except Exception as exc:
            if is_cost_limit_error(exc) and group_size > 1:
                print(f"    {year}: {label} chunk exceeded CDS cost limit "
                      f"→ trying smaller chunks")
                for chunk in temp_files:
                    chunk.unlink(missing_ok=True)
                continue
            raise

        finally:
            if combined is not None:
                try:
                    combined.close()
                except Exception:
                    pass
            for ds in loaded:
                try:
                    ds.close()
                except Exception:
                    pass

    raise RuntimeError(f"Could not download {year}: even monthly chunks failed.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Extract city d2m pixels from one annual NetCDF
# ─────────────────────────────────────────────────────────────────────────────
def extract_city_d2m(nc_path: Path, coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract d2m_max and d2m_min at each FUA city centroid.

    ERA5 dewpoint is stored in Kelvin; values > 100 are converted to °C.
    The lon wrapping fix (% 360) matches 05c and 05d.

    Returns DataFrame: city_id | date | d2m_max_c | d2m_min_c  (°C)
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    lat_name  = next(c for c in ("latitude",  "lat")             if c in ds.coords)
    lon_name  = next(c for c in ("longitude", "lon")             if c in ds.coords)
    time_name = next(c for c in ("valid_time", "time", "date")   if c in ds.coords)

    lats  = ds[lat_name].values
    lons  = ds[lon_name].values
    dates = pd.to_datetime(ds[time_name].values)

    # Detect Kelvin vs °C: ERA5 d2m is in K (~250-310 K), so mean > 100
    sample_max = float(ds["d2m_max"].isel({time_name: 0, lat_name: 360, lon_name: 720}).values)
    is_kelvin  = sample_max > 100.0

    records = []
    for _, row in coords_df.iterrows():
        lat_idx      = int(np.argmin(np.abs(lats - float(row["lat"]))))
        city_lon_adj = float(row["lon"]) % 360
        dists_raw    = np.abs(lons - city_lon_adj)
        dists_wrap   = np.abs(lons - (city_lon_adj - 360))
        lon_idx      = int(np.argmin(np.minimum(dists_raw, dists_wrap)))

        d2m_max_vals = ds["d2m_max"].isel({lat_name: lat_idx, lon_name: lon_idx}).values.astype(np.float32)
        d2m_min_vals = ds["d2m_min"].isel({lat_name: lat_idx, lon_name: lon_idx}).values.astype(np.float32)

        if is_kelvin:
            d2m_max_vals -= 273.15
            d2m_min_vals -= 273.15

        records.append(pd.DataFrame({
            "city_id":   row["city_id"],
            "date":      dates,
            "d2m_max_c": d2m_max_vals,
            "d2m_min_c": d2m_min_vals,
        }))

    ds.close()
    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Build baseline wet-bulb series (1991-2020)
# ─────────────────────────────────────────────────────────────────────────────
def build_wetbulb_baseline(
    coords_df:   pd.DataFrame,
    patch_only:  bool = False,
) -> pd.DataFrame:
    """
    For each baseline year:
      1. Download d2m (or load existing NetCDF if --patch-only)
      2. Extract city d2m pixels
      3. Merge with temperature from city_baseline_all_years.parquet
      4. Compute tw_max = Stull(Tmax, RH from Tmax + d2m_min)
         and    tw_min = Stull(Tmin, RH from Tmin + d2m_max)
      5. Delete the annual d2m NetCDF to save disk space
         (city pixels have been extracted into memory)

    Returns DataFrame: city_id | date | doy | tw_max | tw_min
    """
    # Load temperature baseline (already generated by 05c)
    temp_path = BASELINE_DIR / "city_baseline_all_years.parquet"
    if not temp_path.exists():
        raise FileNotFoundError(
            f"Temperature baseline missing: {temp_path}\n"
            "Run 05c_download_era5_baseline.py first."
        )
    print(f"  Loading temperature baseline ({temp_path.stat().st_size / 1e6:.0f} MB) …")
    temp_base = pd.read_parquet(temp_path)
    temp_base["date"] = pd.to_datetime(temp_base["date"])
    n_cities = temp_base["city_id"].nunique()
    n_years  = temp_base["date"].dt.year.nunique()
    print(f"  {len(temp_base):,} rows  |  {n_cities} cities  |  {n_years} years")

    client   = None if patch_only else get_cds_client()
    all_wb   = []

    for yi, year in enumerate(BASELINE_YEARS):
        t0 = time.time()
        nc_path = DEWPOINT_DIR / f"d2m_daily_{year}.nc"

        if patch_only:
            if not nc_path.exists():
                print(f"  [{yi+1}/{len(BASELINE_YEARS)}] {year}: "
                      f"d2m_daily_{year}.nc not found — skipping")
                continue
            print(f"  [{yi+1}/{len(BASELINE_YEARS)}] {year}: "
                  f"loading existing NetCDF ({nc_path.stat().st_size / 1e6:.0f} MB)")
        else:
            print(f"\n  [{yi+1}/{len(BASELINE_YEARS)}] {year}: downloading d2m …")
            nc_path = download_d2m_year(client, year)

        # Extract d2m at city centroids
        df_d2m = extract_city_d2m(nc_path, coords_df)
        df_d2m["date"] = pd.to_datetime(df_d2m["date"])

        # Merge with temperature for this year
        temp_yr = temp_base[temp_base["date"].dt.year == year].copy()
        merged  = temp_yr.merge(df_d2m, on=["city_id", "date"], how="inner")

        if len(merged) == 0:
            print(f"  [{year}] WARNING: 0 rows after merge — city_id mismatch?")
            if not patch_only:
                nc_path.unlink(missing_ok=True)
            continue

        # Compute wet-bulb
        tw_max, tw_min = compute_wetbulb_daily(
            merged["tmax_c"].values,
            merged["tmin_c"].values,
            merged["d2m_max_c"].values,
            merged["d2m_min_c"].values,
        )
        merged["tw_max"] = tw_max
        merged["tw_min"] = tw_min

        elapsed = time.time() - t0
        n_days  = merged["date"].nunique()
        print(f"  [{year}] {n_days} days × {merged['city_id'].nunique()} cities | "
              f"tw_max {tw_max.min():.1f}–{tw_max.max():.1f}°C | "
              f"tw_min {tw_min.min():.1f}–{tw_min.max():.1f}°C | "
              f"{elapsed:.0f}s")

        all_wb.append(merged[["city_id", "date", "tw_max", "tw_min"]])

        # Delete baseline d2m NetCDF; city pixels now in memory
        if not patch_only:
            nc_path.unlink(missing_ok=True)
            print(f"  [{year}] d2m NetCDF deleted — pixels extracted")

    if not all_wb:
        raise RuntimeError("No baseline years processed. Check downloads / --patch-only flag.")

    baseline_wb = pd.concat(all_wb, ignore_index=True)
    baseline_wb["doy"] = baseline_wb["date"].dt.dayofyear
    # Map Feb-29 (DOY 366 in leap years) → DOY 60 so it pools with late February
    baseline_wb.loc[baseline_wb["doy"] == 366, "doy"] = 60
    return baseline_wb


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Compute DOY-windowed wet-bulb thresholds
# ─────────────────────────────────────────────────────────────────────────────
def compute_wetbulb_thresholds(baseline_wb: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised per-city DOY-windowed threshold computation.

    Same algorithm as 05c compute_doy_thresholds():
      For each city × DOY, pool all observations within ±DOY_WINDOW days
      across all 30 baseline years using a circular-distance broadcast,
      then compute p90 of tw_max and p90 of tw_min.

    Expected: ~5–15 minutes for 1,583 cities.

    Returns DataFrame: city_id | doy | tw_max_p90 | tw_min_p90 | n_samples_wb
    """
    print(f"\n  Computing ±{DOY_WINDOW}-day DOY-windowed wet-bulb thresholds …")
    print(f"  Baseline: {BASELINE_YEARS[0]}–{BASELINE_YEARS[-1]} "
          f"({len(BASELINE_YEARS)} years)")
    print(f"  Expected samples per city×DOY: ~{(2*DOY_WINDOW+1) * len(BASELINE_YEARS)}")

    target_doys = np.arange(1, 367, dtype=np.int16)
    cities      = baseline_wb["city_id"].unique()
    n_cities    = len(cities)
    t0_total    = time.time()
    all_records = []

    for ci, city_id in enumerate(cities):
        if ci > 0 and ci % 200 == 0:
            elapsed   = time.time() - t0_total
            rate      = ci / elapsed
            remaining = (n_cities - ci) / rate / 60
            print(f"    {ci}/{n_cities} cities … ({remaining:.0f} min remaining)")

        cd = baseline_wb[baseline_wb["city_id"] == city_id]
        obs_doys = cd["doy"].values.astype(np.int16)
        tw_max_a = cd["tw_max"].values.astype(np.float32)
        tw_min_a = cd["tw_min"].values.astype(np.float32)

        # Vectorised circular distance: (N_obs, 366) boolean mask
        abs_diff  = np.abs(obs_doys[:, None] - target_doys[None, :])
        circ_dist = np.minimum(abs_diff, 365 - abs_diff)
        in_window = circ_dist <= DOY_WINDOW                   # (N, 366)

        tw_max_p90 = np.empty(366, dtype=np.float32)
        tw_min_p90 = np.empty(366, dtype=np.float32)
        n_samp     = np.empty(366, dtype=np.int32)

        for j in range(366):
            mask       = in_window[:, j]
            n          = int(mask.sum())
            n_samp[j]  = n
            if n > 0:
                tw_max_p90[j] = np.percentile(tw_max_a[mask], HUMID_HT_PCT)
                tw_min_p90[j] = np.percentile(tw_min_a[mask], WARM_NT_PCT)
            else:
                tw_max_p90[j] = np.nan
                tw_min_p90[j] = np.nan

        all_records.append(pd.DataFrame({
            "city_id":      city_id,
            "doy":          target_doys.astype(int),
            "tw_max_p90":   tw_max_p90,
            "tw_min_p90":   tw_min_p90,
            "n_samples_wb": n_samp,
        }))

    thresholds_wb = pd.concat(all_records, ignore_index=True)
    elapsed_min   = (time.time() - t0_total) / 60
    print(f"  ✓ Wet-bulb thresholds: {len(thresholds_wb):,} city×DOY rows "
          f"({elapsed_min:.1f} min)")
    print(f"    tw_max_p90 range: {thresholds_wb['tw_max_p90'].min():.1f}°C – "
          f"{thresholds_wb['tw_max_p90'].max():.1f}°C")
    print(f"    tw_min_p90 range: {thresholds_wb['tw_min_p90'].min():.1f}°C – "
          f"{thresholds_wb['tw_min_p90'].max():.1f}°C")
    print(f"    Median samples per city×DOY: "
          f"{thresholds_wb['n_samples_wb'].median():.0f}")
    return thresholds_wb


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Patch threshold_table_doy.parquet
# ─────────────────────────────────────────────────────────────────────────────
def patch_threshold_table(thresholds_wb: pd.DataFrame):
    """
    Merge tw_max_p90 and tw_min_p90 into the existing threshold table.

    The original table (from 05c) has columns:
        city_id | doy | tmax_p90 | tmin_p10 | n_samples
    After patching it will have:
        city_id | doy | tmax_p90 | tmin_p10 | n_samples | tw_max_p90 | tw_min_p90

    A backup is created before any modification.
    """
    thr_path = BASELINE_DIR / "threshold_table_doy.parquet"
    if not thr_path.exists():
        raise FileNotFoundError(
            f"Threshold table not found: {thr_path}\n"
            "Run 05c_download_era5_baseline.py first."
        )

    # Safety backup (only create once)
    backup = thr_path.with_suffix(".backup_pre_d2m.parquet")
    if not backup.exists():
        shutil.copy2(thr_path, backup)
        print(f"  Backup → {backup.name}")

    existing = pd.read_parquet(thr_path)
    print(f"  Existing threshold table: shape={existing.shape}  "
          f"cols={list(existing.columns)}")

    # Drop any previous wet-bulb columns (idempotent re-run)
    for col in ("tw_max_p90", "tw_min_p90"):
        if col in existing.columns:
            existing = existing.drop(columns=[col])

    patch_cols = ["city_id", "doy", "tw_max_p90", "tw_min_p90"]
    patched = existing.merge(thresholds_wb[patch_cols], on=["city_id", "doy"], how="left")

    n_missing = patched["tw_max_p90"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing:,} rows with NaN tw_max_p90 — "
              f"check city_id alignment between 05c and 05e")

    patched.to_parquet(thr_path, index=False)
    size_mb = thr_path.stat().st_size / 1e6
    print(f"  ✓ Patched → {thr_path.name}  shape={patched.shape}  "
          f"({size_mb:.1f} MB)")
    print(f"  Columns: {list(patched.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Download study-period d2m for 05d
# ─────────────────────────────────────────────────────────────────────────────
def download_study_period_d2m():
    """
    Download d2m_max + d2m_min for 2019-2021 → ERA5/dewpoint/d2m_daily_{year}.nc.

    These files are read by the updated 05d_extract_era5_cities.py to compute
    tw_max and tw_min at each FUA city centroid for the study period.

    Total: 3 × 2 = 6 CDS requests; estimated 3–6 hours.
    """
    client = get_cds_client()
    for year in STUDY_YEARS:
        print(f"\n  Study period: downloading d2m {year} …")
        nc = download_d2m_year(client, year)
        print(f"  ✓ {nc.name}  ({nc.stat().st_size / 1e6:.0f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 d2m and extend pipeline with wet-bulb thresholds"
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--only-study", action="store_true",
                   help="Download only 2019-2021 study-period d2m (6 CDS requests)")
    g.add_argument("--only-base",  action="store_true",
                   help="Download 1991-2020 baseline d2m and patch threshold table")
    g.add_argument("--patch-only", action="store_true",
                   help="Skip downloads; recompute thresholds from existing "
                        "d2m NetCDFs in ERA5/dewpoint/")
    args = parser.parse_args()

    DEWPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ERA5 DEWPOINT / WET-BULB EXTENSION  (05e)")
    print(f"  Baseline : {BASELINE_YEARS[0]}–{BASELINE_YEARS[-1]}  "
          f"({len(BASELINE_YEARS)} years)")
    print(f"  Study    : {STUDY_YEARS[0]}–{STUDY_YEARS[-1]}")
    n_base = len(BASELINE_YEARS) * 2
    n_study = len(STUDY_YEARS) * 2
    if not args.only_study and not args.patch_only:
        print(f"  CDS reqs : {n_base} baseline  +  {n_study} study  =  "
              f"{n_base + n_study} total")
    print(f"  Output   : ERA5/dewpoint/  +  ERA5/baseline/threshold_table_doy.parquet")
    print("=" * 65)

    coords_df = load_city_coords()

    # ── A: Baseline threshold extension ──────────────────────────────────────
    if not args.only_study:
        print("\n[A] BASELINE 1991-2020: d2m download + wet-bulb thresholds …")
        baseline_wb   = build_wetbulb_baseline(coords_df, patch_only=args.patch_only)
        thresholds_wb = compute_wetbulb_thresholds(baseline_wb)
        print("\n  Patching threshold_table_doy.parquet …")
        patch_threshold_table(thresholds_wb)

    # ── B: Study period download (for 05d) ───────────────────────────────────
    if not args.only_base and not args.patch_only:
        print("\n[B] STUDY PERIOD 2019-2021: downloading d2m NetCDFs for 05d …")
        download_study_period_d2m()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DONE")
    thr_path = BASELINE_DIR / "threshold_table_doy.parquet"
    if thr_path.exists():
        thr = pd.read_parquet(thr_path)
        print(f"  threshold_table_doy.parquet : {thr.shape}  "
              f"cols: {list(thr.columns)}")
    for year in STUDY_YEARS:
        nc = DEWPOINT_DIR / f"d2m_daily_{year}.nc"
        if nc.exists():
            print(f"  d2m_daily_{year}.nc         : {nc.stat().st_size / 1e6:.0f} MB")

    print("\nNext steps:")
    print("  1. python 05d_extract_era5_cities.py   (now generates tw_max + tw_min)")
    print("  2. python 04_define_events.py           (humid_heat_event + warm_night_event)")
    print("=" * 65)


if __name__ == "__main__":
    main()
