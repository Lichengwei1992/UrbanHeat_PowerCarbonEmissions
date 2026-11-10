"""
05c_download_era5_baseline.py
=============================
Download ERA5 1991–2020 climate baseline and compute city-specific,
day-of-year (DOY) percentile thresholds for heatwave and cold wave detection.

Method
------
WMO standard: for each city and each calendar DOY, pool observations from a
±15-day window centred on that DOY across all 30 baseline years (1991–2020).
This gives ~930 samples per city×DOY, yielding robust city- and season-specific
percentile thresholds that preserve the local climate cycle.

  tmax_p90(city, DOY) = p90 of {Tmax : year ∈ 1991–2020, |DOY_obs − DOY| ≤ 15}
  tmin_p10(city, DOY) = p10 of {Tmin : year ∈ 1991–2020, |DOY_obs − DOY| ≤ 15}

Why NOT a single annual threshold per city
------------------------------------------
A flat annual p90 does not distinguish seasons: a p90 threshold computed over
the whole year in Stockholm will be ~20°C — a comfortable summer day, not an
extreme. The DOY-windowed approach compares each day only against climatologically
similar days, so January cold extremes are evaluated against January baselines
and July heat extremes against July baselines.

Download strategy (storage-efficient)
--------------------------------------
- Variables: daily Tmax + Tmin only (not Tmean — not needed for thresholds)
- Resolution: 0.25° — matches study-period (05d) extraction to eliminate
  spatial mismatch in thresholds (especially island/mountain/coastal cities)
- Mode A (USE_DERIVED_DAILY_STATS=True): derived-era5-single-levels-daily-statistics
  with 1-hourly daily_max/daily_min — fully consistent with study period
- Mode B (USE_DERIVED_DAILY_STATS=False): raw ERA5 with 8 sub-daily snapshots
  (every 3h) — ~0.1-0.3°C underestimate of true Tmax relative to Mode A
- Derived mode uses adaptive chunks: 12 months → 6 months → 3 months → 1 month
  if CDS rejects a large request; city pixels are extracted year by year and
  temporary NetCDF files are deleted
- Final output: ERA5/baseline/threshold_table_doy.parquet (~10 MB)
  and ERA5/baseline/city_baseline_all_years.parquet (~30 MB, optional keep)

Output
------
    ERA5/baseline/threshold_table_doy.parquet
        Columns: city_id | doy | tmax_p90 | tmin_p10 | n_samples
        Rows: 1,583 cities × 366 DOYs = ~579,378 rows

    ERA5/baseline/city_baseline_all_years.parquet   (optional, --keep-raw)
        Columns: city_id | date | tmax_c | tmin_c
        Used by 04_define_events.py to look up DOY-specific thresholds.

PRE-REQUISITES
--------------
    Same CDS Terms of Use acceptance as 05_download_era5.py:
    https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics
    pip install "cdsapi>=0.7.7" xarray netcdf4 geopandas pandas numpy pyarrow

Run
---
    python 05c_download_era5_baseline.py
    python 05c_download_era5_baseline.py --keep-raw     # keep per-year city parquets
    python 05c_download_era5_baseline.py --thresholds-only  # skip download, recompute only
"""

import argparse
import calendar
import shutil
import sys
import time
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
BASELINE_DIR = BASE / "ERA5" / "baseline"
SHPFILE      = BASE / "ShapefileData" / "FUA_cities_final.shp"

BASELINE_YEARS   = list(range(1991, 2021))   # 30-year WMO standard period
MONTHS           = [f"{m:02d}" for m in range(1, 13)]
HEAT_PERCENTILE  = 90
COLD_PERCENTILE  = 10
DOY_WINDOW       = 15   # ± days around target DOY for percentile pooling

# ── Download mode ────────────────────────────────────────────────────────────
#
# USE_DERIVED_DAILY_STATS = True  (RECOMMENDED)
#   Dataset : derived-era5-single-levels-daily-statistics
#   Method  : true 1-hourly daily_max / daily_min (same as study period 05d)
#   File    : ~50-80 MB/month at 0.25° (daily aggregated, one value/day/pixel)
#   Pros    : fully consistent with study-period daily extremes — no approximation gap
#   Cons    : an older CDS/MARS ambiguity bug sometimes caused 400 errors for some
#             historical years. Try this mode first; switch to False if you get errors.
#
# USE_DERIVED_DAILY_STATS = False  (FALLBACK)
#   Dataset : reanalysis-era5-single-levels (raw hourly ERA5)
#   Method  : 8 snapshots every 3h; daily max/min approximated by resampling
#   File    : ~200-250 MB/month at 0.25° × 8 snapshots
#   Pros    : no CDS product availability issues for historical years
#   Cons    : 8-snapshot daily max/min underestimates true Tmax by ~0.1-0.3°C
#             relative to the 1-hourly derived stats used in the study period;
#             this systematic gap slightly lowers p90 thresholds → marginally more
#             heatwave events flagged in study period than a fully consistent method
#             would produce. Document as methodological limitation.
#
USE_DERIVED_DAILY_STATS = True   # ← change to False if CDS returns 400 errors

# Dataset names
_DATASET_DERIVED = "derived-era5-single-levels-daily-statistics"
_DATASET_RAW     = "reanalysis-era5-single-levels"

CDS_VARIABLE = "2m_temperature"
# Derived daily-statistics mode: request one statistic at a time. This mirrors
# 05_download_era5.py and the CDS API example, and avoids multi-NetCDF archives.
DERIVED_STATISTICS = {
    "daily_max": "t2m_max",
    "daily_min": "t2m_min",
}
# Raw-mode only: 8 sub-daily snapshots every 3h (ignored in derived-stats mode)
TIMES = ["00:00", "03:00", "06:00", "09:00",
         "12:00", "15:00", "18:00", "21:00"]

# Resolution: 0.25° — matches study-period (05d) extraction
GRID_DEG = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def valid_days(year: int, month: int) -> int:
    """Return the number of days in a given month (accounts for leap years)."""
    return calendar.monthrange(year, month)[1]


def get_cds_client():
    try:
        import cdsapi
        return cdsapi.Client()
    except ImportError:
        sys.exit('pip install "cdsapi>=0.7.7"')
    except Exception as e:
        sys.exit(f"CDS error: {e}\nCheck ~/.cdsapirc")


def retrieve_and_unzip(client, request: dict, out: Path, dataset: str | None = None):
    """Download from CDS atomically; unzip if CDS wraps result in an archive.

    Args:
        dataset: CDS dataset name. Defaults to the mode-appropriate name
                 (_DATASET_DERIVED or _DATASET_RAW) based on USE_DERIVED_DAILY_STATS.
    """
    if dataset is None:
        dataset = _DATASET_DERIVED if USE_DERIVED_DAILY_STATS else _DATASET_RAW
    download = out.with_suffix(".download")
    partial  = out.with_suffix(".part.nc")
    download.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    try:
        client.retrieve(dataset, request, str(download))
        if not download.exists() or download.stat().st_size == 0:
            raise RuntimeError(f"CDS returned an empty file for {out.name}")

        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as archive:
                members = [m for m in archive.namelist()
                           if m.lower().endswith((".nc", ".nc4", ".netcdf"))]
                if len(members) != 1:
                    raise RuntimeError(
                        f"Expected 1 NetCDF in CDS archive, found {len(members)}: {members}"
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
    """Return True for CDS requests rejected because the selection is too large."""
    text = str(exc).lower()
    return "cost limits exceeded" in text or "request is too large" in text


def month_groups(group_size: int) -> list[list[str]]:
    """Split MONTHS into consecutive chunks such as yearly, half-year, quarter, month."""
    return [MONTHS[i:i + group_size] for i in range(0, len(MONTHS), group_size)]


def derived_request(year: int, months: list[str], statistic: str) -> dict:
    """Build one derived daily-statistics CDS request."""
    return {
        "product_type":    "reanalysis",
        "variable":        [CDS_VARIABLE],
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
    """Handle time-coordinate names used by different CDS NetCDF converters."""
    for name in ("valid_time", "time", "date"):
        if name in obj.coords:
            return name
    raise RuntimeError(
        f"No time coordinate found. Coordinates: {list(obj.coords)}; "
        f"dims: {list(obj.dims)}"
    )


def load_daily_stat_dataset(path: Path, output_name: str, statistic: str):
    """Load a derived daily-statistics file and rename its data variable."""
    import xarray as xr

    with xr.open_dataset(path) as source:
        time_name = find_time_name(source)
        candidates = [
            name for name, variable in source.data_vars.items()
            if time_name in variable.dims and variable.dtype.kind in "iuf"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot identify the daily-stat variable in {path.name}; "
                f"data variables are {list(source.data_vars)}"
            )
        ds = source[[candidates[0]]].rename({candidates[0]: output_name}).load()
        ds[output_name].attrs["daily_statistic"] = statistic
        return ds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load city coordinates
# ─────────────────────────────────────────────────────────────────────────────
def load_city_coords():
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("pip install geopandas")
    print(f"  Loading city coordinates from {SHPFILE.name} …")
    gdf = gpd.read_file(SHPFILE)
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["city_id"] = gdf.index.astype(str).str.zfill(4)
    coords = gdf[["city_id", "city", "country", "lon", "lat"]].copy()
    print(f"  {len(coords)} cities loaded")
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Download 1 month of Tmax + Tmin at 0.25°
# ─────────────────────────────────────────────────────────────────────────────
def download_baseline_month(client, year: int, month: int) -> Path:
    """
    Download ERA5 2m temperature for one month at 0.25° resolution.

    Mode A — derived daily stats (USE_DERIVED_DAILY_STATS = True):
        Dataset : derived-era5-single-levels-daily-statistics
        Content : true 1-hourly daily_max + daily_min (one value/day/pixel)
        File    : ~50–80 MB/month
        Method  : fully consistent with study-period (05d) daily extremes

    Mode B — raw reanalysis with 8 sub-daily snapshots (USE_DERIVED_DAILY_STATS = False):
        Dataset : reanalysis-era5-single-levels
        Content : raw T2m at 8 times/day (00/03/06/09/12/15/18/21 UTC)
        File    : ~200–250 MB/month
        Method  : daily max/min approximated by resampling 8 snapshots

    Mode A uses 720 total requests (30 years × 12 months × 2 statistics).
    Mode B uses 360 total requests (30 years × 12 months). Temporary files are
    deleted after city-pixel extraction (see extract_city_pixels).

    Returns path to the downloaded NetCDF file (temporary).
    """
    out = BASELINE_DIR / f"_temp_{year}_{month:02d}.nc"
    if out.exists() and out.stat().st_size > 0:
        return out   # resume-safe: skip already-downloaded months

    n_days = valid_days(year, month)
    days   = [f"{d:02d}" for d in range(1, n_days + 1)]

    if USE_DERIVED_DAILY_STATS:
        # ── Mode A: derived daily statistics ─────────────────────────────
        # Request one daily statistic at a time. The CDS API example exposes
        # daily_statistic as a scalar (daily_mean/daily_max/daily_min/daily_sum),
        # and this avoids archives containing multiple NetCDF files.
        import xarray as xr

        datasets = []
        temp_files = []
        try:
            for statistic, output_name in DERIVED_STATISTICS.items():
                chunk = BASELINE_DIR / f"_temp_{year}_{month:02d}_{output_name}.nc"
                temp_files.append(chunk)

                if not chunk.exists() or chunk.stat().st_size == 0:
                    request = {
                        "product_type":    "reanalysis",
                        "variable":        [CDS_VARIABLE],
                        "year":            str(year),
                        "month":           f"{month:02d}",
                        "day":             days,
                        "daily_statistic": statistic,
                        "time_zone":       "utc+00:00",   # same convention as study period
                        "frequency":       "1_hourly",    # aggregate over all 24 hourly values
                        "grid":            [GRID_DEG, GRID_DEG],  # 0.25° × 0.25°
                        "data_format":     "netcdf",
                        "download_format": "unarchived",
                    }
                    retrieve_and_unzip(client, request, chunk, dataset=_DATASET_DERIVED)

                datasets.append(load_daily_stat_dataset(chunk, output_name, statistic))

            combined = xr.merge(datasets)
            partial = out.with_suffix(".part.nc")
            partial.unlink(missing_ok=True)
            combined.to_netcdf(
                partial,
                engine="netcdf4",
                encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32"}
                          for name in combined.data_vars},
            )
            partial.replace(out)
            combined.close()
        finally:
            for ds in datasets:
                ds.close()
            for chunk in temp_files:
                chunk.unlink(missing_ok=True)
    else:
        # ── Mode B: raw ERA5 + resample 8 sub-daily snapshots ─────────────
        request = {
            "product_type":    "reanalysis",
            "variable":        CDS_VARIABLE,
            "year":            str(year),
            "month":           f"{month:02d}",
            "day":             days,
            "time":            TIMES,          # 8 snapshots every 3h
            "grid":            [GRID_DEG, GRID_DEG],  # 0.25° × 0.25°
            "data_format":     "netcdf",
            "download_format": "unarchived",
        }
        retrieve_and_unzip(client, request, out, dataset=_DATASET_RAW)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Extract city pixels from a single-year NetCDF
# ─────────────────────────────────────────────────────────────────────────────
def extract_city_pixels(nc_path: Path, coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract daily Tmax and Tmin for each city centroid from a monthly ERA5 NetCDF.

    Handles both file types automatically:

    Derived-stats file (USE_DERIVED_DAILY_STATS = True):
        Variables: one with "max" in name, one with "min" in name (daily aggregated)
        Time dim:  one value per day → read directly, no resampling
        Units:     Kelvin (K)

    Raw-reanalysis file (USE_DERIVED_DAILY_STATS = False):
        Variable:  t2m (sub-daily, 8 values per day)
        Time dim:  n_days × 8 values → resample to daily max/min
        Units:     Kelvin (K)

    Longitude wrapping: ERA5 uses 0–360°; city centroids use −180–180°.
    Conversion uses `% 360` plus a circular-wrap check near the 0°/360° boundary.

    Returns DataFrame: city_id | date | tmax_c | tmin_c
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    vars_list = list(ds.data_vars)

    # Detect whether this is a derived-stats file (has separate max/min variables)
    # or a raw sub-daily file (single T2m variable to be resampled)
    vmax = next((v for v in vars_list if "max" in v.lower()), None)
    vmin = next((v for v in vars_list if "min" in v.lower()), None)
    is_derived = (vmax is not None and vmin is not None)

    if not is_derived:
        # Raw sub-daily file: find the T2m variable
        t2m_var = next(
            (v for v in vars_list
             if v.lower() in ("t2m", "var_167", "var167", "2m_temperature")),
            None
        )
        if t2m_var is None:
            t2m_var = vars_list[0] if len(vars_list) == 1 else None
        if t2m_var is None:
            ds.close()
            raise RuntimeError(
                f"Cannot identify T2m variable in {nc_path.name}. "
                f"Available: {vars_list}"
            )

    # Resolve coordinate names
    lat_name  = next((c for c in ("latitude",  "lat")          if c in ds.coords), None)
    lon_name  = next((c for c in ("longitude", "lon")          if c in ds.coords), None)
    time_name = next((c for c in ("valid_time", "time", "date") if c in ds.coords), None)
    if not all([lat_name, lon_name, time_name]):
        ds.close()
        raise RuntimeError(
            f"Missing coordinates in {nc_path.name}. Coords: {list(ds.coords)}"
        )

    # Pre-load coordinate arrays once
    lats = ds[lat_name].values   # (n_lat,)
    lons = ds[lon_name].values   # (n_lon,)  — ERA5 uses 0–360° convention

    records = []
    for _, row in coords_df.iterrows():
        try:
            # ── Longitude wrapping fix ─────────────────────────────────────
            # ERA5 stores lon in 0–360°; city centroids in −180–180°.
            # Without % 360 conversion, cities with lon < 0 snap to 0°E.
            # Circular-wrap check also handles cities near the 0°/360° boundary
            # (e.g. London at −0.13° → 359.87° is closest to 359.75° cell, not 0°).
            city_lon_adj = float(row["lon"]) % 360
            lat_idx = int(np.argmin(np.abs(lats - float(row["lat"]))))
            dists_raw  = np.abs(lons - city_lon_adj)
            dists_wrap = np.abs(lons - (city_lon_adj - 360))
            lon_idx = int(np.argmin(np.minimum(dists_raw, dists_wrap)))

            if is_derived:
                # ── Derived-stats file: read daily max/min directly ────────
                # Values are already K; convert to °C
                tmax_vals = (ds[vmax].isel({lat_name: lat_idx, lon_name: lon_idx}).values
                             - 273.15).astype(np.float32)
                tmin_vals = (ds[vmin].isel({lat_name: lat_idx, lon_name: lon_idx}).values
                             - 273.15).astype(np.float32)
                dates = pd.to_datetime(ds[time_name].values)

                df_city = pd.DataFrame({
                    "city_id": row["city_id"],
                    "date":    dates,
                    "tmax_c":  tmax_vals,
                    "tmin_c":  tmin_vals,
                })
            else:
                # ── Raw sub-daily file: resample 8 snapshots → daily ───────
                pixel = (ds[t2m_var].isel({lat_name: lat_idx, lon_name: lon_idx})
                         - 273.15).values
                times_dt = pd.to_datetime(ds[time_name].values)
                ts = pd.Series(pixel, index=times_dt)
                daily_max = ts.resample("1D").max()
                daily_min = ts.resample("1D").min()

                df_city = pd.DataFrame({
                    "city_id": row["city_id"],
                    "date":    daily_max.index,
                    "tmax_c":  daily_max.values.astype(np.float32),
                    "tmin_c":  daily_min.values.astype(np.float32),
                })

            records.append(df_city)
        except Exception as e:
            label = row.get("city", row["city_id"]) if hasattr(row, "get") else row["city_id"]
            print(f"    Warning: {label} ({row['city_id']}) — {e}")

    ds.close()
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4a: Adaptive chunk download (derived-stats mode only)
# ─────────────────────────────────────────────────────────────────────────────
def download_baseline_year(client, year: int) -> Path:
    """
    Download one year of ERA5 derived daily stats with adaptive chunking.

    The CDS cost envelope can reject a full-year 0.25° request. To keep the
    queue count low without failing, try chunks in this order:
        12 months → 6 months → 3 months → 1 month

    Each chunk is requested separately for daily_max and daily_min, then all
    chunks are concatenated into one annual NetCDF used by city extraction.
    """
    import xarray as xr

    out = BASELINE_DIR / f"_temp_{year}_annual.nc"
    if out.exists() and out.stat().st_size > 0:
        return out

    for group_size, label in [(12, "annual"), (6, "half-year"), (3, "quarterly"), (1, "monthly")]:
        groups = month_groups(group_size)
        temp_files = []
        loaded = []
        combined = None

        try:
            for statistic, output_name in DERIVED_STATISTICS.items():
                for months in groups:
                    month_label = f"{months[0]}-{months[-1]}"
                    chunk = BASELINE_DIR / f"_temp_{year}_{output_name}_{month_label}.nc"
                    temp_files.append(chunk)

                    if not chunk.exists() or chunk.stat().st_size == 0:
                        print(f"    {year} {output_name} {month_label}: requesting {label} chunk")
                        retrieve_and_unzip(
                            client,
                            derived_request(year, months, statistic),
                            chunk,
                            dataset=_DATASET_DERIVED,
                        )

            stat_datasets = []
            for statistic, output_name in DERIVED_STATISTICS.items():
                parts = []
                for months in groups:
                    month_label = f"{months[0]}-{months[-1]}"
                    chunk = BASELINE_DIR / f"_temp_{year}_{output_name}_{month_label}.nc"
                    part = load_daily_stat_dataset(chunk, output_name, statistic)
                    parts.append(part)
                    loaded.append(part)

                time_name = find_time_name(parts[0])
                stat_ds = xr.concat(parts, dim=time_name).sortby(time_name)
                stat_datasets.append(stat_ds)
                loaded.append(stat_ds)

            combined = xr.merge(stat_datasets)
            partial = out.with_suffix(".part.nc")
            partial.unlink(missing_ok=True)
            combined.to_netcdf(
                partial,
                engine="netcdf4",
                encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32"}
                          for name in combined.data_vars},
            )
            partial.replace(out)

            for chunk in temp_files:
                chunk.unlink(missing_ok=True)
            print(f"    {year}: {label} chunks accepted by CDS")
            return out

        except Exception as exc:
            if is_cost_limit_error(exc) and group_size > 1:
                print(f"    {year}: {label} chunk exceeded CDS cost limit; trying smaller chunks")
                for chunk in temp_files:
                    chunk.unlink(missing_ok=True)
                continue
            raise

        finally:
            if combined is not None:
                combined.close()
            for ds in loaded:
                try:
                    ds.close()
                except Exception:
                    pass

    raise RuntimeError(f"Could not download {year}; even monthly chunks failed.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b: Download all years, extract, accumulate in memory
# ─────────────────────────────────────────────────────────────────────────────
def build_baseline_series(coords_df: pd.DataFrame, thresholds_only: bool = False) -> pd.DataFrame:
    """
    Download 1991-2020 and extract city pixels year by year.

    Derived-stats mode (USE_DERIVED_DAILY_STATS = True) — adaptive batching:
        Try 12-month chunks first; if CDS rejects the request as too large,
        automatically fall back to 6-month, 3-month, then 1-month chunks.
        Each chunk is requested once for daily_max and once for daily_min.

    Raw ERA5 mode (USE_DERIVED_DAILY_STATS = False) — monthly batching:
        1 CDS request/month × 12 × 30 = 360 total requests.
        A full raw year (8 snapshots/day) ≈ 3 B values → exceeds CDS limit,
        so monthly batching is kept for this mode.

    Each year's extracted city series is saved as a per-year parquet, then
    combined into city_baseline_all_years.parquet at the end.
    """
    raw_all = BASELINE_DIR / "city_baseline_all_years.parquet"
    if raw_all.exists():
        print(f"  ✓ {raw_all.name} already exists — loading cached baseline series")
        return pd.read_parquet(raw_all)

    if thresholds_only:
        sys.exit(
            f"--thresholds-only requires {raw_all.name} to already exist.\n"
            "Run without that flag first to download the baseline."
        )

    client = get_cds_client()
    all_years = []
    total_years = len(BASELINE_YEARS)
    n_cities = len(coords_df)

    for yi, year in enumerate(BASELINE_YEARS):
        year_cache = BASELINE_DIR / f"city_baseline_{year}.parquet"

        if year_cache.exists():
            df_year = pd.read_parquet(year_cache)
            all_years.append(df_year)
            print(f"  [{yi+1}/{total_years}] {year} — loaded from cache "
                  f"({len(df_year):,} rows)")
            continue

        batch_label = "adaptive chunks (12→6→3→1 months)" if USE_DERIVED_DAILY_STATS else "12 monthly requests"
        print(f"\n  [{yi+1}/{total_years}] Year {year} — downloading via {batch_label} …")
        t0_year = time.time()

        if USE_DERIVED_DAILY_STATS:
            # ── Annual batch: 2 requests for the full year ──────────────────
            nc_path  = download_baseline_year(client, year)
            size_mb  = nc_path.stat().st_size / 1e6
            t0       = time.time()
            df_year  = extract_city_pixels(nc_path, coords_df)
            df_year["date"] = pd.to_datetime(df_year["date"])
            nc_path.unlink()
            elapsed  = time.time() - t0
            n_days_yr = df_year["date"].nunique()
            print(f"    {year}: {size_mb:.0f} MB → {n_days_yr} days × "
                  f"{n_cities} cities extracted ({elapsed:.0f}s)")
        else:
            # ── Monthly batching for raw ERA5 (8 snapshots too large annually) ─
            monthly_parts = []
            for month in range(1, 13):
                t0      = time.time()
                nc_path = download_baseline_month(client, year, month)
                size_mb = nc_path.stat().st_size / 1e6
                df_month = extract_city_pixels(nc_path, coords_df)
                df_month["date"] = pd.to_datetime(df_month["date"])
                nc_path.unlink()
                elapsed = time.time() - t0
                monthly_parts.append(df_month)
                n_days = valid_days(year, month)
                print(f"    {year}-{month:02d}: {size_mb:.0f} MB → {n_days} days × "
                      f"{n_cities} cities extracted ({elapsed:.0f}s)")
            df_year = pd.concat(monthly_parts, ignore_index=True)
        df_year.to_parquet(year_cache, index=False)
        all_years.append(df_year)

        elapsed_year = time.time() - t0_year
        print(f"  ✓ {year} complete — {len(df_year):,} rows, "
              f"{elapsed_year/60:.1f} min")

    combined = pd.concat(all_years, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined.to_parquet(raw_all, index=False)
    size_mb = raw_all.stat().st_size / 1e6
    print(f"\n  Full baseline series saved → {raw_all.name} ({size_mb:.0f} MB)")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Compute ±15-day DOY-windowed percentile thresholds
# ─────────────────────────────────────────────────────────────────────────────
def compute_doy_thresholds(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each city and each DOY (1-366), pool observations from ±DOY_WINDOW days
    across all baseline years and compute p90 of Tmax and p10 of Tmin.

    The window wraps circularly (e.g., DOY 3 pools from DOY 353-366 and 1-18).
    DOY 366 (Feb 29 in leap years) is mapped to DOY 60 before windowing so
    leap-day observations contribute to the late-February pool in all years.

    Vectorized per-city: builds a (N_obs, 366) boolean mask in one numpy
    broadcast, then calls np.percentile 2×366 times on ~930-element arrays.
    Typical runtime: 5–15 minutes for 1,583 cities.

    Returns DataFrame: city_id | doy | tmax_p90 | tmin_p10 | n_samples
    """
    print(f"\n[Threshold] Computing ±{DOY_WINDOW}-day DOY-windowed percentiles …")
    print(f"  Baseline: {BASELINE_YEARS[0]}–{BASELINE_YEARS[-1]} ({len(BASELINE_YEARS)} years)")
    print(f"  Samples per city×DOY: ~{(2*DOY_WINDOW+1) * len(BASELINE_YEARS)}")

    baseline_df = baseline_df.copy()
    baseline_df["doy"] = baseline_df["date"].dt.dayofyear
    # Map Feb 29 (DOY 366 in leap years) → DOY 60 so it pools with late February
    baseline_df.loc[baseline_df["doy"] == 366, "doy"] = 60

    target_doys = np.arange(1, 367, dtype=np.int16)   # 1 … 366
    cities      = baseline_df["city_id"].unique()
    n_cities    = len(cities)
    t0_total    = time.time()

    all_records = []

    for ci, city_id in enumerate(cities):
        if ci % 200 == 0 and ci > 0:
            elapsed = time.time() - t0_total
            rate    = ci / elapsed
            remaining = (n_cities - ci) / rate / 60
            print(f"  {ci}/{n_cities} cities … ({remaining:.0f} min remaining)")

        cd = baseline_df[baseline_df["city_id"] == city_id]
        obs_doys  = cd["doy"].values.astype(np.int16)     # (N,)
        tmax_arr  = cd["tmax_c"].values.astype(np.float32) # (N,)
        tmin_arr  = cd["tmin_c"].values.astype(np.float32) # (N,)

        # Vectorised circular distance: shape (N, 366)
        #   dist[i, j] = circular distance from obs_doys[i] to target_doys[j]
        abs_diff = np.abs(obs_doys[:, None] - target_doys[None, :])  # (N, 366)
        circ_dist = np.minimum(abs_diff, 365 - abs_diff)             # (N, 366)
        in_window = circ_dist <= DOY_WINDOW                          # (N, 366) bool

        # Compute percentiles for each target DOY
        tmax_p90 = np.empty(366, dtype=np.float32)
        tmin_p10 = np.empty(366, dtype=np.float32)
        n_samp   = np.empty(366, dtype=np.int32)

        for j in range(366):
            mask = in_window[:, j]
            tx   = tmax_arr[mask]
            tn   = tmin_arr[mask]
            n    = len(tx)
            n_samp[j]   = n
            tmax_p90[j] = np.percentile(tx, HEAT_PERCENTILE) if n > 0 else np.nan
            tmin_p10[j] = np.percentile(tn, COLD_PERCENTILE) if n > 0 else np.nan

        city_df = pd.DataFrame({
            "city_id":   city_id,
            "doy":       target_doys.astype(int),
            "tmax_p90":  tmax_p90,
            "tmin_p10":  tmin_p10,
            "n_samples": n_samp,
        })
        all_records.append(city_df)

    thresholds = pd.concat(all_records, ignore_index=True)
    elapsed_min = (time.time() - t0_total) / 60
    print(f"  ✓ Thresholds computed: {len(thresholds):,} city×DOY rows ({elapsed_min:.1f} min)")
    print(f"  Tmax p90 range: {thresholds['tmax_p90'].min():.1f}°C – "
          f"{thresholds['tmax_p90'].max():.1f}°C")
    print(f"  Tmin p10 range: {thresholds['tmin_p10'].min():.1f}°C – "
          f"{thresholds['tmin_p10'].max():.1f}°C")
    print(f"  Median samples per city×DOY: {thresholds['n_samples'].median():.0f}")
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Storage cleanup
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_per_year_caches(keep_raw: bool):
    """Remove per-year city parquets after the merged file is saved."""
    if keep_raw:
        print("  --keep-raw: per-year city parquets retained")
        return
    removed = 0
    for f in BASELINE_DIR.glob("city_baseline_????.parquet"):
        f.unlink()
        removed += 1
    if removed:
        print(f"  Removed {removed} per-year cache files")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Storage summary
# ─────────────────────────────────────────────────────────────────────────────
def print_storage_summary():
    print("\n" + "─" * 60)
    print("BASELINE STORAGE SUMMARY")
    print("─" * 60)
    total = 0
    for f in sorted(BASELINE_DIR.iterdir()):
        if f.is_file():
            mb = f.stat().st_size / 1e6
            total += mb
            print(f"  {f.name:<50} {mb:6.1f} MB")
    print(f"  {'TOTAL':<50} {total:6.1f} MB")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compute WMO-standard city DOY climate thresholds from ERA5 1991-2020"
    )
    parser.add_argument("--thresholds-only", action="store_true",
                        help="Skip download; recompute thresholds from cached series")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep per-year city parquets after merging")
    args = parser.parse_args()

    print("=" * 60)
    print("ERA5 BASELINE THRESHOLD COMPUTATION")
    print(f"Period: {BASELINE_YEARS[0]}–{BASELINE_YEARS[-1]} "
          f"({len(BASELINE_YEARS)} years, WMO standard)")
    print(f"Method: ±{DOY_WINDOW}-day DOY window per city")
    print(f"Output: ERA5/baseline/threshold_table_doy.parquet")
    print("=" * 60)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    out_thresholds = BASELINE_DIR / "threshold_table_doy.parquet"
    if out_thresholds.exists():
        print(f"\n  ✓ {out_thresholds.name} already exists.")
        print("  Delete it and rerun to recompute from scratch.")
        print_storage_summary()
        return

    # Load city coordinates
    coords_df = load_city_coords()

    # Download + extract baseline series (year by year, delete NetCDF after each)
    mode_label = "derived daily stats, 1-hourly" if USE_DERIVED_DAILY_STATS else \
                 f"raw ERA5, {len(TIMES)} snapshots/day"
    temp_note = "depends on accepted chunk size; allow a few GB transient" if USE_DERIVED_DAILY_STATS else \
                 "~250 MB/month, deleted after extraction"
    if USE_DERIVED_DAILY_STATS:
        batch_note = "adaptive chunks: 12→6→3→1 months (60 best case; 720 monthly fallback)"
        queue_note = "20–60 min/request; actual count depends on largest accepted chunk"
    else:
        n_requests = len(BASELINE_YEARS) * 12  # 1 per month
        batch_note = f"monthly, {n_requests} total CDS requests"
        queue_note = f"20–60 min/request × {n_requests} requests"

    print(f"\n[Download] Downloading {len(BASELINE_YEARS)} years at {GRID_DEG}° "
          f"({mode_label}) …")
    print(f"  Max temp disk usage: {temp_note}")
    print(f"  CDS requests: {batch_note}")
    print(f"  Est. CDS queue time: {queue_note}")
    print(f"  Run overnight or submit and check progress")

    baseline_df = build_baseline_series(coords_df, thresholds_only=args.thresholds_only)

    # Compute DOY-windowed percentiles
    thresholds = compute_doy_thresholds(baseline_df)

    # Save threshold table
    thresholds.to_parquet(out_thresholds, index=False)
    size_mb = out_thresholds.stat().st_size / 1e6
    print(f"\n  ✓ Saved → {out_thresholds.name} ({size_mb:.1f} MB)")

    # Cleanup
    cleanup_per_year_caches(keep_raw=args.keep_raw)
    print_storage_summary()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  04_define_events.py will automatically load:")
    print(f"    {out_thresholds}")
    print("  and apply DOY-specific p90/p10 thresholds per city per day.")
    print("=" * 60)


if __name__ == "__main__":
    main()
