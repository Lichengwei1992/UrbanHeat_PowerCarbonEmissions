"""
05d_extract_era5_cities.py
==========================
Extract study-period (2019–2021) daily climate variables for all 1,583 FUA
city centroids from the already-downloaded ERA5 annual NetCDF files, and save
a single parquet that 04_define_events.py can consume directly.

Why a separate script?
----------------------
04_define_events.py mixes ERA5 extraction with event detection. The extraction
from global 0.25° NetCDF files (2 GB each) is slow and only needs to run once.
Pre-extracting here saves ~20 min on every re-run of the event-detection step.

Inputs
------
  ERA5/temperature/era5_temp_daily_{YYYY}.nc   (from 05_download_era5.py)
      vars: t2m_max, t2m_min, t2m_mean  (°C or K — auto-detected)
      coords: latitude, longitude, valid_time
      resolution: 0.25°, global

  ERA5/utci/utci_daily_{YYYY}.nc              (from 05_download_era5.py)
      vars: utci_mean, utci_max, utci_min   (°C)
      coords: lat, lon, time
      resolution: 0.25°, lat 60°S – 90°N

  ERA5/dewpoint/d2m_daily_{YYYY}.nc          (from 05e_download_era5_dewpoint.py)
      vars: d2m_max, d2m_min  (°C or K — auto-detected)
      coords: latitude, longitude, valid_time
      resolution: 0.25°, global
      NOTE: optional — if absent, tw_max/tw_min columns are NaN

  ShapefileData/FUA_cities_final.shp          (1,583 FUA polygons)
      used for centroid coordinates

Outputs
-------
  outputs/city_climate_2019_2021.parquet
      city_id | date | tmax | tmin | tmean | utci_mean | utci_max
              | tw_max | tw_min
      city_id = zero-padded shapefile row index ("0000"…"1582")
      temperatures in °C

      tw_max — daytime wet-bulb temperature (Stull 2011) from Tmax + d2m_min
               used for humid_heat_event detection in 04_define_events.py
      tw_min — nighttime wet-bulb temperature from Tmin + d2m_max
               used for warm_night_event detection in 04_define_events.py

  outputs/fua_city_lookup.csv
      city_id | city | country | fuacode | lat | lon
      Key lookup table: maps shapefile index → city name/country.
      Used by 04_define_events.py to join Carbon Monitor cities.

Run
---
    python 05d_extract_era5_cities.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
ERA5_DIR = BASE / "ERA5"
SHP_DIR  = BASE / "ShapefileData"
OUT_DIR  = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_YEARS = [2019, 2020, 2021]

TEMP_TEMPLATE = ERA5_DIR / "temperature" / "era5_temp_daily_{year}.nc"
UTCI_TEMPLATE = ERA5_DIR / "utci"        / "utci_daily_{year}.nc"
D2M_TEMPLATE  = ERA5_DIR / "dewpoint"    / "d2m_daily_{year}.nc"
SHPFILE       = SHP_DIR  / "FUA_cities_final.shp"

OUT_CLIMATE = OUT_DIR / "city_climate_2019_2021.parquet"
OUT_LOOKUP  = OUT_DIR / "fua_city_lookup.csv"


# ─────────────────────────────────────────────────────────────────────────────
# WET-BULB PHYSICS  (Stull 2011; Bolton 1980)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from dry-bulb T and dewpoint Td (°C).
    August-Roche-Magnus approximation; result clipped to [1, 100] %."""
    gamma_t  = 17.625 * t_c  / (243.04 + t_c)
    gamma_td = 17.625 * td_c / (243.04 + td_c)
    return np.clip(100.0 * np.exp(gamma_td - gamma_t), 1.0, 100.0).astype(np.float32)


def _stull_wetbulb(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Wet-bulb temperature (°C) via Stull (2011).
    Valid for 5% ≤ RH ≤ 99%, −20°C ≤ T ≤ 50°C (±0.35°C accuracy)."""
    rh = np.clip(rh, 5.0, 99.0)
    return (
        t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load city centroids from FUA shapefile
# ─────────────────────────────────────────────────────────────────────────────
def load_city_coords() -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("pip install geopandas")

    print(f"  Loading {SHPFILE.name} …")
    gdf = gpd.read_file(SHPFILE)

    # city_id: zero-padded row index — must match 05c threshold table
    gdf["city_id"] = gdf.index.astype(str).str.zfill(4)
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y

    keep = ["city_id", "city", "country", "fuacode", "lat", "lon"]
    coords = gdf[[c for c in keep if c in gdf.columns]].copy()
    print(f"  {len(coords)} cities loaded")

    # Save lookup table for 04_define_events.py
    coords.to_csv(OUT_LOOKUP, index=False)
    print(f"  Lookup saved → {OUT_LOOKUP.name}")
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Extract city pixels from one annual temperature NetCDF
# ─────────────────────────────────────────────────────────────────────────────
def extract_temp_year(nc_path: Path, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Extract t2m_max, t2m_min, t2m_mean for all cities from one annual file.

    Strategy: load the full dataset into memory once, then use vectorised
    nearest-neighbour selection across all cities. This is faster than opening
    the file 1,583 times.

    Returns DataFrame: city_id | date | tmax | tmin | tmean
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # Identify temperature variables
    vmax  = next((v for v in ds.data_vars if "max"  in v.lower()), None)
    vmin  = next((v for v in ds.data_vars if "min"  in v.lower()), None)
    vmean = next((v for v in ds.data_vars if "mean" in v.lower()), None)

    if not all([vmax, vmin, vmean]):
        ds.close()
        raise RuntimeError(
            f"Cannot find max/min/mean in {nc_path.name}. "
            f"Vars: {list(ds.data_vars)}"
        )

    lat_name  = next(c for c in ("latitude", "lat")         if c in ds.coords)
    lon_name  = next(c for c in ("longitude", "lon")        if c in ds.coords)
    time_name = next(c for c in ("valid_time", "time", "date") if c in ds.coords)

    lats = ds[lat_name].values   # (721,)
    lons = ds[lon_name].values   # (1440,)
    dates = pd.to_datetime(ds[time_name].values)

    # Detect K vs °C (ERA5 is in K; should be above 200)
    sample_val = float(ds[vmax].isel({time_name: 0, lat_name: 360, lon_name: 720}).values)
    to_celsius = sample_val > 200.0   # True if Kelvin

    records = []
    for _, row in coords.iterrows():
        lat_idx = int(np.argmin(np.abs(lats - row["lat"])))
        # ERA5 NetCDF uses 0–360° longitude; city shapefile uses −180–180°.
        # Without the % 360 conversion, cities with lon < 0 (Americas, etc.)
        # all snap to lon ≈ 0° (Atlantic Ocean) → completely wrong pixels.
        # The circular-wrap check handles cities near the 0°/360° boundary
        # (e.g. London at −0.13° should map to 0.0°, not 359.75°).
        city_lon_adj = float(row["lon"]) % 360   # e.g. −97.5 → 262.5
        dists_raw  = np.abs(lons - city_lon_adj)
        dists_wrap = np.abs(lons - (city_lon_adj - 360))  # wrap check near 0°
        lon_idx = int(np.argmin(np.minimum(dists_raw, dists_wrap)))

        tmax_vals  = ds[vmax ].isel({lat_name: lat_idx, lon_name: lon_idx}).values
        tmin_vals  = ds[vmin ].isel({lat_name: lat_idx, lon_name: lon_idx}).values
        tmean_vals = ds[vmean].isel({lat_name: lat_idx, lon_name: lon_idx}).values

        if to_celsius:
            tmax_vals  = tmax_vals  - 273.15
            tmin_vals  = tmin_vals  - 273.15
            tmean_vals = tmean_vals - 273.15

        records.append(pd.DataFrame({
            "city_id": row["city_id"],
            "date":    dates,
            "tmax":    tmax_vals.astype(np.float32),
            "tmin":    tmin_vals.astype(np.float32),
            "tmean":   tmean_vals.astype(np.float32),
        }))

    ds.close()
    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Extract city pixels from one annual UTCI NetCDF
# ─────────────────────────────────────────────────────────────────────────────
def extract_utci_year(nc_path: Path, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Extract utci_mean and utci_max for all cities from one annual UTCI file.

    UTCI NetCDF covers lat 60°S – 90°N (601 points at 0.25°).
    Cities outside this range (south of 60°S) get NaN — acceptable since
    no FUA cities exist there.

    Returns DataFrame: city_id | date | utci_mean | utci_max
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    lat_name  = next(c for c in ("lat", "latitude")          if c in ds.coords)
    lon_name  = next(c for c in ("lon", "longitude")         if c in ds.coords)
    time_name = next(c for c in ("time", "valid_time", "date") if c in ds.coords)

    lats  = ds[lat_name].values
    lons  = ds[lon_name].values
    dates = pd.to_datetime(ds[time_name].values)

    lat_min, lat_max = float(lats.min()), float(lats.max())

    vumean = next((v for v in ds.data_vars if "mean" in v.lower()), None)
    vumax  = next((v for v in ds.data_vars if "max"  in v.lower()), None)
    # utci_min is required downstream (utci_coldstress / utci_strongcold in
    # 10_define_events.py, peak_utci in 11_event_statistics.py). It is taken
    # from the same annual NetCDF by the same centroid method as mean/max so
    # that every study year shares one extraction path. Splicing in the
    # polygon-mean files written by 05g would mix two spatial methods AND two
    # unit conventions -- 05g leaves UTCI in kelvin.
    vumin  = next((v for v in ds.data_vars if "min"  in v.lower()), None)

    if not vumean or not vumax:
        ds.close()
        raise RuntimeError(
            f"Cannot find utci_mean/utci_max in {nc_path.name}. "
            f"Vars: {list(ds.data_vars)}"
        )

    records = []
    for _, row in coords.iterrows():
        city_lat = row["lat"]
        city_lon = row["lon"]

        # Cities outside UTCI coverage (unlikely but guard anyway)
        if city_lat < lat_min or city_lat > lat_max:
            n_days = len(dates)
            records.append(pd.DataFrame({
                "city_id":   row["city_id"],
                "date":      dates,
                "utci_mean": np.full(n_days, np.nan, dtype=np.float32),
                "utci_max":  np.full(n_days, np.nan, dtype=np.float32),
                "utci_min":  np.full(n_days, np.nan, dtype=np.float32),
            }))
            continue

        lat_idx = int(np.argmin(np.abs(lats - city_lat)))
        city_lon_adj = city_lon % 360   # ERA5 0–360 convention; city lon is −180–180
        dists_raw  = np.abs(lons - city_lon_adj)
        dists_wrap = np.abs(lons - (city_lon_adj - 360))  # wrap check near 0°/360°
        lon_idx = int(np.argmin(np.minimum(dists_raw, dists_wrap)))

        utci_mean_vals = ds[vumean].isel({lat_name: lat_idx, lon_name: lon_idx}).values.astype(np.float32)
        utci_max_vals  = ds[vumax ].isel({lat_name: lat_idx, lon_name: lon_idx}).values.astype(np.float32)
        utci_min_vals  = (ds[vumin].isel({lat_name: lat_idx, lon_name: lon_idx})
                          .values.astype(np.float32)
                          if vumin is not None
                          else np.full(len(dates), np.nan, dtype=np.float32))

        # ERA5-HEAT UTCI is stored in Kelvin — convert to °C
        # (no units attribute in file; confirmed by sample values ~290 K)
        if utci_mean_vals.mean() > 100:
            utci_mean_vals -= 273.15
            utci_max_vals  -= 273.15
            utci_min_vals  -= 273.15

        records.append(pd.DataFrame({
            "city_id":   row["city_id"],
            "date":      dates,
            "utci_mean": utci_mean_vals,
            "utci_max":  utci_max_vals,
            "utci_min":  utci_min_vals,
        }))

    ds.close()
    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Extract city pixels from one annual dewpoint NetCDF and compute Tw
# ─────────────────────────────────────────────────────────────────────────────
def extract_d2m_year(nc_path: Path, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Extract d2m_max and d2m_min at each city centroid, then compute:
      tw_max = Stull(tmax, RH) where RH = f(tmax, d2m_min)
               afternoon pairing: both Tmax and d2m_min occur in the afternoon
      tw_min = Stull(tmin, RH) where RH = f(tmin, d2m_max)
               morning pairing:   both Tmin and d2m_max occur in early morning

    NOTE: this function only returns tw_max and tw_min, not tmax/tmin — those
    come from extract_temp_year(). The caller must merge on (city_id, date).

    Returns DataFrame: city_id | date | tw_max | tw_min
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    lat_name  = next(c for c in ("latitude",  "lat")             if c in ds.coords)
    lon_name  = next(c for c in ("longitude", "lon")             if c in ds.coords)
    time_name = next(c for c in ("valid_time", "time", "date")   if c in ds.coords)

    lats  = ds[lat_name].values
    lons  = ds[lon_name].values
    dates = pd.to_datetime(ds[time_name].values)

    # Auto-detect Kelvin vs °C (ERA5 dewpoint stored in K, so mean ≈ 275 K)
    sample = float(ds["d2m_max"].isel({time_name: 0, lat_name: 360, lon_name: 720}).values)
    is_kelvin = sample > 100.0

    records = []
    for _, row in coords.iterrows():
        lat_idx      = int(np.argmin(np.abs(lats - row["lat"])))
        city_lon_adj = float(row["lon"]) % 360
        dists_raw    = np.abs(lons - city_lon_adj)
        dists_wrap   = np.abs(lons - (city_lon_adj - 360))
        lon_idx      = int(np.argmin(np.minimum(dists_raw, dists_wrap)))

        d2m_max = ds["d2m_max"].isel({lat_name: lat_idx, lon_name: lon_idx}).values.astype(np.float32)
        d2m_min = ds["d2m_min"].isel({lat_name: lat_idx, lon_name: lon_idx}).values.astype(np.float32)

        if is_kelvin:
            d2m_max -= 273.15
            d2m_min -= 273.15

        # Wet-bulb is computed later after merge with tmax/tmin from extract_temp_year.
        # Store raw d2m here; tw computation happens in main() after the merge.
        records.append(pd.DataFrame({
            "city_id":   row["city_id"],
            "date":      dates,
            "d2m_max_c": d2m_max,
            "d2m_min_c": d2m_min,
        }))

    ds.close()
    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Years and output path are parameterised so the panel can be extended
    # without editing code. Defaults reproduce the original 2019-2021 run and
    # the original file name, so nothing already on disk is invalidated.
    # Declared up front: the argparse defaults below READ these module-level
    # names, and Python forbids `global` after a name is used in the scope.
    global STUDY_YEARS, OUT_CLIMATE

    import argparse
    ap = argparse.ArgumentParser(
        description="Extract daily city-level ERA5 climate for the FUA sample")
    ap.add_argument("--years", nargs="+", type=int, default=STUDY_YEARS,
                    help="Years to extract, e.g. --years 2019 2020 2021 2022")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output parquet. Default: outputs/city_climate_<first>_<last>.parquet "
                         "(or the original name when the default years are used).")
    args = ap.parse_args()

    STUDY_YEARS = sorted(args.years)
    if args.out is not None:
        OUT_CLIMATE = args.out
    elif STUDY_YEARS != [2019, 2020, 2021]:
        OUT_CLIMATE = OUT_DIR / (
            f"city_climate_{STUDY_YEARS[0]}_{STUDY_YEARS[-1]}.parquet")

    print("=" * 65)
    print(f"ERA5 CITY-LEVEL EXTRACTION  (years {STUDY_YEARS[0]}-{STUDY_YEARS[-1]})")
    print(f"  output: {OUT_CLIMATE.name}")
    print("=" * 65)

    if OUT_CLIMATE.exists():
        print(f"\n  ✓ {OUT_CLIMATE.name} already exists.")
        print("  Delete it and rerun to regenerate.")
        df = pd.read_parquet(OUT_CLIMATE)
        print(f"  Shape: {df.shape}  |  cities: {df['city_id'].nunique()}")
        print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"  Columns: {list(df.columns)}")
        return

    # Load city centroids
    print("\n[1] Loading city centroids …")
    coords = load_city_coords()

    temp_parts = []
    utci_parts = []
    d2m_parts  = []

    for year in STUDY_YEARS:
        temp_nc = Path(str(TEMP_TEMPLATE).format(year=year))
        utci_nc = Path(str(UTCI_TEMPLATE).format(year=year))
        d2m_nc  = Path(str(D2M_TEMPLATE).format(year=year))

        # ── Temperature ────────────────────────────────────────────────────
        if not temp_nc.exists():
            print(f"  ✗ Missing: {temp_nc} — skipping {year} temperature")
        else:
            t0 = time.time()
            print(f"\n[2] Extracting temperature {year} from {temp_nc.name} …")
            df_t = extract_temp_year(temp_nc, coords)
            temp_parts.append(df_t)
            mb = temp_nc.stat().st_size / 1e6
            print(f"    {year}: {len(df_t):,} city-day rows  |  file {mb:.0f} MB  "
                  f"({time.time()-t0:.0f}s)")

        # ── UTCI ───────────────────────────────────────────────────────────
        if not utci_nc.exists():
            print(f"  ✗ Missing: {utci_nc} — skipping {year} UTCI")
        else:
            t0 = time.time()
            print(f"[3] Extracting UTCI      {year} from {utci_nc.name} …")
            df_u = extract_utci_year(utci_nc, coords)
            utci_parts.append(df_u)
            mb = utci_nc.stat().st_size / 1e6
            print(f"    {year}: {len(df_u):,} city-day rows  |  file {mb:.0f} MB  "
                  f"({time.time()-t0:.0f}s)")

        # ── Dewpoint (optional; generated by 05e) ──────────────────────────
        if not d2m_nc.exists():
            print(f"  ⚠ Missing: {d2m_nc} — tw_max/tw_min will be NaN for {year}")
            print(f"    Run: python 05e_download_era5_dewpoint.py --only-study")
        else:
            t0 = time.time()
            print(f"[4] Extracting dewpoint  {year} from {d2m_nc.name} …")
            df_d = extract_d2m_year(d2m_nc, coords)
            d2m_parts.append(df_d)
            mb = d2m_nc.stat().st_size / 1e6
            print(f"    {year}: {len(df_d):,} city-day rows  |  file {mb:.0f} MB  "
                  f"({time.time()-t0:.0f}s)")

    if not temp_parts:
        sys.exit("No temperature data extracted — check ERA5/temperature/ directory.")

    # Combine temperature across years
    temp_all = pd.concat(temp_parts, ignore_index=True)
    temp_all["date"] = pd.to_datetime(temp_all["date"]).dt.normalize()

    # Combine UTCI across years and merge
    if utci_parts:
        utci_all = pd.concat(utci_parts, ignore_index=True)
        utci_all["date"] = pd.to_datetime(utci_all["date"]).dt.normalize()
        climate = temp_all.merge(utci_all, on=["city_id", "date"], how="left")
    else:
        climate = temp_all.copy()
        climate["utci_mean"] = np.nan
        climate["utci_max"]  = np.nan
        climate["utci_min"]  = np.nan

    # Merge dewpoint and compute wet-bulb temperatures
    if d2m_parts:
        d2m_all = pd.concat(d2m_parts, ignore_index=True)
        d2m_all["date"] = pd.to_datetime(d2m_all["date"]).dt.normalize()
        climate = climate.merge(d2m_all, on=["city_id", "date"], how="left")
        # tw_max: afternoon pairing (Tmax + d2m_min)
        rh_day   = _compute_rh(climate["tmax"].values, climate["d2m_min_c"].values)
        # tw_min: morning pairing (Tmin + d2m_max)
        rh_night = _compute_rh(climate["tmin"].values, climate["d2m_max_c"].values)
        climate["tw_max"] = _stull_wetbulb(climate["tmax"].values, rh_day)
        climate["tw_min"] = _stull_wetbulb(climate["tmin"].values, rh_night)
        # Drop raw d2m columns — tw_max/tw_min carry the information forward
        climate = climate.drop(columns=["d2m_max_c", "d2m_min_c"])
    else:
        climate["tw_max"] = np.nan
        climate["tw_min"] = np.nan

    climate = climate.sort_values(["city_id", "date"]).reset_index(drop=True)

    # Sanity checks
    print("\n[5] Validation …")
    print(f"  Rows:    {len(climate):,}")
    print(f"  Cities:  {climate['city_id'].nunique()}")
    print(f"  Dates:   {climate['date'].min().date()} → {climate['date'].max().date()}")
    print(f"  Columns: {list(climate.columns)}")
    print(f"  tmax range:     {climate['tmax'].min():.1f} – {climate['tmax'].max():.1f} °C")
    print(f"  tmin range:     {climate['tmin'].min():.1f} – {climate['tmin'].max():.1f} °C")
    if "utci_mean" in climate.columns and climate["utci_mean"].notna().any():
        print(f"  utci_mean range:{climate['utci_mean'].min():.1f} – "
              f"{climate['utci_mean'].max():.1f} °C")
    if climate["tw_max"].notna().any():
        print(f"  tw_max range:   {climate['tw_max'].min():.1f} – "
              f"{climate['tw_max'].max():.1f} °C")
        print(f"  tw_min range:   {climate['tw_min'].min():.1f} – "
              f"{climate['tw_min'].max():.1f} °C")
    else:
        print("  tw_max/tw_min:  NaN (run 05e --only-study first)")

    missing_tmax = climate["tmax"].isna().sum()
    if missing_tmax > 0:
        print(f"  Warning: {missing_tmax} rows with NaN tmax")

    # Save
    climate.to_parquet(OUT_CLIMATE, index=False)
    size_mb = OUT_CLIMATE.stat().st_size / 1e6
    print(f"\n  ✓ Saved → {OUT_CLIMATE.name}  ({size_mb:.1f} MB)")

    print("\n" + "=" * 65)
    print("NEXT steps:")
    print("  python 04_define_events.py   (uses tw_max + tw_min for humid heat)")
    print("=" * 65)


if __name__ == "__main__":
    main()
