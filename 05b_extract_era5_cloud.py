"""
05b_extract_era5_cloud.py
=========================
Extract ERA5 climate data for 1,583 FUA cities WITHOUT downloading global files.

Strategy
--------
Temperature (Tmax / Tmin / Tmean):
    ARCO-ERA5 on Google Cloud Storage — public zarr store, no account needed.
    xarray reads only the spatial chunks covering your city pixels.
    ~0 GB local storage for raw data; output is a ~5 MB parquet file.

UTCI (Universal Thermal Climate Index):
    Unavailable in ARCO-ERA5 (requires radiation-based MRT computation).
    Downloaded from CDS with:
      · area subsetting   (bounding box per city cluster)
      · 1.0° resolution   (vs default 0.25°: 16× fewer grid cells)
    ~300–500 MB total vs ~5–7 GB for global 0.25°.

Output
------
    ERA5/city_temperature_daily.parquet   — Tmax, Tmin, Tmean per city per day
    ERA5/city_utci_daily.parquet          — UTCI mean & max per city per day
    ERA5/utci/  (small CDS files)         — raw UTCI NetCDF chunks (~300-500 MB)

PRE-REQUISITES
--------------
    pip install gcsfs zarr xarray netcdf4 dask geopandas pandas pyarrow
    pip install "cdsapi>=0.7.7"            # only needed for UTCI
    Accept Terms of Use for UTCI:
    https://cds.climate.copernicus.eu/datasets/derived-utci-historical

Run
---
    python 05b_extract_era5_cloud.py                  # full pipeline
    python 05b_extract_era5_cloud.py --only-temp      # temperature only
    python 05b_extract_era5_cloud.py --only-utci      # UTCI only
    python 05b_extract_era5_cloud.py --skip-download  # process already-downloaded UTCI
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
ERA5_DIR  = BASE / "ERA5"
UTCI_DIR  = ERA5_DIR / "utci"
SHPFILE   = BASE / "ShapefileData" / "FUA_cities_final.shp"

STUDY_START = "2019-01-01"
STUDY_END   = "2021-12-31"
YEARS       = ["2019", "2020", "2021"]
MONTHS      = [f"{m:02d}" for m in range(1, 13)]

# CDS key for UTCI download (auto-read from ~/.cdsapirc)
# UTCI at 1° resolution dramatically cuts download size
UTCI_RESOLUTION = "1.0"   # degrees; 0.25 = full resolution, 1.0 = coarse
UTCI_TIMES      = ["00:00", "06:00", "12:00", "18:00"]

# ARCO-ERA5 GCS paths (public, no authentication needed)
# Full ERA5 single-levels archive: hourly, 0.25°, 1940-present
ARCO_BUCKET = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def check_packages():
    missing = []
    for pkg in ["gcsfs", "zarr", "xarray", "geopandas", "pyarrow"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            f"Install missing packages first:\n"
            f"  pip install {' '.join(missing)} netcdf4 dask"
        )


def load_city_coords():
    """Load city centroids from FUA shapefile. Returns DataFrame with
    city_id, city, country, lon, lat."""
    import geopandas as gpd
    print(f"  Loading city polygons from {SHPFILE.name} …")
    gdf = gpd.read_file(SHPFILE)
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["city_id"] = gdf.index.astype(str).str.zfill(4)
    coords = gdf[["city_id", "city", "country", "lon", "lat"]].copy()
    print(f"  {len(coords)} cities loaded")
    return coords


def nearest_grid(value, step=0.25):
    """Snap a coordinate to the nearest ERA5 grid point."""
    return round(round(value / step) * step, 6)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Temperature via ARCO-ERA5 (zarr, cloud-native)
# ─────────────────────────────────────────────────────────────────────────────
def extract_temperature_cloud(coords_df):
    """
    Open ARCO-ERA5 from GCS, extract 2m temperature for each city centroid,
    aggregate to daily Tmax/Tmin/Tmean, return DataFrame.

    Data read from cloud: only the lat/lon chunks covering your 1,583 cities.
    No global file is downloaded to disk.
    """
    try:
        import xarray as xr
        import gcsfs
        import zarr
    except ImportError:
        sys.exit("Install: pip install xarray gcsfs zarr dask")

    out = ERA5_DIR / "city_temperature_daily.parquet"
    if out.exists():
        print(f"  ✓ {out.name} already exists — skipping temperature extraction")
        return pd.read_parquet(out)

    print(f"\n[Temperature] Opening ARCO-ERA5 from GCS …")
    print(f"  URL: {ARCO_BUCKET}")
    print(f"  (Only city-specific chunks will be fetched — no global download)")

    t0 = time.time()

    # Open zarr store anonymously (public bucket)
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ARCO_BUCKET)
    ds = xr.open_zarr(store, consolidated=True, chunks={"time": 24})

    # Select study period
    ds_period = ds.sel(time=slice(STUDY_START, STUDY_END))

    # Check available variable names (ARCO-ERA5 uses short names)
    t2m_var = None
    for candidate in ("2m_temperature", "t2m", "VAR_2T"):
        if candidate in ds_period:
            t2m_var = candidate
            break
    if t2m_var is None:
        print(f"  Available variables: {list(ds_period.data_vars)[:20]}")
        sys.exit("Cannot find 2m temperature variable in ARCO-ERA5 store")

    print(f"  Using variable: '{t2m_var}'")
    t2m = ds_period[t2m_var]

    # ARCO-ERA5 latitude may be stored south-to-north or north-to-south
    lats = t2m.latitude.values
    lons = t2m.longitude.values
    print(f"  Grid: lat {lats.min():.1f}–{lats.max():.1f}, "
          f"lon {lons.min():.1f}–{lons.max():.1f}, "
          f"n_times={len(ds_period.time)}")

    records = []
    n = len(coords_df)
    print(f"  Extracting time series for {n} cities …")

    for i, row in coords_df.iterrows():
        if i % 100 == 0:
            print(f"    {i}/{n} cities …")

        # Snap to nearest grid point
        lat_snap = nearest_grid(float(row["lat"]))
        lon_snap  = nearest_grid(float(row["lon"]))

        # Extract hourly time series for this grid point
        try:
            ts = t2m.sel(
                latitude=lat_snap, longitude=lon_snap, method="nearest"
            ).compute()   # pulls only this pixel's data from GCS
        except Exception as e:
            print(f"    Warning: city {row['city_id']} ({row['city']}) failed: {e}")
            continue

        # Convert K → °C
        ts_c = ts - 273.15

        # Aggregate to daily Tmax / Tmin / Tmean
        ts_daily = ts_c.resample(time="1D")
        tmax = ts_daily.max().values
        tmin = ts_daily.min().values
        tmean = ts_daily.mean().values
        dates = pd.date_range(STUDY_START, STUDY_END, freq="D")

        n_days = min(len(tmax), len(dates))
        df_city = pd.DataFrame({
            "city_id": row["city_id"],
            "date": dates[:n_days],
            "tmax_c": tmax[:n_days],
            "tmin_c": tmin[:n_days],
            "tmean_c": tmean[:n_days],
        })
        records.append(df_city)

    ds.close()

    result = pd.concat(records, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])

    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    elapsed = time.time() - t0
    size_mb = out.stat().st_size / 1e6
    print(f"  ✓ Saved → {out.name} ({len(result):,} rows, {size_mb:.1f} MB, "
          f"{elapsed/60:.1f} min)")
    print(f"  Local storage used: {size_mb:.1f} MB  (vs ~4–5 GB global download)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2a: UTCI download from CDS with area + resolution subsetting
# ─────────────────────────────────────────────────────────────────────────────
def compute_city_bbox(coords_df, padding=2.0):
    """Compute bounding box covering all cities, with padding in degrees."""
    n = float(coords_df["lat"].max()) + padding
    s = float(coords_df["lat"].min()) - padding
    w = float(coords_df["lon"].min()) - padding
    e = float(coords_df["lon"].max()) + padding
    # Clamp to valid range
    n = min(n, 90.0); s = max(s, -90.0)
    w = max(w, -180.0); e = min(e, 180.0)
    return n, s, w, e


def download_utci_subset(coords_df):
    """
    Download UTCI from CDS at reduced resolution + city bounding box.
    At 1.0° resolution the global grid is 360×180 = 64,800 cells vs 1,038,240
    at 0.25°. Combined with area subsetting, this is ~16–64× smaller.
    """
    UTCI_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import cdsapi
    except ImportError:
        sys.exit('Install cdsapi:  pip install "cdsapi>=0.7.7"')

    # Compute bounding box covering all FUA cities
    n, s, w, e = compute_city_bbox(coords_df)
    print(f"\n[UTCI] Bounding box: N={n:.1f} S={s:.1f} W={w:.1f} E={e:.1f}")
    print(f"       Resolution: {UTCI_RESOLUTION}° (coarser than default 0.25°)")
    print(f"       Estimated download: ~300–500 MB (vs ~5–7 GB for global 0.25°)")

    client = cdsapi.Client()

    for year in YEARS:
        out = UTCI_DIR / f"utci_{UTCI_RESOLUTION}deg_{year}.nc"
        if out.exists():
            size_mb = out.stat().st_size / 1e6
            print(f"  ✓ {out.name} already exists ({size_mb:.0f} MB) — skipping")
            continue

        print(f"\n  Requesting UTCI {year} …")
        request = {
            "product_type": "consolidated_dataset",
            "variable": "universal_thermal_climate_index",
            "year": year,
            "month": MONTHS,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": UTCI_TIMES,
            "area": [n, w, s, e],          # [N, W, S, E] — spatial subset
            "grid": [UTCI_RESOLUTION, UTCI_RESOLUTION],  # coarser resolution
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        t0 = time.time()
        try:
            client.retrieve("derived-utci-historical", request, str(out))
            elapsed = time.time() - t0
            size_mb = out.stat().st_size / 1e6
            print(f"  ✓ {out.name} ({size_mb:.0f} MB, {elapsed/60:.1f} min)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print(
                "  If 'terms not accepted': "
                "https://cds.climate.copernicus.eu/datasets/derived-utci-historical"
            )
            raise


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2b: Extract UTCI city time series from downloaded NetCDF files
# ─────────────────────────────────────────────────────────────────────────────
def extract_utci_cities(coords_df):
    """
    Open the (small) UTCI NetCDF files and extract daily mean + max
    for each city centroid. Output: ERA5/city_utci_daily.parquet
    """
    try:
        import xarray as xr
    except ImportError:
        sys.exit("Install xarray:  pip install xarray netcdf4")

    out = ERA5_DIR / "city_utci_daily.parquet"
    if out.exists():
        print(f"  ✓ {out.name} already exists — skipping UTCI extraction")
        return pd.read_parquet(out)

    utci_files = sorted(UTCI_DIR.glob(f"utci_{UTCI_RESOLUTION}deg_*.nc"))
    if not utci_files:
        sys.exit(
            f"No UTCI files found in {UTCI_DIR}\n"
            "Run without --skip-download first."
        )

    print(f"\n[UTCI] Extracting city time series from {len(utci_files)} files …")
    ds = xr.open_mfdataset(
        [str(f) for f in utci_files],
        combine="by_coords",
        parallel=False,
    )

    # Find UTCI variable name
    utci_var = next((v for v in ds.data_vars if "utci" in v.lower()), None)
    if utci_var is None:
        sys.exit(f"UTCI variable not found. Available: {list(ds.data_vars)}")

    utci = ds[utci_var]
    time_name = next((c for c in ("time", "valid_time") if c in utci.dims), utci.dims[0])

    records = []
    n = len(coords_df)
    print(f"  Extracting for {n} cities …")

    for i, row in coords_df.iterrows():
        if i % 100 == 0:
            print(f"    {i}/{n} …")
        try:
            ts = utci.sel(
                latitude=float(row["lat"]),
                longitude=float(row["lon"]),
                method="nearest",
            )
            # Aggregate 6-hourly → daily
            daily = ts.resample({time_name: "1D"})
            utci_mean = daily.mean().values
            utci_max  = daily.max().values
            dates = pd.date_range(STUDY_START, STUDY_END, freq="D")
            n_days = min(len(utci_mean), len(dates))
            df_city = pd.DataFrame({
                "city_id":   row["city_id"],
                "date":      dates[:n_days],
                "utci_mean": utci_mean[:n_days],
                "utci_max":  utci_max[:n_days],
            })
            records.append(df_city)
        except Exception as e:
            print(f"    Warning: {row['city']} — {e}")

    ds.close()

    result = pd.concat(records, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result.to_parquet(out, index=False)
    size_mb = out.stat().st_size / 1e6
    print(f"  ✓ Saved → {out.name} ({len(result):,} rows, {size_mb:.1f} MB)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Storage summary
# ─────────────────────────────────────────────────────────────────────────────
def print_storage_summary():
    print("\n" + "─" * 60)
    print("STORAGE COMPARISON")
    print("─" * 60)
    print("  Approach              Temperature    UTCI       Total")
    print("  ─────────────────────────────────────────────────────")
    print("  Original (global DL)  4–5 GB        5–7 GB     10–12 GB")
    print("  This script           ~5 MB*        300–500 MB ~0.5 GB")
    print("  ─────────────────────────────────────────────────────")
    print("  * Temperature: ARCO-ERA5 zarr on GCS, only city pixels fetched")
    print("  * UTCI: CDS download at 1° resolution, city bounding box only")
    print("─" * 60)

    # Check actual files
    temp_out = ERA5_DIR / "city_temperature_daily.parquet"
    utci_out = ERA5_DIR / "city_utci_daily.parquet"
    raw_nc = list(UTCI_DIR.glob("*.nc"))

    total_mb = 0
    for f in [temp_out, utci_out] + raw_nc:
        if f.exists():
            mb = f.stat().st_size / 1e6
            total_mb += mb
            print(f"  {f.name:<45} {mb:6.1f} MB")
    print(f"  {'TOTAL':<45} {total_mb:6.1f} MB")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract ERA5 for FUA cities via cloud zarr + small CDS download"
    )
    parser.add_argument("--only-temp",      action="store_true")
    parser.add_argument("--only-utci",      action="store_true")
    parser.add_argument("--skip-download",  action="store_true",
                        help="Skip CDS download; extract from existing UTCI files")
    args = parser.parse_args()

    print("=" * 60)
    print("ERA5 CLOUD EXTRACTION — Heat-Carbon Study")
    print(f"Period: {STUDY_START} → {STUDY_END}")
    print(f"Cities: 1,583 FUA cities")
    print("=" * 60)

    check_packages()
    print_storage_summary()

    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    UTCI_DIR.mkdir(parents=True, exist_ok=True)

    coords_df = load_city_coords()

    # Temperature: cloud zarr (no local global file)
    if not args.only_utci:
        extract_temperature_cloud(coords_df)

    # UTCI: small CDS download + extraction
    if not args.only_temp:
        if not args.skip_download:
            download_utci_subset(coords_df)
        extract_utci_cities(coords_df)

    print_storage_summary()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  Run 04_define_events.py")
    print("  It expects:")
    print(f"    {ERA5_DIR}/city_temperature_daily.parquet")
    print(f"    {ERA5_DIR}/city_utci_daily.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()
