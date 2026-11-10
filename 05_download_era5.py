"""
05_download_era5.py
===================
Download ERA5 climate data for the heat-carbon study.
Targets: daily Tmax, Tmin, Tmean      (for heatwave/cold wave detection)
         daily UTCI mean, max and min (thermal stress regressors)

Resolution decision: DAILY ONLY — no hourly data needed.
  - Carbon Monitor Cities data is daily; sub-daily ERA5 adds no analytical value.
  - Day/night split is impossible without sub-daily carbon data.
  - Pre-computed daily statistics keep retained storage modest.

Storage estimate scales with the active year list, including --years.
For the default 2019–2021 period (global 0.25°):
  Temperature (Tmax + Tmin + Tmean):   ~4–5 GB
  UTCI (hourly source → daily stats):  temporary monthly data + ~2–3 GB final
  ─────────────────────────────────────────────
  TOTAL:                               ~10–12 GB  (well within 200 GB budget)

PRE-REQUISITES — MUST DO MANUALLY BEFORE RUNNING:
  ① Accept Terms of Use for temperature dataset:
     https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics
  ② Accept Terms of Use for UTCI dataset:
     https://cds.climate.copernicus.eu/datasets/derived-utci-historical
  ③ Install download and NetCDF processing packages:
     pip install "cdsapi>=0.7.7" xarray netcdf4 "dask[array]"

Run:
  python 05_download_era5.py
  python 05_download_era5.py --only-temp   # temperature only
  python 05_download_era5.py --only-utci   # UTCI only
  python 05_download_era5.py --postprocess # aggregate UTCI to daily stats (no download)
"""

import os
import sys
import argparse
import time
import calendar
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

# Keep progress symbols readable in Windows terminals that default to GBK/cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
ERA5_DIR = BASE / "ERA5"
TEMP_DIR = ERA5_DIR / "temperature"
UTCI_DIR = ERA5_DIR / "utci"
TEMP_CHUNK_DIR = TEMP_DIR / "monthly"
UTCI_CHUNK_DIR = UTCI_DIR / "monthly"

# Study years. Kept as a module-level default so every existing call site and
# the 2019-2021 outputs already on disk are unchanged; `--years` overrides it at
# run time so extending the panel needs no code edit. main() rebinds this name
# before any downloader function runs.
YEARS  = ["2019", "2020", "2021"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]

TEMP_STATISTICS = {
    "daily_max": "t2m_max",
    "daily_min": "t2m_min",
    "daily_mean": "t2m_mean",
}


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_directories():
    for d in [ERA5_DIR, TEMP_DIR, UTCI_DIR, TEMP_CHUNK_DIR, UTCI_CHUNK_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  Output directories ready under: {ERA5_DIR}")


def setup_cdsapirc():
    """Verify that CDS credentials are stored outside the source code."""
    rc_path = Path.home() / ".cdsapirc"
    if rc_path.exists():
        print("  ✓ ~/.cdsapirc found")
        return
    sys.exit(
        f"CDS credentials not found: {rc_path}\n"
        "Copy your personal API configuration from:\n"
        "https://cds.climate.copernicus.eu/how-to-api"
    )


def get_client():
    try:
        import cdsapi
        return cdsapi.Client()
    except ImportError:
        sys.exit('Install cdsapi first:  pip install "cdsapi>=0.7.7"')
    except Exception as e:
        sys.exit(f"CDS API error: {e}\n"
                 "Make sure ~/.cdsapirc exists with correct key.")


def valid_days(year: str, month: str):
    """Return only the calendar days that exist in a given month."""
    n_days = calendar.monthrange(int(year), int(month))[1]
    return [f"{day:02d}" for day in range(1, n_days + 1)]


def retrieve_chunk(client, dataset: str, request: dict, out: Path):
    """Download atomically so an interrupted request is never treated as complete."""
    download = out.with_suffix(".download")
    partial = out.with_suffix(".part.nc")
    download.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    try:
        client.retrieve(dataset, request, str(download))
        if not download.exists() or download.stat().st_size == 0:
            raise RuntimeError(f"CDS returned an empty file for {out.name}")

        # The daily-statistics dataset currently always returns a ZIP archive,
        # even when a NetCDF filename is supplied as the target.
        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as archive:
                members = [
                    member for member in archive.namelist()
                    if member.lower().endswith((".nc", ".nc4", ".netcdf"))
                ]
                if len(members) != 1:
                    raise RuntimeError(
                        f"Expected one NetCDF file in CDS archive, found {len(members)}"
                    )
                with archive.open(members[0]) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target)
        else:
            download.replace(partial)

        partial.replace(out)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        download.unlink(missing_ok=True)


def combine_monthly_files(files, out: Path, label: str):
    """Combine monthly NetCDF chunks without loading the full global data into RAM."""
    try:
        import xarray as xr
        import dask  # noqa: F401 - required by xarray.open_mfdataset
        import netCDF4  # noqa: F401 - NetCDF4 output engine
    except ImportError:
        sys.exit(
            "\nMonthly downloads completed, but the annual files still need merging.\n"
            "Install the processing packages in this environment, then rerun:\n"
            "  pip install xarray netcdf4 dask\n"
        )

    if out.exists() and out.stat().st_size > 0:
        print(f"  ✓ {out.name} already exists — skipping merge")
        return out

    print(f"  Combining {len(files)} monthly {label} files → {out.name} …")
    ds = xr.open_mfdataset(
        [str(path) for path in files],
        combine="by_coords",
        parallel=False,
        chunks="auto",
    )

    time_name = next(
        (name for name in ("valid_time", "time", "date") if name in ds.coords),
        None,
    )
    if time_name is not None:
        ds = ds.sortby(time_name)

    partial = out.with_suffix(".part.nc")
    partial.unlink(missing_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, variable in ds.data_vars.items()
        if variable.dtype.kind in "iuf"
    }
    try:
        ds.to_netcdf(partial, engine="netcdf4", encoding=encoding)
        partial.replace(out)
    finally:
        ds.close()
        partial.unlink(missing_ok=True)

    size_mb = out.stat().st_size / 1e6
    print(f"  ✓ Saved → {out.name} ({size_mb:.0f} MB)")
    return out


def find_time_name(obj):
    """Handle the time-coordinate names used by different CDS NetCDF converters."""
    for name in ("valid_time", "time", "date"):
        if name in obj.coords or name in obj.dims:
            return name
    raise ValueError(
        f"No time coordinate found. Coordinates: {list(obj.coords)}; "
        f"dimensions: {list(obj.dims)}"
    )


def rename_temperature_variable(path: Path, output_name: str, statistic: str):
    """Give each downloaded statistic a unique variable name before merging."""
    import xarray as xr

    with xr.open_dataset(path) as source:
        time_name = find_time_name(source)
        candidates = [
            name for name, variable in source.data_vars.items()
            if time_name in variable.dims and variable.dtype.kind in "iuf"
        ]
        if output_name in candidates:
            return
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot identify the temperature variable in {path.name}; "
                f"data variables are {list(source.data_vars)}"
            )
        ds = source.rename({candidates[0]: output_name})
        ds[output_name].attrs["daily_statistic"] = statistic
        partial = path.with_suffix(".normalized.nc")
        partial.unlink(missing_ok=True)
        try:
            ds.to_netcdf(
                partial,
                engine="netcdf4",
                encoding={output_name: {"zlib": True, "complevel": 4}},
            )
        finally:
            ds.close()
    partial.replace(path)


def download_and_aggregate_utci_month(
    client,
    year: str,
    month: str,
    out: Path,
):
    """
    Download one month of hourly ERA5-HEAT and immediately aggregate it daily.

    The current CDS UTCI dataset does not support a time-of-day selection and
    normally returns an archive containing one or more NetCDF files.
    """
    import xarray as xr

    download = out.with_suffix(".download")
    partial = out.with_suffix(".part.nc")
    download.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)

    request = {
        "variable": ["universal_thermal_climate_index"],
        "version": "1_1",
        "product_type": "consolidated_dataset",
        "year": [year],
        "month": [month],
        "day": valid_days(year, month),
    }

    try:
        client.retrieve("derived-utci-historical", request, str(download))
        if not download.exists() or download.stat().st_size == 0:
            raise RuntimeError(f"CDS returned an empty file for {year}-{month}")

        # Keep extraction on the ERA5 drive because a monthly archive can be large.
        with tempfile.TemporaryDirectory(
            prefix=f"utci_{year}_{month}_",
            dir=UTCI_CHUNK_DIR,
        ) as temp_dir:
            temp_dir = Path(temp_dir)
            netcdf_files = []

            if zipfile.is_zipfile(download):
                with zipfile.ZipFile(download) as archive:
                    members = [
                        member for member in archive.infolist()
                        if not member.is_dir()
                        and member.filename.lower().endswith((".nc", ".nc4", ".netcdf"))
                    ]
                    for index, member in enumerate(members):
                        extracted = temp_dir / f"{index:04d}_{Path(member.filename).name}"
                        with archive.open(member) as source, extracted.open("wb") as target:
                            shutil.copyfileobj(source, target)
                        netcdf_files.append(extracted)
            elif tarfile.is_tarfile(download):
                with tarfile.open(download, "r:*") as archive:
                    members = [
                        member for member in archive.getmembers()
                        if member.isfile()
                        and member.name.lower().endswith((".nc", ".nc4", ".netcdf"))
                    ]
                    for index, member in enumerate(members):
                        extracted = temp_dir / f"{index:04d}_{Path(member.name).name}"
                        source = archive.extractfile(member)
                        if source is None:
                            continue
                        with source, extracted.open("wb") as target:
                            shutil.copyfileobj(source, target)
                        netcdf_files.append(extracted)
            else:
                extracted = temp_dir / "utci.nc"
                shutil.copyfile(download, extracted)
                netcdf_files.append(extracted)

            if not netcdf_files:
                raise RuntimeError(
                    f"No NetCDF files were found in the UTCI response for {year}-{month}"
                )

            ds = xr.open_mfdataset(
                [str(path) for path in netcdf_files],
                combine="by_coords",
                parallel=False,
                chunks="auto",
            )
            try:
                utci_name = next(
                    (name for name in ds.data_vars if "utci" in name.lower()),
                    None,
                )
                if utci_name is None:
                    raise RuntimeError(
                        f"Cannot find UTCI variable; available: {list(ds.data_vars)}"
                    )

                utci = ds[utci_name]
                time_name = find_time_name(utci)
                utci = utci.sortby(time_name)
                daily = utci.resample({time_name: "1D"})
                ds_daily = xr.Dataset({
                    "utci_mean": daily.mean(),
                    "utci_max": daily.max(),
                    "utci_min": daily.min(),
                })
                encoding = {
                    name: {"zlib": True, "complevel": 4, "dtype": "float32"}
                    for name in ds_daily.data_vars
                }
                ds_daily.to_netcdf(
                    partial,
                    engine="netcdf4",
                    encoding=encoding,
                )
                ds_daily.close()
            finally:
                ds.close()

        partial.replace(out)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        download.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: ERA5 Daily Temperature Statistics
# Dataset: derived-era5-single-levels-daily-statistics
# Provides: pre-computed Tmax, Tmin, Tmean from hourly ERA5 — NO hourly DL needed
# Output: ERA5/temperature/era5_temp_daily_{year}.nc
# ─────────────────────────────────────────────────────────────────────────────
def download_temperature(client, year: str):
    out = TEMP_DIR / f"era5_temp_daily_{year}.nc"
    if out.exists() and out.stat().st_size > 0:
        size_mb = out.stat().st_size / 1e6
        print(f"  ✓ {out.name} already exists ({size_mb:.0f} MB) — skipping")
        return out

    print(
        f"\n  Requesting ERA5 daily temperature stats: {year} "
        f"(12 months × {len(TEMP_STATISTICS)} statistics) …"
    )
    t0 = time.time()
    monthly_files = []
    for month in MONTHS:
        for statistic, output_name in TEMP_STATISTICS.items():
            chunk = TEMP_CHUNK_DIR / (
                f"era5_temp_{output_name}_{year}_{month}.nc"
            )
            monthly_files.append(chunk)
            if chunk.exists() and chunk.stat().st_size > 0:
                rename_temperature_variable(chunk, output_name, statistic)
                print(f"    ✓ {year}-{month} {statistic} already downloaded — skipping")
                continue

            request = {
                "product_type": "reanalysis",
                "variable": ["2m_temperature"],
                "year": year,
                "month": [month],
                "day": valid_days(year, month),
                "daily_statistic": statistic,
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
            }
            print(f"    ↓ {year}-{month} {statistic} …")
            try:
                retrieve_chunk(
                    client,
                    "derived-era5-single-levels-daily-statistics",
                    request,
                    chunk,
                )
                rename_temperature_variable(chunk, output_name, statistic)
            except Exception as e:
                print(f"  ✗ Failed for {year}-{month} {statistic}: {e}")
                print(
                    "\n  If the message says 'Terms and conditions not accepted', visit:\n"
                    "  https://cds.climate.copernicus.eu/datasets/"
                    "derived-era5-single-levels-daily-statistics\n"
                    "  Otherwise rerun later; completed chunks will be skipped."
                )
                raise

    result = combine_monthly_files(monthly_files, out, "temperature")
    elapsed = time.time() - t0
    print(f"  ✓ Temperature {year} complete ({elapsed/60:.1f} min)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ERA5-HEAT UTCI
# Dataset: derived-utci-historical
# Provides: Universal Thermal Climate Index (UTCI) based on ERA5 inputs
#           (T2m, wind, solar radiation, humidity — accounts for compound
#            heat-humidity effects automatically)
# Strategy: Download each month's hourly source data, immediately aggregate it
#           to daily mean/max/min, and delete the temporary hourly archive.
# Output: ERA5/utci/utci_daily_{year}.nc
# ─────────────────────────────────────────────────────────────────────────────
def download_utci(client, year: str):
    out = UTCI_DIR / f"utci_daily_{year}.nc"
    if out.exists() and out.stat().st_size > 0:
        size_mb = out.stat().st_size / 1e6
        print(f"  ✓ {out.name} already exists ({size_mb:.0f} MB) — skipping")
        return out

    print(f"\n  Requesting ERA5-HEAT UTCI: {year} (12 monthly chunks) …")
    print("  CDS provides all hourly values; each month is aggregated immediately")
    print("  to daily mean/max/min, then the temporary hourly files are removed.")
    t0 = time.time()
    monthly_files = []
    for month in MONTHS:
        chunk = UTCI_CHUNK_DIR / f"utci_daily_{year}_{month}.nc"
        monthly_files.append(chunk)
        if chunk.exists() and chunk.stat().st_size > 0:
            print(f"    ✓ {year}-{month} already downloaded — skipping")
            continue

        print(f"    ↓ {year}-{month} hourly → daily …")
        try:
            download_and_aggregate_utci_month(client, year, month, chunk)
        except Exception as e:
            print(f"  ✗ Failed for {year}-{month}: {e}")
            print(
                "\n  If the message says 'Terms and conditions not accepted', visit:\n"
                "  https://cds.climate.copernicus.eu/datasets/derived-utci-historical\n"
                "  Otherwise rerun later; completed months will be skipped."
            )
            raise

    result = combine_monthly_files(monthly_files, out, "daily UTCI")
    elapsed = time.time() - t0
    print(f"  ✓ UTCI {year} complete ({elapsed/60:.1f} min)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Post-process UTCI → daily mean and max
# Aggregates 4 daily timesteps (00/06/12/18) to daily mean and daily max.
# Output: ERA5/utci/utci_daily_{year}.nc
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_utci(year: str):
    try:
        import xarray as xr
        import numpy as np
    except ImportError:
        sys.exit("Install xarray:  pip install xarray netcdf4")

    raw   = UTCI_DIR / f"utci_6h_{year}.nc"
    out   = UTCI_DIR / f"utci_daily_{year}.nc"

    if not raw.exists():
        print(f"  ✗ Raw UTCI file not found: {raw.name} — run download first")
        return

    if out.exists():
        print(f"  ✓ {out.name} already exists — skipping post-process")
        return

    print(f"  Post-processing UTCI {year}: 6-hourly → daily mean + max …")
    ds = xr.open_dataset(raw)

    # Identify the UTCI variable (name varies: 'utci', 'UTCI', 'utci_mean')
    utci_var = None
    for v in ds.data_vars:
        if "utci" in v.lower():
            utci_var = v
            break
    if utci_var is None:
        print(f"  ✗ Cannot find UTCI variable. Available: {list(ds.data_vars)}")
        return

    utci = ds[utci_var]

    # Determine time dimension name (new CDS files commonly use "valid_time")
    time_dim = find_time_name(utci)

    # Resample to daily
    utci_daily = utci.resample({time_dim: "1D"})
    ds_daily = xr.Dataset({
        "utci_mean": utci_daily.mean(),
        "utci_max":  utci_daily.max(),
        "utci_min":  utci_daily.min(),
    })

    # Encode for compression
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds_daily.data_vars}
    ds_daily.to_netcdf(out, encoding=encoding)
    size_mb = out.stat().st_size / 1e6
    print(f"  ✓ Saved → {out.name}  ({size_mb:.0f} MB)")
    ds.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Validate downloaded files
# ─────────────────────────────────────────────────────────────────────────────
def validate_downloads():
    try:
        import xarray as xr
    except ImportError:
        print("  Install xarray to validate:  pip install xarray netcdf4")
        return

    print("\n[Validation]")
    all_ok = True

    # Temperature
    for year in YEARS:
        f = TEMP_DIR / f"era5_temp_daily_{year}.nc"
        if not f.exists():
            print(f"  ✗ MISSING: {f.name}")
            all_ok = False
            continue
        try:
            ds = xr.open_dataset(f)
            n_days = ds.sizes[find_time_name(ds)]
            vars_found = list(ds.data_vars)
            size_mb = f.stat().st_size / 1e6
            print(f"  ✓ {f.name}: {n_days} days, vars={vars_found}, {size_mb:.0f} MB")
            ds.close()
        except Exception as e:
            print(f"  ✗ {f.name}: corrupt or unreadable ({e})")
            all_ok = False

    # UTCI (daily)
    for year in YEARS:
        f = UTCI_DIR / f"utci_daily_{year}.nc"
        if not f.exists():
            print(f"  ✗ MISSING: {f.name} (run --postprocess)")
            all_ok = False
            continue
        try:
            ds = xr.open_dataset(f)
            n_days = ds.sizes[find_time_name(ds)]
            size_mb = f.stat().st_size / 1e6
            print(f"  ✓ {f.name}: {n_days} days, vars={list(ds.data_vars)}, {size_mb:.0f} MB")
            ds.close()
        except Exception as e:
            print(f"  ✗ {f.name}: corrupt or unreadable ({e})")
            all_ok = False

    # Storage summary
    total_mb = sum(
        f.stat().st_size for f in ERA5_DIR.rglob("*.nc") if f.exists()
    ) / 1e6
    print(f"\n  Total ERA5 storage used: {total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
    if all_ok:
        print("  ✓ All files validated successfully")
    else:
        print("  ⚠ Some files are missing — re-run download")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Storage dry-run estimate
# ─────────────────────────────────────────────────────────────────────────────
def print_storage_estimate():
    low_gb = 3.7 * len(YEARS)
    high_gb = 5.0 * len(YEARS)
    budget_gb = 200.0
    remaining_low = budget_gb - high_gb
    remaining_high = budget_gb - low_gb

    print("\n" + "─" * 55)
    print(f"STORAGE ESTIMATE ({YEARS[0]}–{YEARS[-1]}, global 0.25°)")
    print("─" * 55)
    print("  Temperature daily stats (Tmax+Tmin+Tmean):")
    print(f"    3 stats × 365 days × {len(YEARS)} years × ~1.5 MB   "
          f"≈  {1.5 * len(YEARS):.0f}–{1.7 * len(YEARS):.0f} GB")
    print("  UTCI hourly source:")
    print("    Downloaded one month at a time and deleted after aggregation")
    print("  UTCI daily output (mean+max+min):")
    print(f"    3 stats × 365 days × {len(YEARS)} years × ~0.8 MB   "
          f"≈  {0.8 * len(YEARS):.0f}–{1.0 * len(YEARS):.0f} GB")
    print("─" * 55)
    print(f"  TOTAL (all files):                          "
          f"≈  {low_gb:.0f}–{high_gb:.0f} GB")
    print(f"  Your budget:                                   {budget_gb:.0f} GB")
    print(f"  Remaining after download:               ≈  "
          f"{remaining_low:.0f}–{remaining_high:.0f} GB ✓")
    print("─" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download ERA5 data for heat-carbon study")
    parser.add_argument("--only-temp",    action="store_true", help="Download temperature only")
    parser.add_argument("--only-utci",    action="store_true", help="Download UTCI only")
    parser.add_argument("--postprocess",  action="store_true", help="Post-process UTCI only (no download)")
    parser.add_argument("--validate",     action="store_true", help="Validate existing files only")
    parser.add_argument("--dry-run",      action="store_true", help="Show storage estimate only")
    parser.add_argument("--years", nargs="+", default=None,
                        help="Years to download, e.g. --years 2022. "
                             "Default: 2019 2020 2021 (the original study period).")
    args = parser.parse_args()

    if args.years:
        global YEARS
        YEARS = [str(y) for y in args.years]

    print("=" * 55)
    print("ERA5 DOWNLOAD — Heat-Carbon Study")
    print(f"Years: {', '.join(YEARS)}")
    print(f"Target: {ERA5_DIR}")
    print("=" * 55)

    if args.dry_run:
        print_storage_estimate()
        return

    if args.validate:
        validate_downloads()
        return

    # Setup
    setup_directories()
    setup_cdsapirc()
    print_storage_estimate()

    if args.postprocess:
        print("\n[Post-processing UTCI → daily stats]")
        for year in YEARS:
            postprocess_utci(year)
        validate_downloads()
        return

    client = get_client()

    # Download temperature
    if not args.only_utci:
        print("\n" + "═" * 55)
        print("[1/2] ERA5 DAILY TEMPERATURE STATISTICS")
        print("      Variables: Tmax, Tmin, Tmean  |  UTC daily")
        print("═" * 55)
        print("\n  ⚠ If this fails with 'terms not accepted', visit:")
        print("    https://cds.climate.copernicus.eu/datasets/"
              "derived-era5-single-levels-daily-statistics")
        for year in YEARS:
            download_temperature(client, year)

    # Download UTCI
    if not args.only_temp:
        print("\n" + "═" * 55)
        print("[2/2] ERA5-HEAT UTCI (Universal Thermal Climate Index)")
        print("      Hourly source → daily mean/max/min")
        print("═" * 55)
        print("\n  ⚠ If this fails with 'terms not accepted', visit:")
        print("    https://cds.climate.copernicus.eu/datasets/derived-utci-historical")
        for year in YEARS:
            download_utci(client, year)

    # Validate everything
    validate_downloads()

    print("\n" + "=" * 55)
    print("NEXT STEPS")
    print("=" * 55)
    print("  Files ready in ERA5/temperature/ and ERA5/utci/")
    print("  Run 04_define_events.py to flag heatwave / cold wave days")
    print("  ERA5 file paths expected by 04_define_events.py:")
    print(f"    {TEMP_DIR}/era5_temp_daily_YYYY.nc")
    print(f"    {UTCI_DIR}/utci_daily_YYYY.nc")
    print("=" * 55)


if __name__ == "__main__":
    main()
