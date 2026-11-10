#!/usr/bin/env python
"""
05g_download_era5_hourly_utci.py
================================

Download hourly ERA5-HEAT UTCI source data and reduce it to FUA-polygon city
means, one month at a time.

This is the UTCI counterpart to 05f_download_era5_hourly_t2m.py.  The existing
05_download_era5.py keeps only global daily UTCI files; it downloads hourly
source data temporarily, aggregates to daily mean/max/min, and deletes the
hourly source.  That is not enough when downstream exposure models need
city-hour UTCI.

OUTPUT
------
  ERA5/processed_hourly_polygon/utci/{year}/era5_utci_{yyyy}-{mm}_city_hour.parquet
      city_id | datetime (UTC) | utci (degC) | n_grid_cells
  ERA5/processed_daily_polygon/utci/{year}/era5_utci_{yyyy}-{mm}_city_day.parquet
      city_id | date | utci_mean | utci_max | utci_min | n_grid_cells

Safe to interrupt and rerun: completed months are skipped.

USAGE
-----
  python 05g_download_era5_hourly_utci.py --years 2022 2023 2024
  python 05g_download_era5_hourly_utci.py --years 2024 --months 7
  python 05g_download_era5_hourly_utci.py --dry-run
  python 05g_download_era5_hourly_utci.py --keep-raw
"""

from __future__ import annotations

import argparse
import calendar
import importlib.util
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ERA5_DIR = BASE / "ERA5"
RAW_DIR = ERA5_DIR / "raw_hourly"
CACHE_DIR = ERA5_DIR / "cache"
UTCI_DIR = ERA5_DIR / "utci"
UTCI_CHUNK_DIR = UTCI_DIR / "monthly"
DEFAULT_SHPFILE = BASE / "ShapefileData" / "FUA_cities_final.shp"

CDS_DATASET = "derived-utci-historical"
INDEX = "utci"
YEARS = [2019, 2020, 2021]

# Conservative sanity check. A global monthly hourly UTCI NetCDF should be far
# larger than this, but the threshold is intentionally low to avoid false
# failures if CDS changes compression.
MIN_EXPECTED_BYTES = 50 * 1024 * 1024


def load_chi_module():
    """Import 07_download_chi_indices.py (filename starts with a digit)."""
    path = BASE / "07_download_chi_indices.py"
    if not path.exists():
        sys.exit(f"Required helper module not found: {path}")
    spec = importlib.util.spec_from_file_location("chi_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # safe: 07 guards with __main__
    return module


def raw_file(year: int, month: int) -> Path:
    return RAW_DIR / INDEX / f"{year}" / f"era5_{INDEX}_{year}-{month:02d}.nc"


def hourly_file(year: int, month: int) -> Path:
    return (
        ERA5_DIR / "processed_hourly_polygon" / INDEX / f"{year}"
        / f"era5_{INDEX}_{year}-{month:02d}_city_hour.parquet"
    )


def daily_file(year: int, month: int) -> Path:
    return (
        ERA5_DIR / "processed_daily_polygon" / INDEX / f"{year}"
        / f"era5_{INDEX}_{year}-{month:02d}_city_day.parquet"
    )


def global_daily_month_file(year: int, month: int) -> Path:
    return UTCI_CHUNK_DIR / f"utci_daily_{year}_{month:02d}.nc"


def global_daily_year_file(year: int) -> Path:
    return UTCI_DIR / f"utci_daily_{year}.nc"


def get_client():
    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        sys.exit(
            f"CDS credentials not found: {rc}\n"
            "Copy your personal API configuration from "
            "https://cds.climate.copernicus.eu/how-to-api"
        )
    try:
        import cdsapi
    except ImportError:
        sys.exit('Install the CDS client first:  pip install "cdsapi>=0.7.7"')
    try:
        return cdsapi.Client()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not initialise the CDS client: {exc}")


def build_request(year: int, month: int) -> dict:
    n_days = calendar.monthrange(year, month)[1]
    return {
        "variable": ["universal_thermal_climate_index"],
        "version": "1_1",
        "product_type": "consolidated_dataset",
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, n_days + 1)],
    }


def find_time_name(ds) -> str:
    for name in ("valid_time", "time", "date"):
        if name in ds.coords or name in ds.dims:
            return name
    raise RuntimeError(f"No time coordinate found. Coordinates: {list(ds.coords)}")


def write_combined_netcdf(netcdf_files: list[Path], partial: Path) -> None:
    """Write one monthly NetCDF if CDS returns multiple NetCDF members."""
    import xarray as xr

    ds = xr.open_mfdataset(
        [str(path) for path in netcdf_files],
        combine="by_coords",
        parallel=False,
        chunks="auto",
    )
    try:
        time_name = find_time_name(ds)
        ds = ds.sortby(time_name)
        encoding = {
            name: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for name, variable in ds.data_vars.items()
            if variable.dtype.kind in "iuf"
        }
        ds.to_netcdf(partial, engine="netcdf4", encoding=encoding)
    finally:
        ds.close()


def detect_utci_variable(ds) -> str:
    name = next((var for var in ds.data_vars if "utci" in var.lower()), None)
    if name is None:
        raise RuntimeError(f"Cannot find UTCI variable; available: {list(ds.data_vars)}")
    return name


def write_global_daily_month(raw: Path, out: Path) -> None:
    """Aggregate one hourly global UTCI month to daily mean/max/min."""
    if out.exists() and out.stat().st_size > 0:
        return
    import xarray as xr

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".part.nc")
    partial.unlink(missing_ok=True)
    ds = xr.open_dataset(raw, chunks="auto")
    try:
        utci_name = detect_utci_variable(ds)
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
        ds_daily.to_netcdf(partial, engine="netcdf4", encoding=encoding)
        ds_daily.close()
        partial.replace(out)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        ds.close()


def combine_global_daily_year(year: int) -> None:
    """Combine 12 monthly global UTCI daily chunks into utci_daily_YYYY.nc."""
    out = global_daily_year_file(year)
    if out.exists() and out.stat().st_size > 0:
        return
    monthly = [global_daily_month_file(year, month) for month in range(1, 13)]
    if not all(path.exists() and path.stat().st_size > 0 for path in monthly):
        return
    import xarray as xr

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".part.nc")
    partial.unlink(missing_ok=True)
    ds = xr.open_mfdataset(
        [str(path) for path in monthly],
        combine="by_coords",
        parallel=False,
        chunks="auto",
    )
    try:
        time_name = find_time_name(ds)
        ds = ds.sortby(time_name)
        encoding = {
            name: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for name, variable in ds.data_vars.items()
            if variable.dtype.kind in "iuf"
        }
        ds.to_netcdf(partial, engine="netcdf4", encoding=encoding)
        partial.replace(out)
        print(f"  combined global daily UTCI year file: {out.name}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        ds.close()


def extract_archive(download: Path, temp_dir: Path) -> list[Path]:
    netcdf_files: list[Path] = []
    suffixes = (".nc", ".nc4", ".netcdf")

    if zipfile.is_zipfile(download):
        with zipfile.ZipFile(download) as archive:
            members = [
                member for member in archive.infolist()
                if not member.is_dir()
                and member.filename.lower().endswith(suffixes)
            ]
            for index, member in enumerate(members):
                extracted = temp_dir / f"{index:04d}_{Path(member.filename).name}"
                with archive.open(member) as source, extracted.open("wb") as target:
                    shutil.copyfileobj(source, target)
                netcdf_files.append(extracted)
        return netcdf_files

    if tarfile.is_tarfile(download):
        with tarfile.open(download, "r:*") as archive:
            members = [
                member for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(suffixes)
            ]
            for index, member in enumerate(members):
                extracted = temp_dir / f"{index:04d}_{Path(member.name).name}"
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, extracted.open("wb") as target:
                    shutil.copyfileobj(source, target)
                netcdf_files.append(extracted)
        return netcdf_files

    return []


def retrieve_month(client, year: int, month: int, out: Path) -> None:
    """Download one month atomically; a partial transfer never looks complete."""
    out.parent.mkdir(parents=True, exist_ok=True)
    download = out.with_suffix(".download")
    partial = out.with_suffix(".part.nc")
    download.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)

    try:
        client.retrieve(CDS_DATASET, build_request(year, month), str(download))
        if not download.exists() or download.stat().st_size == 0:
            raise RuntimeError(f"CDS returned an empty file for {out.name}")

        with tempfile.TemporaryDirectory(prefix=f"utci_{year}_{month:02d}_", dir=out.parent) as tmp:
            temp_dir = Path(tmp)
            netcdf_files = extract_archive(download, temp_dir)
            if len(netcdf_files) == 1:
                shutil.copyfile(netcdf_files[0], partial)
            elif len(netcdf_files) > 1:
                write_combined_netcdf(netcdf_files, partial)
            else:
                # CDS may return a plain NetCDF file.
                download.replace(partial)

        size = partial.stat().st_size
        if size < MIN_EXPECTED_BYTES:
            raise RuntimeError(
                f"{out.name} is only {size / 1024**2:.0f} MB, which is below "
                "the expected size for global hourly UTCI. Treating this as "
                "a truncated download."
            )
        partial.replace(out)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        download.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download hourly ERA5-HEAT UTCI and reduce it to FUA city means."
    )
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHPFILE)
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep raw monthly NetCDF instead of deleting it after processing."
    )
    parser.add_argument(
        "--no-write-hourly", dest="write_hourly", action="store_false", default=True,
        help="Write only daily parquet."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    months = [m for m in args.months if 1 <= m <= 12]
    todo: list[tuple[int, int]] = []
    for year in args.years:
        for month in months:
            h_out, d_out = hourly_file(year, month), daily_file(year, month)
            if d_out.exists() and (not args.write_hourly or h_out.exists()):
                continue
            todo.append((year, month))

    print("=" * 74)
    print("ERA5-HEAT hourly UTCI -> FUA city means")
    print("=" * 74)
    print(f"  months requested : {len(args.years) * len(months)}")
    print(f"  months remaining : {len(todo)}")
    print(f"  output root      : {ERA5_DIR}")
    print(f"  raw NetCDFs      : {'kept' if args.keep_raw else 'deleted after each month'}")
    print(f"  dataset          : {CDS_DATASET}")
    print("  output index     : utci")

    if not todo:
        print("\nNothing to do - every requested month is already processed.")
        return
    if args.dry_run:
        print("\nDry run. Months that would be processed:")
        for year, month in todo:
            print(f"  {year}-{month:02d}")
        return

    chi = load_chi_module()
    print("\nLoading FUA polygons ...")
    polygons = chi.load_fua_polygons(args.shapefile)
    print(f"  {len(polygons):,} FUA polygons loaded")

    client = get_client()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    done, failed = 0, []
    t_start = time.time()

    for idx, (year, month) in enumerate(todo, start=1):
        tag = f"{year}-{month:02d}"
        raw = raw_file(year, month)
        h_out, d_out = hourly_file(year, month), daily_file(year, month)
        print(f"\n[{idx}/{len(todo)}] {tag}")

        try:
            if raw.exists() and raw.stat().st_size >= MIN_EXPECTED_BYTES:
                print(f"  raw file already present, reusing: {raw.name}")
            else:
                for attempt in range(1, args.retries + 1):
                    try:
                        print(f"  requesting from CDS (attempt {attempt}/{args.retries}) ...")
                        retrieve_month(client, year, month, raw)
                        break
                    except Exception as exc:  # noqa: BLE001
                        if attempt == args.retries:
                            raise
                        wait = 60 * attempt
                        print(f"  CDS request failed: {exc}\n  retrying in {wait}s ...")
                        time.sleep(wait)
                print(f"  downloaded {raw.stat().st_size / 1024**3:.2f} GB")

            print("  extracting FUA polygon means ...")
            monthly_global = global_daily_month_file(year, month)
            if monthly_global.exists() and monthly_global.stat().st_size > 0:
                print(f"  global daily UTCI chunk already exists: {monthly_global.name}")
            else:
                print("  aggregating global hourly UTCI to daily mean/max/min ...")
                write_global_daily_month(raw, monthly_global)

            hourly = chi.extract_city_hourly_polygon(raw, INDEX, polygons, CACHE_DIR)
            daily = chi.daily_from_hourly(hourly, INDEX)

            if args.write_hourly:
                h_out.parent.mkdir(parents=True, exist_ok=True)
                tmp = h_out.with_suffix(".part.parquet")
                hourly.to_parquet(tmp, index=False)
                tmp.replace(h_out)
            d_out.parent.mkdir(parents=True, exist_ok=True)
            tmp = d_out.with_suffix(".part.parquet")
            daily.to_parquet(tmp, index=False)
            tmp.replace(d_out)

            print(
                f"  wrote {len(hourly):,} city-hours / {len(daily):,} city-days "
                f"({hourly['city_id'].nunique():,} cities)"
            )

            if not args.keep_raw:
                raw.unlink(missing_ok=True)
                print("  raw NetCDF deleted")

            done += 1
            combine_global_daily_year(year)
            elapsed = time.time() - t_start
            remaining = (elapsed / done) * (len(todo) - done)
            print(f"  progress {done}/{len(todo)} | est. remaining {remaining / 3600:.1f} h")

        except KeyboardInterrupt:
            print("\nInterrupted. Rerun the same command to resume.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {tag}: {exc}")
            failed.append(tag)

    print("\n" + "=" * 74)
    print(f"Completed {done}/{len(todo)} months in {(time.time() - t_start) / 3600:.1f} h")
    if failed:
        print(f"Failed months ({len(failed)}): {', '.join(failed)}")
        print("Rerun the same command to retry only the failures.")
    else:
        print("Hourly UTCI FUA parquet outputs are ready.")
    print("=" * 74)


if __name__ == "__main__":
    main()
