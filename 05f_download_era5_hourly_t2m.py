#!/usr/bin/env python
"""
05f_download_era5_hourly_t2m.py
================================

Download HOURLY ERA5-Land 2 m dry-bulb air temperature (t2m) at 0.1 deg and
reduce it to FUA-polygon city means, one month at a time.

WHY THIS SCRIPT EXISTS
----------------------
`05_download_era5.py` deliberately used the *pre-aggregated* CDS product
`derived-era5-single-levels-daily-statistics` (its header says
"Resolution decision: DAILY ONLY"). That gives t2m_max / t2m_min / t2m_mean
only. The CHI archive likewise supplies no dry-bulb temperature.

Consequently the following are currently NOT computable:
  * hourly wet-bulb depression   WBD_h = T_h - Twb_h
  * sensible / latent cooling load decomposition
  * humidity ratio and moist-air enthalpy
  * local-night vs local-day degree-hour partitions

All of them need hourly dry-bulb T. This script supplies it.

WHY ONLY t2m (NO DEWPOINT)
--------------------------
Hourly Twb already exists in CHI_Data/processed_hourly_polygon/Twb/, and CHI
defines Twb = Stull(T2m, rh). Stull is monotone in rh, so rh is recovered by
inverting it once T2m is known -- exactly, and without pressure. Dewpoint
would be redundant and would double the download.

    rh  = Stull^-1(T2m, Twb)                  exact, pressure-free
    e   = e_s(T2m) * rh / 100                 vapour pressure, pressure-free
    w   = 0.622 * e / (P - e)                 humidity ratio  -> latent load
    h   = 1.006*T + w*(2501 + 1.86*T)         moist-air enthalpy

Only w and h need surface pressure; rh, e and WBD do not.
See 08_build_moist_air.py, which consumes this script's output.

ZERO-DOWNLOAD CROSS-CHECK
-------------------------
HI is also a pure function of (T2m, rh) -- Rothfusz, CHI Eq. (16). So the
already-downloaded pair {Twb, HI} is two equations in two unknowns and can
recover (T2m, rh) with no download at all. It is well conditioned only on hot
days, because below ~27 C the NWS branch of HI depends on rh only weakly, so
it is a validation tool rather than a substitute. Use it to confirm this
script's polygon-mean t2m and to quantify the Jensen gap from averaging
nonlinear indices over FUA polygons.

CONSISTENCY WITH CHI (resolved from the CHI data descriptor)
------------------------------------------------------------
Malik et al. (2026), "A Global High-Resolution Comprehensive Heat Indices
Dataset from 1950 to 2024", Scientific Data 13:200, doi:10.1038/s41597-025-06519-y

  * Methods, Input data: every CHI input variable comes from ERA5-Land
    (0.1 deg) except total sky direct solar radiation.
  * Table 3, row 5:  Twb | inputs T2m, rh | method Stull | code thermofeel.
  * Eq. (7) is the Stull (2011) expression verbatim.

So CHI Twb is the EMPIRICAL Stull wet-bulb of ERA5-Land T2m and rh -- not a
thermodynamic wet-bulb. Two consequences:

  1. This script must pull ERA5-LAND 2m_temperature at 0.1 deg so dry-bulb
     and wet-bulb share one grid.
  2. rh is recovered EXACTLY by inverting Stull for rh given (T2m, Twb), with
     no pressure term at all. See 08_build_moist_air.py --mode stull.

VOLUME
------
ERA5-Land global hourly t2m is ~19 GB/month uncompressed (1801 x 3600 x 744
x 4 bytes); CDS typically delivers 5-9 GB per month because the ocean is
masked and the file is compressed. Each month is downloaded, reduced to city
means, written to parquet, and the raw NetCDF deleted, so peak disk use stays
near one month's file. Final parquet output is ~200 MB for 2019-2021.

Budget real time for this: ERA5-Land hourly requests queue slowly at CDS.
36 months will take days, not hours.

OUTPUT
------
  ERA5/processed_hourly_polygon/t2m/{year}/era5_t2m_{yyyy}-{mm}_city_hour.parquet
      city_id | datetime (UTC) | t2m (degC) | n_grid_cells
  ERA5/processed_daily_polygon/t2m/{year}/era5_t2m_{yyyy}-{mm}_city_day.parquet
      city_id | date | t2m_mean | t2m_max | t2m_min | n_grid_cells

Schema deliberately mirrors the CHI parquet layout so downstream code can
treat t2m as a sixth index.

USAGE
-----
  python 05f_download_era5_hourly_t2m.py                    # all of 2019-2021
  python 05f_download_era5_hourly_t2m.py --years 2019       # one year
  python 05f_download_era5_hourly_t2m.py --years 2019 --months 1 2 3
  python 05f_download_era5_hourly_t2m.py --dry-run          # show plan only
  python 05f_download_era5_hourly_t2m.py --keep-raw         # keep NetCDFs

Safe to interrupt and rerun: completed months are skipped.

REQUIREMENTS
------------
  pip install "cdsapi>=0.7.7" xarray netcdf4 geopandas pyarrow
  ~/.cdsapirc with your CDS key
  Accept the licence for "ERA5 hourly data on single levels from 1940 to present":
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
"""

from __future__ import annotations

import argparse
import calendar
import importlib.util
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
ERA5_DIR = BASE / "ERA5"
RAW_DIR = ERA5_DIR / "raw_hourly"
CACHE_DIR = ERA5_DIR / "cache"
DEFAULT_SHPFILE = BASE / "ShapefileData" / "FUA_cities_final.shp"

# ERA5-Land, NOT ERA5. Malik et al. (2026), Sci Data 13:200, Methods:
# "All variables in Table 1 were obtained from ERA5-Land, except for the total
# sky direct solar radiation". CHI is therefore on the ERA5-Land 0.1 deg grid,
# and Twb = Stull(T2m, rh) uses ERA5-Land T2m. Downloading ERA5 at 0.25 deg
# would put dry-bulb on a coarser grid than wet-bulb, so WBD = T - Twb would
# carry a spatial-support artefact instead of a physical signal.
CDS_DATASET = "reanalysis-era5-land"
CDS_VARIABLE = "2m_temperature"
INDEX = "t2m"                      # column name in the output parquet
YEARS = [2019, 2020, 2021]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

# ERA5-Land global hourly t2m is ~19 GB/month uncompressed (1801 x 3600 x 744
# x 4 bytes); CDS usually delivers 5-9 GB after compression because the ocean
# is masked. Used only to detect a truncated download.
MIN_EXPECTED_BYTES = 500 * 1024 * 1024


# --------------------------------------------------------------------------
# Reuse the tested polygon-extraction machinery from 07 rather than cloning it
# --------------------------------------------------------------------------
def load_chi_module():
    """Import 07_download_chi_indices.py (name starts with a digit)."""
    path = BASE / "07_download_chi_indices.py"
    if not path.exists():
        sys.exit(f"Required helper module not found: {path}")
    spec = importlib.util.spec_from_file_location("chi_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # safe: 07 guards with __main__
    return module


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# CDS retrieval
# --------------------------------------------------------------------------
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
    except Exception as exc:                                  # noqa: BLE001
        sys.exit(f"Could not initialise the CDS client: {exc}")


def build_request(year: int, month: int) -> dict:
    # NOTE: reanalysis-era5-land takes no "product_type" key (unlike
    # reanalysis-era5-single-levels); including it makes the request fail.
    n_days = calendar.monthrange(year, month)[1]
    return {
        "variable": [CDS_VARIABLE],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, n_days + 1)],
        "time": ALL_HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


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

        # CDS occasionally wraps the payload in a ZIP even when netcdf is asked
        # for; handle both shapes.
        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as archive:
                members = [
                    m for m in archive.namelist()
                    if m.lower().endswith((".nc", ".nc4", ".netcdf"))
                ]
                if len(members) != 1:
                    raise RuntimeError(
                        f"Expected exactly one NetCDF inside the CDS archive, "
                        f"found {len(members)}"
                    )
                with archive.open(members[0]) as src, partial.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            download.replace(partial)

        size = partial.stat().st_size
        if size < MIN_EXPECTED_BYTES:
            raise RuntimeError(
                f"{out.name} is only {size / 1024**2:.0f} MB, which is far below "
                f"the expected size for a month of hourly global t2m. "
                "Treating this as a truncated download."
            )
        partial.replace(out)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        download.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download hourly ERA5 t2m and reduce it to FUA city means."
    )
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHPFILE)
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep the raw monthly NetCDF instead of deleting it after processing."
    )
    parser.add_argument(
        "--no-write-hourly", dest="write_hourly", action="store_false", default=True,
        help="Write only the daily parquet. Not recommended: the hourly file is "
             "the entire reason for this script."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the months that would be downloaded, then exit."
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Retry attempts per month on CDS failure (default 3)."
    )
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
    print("ERA5 hourly 2 m dry-bulb temperature -> FUA city means")
    print("=" * 74)
    print(f"  months requested : {len(args.years) * len(months)}")
    print(f"  months remaining : {len(todo)}")
    print(f"  output root      : {ERA5_DIR}")
    print(f"  raw NetCDFs      : {'kept' if args.keep_raw else 'deleted after each month'}")
    print(f"  dataset          : {CDS_DATASET} (0.1 deg, matches the CHI grid)")
    print(f"  est. transfer    : ~{len(todo) * 7:.0f} GB total, ~5-9 GB per month")

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
                    except Exception as exc:                   # noqa: BLE001
                        if attempt == args.retries:
                            raise
                        wait = 60 * attempt
                        print(f"  CDS request failed: {exc}\n  retrying in {wait}s ...")
                        time.sleep(wait)
                print(f"  downloaded {raw.stat().st_size / 1024**3:.2f} GB")

            print("  extracting FUA polygon means ...")
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
            elapsed = time.time() - t_start
            remaining = (elapsed / done) * (len(todo) - done)
            print(f"  progress {done}/{len(todo)} | est. remaining {remaining / 3600:.1f} h")

        except KeyboardInterrupt:
            print("\nInterrupted. Rerun the same command to resume.")
            return
        except Exception as exc:                                # noqa: BLE001
            print(f"  FAILED {tag}: {exc}")
            failed.append(tag)

    print("\n" + "=" * 74)
    print(f"Completed {done}/{len(todo)} months in {(time.time() - t_start) / 3600:.1f} h")
    if failed:
        print(f"Failed months ({len(failed)}): {', '.join(failed)}")
        print("Rerun the same command to retry only the failures.")
    else:
        print("Next step:  python 08_build_moist_air.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
