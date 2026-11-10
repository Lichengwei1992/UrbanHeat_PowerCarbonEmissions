#!/usr/bin/env python
"""
27_run_2019_2022.py  --  driver for the extended 2019-2022 result set
======================================================================

Runs the whole chain for the four-year panel and writes every figure and table
into a SEPARATE results root, so the published 2019-2021 set under `Results/`
is never touched.

WHY A DRIVER
------------
The chain is eight steps with several year-stamped file names threaded between
them. Running them by hand invites a mismatch -- for instance pointing
10_define_events.py at the four-year climate file but the three-year UTCI file,
which would silently drop UTCI for 2022 rather than fail. Every path is derived
here from one place.

WHAT IS PRESERVED
-----------------
  Results/                          untouched, remains the 2019-2021 set
  outputs/04_events_2019_2021/      archived copy of the 3-year event panel
  outputs/exposure_2019_2021/       archived copy of the 3-year exposure panel
  outputs/city_climate_2019_2021.parquet   untouched

WHAT IS WRITTEN
---------------
  outputs/city_climate_2019_2022.parquet
  outputs/utci/city_utci_daily_2019_2022.parquet
  outputs/moist_air/city_day/moist_2022-*.parquet      (added alongside 2019-21)
  outputs/04_events/*                                   REGENERATED for 4 years
  outputs/exposure/*                                    REGENERATED for 4 years
  Results_2019_2022/                                    the new figures/tables

A UNIT AND METHOD TRAP THAT THIS SCRIPT AVOIDS
----------------------------------------------
`05g_download_era5_hourly_utci.py` writes polygon-mean UTCI to
ERA5/processed_daily_polygon/utci/ **in kelvin**, while the city UTCI series
used by the 2019-2021 analysis is a centroid extraction **in Celsius**. Splicing
the two would introduce both a 273.15 offset and a spatial-method change at the
2022 boundary. This driver instead re-extracts UTCI for all four years through
05d, which now also emits utci_min, so one method and one unit span the panel.

USAGE
-----
  python 27_run_2019_2022.py              # full chain
  python 27_run_2019_2022.py --from 4     # resume at step 4
  python 27_run_2019_2022.py --dry-run    # print the commands only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS_NEW = BASE / "Results_2019_2022"

CLIMATE = BASE / "outputs" / "city_climate_2019_2022.parquet"
UTCI_OUT = BASE / "outputs" / "utci" / "city_utci_daily_2019_2022.parquet"
START, END = "2019-01-01", "2022-12-31"

PY = sys.executable


def step_cmds() -> list[tuple[str, list[str]]]:
    return [
        ("1  city climate + UTCI, 4 years (slow: reads 4 annual NetCDFs)",
         [PY, "05d_extract_era5_cities.py",
          "--years", "2019", "2020", "2021", "2022",
          "--out", str(CLIMATE)]),

        ("2  build the 4-year city UTCI parquet from the climate file",
         [PY, "-c", _UTCI_SNIPPET]),

        ("3  moist air for 2022 (needs CHI hourly Twb + ERA5 hourly t2m)",
         [PY, "08_build_moist_air.py", "--years", "2022"]),

        ("4  event taxonomy, 2019-2022",
         [PY, "10_define_events.py", "--start", START, "--end", END,
          "--climate", str(CLIMATE), "--utci", str(UTCI_OUT)]),

        ("5  event statistics", [PY, "11_event_statistics.py"]),
        ("6  event windows", [PY, "12_event_windows.py"]),
        ("7  national city-weighted exposure", [PY, "14_national_exposure.py"]),

        # The response analyses read emissions through HC_CARBON_DIR. Without
        # this merge they would silently regress a 4-year exposure panel on a
        # 3-year emissions panel: same n, same coefficients, no error raised.
        ("7b merge the 2022 emissions release", [PY, "28_merge_emissions.py"]),

        # Estimation must run before 19_results_figures.py: that script only
        # PLOTS, reading r3_lag_profiles_tier*.csv (from 16) and
        # r2_benchmarks_tier*.csv (from 18). Omitting these would silently
        # produce a 2019-2022 figure set drawn from 2019-2021 coefficients.
        ("8  R3 dynamic response, Tier A+B",
         [PY, "16_r3_dynamic_response.py", "--tier", "AB"]),
        ("9  R3 dynamic response, Tier C",
         [PY, "16_r3_dynamic_response.py", "--tier", "C"]),
        ("10 national benchmark exposure (GADM raster, slow)",
         [PY, "17_national_benchmarks.py",
          "--years", "2019", "2020", "2021", "2022"]),
        ("11 R2 benchmark test, Tier A+B",
         [PY, "18_r2_benchmark_test.py", "--tier", "AB"]),
        ("12 R2 benchmark test, Tier C",
         [PY, "18_r2_benchmark_test.py", "--tier", "C"]),

        ("13 R1-R3 figures and tables", [PY, "19_results_figures.py"]),
        ("14 R4 humidity, Tier A+B", [PY, "20_r4_humidity.py", "--tier", "AB"]),
        ("15 R4 humidity, Tier C", [PY, "20_r4_humidity.py", "--tier", "C"]),
        ("16 SI threshold scan, Tier A+B",
         [PY, "21_chhe_threshold_scan.py", "--tier", "AB"]),
        ("17 SI threshold scan, Tier C",
         [PY, "21_chhe_threshold_scan.py", "--tier", "C", "--no-figure"]),
        ("18 R5 thermal asymmetry", [PY, "22_r5_asymmetry.py", "--tier", "AB"]),
        ("19 R6 built-environment weights", [PY, "23_built_environment_weights.py"]),
        ("20 R6 weighting competition", [PY, "24_r6_weight_competition.py", "--tier", "AB"]),
        # Produces FigS3 / TS4 / TS5. Without it the SI section of the new tree
        # would be missing three items the published Results/SI contains.
        ("21 SI OxCGRT policy robustness",
         [PY, "25_oxcgrt_robustness.py", "--tier", "AB"]),
        ("22 construct-mismatch diagnostic", [PY, "26_construct_mismatch.py", "--tier", "AB"]),

        # Scale-free units for R5. The country screen still uses raw
        # degree-hours, so these runs differ from the one above only in the
        # unit the two sides are expressed in.
        ("23 R5 asymmetry, per-SD units",
         [PY, "22_r5_asymmetry.py", "--tier", "AB", "--unit", "perSD",
          "--no-figure"]),
        ("24 R5 asymmetry, per-p95 units",
         [PY, "22_r5_asymmetry.py", "--tier", "AB", "--unit", "perP95",
          "--no-figure"]),
        # Base-temperature scan: the result that the sign of the asymmetry is
        # set by a pair of conventions. Slow -- 270 fits.
        ("25 base-temperature scan", [PY, "32_base_temperature_scan.py"]),
        ("26 resolution and humidity encompassing tests",
         [PY, "31_r3r5_diagnostics.py"]),
        ("27 why the response is estimated at country scale",
         [PY, "33_city_scale_feasibility.py"]),

        # Verification. These are cheap and they close the two failure modes
        # that actually occurred in this project: a tree that looks complete
        # but is missing an item or a format (29), and prose whose numbers no
        # longer match the tables they were copied from (30). Both exit
        # non-zero on failure, so a bad run cannot end with "Done" quietly.
        ("28 results-tree parity against the published layout",
         [PY, "29_check_results_parity.py"]),
        ("29 narrative numbers against the source tables",
         [PY, "30_verify_narrative.py"]),
    ]


# Built as a snippet rather than a separate file: it is four lines of glue whose
# only job is to project the climate table onto the columns 10_define_events.py
# expects from a UTCI file.
_UTCI_SNIPPET = (
    "import pandas as pd, pathlib; "
    "c = pd.read_parquet(r'{clim}'); "
    "cols = [x for x in ['city_id','date','utci_mean','utci_max','utci_min'] "
    "if x in c.columns]; "
    "assert 'utci_min' in cols, 'utci_min missing - rerun 05d after the patch'; "
    "out = pathlib.Path(r'{utci}'); out.parent.mkdir(parents=True, exist_ok=True); "
    "c[cols].to_parquet(out, index=False); "
    "print(f'  wrote {{out.name}}: {{len(c):,}} rows, cols={{cols}}')"
).format(clim=CLIMATE, utci=UTCI_OUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start_at", type=int, default=1,
                    help="Resume at this step number (1-based).")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="Run only these step numbers.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ)
    env["HC_RESULTS_ROOT"] = str(RESULTS_NEW)
    env["HC_CARBON_DIR"] = str(BASE / "CarbonData" / "merged_2019_2022")
    RESULTS_NEW.mkdir(parents=True, exist_ok=True)

    steps = step_cmds()
    print("=" * 78)
    print("Extended run: 2019-2022")
    print(f"  results root : {RESULTS_NEW}")
    print(f"  climate file : {CLIMATE.name}")
    print(f"  utci file    : {UTCI_OUT.name}")
    print(f"  window       : {START} .. {END}")
    print(f"  emissions    : {env['HC_CARBON_DIR']}")
    print("=" * 78)

    t0 = time.time()
    for i, (label, cmd) in enumerate(steps, start=1):
        if args.only and i not in args.only:
            continue
        if not args.only and i < args.start_at:
            print(f"[{i:2d}] SKIP  {label}")
            continue
        shown = " ".join(c if len(c) < 80 else c[:60] + "..." for c in cmd)
        print(f"\n[{i:2d}] {label}\n     $ {shown}")
        if args.dry_run:
            continue
        t1 = time.time()
        r = subprocess.run(cmd, cwd=BASE, env=env)
        if r.returncode != 0:
            print(f"\n  STEP {i} FAILED (exit {r.returncode}). Fix, then resume with:")
            print(f"    python 27_run_2019_2022.py --from {i}")
            sys.exit(r.returncode)
        print(f"     done in {time.time() - t1:.0f}s")

    print(f"\nAll requested steps finished in {(time.time() - t0) / 60:.1f} min")
    print(f"New result set: {RESULTS_NEW}")
    print(f"Original 2019-2021 set untouched at: {BASE / 'Results'}")


if __name__ == "__main__":
    main()
