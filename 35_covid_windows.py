#!/usr/bin/env python
"""
35_covid_windows.py  --  the COVID sub-window diagnostic, for all five sectors
==============================================================================

Re-estimates the R5 asymmetry on three windows -- the full panel, the panel with
2020 removed, and 2020 alone -- for every sector and every unit, and writes

    R5/tables/T8_covid_window_tier<TIER>[_<unit>].csv

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
The diagnostic is 5 sectors x 3 windows x 3 units = 45 fits of the same
distributed lead-lag model that 22_r5_asymmetry.py fits once. Folding that into
22 made a script that used to finish in a couple of minutes take most of an
hour, and a single interruption threw away every completed cell.

So the work is cached per (tier, unit, sector, window) cell in a parquet
sidecar. Rerunning skips whatever is already there, which means the run can be
stopped and resumed, and a single sector can be refitted without touching the
rest. `--budget` stops cleanly after a time limit rather than being killed
mid-fit, which is what makes it usable from a caller with a short timeout.

WHAT THE DIAGNOSTIC IS FOR
--------------------------
2019-2022 contains a demand collapse and a recovery that are not weather. If
the thermal slopes were picking that up rather than the weather, dropping 2020
would move them. Running all five sectors is what makes the answer legible: a
shift confined to Power and Industry is a different fact from a shift shared by
the two sectors with no cooling channel.

USAGE
-----
  python 35_covid_windows.py --tier AB --unit perSD
  python 35_covid_windows.py --tier AB --unit perSD --sectors Power Industry
  python 35_covid_windows.py --tier AB --unit all --budget 150
  python 35_covid_windows.py --tier AB --unit all --status
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", str(BASE / "Results")))
CACHE = BASE / "outputs" / "response" / "r5_covid_windows_cache.parquet"

UNITS = ["per100", "perSD", "perP95"]
CACHE_KEYS = ["tier", "unit", "sector", "window"]


def load_r5():
    """Import 22_r5_asymmetry by path; its name is not a valid identifier."""
    spec = importlib.util.spec_from_file_location(
        "r5_asymmetry", BASE / "22_r5_asymmetry.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r5_asymmetry"] = mod
    spec.loader.exec_module(mod)
    return mod


def set_unit(r5, ex: pd.DataFrame, unit: str) -> tuple[float, float]:
    """Put r5's module-level unit globals on the requested scale.

    Mirrors what 22's main() does. The identification screen inside estimate()
    reads the raw degree-hour columns, so the country set does not move.
    """
    r5.UNIT_NAME = unit
    if unit == "per100":
        r5.UNIT_C = r5.UNIT_H = r5.SCALE
    else:
        def _u(x: np.ndarray) -> float:
            if unit == "perSD":
                return float(np.nanstd(x))
            pos = x[np.isfinite(x) & (x > 0)]
            return float(np.quantile(pos, 0.95)) if pos.size >= 100 else np.nan
        r5.UNIT_C = _u(ex["w_SCDH24"].to_numpy(float))
        r5.UNIT_H = _u(ex["w_HDH15"].to_numpy(float))
        if not (np.isfinite(r5.UNIT_C) and np.isfinite(r5.UNIT_H)):
            raise SystemExit(f"cannot form the {unit} unit on this tier")
    return r5.UNIT_C, r5.UNIT_H


def read_cache() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    return pd.DataFrame(columns=CACHE_KEYS)


def write_cache(df: pd.DataFrame) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)


def publish(cache: pd.DataFrame, tier: str, unit: str) -> Path | None:
    """Write the results table for one (tier, unit) if it is complete."""
    d = cache[(cache.tier == tier) & (cache.unit == unit)].copy()
    if d.empty:
        return None
    order = {w: i for i, w in enumerate(d.window_order.unique())} if \
        "window_order" in d.columns else {}
    d = d.sort_values(["sector_order", "window_order"]) if \
        "sector_order" in d.columns else d
    keep = ["sector", "window", "heat", "heat_se", "heat_z",
            "cold", "cold_se", "cold_z", "diff", "diff_se", "diff_z"]
    keep = [c for c in keep if c in d.columns]
    out = RESULTS / "R5" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    suf = "" if unit == "per100" else f"_{unit}"
    path = out / f"T8_covid_window_tier{tier}{suf}.csv"
    d[keep].to_csv(path, index=False)
    del order
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C"])
    ap.add_argument("--unit", default="all",
                    choices=UNITS + ["all"])
    ap.add_argument("--sectors", nargs="+", default=None)
    ap.add_argument("--budget", type=float, default=0.0,
                    help="stop cleanly after this many seconds (0 = no limit)")
    ap.add_argument("--status", action="store_true",
                    help="report what is cached and exit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    r5 = load_r5()
    sectors = args.sectors or r5.ALL_SECTORS
    units = UNITS if args.unit == "all" else [args.unit]

    cache = read_cache()
    if args.force:
        m = ~(cache.tier.eq(args.tier) & cache.unit.isin(units)
              & cache.sector.isin(sectors)) if len(cache) else \
            pd.Series(dtype=bool)
        cache = cache[m] if len(cache) else cache

    ex = pd.read_parquet(r5.EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"]}[args.tier]
    ex = ex[ex.tier.isin(keep)][["country", "date", "w_SCDH24", "w_HDH15"]]
    y0, y1 = int(ex.date.dt.year.min()), int(ex.date.dt.year.max())
    windows = [(f"{y0}-{y1}", None),
               ("drop 2020", {y for y in range(y0, y1 + 1) if y != 2020}),
               ("2020 only", {2020})]

    todo = [(u, s, w, yrs) for u in units for s in sectors
            for w, yrs in windows]
    if len(cache):
        done = set(map(tuple, cache[CACHE_KEYS].to_numpy().tolist()))
        todo = [t for t in todo
                if (args.tier, t[0], t[1], t[2]) not in done]

    print("=" * 78)
    print(f"R5 COVID sub-windows -- Tier {args.tier}, {y0}-{y1}")
    print("=" * 78)
    print(f"    cells cached: {len(cache)}   still to fit: {len(todo)}")
    if args.status:
        if len(cache):
            print(cache.groupby(["unit", "sector"]).size()
                  .unstack(fill_value=0).to_string())
        return
    if not todo:
        print("    nothing to do")

    t0 = time.time()
    rows = []
    cur_unit = None
    for unit, sec, wlab, yrs in todo:
        if args.budget and time.time() - t0 > args.budget:
            print(f"    budget of {args.budget:.0f}s reached; "
                  f"{len(todo) - len(rows)} cells left -- rerun to continue")
            break
        if unit != cur_unit:
            uc, uh = set_unit(r5, ex, unit)
            cur_unit = unit
            print(f"\n  [{unit}]  cooling unit {uc:.2f} K.h, "
                  f"heating unit {uh:.2f} K.h")
        t1 = time.time()
        _, web, _ = r5.estimate(sec, ex, naive=False, verbose=False, years=yrs)
        row = {"tier": args.tier, "unit": unit, "sector": sec, "window": wlab,
               "sector_order": r5.ALL_SECTORS.index(sec),
               "window_order": [w for w, _ in windows].index(wlab)}
        cells = []
        for tag in ("heat", "cold", "diff"):
            if tag in web:
                row[tag] = 100 * web[tag]["mu"]
                row[tag + "_se"] = 100 * web[tag]["mu_se"]
                row[tag + "_z"] = web[tag]["mu"] / web[tag]["mu_se"]
                cells.append(f"{row[tag]:+7.2f}({row[tag + '_z']:+.1f})")
            else:
                cells.append("      --")
        rows.append(row)
        print(f"    {sec:18s}{wlab:12s}{cells[0]:>13}{cells[1]:>13}"
              f"{cells[2]:>13}   {time.time() - t1:5.1f}s")
        # Written after every cell, so an interrupted run keeps its work.
        write_cache(pd.concat([cache, pd.DataFrame(rows)], ignore_index=True))

    cache = read_cache()
    print()
    for unit in units:
        d = cache[(cache.tier == args.tier) & (cache.unit == unit)]
        n_need = len(sectors) * len(windows)
        if len(d) >= n_need:
            p = publish(cache, args.tier, unit)
            print(f"    {p.relative_to(RESULTS)}  ({len(d)} cells)")
        else:
            print(f"    {unit}: {len(d)}/{n_need} cells -- table not written yet")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
