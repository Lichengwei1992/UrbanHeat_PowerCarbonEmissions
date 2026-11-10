#!/usr/bin/env python
"""
09_extend_thresholds.py
=======================

Build the full DOY-windowed percentile table needed by the v2 Part III event
taxonomy, from the 1991-2020 ERA5 city baseline.

WHY
---
`ERA5/baseline/threshold_table_doy.parquet` (from 05c) carries only
`tmax_p90` and `tmin_p10`. The Part III taxonomy additionally needs:

  * `tmin_p90`  -> hot nights (HN) and compound day-night heat (CDNH)
  * `tmax_p10`  -> cold days for the cold-side taxonomy
  * p95 / p99 / p05 / p01  -> sensitivity analysis on the extremeness cut
  * climatological means and SDs -> anomalies and standardised indices
  * `tmax_p25` / `tmax_p75` -> the HWMId denominator (Russo et al. 2015)

METHOD (matches 05c so the two tables are interchangeable)
----------------------------------------------------------
For each city and each day-of-year d, pool all daily values falling in a
+/-15-day window around d across all 30 baseline years:

    31 days x 30 years = 930 samples per city-DOY

which reproduces the `n_samples = 930` in the existing table. Percentiles are
computed on that pooled sample. DOY 366 reuses the DOY 365 window.

INPUT
-----
  ERA5/baseline/city_baseline_all_years.parquet   (1991-2020 daily tmax_c/tmin_c)

OUTPUT
------
  ERA5/baseline/threshold_table_doy_full.parquet
    city_id | doy | n_samples
    tmax_p01 tmax_p05 tmax_p10 tmax_p25 tmax_p75 tmax_p90 tmax_p95 tmax_p99
    tmin_p01 tmin_p05 tmin_p10 tmin_p90 tmin_p95 tmin_p99
    tmax_mean tmax_sd tmin_mean tmin_sd

USAGE
-----
  python 09_extend_thresholds.py
  python 09_extend_thresholds.py --window 15 --validate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
BASELINE = BASE / "ERA5" / "baseline" / "city_baseline_all_years.parquet"
EXISTING = BASE / "ERA5" / "baseline" / "threshold_table_doy.parquet"
OUT = BASE / "ERA5" / "baseline" / "threshold_table_doy_full.parquet"

TMAX_PCTS = [1, 5, 10, 25, 75, 90, 95, 99]
TMIN_PCTS = [1, 5, 10, 90, 95, 99]
N_DOY = 366

# Month starts in a leap-year template. Mapping every observation by month/day
# avoids the one-day post-February shift produced by pandas dayofyear when leap
# and non-leap years are pooled. Non-leap years simply have no observation at
# template DOY 60 (29 February).
LEAP_MONTH_START = np.array(
    [0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
    dtype=np.int16,
)


def pct_name(prefix: str, percentile: int) -> str:
    """Stable percentile column names (p01/p05, then p10/p90, etc.)."""
    return f"{prefix}_p{percentile:02d}"


def climatological_doy(dates: pd.Series) -> np.ndarray:
    """Map month/day to a fixed 366-day climatological calendar."""
    month = dates.dt.month.to_numpy(dtype=np.int16)
    day = dates.dt.day.to_numpy(dtype=np.int16)
    return LEAP_MONTH_START[month] + day


def build_cube(df: pd.DataFrame, cities: np.ndarray, years: np.ndarray):
    """Return arrays [city, year, doy] for tmax and tmin (NaN where absent)."""
    cidx = pd.Series(np.arange(len(cities)), index=cities)
    yidx = pd.Series(np.arange(len(years)), index=years)

    ci = df["city_id"].map(cidx).to_numpy()
    yi = df["date"].dt.year.map(yidx).to_numpy()
    di = climatological_doy(df["date"]) - 1

    shape = (len(cities), len(years), N_DOY)
    tmax = np.full(shape, np.nan, dtype="float32")
    tmin = np.full(shape, np.nan, dtype="float32")
    tmax[ci, yi, di] = df["tmax_c"].to_numpy(dtype="float32")
    tmin[ci, yi, di] = df["tmin_c"].to_numpy(dtype="float32")
    return tmax, tmin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=15,
                    help="Half-width of the DOY window in days (default 15 -> 31-day window).")
    ap.add_argument("--validate", action="store_true",
                    help="Compare the recomputed tmax_p90/tmin_p10 against the existing table.")
    args = ap.parse_args()

    if not BASELINE.exists():
        raise SystemExit(f"Baseline not found: {BASELINE}")

    t0 = time.time()
    print("=" * 70)
    print("Extending the DOY threshold table from the 1991-2020 baseline")
    print("=" * 70)

    df = pd.read_parquet(BASELINE)
    df["city_id"] = df["city_id"].astype(str).str.zfill(4)
    cities = np.sort(df["city_id"].unique())
    years = np.sort(df["date"].dt.year.unique())
    print(f"  {len(df):,} city-days | {len(cities):,} cities | "
          f"{years.min()}-{years.max()} ({len(years)} years)")

    tmax, tmin = build_cube(df, cities, years)
    del df
    print(f"  built [city, year, doy] cube {tmax.shape}  ({time.time() - t0:.0f}s)")

    w = args.window
    n_city = len(cities)
    cols = {pct_name("tmax", p): np.empty((n_city, N_DOY), "float32")
            for p in TMAX_PCTS}
    cols |= {pct_name("tmin", p): np.empty((n_city, N_DOY), "float32")
             for p in TMIN_PCTS}
    for name in ("tmax_mean", "tmax_sd", "tmin_mean", "tmin_sd"):
        cols[name] = np.empty((n_city, N_DOY), "float32")
    n_samples = np.empty((n_city, N_DOY), "int32")

    for d in range(N_DOY):
        # Circular +/- w window on a fixed leap-year calendar. February 29 is
        # sparse but receives a full surrounding window from all 30 years.
        idx = (np.arange(d - w, d + w + 1) % N_DOY)
        wx = tmax[:, :, idx].reshape(n_city, -1)
        wn = tmin[:, :, idx].reshape(n_city, -1)

        qx = np.nanpercentile(wx, TMAX_PCTS, axis=1)
        for k, p in enumerate(TMAX_PCTS):
            cols[pct_name("tmax", p)][:, d] = qx[k]
        qn = np.nanpercentile(wn, TMIN_PCTS, axis=1)
        for k, p in enumerate(TMIN_PCTS):
            cols[pct_name("tmin", p)][:, d] = qn[k]

        cols["tmax_mean"][:, d] = np.nanmean(wx, axis=1)
        cols["tmax_sd"][:, d] = np.nanstd(wx, axis=1)
        cols["tmin_mean"][:, d] = np.nanmean(wn, axis=1)
        cols["tmin_sd"][:, d] = np.nanstd(wn, axis=1)
        n_samples[:, d] = np.isfinite(wx).sum(axis=1)

        if (d + 1) % 60 == 0 or d == N_DOY - 1:
            print(f"    DOY {d + 1}/{N_DOY}  ({time.time() - t0:.0f}s)")

    out = pd.DataFrame({
        "city_id": np.repeat(cities, N_DOY),
        "doy": np.tile(np.arange(1, N_DOY + 1), n_city),
        "n_samples": n_samples.ravel(),
    })
    for name, arr in cols.items():
        out[name] = arr.ravel()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n  saved {OUT}  ({len(out):,} rows, {OUT.stat().st_size / 1024**2:.1f} MB)")
    print(f"  median samples per city-DOY: {np.median(n_samples):.0f}")

    print("\n  threshold ranges across cities (DOY-pooled):")
    for c in ["tmax_p90", "tmax_p95", "tmax_p99", "tmin_p10", "tmin_p90", "tmax_p10"]:
        print(f"    {c:10s} {out[c].min():7.1f} .. {out[c].max():6.1f} C   "
              f"median {out[c].median():6.1f}")

    # Human-readable audit tables. The parquet remains the machine input;
    # these CSVs make coverage and threshold ranges directly inspectable.
    value_cols = [c for c in out.columns if c not in ("city_id", "doy")]
    summary = []
    for c in value_cols:
        s = out[c]
        summary.append({
            "variable": c,
            "n_rows": len(s),
            "n_missing": int(s.isna().sum()),
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p50": float(s.median()),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        })
    summary_path = OUT.with_name("threshold_table_doy_full_summary.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    method_path = OUT.with_name("threshold_table_doy_full_method.csv")
    pd.DataFrame([{
        "baseline_start": int(years.min()),
        "baseline_end": int(years.max()),
        "n_years": len(years),
        "n_cities": len(cities),
        "calendar": "fixed 366-day month-day template",
        "window_half_width_days": w,
        "window_total_days": 2 * w + 1,
        "feb29_handling": "explicit template DOY 60; absent in non-leap years",
    }]).to_csv(method_path, index=False)
    print(f"  wrote audit tables: {summary_path.name}, {method_path.name}")

    if args.validate and EXISTING.exists():
        old = pd.read_parquet(EXISTING)
        old["city_id"] = old["city_id"].astype(str).str.zfill(4)
        chk = out.merge(old, on=["city_id", "doy"], suffixes=("_new", "_old"))
        for c in ("tmax_p90", "tmin_p10"):
            d = (chk[f"{c}_new"] - chk[f"{c}_old"]).abs()
            print(f"\n  validation vs 05c table | {c}: "
                  f"median |diff| = {d.median():.4f} K, p99 = {d.quantile(.99):.4f} K, "
                  f"max = {d.max():.4f} K")
        print("  (small non-zero differences are expected only if 05c used a "
              "different window or leap-day rule)")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
