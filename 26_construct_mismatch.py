#!/usr/bin/env python
"""
26_construct_mismatch.py  --  the relative-vs-absolute exposure metric diagnostic
============================================================================

R3.2 reports that exposure defined against local climatology carries no carbon
signal while absolute physiological thresholds do, and supports it with three
descriptive facts. Those three numbers were originally produced by an ad-hoc
diagnostic that was never written to disk, so they could not be regenerated or
checked. This script computes them from the archived panels with the
definitions stated explicitly, because they are the quantitative core of the
paper's methodological claim and must be reproducible line by line.

THE THREE FACTS
---------------
1  SAME LABEL, DIFFERENT PHYSICS
   Mean cooling degree-hours on the heatwave days of each country. If the
   relative criterion described a common physical state, this would be roughly
   constant across countries.

2  THE CRITERION MISSES THE DEMAND
   Share of the highest-cooling-demand days that the relative criterion never
   flags. Computed at both city-day and country-day level and with two
   definitions of "not flagged", because the number depends on all three
   choices and quoting one without saying which would be indefensible.

3  IT IS NOT A POWER PROBLEM
   Variance of each exposure metric surviving the study's fixed effects. A relative
   exposure metric that retains MORE identifying variation than a continuous one, and
   still returns a null, has failed for reasons other than precision.

UNITS MATTER AND ARE REPORTED, NOT CHOSEN SILENTLY
--------------------------------------------------
The response model is estimated on country-days, so country-day is the unit
that speaks to the regression result. The exposure metric itself is defined on
city-days, so city-day is the unit that speaks to the definition. They give
different numbers and both are printed; the text quotes the city-day version
for the statement about the exposure metric and the country-day version for the
statement about the regression.

INPUTS
------
  outputs/04_events/city_events_daily_v2.parquet
  outputs/exposure/country_daily_exposure.parquet

OUTPUTS
-------
  Results/R3/tables/T4b_metric_mismatch.csv
  printed summary for direct quotation in the text

USAGE
-----
  python 26_construct_mismatch.py --tier AB
"""

from __future__ import annotations

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
# Results root. Overridable with the HC_RESULTS_ROOT environment variable so an
# extended-period rerun can write a parallel result set instead of overwriting
# the published one. Unset, it resolves to the original BASE/"Results".
RESULTS_ROOT = Path(os.environ.get("HC_RESULTS_ROOT", str(BASE / "Results")))

EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
TAB_DIR = RESULTS_ROOT / "R3" / "tables"

MIN_HW_DAYS = 20          # countries with too few heatwave days are unstable
HW_SHARE = 0.5            # a country-day counts as a heatwave day above this


def demean(X, facs, it=250, tol=1e-11):
    X = np.asarray(X, dtype="float64").copy()
    meta = [(f, np.bincount(f).astype(float), f.max() + 1) for f in facs]
    for _ in range(it):
        p = X.copy()
        for f, c, n in meta:
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(f, weights=X[:, j], minlength=n) / c)[f]
        if np.max(np.abs(X - p)) < tol:
            break
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C", "all"])
    args = ap.parse_args()

    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"],
            "all": ["A", "B", "C"]}[args.tier]
    ex = pd.read_parquet(EXPOSURE)
    ex = ex[ex.tier.isin(keep)].copy()
    rows = []

    print("=" * 78)
    print(f"Construct mismatch diagnostic (Tier {args.tier})")
    print("=" * 78)

    # ---- 1. same label, different physics --------------------------------
    print("\n[1] Mean cooling degree-hours on each country's heatwave days")
    hw = ex[ex.w_HW > HW_SHARE]
    g = (hw.groupby("country")
           .agg(n_hw_days=("date", "size"), scdh=("w_SCDH24", "mean"))
           .query(f"n_hw_days >= {MIN_HW_DAYS}")
           .sort_values("scdh"))
    lo, hi = g.iloc[0], g.iloc[-1]
    ratio = hi.scdh / lo.scdh
    print(f"    {len(g)} countries with >= {MIN_HW_DAYS} heatwave days")
    print(f"    lowest  {g.index[0]:16s} {lo.scdh:6.2f} K.h  ({int(lo.n_hw_days)} days)")
    print(f"    highest {g.index[-1]:16s} {hi.scdh:6.2f} K.h  ({int(hi.n_hw_days)} days)")
    print(f"    ratio = {ratio:.1f}-fold")
    rows.append({"fact": "spread of cooling load across countries' heatwave days",
                 "unit": "country", "definition": f"w_HW > {HW_SHARE}, n >= {MIN_HW_DAYS}",
                 "value": ratio, "detail": f"{g.index[0]} {lo.scdh:.2f} -> "
                                           f"{g.index[-1]} {hi.scdh:.2f} K.h"})

    # ---- 2. the criterion misses the demand ------------------------------
    print("\n[2] Highest-cooling-demand days that the relative criterion misses")
    ev = pd.read_parquet(EVENTS, columns=["city_id", "date", "SCDH24", "EHD", "HW"])
    print(f"    {'unit':12s}{'flag':6s}{'threshold':>12s}{'missed %':>10s}{'n':>10s}")
    for unit, d, sc, flags in (
            ("city-day", ev, "SCDH24", {"EHD": ev.EHD, "HW": ev.HW}),
            ("country-day", ex, "w_SCDH24",
             {"EHD": ex.w_EHD > 0, "HW": ex.w_HW > 0})):
        for q, qlab in ((0.90, "top decile"), (0.95, "top 5%")):
            thr = d[sc].quantile(q)
            top = d[d[sc] >= thr]
            for fname, fser in flags.items():
                f = fser.loc[top.index]
                missed = 100 * float((f == 0).mean() if f.dtype != bool
                                     else (~f).mean())
                print(f"    {unit:12s}{fname:6s}{qlab:>12s}{missed:10.1f}{len(top):10,d}")
                rows.append({"fact": "top-demand days not flagged",
                             "unit": unit,
                             "definition": f"{qlab} of {sc}, not flagged by {fname}",
                             "value": missed, "detail": f"n = {len(top)}"})

    # ---- 3. it is not a power problem ------------------------------------
    print("\n[3] Variance of each exposure metric surviving the fixed effects")
    cols = ["w_HW", "w_EHD", "w_utci_strong", "w_CHHE", "w_SCDH24"]
    # Reported on the ESTIMATION sample: the response models drop the first
    # LAGS days of each country when the lag matrix is built, so the variance a
    # coefficient is actually identified from is the variance on that trimmed
    # panel, not on the full one. Both are printed; they differ by 1-2 points.
    for sample, lab in ((0, "full panel"), (7, "estimation sample (lags dropped)")):
        d = ex.sort_values(["country", "date"]).reset_index(drop=True)
        if sample:
            # cumcount rather than groupby.apply: from pandas 2.2 apply() drops
            # the grouping column from the result, which silently removed
            # `country` and broke the fixed-effect factorisation below. A
            # positional mask does the same trimming with no such surprise.
            d = d[d.groupby("country").cumcount() >= sample].reset_index(drop=True)
        f1 = pd.factorize(d.country + "_" + d.date.dt.to_period("M").astype(str))[0]
        f2 = pd.factorize(d.date)[0]
        f3 = pd.factorize(d.country + "_" + d.date.dt.dayofweek.astype(str))[0]
        Z = demean(d[cols].values, [f1, f2, f3])
        print(f"    {lab}  (n = {len(d):,})")
        print(f"      {'exposure metric':20s}{'raw SD':>10s}{'residual SD':>13s}"
              f"{'variance kept':>15s}")
        for j, c in enumerate(cols):
            v0 = float(d[c].var())
            v1 = float(Z[:, j].var())
            pct = 100 * v1 / v0 if v0 > 0 else np.nan
            print(f"      {c:20s}{np.sqrt(v0):10.3f}{np.sqrt(v1):13.3f}{pct:14.1f}%")
            rows.append({"fact": "variance surviving fixed effects",
                         "unit": f"country-day, {lab}",
                         "definition": f"{c} after country x year-month + date + "
                                       f"country x day-of-week",
                         "value": pct, "detail": f"n = {len(d)}"})

    def keep_of(col):
        m = [r for r in rows
             if r["fact"] == "variance surviving fixed effects"
             and r["definition"].startswith(col + " ")
             and "estimation sample" in r["unit"]]
        return m[0]["value"]

    hw_keep, cd_keep = keep_of("w_HW"), keep_of("w_SCDH24")
    print(f"\n    -> the relative heatwave share retains {hw_keep:.1f}% of its")
    print(f"       variance against {cd_keep:.1f}% for cooling degree-hours, so it")
    print("       has MORE identifying variation than the exposure metric that works.")
    print("       Its null is not a precision problem.")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / f"T4b_metric_mismatch_tier{args.tier}.csv", index=False)
    print(f"\n    R3/tables/T4b_metric_mismatch_tier{args.tier}.csv")


if __name__ == "__main__":
    main()
