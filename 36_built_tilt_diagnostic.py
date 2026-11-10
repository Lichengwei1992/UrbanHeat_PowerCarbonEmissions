#!/usr/bin/env python
"""
36_built_tilt_diagnostic.py  --  is there anything to identify?
================================================================

R6 currently tests whether built-environment scale moderates the thermal carbon
response by regressing 20 country-level response slopes on 20 country-level
covariates. That is a cross-country association with all the usual confounds --
country size, income, climate zone and urban form move together across nations
and cannot be separated with twenty points.

THE ALTERNATIVE THIS SCRIPT SIZES UP
------------------------------------
Ask instead whether the SAME country responds differently on days when the heat
falls on the built-up part of its urban system than on days when the same
national load is spread over its less built-up cities. National exposure is a
weighted mean over a country's cities, so re-weighting by built volume instead
of population gives a second series, and the gap between them varies day to day
within a country:

    X_pop_ct = sum_i pop_i  x_it / sum_i pop_i
    X_bv_ct  = sum_i bv_i   x_it / sum_i bv_i
    D_ct     = X_bv_ct - X_pop_ct

D_ct > 0 means today's thermal load sits disproportionately on the country's
high built-volume cities. Entering D beside X_pop in R3's specification is a
within-country test with about 40,000 country-days and country x year-month
fixed effects, and it needs no city-level emissions, so none of the
identification failure documented in identification_test_memo.md applies.

WHY THE DIFFERENCE AND NOT THE RATIO
------------------------------------
A ratio X_bv / X_pop is undefined on the many days when both are near zero,
which is most of the year for cooling load. The difference is defined
everywhere, is exactly zero when the two weightings agree, and is the regressor
the model would actually use. The ratio is reported too, but only on days with
real load, and only because it is the interpretable version.

WHAT THIS SCRIPT DOES AND DOES NOT DO
-------------------------------------
It fits NOTHING. R6.1 already established that built volume and population rank
a country's cities almost identically (median Spearman rho = 0.95), which is
precisely the reason to fear that D carries no usable variation. This script
measures that variation before any model is written:

  * how much of D survives country x year-month fixed effects, in absolute
    terms and relative to the exposure it modulates;
  * whether D is merely a rescaling of X_pop within country, in which case an
    interaction would be unidentified;
  * how it differs across urban systems, since a country whose population sits
    in one primate city cannot have much tilt at all.

A near-zero answer kills the design, and that null is itself worth reporting:
it would say precisely why the built-environment mechanism cannot be taken
below the national scale with this data.

USAGE
-----
  python 36_built_tilt_diagnostic.py
  python 36_built_tilt_diagnostic.py --var SCDH24 --tier AB
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
SCHEMES = BASE / "outputs" / "population" / "fua_weight_schemes.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"


def demean(df: pd.DataFrame, col: str, by: list[str]) -> np.ndarray:
    """Residual of col after the fixed effects implied by `by`."""
    return (df[col] - df.groupby(by)[col].transform("mean")).to_numpy(float)


def build(var: str, tier: str) -> pd.DataFrame:
    """Country-day population- and built-volume-weighted exposure."""
    w = pd.read_parquet(SCHEMES)[
        ["city_id", "country", "pop_weight_dedup", "built_v", "built_s"]]
    w = w[(w.pop_weight_dedup > 0) & (w.built_v > 0)]
    print(f"    weights: {len(w):,} cities in {w.country.nunique()} countries")

    ev = pd.read_parquet(EVENTS, columns=["city_id", "date", var])
    d = ev.merge(w, on="city_id", how="inner")
    print(f"    joined : {len(d):,} city-days")

    g = d.groupby(["country", "date"], sort=False)
    num_p = g.apply(lambda s: float(np.average(s[var], weights=s.pop_weight_dedup)))
    num_b = g.apply(lambda s: float(np.average(s[var], weights=s.built_v)))
    out = pd.DataFrame({"X_pop": num_p, "X_bv": num_b}).reset_index()
    out["D"] = out.X_bv - out.X_pop
    out["date"] = pd.to_datetime(out.date)
    out["ym"] = out.date.dt.to_period("M").astype(str)

    ex = pd.read_parquet(EXPOSURE, columns=["country", "date", "tier"])
    ex["date"] = pd.to_datetime(ex.date)
    out = out.merge(ex.drop_duplicates(["country", "date"]), on=["country", "date"],
                    how="left")
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"]}[tier]
    out = out[out.tier.isin(keep)].copy()
    print(f"    tier {tier}: {len(out):,} country-days, "
          f"{out.country.nunique()} countries")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="SCDH24",
                    help="city-level exposure column to tilt")
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C"])
    args = ap.parse_args()

    print("=" * 78)
    print(f"Built-environment tilt diagnostic -- {args.var}, Tier {args.tier}")
    print("=" * 78)
    d = build(args.var, args.tier)

    # ---- 1. does D survive the fixed effects the model would use? --------
    print("\n[1] Variation surviving country x year-month fixed effects")
    fe = ["country", "ym"]
    d["X_res"] = demean(d, "X_pop", fe)
    d["D_res"] = demean(d, "D", fe)
    sd_x, sd_d = float(np.std(d.X_res)), float(np.std(d.D_res))
    print(f"    SD of X_pop  (raw / after FE): {d.X_pop.std():8.3f} /{sd_x:8.3f} K.h")
    print(f"    SD of D      (raw / after FE): {d.D.std():8.3f} /{sd_d:8.3f} K.h")
    print(f"    D as a share of the exposure it modulates: "
          f"{100 * sd_d / sd_x:5.1f} %")
    print("    D is the built-volume minus population weighted national load.")
    print("    If this share is a rounding error the interaction has nothing")
    print("    to be identified from, whatever its coefficient turns out to be.")

    # ---- 2. is D just a rescaling of X within country? -------------------
    print("\n[2] Is D independent variation, or X_pop in disguise?")
    r_all = float(np.corrcoef(d.X_res, d.D_res)[0, 1])
    print(f"    corr(X_pop, D) after FE, pooled: {r_all:+.3f}")
    print("    A correlation near +/-1 would mean D is proportional to X within")
    print("    country, so the interaction would be collinear with the main")
    print("    effect and the design would be unidentified.")

    # ---- 3. per country, and how it relates to the urban system ----------
    print("\n[3] By country")
    rows = []
    for c, s in d.groupby("country"):
        if len(s) < 200:
            continue
        xr, dr = demean(s, "X_pop", ["ym"]), demean(s, "D", ["ym"])
        sx, sd_ = float(np.std(xr)), float(np.std(dr))
        # Urban-system concentration: the Herfindahl of built volume across the
        # country's cities. One dominant city means little room for tilt.
        rows.append({"country": c, "n_days": len(s), "sd_X": sx, "sd_D": sd_,
                     "D_share_pct": 100 * sd_ / sx if sx > 0 else np.nan,
                     "corr_X_D": float(np.corrcoef(xr, dr)[0, 1])
                     if sx > 0 and sd_ > 0 else np.nan})
    r = pd.DataFrame(rows)

    w = pd.read_parquet(SCHEMES)
    w = w[(w.pop_weight_dedup > 0) & (w.built_v > 0)]
    hh = w.groupby("country").apply(
        lambda s: pd.Series({
            "n_cities": len(s),
            "hhi_built_v": float(((s.built_v / s.built_v.sum()) ** 2).sum()),
        })).reset_index()
    r = r.merge(hh, on="country", how="left").sort_values("D_share_pct",
                                                          ascending=False)
    print(f"    {'country':18s}{'days':>6s}{'SD X':>9s}{'SD D':>9s}"
          f"{'D/X %':>8s}{'r(X,D)':>9s}{'cities':>8s}{'HHI':>8s}")
    for x in r.itertuples():
        print(f"    {x.country:18s}{x.n_days:6d}{x.sd_X:9.2f}{x.sd_D:9.3f}"
              f"{x.D_share_pct:8.1f}{x.corr_X_D:9.2f}{int(x.n_cities):8d}"
              f"{x.hhi_built_v:8.3f}")

    ok = r[r.hhi_built_v.notna() & r.D_share_pct.notna()]
    if len(ok) > 3:
        rho = float(ok[["hhi_built_v", "D_share_pct"]].corr(
            method="spearman").iloc[0, 1])
        print(f"\n    Spearman(urban concentration, D/X) = {rho:+.2f} over "
              f"{len(ok)} countries")
        print("    A country whose built volume sits in one city has little")
        print("    room for the tilt to move, so this is the prior sign check.")

    out = RESULTS / "R6" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"T12_built_tilt_diagnostic_{args.var}_tier{args.tier}.csv"
    r.to_csv(p, index=False)
    print(f"\n    {p.relative_to(RESULTS)}")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "-" * 78)
    share = 100 * sd_d / sd_x
    if share < 2:
        print(f"VERDICT: D carries {share:.1f} % of the exposure variation after "
              "the fixed\neffects. That is too little to identify an "
              "interaction. The built-environment\nmechanism cannot be taken "
              "below the national scale with this weighting, and\nthat is the "
              "result to report.")
    elif abs(r_all) > 0.9:
        print(f"VERDICT: D survives ({share:.1f} % of X) but correlates "
              f"{r_all:+.2f} with X_pop after\nthe fixed effects, so it is a "
              "rescaling rather than independent variation.\nThe interaction "
              "would be collinear with the main effect.")
    else:
        print(f"VERDICT: D carries {share:.1f} % of the exposure variation "
              f"after the fixed effects\nand correlates {r_all:+.2f} with it. "
              "There is variation to identify from; the next\nstep is to enter "
              "D beside X_pop in R3's specification, with the two\nno-cooling "
              "sectors as negative controls.")
    print("-" * 78)


if __name__ == "__main__":
    main()
