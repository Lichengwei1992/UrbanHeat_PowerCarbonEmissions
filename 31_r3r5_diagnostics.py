#!/usr/bin/env python
"""
31_r3r5_diagnostics.py  --  three checks R3-R5 currently assert without testing
==============================================================================

Diagnostic only. Nothing here writes into the R3-R6 figures or narrative; the
point is to find out whether three claims survive before deciding how to frame
the response half of the paper.

D1  DOES THE STUDY NEED CITY-SCALE DATA FOR R3-R6?
    SI.1 compares FUA against national resolution for daily-Tmax cooling
    degree-hours only, and finds national population weighting is as good or
    better. It never asks the same question of the constructs that actually
    carry the R3.2 result. A national WBGT series cannot be built here -- the
    gridded ERA5 cache holds temperature only, and WBGT comes from CHI at city
    scale -- so the question is instead decomposed into its two parts:

        does city RESOLUTION add anything over national temperature?
        does the HUMIDITY-inclusive construct add anything over either?

    Answered with encompassing regressions rather than out-of-sample skill,
    because the question is whether one construct's information is contained in
    another's, not which predicts better.

D2  IS R5's ASYMMETRY AN ARTEFACT OF THE BASE TEMPERATURES?
    R5 subtracts a heating slope in %/100 K.h from a cooling slope in the same
    units, but cooling degree-hours use base 24 C and heating degree-hours base
    15 C, and their annual totals differ by an order of magnitude. Those bases
    are conventions. The comparison is repeated at cooling base 18 C and with
    the national population-weighted series.

    The same run also reports the standard deviations behind R3.1's "cold is
    3.3 times heat" statement. That ratio is computed per standard deviation
    while R5 works per 100 K.h, so the two are different quantities and the
    text should not present the first as an indication of the second.

D3  IS THE HETEROGENEITY IN R5 ACTUALLY SIGNIFICANT?
    R5.4 makes between-country heterogeneity its headline on the strength of a
    between-country SD and a vote count of signed countries. Cochran's Q and
    I-squared are the standard tests and are trivial to compute from T7.

USAGE
-----
  python 31_r3r5_diagnostics.py                 # Tier A+B, Power
  python 31_r3r5_diagnostics.py --sector Industry
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
BENCH = BASE / "outputs" / "_cache"


def load_25():
    """Reuse the estimator from 25_oxcgrt_robustness.py rather than restate it.

    A second copy of a distributed-lag FE fit is a second chance to get the
    variance estimator subtly wrong, and any difference between the two would
    be indistinguishable from a real finding.
    """
    spec = importlib.util.spec_from_file_location(
        "ox", BASE / "25_oxcgrt_robustness.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ox"] = m
    argv = sys.argv
    sys.argv = ["25_oxcgrt_robustness.py"]      # its argparse runs at import
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def national_benchmarks() -> pd.DataFrame:
    f = sorted(glob.glob(str(BENCH / "nat_bench_*.parquet")))
    if not f:
        raise SystemExit("national benchmark cache missing; run "
                         "17_national_benchmarks.py first")
    b = pd.concat([pd.read_parquet(x) for x in f], ignore_index=True)
    b["date"] = pd.to_datetime(b["date"])
    return b


def build_panel(ox, sector: str, tier: str) -> pd.DataFrame:
    ex = pd.read_parquet(EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"]}[tier]
    ex = ex[ex.tier.isin(keep)]
    em = ox.load_sector(sector)
    p = em.merge(ex, on=["country", "date"], how="inner")
    p = p.merge(national_benchmarks(), on=["country", "date"], how="left")
    p = p[p.v > 0].copy()
    p["y"] = np.log(p.v)
    return p.sort_values(["country", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
def d1_resolution_vs_humidity(ox, p: pd.DataFrame) -> pd.DataFrame:
    """Encompassing tests: what does city scale, and humidity, each add?"""
    print("\n" + "=" * 78)
    print("D1  Does R3-R6 need city-scale data?")
    print("=" * 78)
    need = ["nat_pop_scdh24", "w_SCDH24", "w_twbg_danger"]
    miss = [c for c in need if c not in p.columns or p[c].isna().all()]
    if miss:
        print(f"    missing columns: {miss}; skipping")
        return pd.DataFrame()

    d = p.dropna(subset=need).copy()
    print(f"    n = {len(d):,} country-days, "
          f"{d.country.nunique()} countries\n")

    runs = [
        ("national temperature alone", {"nat": "nat_pop_scdh24"}),
        ("city temperature alone", {"city": "w_SCDH24"}),
        ("city WBGT>28 alone", {"wbgt": "w_twbg_danger"}),
        ("national + city temperature",
         {"nat": "nat_pop_scdh24", "city": "w_SCDH24"}),
        ("national temperature + city WBGT>28",
         {"nat": "nat_pop_scdh24", "wbgt": "w_twbg_danger"}),
        ("city temperature + city WBGT>28",
         {"city": "w_SCDH24", "wbgt": "w_twbg_danger"}),
    ]
    rows = []
    for lab, blocks in runs:
        r = ox.fit(d, blocks)
        if r is None:
            print(f"    {lab:38s} did not converge")
            continue
        parts = []
        for tag in blocks:
            parts.append(f"{tag} {100 * r[tag]['cum']:+6.2f}% "
                         f"(t={r[tag]['t']:+5.2f})")
            rows.append({"specification": lab, "term": tag,
                         "column": blocks[tag], "cum_pct": 100 * r[tag]["cum"],
                         "se_pct": 100 * r[tag]["se"], "t": r[tag]["t"],
                         "n": r["n"]})
        print(f"    {lab:38s} " + "   ".join(parts))
    print("\n    Read the joint rows against the collinearity below: a joint")
    print("    test can only separate two constructs that are not already the")
    print("    same variable.")
    dm = d.groupby("country")[need].transform(lambda s: s - s.mean())
    cc = dm.corr()
    print(f"    within-country r: national vs city temperature "
          f"{cc.loc['nat_pop_scdh24', 'w_SCDH24']:+.3f};  "
          f"city temperature vs WBGT>28 "
          f"{cc.loc['w_SCDH24', 'w_twbg_danger']:+.3f};  "
          f"national temperature vs WBGT>28 "
          f"{cc.loc['nat_pop_scdh24', 'w_twbg_danger']:+.3f}")
    for a in need:
        for b in need:
            if a < b:
                rows.append({"specification": "within-country correlation",
                             "term": f"{a} vs {b}", "column": "",
                             "cum_pct": np.nan, "se_pct": np.nan,
                             "t": cc.loc[a, b], "n": len(d)})
    return pd.DataFrame(rows)


def d1b_resolution_matched(ox, p: pd.DataFrame) -> pd.DataFrame:
    """Resolution alone, with the temporal treatment held identical.

    D1 compared a national DAILY proxy against a city HOURLY integral, because
    17_national_benchmarks.py builds its national degree-hours as
    max(Tmax - 24, 0) x 24 rather than integrating hours. That conflates spatial
    resolution with temporal integration -- the exact confound SI.1's "matched"
    set was created to avoid. Here both sides are daily-Tmax cooling degree
    days, so resolution is the only thing that differs.
    """
    print("\n" + "=" * 78)
    print("D1b Resolution alone: both sides on daily Tmax")
    print("=" * 78)
    fua = ox_build_fua()
    if fua is None:
        print("    could not build the FUA daily-Tmax series; skipping")
        return pd.DataFrame()
    d = p.merge(fua, on=["country", "date"], how="inner")
    d["nat_pop_cdd"] = d["nat_pop_scdh24"] / 24.0
    need = ["nat_pop_cdd", "fua_pop_cdd", "fua_equal_cdd"]
    d = d.dropna(subset=need)
    print(f"    n = {len(d):,} country-days\n")

    rows = []
    for lab, blocks in [
            ("national pop-weighted alone", {"nat": "nat_pop_cdd"}),
            ("FUA pop-weighted alone", {"fua": "fua_pop_cdd"}),
            ("FUA equal-weighted alone", {"fuae": "fua_equal_cdd"}),
            ("national + FUA, both pop-weighted",
             {"nat": "nat_pop_cdd", "fua": "fua_pop_cdd"})]:
        r = ox.fit(d, blocks)
        if r is None:
            print(f"    {lab:36s} did not converge")
            continue
        parts = [f"{k} {100 * r[k]['cum']:+6.2f}% (t={r[k]['t']:+5.2f})"
                 for k in blocks]
        print(f"    {lab:36s} " + "   ".join(parts))
        for k in blocks:
            rows.append({"specification": lab, "term": k,
                         "column": blocks[k], "cum_pct": 100 * r[k]["cum"],
                         "se_pct": 100 * r[k]["se"], "t": r[k]["t"],
                         "n": r["n"]})
    dm = d.groupby("country")[need].transform(lambda s: s - s.mean())
    rr = float(dm[["nat_pop_cdd", "fua_pop_cdd"]].corr().iloc[0, 1])
    print(f"\n    within-country r(national, FUA) = {rr:+.3f}")
    rows.append({"specification": "within-country correlation",
                 "term": "nat_pop_cdd vs fua_pop_cdd", "column": "",
                 "cum_pct": np.nan, "se_pct": np.nan, "t": rr, "n": len(d)})
    return pd.DataFrame(rows)


def ox_build_fua():
    """FUA daily-Tmax cooling degree days, reusing 18_r2_benchmark_test.py."""
    spec = importlib.util.spec_from_file_location(
        "bench", BASE / "18_r2_benchmark_test.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = m
    argv = sys.argv
    sys.argv = ["18_r2_benchmark_test.py"]
    try:
        spec.loader.exec_module(m)
        return m.build_fua_tmax()
    except Exception as e:                       # noqa: BLE001
        print(f"    {type(e).__name__}: {e}")
        return None
    finally:
        sys.argv = argv


# --------------------------------------------------------------------------
def d2_base_sensitivity(ox, p: pd.DataFrame) -> pd.DataFrame:
    """Is the cooling-minus-heating difference specific to bases 24 and 15?"""
    print("\n" + "=" * 78)
    print("D2  Is R5's asymmetry an artefact of the base temperatures?")
    print("=" * 78)

    sd_c, sd_h = p.w_SCDH24.std(), p.w_HDH15.std()
    print(f"    SD of the daily series: cooling(24) {sd_c:8.2f} K.h, "
          f"heating(15) {sd_h:8.2f} K.h, ratio {sd_h / sd_c:.2f}")
    print("    R3.1 reports coefficients per SD and R5 per 100 K.h. With SDs")
    print("    this far apart the two normalisations answer different")
    print("    questions, and a per-SD ratio is not evidence of a per-load")
    print("    asymmetry.\n")

    # fit_r5 hardcodes w_SCDH24 / w_HDH15, so each variant is fed in under
    # those names rather than duplicating the estimator.
    variants = [
        ("cooling base 24 C, heating base 15 C  (published)",
         "w_SCDH24", "w_HDH15"),
        ("cooling base 18 C, heating base 15 C", "w_SCDH18", "w_HDH15"),
        ("national pop-weighted, bases 24 / 15",
         "nat_pop_scdh24", "nat_pop_hdh15"),
    ]
    rows = []
    for lab, ccol, hcol in variants:
        if ccol not in p.columns or hcol not in p.columns:
            print(f"    {lab:44s} missing {ccol}/{hcol}")
            continue
        q = p.dropna(subset=[ccol, hcol]).copy()
        q["w_SCDH24"] = q[ccol]
        q["w_HDH15"] = q[hcol]
        r = ox.fit_r5(q)
        if r is None or "diff" not in r:
            print(f"    {lab:44s} did not converge")
            continue
        # fit_r5 returns {tag: {cum, se, t}} in log units per 100 K.h
        g = {k: 100 * r[k]["cum"] for k in ("heat", "cold", "diff")}
        z = r["diff"]["t"]
        print(f"    {lab:44s} cool {g['heat']:+7.2f}  heat {g['cold']:+7.2f}  "
              f"diff {g['diff']:+7.2f} (z={z:+5.2f})  "
              f"n_slope={r.get('n_slope', '?')}")
        rows.append({"variant": lab, "cooling_col": ccol, "heating_col": hcol,
                     "cooling": g["heat"], "heating": g["cold"],
                     "difference": g["diff"], "diff_z": z,
                     "n_slope": r.get("n_slope", np.nan),
                     "n": r.get("n", np.nan)})
    print("\n    If the difference changes sign or significance across rows,")
    print("    the published number is a property of the base pair, not of")
    print("    the climate.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def d3_heterogeneity(sector: str, tier: str) -> pd.DataFrame:
    """Cochran's Q and I-squared on the country-specific slopes."""
    print("\n" + "=" * 78)
    print("D3  Is the between-country heterogeneity in R5 significant?")
    print("=" * 78)
    f = RESULTS / "R5" / "tables" / \
        f"T6_thermal_asymmetry_{sector.replace(' ', '_')}_tier{tier}.csv"
    if not f.exists():
        print(f"    {f.name} not found; run 22_r5_asymmetry.py first")
        return pd.DataFrame()
    t = pd.read_csv(f)

    rows = []
    for lab, est, se, sub in [
            ("cooling slope", "heat", "heat_se", t[t.heat_ok.fillna(False)]),
            ("heating slope", "cold", "cold_se", t[t.cold_ok.fillna(False)]),
            ("cooling - heating", "diff", "diff_se", t[t.both.fillna(False)])]:
        s = sub.dropna(subset=[est, se])
        if len(s) < 3:
            continue
        y = 100 * s[est].values          # %/100 K.h, as reported
        v = (100 * s[se].values) ** 2
        w = 1.0 / v
        mu = float((w * y).sum() / w.sum())
        Q = float((w * (y - mu) ** 2).sum())
        df = len(s) - 1
        pQ = float(stats.chi2.sf(Q, df))
        I2 = max(0.0, 100 * (Q - df) / Q) if Q > 0 else 0.0
        tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum()))
        print(f"    {lab:20s} k={len(s):3d}  pooled {mu:+7.2f}  "
              f"Q={Q:9.1f} (df={df})  p={pQ:.3g}  I2={I2:5.1f}%  "
              f"tau={np.sqrt(tau2):6.2f}")
        rows.append({"quantity": lab, "k": len(s), "pooled": mu, "Q": Q,
                     "df": df, "p_Q": pQ, "I2_pct": I2,
                     "tau": float(np.sqrt(tau2))})
    print("\n    Q tests the null that every country shares one slope. I2 is")
    print("    the share of observed variation not attributable to sampling")
    print("    error. R5.4 currently asserts heterogeneity from a between-")
    print("    country SD and a count of signed countries; these are the")
    print("    standard tests of the same claim.")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="Power")
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C"])
    ap.add_argument("--out", default=None,
                    help="directory for the three CSVs (default: no write)")
    args = ap.parse_args()

    ox = load_25()
    print("=" * 78)
    print(f"R3-R5 diagnostics: {args.sector}, Tier {args.tier}")
    print("=" * 78)
    p = build_panel(ox, args.sector, args.tier)
    print(f"panel: {len(p):,} country-days, {p.country.nunique()} countries, "
          f"{p.date.min().date()} .. {p.date.max().date()}")

    d1 = d1_resolution_vs_humidity(ox, p)
    d1b = d1b_resolution_matched(ox, p)
    d2 = d2_base_sensitivity(ox, p)
    d3 = d3_heterogeneity(args.sector, args.tier)

    # The two encompassing tests answer a question the paper has to address --
    # whether city resolution and humidity add anything -- so they belong with
    # the results rather than in a scratch directory. The T-number is
    # provisional until the response sections are renumbered. D2 and D3 stay
    # diagnostics: D2 is superseded by the full scan in 32, and D3 is now
    # produced inside 22 alongside the per-country table.
    enc = pd.concat([d for d in (d1, d1b) if len(d)], ignore_index=True) \
        if (len(d1) or len(d1b)) else pd.DataFrame()
    if len(enc):
        tab = RESULTS / "R3" / "tables"
        tab.mkdir(parents=True, exist_ok=True)
        f = tab / f"T4c_resolution_and_humidity_tier{args.tier}.csv"
        enc.round(5).to_csv(f, index=False)
        print(f"\n    wrote {f}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in [("D1_resolution_vs_humidity", d1),
                         ("D1b_resolution_matched", d1b),
                         ("D2_base_sensitivity", d2),
                         ("D3_heterogeneity", d3)]:
            if len(df):
                df.round(5).to_csv(out / f"{name}.csv", index=False)
                print(f"    wrote {out / f'{name}.csv'}")


if __name__ == "__main__":
    main()
