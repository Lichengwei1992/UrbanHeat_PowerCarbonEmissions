#!/usr/bin/env python
"""
15_power_response.py  --  R3 minimal Power model, the Go/No-Go gate
====================================================================

Step 3 of the revised pipeline. Estimates the distributed-lag response of
national daily sectoral CO2 to city-weighted thermal exposure, and evaluates
the five Go/No-Go criteria agreed before any results were seen.

OUTCOME SCALE
-------------
Emissions are summed over the sampled cities of each country. Within a
calendar month every city's share of the national total is constant to
floating-point precision, so

    sum_i E_(i,t) = (sum_i share_(i,m)) * CM_(c,t)

and the coverage term is absorbed entirely by the country x year-month fixed
effect. The estimates are therefore identical to using the national Carbon
Monitor series directly, while retaining 46 countries instead of the 11 in
`Global_daily_sectoral`.

MODEL
-----
    ln E_(c,t) = mu_(c,ym) + lambda_t
                 + sum_(k=-7..+7) beta_k * Exposure_(c,t-k)
                 + country-specific day-of-week + eps

  * mu_(c,ym)  country x year-month FE: absorbs coverage, season, trend,
               national policy and fuel-mix drift within a month
  * lambda_t   global date FE: absorbs worldwide common shocks (COVID timing)
  * k < 0      leads, used only as the parallel-trends check
  * SE         two-way clustered on country and date

FIXED EFFECTS ARE PARTIALLED OUT BY ALTERNATING PROJECTIONS with an explicit
convergence check -- the residual group means are reported, because a silent
non-convergence here would invalidate every coefficient.

GO / NO-GO CRITERIA (predefined)
--------------------------------
  1. City-weighted exposure beats equal-weighted (incremental fit)
  2. Power cumulative response is sign-sensible and survives dropping 2020
  3. Leads jointly insignificant (no systematic anticipation)
  4. Aviation / Industry do not show a stronger heat response than Power
  5. Result is not driven by the US, China or Japan alone

INPUTS
------
  CarbonData/Near-real-time .../carbon-monitor-cities-*-v0325.csv
  outputs/exposure/country_daily_exposure.parquet

OUTPUT
------
  outputs/response/power_gate_results.parquet
  outputs/response/tables/*.csv

USAGE
-----
  python 15_power_response.py
  python 15_power_response.py --exposure w_SCDH24 --tier A
"""

from __future__ import annotations

import os
import argparse
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
# Emissions directory. Every loader in this project globs
# carbon-monitor-cities-<Country>-v0325.csv from here. HC_CARBON_DIR points the
# whole pipeline at an alternative directory -- e.g. a 2019-2022 merge built by
# 28_merge_emissions.py -- without touching the published 2019-2021 files.
CD = Path(os.environ.get(
    "HC_CARBON_DIR",
    str(BASE / "CarbonData" /
        "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide")))
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
OUT_DIR = BASE / "outputs" / "response"
TABLE_DIR = OUT_DIR / "tables"

SECTORS = ["Power", "Ground Transport", "Industry", "Aviation", "Residential"]
LEADS, LAGS = 7, 7


# --------------------------------------------------------------------------
def load_emissions() -> pd.DataFrame:
    """Country-day-sector emissions, summed over sampled cities."""
    rows = []
    for f in sorted(glob.glob(str(CD / "carbon-monitor-cities-*-v0325.csv"))):
        name = Path(f).stem.replace("carbon-monitor-cities-", "").replace("-v0325", "")
        if "all-cities" in name or "GADM" in name or "Counties" in name:
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip().lower().replace("﻿", "") for c in d.columns]
        if "city" not in d.columns:
            continue
        d = d.rename(columns={"value (ktco2 per day)": "v"})
        d = d[d["sector"].isin(SECTORS)]
        d["country"] = name
        rows.append(d[["country", "date", "sector", "v"]])
    e = pd.concat(rows, ignore_index=True)
    e["date"] = pd.to_datetime(e["date"])
    return (e.groupby(["country", "date", "sector"], as_index=False)["v"].sum())


def demean_fe(X: np.ndarray, f1: np.ndarray, f2: np.ndarray,
              tol: float = 1e-10, max_iter: int = 200) -> tuple[np.ndarray, float]:
    """Two-way within transformation by alternating projections."""
    X = np.asarray(X, dtype="float64").copy()
    n1, n2 = f1.max() + 1, f2.max() + 1
    c1 = np.bincount(f1, minlength=n1).astype(float)
    c2 = np.bincount(f2, minlength=n2).astype(float)
    for it in range(max_iter):
        prev = X.copy()
        for j in range(X.shape[1]):
            X[:, j] -= (np.bincount(f1, weights=X[:, j], minlength=n1) / c1)[f1]
        for j in range(X.shape[1]):
            X[:, j] -= (np.bincount(f2, weights=X[:, j], minlength=n2) / c2)[f2]
        delta = np.max(np.abs(X - prev))
        if delta < tol:
            break
    # convergence diagnostic: largest surviving group mean
    resid = max(
        max(abs(np.bincount(f, weights=X[:, j], minlength=n) / c).max()
            for j in range(X.shape[1]))
        for f, c, n in ((f1, c1, n1), (f2, c2, n2)))
    return X, resid


def twoway_cluster_vcov(X: np.ndarray, e: np.ndarray,
                        g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    XtX_inv = np.linalg.inv(X.T @ X)

    def meat(g):
        n = g.max() + 1
        S = np.zeros((n, X.shape[1]))
        Xe = X * e[:, None]
        for j in range(X.shape[1]):
            S[:, j] = np.bincount(g, weights=Xe[:, j], minlength=n)
        return S.T @ S

    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]
    M = meat(g1) + meat(g2) - meat(g12)
    return XtX_inv @ M @ XtX_inv


def fit(df: pd.DataFrame, xcol: str, label: str, verbose: bool = True):
    """Distributed-lag FE regression; returns coefficient table."""
    d = df.sort_values(["country", "date"]).reset_index(drop=True).copy()
    g = d.groupby("country")[xcol]
    cols = []
    for s in range(-LEADS, LAGS + 1):
        nm = f"L{'m' if s < 0 else 'p'}{abs(s)}"
        d[nm] = g.shift(s)          # s>0: exposure s days earlier
        cols.append(nm)
    d = d.dropna(subset=cols + ["y"]).reset_index(drop=True)
    if len(d) < 500:
        return None

    dow = pd.get_dummies(d["country"] + "_" + d["date"].dt.dayofweek.astype(str),
                         drop_first=True).astype(float)
    M = np.column_stack([d[cols].to_numpy(dtype="float64"), dow.to_numpy(),
                         d["y"].to_numpy()])
    f1 = pd.factorize(d["country"] + "_" + d["date"].dt.to_period("M").astype(str))[0]
    f2 = pd.factorize(d["date"])[0]
    Z, conv = demean_fe(M, f1, f2)
    X, y = Z[:, :-1], Z[:, -1]

    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-10
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    V = twoway_cluster_vcov(X, e, pd.factorize(d["country"])[0], f2)
    se = np.sqrt(np.diag(V))
    r2 = 1 - e.var() / y.var()

    k = len(cols)
    res = pd.DataFrame({"lag": list(range(-LEADS, LAGS + 1)),
                        "beta": b[:k], "se": se[:k]})
    res["t"] = res.beta / res.se
    cum = b[LEADS:k].sum()
    cum_se = np.sqrt(V[LEADS:k, LEADS:k].sum())
    lead_idx = np.arange(0, LEADS)
    Vl = V[np.ix_(lead_idx, lead_idx)]
    wald = b[lead_idx] @ np.linalg.solve(Vl, b[lead_idx])

    if verbose:
        print(f"    {label:34s} n={len(d):6,}  R2={r2:.4f}  conv={conv:.1e}")
        print(f"      cumulative beta(0..{LAGS}) = {100*cum:+7.3f}%  "
              f"(SE {100*cum_se:.3f}, t={cum/cum_se:+.2f})"
              f"   lead Wald chi2({LEADS})={wald:.1f}")
    return {"table": res, "cum": cum, "cum_se": cum_se, "lead_wald": wald,
            "n": len(d), "r2": r2, "conv": conv, "label": label}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", default="w_SCDH24",
                    help="Primary exposure column (default w_SCDH24).")
    ap.add_argument("--tier", default="A", choices=["A", "AB", "all"],
                    help="Coverage tier restriction for the main sample.")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("R3 gate: sectoral CO2 response to city-weighted thermal exposure")
    print("=" * 78)

    print("\n[1] Emissions")
    em = load_emissions()
    print(f"    {len(em):,} country-day-sector | {em.country.nunique()} countries")

    print("[2] Exposure")
    ex = pd.read_parquet(EXPOSURE)
    keep_tier = {"A": ["A"], "AB": ["A", "B"], "all": ["A", "B", "C"]}[args.tier]
    ex_m = ex[ex.tier.isin(keep_tier)]
    print(f"    tier {args.tier}: {ex_m.country.nunique()} countries, "
          f"{len(ex_m):,} country-days")

    xcol = args.exposure
    ecol = xcol.replace("w_", "e_", 1)
    panel = em.merge(ex_m, on=["country", "date"], how="inner")
    panel = panel[panel.v > 0].copy()
    panel["y"] = np.log(panel.v)
    # standardise exposure so coefficients read as "per SD of exposure"
    for c in (xcol, ecol):
        if c in panel.columns:
            sd = panel[c].std()
            panel[c] = panel[c] / (sd if sd > 0 else 1)
    print(f"    merged panel: {len(panel):,} rows, "
          f"{panel.country.nunique()} countries, {panel.sector.nunique()} sectors")

    results = {}
    print(f"\n[3] Distributed-lag response to {xcol} (per SD, %)")
    for sec in SECTORS:
        sub = panel[panel.sector == sec]
        if len(sub) < 1000:
            continue
        r = fit(sub, xcol, sec)
        if r:
            results[sec] = r

    # ---- Go/No-Go -------------------------------------------------------
    print("\n" + "=" * 78)
    print("GO / NO-GO")
    print("=" * 78)
    verdict = {}

    # 1 weighted vs equal-weighted
    print("\n[1] City-weighted vs equal-weighted exposure (Power)")
    p = panel[panel.sector == "Power"]
    rw = results.get("Power")
    re_ = fit(p, ecol, "Power / equal-weighted") if ecol in p.columns else None
    if rw and re_:
        better = rw["r2"] > re_["r2"]
        print(f"    weighted R2 {rw['r2']:.5f}  vs  equal {re_['r2']:.5f}  "
              f"-> {'PASS' if better else 'FAIL'}")
        verdict["1_weighting"] = bool(better)

    # 2 sign + drop 2020
    print("\n[2] Power cumulative response, sign and 2020 exclusion")
    if rw:
        r20 = fit(p[p.date.dt.year != 2020], xcol, "Power / excl. 2020")
        same = r20 and np.sign(r20["cum"]) == np.sign(rw["cum"])
        sig = abs(rw["cum"] / rw["cum_se"]) > 1.96
        print(f"    full {100*rw['cum']:+.3f}%  excl-2020 "
              f"{100*r20['cum']:+.3f}%  -> "
              f"{'PASS' if (same and sig) else 'FAIL'}"
              f" (sign stable={bool(same)}, |t|>1.96={bool(sig)})")
        verdict["2_sign_stability"] = bool(same and sig)

    # 3 leads
    print("\n[3] Parallel trends (leads jointly zero)")
    if rw:
        from scipy import stats
        pv = 1 - stats.chi2.cdf(rw["lead_wald"], LEADS)
        print(f"    Wald chi2({LEADS}) = {rw['lead_wald']:.1f}, p = {pv:.3f}  -> "
              f"{'PASS' if pv > 0.05 else 'FAIL'}")
        verdict["3_no_pretrend"] = bool(pv > 0.05)

    # 4 placebo ordering
    print("\n[4] Power vs Aviation / Industry")
    if rw:
        others = {s: results[s]["cum"] for s in ("Aviation", "Industry")
                  if s in results}
        ok = all(abs(v) < abs(rw["cum"]) for v in others.values()) if others else None
        for s, v in others.items():
            print(f"    {s:18s} {100*v:+.3f}%   vs Power {100*rw['cum']:+.3f}%")
        print(f"    -> {'PASS' if ok else 'FAIL'}")
        verdict["4_placebo_order"] = bool(ok)

    # 5 leave-one-country-out
    print("\n[5] Leave-one-country-out (US / China / Japan)")
    if rw:
        base = rw["cum"]
        rows = []
        for c in ("United States", "China", "Japan"):
            if c not in p.country.unique():
                continue
            rr = fit(p[p.country != c], xcol, f"Power / drop {c}", verbose=False)
            if rr:
                rows.append((c, rr["cum"]))
                print(f"    drop {c:16s} {100*rr['cum']:+.3f}%  "
                      f"(base {100*base:+.3f}%)")
        ok = all(np.sign(v) == np.sign(base) and abs(v) > 0.3 * abs(base)
                 for _, v in rows) if rows else None
        print(f"    -> {'PASS' if ok else 'FAIL'}")
        verdict["5_not_driven_by_one"] = bool(ok)

    print("\n" + "-" * 78)
    n_pass = sum(1 for v in verdict.values() if v)
    for k, v in verdict.items():
        print(f"    {k:24s} {'PASS' if v else 'FAIL'}")
    print(f"\n    {n_pass}/{len(verdict)} criteria passed -> "
          f"{'GO' if n_pass >= 4 else 'NO-GO / redesign'}")

    # ---- save -----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    allres = []
    for sec, r in results.items():
        t = r["table"].copy()
        t["sector"] = sec
        t["n"] = r["n"]
        t["r2"] = r["r2"]
        allres.append(t)
    if allres:
        out = pd.concat(allres, ignore_index=True)
        out.to_parquet(OUT_DIR / "power_gate_results.parquet", index=False)
        out.to_csv(TABLE_DIR / "sector_lag_coefficients.csv", index=False)
        pd.DataFrame([{"criterion": k, "pass": v} for k, v in verdict.items()]).to_csv(
            TABLE_DIR / "go_no_go.csv", index=False)
        print(f"\n    saved {OUT_DIR/'power_gate_results.parquet'}")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
