#!/usr/bin/env python
"""
24_r6_weight_competition.py  --  R6: does built-environment scale matter?
=========================================================================

R6 was originally framed as a competition between aggregation weights: build
national exposure by weighting cities with population, with built-up surface,
and with built-up volume, and see which best predicts daily sectoral CO2. That
comparison is run here in full (R6.1), but a diagnostic computed before any
model is fitted says what to expect: within a country, built-up surface and
built-up volume rank cities almost exactly as population does (median Spearman
rho = 0.94 and 0.95). Two weights that order the same cities the same way
produce nearly the same weighted mean, so the horse race is a test with very
little to separate.

The scientifically informative version of the same question does not live in
the weights at all. It lives in the SLOPE: does a country with more built
volume per person convert a degree-hour into more carbon? That is a
country-level MODERATOR, and it is the form in which built-environment scale
can actually be tested with this data (R6.2). It is put in competition against
the obvious rival mechanism -- air-conditioning penetration -- which is also
only available per country.

WHY AC PENETRATION IS NOT A WEIGHT
----------------------------------
Any factor constant within a country cancels exactly out of a weighted mean
over that country's cities. AC penetration is a national statistic, so it can
never be a weighting scheme, only a moderator. The same is true of any
country-level covariate, and it is the reason R6.2 rather than R6.1 carries
the mechanism content.

THE PHYSICAL CLAIM THAT MAY NOT BE MADE
---------------------------------------
GHS_BUILT_S is built-up surface and GHS_BUILT_V is gross built volume. Neither
is the thermal envelope area A of Q = U A dT, and this data has no U, no indoor
set-point and no operating hours. A positive moderator coefficient is reported
as CONSISTENT WITH built-environment scale modulating heat-sensitive energy
demand, and never as evidence for a conduction model.

THE SAMPLE CONSTRAINT THAT DOES NOT GO AWAY
-------------------------------------------
Comparing five weighting schemes does not create new independent information:
the number of independent units is still the number of countries (28 in Tier
A+B, 16 with both thermal sides identified). A moderator regression on 28
country-level slopes has low power and is reported as descriptive.

PROTOCOL
--------
Model comparison is out-of-sample by leave-one-country-out and leave-one-year-
out, never random splits of country-days: daily emissions are strongly
autocorrelated and a random split leaks across the fold boundary. Encompassing
tests enter two weightings jointly, which has more power than comparing two
tiny out-of-sample R2 values.

INPUTS
------
  outputs/04_events/city_events_daily_v2.parquet
  outputs/population/fua_weight_schemes.parquet
  outputs/exposure/country_daily_exposure.parquet      (tiers)
  outputs/response/r5_asymmetry_tierAB.parquet          (country slopes)
  AuxData/ac_penetration_country.csv
  CarbonData/Near-real-time .../carbon-monitor-cities-*-v0325.csv

OUTPUTS
-------
  outputs/exposure/country_exposure_by_weight.parquet
  Results/R6/tables/T10_weight_race_tier{AB,C}.csv
  Results/R6/tables/T11_moderators_tier{AB,C}.csv
  Results/R6/figures/Fig7_built_environment.{png,pdf,svg}

USAGE
-----
  python 24_r6_weight_competition.py --tier AB
"""

from __future__ import annotations

import os
import argparse
import glob
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
# Results root. Overridable with the HC_RESULTS_ROOT environment variable so an
# extended-period rerun can write a parallel result set instead of overwriting
# the published one. Unset, it resolves to the original BASE/"Results".
RESULTS_ROOT = Path(os.environ.get("HC_RESULTS_ROOT", str(BASE / "Results")))

# Emissions directory. Every loader in this project globs
# carbon-monitor-cities-<Country>-v0325.csv from here. HC_CARBON_DIR points the
# whole pipeline at an alternative directory -- e.g. a 2019-2022 merge built by
# 28_merge_emissions.py -- without touching the published 2019-2021 files.
CD = Path(os.environ.get(
    "HC_CARBON_DIR",
    str(BASE / "CarbonData" /
        "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide")))
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
SCHEMES = BASE / "outputs" / "population" / "fua_weight_schemes.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
R5 = BASE / "outputs" / "response" / "r5_asymmetry_tierAB.parquet"
AC_CSV = BASE / "AuxData" / "ac_penetration_country.csv"
OUT_DIR = BASE / "outputs" / "exposure"
FIG_DIR = RESULTS_ROOT / "R6" / "figures"
TAB_DIR = RESULTS_ROOT / "R6" / "tables"

LAGS = 7
WEIGHTS = ["w_equal", "w_pop", "w_built_s", "w_built_v", "w_pop_vint"]
LABEL = {"w_equal": "equal", "w_pop": "population", "w_built_s": "built surface",
         "w_built_v": "built volume", "w_pop_vint": "pop x volume intensity"}
SAVE_FORMATS = ("png", "pdf", "svg")
SAVE_DPI = 600
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "savefig.dpi": SAVE_DPI, "figure.dpi": 110,
                     "svg.fonttype": "none"})


# --------------------------------------------------------------------------
def load_power() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(CD / "carbon-monitor-cities-*-v0325.csv"))):
        n = Path(f).stem.replace("carbon-monitor-cities-", "").replace("-v0325", "")
        if "all-cities" in n or "GADM" in n or "Counties" in n:
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip().lower().replace("﻿", "") for c in d.columns]
        if "city" not in d.columns:
            continue
        d = d.rename(columns={"value (ktco2 per day)": "v"})
        d = d[d.sector == "Power"]
        d["country"] = n
        rows.append(d[["country", "date", "v"]])
    e = pd.concat(rows, ignore_index=True)
    e["date"] = pd.to_datetime(e["date"])
    return e.groupby(["country", "date"], as_index=False)["v"].sum()


def build_exposure_by_weight() -> pd.DataFrame:
    """Country-day SCDH24 and HDH15 under every weighting scheme."""
    ev = pd.read_parquet(EVENTS, columns=["city_id", "date", "SCDH24", "HDH15"])
    ev["city_id"] = ev["city_id"].astype(str).str.zfill(4)
    ev["date"] = pd.to_datetime(ev["date"])
    w = pd.read_parquet(SCHEMES)
    w["city_id"] = w["city_id"].astype(str).str.zfill(4)
    d = ev.merge(w[["city_id", "country"] + WEIGHTS], on="city_id", how="inner")
    print(f"    {len(d):,} city-days, {d.city_id.nunique():,} cities, "
          f"{d.country.nunique()} countries")

    out = None
    for wt in WEIGHTS:
        num = pd.DataFrame({
            "country": d["country"].values, "date": d["date"].values,
            "num_c": (d["SCDH24"] * d[wt]).values,
            "num_h": (d["HDH15"] * d[wt]).values,
            "den": d[wt].values})
        g = num.groupby(["country", "date"], sort=False).sum()
        r = pd.DataFrame({
            f"{wt}__scdh": g["num_c"] / g["den"],
            f"{wt}__hdh": g["num_h"] / g["den"]}).reset_index()
        out = r if out is None else out.merge(r, on=["country", "date"])
    return out


def demean(X, facs, it=200, tol=1e-11):
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


def oos_score(panel: pd.DataFrame, cols: list[str], folds: str) -> float:
    """Leave-one-group-out OOS R2 for a distributed-lag FE model."""
    d = panel.sort_values(["country", "date"]).reset_index(drop=True).copy()
    nm = []
    for c in cols:
        sd = d[c].std()
        d[c + "_z"] = d[c] / (sd if sd > 0 else 1)
        g = d.groupby("country")[c + "_z"]
        for s in range(LAGS + 1):
            k = f"{c}_L{s}"
            d[k] = g.shift(s)
            nm.append(k)
    d = d.dropna(subset=nm + ["y"]).reset_index(drop=True)
    groups = d["country"] if folds == "country" else d["date"].dt.year
    sse = sst = 0.0
    for gv in groups.unique():
        tr, te = d[groups != gv], d[groups == gv]
        if len(te) < 50 or len(tr) < 500:
            continue
        f1 = pd.factorize(tr.country + "_" + tr.date.dt.to_period("M").astype(str))[0]
        f2 = pd.factorize(tr.date)[0]
        Z = demean(np.column_stack([tr[nm].values, tr.y.values]), [f1, f2])
        X, yy = Z[:, :-1], Z[:, -1]
        b = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * 1e-8, X.T @ yy)
        h1 = pd.factorize(te.country + "_" + te.date.dt.to_period("M").astype(str))[0]
        h2 = pd.factorize(te.date)[0]
        fac = [h1] + ([h2] if len(te) / max(h2.max() + 1, 1) >= 2 else [])
        Zt = demean(np.column_stack([te[nm].values, te.y.values]), fac)
        Xt, yt = Zt[:, :-1], Zt[:, -1]
        sse += float(((yt - Xt @ b) ** 2).sum())
        sst += float((yt ** 2).sum())
    return 1 - sse / sst if sst > 0 else np.nan


def fit_cum(d: pd.DataFrame, blocks: dict[str, str]) -> dict | None:
    """Distributed-lag FE fit with two-way clustered, PSD-projected variance."""
    d = d.sort_values(["country", "date"]).reset_index(drop=True).copy()
    names, index = [], {}
    for tag, col in blocks.items():
        sd = d[col].std()
        if not np.isfinite(sd) or sd == 0:
            return None
        d[col + "_z"] = d[col] / sd
        g = d.groupby("country")[col + "_z"]
        idx = []
        for s in range(LAGS + 1):
            nm = f"{tag}_L{s}"
            d[nm] = g.shift(s)
            names.append(nm)
            idx.append(len(names) - 1)
        index[tag] = np.array(idx)
    d = d.dropna(subset=names + ["y"]).reset_index(drop=True)
    if len(d) < 500:
        return None
    f1 = pd.factorize(d.country + "_" + d.date.dt.to_period("M").astype(str))[0]
    f2 = pd.factorize(d.date)[0]
    f3 = pd.factorize(d.country + "_" + d.date.dt.dayofweek.astype(str))[0]
    Z = demean(np.column_stack([d[names].values, d.y.values]), [f1, f2, f3])
    X, y = Z[:, :-1], Z[:, -1]
    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-10
    inv = np.linalg.inv(XtX)
    b = inv @ (X.T @ y)
    e = y - X @ b

    def meat(g):
        n = g.max() + 1
        Xe = X * e[:, None]
        S = np.zeros((n, X.shape[1]))
        for j in range(X.shape[1]):
            S[:, j] = np.bincount(g, weights=Xe[:, j], minlength=n)
        return S.T @ S

    g1 = pd.factorize(d.country)[0]
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(f2).astype(str))[0]
    M = meat(g1) + meat(f2) - meat(g12)
    M = 0.5 * (M + M.T)
    wv, Q = np.linalg.eigh(M)
    M = (Q * np.clip(wv, 0, None)) @ Q.T
    V = inv @ M @ inv
    out = {"n": len(d)}
    for tag, ix in index.items():
        cum = float(b[ix].sum())
        se = float(np.sqrt(max(float(V[np.ix_(ix, ix)].sum()), 0)))
        out[tag] = {"cum": cum, "se": se, "t": cum / se if se else np.nan}
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C", "all"])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("R6: built-environment weighting and moderation")
    print("=" * 78)

    print("\n[1] Country-day exposure under every weighting scheme")
    ex = build_exposure_by_weight()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ex.to_parquet(OUT_DIR / "country_exposure_by_weight.parquet", index=False)

    tiers = pd.read_parquet(EXPOSURE)[["country", "date", "tier"]]
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"],
            "all": ["A", "B", "C"]}[args.tier]
    ex = ex.merge(tiers, on=["country", "date"], how="inner")
    ex = ex[ex.tier.isin(keep)].drop(columns=["tier"])

    p = load_power().merge(ex, on=["country", "date"], how="inner")
    p = p[p.v > 0].copy()
    p["y"] = np.log(p.v)
    print(f"    Tier {args.tier}: {p.country.nunique()} countries, "
          f"{len(p):,} country-days")

    # ---- how different are the resulting series at all? -------------------
    print("\n[2] Correlation of the national exposure series they produce")
    base = "w_pop__scdh"
    print(f"    {'scheme':24s}{'corr with pop':>15s}{'mean abs diff':>16s}")
    corr_rows = []
    for wt in WEIGHTS:
        col = f"{wt}__scdh"
        r = float(np.corrcoef(p[col], p[base])[0, 1])
        md = float(np.abs(p[col] - p[base]).mean())
        corr_rows.append({"scheme": LABEL[wt], "corr_with_pop": r,
                          "mean_abs_diff_Kh": md})
        print(f"    {LABEL[wt]:24s}{r:15.4f}{md:16.2f}")
    print("    two exposure series correlated at r > 0.99 cannot be")
    print("    distinguished by any model fitted to 30,000 country-days")

    # ---- R6.1 the weighting race -----------------------------------------
    print("\n[3] Weighting race: out-of-sample skill, cooling exposure")
    print(f"    {'scheme':24s}{'OOS R2 (LOCO)':>15s}{'OOS R2 (LOYO)':>15s}"
          f"{'sum beta/SD (%)':>17s}{'t':>7s}")
    race = []
    for wt in WEIGHTS:
        col = f"{wt}__scdh"
        a = oos_score(p, [col], "country")
        b = oos_score(p, [col], "year")
        r = fit_cum(p, {"x": col})
        cum = 100 * r["x"]["cum"] if r else np.nan
        tt = r["x"]["t"] if r else np.nan
        race.append({"scheme": LABEL[wt], "column": col, "oos_r2_loco": a,
                     "oos_r2_loyo": b, "pct_per_sd": cum, "t": tt})
        print(f"    {LABEL[wt]:24s}{a:15.5f}{b:15.5f}{cum:17.3f}{tt:7.2f}")

    rdf = pd.DataFrame(race)
    best_loco = rdf.loc[rdf.oos_r2_loco.idxmax(), "scheme"]
    best_loyo = rdf.loc[rdf.oos_r2_loyo.idxmax(), "scheme"]
    spread = rdf.oos_r2_loco.max() - rdf.oos_r2_loco.min()
    print(f"    best by LOCO: {best_loco};  best by LOYO: {best_loyo}")
    print(f"    spread across all five schemes: {spread:.5f} in OOS R2")

    # ---- encompassing tests ----------------------------------------------
    print("\n[4] Encompassing tests: does a built weight add to population?")
    print(f"    {'pair':38s}{'pop':>16s}{'challenger':>16s}")
    enc = []
    for wt in ("w_built_s", "w_built_v", "w_pop_vint", "w_equal"):
        r = fit_cum(p, {"pop": "w_pop__scdh", "alt": f"{wt}__scdh"})
        if r is None:
            continue
        enc.append({"challenger": LABEL[wt],
                    "pop_cum_pct": 100 * r["pop"]["cum"], "pop_t": r["pop"]["t"],
                    "alt_cum_pct": 100 * r["alt"]["cum"], "alt_t": r["alt"]["t"],
                    "n": r["n"]})
        print(f"    population vs {LABEL[wt]:24s}"
              f"{100 * r['pop']['cum']:9.2f}({r['pop']['t']:+.1f})"
              f"{100 * r['alt']['cum']:9.2f}({r['alt']['t']:+.1f})")
    print("    with two near-collinear regressors both coefficients become")
    print("    unstable; this is a symptom of the collinearity in [2], not")
    print("    evidence that either weight is wrong")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(corr_rows).merge(rdf, on="scheme").to_csv(
        TAB_DIR / f"T10_weight_race_tier{args.tier}.csv", index=False)
    if enc:
        pd.DataFrame(enc).to_csv(
            TAB_DIR / f"T10b_encompassing_tier{args.tier}.csv", index=False)

    # ---- R6.2 moderators --------------------------------------------------
    print("\n[5] Country-level moderators of the cooling response")
    mod = moderator_test(args.tier)
    if mod is not None:
        mod.to_csv(TAB_DIR / f"T11_moderators_tier{args.tier}.csv", index=False)
        print(f"    R6/tables/T11_moderators_tier{args.tier}.csv")

    print("\n    WORDING: GHS_BUILT_S/V are built surface and gross volume, not")
    print("    envelope area; no U, set-point or operating hours exist here. A")
    print("    positive moderator is CONSISTENT WITH built-environment scale")
    print("    modulating heat-sensitive demand, and is not a test of Q=UAdT.")

    if not args.no_figure:
        make_figure(p, rdf, pd.DataFrame(corr_rows), mod, args.tier)
    print(f"\nDone in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------
def moderator_test(tier: str) -> pd.DataFrame | None:
    """Do country slopes rise with built volume per capita, or with AC stock?

    The country-specific cooling slopes estimated in R5 are regressed on
    country-level candidate moderators. With 18 identified countries this is a
    descriptive association with wide intervals, weighted by the precision of
    each slope; it is not a causal decomposition and no multiple-moderator
    model is fitted beyond the two-variable horse race.
    """
    if not R5.exists():
        print("    R5 slopes not found; run 22_r5_asymmetry.py first")
        return None
    r5 = pd.read_parquet(R5)
    r5 = r5[r5.heat_ok.fillna(False) & r5["heat"].notna()].copy()

    w = pd.read_parquet(SCHEMES)
    # NOTE ON THE SCHEMES TABLE. In fua_weight_schemes.parquet the columns
    # `built_s` and `built_v` hold the zonal sum for the whole FUA, repeated on
    # every city row that shares a fuacode; only `w_built_s` and `w_built_v` are
    # divided by n_share. `pop_weight_dedup` is already deduplicated. Summing or
    # weighting by the raw columns therefore double counts 34 cities in 5
    # countries -- 24 of them in China, which inflates its built volume per capita
    # by 31 % -- while leaving population correct. The deduplicated columns are
    # used everywhere below.
    agg = w.groupby("country").agg(built_v=("w_built_v", "sum"),
                                   built_s=("w_built_s", "sum"),
                                   pop=("pop_weight_dedup", "sum")).reset_index()
    agg["vol_per_cap"] = agg.built_v / agg["pop"]
    agg["surf_per_cap"] = agg.built_s / agg["pop"]
    d = r5.merge(agg, on="country", how="left")
    if AC_CSV.exists():
        d = d.merge(pd.read_csv(AC_CSV), on="country", how="left")

    cands = [("vol_per_cap", "built volume per capita (m3)"),
             ("surf_per_cap", "built surface per capita (m2)"),
             ("ac_ownership_pct", "AC penetration (%)")]

    def wls(s, cols):
        """Precision-weighted regression of country slopes on moderators."""
        X = np.column_stack([np.ones(len(s))] + [
            (s[c].values.astype(float) - s[c].values.astype(float).mean())
            / s[c].values.astype(float).std() for c in cols])
        yv = 100 * s["heat"].values
        wt = 1.0 / np.maximum((100 * s["heat_se"].values) ** 2, 1e-9)
        XtWX = X.T @ (X * wt[:, None])
        b = np.linalg.solve(XtWX, X.T @ (wt * yv))
        resid = yv - X @ b
        dof = max(len(s) - X.shape[1], 1)
        s2 = float((wt * resid ** 2).sum() / dof)
        V = s2 * np.linalg.inv(XtWX)
        ss = float(((yv - yv.mean()) ** 2 * wt).sum())
        r2 = 1 - float((wt * resid ** 2).sum()) / ss if ss > 0 else np.nan
        return b, V, r2

    rows = []
    print(f"    {'moderator':34s}{'n':>4s}{'slope':>12s}{'t':>7s}{'R2':>7s}")
    for col, lab in cands:
        s = d[d[col].notna() & d["heat"].notna()]
        if len(s) < 6:
            print(f"    {lab:34s}{len(s):4d}   too few countries")
            continue
        b, V, r2 = wls(s, [col])
        t = b[1] / np.sqrt(V[1, 1])
        rows.append({"moderator": lab, "column": col, "n_countries": len(s),
                     "slope_per_sd": b[1], "se": float(np.sqrt(V[1, 1])),
                     "t": t, "r2": r2})
        print(f"    {lab:34s}{len(s):4d}{b[1]:12.2f}{t:7.2f}{r2:7.3f}")
    print("    slope is the change in the cooling response (%/100 K.h) per one")
    print("    SD of the moderator, precision-weighted across countries")

    # ---- are the moderators telling the same story? ----------------------
    print("\n[5b] Are the moderators distinguishable from each other?")
    ok = d[d.vol_per_cap.notna() & d.ac_ownership_pct.notna()
           & d["heat"].notna()]
    if len(ok) >= 6:
        rho = float(ok[["vol_per_cap", "ac_ownership_pct"]].corr().iloc[0, 1])
        print(f"    corr(volume per capita, AC penetration) = {rho:+.3f} "
              f"over {len(ok)} countries")
        b, V, r2 = wls(ok, ["vol_per_cap", "ac_ownership_pct"])
        print(f"    entered jointly (n = {len(ok)}):")
        for i, lab in enumerate(["built volume per capita", "AC penetration"],
                                start=1):
            print(f"      {lab:26s}{b[i]:9.2f}  (t = "
                  f"{b[i] / np.sqrt(V[i, i]):+.2f})")
        print("    both are proxies for national income; with this many")
        print("    countries they cannot be separated, and neither is claimed")
        print("    to be the operative mechanism")
        rows.append({"moderator": "joint: built volume per capita",
                     "column": "vol_per_cap|joint", "n_countries": len(ok),
                     "slope_per_sd": b[1], "se": float(np.sqrt(V[1, 1])),
                     "t": b[1] / np.sqrt(V[1, 1]), "r2": r2})
        rows.append({"moderator": "joint: AC penetration",
                     "column": "ac|joint", "n_countries": len(ok),
                     "slope_per_sd": b[2], "se": float(np.sqrt(V[2, 2])),
                     "t": b[2] / np.sqrt(V[2, 2]), "r2": r2})

    # ---- is one country carrying it? -------------------------------------
    print("\n[5c] Leave-one-country-out on the volume-per-capita slope")
    s = d[d.vol_per_cap.notna() & d["heat"].notna()]
    jack = []
    for c in s.country:
        b, V, _ = wls(s[s.country != c], ["vol_per_cap"])
        jack.append({"dropped": c, "slope": b[1],
                     "t": b[1] / np.sqrt(V[1, 1])})
    j = pd.DataFrame(jack)
    print(f"    slope range {j.slope.min():.2f} to {j.slope.max():.2f}; "
          f"t range {j.t.min():.2f} to {j.t.max():.2f}")
    flip = j[j.t < 1.96]
    print(f"    countries whose removal drops |t| below 1.96: "
          f"{len(flip)}/{len(j)}"
          + (f"  ({', '.join(flip.dropped)})" if len(flip) else ""))

    # ---- placebo: does the same moderator move Aviation? -----------------
    print("\n[5d] Placebo: the same moderator on Aviation cooling slopes")
    av = TAB_DIR.parent / "tables" / "T6_thermal_asymmetry_Aviation_tierAB.csv"
    av = RESULTS_ROOT / "R5" / "tables" / "T6_thermal_asymmetry_Aviation_tierAB.csv"
    if av.exists():
        a = pd.read_csv(av)
        a = a[a.heat_ok.fillna(False) & a["heat"].notna()].merge(
            agg, on="country", how="left")
        a = a[a.vol_per_cap.notna()]
        if len(a) >= 6:
            b, V, r2 = wls(a, ["vol_per_cap"])
            t = b[1] / np.sqrt(V[1, 1])
            print(f"    Aviation (n = {len(a)}): slope {b[1]:+.2f}, t = {t:+.2f}")
            print("    a null here means the moderator is not simply tracking a")
            print("    general national factor that moves every sector")
            rows.append({"moderator": "PLACEBO Aviation: volume per capita",
                         "column": "vol_per_cap|aviation",
                         "n_countries": len(a), "slope_per_sd": b[1],
                         "se": float(np.sqrt(V[1, 1])), "t": t, "r2": r2})
    else:
        print("    Aviation slopes not found; run 22_r5_asymmetry.py first")

    return pd.DataFrame(rows) if rows else None


def make_figure(p, rdf, cdf, mod, tier: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))

    # (a) the schemes barely differ
    ax = axes[0]
    s = cdf.set_index("scheme").reindex([LABEL[w] for w in WEIGHTS])
    yy = np.arange(len(s))[::-1]
    ax.barh(yy, s.corr_with_pop, color="#34495e", height=0.55)
    ax.set_xlim(0.9, 1.001)
    ax.axvline(0.99, color="#c0392b", lw=0.9, ls=":")
    ax.set_yticks(yy)
    ax.set_yticklabels(s.index, fontsize=6.5)
    ax.set_xlabel("correlation with the population-weighted series")
    ax.set_title("a  The schemes produce near-identical exposure")

    # (b) the race
    ax = axes[1]
    s = rdf.set_index("scheme").reindex([LABEL[w] for w in WEIGHTS])
    yy = np.arange(len(s))[::-1]
    ax.scatter(s.oos_r2_loco, yy + 0.14, s=26, color="#2980b9", label="LOCO")
    ax.scatter(s.oos_r2_loyo, yy - 0.14, s=26, color="#e67e22", marker="s",
               label="LOYO")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels(s.index, fontsize=6.5)
    ax.set_xlabel("out-of-sample R²")
    ax.set_title("b  No scheme separates from the others")
    ax.legend(frameon=False, fontsize=6.5)

    # (c) moderators, where the mechanism question actually lives
    ax = axes[2]
    if mod is not None and len(mod):
        yy = np.arange(len(mod))[::-1]
        ax.errorbar(mod.slope_per_sd, yy, xerr=1.96 * mod.se, fmt="o", ms=5,
                    lw=1.1, color="#7d3c98", ecolor="#b9a0c9", capsize=0)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels(mod.moderator, fontsize=6.5)
        ax.set_xlabel("Δ response per SD of moderator (%/100 K·h)")
        ax.set_title("c  Country-level moderators, 95% CI")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Figure 7  Built-environment weighting and moderation "
                 f"(Power, Tier {tier})", y=1.03, fontsize=9.5)
    fig.tight_layout()
    name = "Fig7_built_environment"
    for ext in SAVE_FORMATS:
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)
    print(f"\n    R6/figures/{name}  [{', '.join(SAVE_FORMATS)}]  {SAVE_DPI} dpi")


if __name__ == "__main__":
    main()
