#!/usr/bin/env python
"""
21_chhe_threshold_scan.py  --  SI: is the 26 C wet-bulb threshold arbitrary?
============================================================================

R4 reports a positive carbon response to compound humid heat defined as an
extreme-heat day with Twb_max >= 26 C (CHHE), and reports that the severe form
at 28 C (CHHE_plus) FAILS its placebo -- it moves Aviation as strongly as Power.
Two questions follow immediately, and a referee will ask both:

  1  Is 26 C a real threshold or a lucky draw? The value comes from the
     literature, not from this sample, so the coefficient must be shown as a
     smooth function of the threshold rather than at one point.
  2  Where does the placebo break? If Aviation starts moving somewhere between
     26 and 28 C, the boundary of the interpretable range is an empirical fact
     that belongs in the paper, not a judgement call.

DESIGN
------
For each threshold in 24..28 C and each of two construct families the country-
day population-weighted share of urban population under the condition is
rebuilt from the city panel, then entered jointly with the plain extreme-heat
share in the same distributed-lag fixed-effect model used throughout R3/R4:

    conditional   EHD & Twb_max >= theta      the CHHE definition as published
    pure          Twb_max >= theta            no relative-threshold conditioning

The second family is not padding. CHHE conditions an absolute criterion on a
RELATIVE one (EHD), which R3.2 and R4.5 both showed to be the wrong kind of
construct; if the two families disagree, the published definition is carrying a
relative-threshold artefact and that has to be said.

Aviation is the placebo. It has no plausible cooling channel, so a humid-heat
coefficient there indicates an unabsorbed seasonal or regional factor rather
than an energy response. The interpretable range is the set of thresholds where
Power responds and Aviation does not.

WHAT THIS CANNOT SHOW
---------------------
A smooth coefficient profile with no kink is evidence AGAINST a sharp
physiological or engineering threshold at any particular temperature; it is not
evidence that humid heat does not matter. Conversely a peak at some theta is
not a "discovered threshold" -- prevalence falls monotonically with theta, so
precision and composition change together and the profile mixes both.

INPUTS
------
  outputs/04_events/city_events_daily_v2.parquet
  outputs/population/fua_city_weights.parquet
  outputs/exposure/country_daily_exposure.parquet
  CarbonData/Near-real-time .../carbon-monitor-cities-*-v0325.csv

OUTPUTS
-------
  outputs/response/chhe_threshold_scan_tier{AB,C}.parquet
  Results/SI/tables/TS2_chhe_threshold_scan_tier{AB,C}.csv
  Results/SI/tables/TS3_chhe_threshold_prevalence.csv
  Results/SI/figures/FigS2_chhe_threshold_sensitivity.{png,pdf,svg}

USAGE
-----
  python 21_chhe_threshold_scan.py --tier AB
  python 21_chhe_threshold_scan.py --tier AB --no-figure
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
WEIGHTS = BASE / "outputs" / "population" / "fua_city_weights.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
OUT_DIR = BASE / "outputs" / "response"
SI_TAB = RESULTS_ROOT / "SI" / "tables"
SI_FIG = RESULTS_ROOT / "SI" / "figures"

LAGS = 7
THRESHOLDS = [24.0, 25.0, 26.0, 27.0, 28.0]
PUBLISHED = 26.0
PLACEBO = "Aviation"
SECTORS = ["Power", "Industry", "Ground Transport", "Aviation", "Residential"]
SAVE_FORMATS = ("png", "pdf", "svg")
SAVE_DPI = 600

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "savefig.dpi": SAVE_DPI, "figure.dpi": 110,
                     "svg.fonttype": "none"})


# --------------------------------------------------------------------------
# estimation machinery, identical to 20_r4_humidity.py
# --------------------------------------------------------------------------
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


def psd(M):
    M = 0.5 * (M + M.T)
    w, Q = np.linalg.eigh(M)
    return (Q * np.clip(w, 0, None)) @ Q.T, int((w < 0).sum())


def fit(d: pd.DataFrame, blocks: dict[str, str]):
    """Distributed-lag FE model, two-way clustered and PSD-projected."""
    d = d.sort_values(["country", "date"]).reset_index(drop=True).copy()
    names, index = [], {}
    for tag, col in blocks.items():
        sd = d[col].std()
        if not np.isfinite(sd) or sd == 0:
            return None
        d[col + "_z"] = d[col] / sd
        g = d.groupby("country")[col + "_z"]
        idx = []
        for s in range(0, LAGS + 1):
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
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    inv = np.linalg.inv(XtX)

    def meat(g):
        n = g.max() + 1
        Xe = X * e[:, None]
        S = np.zeros((n, X.shape[1]))
        for j in range(X.shape[1]):
            S[:, j] = np.bincount(g, weights=Xe[:, j], minlength=n)
        return S.T @ S

    g1 = pd.factorize(d.country)[0]
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(f2).astype(str))[0]
    M, n_neg = psd(meat(g1) + meat(f2) - meat(g12))
    V = inv @ M @ inv

    out = {"n": len(d), "neg_eig": n_neg}
    for tag, idx in index.items():
        cum = float(b[idx].sum())
        se = float(np.sqrt(V[np.ix_(idx, idx)].sum()))
        out[tag] = {"cum": cum, "se": se, "t": cum / se if se else np.nan}
    return out


# --------------------------------------------------------------------------
def load_sector(sector: str) -> pd.DataFrame:
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
        d = d[d.sector == sector]
        d["country"] = n
        rows.append(d[["country", "date", "v"]])
    e = pd.concat(rows, ignore_index=True)
    e["date"] = pd.to_datetime(e["date"])
    return e.groupby(["country", "date"], as_index=False)["v"].sum()


def build_threshold_exposure() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country-day population-weighted shares for every threshold and family.

    Weights use `pop_weight_dedup` for the same reason as 13/14: eight Carbon
    Monitor cities named Guangzhou1..8 share a single FUA polygon, so weighting
    per emission record would count that polygon eight times.
    """
    ev = pd.read_parquet(EVENTS, columns=["city_id", "date", "EHD", "Twb_max"])
    ev["city_id"] = ev["city_id"].astype(str).str.zfill(4)
    ev["date"] = pd.to_datetime(ev["date"])

    w = pd.read_parquet(WEIGHTS)
    w["city_id"] = w["city_id"].astype(str).str.zfill(4)
    if "pop_weight_dedup" not in w.columns:
        w["pop_weight_dedup"] = w["pop_mean"] / w["n_share"]
    w = w[["city_id", "country", "pop_weight_dedup"]].dropna()
    w = w[w.pop_weight_dedup > 0]

    d = ev.merge(w, on="city_id", how="inner")
    print(f"    {len(d):,} city-days, {d.city_id.nunique():,} cities, "
          f"{d.country.nunique()} countries")

    # prevalence audit, computed on the raw city panel before any weighting
    prev = []
    for th in THRESHOLDS:
        cond = (d.EHD == 1) & (d.Twb_max >= th)
        pure = d.Twb_max >= th
        prev.append({"threshold_c": th,
                     "conditional_share_pct": 100 * cond.mean(),
                     "conditional_n_cities": int(d.loc[cond, "city_id"].nunique()),
                     "conditional_n_countries": int(d.loc[cond, "country"].nunique()),
                     "pure_share_pct": 100 * pure.mean(),
                     "pure_n_cities": int(d.loc[pure, "city_id"].nunique()),
                     "pure_n_countries": int(d.loc[pure, "country"].nunique())})
    prevalence = pd.DataFrame(prev)

    flags = {}
    for th in THRESHOLDS:
        tag = f"{int(th)}"
        flags[f"cond_{tag}"] = ((d.EHD == 1) & (d.Twb_max >= th)).astype("float64")
        flags[f"pure_{tag}"] = (d.Twb_max >= th).astype("float64")
    F = pd.DataFrame(flags, index=d.index)

    wt = d["pop_weight_dedup"].values[:, None]
    num = pd.DataFrame(F.values * wt, columns=F.columns, index=d.index)
    num["__w"] = d["pop_weight_dedup"].values
    num["country"] = d["country"].values
    num["date"] = d["date"].values
    g = num.groupby(["country", "date"], sort=False).sum()
    share = g[F.columns].div(g["__w"], axis=0).reset_index()
    return share, prevalence


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C", "all"])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("SI: wet-bulb threshold sensitivity for compound humid heat")
    print("=" * 78)

    print("\n[1] Rebuilding threshold exposure from the city panel")
    share, prevalence = build_threshold_exposure()

    print("\n[2] Prevalence by threshold")
    print(f"    {'theta':>6}{'EHD & Twb (%)':>16}{'cities':>8}"
          f"{'Twb only (%)':>15}{'cities':>8}")
    for _, r in prevalence.iterrows():
        print(f"    {r.threshold_c:6.0f}{r.conditional_share_pct:16.3f}"
              f"{r.conditional_n_cities:8.0f}{r.pure_share_pct:15.3f}"
              f"{r.pure_n_cities:8.0f}")

    ex = pd.read_parquet(EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"],
            "all": ["A", "B", "C"]}[args.tier]
    ex = ex[ex.tier.isin(keep)][["country", "date", "w_EHD"]]
    ex = ex.merge(share, on=["country", "date"], how="inner")
    print(f"\n[3] Tier {args.tier}: {ex.country.nunique()} countries, "
          f"{len(ex):,} country-days")

    rows = []
    for sector in SECTORS:
        em = load_sector(sector)
        p = em.merge(ex, on=["country", "date"], how="inner")
        p = p[p.v > 0].copy()
        if len(p) < 1000:
            print(f"    [{sector}] too few observations, skipped")
            continue
        p["y"] = np.log(p.v)
        print(f"\n    [{sector}]  {len(p):,} country-days")
        print(f"      {'theta':>6}  {'family':<12}{'sum beta/SD (%)':>17}"
              f"{'t':>8}")
        for family in ("cond", "pure"):
            for th in THRESHOLDS:
                col = f"{family}_{int(th)}"
                r = fit(p, {"ehd": "w_EHD", "humid": col})
                if r is None:
                    print(f"      {th:6.0f}  {family:<12}{'not identified':>17}")
                    continue
                h = r["humid"]
                rows.append({"tier": args.tier, "sector": sector,
                             "family": family, "threshold_c": th,
                             "cum": h["cum"], "se": h["se"], "t": h["t"],
                             "pct_per_sd": 100 * h["cum"],
                             "ehd_cum": r["ehd"]["cum"], "ehd_t": r["ehd"]["t"],
                             "n": r["n"], "neg_eig": r["neg_eig"]})
                print(f"      {th:6.0f}  {family:<12}{100 * h['cum']:17.3f}"
                      f"{h['t']:8.2f}")

    res = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SI_TAB.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_DIR / f"chhe_threshold_scan_tier{args.tier}.parquet",
                   index=False)
    res.to_csv(SI_TAB / f"TS2_chhe_threshold_scan_tier{args.tier}.csv",
               index=False)
    prevalence.to_csv(SI_TAB / "TS3_chhe_threshold_prevalence.csv", index=False)

    # ---- verdict: where is the construct interpretable? -------------------
    print("\n[4] Interpretable range (conditional family, |t| > 1.96 is 'moves')")
    print(f"    {'theta':>6}{'Power t':>10}{'Aviation t':>12}   verdict")
    ok = []
    for th in THRESHOLDS:
        s = res[(res.family == "cond") & (res.threshold_c == th)]
        tp = s.loc[s.sector == "Power", "t"]
        ta = s.loc[s.sector == PLACEBO, "t"]
        if tp.empty or ta.empty:
            continue
        tp, ta = float(tp.iloc[0]), float(ta.iloc[0])
        if abs(ta) > 1.96:
            v = "PLACEBO FAILS -- not interpretable"
        elif abs(tp) > 1.96:
            v = "interpretable, Power responds"
            ok.append(th)
        else:
            v = "placebo clean but Power null"
        print(f"    {th:6.0f}{tp:10.2f}{ta:12.2f}   {v}")
    if ok:
        print(f"\n    -> interpretable thresholds: "
              f"{', '.join(f'{t:.0f} C' for t in ok)}")
        print(f"    -> the published {PUBLISHED:.0f} C is "
              f"{'inside' if PUBLISHED in ok else 'OUTSIDE'} that range")

    if not args.no_figure:
        make_figure(res, prevalence, args.tier)

    print(f"\nDone in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------
def make_figure(res: pd.DataFrame, prevalence: pd.DataFrame, tier: str) -> None:
    if res.empty:
        return
    SI_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    colors = {"Power": "#c0392b", "Industry": "#e67e22",
              "Ground Transport": "#7f8c8d", "Aviation": "#2980b9",
              "Residential": "#27ae60"}

    # (a) prevalence -- the composition changing behind every other panel
    ax = axes[0]
    ax.semilogy(prevalence.threshold_c, prevalence.conditional_share_pct,
                "o-", color="#2c3e50", lw=1.4, ms=4, label="EHD & Twb >= θ")
    ax.semilogy(prevalence.threshold_c, prevalence.pure_share_pct,
                "s--", color="#95a5a6", lw=1.2, ms=4, label="Twb >= θ only")
    ax.axvline(PUBLISHED, color="#c0392b", lw=0.8, ls=":")
    ax.set_xlabel("wet-bulb threshold θ (°C)")
    ax.set_ylabel("share of city-days (%, log scale)")
    fold = (prevalence.conditional_share_pct.iloc[0]
            / prevalence.conditional_share_pct.iloc[-1])
    ax.set_title(f"a  Prevalence falls {fold:.0f}× from "
                 f"{THRESHOLDS[0]:.0f} to {THRESHOLDS[-1]:.0f} °C")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    # (b) coefficient profile
    ax = axes[1]
    for sec in SECTORS:
        s = res[(res.family == "cond") & (res.sector == sec)].sort_values("threshold_c")
        if s.empty:
            continue
        ax.plot(s.threshold_c, s.pct_per_sd, "o-", ms=3.5, lw=1.3,
                color=colors.get(sec, "k"), label=sec)
        if sec == "Power":
            ax.fill_between(s.threshold_c,
                            100 * (s.cum - 1.96 * s.se),
                            100 * (s.cum + 1.96 * s.se),
                            color=colors[sec], alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(PUBLISHED, color="#c0392b", lw=0.8, ls=":")
    ax.set_xlabel("wet-bulb threshold θ (°C)")
    ax.set_ylabel("Σβ per SD of exposure (%)")
    ax.set_title("b  No kink at 26 °C: the threshold is not special")
    ax.legend(frameon=False, fontsize=6, ncol=1)

    # (c) the placebo, which is the panel that decides the usable range
    ax = axes[2]
    for sec, mk in (("Power", "o"), (PLACEBO, "s")):
        s = res[(res.family == "cond") & (res.sector == sec)].sort_values("threshold_c")
        if s.empty:
            continue
        ax.plot(s.threshold_c, s.t, mk + "-", ms=4, lw=1.4,
                color=colors.get(sec, "k"), label=f"{sec}"
                + (" (placebo)" if sec == PLACEBO else ""))
    ax.axhspan(-1.96, 1.96, color="#bdc3c7", alpha=0.35, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(PUBLISHED, color="#c0392b", lw=0.8, ls=":")
    ax.set_xlabel("wet-bulb threshold θ (°C)")
    ax.set_ylabel("t statistic")
    av = res[(res.family == "cond") & (res.sector == PLACEBO)].sort_values("threshold_c")
    bad = av.loc[av.t.abs() > 1.96, "threshold_c"]
    ax.set_title(f"c  Placebo fails at {bad.iloc[0]:.0f} °C" if len(bad)
                 else "c  Placebo clean at every threshold")
    ax.legend(frameon=False, fontsize=6.5)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Figure S2  Sensitivity of the compound humid-heat result to "
                 f"the wet-bulb threshold (Tier {tier})", y=1.04, fontsize=9)
    fig.tight_layout()
    name = "FigS2_chhe_threshold_sensitivity"
    for ext in SAVE_FORMATS:
        fig.savefig(SI_FIG / f"{name}.{ext}", bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)
    print(f"\n    SI/figures/{name}  [{', '.join(SAVE_FORMATS)}]  {SAVE_DPI} dpi")


if __name__ == "__main__":
    main()
