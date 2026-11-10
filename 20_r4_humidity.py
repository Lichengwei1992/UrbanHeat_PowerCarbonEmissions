#!/usr/bin/env python
"""
20_r4_humidity.py  --  R4: humidity amplification and day-night structure
==========================================================================

The review was explicit that humidity amplification cannot be claimed by
comparing the fit of a wet-bulb model against a dry-bulb model: the two are
highly collinear and the better fit could come from either dimension. The claim
is only supported if the humidity dimension retains an incremental effect
AFTER conditioning on dry-bulb cooling demand. Five tests are run to that
standard, all on the same fixed-effect structure as R3.

  T1  same-load interaction   ln E ~ SCDH24 + WBD + SCDH24 x 1[low WBD]
                              beta3 > 0 means that at an identical sensible
                              cooling load, humid conditions cost more carbon
  T1b orthogonalised humidity SCDH24 plus wet-bulb depression residualised on
                              SCDH24 within country -- the cleanest reading of
                              "extra humidity at the same cooling load"
  T1c peak-load subsample     T1b restricted to the upper tercile of cooling
                              demand, to test whether the humidity term is
                              concentrated where peak loads actually occur
  T2  humid vs dry events     humid and dry extreme-heat population shares
                              enter jointly; H0: sum(beta_humid) = sum(beta_dry).
                              A DIAGNOSTIC of the v2 typing rather than a test
                              of humidity: see the construct check below
  T3  day vs night            SCDH24 split into daytime and night-time
                              components (local time) entered together
  T4  compound day-night      extreme-heat share plus the compound day-night
                              share; the second coefficient is the extra cost
                              of no night-time recovery
  T5  latent vs sensible      sensible cooling degree-hours plus latent
                              degree-hours in humidity ratio

A CONSTRUCT FAILURE THAT HAD TO BE DIAGNOSED
--------------------------------------------
The v2 humid/dry typing splits extreme-heat days at the within-city terciles of
wet-bulb depression. Wet-bulb depression grows with temperature at fixed
relative humidity, so the split does not isolate humidity: in this sample the
"humid" class averages Tmax 19.3 C and the "dry" class 30.5 C, an 11.3 K gap.
The typing therefore contrasts cool damp days against hot dry ones, and its
coefficients cannot be read as a humidity effect. The script reports the raw
contrast for transparency and then re-runs the test on a humidity dimension
orthogonalised to cooling load.

WORDING CONSTRAINT
------------------
Without load or metered cooling-electricity data none of these tests can
identify a latent-load or coefficient-of-performance mechanism. A positive
result is reported as *consistent with cooling-related amplification*, and the
script prints that phrasing alongside the numbers so it cannot drift.

INPUTS
------
  CarbonData/Near-real-time .../carbon-monitor-cities-*-v0325.csv
  outputs/exposure/country_daily_exposure.parquet

OUTPUTS
-------
  outputs/response/r4_humidity_tests.parquet
  Results/R4/tables/T5_humidity_tests_tier{AB,C}.csv

USAGE
-----
  python 20_r4_humidity.py --tier AB
  python 20_r4_humidity.py --tier C
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
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
OUT_DIR = BASE / "outputs" / "response"
TAB = RESULTS_ROOT / "R4" / "tables"

LAGS = 7
SECTORS = ["Power", "Industry", "Ground Transport",
           "Aviation", "Residential"]


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


def fit(d: pd.DataFrame, blocks: dict[str, str], extra: list[str] | None = None):
    """Distributed-lag FE model with several exposure blocks entered jointly.

    `blocks` maps a short tag to a column; each gets its own 0..LAGS lag set.
    `extra` holds contemporaneous-only controls (e.g. an interaction term).
    Returns cumulative sums per block with a two-way clustered, PSD-projected
    variance.
    """
    d = d.sort_values(["country", "date"]).reset_index(drop=True).copy()
    names, index = [], {}
    for tag, col in blocks.items():
        sd = d[col].std()
        d[col + "_z"] = d[col] / (sd if sd > 0 else 1)
        g = d.groupby("country")[col + "_z"]
        idx = []
        for s in range(0, LAGS + 1):
            nm = f"{tag}_L{s}"
            d[nm] = g.shift(s)
            names.append(nm)
            idx.append(len(names) - 1)
        index[tag] = np.array(idx)
    for c in (extra or []):
        sd = d[c].std()
        d[c + "_z"] = d[c] / (sd if sd > 0 else 1)
        names.append(c + "_z")
        index[c] = np.array([len(names) - 1])

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

    out = {"n": len(d), "r2": 1 - e.var() / y.var(), "neg_eig": n_neg}
    for tag, idx in index.items():
        cum = float(b[idx].sum())
        se = float(np.sqrt(V[np.ix_(idx, idx)].sum()))
        out[tag] = {"cum": cum, "se": se, "t": cum / se if se else np.nan}
    out["_b"], out["_V"], out["_index"] = b, V, index
    return out


def contrast_test(res, tag_a: str, tag_b: str) -> dict:
    """Wald test of H0: sum(beta_a) = sum(beta_b)."""
    ia, ib = res["_index"][tag_a], res["_index"][tag_b]
    c = np.zeros(len(res["_b"]))
    c[ia] = 1.0
    c[ib] = -1.0
    diff = float(c @ res["_b"])
    var = float(c @ res["_V"] @ c)
    se = np.sqrt(max(var, 0))
    z = diff / se if se > 0 else np.nan
    return {"diff": diff, "se": se, "z": z,
            "p": float(2 * (1 - stats.norm.cdf(abs(z)))) if se > 0 else np.nan}


# --------------------------------------------------------------------------
def run_sector(p: pd.DataFrame, sector: str, tier: str) -> list[dict]:
    """All humidity tests for one sector; returns tidy rows."""
    rows = []

    def rec(test, tag, v):
        rows.append({"tier": tier, "sector": sector, "test": test,
                     "term": tag, **v})

    # T1 interaction on the sensible load
    r = fit(p, {"scdh": "w_SCDH24", "wbd": "w_WBD_at_tmax"},
            extra=["SCDH_x_humid"])
    if r:
        for tag in ("scdh", "wbd", "SCDH_x_humid"):
            rec("T1_same_load", tag, r[tag] | {"n": r["n"]})

    # T1b humidity orthogonalised to cooling load
    r = fit(p, {"scdh": "w_SCDH24", "humid": "humid_orth"})
    if r:
        for tag in ("scdh", "humid"):
            rec("T1b_orth_humidity", tag, r[tag] | {"n": r["n"]})

    # T1c the same, restricted to peak cooling demand
    hi = p[p.w_SCDH24 > p.w_SCDH24.quantile(2 / 3)]
    r = fit(hi, {"scdh": "w_SCDH24", "humid": "humid_orth"})
    if r:
        for tag in ("scdh", "humid"):
            rec("T1c_peak_load", tag, r[tag] | {"n": r["n"]})

    # T2 the v2 relative humid/dry typing (diagnostic only)
    r = fit(p, {"humid": "w_HEH", "dry": "w_DEH"})
    if r:
        for tag in ("humid", "dry"):
            rec("T2_humid_vs_dry_relative", tag, r[tag] | {"n": r["n"]})
        ct = contrast_test(r, "humid", "dry")
        rec("T2_humid_vs_dry_relative", "humid_minus_dry",
            {"cum": ct["diff"], "se": ct["se"], "t": ct["z"], "p": ct["p"],
             "n": r["n"]})

    # T2b ABSOLUTE humid heat: CHHE and its severe form against plain extreme heat
    for tag_hi, col_hi in (("chhe", "w_CHHE"), ("chhe_plus", "w_CHHE_plus")):
        if col_hi not in p.columns or p[col_hi].std() == 0:
            continue
        r = fit(p, {"ehd": "w_EHD", tag_hi: col_hi})
        if r:
            for tag in ("ehd", tag_hi):
                rec(f"T2b_absolute_{tag_hi}", tag, r[tag] | {"n": r["n"]})

    # T3 day versus night cooling degree-hours
    r = fit(p, {"day": "w_SCDH24_day", "night": "w_SCDH24_night"})
    if r:
        for tag in ("day", "night"):
            rec("T3_day_night", tag, r[tag] | {"n": r["n"]})
        ct = contrast_test(r, "night", "day")
        rec("T3_day_night", "night_minus_day",
            {"cum": ct["diff"], "se": ct["se"], "t": ct["z"], "p": ct["p"],
             "n": r["n"]})

    # T3b wet-bulb cooling degree-hours, day versus night
    if "w_CDH_Twb26_day" in p.columns:
        r = fit(p, {"day": "w_CDH_Twb26_day", "night": "w_CDH_Twb26_night"})
        if r:
            for tag in ("day", "night"):
                rec("T3b_twb_day_night", tag, r[tag] | {"n": r["n"]})

    # T4 extra cost of no night-time recovery
    r = fit(p, {"ehd": "w_EHD", "cdnh": "w_CDNH"})
    if r:
        for tag in ("ehd", "cdnh"):
            rec("T4_compound_night", tag, r[tag] | {"n": r["n"]})

    # T5 enthalpy and latent against sensible
    r = fit(p, {"sensible": "w_SCDH24", "enthalpy": "w_EDH_h"})
    if r:
        for tag in ("sensible", "enthalpy"):
            rec("T5b_enthalpy", tag, r[tag] | {"n": r["n"]})
    r = fit(p, {"sensible": "w_SCDH24", "latent": "w_LDH_w"})
    if r:
        for tag in ("sensible", "latent"):
            rec("T5_latent_sensible", tag, r[tag] | {"n": r["n"]})
    return rows


def prepare(p: pd.DataFrame) -> pd.DataFrame:
    """Derived exposure columns shared by every sector."""
    p = p.copy()
    p["w_SCDH24_day"] = (p["w_SCDH24"] - p["w_SCDH24_night"]).clip(lower=0)
    p["w_CDH_Twb26_day"] = (p["w_CDH_Twb26"] - p["w_CDH_Twb26_night"]).clip(lower=0)
    med = p.groupby("country")["w_WBD_at_tmax"].transform("median")
    p["humid_flag"] = (p["w_WBD_at_tmax"] < med).astype(float)
    p["SCDH_x_humid"] = p["w_SCDH24"] * p["humid_flag"]
    # humidity orthogonal to cooling load, within country
    p = p.reset_index(drop=True)
    p["wbd_orth"] = np.nan
    for c, g in p.groupby("country"):
        X = np.column_stack([np.ones(len(g)), g["w_SCDH24"].values])
        b = np.linalg.lstsq(X, g["w_WBD_at_tmax"].values, rcond=None)[0]
        p.loc[g.index, "wbd_orth"] = g["w_WBD_at_tmax"].values - X @ b
    p["humid_orth"] = -p["wbd_orth"]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C", "all"])
    ap.add_argument("--sectors", nargs="+", default=SECTORS)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 82)
    print(f"R4  humidity amplification and day-night structure  |  Tier {args.tier}")
    print("=" * 82)

    ex = pd.read_parquet(EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"],
            "all": ["A", "B", "C"]}[args.tier]
    ex = ex[ex.tier.isin(keep)]

    # construct diagnosis, reported once
    print("\n[construct] Does the v2 relative humid/dry typing isolate humidity?")
    ev = pd.read_parquet(
        BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet",
        columns=["HEH", "DEH", "CHHE", "CHHE_plus", "Twb_max", "WBD_at_tmax",
                 "tmax_c"])
    h, dd = ev[ev.HEH == 1], ev[ev.DEH == 1]
    ch = ev[ev.CHHE == 1]
    print(f"    HEH  (relative humid): Tmax {h.tmax_c.mean():5.2f} C, "
          f"Twb {h.Twb_max.mean():5.2f} C, WBD {h.WBD_at_tmax.mean():5.2f} K")
    print(f"    DEH  (relative dry)  : Tmax {dd.tmax_c.mean():5.2f} C, "
          f"Twb {dd.Twb_max.mean():5.2f} C, WBD {dd.WBD_at_tmax.mean():5.2f} K")
    print(f"    CHHE (absolute humid): Tmax {ch.tmax_c.mean():5.2f} C, "
          f"Twb {ch.Twb_max.mean():5.2f} C, WBD {ch.WBD_at_tmax.mean():5.2f} K")
    print(f"    -> the relative 'humid' class is "
          f"{dd.tmax_c.mean() - h.tmax_c.mean():.2f} K COOLER than the 'dry' class;")
    print(f"       the absolute CHHE class is "
          f"{ch.tmax_c.mean() - dd.tmax_c.mean():+.2f} K relative to it.")

    all_rows = []
    for sector in args.sectors:
        em = load_sector(sector)
        p = em.merge(ex, on=["country", "date"], how="inner")
        p = p[p.v > 0].copy()
        if len(p) < 1000:
            print(f"\n[{sector}] too few observations, skipped")
            continue
        p["y"] = np.log(p.v)
        p = prepare(p)
        rows = run_sector(p, sector, args.tier)
        all_rows += rows
        r = pd.DataFrame(rows)

        def get(test, term):
            m = r[(r.test == test) & (r.term == term)]
            return (m.cum.iloc[0], m.t.iloc[0]) if len(m) else (np.nan, np.nan)

        print(f"\n[{sector}]  n={p.shape[0]:,}  countries={p.country.nunique()}")
        print(f"    {'test':30s}{'Σβ (%)':>10s}{'t':>8s}")
        for test, term, lab in (
                ("T1b_orth_humidity", "humid", "humidity at same load"),
                ("T1c_peak_load", "humid", "  ... peak-load days only"),
                ("T2b_absolute_chhe", "chhe", "absolute humid heat (CHHE)"),
                ("T2b_absolute_chhe_plus", "chhe_plus", "severe humid heat (CHHE+)"),
                ("T2_humid_vs_dry_relative", "humid_minus_dry", "relative humid - dry"),
                ("T3_day_night", "day", "daytime cooling"),
                ("T3_day_night", "night", "night-time cooling"),
                ("T4_compound_night", "cdnh", "compound day-night heat"),
                ("T5b_enthalpy", "enthalpy", "enthalpy (total heat)"),
                ("T5_latent_sensible", "latent", "latent degree-hours")):
            c, t = get(test, term)
            star = "" if not np.isfinite(t) else (
                "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "")
            print(f"    {lab:30s}{100*c:10.3f}{t:8.2f}{star}")

    res = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    f = OUT_DIR / f"r4_humidity_tests_tier{args.tier}.parquet"
    res.to_parquet(f, index=False)
    # Table 6 in the published sequence: the R2 benchmark table moved to the SI
    # (Table S1), so everything after it shifts down by one. See RENUMBER in
    # 19_results_figures.py, which is the single source of truth for numbering.
    res.to_csv(TAB / f"T5_humidity_tests_tier{args.tier}.csv", index=False)
    print(f"\nSaved {f.name}  ({len(res)} rows, {res.sector.nunique()} sectors)")

    print("\nREPORTING CONSTRAINT: a positive humidity term is described as")
    print("  'consistent with cooling-related amplification'. Without load or")
    print("  metered cooling-electricity data these tests cannot identify a")
    print("  latent-load or COP mechanism.")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
