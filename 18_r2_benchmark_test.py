#!/usr/bin/env python
"""
18_r2_benchmark_test.py  --  R2: does city-resolved exposure beat national?
============================================================================

The review's central objection to R2 was that comparing city-weighted exposure
only against area-weighted national temperature sets up a straw man: for a
large, sparsely populated country the two differ almost by construction. The
serious opponent is POPULATION-WEIGHTED national temperature, which is an
established method in its own right. The contribution therefore has to be the
pairing of city-resolved exposure with high-frequency sectoral carbon, and
that claim has to be tested rather than asserted.

FOUR BENCHMARKS, ONE FOOTING
----------------------------
  1  nat_area   area-weighted national        naive
  2  nat_pop    population-weighted national  established method
  3  fua_equal  FUA equal-weighted            city sample, unweighted
  4  fua_pop    FUA population-weighted       this study

A FAIRNESS PROBLEM THAT MUST NOT BE SWEPT UP
--------------------------------------------
The city-side SCDH24 is integrated from HOURLY temperature; the national
benchmark can only be built from daily Tmax (no hourly global field was
downloaded). Comparing them directly would confound two things at once --
"city vs national" and "hourly vs daily". So the comparison is run twice:

  matched   all four built from daily Tmax alone (isolates city vs national)
  native    each measure in its best available form (the practical question)

Only the matched comparison can support a claim about spatial resolution.

OUT-OF-SAMPLE PROTOCOL
----------------------
Leave-one-country-out and leave-one-year-out, never random splits of
country-days: daily emissions are strongly autocorrelated, so a random split
leaks information across the fold boundary and inflates every model equally.

INPUTS
------
  outputs/exposure/country_daily_exposure.parquet             from 14
  outputs/exposure/national_benchmark_exposure.parquet        from 17
  outputs/04_events/city_events_daily_v2.parquet              for FUA Tmax
  outputs/population/fua_city_weights.parquet

OUTPUT
------
  outputs/response/r2_benchmark_comparison.parquet
  outputs/response/tables/r2_benchmarks.csv
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
NATBENCH = BASE / "outputs" / "exposure" / "national_benchmark_exposure.parquet"
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
WEIGHTS = BASE / "outputs" / "population" / "fua_city_weights.parquet"
OUT_DIR = BASE / "outputs" / "response"
TABLE_DIR = OUT_DIR / "tables"

BASE_C = 24.0
LAGS = 7


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


def build_fua_tmax() -> pd.DataFrame:
    """FUA equal- and population-weighted daily Tmax and Tmax-based CDD."""
    w = pd.read_parquet(WEIGHTS)
    w["city_id"] = w["city_id"].astype(str).str.zfill(4)
    if "pop_weight_dedup" not in w.columns:
        w["pop_weight_dedup"] = w["pop_mean"] / w["n_share"]
    w = w[["city_id", "country", "pop_weight_dedup"]].dropna()

    ev = pd.read_parquet(EVENTS, columns=["city_id", "date", "event_tmax"]) \
        if "event_tmax" in pd.read_parquet(EVENTS).columns[:80] else None
    if ev is None:
        ev = pd.read_parquet(EVENTS, columns=["city_id", "date", "tmax_c"])
        ev = ev.rename(columns={"tmax_c": "event_tmax"})
    ev["city_id"] = ev["city_id"].astype(str).str.zfill(4)

    d = ev.merge(w, on="city_id", how="inner")
    d["cdd"] = np.clip(d["event_tmax"] - BASE_C, 0, None)
    g = d.groupby(["country", "date"])
    out = g.agg(fua_equal_tmax=("event_tmax", "mean"),
                fua_equal_cdd=("cdd", "mean"),
                wsum=("pop_weight_dedup", "sum")).reset_index()
    for src, dst in (("event_tmax", "fua_pop_tmax"), ("cdd", "fua_pop_cdd")):
        num = (d[src] * d["pop_weight_dedup"]).groupby(
            [d["country"], d["date"]]).sum().rename(dst).reset_index()
        out = out.merge(num, on=["country", "date"])
        out[dst] = out[dst] / out["wsum"]
    return out.drop(columns=["wsum"])


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


def oos_score(panel: pd.DataFrame, xcol: str, folds: str) -> dict:
    """Leave-one-group-out out-of-sample R2 for a distributed-lag FE model."""
    d = panel.sort_values(["country", "date"]).reset_index(drop=True).copy()
    d["x"] = d[xcol] / d[xcol].std()
    g = d.groupby("country")["x"]
    nm = []
    for s in range(0, LAGS + 1):
        k = f"L{s}"
        d[k] = g.shift(s)
        nm.append(k)
    d = d.dropna(subset=nm + ["y"]).reset_index(drop=True)

    groups = d["country"] if folds == "country" else d["date"].dt.year
    sse, sst = 0.0, 0.0
    for gv in groups.unique():
        tr, te = d[groups != gv], d[groups == gv]
        if len(te) < 50 or len(tr) < 500:
            continue
        f1 = pd.factorize(tr["country"] + "_" + tr["date"].dt.to_period("M").astype(str))[0]
        f2 = pd.factorize(tr["date"])[0]
        Z = demean(np.column_stack([tr[nm].values, tr["y"].values]), [f1, f2])
        X, yy = Z[:, :-1], Z[:, -1]
        b = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * 1e-8, X.T @ yy)
        # Score the held-out fold after removing its own FE, so only the
        # exposure coefficients transfer across folds. Under leave-one-country-
        # out the fold contains a single country, so a date fixed effect has
        # exactly one observation per level and would annihilate the data; it
        # is therefore included only when it is identified.
        h1 = pd.factorize(te["country"] + "_" + te["date"].dt.to_period("M").astype(str))[0]
        h2 = pd.factorize(te["date"])[0]
        test_facs = [h1]
        if len(te) / max(h2.max() + 1, 1) >= 2:
            test_facs.append(h2)
        Zt = demean(np.column_stack([te[nm].values, te["y"].values]), test_facs)
        Xt, yt = Zt[:, :-1], Zt[:, -1]
        sse += float(((yt - Xt @ b) ** 2).sum())
        sst += float((yt ** 2).sum())
    return {"oos_r2": 1 - sse / sst if sst > 0 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="A", choices=["A", "AB", "C", "all"])
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 76)
    print("R2: city-resolved vs national exposure, matched and native forms")
    print("=" * 76)

    pw = load_power()
    ex = pd.read_parquet(EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"], "all": ["A", "B", "C"]}[args.tier]
    ex = ex[ex.tier.isin(keep)]
    nb = pd.read_parquet(NATBENCH)
    print(f"\nTier {args.tier}: {ex.country.nunique()} countries")

    print("[1] Building FUA Tmax benchmarks")
    fua = build_fua_tmax()

    p = (pw.merge(ex[["country", "date", "w_SCDH24", "e_SCDH24"]], on=["country", "date"])
           .merge(nb, on=["country", "date"], how="left")
           .merge(fua, on=["country", "date"], how="left"))
    p = p[p.v > 0].copy()
    p["y"] = np.log(p.v)
    p["nat_area_cdd"] = p["nat_area_scdh24"] / 24.0
    p["nat_pop_cdd"] = p["nat_pop_scdh24"] / 24.0
    print(f"    panel {len(p):,} country-days, {p.country.nunique()} countries")

    sets = {
        "matched (daily Tmax CDD)": [
            ("1 nat_area", "nat_area_cdd"), ("2 nat_pop", "nat_pop_cdd"),
            ("3 fua_equal", "fua_equal_cdd"), ("4 fua_pop", "fua_pop_cdd")],
        "native (best available)": [
            ("1 nat_area", "nat_area_scdh24"), ("2 nat_pop", "nat_pop_scdh24"),
            ("3 fua_equal", "e_SCDH24"), ("4 fua_pop", "w_SCDH24")],
    }

    rows = []
    for label, items in sets.items():
        print(f"\n[2] {label}")
        print(f"    {'benchmark':14s}{'OOS R2 (LOCO)':>15s}{'OOS R2 (LOYO)':>15s}")
        for nm, col in items:
            if col not in p.columns or p[col].isna().all():
                print(f"    {nm:14s}{'missing':>15s}")
                continue
            sub = p.dropna(subset=[col])
            a = oos_score(sub, col, "country")["oos_r2"]
            b = oos_score(sub, col, "year")["oos_r2"]
            rows.append({"set": label, "benchmark": nm, "column": col,
                         "oos_r2_loco": a, "oos_r2_loyo": b, "n": len(sub)})
            print(f"    {nm:14s}{a:15.5f}{b:15.5f}")

    res = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_DIR / "r2_benchmark_comparison.parquet", index=False)
    res.to_csv(TABLE_DIR / f"r2_benchmarks_tier{args.tier}.csv", index=False)

    print("\n[3] Verdict")
    for label in sets:
        s = res[res.set == label]
        if s.empty:
            continue
        if s.oos_r2_loco.notna().sum() == 0:
            print(f"    {label}: LOCO undefined")
            continue
        best = s.loc[s.oos_r2_loco.idxmax()]
        natpop = s[s.benchmark == "2 nat_pop"]
        fuapop = s[s.benchmark == "4 fua_pop"]
        print(f"    {label}")
        print(f"      best by LOCO: {best.benchmark}")
        if len(natpop) and len(fuapop):
            gain = float(fuapop.oos_r2_loco.iloc[0] - natpop.oos_r2_loco.iloc[0])
            print(f"      fua_pop - nat_pop = {gain:+.5f}  -> "
                  f"{'city resolution adds' if gain > 0 else 'no gain from city resolution'}")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
