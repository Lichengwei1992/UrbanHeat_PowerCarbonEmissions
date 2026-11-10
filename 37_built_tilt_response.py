#!/usr/bin/env python
"""
37_built_tilt_response.py  --  does WHERE the heat falls change the response?
==============================================================================

R6 as it stands regresses twenty country-level response slopes on twenty
country-level built-environment covariates. Country size, income, climate zone
and urban form all move together across nations, so that design cannot separate
them, and it is the reason R6 reads as an association rather than a mechanism.

THE DESIGN HERE
---------------
National exposure is a weighted mean over a country's cities. Re-weighting by
built volume instead of population gives a second series of the same physical
quantity, and the gap between them moves day to day within a country as weather
systems sit over different parts of the urban system:

    X_ct = sum_i pop_i x_it / sum_i pop_i          the published exposure
    D_ct = sum_i bv_i  x_it / sum_i bv_i  -  X_ct  today's tilt toward built-up

D > 0 means today's thermal load sits disproportionately on the country's high
built-volume cities. Entering D beside X in R3's specification asks whether the
SAME country responds more strongly on those days. Identification is
within-country and within-month, off roughly forty thousand country-days, and
no city-level emissions series is used, so the pseudo-replication documented in
identification_test_memo.md does not apply.

36_built_tilt_diagnostic.py sized the variation before this was written: after
country x year-month fixed effects D carries 10.3 % of the cooling exposure's
variation and 5.2 % of the heating exposure's, and correlates -0.07 and -0.18
with the exposure it modulates. There is something to identify from, and it is
close to orthogonal to the main effect.

WHY ALL FOUR BLOCKS ENTER TOGETHER
----------------------------------
Cooling and heating exposure enter jointly in R3 so that neither is absorbed
into the other. The same applies to their tilts: a model with only the cooling
side and its tilt would push the heating season into the residual. So one fit
carries four distributed-lag blocks -- cooling, cooling tilt, heating, heating
tilt -- and both tilts are read from it.

WHAT WOULD FALSIFY IT
---------------------
Ground Transport and Aviation have no cooling channel. If the tilt moves them
too, it is not a built-environment mechanism: it is the spatial allocation of
the emissions product responding to which cities are hot, which would
contaminate every sector equally. The two control sectors are therefore fitted
under the identical specification and reported beside Power.

A note on the collinearity screen that is NOT used by default. In several
countries D is collinear with X within country (Chile +0.91, Norway -0.98 on
the cooling side; Brazil +0.91, Colombia -0.89 on the heating side), and an
earlier version dropped them. That screen removed ten of twenty-eight countries
including China and Australia -- the two with the largest tilt variation on the
cooling side -- so it discarded most of the identification in the name of
protecting it. It is kept behind --samples noncollinear for inspection and is
not part of the reported specification.

USAGE
-----
  python 37_built_tilt_response.py --tier AB
  python 37_built_tilt_response.py --sectors Power --budget 150
  python 37_built_tilt_response.py --status
"""

from __future__ import annotations

import argparse
import glob
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
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
CD = Path(os.environ.get(
    "HC_CARBON_DIR",
    str(BASE / "CarbonData" /
        "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide")))
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
SCHEMES = BASE / "outputs" / "population" / "fua_weight_schemes.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
TILT = BASE / "outputs" / "exposure" / "country_daily_tilt.parquet"
CACHE = BASE / "outputs" / "response" / "r6_tilt_cache.parquet"

SECTORS = ["Power", "Industry", "Ground Transport", "Aviation", "Residential"]
COOL, HEAT = "SCDH24", "HDH15"
FORM = BASE / "outputs" / "population" / "fua_form_weights.parquet"
# Alternative weightings, each generating a tilt against the population-weighted
# series. Scale first, then the two form weights built by 38_form_weights.py.
# w_compact is the one that ranks cities differently from built volume (median
# within-country Spearman 0.78 against 0.98 for w_tall), so it is the only one
# that can carry information the scale weight does not.
# NOTE ON THE SCHEMES TABLE. In fua_weight_schemes.parquet the columns
# `built_s` and `built_v` hold the zonal sum for the whole FUA, repeated on
# every city row that shares a fuacode; only `w_built_s` and `w_built_v` are
# divided by n_share. `pop_weight_dedup` is already deduplicated. Summing or
# weighting by the raw columns therefore double counts 34 cities in 5
# countries -- 24 of them in China, which inflates its built volume per capita
# by 31 % -- while leaving population correct. The deduplicated columns are
# used everywhere below.
TILTS = {"bv": "w_built_v", "tall": "w_tall", "compact": "w_compact"}
# LCZ variants from 39_lcz_variants.py. The sweep there showed that sharpening
# the compact share is the only lever that widens the gap from population
# weighting far enough to matter: compact^4 keeps 62 % of the exposure
# variation against 10 % for built volume.
LCZ_W = BASE / "outputs" / "population" / "fua_lcz_weights.parquet"
LCZ_TILTS = {"cp4": "w_compact_p4", "cp2": "w_compact_p2", "hi1": "w_hi1"}
ALL_TILTS = {**TILTS, **LCZ_TILTS}
LEADS, LAGS = 7, 14
CACHE_KEYS = ["tier", "spec", "sample", "sector"]


def load_r3():
    """Reuse R3's absorption and clustered VCV so the two are comparable."""
    spec = importlib.util.spec_from_file_location(
        "r3_dyn", BASE / "16_r3_dynamic_response.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["r3_dyn"] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------
def build_tilt(force: bool = False, tilts: list[str] | None = None
               ) -> pd.DataFrame:
    """Population- and alternatively weighted national exposure, and the gaps.

    The positivity screen applies ONLY to the weights this fit will use. An
    earlier version required every candidate weight to be positive, which let
    the most restrictive one decide the sample for all of them: w_hi1 is zero
    throughout Greece and Portugal, since neither has a compact high-rise LCZ
    cell, so both countries vanished from every specification including the
    ones that did not use w_hi1. Greece carries the largest heatwave-day
    cooling load in the sample, so this was not a harmless loss.
    """
    use = {k: v for k, v in ALL_TILTS.items() if tilts is None or k in tilts}
    key = "+".join(sorted(use))
    path = TILT.with_name(f"country_daily_tilt__{key}.parquet")
    if path.exists() and not force:
        return pd.read_parquet(path)

    w = pd.read_parquet(SCHEMES)[["city_id", "country", "pop_weight_dedup",
                                  "w_built_v"]]
    if FORM.exists():
        f = pd.read_parquet(FORM)[["city_id", "w_tall", "w_compact"]]
        w = w.merge(f, on="city_id", how="left")
    if LCZ_W.exists():
        z = pd.read_parquet(LCZ_W)[["city_id"] + list(LCZ_TILTS.values())]
        w = w.merge(z, on="city_id", how="left")
        TILTS.update(LCZ_TILTS)
    cols = ["pop_weight_dedup"] + [c for c in use.values() if c in w.columns]
    # A city missing any weight would enter some national means and not others,
    # so the set of cities behind every weighting is held identical.
    w = w[(w[cols] > 0).all(axis=1)]
    ev = pd.read_parquet(EVENTS, columns=["city_id", "date", COOL, HEAT])
    d = ev.merge(w, on="city_id", how="inner")

    # Weighted means without a per-group python call: numerator and denominator
    # are summed separately, which on two million rows is the difference
    # between seconds and minutes.
    out = {}
    pairs = [("pop_weight_dedup", "pop")] + [
        (c, t) for t, c in use.items() if c in d.columns]
    for wt, tag in pairs:
        num = pd.DataFrame({
            "country": d.country, "date": d.date,
            COOL: d[COOL] * d[wt], HEAT: d[HEAT] * d[wt], "wt": d[wt]})
        g = num.groupby(["country", "date"], sort=False)[[COOL, HEAT, "wt"]].sum()
        for c in (COOL, HEAT):
            out[f"{c}_{tag}"] = g[c] / g["wt"]
    t = pd.DataFrame(out).reset_index()
    for c in (COOL, HEAT):
        for tag in use:
            if f"{c}_{tag}" in t.columns:
                t[f"D_{tag}_{c}"] = t[f"{c}_{tag}"] - t[f"{c}_pop"]
    t["date"] = pd.to_datetime(t.date)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(path, index=False)
    print(f"    built the tilt panel for [{key}]: {len(t):,} country-days")
    return t


def collinear_countries(t: pd.DataFrame, ex: pd.DataFrame,
                        thresh: float = 0.7) -> list[str]:
    """Countries where D is collinear with X within country, either side."""
    d = t.merge(ex[["country", "date", "tier"]], on=["country", "date"])
    d = d[d.tier.isin(["A", "B"])]
    d["ym"] = d.date.dt.to_period("M").astype(str)
    bad = set()
    for c, s in d.groupby("country"):
        for v in (COOL, HEAT):
            dcol = f"D_bv_{v}"
            if dcol not in s.columns:
                continue
            x = s[f"{v}_pop"] - s.groupby("ym")[f"{v}_pop"].transform("mean")
            y = s[dcol] - s.groupby("ym")[dcol].transform("mean")
            if x.std() > 0 and y.std() > 0:
                if abs(float(np.corrcoef(x, y)[0, 1])) > thresh:
                    bad.add(c)
    return sorted(bad)


# --------------------------------------------------------------------------
def fit_cell(r3, panel: pd.DataFrame, sector: str,
             tilts: list[str]) -> dict | None:
    """Four distributed-lag blocks, R3's fixed effects and clustered VCV.

    Blocks are cooling, cooling tilt, heating, heating tilt. The cumulative
    post-onset sum over lags 0..LAGS is the reported quantity for each, with
    the joint test on the leads reported as the pre-trend as elsewhere.
    """
    d = panel[panel.sector == sector].sort_values(["country", "date"])
    d = d.reset_index(drop=True).copy()
    d = d[d.v > 0].copy()
    if len(d) < 2000:
        return None
    d["y"] = np.log(d.v)

    blocks = [("X_cool", f"{COOL}_pop"), ("X_heat", f"{HEAT}_pop")]
    for tag in tilts:
        blocks += [(f"D{tag}_cool", f"D_{tag}_{COOL}"),
                   (f"D{tag}_heat", f"D_{tag}_{HEAT}")]
    blocks = [b for b in blocks if b[1] in d.columns]
    names, idx_of = [], {}
    for tag, col in blocks:
        g = d.groupby("country")[col]
        start = len(names)
        for s in range(-LEADS, LAGS + 1):
            nm = f"{tag}{s:+d}"
            d[nm] = g.shift(s)
            names.append(nm)
        idx_of[tag] = np.arange(start, len(names))
    d = d.dropna(subset=names + ["y"]).reset_index(drop=True)
    if len(d) < 2000:
        return None

    M = np.column_stack([d[names].to_numpy("float64"), d["y"].to_numpy()])
    f_cym = pd.factorize(d.country + "_" +
                         d.date.dt.to_period("M").astype(str))[0]
    f_date = pd.factorize(d.date)[0]
    f_cdow = pd.factorize(d.country + "_" +
                          d.date.dt.dayofweek.astype(str))[0]
    Z, conv = r3.demean_factors(M, [f_cym, f_date, f_cdow])
    X, y = Z[:, :-1], Z[:, -1]
    b = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * 1e-10, X.T @ y)
    e = y - X @ b
    V, _ = r3.twoway_vcov(X, e, pd.factorize(d.country)[0], f_date)

    lag = np.arange(-LEADS, LAGS + 1)
    row = {"sector": sector, "n": len(d), "n_country": d.country.nunique(),
           "conv": conv, "r2": 1 - e.var() / y.var()}
    for tag, _ in blocks:
        ix = idx_of[tag]
        post = ix[lag >= 0]
        cvec = np.zeros(len(b))
        cvec[post] = 1.0
        cum = float(cvec @ b)
        cse = float(np.sqrt(max(cvec @ V @ cvec, 0)))
        row[f"{tag}_cum"] = 100 * cum
        row[f"{tag}_se"] = 100 * cse
        row[f"{tag}_t"] = cum / cse if cse > 0 else np.nan
        # pre-trend: joint Wald on the leads
        pre = ix[lag < 0]
        Vp = V[np.ix_(pre, pre)]
        try:
            w = float(b[pre] @ np.linalg.solve(Vp, b[pre]))
            from scipy import stats
            row[f"{tag}_pre_p"] = float(stats.chi2.sf(w, len(pre)))
        except Exception:                                    # noqa: BLE001
            row[f"{tag}_pre_p"] = np.nan
    return row


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="AB", choices=["A", "AB"])
    ap.add_argument("--sectors", nargs="+", default=None)
    ap.add_argument("--samples", nargs="+", default=["all"],
                    help="'noncollinear' is retained but no longer default: "
                         "it drops the countries that carry most of the "
                         "identifying variation")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tilts", nargs="+", default=["bv", "tall", "compact"],
                    choices=list(ALL_TILTS))
    ap.add_argument("--spec", default=None,
                    help="label for this tilt set; defaults to the tilts joined")
    args = ap.parse_args()
    spec = args.spec or "+".join(args.tilts)

    print("=" * 78)
    print(f"R6 built-environment tilt -- Tier {args.tier}, spec {spec}")
    print("=" * 78)

    t = build_tilt(force=args.force, tilts=args.tilts)
    ex = pd.read_parquet(EXPOSURE, columns=["country", "date", "tier"])
    ex["date"] = pd.to_datetime(ex.date)
    ex = ex.drop_duplicates(["country", "date"])
    bad = collinear_countries(t, ex)
    print(f"    collinear countries (|corr(X, D)| > 0.7 within country): "
          f"{', '.join(bad) if bad else 'none'}")

    r3 = load_r3()
    em = r3.load_emissions()
    panel = em.merge(t, on=["country", "date"]).merge(ex, on=["country", "date"])
    keep = {"A": ["A"], "AB": ["A", "B"]}[args.tier]
    panel = panel[panel.tier.isin(keep)]
    print(f"    panel: {len(panel):,} country-day-sector rows, "
          f"{panel.country.nunique()} countries")

    sectors = args.sectors or SECTORS
    cache = pd.read_parquet(CACHE) if CACHE.exists() else \
        pd.DataFrame(columns=CACHE_KEYS)
    if args.force and len(cache) and "spec" in cache.columns:
        cache = cache[~(cache.tier.eq(args.tier) & cache.spec.eq(spec)
                        & cache.sector.isin(sectors))]

    todo = [(s, sec) for s in args.samples for sec in sectors]
    if len(cache) and "spec" in cache.columns:
        done = {(r.tier, r.spec, r.sample, r.sector) for r in cache.itertuples()}
        todo = [x for x in todo if (args.tier, spec, x[0], x[1]) not in done]
    print(f"    cells cached: {len(cache)}   still to fit: {len(todo)}")
    if args.status:
        if len(cache):
            print(cache[["sample", "sector", "D_cool_t", "D_heat_t"]]
                  .round(2).to_string(index=False))
        return

    t0, rows = time.time(), []
    for sample, sec in todo:
        if args.budget and time.time() - t0 > args.budget:
            print(f"    budget reached; {len(todo) - len(rows)} cells left")
            break
        p = panel if sample == "all" else panel[~panel.country.isin(bad)]
        t1 = time.time()
        r = fit_cell(r3, p, sec, args.tilts)
        if r is None:
            print(f"    {sample:14s}{sec:18s} not estimable")
            continue
        r.update({"tier": args.tier, "spec": spec, "sample": sample})
        rows.append(r)
        cells = "".join(
            f"  {tag}:{r.get(f'D{tag}_cool_t', float('nan')):+5.2f}/"
            f"{r.get(f'D{tag}_heat_t', float('nan')):+5.2f}"
            for tag in args.tilts)
        print(f"    {sec:18s} X {r['X_cool_t']:+5.2f}/{r['X_heat_t']:+5.2f}"
              f" | tilt t cool/heat{cells}   {time.time() - t1:5.1f}s",
              flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([cache, pd.DataFrame(rows)],
                  ignore_index=True).to_parquet(CACHE, index=False)

    cache = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
    done = cache[cache.tier == args.tier] if len(cache) else cache
    this_spec = done[done.spec == spec] if len(done) and "spec" in done else done
    if len(this_spec) >= len(args.samples) * len(sectors):
        out = RESULTS / "R6" / "tables"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"T12b_built_tilt_response_tier{args.tier}.csv"
        # every spec for this tier, not only the one just fitted
        done.sort_values(["spec", "sample", "sector"]).to_csv(p, index=False)
        print(f"\n    {p.relative_to(RESULTS)}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
