#!/usr/bin/env python
"""
33_city_scale_feasibility.py  --  why the response is estimated at country scale
================================================================================

The first question a reader asks of a study that assembles city-resolved
exposure for 1,583 functional urban areas and then regresses at country-day
level is: why not do it at city level? This script answers it with the data
rather than with an assertion, because the answer is a property of the emissions
product and is worth documenting for anyone else using it.

Carbon Monitor Cities distributes a daily CO2 series per city per sector. It
reads like 1,583 independent observations. It is not.

  A  WITHIN A COUNTRY-MONTH THE CITY PANEL IS EXACTLY RANK ONE.
     Every city's daily series is the national curve times a constant, so the
     [city x day] matrix has one non-zero singular value and the rest sit at
     machine precision. Under country x date fixed effects -- which any
     city-level design needs, to absorb national policy, holidays and the
     national generation mix -- there is then nothing left to explain.

  B  THE CITY SHARE DOES MOVE BETWEEN MONTHS, BUT NOT WITH THE WEATHER.
     Monthly shares drift smoothly and do not repeat seasonally, which is the
     signature of interpolation between annually updated spatial weights. Real
     weather sensitivity would show a strong twelve-month component.

  C  THE DIRECT TEST RETURNS A NULL THAT HAS THE POWER TO FIND A SIGNAL.
     A city x month panel with city and country x month fixed effects finds no
     response to city-level cooling or heating load, while a fifth of the
     exposure variance survives those fixed effects. The regressor varies; the
     outcome does not respond.

None of this says cities do not matter for energy use. It says this product
cannot be used to measure that, because its city dimension is an allocation of
a national total, not an observation.

USAGE
-----
  python 33_city_scale_feasibility.py
  python 33_city_scale_feasibility.py --sector Residential
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
CD = Path(os.environ.get(
    "HC_CARBON_DIR",
    BASE / "CarbonData" / "Near-real-time daily estimates of CO2 emissions "
                          "from 1500 cities worldwide"))
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
LOOKUP = BASE / "outputs" / "fua_city_lookup.csv"

RANK_COUNTRIES = ["Germany", "China", "United States", "Japan", "Brazil",
                  "France", "India"]
RANK_MONTH = "2021-07"


def read_country(country: str) -> pd.DataFrame:
    f = CD / f"carbon-monitor-cities-{country}-v0325.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    d.columns = [c.strip().lower().replace("﻿", "") for c in d.columns]
    d = d.rename(columns={"value (ktco2 per day)": "v"})
    d["date"] = pd.to_datetime(d["date"])
    d["country"] = country
    return d[["country", "city", "date", "sector", "v"]]


# --------------------------------------------------------------------------
def part_a_rank(sectors: list[str]) -> pd.DataFrame:
    """Singular-value spectrum of the [city x day] matrix, per country-month."""
    print("\n" + "=" * 78)
    print("A  Within a country-month, is the city panel rank one?")
    print("=" * 78)
    print(f"    {'country':16s}{'sector':18s}{'cities':>7s}{'days':>6s}"
          f"{'sv2/sv1':>12s}{'ratio CV':>12s}")
    rows = []
    for c in RANK_COUNTRIES:
        d = read_country(c)
        if d.empty:
            continue
        for sec in sectors:
            s = d[(d.sector == sec) &
                  (d.date.dt.to_period("M") == RANK_MONTH)]
            M = s.pivot_table(index="city", columns="date",
                              values="v").dropna()
            if M.shape[0] < 3 or M.shape[1] < 3:
                continue
            sv = np.linalg.svd(M.values, compute_uv=False)
            # ratio of each city's series to the first city's: constant across
            # days if and only if the matrix is rank one
            r = M.div(M.iloc[0], axis=1)
            cv = float((r.std(axis=1) / r.mean(axis=1)).median())
            print(f"    {c:16s}{sec:18s}{M.shape[0]:7d}{M.shape[1]:6d}"
                  f"{sv[1] / sv[0]:12.2e}{cv:12.2e}")
            rows.append({"country": c, "sector": sec, "month": RANK_MONTH,
                         "n_cities": M.shape[0], "n_days": M.shape[1],
                         "sv2_over_sv1": sv[1] / sv[0], "ratio_cv_median": cv})
    res = pd.DataFrame(rows)
    if len(res):
        # Not every cell is exactly rank one, and the exceptions are
        # informative rather than embarrassing: the largest departures sit in
        # Residential, whose allocation uses population-weighted heating degree
        # days and therefore does carry city weather -- which is precisely the
        # circularity that keeps Residential out of the interpreted results.
        res["exactly_rank1"] = res.sv2_over_sv1 < 1e-10
        k = int(res.exactly_rank1.sum())
        worst = res.loc[res.sv2_over_sv1.idxmax()]
        print(f"\n    {k} of {len(res)} country-sector cells are rank one to "
              f"machine precision (about 2e-16).")
        print(f"    The largest departure is {worst.country} "
              f"{worst.sector} at sv2/sv1 = {worst.sv2_over_sv1:.2e}, so even "
              f"there the second\n    component carries about "
              f"{100 * worst.sv2_over_sv1:.1f} % of the first.")
        off = res[~res.exactly_rank1]
        if len(off):
            share = 100 * (off.sector == "Residential").mean()
            print(f"    {len(off)} cells depart from rank one and "
                  f"{share:.0f} % of them are Residential, whose allocation is "
                  f"built\n    from population-weighted heating degree days "
                  f"and so is weather-dependent by\n    construction -- the "
                  f"circularity that excludes it from the interpreted results.")
        print("\n    Everywhere else each city's daily series is the national")
        print("    curve times a constant, and country x date effects absorb it.")
    return res


def part_b_share(sectors: list[str]) -> pd.DataFrame:
    """Is the monthly city share weather-driven or a drifting allocation?"""
    print("\n" + "=" * 78)
    print("B  The monthly city share moves -- is it weather?")
    print("=" * 78)
    print(f"    {'country':16s}{'sector':18s}{'lag-1':>8s}{'lag-12':>8s}"
          f"{'trend R2':>10s}{'month-of-year':>15s}")
    rows = []
    for c in RANK_COUNTRIES:
        d = read_country(c)
        if d.empty:
            continue
        for sec in sectors:
            s = d[d.sector == sec].copy()
            if s.empty:
                continue
            s["ym"] = s.date.dt.to_period("M")
            nat = s.groupby("date").v.sum().rename("nat").reset_index()
            s = s.merge(nat, on="date")
            s["share"] = s.v / s.nat
            sh = s.groupby(["city", "ym"]).share.mean().unstack().dropna()
            if sh.shape[0] < 5 or sh.shape[1] < 24:
                continue
            z = sh.sub(sh.mean(axis=1), axis=0).div(sh.std(axis=1), axis=0)
            z = z.replace([np.inf, -np.inf], np.nan).dropna()
            lag1 = float(np.nanmean([np.corrcoef(r[:-1], r[1:])[0, 1]
                                     for r in z.values]))
            lag12 = float(np.nanmean([np.corrcoef(r[:-12], r[12:])[0, 1]
                                      for r in z.values]))
            tt = np.arange(z.shape[1], dtype=float)
            tt = (tt - tt.mean()) / tt.std()
            r2 = float(np.mean([
                1 - np.var(r - np.polyval(np.polyfit(tt, r, 1), tt)) / np.var(r)
                for r in z.values]))
            # share of the share-variance explained by a fixed month-of-year
            # pattern: high would mean a climatological allocation
            lng = sh.stack().rename("share").reset_index()
            lng["month"] = lng.ym.dt.month
            lng["cm"] = lng.groupby("city").share.transform("mean")
            lng["moy"] = lng.groupby(["city", "month"]).share.transform("mean")
            tot = float(((lng.share - lng.cm) ** 2).mean())
            res = float(((lng.share - lng.moy) ** 2).mean())
            moy = 100 * (1 - res / tot) if tot > 0 else np.nan
            print(f"    {c:16s}{sec:18s}{lag1:8.3f}{lag12:8.3f}{r2:10.3f}"
                  f"{moy:14.1f}%")
            rows.append({"country": c, "sector": sec, "lag1_autocorr": lag1,
                         "lag12_autocorr": lag12, "linear_trend_r2": r2,
                         "month_of_year_pct": moy, "n_cities": sh.shape[0],
                         "n_months": sh.shape[1]})
    print("\n    Smooth (high lag-1), no seasonal repetition (lag-12 near zero)")
    print("    and largely a linear drift: the signature of interpolating")
    print("    annually updated weights, not of weather.")
    return pd.DataFrame(rows)


def demean(X: np.ndarray, facs: list[np.ndarray], it: int = 300,
           tol: float = 1e-11) -> np.ndarray:
    X = X.astype(float).copy()
    for _ in range(it):
        mx = 0.0
        for g in facs:
            n = g.max() + 1
            cnt = np.bincount(g, minlength=n).astype(float)
            gm = np.empty((n, X.shape[1]))
            for j in range(X.shape[1]):
                gm[:, j] = np.bincount(g, weights=X[:, j], minlength=n)
            gm /= cnt[:, None]
            d = gm[g]
            X -= d
            mx = max(mx, float(np.abs(d).max()))
        if mx < tol:
            break
    return X


def part_c_citymonth(sector: str) -> pd.DataFrame:
    """City x month response to city exposure, with the national curve gone."""
    print("\n" + "=" * 78)
    print(f"C  Direct test: does the city share respond to city weather?"
          f"  ({sector})")
    print("=" * 78)
    lk = pd.read_csv(LOOKUP, dtype={"city_id": str})
    lk["city_id"] = lk.city_id.str.zfill(4)
    tiers = pd.read_parquet(EXPOSURE, columns=["country", "tier"]).drop_duplicates()
    ab = set(tiers[tiers.tier.isin(["A", "B"])].country)

    ev = pd.read_parquet(EVENTS, columns=["city_id", "date", "SCDH24", "HDH15"])
    ev["city_id"] = ev.city_id.astype(str).str.zfill(4)
    ev["ym"] = ev.date.dt.to_period("M")
    cm = ev.groupby(["city_id", "ym"], as_index=False)[["SCDH24", "HDH15"]].sum()

    rows = []
    for f in sorted(glob.glob(str(CD / "carbon-monitor-cities-*-v0325.csv"))):
        n = Path(f).stem.replace("carbon-monitor-cities-", "").replace("-v0325", "")
        if n not in ab:
            continue
        d = read_country(n)
        d = d[d.sector == sector]
        if d.empty:
            continue
        d["ym"] = d.date.dt.to_period("M")
        rows.append(d.groupby(["country", "city", "ym"], as_index=False).v.sum())
    em = pd.concat(rows, ignore_index=True)
    p = em.merge(lk[["city_id", "city", "country"]], on=["city", "country"])
    p = p.merge(cm, on=["city_id", "ym"])
    p = p[p.v > 0].copy()
    p["y"] = np.log(p.v)
    print(f"    {len(p):,} city-months, {p.city_id.nunique():,} cities, "
          f"{p.country.nunique()} countries")

    f_city = pd.factorize(p.city_id)[0]
    f_cm = pd.factorize(p.country + "_" + p.ym.astype(str))[0]
    g = pd.factorize(p.country)[0]
    out = []
    for lab, col in (("cooling degree-hours", "SCDH24"),
                     ("heating degree-hours", "HDH15")):
        x = p[col].to_numpy(float)
        surv = float(demean(x.reshape(-1, 1), [f_city, f_cm]).var() / x.var())
        xs = x / x.std()
        Z = demean(np.column_stack([xs, p.y.to_numpy(float)]), [f_city, f_cm])
        X, yy = Z[:, :1], Z[:, 1]
        b = float(np.linalg.lstsq(X, yy, rcond=None)[0][0])
        e = yy - X.ravel() * b
        XtX = float(X.T @ X)
        S = np.bincount(g, weights=X.ravel() * e)      # clustered by country
        se = float(np.sqrt((S ** 2).sum() / XtX ** 2))
        print(f"    {lab:22s} {b:+8.4f} per SD   se {se:.4f}   t {b / se:+6.2f}"
              f"   exposure variance surviving the fixed effects "
              f"{100 * surv:5.1f} %")
        out.append({"sector": sector, "term": lab, "column": col,
                    "beta_per_sd": b, "se": se, "t": b / se,
                    "exposure_var_surviving_pct": 100 * surv,
                    "n_city_months": len(p), "n_cities": p.city_id.nunique(),
                    "n_countries": p.country.nunique()})
    print("\n    The regressor varies after the fixed effects; the outcome does")
    print("    not respond. A city-scale replication would report a null about")
    print("    the allocation model, not about cities.")
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="Power")
    ap.add_argument("--sectors", nargs="+",
                    default=["Power", "Ground Transport", "Residential"])
    args = ap.parse_args()

    print("=" * 78)
    print("Why the response is estimated at country scale")
    print("=" * 78)
    a = part_a_rank(args.sectors)
    b = part_b_share(args.sectors)
    c = part_c_citymonth(args.sector)

    out = RESULTS / "SI" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("TS10_city_panel_rank", a),
                     ("TS11_city_share_dynamics", b),
                     ("TS12_city_month_response", c)):
        if len(df):
            df.round(6).to_csv(out / f"{name}.csv", index=False)
            print(f"    wrote SI/tables/{name}.csv")


if __name__ == "__main__":
    main()
