#!/usr/bin/env python
"""
22_r5_asymmetry.py  --  R5: observed thermal carbon-response asymmetry
=======================================================================

Estimates, country by country, how much national daily Power-sector CO2
responds to a unit of cooling load versus a unit of heating load, and asks
whether the two are equal. The section is titled OBSERVED asymmetry throughout
and the numbers are never described as a projection: what a country's emissions
did per degree-hour in 2019-2021 is not what they will do under a warmer
climate with a different generation mix.

THREE DESIGN CONSTRAINTS, ALL FROM THE REVIEW
---------------------------------------------
1  PER-COUNTRY COEFFICIENTS ARE UNSTABLE with 28 countries and three years, so
   raw estimates are shrunk toward the pooled mean by empirical Bayes with the
   full posterior interval reported. Raw and shrunk are both shown; a reader
   must be able to see how much of each estimate is data and how much is prior.

2  THE RATIO EXPLODES. A thermal carbon asymmetry index defined as
   beta_heat / beta_cold is unusable when the denominator is near zero, which
   happens for exactly the countries with little heating load. The DIFFERENCE
   beta_heat - beta_cold is the primary quantity; the ratio is computed only to
   document the instability that motivates avoiding it.

3  POWER ONLY. Residential daily emissions in Carbon Monitor are constructed
   from population-weighted heating degree-days (Huo et al. 2022, Eq 6-7), so
   regressing them on a heating-degree-hour series is circular. Industry and
   Ground Transport are reported for completeness but not interpreted.

AN IDENTIFICATION SCREEN THAT DOES REAL WORK
--------------------------------------------
The asymmetry is only defined where BOTH sides are identified, and in this
sample that is far from automatic: the United Kingdom has 36 country-days with
any cooling load at all, Norway 32, and Brazil's mean heating load is 5 K.h.
A country therefore enters the asymmetry comparison only if it has enough days
of real exposure on BOTH sides:

    heat side   >= MIN_DAYS days with w_SCDH24 >= MIN_SCDH
    cold side   >= MIN_DAYS days with w_HDH15  >= MIN_HDH

Countries failing the screen are estimated and reported for the side they do
identify, and excluded from the difference. Reporting an asymmetry for Norway
would be reporting the prior.

UNITS
-----
Coefficients are per 100 population-weighted degree-hours, so heat and cold are
on a common physical scale and their difference is meaningful. A per-own-SD
version is also computed, because the two questions differ: per degree-hour
asks about physical sensitivity, per SD asks which side moves emissions more
given how much of each a country actually experiences.

MODEL
-----
    ln E_(c,t) = mu_(c,ym) + lambda_t + nu_(c,dow)
                 + sum_(s=0..7) betaH_(c,s) SCDH24_(c,t-s)/100
                 + sum_(s=0..7) betaC_(c,s) HDH15_(c,t-s)/100 + eps

This is R3's fixed-effect structure with the slopes allowed to vary by country;
the GLOBAL DATE effect lambda_t is what makes it identified. An earlier version
ran a separate regression per country, which cannot represent a global date
effect because a single-country series has one observation per date. That
version is retained behind --naive because its failure is instructive: it
returns a large, apparently significant heat-minus-cold asymmetry for AVIATION
(+7.5 {unit_label()}, z = +4.5), a sector with no cooling channel, because the 2020
collapse and recovery moved every country on a common trajectory that
correlates with rising cooling load and falling heating load.

Standard errors are Driscoll-Kraay (bandwidth 14 days), which is robust to both
serial correlation within a country and cross-sectional correlation across
countries on the same date, and whose rank is not bounded by 28 countries.

PLACEBO CALIBRATION
-------------------
Aviation and Ground Transport are estimated under the identical specification
and reported alongside. They are not clean zeros -- winter storms genuinely
ground flights, so Aviation has a real cold-side channel -- which is why they
are used as a MAGNITUDE FLOOR rather than as a pass/fail test: a Power
coefficient of the same size as the placebo coefficients carries no
interpretation, whatever its t statistic.

INPUTS
------
  CarbonData/Near-real-time .../carbon-monitor-cities-*-v0325.csv
  outputs/exposure/country_daily_exposure.parquet

OUTPUTS
-------
  outputs/response/r5_asymmetry_tier{AB,C}.parquet
  Results/R5/tables/T6_thermal_asymmetry_<sector>_tier{AB,C}.csv
  Results/SI/figures/FigS11_asymmetry_diagnostics_tier{AB,C}.{png,pdf,svg}
  (Figure 6 itself is drawn by 34_response_figures.py)

USAGE
-----
  python 22_r5_asymmetry.py --tier AB
  python 22_r5_asymmetry.py --tier AB --naive       # reproduce the failure
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
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
OUT_DIR = BASE / "outputs" / "response"
FIG_DIR = RESULTS_ROOT / "R5" / "figures"
TAB_DIR = RESULTS_ROOT / "R5" / "tables"

LAGS = 7
HAC_BW = 14
SCALE = 100.0            # coefficients per 100 degree-hours

# Unit in which the two sides are expressed. Set by --unit; the screen that
# decides which countries identify a side always uses the RAW degree-hours, so
# the country set cannot move with the unit.
#
#   per100  one unit = 100 K.h. Absolute and physical, but how rare 100 K.h is
#           depends on the base temperature, so the two sides are not on a
#           common footing and their difference inherits the base pair.
#   perSD   one unit = one standard deviation of that side's own daily series.
#   perP95  one unit = that side's 95th percentile among days when it is
#           positive: a common probability footing rather than a physical one.
#
# Rescaling a regressor cannot change its t statistic and does not; only the
# DIFFERENCE between the two sides moves, because a difference between two
# differently scaled quantities inherits the exchange rate between them.
UNIT_C = SCALE
UNIT_H = SCALE
UNIT_NAME = "per100"
MIN_DAYS = 100           # days of real exposure needed on a side
MIN_SCDH = 5.0           # K.h -- a day with some genuine cooling load
MIN_HDH = 50.0           # K.h -- a day with some genuine heating load
MIN_OBS = 400
PLACEBOS = ["Aviation", "Ground Transport"]
ALL_SECTORS = ["Power", "Industry", "Ground Transport", "Aviation",
               "Residential"]

SAVE_FORMATS = ("png", "pdf", "svg")
SAVE_DPI = 600
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "savefig.dpi": SAVE_DPI, "figure.dpi": 110,
                     "svg.fonttype": "none"})


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


def newey_west(X: np.ndarray, e: np.ndarray, bw: int) -> np.ndarray:
    """HAC meat matrix with Bartlett kernel, for an ordered time series."""
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, bw + 1):
        w = 1.0 - l / (bw + 1.0)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    return S


def demean(X: np.ndarray, facs: list[np.ndarray], it: int = 200,
           tol: float = 1e-10) -> np.ndarray:
    """Within transformation over several factors by alternating projections."""
    X = np.asarray(X, dtype="float64").copy()
    meta = [(f, np.bincount(f).astype(float), f.max() + 1) for f in facs]
    for _ in range(it):
        prev = X.copy()
        for f, cnt, n in meta:
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(f, weights=X[:, j], minlength=n) / cnt)[f]
        if np.max(np.abs(X - prev)) < tol:
            break
    return X


def driscoll_kraay(X: np.ndarray, e: np.ndarray, tt: np.ndarray,
                   bw: int) -> np.ndarray:
    """Driscoll-Kraay meat: scores summed across countries within a date, then
    Newey-West over dates.

    This is the right variance for this design. The naive per-country
    specification failed its Aviation placebo because a global common factor --
    the 2020 collapse and recovery, which moved every country at once -- was
    not absorbed. Once a global date fixed effect is included the estimator is
    identified, but the residuals remain both serially correlated within a
    country and cross-sectionally correlated across countries on the same date.
    Driscoll-Kraay (1998) handles both, and unlike clustering on 28 countries it
    does not have rank bounded by the number of countries.
    """
    Xe = X * e[:, None]
    nT = tt.max() + 1
    S = np.zeros((nT, X.shape[1]))
    for j in range(X.shape[1]):
        S[:, j] = np.bincount(tt, weights=Xe[:, j], minlength=nT)
    return newey_west(S, np.ones(nT), bw)


def fit_joint(p: pd.DataFrame, scr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One panel regression with country-specific heat and cold slopes.

    The fixed-effect structure is exactly the one validated in R3 -- country x
    year-month, GLOBAL DATE, country x day-of-week -- so the global date effect
    that the per-country specification cannot represent is absorbed here. Only
    the slopes are allowed to vary by country, and only on the side a country
    actually identifies (see the screen), so no column is a near-zero regressor
    standing in for a country with no cooling season.
    """
    p = p.sort_values(["country", "date"]).reset_index(drop=True).copy()
    p["h"] = p["w_SCDH24"] / UNIT_C
    p["c"] = p["w_HDH15"] / UNIT_H
    g = p.groupby("country")
    for s in range(LAGS + 1):
        p[f"h{s}"] = g["h"].shift(s)
        p[f"c{s}"] = g["c"].shift(s)
    lag_cols = [f"{k}{s}" for s in range(LAGS + 1) for k in ("h", "c")]
    p = p.dropna(subset=lag_cols + ["y"]).reset_index(drop=True)

    ok_h = set(scr.loc[scr.heat_ok, "country"])
    ok_c = set(scr.loc[scr.cold_ok, "country"])
    countries = sorted(p.country.unique())

    cols, meta = [], []
    for ctry in countries:
        m = (p.country == ctry).values.astype("float64")
        for side, ok in (("h", ok_h), ("c", ok_c)):
            if ctry not in ok:
                continue
            for s in range(LAGS + 1):
                cols.append(p[f"{side}{s}"].values * m)
                meta.append((ctry, side, s))
    X = np.column_stack(cols)
    y = p["y"].values.astype("float64")

    f1 = pd.factorize(p.country + "_" + p.date.dt.to_period("M").astype(str))[0]
    f2 = pd.factorize(p.date)[0]
    f3 = pd.factorize(p.country + "_" + p.date.dt.dayofweek.astype(str))[0]
    Z = demean(np.column_stack([X, y]), [f1, f2, f3])
    X, y = Z[:, :-1], Z[:, -1]

    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-9
    inv = np.linalg.inv(XtX)
    b = inv @ (X.T @ y)
    e = y - X @ b
    V = inv @ driscoll_kraay(X, e, f2, HAC_BW) @ inv

    mdf = pd.DataFrame(meta, columns=["country", "side", "lag"])
    rows = []
    for ctry in countries:
        ih = mdf.index[(mdf.country == ctry) & (mdf.side == "h")].values
        ic = mdf.index[(mdf.country == ctry) & (mdf.side == "c")].values
        r = {"country": ctry, "n": int((p.country == ctry).sum())}
        for tag, ix in (("heat", ih), ("cold", ic)):
            if len(ix) == 0:
                r[tag] = np.nan
                r[tag + "_se"] = np.nan
                continue
            r[tag] = float(b[ix].sum())
            r[tag + "_se"] = float(np.sqrt(max(float(V[np.ix_(ix, ix)].sum()), 0)))
        if len(ih) and len(ic):
            cv = np.zeros(len(b))
            cv[ih] = 1.0
            cv[ic] = -1.0
            r["diff"] = float(cv @ b)
            r["diff_se"] = float(np.sqrt(max(float(cv @ V @ cv), 0)))
        else:
            r["diff"] = r["diff_se"] = np.nan
        sub = p[p.country == ctry]
        r["sd_scdh"] = float(sub["w_SCDH24"].std())
        r["sd_hdh"] = float(sub["w_HDH15"].std())
        rows.append(r)
    # The country slopes come out of ONE joint regression, so their sampling
    # errors are correlated: V has non-zero off-diagonal blocks between
    # countries, both because the global date effect is shared and because
    # Driscoll-Kraay is explicitly robust to cross-sectional dependence on the
    # same date. Reporting only the diagonal (the *_se columns above) is
    # sufficient for a single country's interval but NOT for any test that
    # compares countries with one another. The aggregation matrix and the full
    # covariance of the aggregates are therefore returned as well.
    #
    #   theta = A b      A is (C x k), one row per country, summing its lags
    #   Sigma = A V A'   the full C x C covariance, off-diagonal included
    agg = {}
    for tag, side in (("heat", "h"), ("cold", "c"), ("diff", None)):
        keep, A_rows = [], []
        for ctry in countries:
            ih = mdf.index[(mdf.country == ctry) & (mdf.side == "h")].values
            ic = mdf.index[(mdf.country == ctry) & (mdf.side == "c")].values
            a = np.zeros(len(b))
            if tag == "diff":
                if not (len(ih) and len(ic)):
                    continue
                a[ih] = 1.0
                a[ic] = -1.0
            else:
                ix = ih if side == "h" else ic
                if not len(ix):
                    continue
                a[ix] = 1.0
            keep.append(ctry)
            A_rows.append(a)
        if len(keep) >= 3:
            A = np.vstack(A_rows)
            agg[tag] = {"countries": keep, "theta": A @ b, "Sigma": A @ V @ A.T}

    info = {"n_obs": len(p), "n_par": X.shape[1],
            "r2": float(1 - e.var() / y.var()), "agg": agg}
    return pd.DataFrame(rows), info


def fit_country(d: pd.DataFrame) -> dict | None:
    """NAIVE per-country model, retained only to document its placebo failure.

    Year-month and day-of-week effects are country-specific here but there is no
    global date effect, because a single-country time series cannot represent
    one. That is precisely the gap the 2020 aviation collapse and recovery
    exploits: it moved every country on a common trajectory that correlates with
    rising cooling load and falling heating load, so this specification returns
    a large, apparently significant heat-minus-cold asymmetry for AVIATION,
    which has no cooling channel. Never used for the reported estimates.
    """
    d = d.sort_values("date").reset_index(drop=True).copy()
    d["h"] = d["w_SCDH24"] / SCALE
    d["c"] = d["w_HDH15"] / SCALE
    names, idx_h, idx_c = [], [], []
    for s in range(LAGS + 1):
        d[f"h{s}"] = d["h"].shift(s)
        d[f"c{s}"] = d["c"].shift(s)
        names += [f"h{s}", f"c{s}"]
        idx_h.append(len(names) - 2)
        idx_c.append(len(names) - 1)
    d = d.dropna(subset=names + ["y"]).reset_index(drop=True)
    if len(d) < MIN_OBS:
        return None

    ym = pd.get_dummies(d.date.dt.to_period("M").astype(str), drop_first=True)
    dw = pd.get_dummies(d.date.dt.dayofweek.astype(str), drop_first=True)
    X = np.column_stack([d[names].values.astype("float64"),
                         ym.values.astype("float64"),
                         dw.values.astype("float64"),
                         np.ones(len(d))])
    y = d["y"].values.astype("float64")

    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        return None
    inv = np.linalg.pinv(XtX)
    b = inv @ (X.T @ y)
    e = y - X @ b
    V = inv @ newey_west(X, e, HAC_BW) @ inv

    ih, ic = np.array(idx_h), np.array(idx_c)
    out = {"n": len(d)}
    for tag, ix in (("heat", ih), ("cold", ic)):
        cum = float(b[ix].sum())
        var = float(V[np.ix_(ix, ix)].sum())
        out[tag] = cum
        out[tag + "_se"] = float(np.sqrt(max(var, 0)))
    cvec = np.zeros(X.shape[1])
    cvec[ih] = 1.0
    cvec[ic] = -1.0
    diff = float(cvec @ b)
    dvar = float(cvec @ V @ cvec)
    out["diff"] = diff
    out["diff_se"] = float(np.sqrt(max(dvar, 0)))
    out["sd_scdh"] = float(d["w_SCDH24"].std())
    out["sd_hdh"] = float(d["w_HDH15"].std())
    return out


# --------------------------------------------------------------------------
def empirical_bayes(b: np.ndarray, s: np.ndarray,
                    it: int = 500, tol: float = 1e-12) -> dict:
    """Normal-normal empirical Bayes with an ML variance component.

    b_c ~ N(theta_c, s_c^2),  theta_c ~ N(mu, tau^2)

    tau^2 is the between-country variance that survives after sampling error is
    removed; when it is small relative to the s_c^2 the posterior collapses onto
    the pooled mean, which is the honest answer when per-country estimates are
    too noisy to distinguish. The shrinkage factor B_c is reported so that a
    reader can see exactly how much of each posterior is prior.
    """
    b = np.asarray(b, float)
    s2 = np.asarray(s, float) ** 2
    tau2 = max(b.var(ddof=1) - s2.mean(), 1e-12)
    mu = b.mean()
    for _ in range(it):
        w = 1.0 / (s2 + tau2)
        mu_new = float((w * b).sum() / w.sum())
        num = (w ** 2 * ((b - mu_new) ** 2 - s2)).sum()
        den = (w ** 2).sum()
        tau2_new = max(float(num / den), 0.0)
        if abs(tau2_new - tau2) < tol and abs(mu_new - mu) < tol:
            mu, tau2 = mu_new, tau2_new
            break
        mu, tau2 = mu_new, tau2_new
    B = s2 / (s2 + tau2) if tau2 > 0 else np.ones_like(s2)
    post = (1 - B) * b + B * mu
    post_se = np.sqrt(s2 * tau2 / (s2 + tau2)) if tau2 > 0 else np.zeros_like(s2)
    w = 1.0 / (s2 + tau2)
    return {"mu": mu, "mu_se": float(np.sqrt(1.0 / w.sum())), "tau2": tau2,
            "tau": float(np.sqrt(tau2)), "post": post, "post_se": post_se,
            "shrink": B,
            "i2": float(tau2 / (tau2 + np.median(s2))) if tau2 > 0 else 0.0}


# --------------------------------------------------------------------------
def unit_label() -> str:
    """How to name one unit of load in printed output.

    Hardcoding f"{unit_label()}" was correct only for the default unit; under --unit
    the same string would have mislabelled a per-SD coefficient as a per-load
    one, which is the exact confusion this option exists to expose.
    """
    return {"per100": "%/100 K.h", "perSD": "%/SD",
            "perP95": "%/p95-day"}.get(UNIT_NAME, "%/unit")


def heterogeneity(res: pd.DataFrame, agg: dict | None = None) -> dict:
    """Test the null that every country shares one slope.

    R5's headline is that the response is not one number. The first version of
    this function argued that with Cochran's Q and I-squared computed from the
    per-country standard errors alone, which assumes the country estimates are
    INDEPENDENT. They are not: they come from a single joint regression, and
    its Driscoll-Kraay covariance is built to be robust to exactly the
    cross-sectional dependence that makes them correlated -- countries share
    the global date effect and share weather on the same day. Using only the
    diagonal is therefore not a conservative simplification; it can push Q in
    either direction, depending on the sign of the covariances.

    Given the full covariance Sigma of the country aggregates, the test becomes
    the GLS form of Q,

        mu   = (1' Sigma^-1 theta) / (1' Sigma^-1 1)
        Q    = (theta - mu 1)' Sigma^-1 (theta - mu 1)   ~ chi2(C - 1)

    which reduces to the classical Q when Sigma is diagonal. Both are reported:
    `Q` is the covariance-consistent statistic used for inference, and
    `Q_indep` is the independence version, kept so the size of the difference
    is visible rather than asserted.

    I-squared has no standard definition under correlated estimates. It is
    reported from the GLS Q by the usual (Q - df)/Q formula and labelled as
    descriptive; the test itself is the Wald statistic.
    """
    out = {}
    for tag, est, se, mask in (("heat", "heat", "heat_se", "heat_ok"),
                               ("cold", "cold", "cold_se", "cold_ok"),
                               ("diff", "diff", "diff_se", "both")):
        if est not in res.columns or mask not in res.columns:
            continue
        sdf = res[res[mask].fillna(False)].dropna(subset=[est, se])
        if len(sdf) < 3:
            continue
        y = 100 * sdf[est].to_numpy(float)
        v = (100 * sdf[se].to_numpy(float)) ** 2
        w = 1.0 / v
        mu_i = float((w * y).sum() / w.sum())
        Q_i = float((w * (y - mu_i) ** 2).sum())
        df = len(sdf) - 1
        c = w.sum() - (w ** 2).sum() / w.sum()
        rec = {"k": len(sdf), "df": df, "Q_indep": Q_i,
               "Q_indep_p": float(stats.chi2.sf(Q_i, df)),
               "fe_pooled": mu_i,
               "tau": float(np.sqrt(max(0.0, (Q_i - df) / c))) if c > 0 else np.nan}

        Q, used_cov = Q_i, False
        if agg and tag in agg:
            a = agg[tag]
            # Line the covariance up with the countries that passed the screen,
            # in the order the table holds them, so theta and Sigma cannot be
            # silently mismatched.
            pos = {c_: i for i, c_ in enumerate(a["countries"])}
            idx = [pos[c_] for c_ in sdf.country if c_ in pos]
            if len(idx) == len(sdf):
                th = 100 * a["theta"][idx]
                S = 1e4 * a["Sigma"][np.ix_(idx, idx)]
                # Driscoll-Kraay covariances are not guaranteed positive
                # definite in finite samples; project onto the PSD cone and
                # use a pseudo-inverse rather than failing or, worse, silently
                # returning a negative quadratic form.
                ev, U = np.linalg.eigh((S + S.T) / 2)
                keep = ev > max(ev.max(), 0) * 1e-10
                Si = (U[:, keep] / ev[keep]) @ U[:, keep].T
                one = np.ones(len(th))
                den = float(one @ Si @ one)
                if den > 0:
                    mu_g = float(one @ Si @ th) / den
                    r = th - mu_g
                    Q = float(r @ Si @ r)
                    rec["fe_pooled_gls"] = mu_g
                    rec["df"] = df = int(keep.sum()) - 1
                    rec["n_eig_kept"] = int(keep.sum())
                    used_cov = True
        rec.update({"Q": Q, "p": float(stats.chi2.sf(Q, df)),
                    "I2": max(0.0, 100 * (Q - df) / Q) if Q > 0 else 0.0,
                    "cov_consistent": used_cov})
        out[tag] = rec
    return out


def screen(p: pd.DataFrame) -> pd.DataFrame:
    s = p.groupby("country").agg(
        n_hot=("w_SCDH24", lambda x: int((x >= MIN_SCDH).sum())),
        n_cold=("w_HDH15", lambda x: int((x >= MIN_HDH).sum()))).reset_index()
    s["heat_ok"] = s.n_hot >= MIN_DAYS
    s["cold_ok"] = s.n_cold >= MIN_DAYS
    s["both"] = s.heat_ok & s.cold_ok
    return s


def estimate(sector: str, ex: pd.DataFrame, naive: bool = False,
             verbose: bool = True, years: set[int] | None = None
             ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Screen, fit and shrink one sector. Returns (per-country table, EB, screen).

    `years` restricts the estimation window. It exists for the COVID diagnostic:
    2020 is the only year in the sample containing a global policy shock, so
    re-estimating on 2019+2021 alone is an assumption-free test of whether the
    pandemic drives the result -- assumption-free in the sense that it needs no
    policy covariate, at the cost of a third of the sample.
    """
    em = load_sector(sector)
    p = em.merge(ex, on=["country", "date"], how="inner")
    p = p[p.v > 0].copy()
    if years is not None:
        p = p[p.date.dt.year.isin(years)].copy()
    p["y"] = np.log(p.v)
    scr = screen(p)

    if verbose:
        print(f"    {p.country.nunique()} countries, {len(p):,} country-days; "
              f"heat side {int(scr.heat_ok.sum())}, cold side "
              f"{int(scr.cold_ok.sum())}, both {int(scr.both.sum())}")
        drop_h = scr.loc[~scr.heat_ok, "country"].tolist()
        drop_c = scr.loc[~scr.cold_ok, "country"].tolist()
        if drop_h:
            print(f"    no heat side: {', '.join(drop_h)}")
        if drop_c:
            print(f"    no cold side: {', '.join(drop_c)}")

    if naive:
        rows = []
        for c, g in p.groupby("country"):
            r = fit_country(g)
            if r is None:
                continue
            r["country"] = c
            rows.append(r)
        res, info = pd.DataFrame(rows), {}
    else:
        res, info = fit_joint(p, scr)
        if verbose:
            print(f"    {info['n_obs']:,} obs, {info['n_par']} slope "
                  f"parameters, within R2 = {info['r2']:.4f}")
    res = res.merge(scr, on="country", how="left")
    # attrs rather than a fourth return value: every caller unpacks three, and
    # the covariance is only needed by heterogeneity().
    res.attrs["agg"] = info.get("agg", {})
    if res.empty:
        return res, {}, scr

    for tag in ("heat", "cold", "diff"):
        res[f"{tag}_pct"] = 100 * res[tag]

    eb = {}
    for tag, mask in (("heat", res.heat_ok), ("cold", res.cold_ok),
                      ("diff", res.both)):
        sub = res[mask.fillna(False) & res[tag].notna()]
        if len(sub) < 4:
            continue
        e = empirical_bayes(sub[tag].values, sub[tag + "_se"].values)
        e["n"] = len(sub)
        eb[tag] = e
        for k, col in (("post", f"{tag}_eb"), ("post_se", f"{tag}_eb_se"),
                       ("shrink", f"{tag}_shrink")):
            res.loc[sub.index, col] = e[k]
        res[f"{tag}_eb_pct"] = 100 * res[f"{tag}_eb"]
        res[f"{tag}_eb_lo"] = 100 * (res[f"{tag}_eb"] - 1.96 * res[f"{tag}_eb_se"])
        res[f"{tag}_eb_hi"] = 100 * (res[f"{tag}_eb"] + 1.96 * res[f"{tag}_eb_se"])

    # per-own-SD version, a different and equally legitimate question
    res["heat_perSD_pct"] = 100 * res["heat"] * res["sd_scdh"] / SCALE
    res["cold_perSD_pct"] = 100 * res["cold"] * res["sd_hdh"] / SCALE
    res["diff_perSD_pct"] = res["heat_perSD_pct"] - res["cold_perSD_pct"]
    res["sector"] = sector
    return res, eb, scr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C", "all"])
    ap.add_argument("--sector", default="Power")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--no-placebo", action="store_true")
    ap.add_argument("--unit", default="per100",
                    choices=["per100", "perSD", "perP95"],
                    help="unit for the two sides; the country screen is "
                         "unaffected")
    ap.add_argument("--naive", action="store_true",
                    help="Per-country fits without a global date effect. "
                         "Retained only to reproduce the placebo failure.")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print(f"R5: observed thermal carbon-response asymmetry -- {args.sector}")
    if args.naive:
        print("*** NAIVE specification: no global date effect. Documented as a")
        print("*** placebo failure; never used for reported estimates.")
    print("=" * 78)

    ex = pd.read_parquet(EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"],
            "all": ["A", "B", "C"]}[args.tier]
    ex = ex[ex.tier.isin(keep)][["country", "date", "w_SCDH24", "w_HDH15"]]

    global UNIT_C, UNIT_H, UNIT_NAME
    UNIT_NAME = args.unit
    if args.unit != "per100":
        def _unit(x: np.ndarray) -> float:
            if args.unit == "perSD":
                return float(np.nanstd(x))
            pos = x[np.isfinite(x) & (x > 0)]
            return float(np.quantile(pos, 0.95)) if pos.size >= 100 else np.nan
        UNIT_C = _unit(ex["w_SCDH24"].to_numpy(float))
        UNIT_H = _unit(ex["w_HDH15"].to_numpy(float))
        if not (np.isfinite(UNIT_C) and np.isfinite(UNIT_H)):
            raise SystemExit(f"cannot form the {args.unit} unit on this tier")
        print(f"\n    unit: {args.unit} -- cooling {UNIT_C:.2f} K.h, "
              f"heating {UNIT_H:.2f} K.h. The identification screen still uses "
              f"raw\n    degree-hours, so the country set does not move with "
              f"the unit.")

    print(f"\n[1] Tier {args.tier}, identification screen "
          f"(>= {MIN_DAYS} days with SCDH24 >= {MIN_SCDH:.0f} / "
          f"HDH15 >= {MIN_HDH:.0f} K.h)")
    if not args.naive:
        print("    FE: country x year-month, global date, country x day-of-week")
        print("    VCV: Driscoll-Kraay, bandwidth 14 days")
    res, eb, scr = estimate(args.sector, ex, naive=args.naive)
    if res.empty:
        print("    nothing estimable; stopping")
        return

    print("\n[2] Empirical-Bayes shrinkage")
    for tag in ("heat", "cold", "diff"):
        if tag not in eb:
            continue
        e = eb[tag]
        print(f"    {tag:5s} n={e['n']:2d}  pooled mu = "
              f"{100 * e['mu']:+.3f} +/- {100 * e['mu_se']:.3f} {unit_label()}  "
              f"tau = {100 * e['tau']:.3f}  median shrinkage = "
              f"{np.median(e['shrink']):.2f}")

    # ---- negative-control floor ----------------------------------------------------
    floor = {}
    placebo_res = {}          # per-country tables, reused when writing [3b]
    if not args.no_placebo:
        print("\n[3] Negative-control sectors, identical specification")
        print(f"    {'sector':20s}{'heat':>12}{'cold':>12}{'diff':>12}")
        for sec in PLACEBOS:
            if sec == args.sector:
                continue
            pres, peb, _ = estimate(sec, ex, naive=args.naive, verbose=False)
            floor[sec] = peb
            placebo_res[sec] = pres
            cells = []
            for tag in ("heat", "cold", "diff"):
                if tag in peb:
                    z = peb[tag]["mu"] / peb[tag]["mu_se"]
                    cells.append(f"{100 * peb[tag]['mu']:+7.2f}"
                                 f"({z:+.1f})")
                else:
                    cells.append("      --")
            print(f"    {sec:20s}{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}")
        print(f"    values are pooled mu ({unit_label()}) with z in brackets")
        mags = [abs(100 * v[t]["mu"]) for v in floor.values()
                for t in ("heat", "cold", "diff") if t in v]
        if mags:
            print(f"    -> detection floor: any coefficient below about "
                  f"{max(mags):.1f} {unit_label()}")
            print("       is within the range these no-cooling sectors produce")

    # ---- all sectors, and the COVID window diagnostic ---------------------
    sectors_tab, covid_tab = [], []
    if not args.no_placebo:
        print("\n[3b] All five sectors, identical specification")
        print(f"    {'sector':20s}{'cooling':>13}{'heating':>13}"
              f"{'difference':>13}   status")
        status = {
            "Power": "interpreted",
            "Industry": "not independent: built from national electricity",
            "Ground Transport": "negative control",
            "Aviation": "negative control (real cold channel)",
            "Residential": "CIRCULAR: built from heating degree-days",
        }
        for sec in ALL_SECTORS:
            sres = None
            if sec == args.sector:
                seb = eb
            elif sec in placebo_res:
                sres, seb = placebo_res[sec], floor[sec]
            else:
                sres, seb, _ = estimate(sec, ex, naive=args.naive, verbose=False)
            # Write the per-country table for EVERY sector, not just the one
            # named by --sector. 24_r6_weight_competition.py reads the Aviation
            # file for its placebo check, and previously that file only existed
            # if 22 had been run a second time with --sector Aviation -- an
            # ordering dependency that failed silently as "slopes not found".
            if sres is not None and not sres.empty:
                TAB_DIR.mkdir(parents=True, exist_ok=True)
                sres.to_csv(
                    TAB_DIR / f"T6_thermal_asymmetry_"
                              f"{sec.replace(' ', '_')}_tier{args.tier}"
                              f"{'' if args.unit == 'per100' else '_' + args.unit}.csv",
                    index=False)
            row = {"sector": sec}
            cells = []
            for tag in ("heat", "cold", "diff"):
                if tag in seb:
                    z = seb[tag]["mu"] / seb[tag]["mu_se"]
                    row[tag] = 100 * seb[tag]["mu"]
                    row[tag + "_se"] = 100 * seb[tag]["mu_se"]
                    row[tag + "_z"] = z
                    cells.append(f"{100 * seb[tag]['mu']:+7.2f}({z:+.1f})")
                else:
                    cells.append("      --")
            het_src = res if sec == args.sector else sres
            if het_src is not None and not het_src.empty:
                for tag, h in heterogeneity(
                        het_src, het_src.attrs.get("agg")).items():
                    row[f"{tag}_k"] = h["k"]
                    row[f"{tag}_Q"] = h["Q"]
                    row[f"{tag}_Q_df"] = h["df"]
                    row[f"{tag}_Q_p"] = h["p"]
                    row[f"{tag}_I2"] = h["I2"]
                    row[f"{tag}_tau"] = h["tau"]
                    # Carried into the table so a reader can see how much the
                    # covariance correction moved Q without rerunning.
                    row[f"{tag}_Q_indep"] = h["Q_indep"]
                    row[f"{tag}_cov_consistent"] = h["cov_consistent"]
            sectors_tab.append(row)
            print(f"    {sec:20s}{cells[0]:>13}{cells[1]:>13}{cells[2]:>13}"
                  f"   {status.get(sec, '')}")

        print("\n[3b-i] Is the between-country spread larger than sampling error?")
        hh = heterogeneity(res, res.attrs.get("agg"))
        print(f"    {'quantity':12s}{'k':>4}{'Q':>10}{'df':>5}{'p':>11}"
              f"{'I2 %':>8}{'Q_indep':>10}{'cov':>5}")
        for tag, lab in (("heat", "cooling"), ("cold", "heating"),
                         ("diff", "difference")):
            if tag in hh:
                h = hh[tag]
                print(f"    {lab:12s}{h['k']:4d}{h['Q']:10.1f}{h['df']:5d}"
                      f"{h['p']:11.2e}{h['I2']:8.1f}{h['Q_indep']:10.1f}"
                      f"{'yes' if h['cov_consistent'] else 'NO':>5}")
        print("    Q is the GLS statistic using the full cross-country")
        print("    covariance; Q_indep is the independence version, shown so")
        print("    the size of the correction is visible.")
        print("    Q tests the null of one shared slope; I2 is the share of the")
        print("    observed spread that sampling error cannot explain. Computed")
        print("    on the unshrunk estimates, since shrinkage is designed to")
        print("    reduce exactly this spread.")

        print("\n[3c] COVID diagnostic: re-estimating on sub-windows")
        print(f"    {'window':16s}{'cooling':>13}{'heating':>13}"
              f"{'difference':>13}")
        # Only --sector here. The all-sector version is 15 refits, which is
        # slower than the rest of this script put together, so it lives in
        # 35_covid_windows.py, which caches per (sector, window) cell and can
        # be resumed. That script owns T9; this block is the quick look.
        y0, y1 = int(ex.date.dt.year.min()), int(ex.date.dt.year.max())
        keep_yrs = {y for y in range(y0, y1 + 1) if y != 2020}
        for label, yrs in ((f"{y0}-{y1}", None), ("drop 2020", keep_yrs),
                           ("2020 only", {2020})):
            web = eb if yrs is None else estimate(
                args.sector, ex, naive=args.naive, verbose=False, years=yrs)[1]
            row = {"sector": args.sector, "window": label}
            cells = []
            for tag in ("heat", "cold", "diff"):
                if tag in web:
                    z = web[tag]["mu"] / web[tag]["mu_se"]
                    row[tag] = 100 * web[tag]["mu"]
                    row[tag + "_se"] = 100 * web[tag]["mu_se"]
                    row[tag + "_z"] = z
                    cells.append(f"{100 * web[tag]['mu']:+7.2f}({z:+.1f})")
                else:
                    cells.append("      --")
            covid_tab.append(row)
            print(f"    {label:16s}{cells[0]:>13}{cells[1]:>13}{cells[2]:>13}")
        print("    if 'drop 2020' agrees with the full window, the pandemic is")
        print("    not driving the estimate and a policy-stringency covariate")
        print("    would be a refinement rather than a correction")

    # ---- why the ratio is not used ---------------------------------------
    print("\n[4] Why the difference and not the ratio")
    sub = res[res.both.fillna(False)].copy()
    sub["ratio"] = sub["heat"] / sub["cold"]
    sub["cold_t"] = sub["cold"] / sub["cold_se"]
    weak = sub[sub.cold_t.abs() < 2]
    print(f"    ratio range across {len(sub)} countries: "
          f"{sub.ratio.min():+.2f} to {sub.ratio.max():+.2f}")
    print(f"    countries whose denominator is within 2 SE of zero: "
          f"{len(weak)}/{len(sub)}"
          + (f"  ({', '.join(weak.country.head(6))})" if len(weak) else ""))
    print("    -> the ratio is not reported as an index; the difference is")
    res = res.merge(sub[["country", "ratio", "cold_t"]], on="country", how="left")

    # ---- verdict ----------------------------------------------------------
    print("\n[5] Observed asymmetry, shrunk (%, per 100 K.h)")
    print(f"    {'country':22s}{'heat':>9}{'cold':>9}{'diff':>9}"
          f"{'95% CI of diff':>20}")
    sub = res[res.both.fillna(False)].sort_values("diff_eb_pct", ascending=False)
    n_pos = n_neg = 0
    for _, r in sub.iterrows():
        star = ""
        if r.diff_eb_lo > 0:
            star, n_pos = "  heat > cold", n_pos + 1
        elif r.diff_eb_hi < 0:
            star, n_neg = "  cold > heat", n_neg + 1
        print(f"    {r.country:22s}{r.heat_eb_pct:9.3f}{r.cold_eb_pct:9.3f}"
              f"{r.diff_eb_pct:9.3f}"
              f"   [{r.diff_eb_lo:6.3f}, {r.diff_eb_hi:6.3f}]{star}")
    print(f"\n    heat > cold in {n_pos}/{len(sub)} countries, "
          f"cold > heat in {n_neg}/{len(sub)}, "
          f"indistinguishable in {len(sub) - n_pos - n_neg}")
    if "diff" in eb:
        e = eb["diff"]
        z = e["mu"] / e["mu_se"]
        print(f"    pooled difference {100 * e['mu']:+.3f} "
              f"+/- {100 * e['mu_se']:.3f} {unit_label()}  (z = {z:+.2f}, "
              f"p = {2 * (1 - stats.norm.cdf(abs(z))):.4f})")
    # Window taken from the data, not hardcoded: the panel now extends past 2021
    # and a fixed string here would misdate every printed result.
    yr0, yr1 = int(ex.date.dt.year.min()), int(ex.date.dt.year.max())
    print(f"\n    WORDING: this is the asymmetry OBSERVED in {yr0}-{yr1} under the")
    print("    generation mix of those years. It is not a projection of the net")
    print("    carbon effect of warming, which also requires changes in the")
    print("    frequency of hot and cold days, grid decarbonisation and")
    print("    technology turnover.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    # per100 keeps the published filenames; the scale-free runs sit beside them
    suf = "" if args.unit == "per100" else f"_{args.unit}"
    res.to_parquet(OUT_DIR / f"r5_asymmetry_tier{args.tier}{suf}.parquet",
                   index=False)
    res.to_csv(TAB_DIR / f"T6_thermal_asymmetry_"
               f"{args.sector.replace(chr(32), chr(95))}_tier{args.tier}{suf}.csv",
               index=False)
    if sectors_tab:
        pd.DataFrame(sectors_tab).to_csv(
            TAB_DIR / f"T7_sector_comparison_tier{args.tier}{suf}.csv",
            index=False)
    # The COVID sub-windows are printed above as a quick look but not written:
    # 35_covid_windows.py owns T8, and a Power-only file written here would sit
    # in the results tree looking like the real table.
    print(f"\n    R5/tables/T6_thermal_asymmetry_"
          f"{args.sector.replace(' ', '_')}_tier{args.tier}{suf}.csv")

    if not args.no_figure and args.sector == "Power":
        mags = [abs(100 * v[t]['mu']) for v in floor.values()
                for t in ('heat', 'cold', 'diff') if t in v]
        make_figure(res, args.tier, max(mags) if mags else None,
                    100 * eb['diff']['mu'] if 'diff' in eb else None,
                    sectors_tab, covid_tab)
    print(f"\nDone in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------
def make_figure(res: pd.DataFrame, tier: str, floor: float | None = None,
                pooled: float | None = None,
                sectors_tab: list[dict] | None = None,
                covid_tab: list[dict] | None = None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.4))
    C_H, C_C, C_D = "#c0392b", "#2980b9", "#7d3c98"

    # (a) raw versus shrunk, both sides
    ax = axes[0, 0]
    d = res[res.heat_ok.fillna(False)].sort_values("heat_pct")
    ax.scatter(d.heat_pct, range(len(d)), s=14, facecolors="none",
               edgecolors=C_H, lw=0.8, label="heat, raw")
    ax.scatter(d.heat_eb_pct, range(len(d)), s=16, color=C_H, label="heat, shrunk")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.heat_pct, r.heat_eb_pct], [i, i], color=C_H, lw=0.6, alpha=0.5)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.country, fontsize=5.5)
    ax.set_xlabel(f"Power response to cooling load ({unit_label()})")
    ax.set_title("a  Raw and empirical-Bayes country slopes, cooling side")
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    # (b) the two sides against each other
    ax = axes[0, 1]
    d = res[res.both.fillna(False)]
    ax.errorbar(d.cold_eb_pct, d.heat_eb_pct,
                xerr=1.96 * 100 * d.cold_eb_se, yerr=1.96 * 100 * d.heat_eb_se,
                fmt="o", ms=4, lw=0.7, color=C_D, ecolor="#c8b6d6", capsize=0)
    span = max(d.cold_eb_pct.max(), d.heat_eb_pct.max()) - \
        min(d.cold_eb_pct.min(), d.heat_eb_pct.min())
    lim = [min(d.cold_eb_pct.min(), d.heat_eb_pct.min()) - 0.10 * span,
           max(d.cold_eb_pct.max(), d.heat_eb_pct.max()) + 0.10 * span]
    ax.plot(lim, lim, ls="--", color="k", lw=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    for _, r in d.iterrows():
        ax.annotate(r.country, (r.cold_eb_pct, r.heat_eb_pct), fontsize=5,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"response to heating load ({unit_label()})")
    ax.set_ylabel(f"response to cooling load ({unit_label()})")
    ax.set_title("b  Cooling against heating slope, by country")

    # (c) the difference, the primary quantity
    ax = axes[1, 0]
    d = res[res.both.fillna(False)].sort_values("diff_eb_pct")
    yy = np.arange(len(d))
    if floor:
        ax.axvspan(-floor, floor, color="#bdc3c7", alpha=0.40, lw=0,
                   label=f"detection floor (±{floor:.1f})")
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.hlines(yy, d.diff_eb_lo, d.diff_eb_hi, color=C_D, lw=1.2)
    ax.scatter(d.diff_eb_pct, yy, s=16, color=C_D, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    if pooled is not None:
        ax.axvline(pooled, color=C_D, lw=0.9, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels(d.country, fontsize=5.5)
    ax.set_xlabel(f"cooling minus heating response ({unit_label()})")
    ax.set_title("c  Cooling − heating difference, empirical-Bayes "
                 "intervals")

    # (d) why the ratio was rejected
    ax = axes[1, 1]
    d = res[res.both.fillna(False) & res.ratio.notna()]
    ax.scatter(d.cold_pct, d.ratio, s=18, color="#7f8c8d",
               edgecolors="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="#c0392b", lw=0.8, ls=":")
    for _, r in d.iterrows():
        if abs(r.ratio) > 3 or abs(r.cold_t) < 2:
            ax.annotate(r.country, (r.cold_pct, r.ratio), fontsize=5,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"heating-load response, the denominator ({unit_label()})")
    ax.set_ylabel("heat / cold ratio")
    ax.set_title("d  Cooling/heating ratio against its denominator")

    # (e) all five sectors -- which ones may be read at all
    ax = axes[0, 2]
    if sectors_tab:
        s = pd.DataFrame(sectors_tab).set_index("sector").reindex(ALL_SECTORS)
        yy = np.arange(len(s))[::-1]
        for tag, col, off, lab in (("heat", C_H, +0.22, "cooling load"),
                                   ("cold", C_C, 0.0, "heating load"),
                                   ("diff", C_D, -0.22, "difference")):
            if tag not in s.columns:
                continue
            ax.errorbar(s[tag], yy + off, xerr=1.96 * s[tag + "_se"], fmt="o",
                        ms=4, lw=0.9, color=col, ecolor=col, capsize=0,
                        label=lab, alpha=0.9)
        ax.axvline(0, color="k", lw=0.8)
        if floor:
            ax.axvspan(-floor, floor, color="#bdc3c7", alpha=0.40, lw=0)
        ax.set_yticks(yy)
        ax.set_yticklabels(
            ["Power\n(interpreted)", "Industry\n(not independent)",
             "Ground Transport\n(negative control)",
             "Aviation\n(negative control)",
             "Residential\n(CIRCULAR)"], fontsize=5.5)
        ax.set_xlabel(f"pooled response ({unit_label()})")
        ax.set_title("e  Pooled response by sector, identical specification")
        ax.legend(frameon=False, fontsize=6, loc="lower left")

    # (f) is the pandemic driving this?
    ax = axes[1, 2]
    if covid_tab:
        s = pd.DataFrame(covid_tab).set_index("window")
        yy = np.arange(len(s))[::-1]
        for tag, col, off, lab in (("heat", C_H, +0.20, "cooling load"),
                                   ("cold", C_C, 0.0, "heating load"),
                                   ("diff", C_D, -0.20, "difference")):
            if tag not in s.columns:
                continue
            ax.errorbar(s[tag], yy + off, xerr=1.96 * s[tag + "_se"], fmt="o",
                        ms=4.5, lw=1.0, color=col, ecolor=col, capsize=0,
                        label=lab)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels(s.index, fontsize=7)
        ax.set_xlabel(f"pooled response ({unit_label()})")
        ax.set_title("f  Pooled response by estimation window, Power")
        ax.legend(frameon=False, fontsize=6, loc="lower right")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Supplementary diagnostics for the observed thermal "
                 f"carbon-response asymmetry, Power sector, "
                 f"Tier {'A+B' if tier == 'AB' else tier}, "
                 f"{unit_label()}", y=1.0, fontsize=9.5)
    fig.tight_layout()
    # Figure 7 itself is drawn by 34_response_figures.py, in a scale-free
    # unit and around the heterogeneity result. What this function produces is
    # the diagnostic plate behind it: shrinkage, the joint distribution of the
    # two sides, the ratio instability that motivates using the difference, and
    # the COVID sub-windows.
    out = RESULTS_ROOT / "SI" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    name = (f"FigS11_asymmetry_diagnostics_tier{tier}"
            f"{'' if UNIT_NAME == 'per100' else '_' + UNIT_NAME}")
    for ext in SAVE_FORMATS:
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)
    print(f"    SI/figures/{name}  [{', '.join(SAVE_FORMATS)}]  {SAVE_DPI} dpi")


if __name__ == "__main__":
    main()
