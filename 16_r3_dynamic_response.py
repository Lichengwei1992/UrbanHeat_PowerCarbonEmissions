#!/usr/bin/env python
"""
16_r3_dynamic_response.py  --  R3 full: sectoral dynamic response (Fig. 4)
===========================================================================

Estimates the before/during/after response of national daily sectoral CO2 to
city-weighted thermal exposure, and reduces each lag profile to the five shape
indicators the conceptual framework asks for.

MODEL
-----
    ln E_(c,t) = mu_(c,ym) + lambda_t + nu_(c,dow)
                 + sum_(k=-L..+M) beta^H_k Heat_(c,t-k)
                 + sum_(k=-L..+M) beta^C_k Cold_(c,t-k) + eps

  mu_(c,ym)   country x year-month   season, trend, coverage, national policy
  lambda_t    global date            worldwide common shocks (COVID timing)
  nu_(c,dow)  country x day-of-week  country-specific weekly cycle
  k < 0       leads: parallel-trends check only, never interpreted as effects
  SE          two-way clustered on country and date

Heat and cold enter jointly so neither is absorbed into the other; the same
window is used for both so the profiles are directly comparable.

TWO EXPOSURE FORMS
------------------
  continuous  w_SCDH24 / w_HDH15 -- population-weighted cooling and heating
              degree-hours. Physically graded but highly autocorrelated, so
              individual lags are imprecise while the cumulative sum is stable.
  event       w_HW / w_CW -- share of urban population under a heatwave or
              cold wave. Closer to the original event-study framing and far
              less collinear across lags.

Both are reported: agreement between them is the real robustness check.

SHAPE INDICATORS
----------------
  onset_jump  beta(0) - beta(-1)                  discontinuity at onset
  peak_lag    argmax |beta(k)|, k >= 0            when the response peaks
  half_life   first k after the peak with |beta| <= half the peak
  rebound     sum of beta(k) over the late window  post-event overshoot
  pretrend    Wald chi2 on all leads jointly       parallel-trends test

FIXED EFFECTS are removed by alternating projections over an arbitrary number
of factors, with the surviving group means reported -- silent non-convergence
here would invalidate every coefficient.

INPUTS
------
  CarbonData/Near-real-time .../carbon-monitor-cities-*-v0325.csv
  outputs/exposure/country_daily_exposure.parquet

OUTPUTS
-------
  outputs/response/r3_lag_profiles.parquet
  outputs/response/r3_shape_indicators.parquet
  outputs/response/tables/*.csv
  outputs/figures/Fig4_sectoral_dynamic_response.png / .pdf

USAGE
-----
  python 16_r3_dynamic_response.py
  python 16_r3_dynamic_response.py --tier AB --leads 14 --lags 21
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
FIG_DIR = BASE / "outputs" / "figures"

# Figure output contract, identical across the project: three formats, 600 dpi,
# text kept as vector inside the SVG.
SAVE_FORMATS = ("png", "pdf", "svg")
SAVE_DPI = 600

SECTORS = ["Power", "Ground Transport", "Industry", "Aviation", "Residential"]
FORMS = {"continuous": ("w_SCDH24", "w_HDH15"),
         "event": ("w_HW", "w_CW")}
LATE_WINDOW = (8, 14)          # rebound is summed over this lag range


# --------------------------------------------------------------------------
def load_emissions() -> pd.DataFrame:
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
    return e.groupby(["country", "date", "sector"], as_index=False)["v"].sum()


def demean_factors(X: np.ndarray, factors: list[np.ndarray],
                   tol: float = 1e-10, max_iter: int = 300):
    """Within transformation over an arbitrary number of factors."""
    X = np.asarray(X, dtype="float64").copy()
    meta = [(f, np.bincount(f).astype(float), f.max() + 1) for f in factors]
    for _ in range(max_iter):
        prev = X.copy()
        for f, cnt, n in meta:
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(f, weights=X[:, j], minlength=n) / cnt)[f]
        if np.max(np.abs(X - prev)) < tol:
            break
    resid = max(max(abs(np.bincount(f, weights=X[:, j], minlength=n) / cnt).max()
                    for j in range(X.shape[1])) for f, cnt, n in meta)
    return X, resid


def cluster_scores(X, e, g):
    """Sum of X_i * e_i within each cluster of g -> (n_clusters, k)."""
    n = g.max() + 1
    Xe = X * e[:, None]
    S = np.zeros((n, X.shape[1]))
    for j in range(X.shape[1]):
        S[:, j] = np.bincount(g, weights=Xe[:, j], minlength=n)
    return S


def psd_correct(M: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Project a symmetric matrix onto the PSD cone by truncating eigenvalues.

    The Cameron-Gelbach-Miller two-way estimator M1 + M2 - M12 is a difference
    of PSD matrices and is therefore NOT guaranteed to be positive
    semi-definite. When it is not, variances can come out negative and Wald
    statistics can be negative -- which is what happened for
    Ground Transport / cold / event (Wald = -1.15). The standard remedy is to
    zero the negative eigenvalues; see Cameron, Gelbach & Miller (2011).
    """
    M = 0.5 * (M + M.T)
    w, Q = np.linalg.eigh(M)
    n_neg = int((w < 0).sum())
    worst = float(w.min() / max(abs(w).max(), 1e-300)) if len(w) else 0.0
    if n_neg:
        M = (Q * np.clip(w, 0, None)) @ Q.T
    return M, n_neg, worst


def twoway_vcov(X, e, g1, g2, correct: bool = True):
    """Two-way clustered VCV with optional PSD projection.

    Returns (V, diagnostics). `diagnostics` records whether the raw meat was
    indefinite, so a silently patched matrix can never be mistaken for a clean
    one in the output tables.
    """
    inv = np.linalg.inv(X.T @ X)
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]
    S1, S2, S12 = (cluster_scores(X, e, g) for g in (g1, g2, g12))
    M_raw = S1.T @ S1 + S2.T @ S2 - S12.T @ S12
    M, n_neg, worst = (psd_correct(M_raw) if correct else (M_raw, 0, 0.0))
    return inv @ M @ inv, {"vcov_neg_eig": n_neg, "vcov_min_eig_ratio": worst}


def wcb_pvalue(X, y, b, cvec, g_boot, g1, g2, B: int = 999,
               seed: int = 20260813) -> float:
    """Restricted wild cluster bootstrap p-value for H0: c'beta = 0.

    Rademacher weights are drawn per bootstrap cluster (country), and the null
    is imposed before resampling, which is the variant that behaves best with
    few clusters (MacKinnon & Webb 2017; Roodman et al. 2019). With 25
    countries the analytic normal approximation is not trustworthy on its own.

    The standard error of c'beta is recomputed in every replication, but only
    as a scalar: with u = (X'X)^-1 c fixed, c'Vc = sum_g (sum_(i in g) z_i e_i)^2
    where z = X u, so no k x k matrix is ever rebuilt.
    """
    rng = np.random.default_rng(seed)
    XtX = X.T @ X
    A = np.linalg.inv(XtX)
    u = A @ cvec
    z = X @ u

    # impose the null: beta_r = beta - A c (c' A c)^-1 c' beta
    denom = float(cvec @ A @ cvec)
    b_r = b - A @ cvec * (float(cvec @ b) / denom)
    fitted_r = X @ b_r
    e_r = y - fitted_r

    def se_of_c(resid):
        s1 = np.bincount(g1, weights=z * resid, minlength=g1.max() + 1)
        s2 = np.bincount(g2, weights=z * resid, minlength=g2.max() + 1)
        s12 = np.bincount(g12v, weights=z * resid, minlength=g12v.max() + 1)
        v = (s1 ** 2).sum() + (s2 ** 2).sum() - (s12 ** 2).sum()
        return np.sqrt(v) if v > 0 else np.nan

    g12v = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]

    e_obs = y - X @ b
    se_obs = se_of_c(e_obs)
    if not np.isfinite(se_obs) or se_obs == 0:
        return np.nan
    t_obs = abs(float(cvec @ b) / se_obs)

    n_g = g_boot.max() + 1
    hits = 0
    for _ in range(B):
        w = rng.choice(np.array([-1.0, 1.0]), size=n_g)[g_boot]
        y_star = fitted_r + w * e_r
        b_star = np.linalg.solve(XtX, X.T @ y_star)
        e_star = y_star - X @ b_star
        se_star = se_of_c(e_star)
        if not np.isfinite(se_star) or se_star == 0:
            continue
        if abs(float(cvec @ b_star) / se_star) >= t_obs:
            hits += 1
    return (hits + 1) / (B + 1)


def shape_indicators(lags: np.ndarray, beta: np.ndarray, V: np.ndarray,
                     idx: np.ndarray, leads: int) -> dict:
    """Reduce a lag profile to interpretable shape statistics."""
    post = lags >= 0
    bp, lp = beta[post], lags[post]
    peak_i = int(np.argmax(np.abs(bp)))
    peak_lag, peak_b = int(lp[peak_i]), float(bp[peak_i])

    half = np.nan
    for k in range(peak_i + 1, len(bp)):
        if abs(bp[k]) <= 0.5 * abs(peak_b):
            half = int(lp[k]) - peak_lag
            break

    onset = np.nan
    if (lags == 0).any() and (lags == -1).any():
        onset = float(beta[lags == 0][0] - beta[lags == -1][0])

    late = (lags >= LATE_WINDOW[0]) & (lags <= LATE_WINDOW[1])
    reb = float(beta[late].sum()) if late.any() else np.nan

    cum_i = idx[post]
    cum = float(beta[post].sum())
    cum_se = float(np.sqrt(V[np.ix_(cum_i, cum_i)].sum()))

    lead_i = idx[lags < 0]
    if len(lead_i):
        bl = beta[lags < 0]
        Vl = V[np.ix_(lead_i, lead_i)]
        # V has already been projected onto the PSD cone, but a near-singular
        # block can still produce a numerically negative quadratic form, so use
        # the pseudo-inverse and clip at zero rather than emitting a negative
        # chi-square.
        wald = float(bl @ np.linalg.pinv(Vl) @ bl)
        wald = max(wald, 0.0)
        pv = float(1 - stats.chi2.cdf(wald, len(lead_i)))
    else:
        wald, pv = np.nan, np.nan

    return {"cum": cum, "cum_se": cum_se, "cum_t": cum / cum_se if cum_se else np.nan,
            "peak_lag": peak_lag, "peak_beta": peak_b, "half_life": half,
            "onset_jump": onset, "rebound": reb,
            "pretrend_wald": wald, "pretrend_p": pv}


def fit_sector(d: pd.DataFrame, heat: str, cold: str,
               leads: int, lags: int, wcb_reps: int = 999):
    d = d.sort_values(["country", "date"]).reset_index(drop=True).copy()
    names, blocks = [], []
    for tag, col in (("H", heat), ("C", cold)):
        g = d.groupby("country")[col]
        for s in range(-leads, lags + 1):
            nm = f"{tag}{s:+d}"
            d[nm] = g.shift(s)
            names.append(nm)
        blocks.append(tag)
    d = d.dropna(subset=names + ["y"]).reset_index(drop=True)
    if len(d) < 500:
        return None

    M = np.column_stack([d[names].to_numpy(dtype="float64"), d["y"].to_numpy()])
    f_cym = pd.factorize(d["country"] + "_" + d["date"].dt.to_period("M").astype(str))[0]
    f_date = pd.factorize(d["date"])[0]
    f_cdow = pd.factorize(d["country"] + "_" + d["date"].dt.dayofweek.astype(str))[0]
    Z, conv = demean_factors(M, [f_cym, f_date, f_cdow])
    X, y = Z[:, :-1], Z[:, -1]

    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-10
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    f_ctry = pd.factorize(d["country"])[0]
    V, vdiag = twoway_vcov(X, e, f_ctry, f_date)
    se = np.sqrt(np.clip(np.diag(V), 0, None))

    nk = leads + lags + 1
    prof, shapes = [], {}
    for bi, tag in enumerate(blocks):
        sl = slice(bi * nk, (bi + 1) * nk)
        lag_arr = np.arange(-leads, lags + 1)
        idx = np.arange(bi * nk, (bi + 1) * nk)
        exp_name = "heat" if tag == "H" else "cold"
        prof.append(pd.DataFrame({"exposure": exp_name, "lag": lag_arr,
                                  "beta": b[sl], "se": se[sl]}))
        sh = shape_indicators(lag_arr, b[sl], V, idx, leads)
        # wild cluster bootstrap on the cumulative post-onset effect
        cvec = np.zeros(X.shape[1])
        cvec[idx[lag_arr >= 0]] = 1.0
        sh["cum_p_wcb"] = wcb_pvalue(X, y, b, cvec, f_ctry, f_ctry, f_date,
                                     B=wcb_reps)
        sh.update(vdiag)
        shapes[exp_name] = sh
    return {"profile": pd.concat(prof, ignore_index=True), "shapes": shapes,
            "n": len(d), "r2": 1 - e.var() / y.var(), "conv": conv,
            "n_country": d.country.nunique()}


# --------------------------------------------------------------------------
def make_figure(prof: pd.DataFrame, shapes: pd.DataFrame, form: str, tier: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # svg.fonttype "none" keeps SVG text as editable text rather than outlines,
    # matching the other figure scripts so the whole set behaves the same way
    # in a vector editor.
    plt.rcParams.update({"savefig.dpi": SAVE_DPI, "svg.fonttype": "none"})

    secs = [s for s in SECTORS if s in prof.sector.unique()]
    fig, axes = plt.subplots(2, len(secs), figsize=(3.1 * len(secs), 6.2),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for j, sec in enumerate(secs):
        for i, exp in enumerate(["heat", "cold"]):
            ax = axes[i, j]
            p = prof[(prof.sector == sec) & (prof.exposure == exp)].sort_values("lag")
            lo, hi = 100 * (p.beta - 1.96 * p.se), 100 * (p.beta + 1.96 * p.se)
            ax.fill_between(p.lag, lo, hi, alpha=0.22,
                            color="#c0392b" if exp == "heat" else "#2471a3", lw=0)
            ax.plot(p.lag, 100 * p.beta, lw=1.7,
                    color="#c0392b" if exp == "heat" else "#2471a3")
            ax.axhline(0, color="0.4", lw=0.8)
            ax.axvline(-0.5, color="0.4", lw=0.8, ls="--")
            s = shapes[(shapes.sector == sec) & (shapes.exposure == exp)]
            if len(s):
                r = s.iloc[0]
                ax.set_title(f"{sec}" if i == 0 else "", fontsize=10)
                ax.text(0.03, 0.94,
                        f"Σβ={100*r.cum:+.2f}% (t={r.cum_t:+.1f})\n"
                        f"peak k={int(r.peak_lag)}  pre-p={r.pretrend_p:.2f}",
                        transform=ax.transAxes, va="top", fontsize=7.2)
            if j == 0:
                ax.set_ylabel(("Heat" if exp == "heat" else "Cold") +
                              " response (%)", fontsize=9)
            if i == 1:
                ax.set_xlabel("lag k (days)", fontsize=9)
            ax.tick_params(labelsize=8)
    fig.suptitle(f"Sectoral CO$_2$ response to city-weighted thermal exposure "
                 f"({form} form, Tier {tier})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Same output contract as every other figure in the project: PNG, PDF and
    # SVG at 600 dpi. This script previously wrote PNG+PDF at 200 dpi, which is
    # below submission resolution and inconsistent with the rest of the set.
    for ext in SAVE_FORMATS:
        fig.savefig(FIG_DIR / f"Fig4_sectoral_dynamic_response_{form}_{tier}.{ext}",
                    dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Fig4_sectoral_dynamic_response_{form}_{tier}  "
          f"[{', '.join(SAVE_FORMATS)}]  {SAVE_DPI} dpi")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="A", choices=["A", "AB", "C", "all"])
    ap.add_argument("--leads", type=int, default=7)
    ap.add_argument("--lags", type=int, default=14)
    ap.add_argument("--wcb", type=int, default=999,
                    help="Wild cluster bootstrap replications (0 to skip).")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 80)
    print(f"R3 full: sectoral dynamic response  |  leads -{args.leads} .. lags +{args.lags}")
    print("=" * 80)

    em = load_emissions()
    ex = pd.read_parquet(EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"], "all": ["A", "B", "C"]}[args.tier]
    ex = ex[ex.tier.isin(keep)]
    print(f"\nTier {args.tier}: {ex.country.nunique()} countries")

    panel = em.merge(ex, on=["country", "date"], how="inner")
    panel = panel[panel.v > 0].copy()
    panel["y"] = np.log(panel.v)

    all_prof, all_shape = [], []
    for form, (heat, cold) in FORMS.items():
        if heat not in panel.columns or cold not in panel.columns:
            print(f"\n[{form}] missing {heat}/{cold}, skipped")
            continue
        p = panel.copy()
        for c in (heat, cold):                      # standardise -> per SD
            sd = p[c].std()
            p[c] = p[c] / (sd if sd > 0 else 1)
        print(f"\n[{form}]  heat={heat}  cold={cold}")
        print(f"    {'sector':18s}{'n':>8s}{'R2':>8s}"
              f"{'heat Σβ':>11s}{'t':>7s}{'p-wcb':>8s}{'peak':>6s}"
              f"{'pre-p':>7s}{'negeig':>8s}")
        for sec in SECTORS:
            sub = p[p.sector == sec]
            if len(sub) < 1000:
                continue
            r = fit_sector(sub, heat, cold, args.leads, args.lags,
                           wcb_reps=args.wcb)
            if r is None:
                continue
            pr = r["profile"]; pr["sector"] = sec; pr["form"] = form
            all_prof.append(pr)
            for exp, sh in r["shapes"].items():
                all_shape.append({"sector": sec, "exposure": exp, "form": form,
                                  "n": r["n"], "r2": r["r2"], "conv": r["conv"],
                                  "n_country": r["n_country"], **sh})
            h = r["shapes"]["heat"]
            print(f"    {sec:18s}{r['n']:8,}{r['r2']:8.4f}"
                  f"{100*h['cum']:+11.3f}{h['cum_t']:+7.2f}"
                  f"{h['cum_p_wcb']:8.3f}{h['peak_lag']:6d}"
                  f"{h['pretrend_p']:7.3f}{h['vcov_neg_eig']:8d}")

    prof = pd.concat(all_prof, ignore_index=True)
    shp = pd.DataFrame(all_shape)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    prof.to_parquet(OUT_DIR / "r3_lag_profiles.parquet", index=False)
    shp.to_parquet(OUT_DIR / "r3_shape_indicators.parquet", index=False)
    prof.to_csv(TABLE_DIR / f"r3_lag_profiles_tier{args.tier}.csv", index=False)
    shp.to_csv(TABLE_DIR / f"r3_shape_indicators_tier{args.tier}.csv", index=False)

    print("\n" + "=" * 80)
    print("SHAPE INDICATORS (heat exposure)")
    print("=" * 80)
    h = shp[shp.exposure == "heat"]
    print(f"{'form':12s}{'sector':18s}{'Σβ(%)':>9s}{'t':>7s}{'onset':>8s}"
          f"{'peak':>6s}{'half':>6s}{'rebound':>9s}{'pre-p':>7s}")
    for _, r in h.iterrows():
        hl = "n/a" if not np.isfinite(r.half_life) else f"{int(r.half_life)}"
        print(f"{r.form:12s}{r.sector:18s}{100*r.cum:+9.3f}{r.cum_t:+7.2f}"
              f"{100*r.onset_jump:+8.3f}{int(r.peak_lag):6d}{hl:>6s}"
              f"{100*r.rebound:+9.3f}{r.pretrend_p:7.3f}")

    print("\nFigures")
    for form in prof.form.unique():
        make_figure(prof[prof.form == form], shp[shp.form == form], form, args.tier)
        print(f"    Fig4_sectoral_dynamic_response_{form}_{args.tier}.png")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
