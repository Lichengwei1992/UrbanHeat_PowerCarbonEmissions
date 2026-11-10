#!/usr/bin/env python
"""
25_oxcgrt_robustness.py  --  SI: does COVID policy stringency change anything?
===============================================================================

Every model in R3-R6 uses country x year-month, global date and country x
day-of-week fixed effects. Between them these absorb a country's monthly policy
level and any worldwide common shock, and R5.8 showed that dropping 2020
entirely does not reverse or attenuate the estimates. One channel is still not
absorbed: COUNTRY-SPECIFIC POLICY TIMING WITHIN A MONTH. A national lockdown
beginning on 17 March moves emissions from the 17th, while a country x
year-month effect only removes the March average, and in the northern
hemisphere spring the residual within-month profile is correlated with rising
cooling load and falling heating load.

This script adds the OxCGRT daily national policy indices as a control block to
the three results that could plausibly be affected, and reports the change:

  R3  cumulative Power response to cooling and heating load
  R4  humidity at a given cooling load, and absolute compound humid heat
  R5  country-specific slopes and the pooled cooling-minus-heating difference
      (re-estimated with the full heterogeneous-slope model, not a proxy)

WHY THIS IS A REFINEMENT AND NOT A CORRECTION
---------------------------------------------
If the pandemic were generating these results, the drop-2020 window in R5.8
would have attenuated them; instead the cooling coefficient grew. The purpose
here is to close the remaining channel explicitly rather than argue that it is
small, and to put the size of the change on the record either way.

WHAT A CONTROL CANNOT FIX
-------------------------
Policy stringency is measured at the national level and daily; it cannot
represent within-country heterogeneity in compliance or in the sectoral
incidence of restrictions. A stable coefficient after controlling is evidence
against a first-order policy confound, not proof of none.

DATA
----
OxCGRT compact national file, from
https://github.com/OxCGRT/covid-policy-dataset
    data/OxCGRT_compact_national_v1.csv
Place it at Other_data/OxCGRT_compact_national_v1.csv (the script also accepts
OxCGRT_latest.csv or any file matching Other_data/*OxCGRT*.csv).

Indices used, both daily and national:
    StringencyIndex_Average          containment and closure + public info
    ContainmentHealthIndex_Average   the above plus health-system measures

INPUTS
------
  Other_data/*OxCGRT*.csv
  outputs/exposure/country_daily_exposure.parquet
  CarbonData/Near-real-time .../carbon-monitor-cities-*-v0325.csv

OUTPUTS
-------
  outputs/response/oxcgrt_country_daily.parquet
  Results/SI/tables/TS4_oxcgrt_robustness_tier{AB,C}.csv
  Results/SI/tables/TS5_oxcgrt_country_match.csv
  Results/SI/figures/FigS3_oxcgrt_robustness.{png,pdf,svg}

USAGE
-----
  python 25_oxcgrt_robustness.py --tier AB
"""

from __future__ import annotations

import os
import argparse
import glob
import shutil
import zipfile
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
OTHER = BASE / "Other_data"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
OUT_DIR = BASE / "outputs" / "response"
SI_TAB = RESULTS_ROOT / "SI" / "tables"
SI_FIG = RESULTS_ROOT / "SI" / "figures"

LAGS = 7
CTRL_LAGS = 7
HAC_BW = 14
INDICES = ["StringencyIndex_Average", "ContainmentHealthIndex_Average"]
SAVE_FORMATS = ("png", "pdf", "svg")
SAVE_DPI = 600
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "savefig.dpi": SAVE_DPI, "figure.dpi": 110,
                     "svg.fonttype": "none"})

# Carbon Monitor country labels come from file names; OxCGRT uses its own
# spellings. Only the genuine mismatches are listed; everything else matches on
# an exact string and the script prints anything it could not resolve.
NAME_MAP = {
    "Korea": "South Korea",
    "Russia": "Russia",
    "United States": "United States",
    "United Kingdom": "United Kingdom",
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",
    "Slovakia": "Slovak Republic",
    "Iran": "Iran",
    "Vietnam": "Vietnam",
    "Taiwan": "Taiwan",
    "Hong Kong": "Hong Kong",
    "UAE": "United Arab Emirates",
    "United Arab Emirates": "United Arab Emirates",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "North Macedonia": "North Macedonia",
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
    "Congo": "Congo",
    "DR Congo": "Democratic Republic of Congo",
    "Egypt": "Egypt",
    "Syria": "Syria",
    "Laos": "Laos",
    "Moldova": "Moldova",
    "Venezuela": "Venezuela",
    "Bolivia": "Bolivia",
    "Tanzania": "Tanzania",
    "Brunei": "Brunei",
    "Kyrgyzstan": "Kyrgyz Republic",
}


# --------------------------------------------------------------------------
WANTED = "OxCGRT_compact_national_v1.csv"


def find_oxcgrt() -> Path:
    """Locate the compact NATIONAL file, extracting it from a repo zip if needed.

    The subnational and vaccine files sit beside it in the distribution and
    have overlapping names, so the search is anchored on the exact file name
    rather than on a glob: picking up OxCGRT_compact_subnational_v1.csv by
    accident would silently produce several rows per country-day and inflate
    every merge downstream.
    """
    direct = [p for p in OTHER.rglob("*.csv") if p.name == WANTED]
    if direct:
        return direct[0]

    for z in sorted(OTHER.rglob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                inner = [i for i in zf.infolist()
                         if Path(i.filename).name == WANTED]
                if not inner:
                    continue
                out = OTHER / WANTED
                print(f"    extracting {WANTED} from {z.name} "
                      f"({inner[0].file_size / 1e6:.0f} MB)")
                tmp = out.with_suffix(".extracting")
                with zf.open(inner[0]) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                tmp.replace(out)
                return out
        except zipfile.BadZipFile:
            continue

    loose = sorted(p for p in OTHER.rglob("*.csv")
                   if "oxcgrt" in p.name.lower() and "subnational" not in
                   p.name.lower() and "vaccines" not in p.name.lower())
    if loose:
        print(f"    WARNING: {WANTED} not found; falling back to {loose[0].name}")
        return loose[0]

    raise FileNotFoundError(
        f"No {WANTED} found under {OTHER}.\n"
        "  Get it from https://github.com/OxCGRT/covid-policy-dataset\n"
        "  (data/OxCGRT_compact_national_v1.csv, or the repo zip)")


def load_oxcgrt(countries: list[str], start: str, end: str
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load national daily policy indices over the SAME window as the panel.

    The window is passed in rather than hardcoded. It used to end 2021-12-31,
    which meant that after the panel was extended to 2022 the control block was
    all zeros for the final year -- the robustness check would then have been
    run on a three-year subsample while being reported as covering four.
    """
    f = find_oxcgrt()
    print(f"    reading {f.name}   window {start[:4]}-{end[:4]}")
    use = ["CountryName", "Jurisdiction", "Date"] + INDICES
    d = pd.read_csv(f, usecols=lambda c: c in use, low_memory=False)
    if "Jurisdiction" in d.columns:
        d = d[d["Jurisdiction"] == "NAT_TOTAL"]
    d["date"] = pd.to_datetime(d["Date"].astype(str), format="%Y%m%d",
                               errors="coerce")
    d = d.dropna(subset=["date"])
    d = d[(d.date >= start) & (d.date <= end)]

    ox_names = set(d["CountryName"].unique())
    rows, frames = [], []
    for c in countries:
        target = NAME_MAP.get(c, c)
        hit = target if target in ox_names else None
        if hit is None:                       # last resort: case-insensitive
            low = {n.lower(): n for n in ox_names}
            hit = low.get(target.lower())
        rows.append({"carbon_monitor": c, "oxcgrt": hit,
                     "matched": hit is not None})
        if hit is not None:
            g = d[d.CountryName == hit][["date"] + INDICES].copy()
            g["country"] = c
            frames.append(g)
    match = pd.DataFrame(rows)

    if not frames:
        raise RuntimeError("No country matched between Carbon Monitor and OxCGRT")
    ox = pd.concat(frames, ignore_index=True)

    # 2019 predates the dataset for most countries: policy was zero, so a
    # missing value there is a true zero rather than an absent measurement.
    full = (pd.MultiIndex
            .from_product([ox.country.unique(), pd.date_range(start, end)],
                          names=["country", "date"]).to_frame(index=False))
    ox = full.merge(ox, on=["country", "date"], how="left")
    ox = ox.sort_values(["country", "date"])
    for c in INDICES:
        ox[c] = ox.groupby("country")[c].ffill()
        ox[c] = ox[c].fillna(0.0)
    return ox, match


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


def fit(d: pd.DataFrame, blocks: dict[str, str],
        controls: list[str] | None = None) -> dict | None:
    """Distributed-lag FE fit, optionally with a lagged control block.

    Controls enter with their own 0..CTRL_LAGS lag structure, standardised the
    same way as the exposure blocks, and are never reported: they exist only to
    absorb variation, and reporting a coefficient on a policy index estimated
    this way would invite a causal reading it cannot support.
    """
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
    for c in (controls or []):
        sd = d[c].std()
        if not np.isfinite(sd) or sd == 0:
            continue
        d[c + "_z"] = d[c] / sd
        g = d.groupby("country")[c + "_z"]
        for s in range(CTRL_LAGS + 1):
            nm = f"ctrl_{c}_L{s}"
            d[nm] = g.shift(s)
            names.append(nm)

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


def fit_r5(p: pd.DataFrame, controls: list[str] | None = None,
           fixed_h: set | None = None, fixed_c: set | None = None) -> dict | None:
    """R5's country-specific-slope model, optionally with a policy control.

    Mirrors fit_joint in 22_r5_asymmetry.py: country-specific cooling and
    heating slopes inside country x year-month, global date and country x
    day-of-week effects, Driscoll-Kraay variance. Only the pooled (precision-
    weighted) heat, cold and difference are returned, since the SI question is
    whether R5's headline moves, not whether any single country's does.
    """
    p = p.sort_values(["country", "date"]).reset_index(drop=True).copy()
    p["h"] = p["w_SCDH24"] / 100.0
    p["c"] = p["w_HDH15"] / 100.0
    g = p.groupby("country")
    for s in range(LAGS + 1):
        p[f"h{s}"] = g["h"].shift(s)
        p[f"c{s}"] = g["c"].shift(s)
    lag_cols = [f"{k}{s}" for s in range(LAGS + 1) for k in ("h", "c")]

    ctrl_cols = []
    for cc in (controls or []):
        sd = p[cc].std()
        if not np.isfinite(sd) or sd == 0:
            continue
        p[cc + "_z"] = p[cc] / sd
        gg = p.groupby("country")[cc + "_z"]
        for s in range(CTRL_LAGS + 1):
            nm = f"ctrl_{cc}_L{s}"
            p[nm] = gg.shift(s)
            ctrl_cols.append(nm)

    p = p.dropna(subset=lag_cols + ctrl_cols + ["y"]).reset_index(drop=True)
    scr = p.groupby("country").agg(
        n_hot=("w_SCDH24", lambda x: int((x >= 5.0).sum())),
        n_cold=("w_HDH15", lambda x: int((x >= 50.0).sum()))).reset_index()
    ok_h = set(scr.loc[scr.n_hot >= 100, "country"])
    ok_c = set(scr.loc[scr.n_cold >= 100, "country"])
    # The >= 100 rule is applied to whatever is in w_SCDH24 / w_HDH15, so a
    # base-temperature scan silently changes the country set as well as the
    # coefficients -- two moving parts where the comparison needs one. Passing
    # fixed_h / fixed_c pins the set; leaving them None preserves the published
    # behaviour exactly.
    if fixed_h is not None:
        ok_h = set(fixed_h)
    if fixed_c is not None:
        ok_c = set(fixed_c)
    countries = sorted(p.country.unique())

    cols, meta = [], []
    for ctry in countries:
        m = (p.country == ctry).values.astype("float64")
        for side, ok in (("h", ok_h), ("c", ok_c)):
            if ctry not in ok:
                continue
            for s in range(LAGS + 1):
                cols.append(p[f"{side}{s}"].values * m)
                meta.append((ctry, side))
    n_slope = len(cols)
    for nm in ctrl_cols:
        cols.append(p[nm].values)
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

    Xe = X * e[:, None]
    nT = f2.max() + 1
    S = np.zeros((nT, X.shape[1]))
    for j in range(X.shape[1]):
        S[:, j] = np.bincount(f2, weights=Xe[:, j], minlength=nT)
    M = S.T @ S
    for l in range(1, HAC_BW + 1):
        w = 1.0 - l / (HAC_BW + 1.0)
        G = S[l:].T @ S[:-l]
        M += w * (G + G.T)
    V = inv @ M @ inv

    mdf = pd.DataFrame(meta, columns=["country", "side"])
    per = {"heat": [], "cold": [], "diff": []}
    for ctry in countries:
        ih = mdf.index[(mdf.country == ctry) & (mdf.side == "h")].values
        ic = mdf.index[(mdf.country == ctry) & (mdf.side == "c")].values
        for tag, ix in (("heat", ih), ("cold", ic)):
            if len(ix):
                cum = float(b[ix].sum())
                se = float(np.sqrt(max(float(V[np.ix_(ix, ix)].sum()), 0)))
                per[tag].append((cum, se))
        if len(ih) and len(ic):
            cv = np.zeros(len(b))
            cv[ih] = 1.0
            cv[ic] = -1.0
            cum = float(cv @ b)
            se = float(np.sqrt(max(float(cv @ V @ cv), 0)))
            per["diff"].append((cum, se))

    out = {"n_slope": n_slope, "n": len(p),
           "countries_h": sorted(ok_h & set(countries)),
           "countries_c": sorted(ok_c & set(countries))}
    for tag, vals in per.items():
        if len(vals) < 4:
            continue
        bb = np.array([v[0] for v in vals])
        ss = np.array([v[1] for v in vals])
        tau2 = max(bb.var(ddof=1) - (ss ** 2).mean(), 1e-12)
        for _ in range(500):
            w = 1.0 / (ss ** 2 + tau2)
            mu = float((w * bb).sum() / w.sum())
            num = (w ** 2 * ((bb - mu) ** 2 - ss ** 2)).sum()
            tau2 = max(float(num / (w ** 2).sum()), 0.0)
        w = 1.0 / (ss ** 2 + tau2)
        mu = float((w * bb).sum() / w.sum())
        mu_se = float(np.sqrt(1.0 / w.sum()))
        out[tag] = {"cum": mu, "se": mu_se, "t": mu / mu_se if mu_se else np.nan}
    return out


def prepare(p: pd.DataFrame) -> pd.DataFrame:
    """Derived exposure columns, matching 20_r4_humidity.py exactly."""
    p = p.reset_index(drop=True).copy()
    p["wbd_orth"] = np.nan
    for c, g in p.groupby("country"):
        X = np.column_stack([np.ones(len(g)), g["w_SCDH24"].values])
        b = np.linalg.lstsq(X, g["w_WBD_at_tmax"].values, rcond=None)[0]
        p.loc[g.index, "wbd_orth"] = g["w_WBD_at_tmax"].values - X @ b
    p["humid_orth"] = -p["wbd_orth"]
    return p


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C", "all"])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("SI: COVID policy-stringency robustness")
    print("=" * 78)

    ex = pd.read_parquet(EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"],
            "all": ["A", "B", "C"]}[args.tier]
    ex = ex[ex.tier.isin(keep)]
    countries = sorted(ex.country.unique())

    print(f"\n[1] OxCGRT national daily indices, Tier {args.tier} "
          f"({len(countries)} countries)")
    # Window taken from the exposure panel so the control always spans the
    # same period as the models it is added to.
    start = str(ex.date.min().date())
    end = str(ex.date.max().date())
    ox, match = load_oxcgrt(countries, start, end)
    SI_TAB.mkdir(parents=True, exist_ok=True)
    match.to_csv(SI_TAB / "TS5_oxcgrt_country_match.csv", index=False)
    miss = match[~match.matched]
    print(f"    matched {int(match.matched.sum())}/{len(match)} countries")
    if len(miss):
        print(f"    UNMATCHED (dropped): {', '.join(miss.carbon_monitor)}")
    s = ox[ox.country.isin(countries)]
    print(f"    stringency: mean {s.StringencyIndex_Average.mean():.1f}, "
          f"max {s.StringencyIndex_Average.max():.1f}")
    yr = s.groupby(s.date.dt.year)["StringencyIndex_Average"].mean()
    print("    annual mean stringency: "
          + ", ".join(f"{y} {v:.1f}" for y, v in yr.items()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ox.to_parquet(OUT_DIR / "oxcgrt_country_daily.parquet", index=False)

    # within-country-month variation is what the fixed effects cannot absorb
    z = s.copy()
    z["ym"] = z.date.dt.to_period("M").astype(str)
    within = z.groupby(["country", "ym"])["StringencyIndex_Average"].std()
    print(f"    within country-month SD of stringency: mean "
          f"{within.mean():.2f}, p90 {within.quantile(0.9):.2f}")
    print("    this is the only part the fixed effects cannot reach, and it is")
    print("    what the control block is added to absorb")

    print("\n[2] Re-estimating with and without the policy control block")
    tests = [
        ("R3 Power: cooling load", "Power", {"heat": "w_SCDH24"}, None),
        ("R3 Power: heating load", "Power", {"cold": "w_HDH15"}, None),
        ("R3 Power: heat+cold joint", "Power",
         {"heat": "w_SCDH24", "cold": "w_HDH15"}, None),
        ("R4 Power: humidity at same load", "Power",
         {"scdh": "w_SCDH24", "humid": "humid_orth"}, "humid"),
        ("R4 Power: absolute humid heat", "Power",
         {"ehd": "w_EHD", "chhe": "w_CHHE"}, "chhe"),
        ("R4 Residential: humidity at same load", "Residential",
         {"scdh": "w_SCDH24", "humid": "humid_orth"}, "humid"),
        ("R4 Industry: absolute humid heat", "Industry",
         {"ehd": "w_EHD", "chhe": "w_CHHE"}, "chhe"),
    ]

    panels: dict[str, pd.DataFrame] = {}
    rows = []
    print(f"    {'test':42s}{'no control':>20s}{'+ policy':>20s}{'Δ':>8s}")
    for label, sector, blocks, focus in tests:
        if sector not in panels:
            em = load_sector(sector)
            p = em.merge(ex, on=["country", "date"], how="inner")
            p = p.merge(ox, on=["country", "date"], how="inner")
            p = p[p.v > 0].copy()
            p["y"] = np.log(p.v)
            panels[sector] = prepare(p)
        p = panels[sector]
        base = fit(p, blocks)
        ctrl = fit(p, blocks, controls=INDICES)
        if base is None or ctrl is None:
            print(f"    {label:42s}{'not estimable':>20s}")
            continue
        for tag in blocks:
            if focus is not None and tag != focus:
                continue
            b0, b1 = base[tag], ctrl[tag]
            d_pct = 100 * (b1["cum"] - b0["cum"])
            rows.append({"test": label, "sector": sector, "term": tag,
                         "base_pct": 100 * b0["cum"], "base_t": b0["t"],
                         "ctrl_pct": 100 * b1["cum"], "ctrl_t": b1["t"],
                         "delta_pct": d_pct,
                         "rel_change": (b1["cum"] - b0["cum"]) / b0["cum"]
                         if b0["cum"] else np.nan,
                         "n_base": base["n"], "n_ctrl": ctrl["n"]})
            nm = label if focus is not None else f"{label} [{tag}]"
            print(f"    {nm:42s}"
                  f"{100 * b0['cum']:12.3f}({b0['t']:+.1f})"
                  f"{100 * b1['cum']:12.3f}({b1['t']:+.1f})"
                  f"{d_pct:+8.3f}")

    # R5's country-specific-slope model is re-estimated separately: its
    # headline is a pooled quantity built from 16-18 country slopes, not a
    # single homogeneous coefficient, so the homogeneous fits above do not
    # speak to it.
    print("\n[2b] R5 country-specific slopes, pooled")
    print(f"    {'quantity':42s}{'no control':>20s}{'+ policy':>20s}{'Δ':>8s}")
    p5 = panels["Power"]
    r0 = fit_r5(p5)
    r1 = fit_r5(p5, controls=INDICES)
    if r0 and r1:
        for tag, lab in (("heat", "R5 pooled: cooling load"),
                         ("cold", "R5 pooled: heating load"),
                         ("diff", "R5 pooled: cooling - heating")):
            if tag not in r0 or tag not in r1:
                continue
            b0, b1 = r0[tag], r1[tag]
            rows.append({"test": lab, "sector": "Power", "term": tag,
                         "base_pct": 100 * b0["cum"], "base_t": b0["t"],
                         "ctrl_pct": 100 * b1["cum"], "ctrl_t": b1["t"],
                         "delta_pct": 100 * (b1["cum"] - b0["cum"]),
                         "rel_change": (b1["cum"] - b0["cum"]) / b0["cum"]
                         if b0["cum"] else np.nan,
                         "n_base": r0["n"], "n_ctrl": r1["n"]})
            print(f"    {lab:42s}"
                  f"{100 * b0['cum']:12.3f}({b0['t']:+.1f})"
                  f"{100 * b1['cum']:12.3f}({b1['t']:+.1f})"
                  f"{100 * (b1['cum'] - b0['cum']):+8.3f}")
        print("    units here are %/100 K.h, not per SD, matching R5")

    res = pd.DataFrame(rows)
    res.to_csv(SI_TAB / f"TS4_oxcgrt_robustness_tier{args.tier}.csv", index=False)

    print("\n[3] Verdict")
    if len(res):
        worst = res.loc[res.rel_change.abs().idxmax()]
        flips = res[(np.sign(res.base_pct) != np.sign(res.ctrl_pct))]
        lost = res[(res.base_t.abs() > 1.96) & (res.ctrl_t.abs() <= 1.96)]
        gained = res[(res.base_t.abs() <= 1.96) & (res.ctrl_t.abs() > 1.96)]
        print(f"    largest relative change: {worst.rel_change:+.1%} "
              f"({worst.test}, {worst.term})")
        print(f"    sign flips: {len(flips)}/{len(res)}")
        print(f"    lose significance: {len(lost)}/{len(res)}"
              + (f"  ({', '.join(lost.test)})" if len(lost) else ""))
        print(f"    gain significance: {len(gained)}/{len(res)}"
              + (f"  ({', '.join(gained.test)})" if len(gained) else ""))
        print("    a control that changes nothing is evidence against a")
        print("    first-order policy confound, not proof that none exists:")
        print("    OxCGRT is national and daily, so within-country compliance")
        print("    and sectoral incidence remain unmeasured")
    print(f"\n    SI/tables/TS4_oxcgrt_robustness_tier{args.tier}.csv")

    if not args.no_figure and len(res):
        make_figure(res, within, yr, args.tier)
    print(f"\nDone in {time.time() - t0:.0f}s")


def make_figure(res: pd.DataFrame, within: pd.Series, yr: pd.Series,
                tier: str) -> None:
    SI_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax = axes[0]
    ax.bar(yr.index.astype(str), yr.values, color="#34495e", width=0.55)
    ax.set_xlabel("year")
    ax.set_ylabel("mean stringency index")
    ax.set_title("a  Policy stringency in the sample")

    ax = axes[1]
    ax.hist(within.dropna().values, bins=40, color="#7f8c8d")
    ax.set_xlabel("within country-month SD of stringency")
    ax.set_ylabel("country-months")
    ax.set_title("b  The part fixed effects cannot absorb")

    ax = axes[2]
    yy = np.arange(len(res))[::-1]
    ax.scatter(res.base_pct, yy + 0.16, s=26, color="#c0392b",
               label="no control")
    ax.scatter(res.ctrl_pct, yy - 0.16, s=26, color="#2980b9", marker="s",
               label="+ policy control")
    for i, (_, r) in enumerate(res.iterrows()):
        ax.plot([r.base_pct, r.ctrl_pct], [yy[i] + 0.16, yy[i] - 0.16],
                color="#95a5a6", lw=0.7, zorder=0)
    ax.axvline(0, color="k", lw=0.8)
    # R3/R4 rows are per SD of exposure; R5 rows are per 100 K.h, matching the
    # units each result is reported in. The blocks are separated so the two
    # scales are never read off against each other.
    is_r5 = res.test.str.startswith("R5").values
    if is_r5.any() and (~is_r5).any():
        ax.axhline(yy[is_r5.argmax()] + 0.5, color="#95a5a6", lw=0.8, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r.test} [{r.term}]" for _, r in res.iterrows()],
                       fontsize=5.5)
    ax.set_xlabel("cumulative effect\n(R3/R4: % per SD;  R5: %/100 K·h)")
    ax.set_title("c  Estimates barely move")
    ax.legend(frameon=False, fontsize=6.5)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Figure S3  COVID policy-stringency robustness "
                 f"(OxCGRT, Tier {tier})", y=1.04, fontsize=9.5)
    fig.tight_layout()
    name = "FigS3_oxcgrt_robustness"
    for ext in SAVE_FORMATS:
        fig.savefig(SI_FIG / f"{name}.{ext}", bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)
    print(f"    SI/figures/{name}  [{', '.join(SAVE_FORMATS)}]  {SAVE_DPI} dpi")


if __name__ == "__main__":
    main()
