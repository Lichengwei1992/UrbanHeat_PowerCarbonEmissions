#!/usr/bin/env python
"""
39_lcz_variants.py  --  can any LCZ weighting reach usable power?
==================================================================

WHAT THE POWER CALCULATION ESTABLISHED
--------------------------------------
The tilt design of 37_built_tilt_response.py returns a null for every built
environment weighting, but the null is uninformative. Writing theta = gamma /
beta -- the share of the way from population weighting to the alternative
weighting that the data support -- the 95 % interval for built volume on the
cooling side is [-7.4, +16.6]. The design cannot even reject theta = 1, the
case where the alternative weighting is entirely correct.

The reason is arithmetic. A tilt D = X^w - X^pop has variance only to the
extent that w ranks a country's cities differently from population. Built
volume ranks them almost identically (median within-country Spearman 0.947), so
D holds 10 % of the exposure variation and its coefficient carries roughly
eighteen times the standard error of the main effect. Detecting theta = 1 would
need a main effect with t near 50; the real one has t = 2.9.

WHAT THIS SCRIPT ASKS
---------------------
LCZ is the one layer that ranks cities differently from both population and
built volume (Spearman 0.78 against built volume), and its tilt already holds
28 % of the exposure variation rather than 10 %. That is a factor of 2.8 in the
right direction, from a single crude definition -- the share of built LCZ cells
that are compact.

So: is there an LCZ weighting whose divergence from population weighting is
large enough to bring theta into a range where the answer means something? The
question is decided before any response model is fitted, by the variance the
tilt retains after the fixed effects, so the sweep is cheap.

TWO KINDS OF CANDIDATE
----------------------
NOMINAL variants slice the class list differently: compact classes only, the
compact high-rise class alone, the open and sparse classes as a contrast, and
so on. They are easy to compute and easy to over-fit, so the whole set is
reported rather than the best one.

PHYSICAL variants assign each LCZ class a canonical property from the Stewart
and Oke scheme -- sky view factor, building surface fraction, impervious
fraction -- and take the area-weighted city mean. These are the defensible
ones: they turn a nominal classification into a continuous physical variable
whose meaning does not depend on where a threshold was drawn.

STAGES
------
  extract   per-FUA histogram over the 17 LCZ classes, chunked and resumable
  sweep     build every candidate weight, report divergence from population
            weighting and the tilt variance surviving country x year-month
  fit       fit the response model for the candidates that clear the screen

USAGE
-----
  python 39_lcz_variants.py --stage extract --limit 600
  python 39_lcz_variants.py --stage sweep
  python 39_lcz_variants.py --stage fit --top 4
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
FUA_SHP = BASE / "ShapefileData" / "FUA_cities_final.shp"
LCZ_TIF = BASE / "Other_data" / "Global_LCZ" / "lcz_filter_v3.tif"
SCHEMES = BASE / "outputs" / "population" / "fua_weight_schemes.parquet"
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
HIST = BASE / "outputs" / "_cache" / "lcz_hist.parquet"
OUT_W = BASE / "outputs" / "population" / "fua_lcz_weights.parquet"

COOL, HEAT = "SCDH24", "HDH15"
CLASSES = list(range(1, 18))          # LCZ 1-10 built, 11-17 natural
BUILT = list(range(1, 11))

# Canonical properties of the built LCZ classes, from the Stewart and Oke
# (2012) definition table. Mid-points of the published ranges are used; these
# are class definitions, not measurements of any particular city, and they are
# what makes a physical variant interpretable without a threshold choice.
#            sky view   building    impervious   height
#            factor     surf frac   frac         (m)
LCZ_PROPS = {
    1:  (0.35, 0.50, 0.45, 37.5),   # compact high-rise
    2:  (0.45, 0.50, 0.45, 17.5),   # compact mid-rise
    3:  (0.45, 0.50, 0.45,  6.0),   # compact low-rise
    4:  (0.65, 0.30, 0.35, 37.5),   # open high-rise
    5:  (0.70, 0.30, 0.35, 17.5),   # open mid-rise
    6:  (0.70, 0.30, 0.35,  6.0),   # open low-rise
    7:  (0.40, 0.75, 0.15,  3.0),   # lightweight low-rise
    8:  (0.90, 0.40, 0.55,  6.5),   # large low-rise
    9:  (0.85, 0.15, 0.15,  6.0),   # sparsely built
    10: (0.75, 0.25, 0.45, 10.0),   # heavy industry
}


def polygons():
    import geopandas as gpd
    g = gpd.read_file(FUA_SHP).to_crs("EPSG:4326")
    return g[["fuacode", "city", "country", "geometry"]].reset_index(drop=True)


def stage_extract(limit: int, log_every: int) -> None:
    import rasterio
    from rasterio.mask import mask

    g = polygons()
    done = pd.read_parquet(HIST) if HIST.exists() else pd.DataFrame(
        columns=["idx"])
    have = set(done["idx"]) if len(done) else set()
    todo = [i for i in range(len(g)) if i not in have]
    print(f"    polygons {len(g)}   cached {len(have)}   to do {len(todo)}")
    if not todo:
        return
    todo = todo[:limit] if limit else todo

    rows, t0 = [], time.time()
    with rasterio.open(LCZ_TIF) as src:
        for n, i in enumerate(todo, 1):
            r = {"idx": i, "fuacode": g.fuacode.iloc[i]}
            try:
                z, _ = mask(src, [g.geometry.iloc[i]], crop=True, filled=True,
                            nodata=0)
                v = z[0].ravel()
                cnt = np.bincount(v[v > 0], minlength=18)
                for c in CLASSES:
                    r[f"lcz{c}"] = int(cnt[c]) if c < len(cnt) else 0
            except Exception as exc:                             # noqa: BLE001
                for c in CLASSES:
                    r[f"lcz{c}"] = 0
                r["err"] = str(exc)[:60]
            rows.append(r)
            if n % log_every == 0 or n == len(todo):
                print(f"      {n}/{len(todo)}  {time.time() - t0:6.1f}s",
                      flush=True)
                HIST.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([done, pd.DataFrame(rows)],
                          ignore_index=True).to_parquet(HIST, index=False)
    HIST.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([done, pd.DataFrame(rows)],
              ignore_index=True).to_parquet(HIST, index=False)
    print(f"    -> {HIST.relative_to(BASE)}")


# --------------------------------------------------------------------------
def candidate_weights() -> pd.DataFrame:
    """One column per candidate weighting, per city. All must be extensive."""
    h = pd.read_parquet(HIST).drop_duplicates("fuacode")
    w = pd.read_parquet(SCHEMES)
    d = w.merge(h, on="fuacode", how="left")
    cls = d[[f"lcz{c}" for c in CLASSES]].fillna(0).to_numpy(float)
    built = cls[:, :10]
    n_built = built.sum(axis=1)
    ok = n_built > 0

    def share(idx):
        s = np.full(len(d), np.nan)
        s[ok] = built[np.ix_(ok, [i - 1 for i in idx])].sum(axis=1) / n_built[ok]
        return s

    def prop_mean(k):
        """Area-weighted mean of a Stewart-Oke class property."""
        v = np.array([LCZ_PROPS[c][k] for c in BUILT])
        s = np.full(len(d), np.nan)
        s[ok] = (built[ok] @ v) / n_built[ok]
        return s

    # Intensive descriptors first, then multiplied by a scale to make weights.
    d["s_compact"] = share([1, 2, 3])          # compact high/mid/low-rise
    d["s_compact12"] = share([1, 2])           # compact high and mid only
    d["s_hi1"] = share([1])                    # compact high-rise alone
    d["s_open"] = share([6, 9])                # open low-rise and sparsely built
    d["s_dense_any"] = share([1, 2, 4, 5])     # any mid or high rise
    d["svf"] = prop_mean(0)
    d["bsf"] = prop_mean(1)
    d["isf"] = prop_mean(2)
    d["lcz_h"] = prop_mean(3)
    # Contrast: compact minus open. Ranges over [-1, 1] and is shifted into a
    # positive weight, because a weight must not be negative.
    d["s_contrast"] = (d.s_compact - d.s_open + 1.0) / 2.0

    # A weight must be EXTENSIVE. Built-up area is the scale carrier; the
    # intensive descriptor selects which part of that stock is emphasised.
    # NOTE ON THE SCHEMES TABLE. In fua_weight_schemes.parquet the columns
    # `built_s` and `built_v` hold the zonal sum for the whole FUA, repeated on
    # every city row that shares a fuacode; only `w_built_s` and `w_built_v` are
    # divided by n_share. `pop_weight_dedup` is already deduplicated. Summing or
    # weighting by the raw columns therefore double counts 34 cities in 5
    # countries -- 24 of them in China, which inflates its built volume per capita
    # by 31 % -- while leaving population correct. The deduplicated columns are
    # used everywhere below.
    scale = d.w_built_s.replace(0, np.nan)
    cands = {
        "compact": d.s_compact, "compact12": d.s_compact12, "hi1": d.s_hi1,
        "denseany": d.s_dense_any, "contrast": d.s_contrast,
        "svf": 1.0 - d.svf,               # low sky view factor = enclosed
        "bsf": d.bsf, "isf": d.isf, "lczh": d.lcz_h,
        # sharpened: raising a share to a power pushes weight onto the extreme
        # cities, which is the only lever that increases divergence without
        # inventing a new variable
        "compact_p2": d.s_compact ** 2, "compact_p4": d.s_compact ** 4,
    }
    for k, v in cands.items():
        d[f"w_{k}"] = scale * v
    d["w_area"] = scale                    # built area alone, as a reference
    keep = ["city_id", "country", "fuacode", "pop_weight_dedup", "w_built_v",
            "w_area"] + [f"w_{k}" for k in cands]
    out = d[keep]
    OUT_W.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_W, index=False)
    return out


def stage_sweep() -> None:
    w = candidate_weights()
    wcols = [c for c in w.columns if c.startswith("w_")]
    print(f"    {len(w)} cities, {len(wcols)} candidate weights\n")

    # 1. how differently does each weight rank cities, within country?
    print("[1] Divergence from population weighting (within-country Spearman)")
    rho = {}
    for c in wcols:
        rs = []
        for _, s in w.groupby("country"):
            s2 = s[["pop_weight_dedup", c]].dropna()
            if len(s2) >= 5 and s2.iloc[:, 0].std() > 0 and s2.iloc[:, 1].std() > 0:
                rs.append(float(s2.corr(method="spearman").iloc[0, 1]))
        rho[c] = float(np.median(rs)) if rs else np.nan
    for c in sorted(rho, key=lambda k: rho[k]):
        print(f"      {c:16s} median rho vs population = {rho[c]:+.3f}")
    print("      a weight that ranks cities as population does produces the")
    print("      same national mean, so a rho near 1 leaves nothing to test.")

    # 2. what does the tilt retain after the fixed effects?
    print("\n[2] Tilt variation surviving country x year-month fixed effects")
    ev = pd.read_parquet(EVENTS, columns=["city_id", "date", COOL, HEAT])
    ex = pd.read_parquet(EXPOSURE, columns=["country", "date", "tier"])
    ex["date"] = pd.to_datetime(ex.date)
    ex = ex.drop_duplicates(["country", "date"])

    base = w[["city_id", "country", "pop_weight_dedup"] + wcols]
    d = ev.merge(base, on="city_id", how="inner")
    d = d[d.pop_weight_dedup > 0]
    series = {}
    for wt in ["pop_weight_dedup"] + wcols:
        m = d[d[wt].notna() & (d[wt] > 0)]
        num = pd.DataFrame({"country": m.country, "date": m.date,
                            COOL: m[COOL] * m[wt], HEAT: m[HEAT] * m[wt],
                            "wt": m[wt]})
        g = num.groupby(["country", "date"], sort=False)[[COOL, HEAT, "wt"]].sum()
        for c in (COOL, HEAT):
            series[f"{c}__{wt}"] = g[c] / g["wt"]
    t = pd.DataFrame(series).reset_index()
    t["date"] = pd.to_datetime(t.date)
    t = t.merge(ex, on=["country", "date"], how="left")
    t = t[t.tier.isin(["A", "B"])]
    t["ym"] = t.date.dt.to_period("M").astype(str)

    def resid(col):
        return (t[col] - t.groupby(["country", "ym"])[col].transform("mean"))

    rows = []
    for side, var in (("cool", COOL), ("heat", HEAT)):
        xr = resid(f"{var}__pop_weight_dedup")
        sx = float(xr.std())
        for c in wcols:
            dr = resid(f"{var}__{c}") - xr
            sd = float(dr.std())
            rows.append({"side": side, "weight": c, "rho_pop": rho[c],
                         "sd_X": sx, "sd_D": sd, "D_share_pct": 100 * sd / sx,
                         "corr_X_D": float(np.corrcoef(xr, dr)[0, 1])
                         if sd > 0 else np.nan})
    r = pd.DataFrame(rows)

    # 3. translate into the quantity that decides usability.
    #    SE(gamma) scales roughly as SD(X)/SD(D) times SE(beta), so the
    #    detectable theta at 80 % power is about 2.80 / (t_beta * SD_D/SD_X).
    T_BETA = {"cool": 2.91, "heat": 3.84}       # R3 Power main effects
    r["theta80_pred"] = 2.80 / (r.side.map(T_BETA) * r.D_share_pct / 100)
    r = r.sort_values(["side", "theta80_pred"])
    print(f"      {'side':6s}{'weight':16s}{'rho':>7s}{'D/X %':>8s}"
          f"{'corr':>7s}{'theta80':>9s}")
    for x in r.itertuples():
        flag = "  <= usable" if x.theta80_pred <= 1 else ""
        print(f"      {x.side:6s}{x.weight:16s}{x.rho_pop:+7.2f}"
              f"{x.D_share_pct:8.1f}{x.corr_X_D:+7.2f}"
              f"{x.theta80_pred:9.2f}{flag}")
    print("\n      theta80 is the smallest effect detectable at 80 % power,")
    print("      in units where 1 means the alternative weighting is entirely")
    print("      correct. Above 1 the test cannot reject even that, and the")
    print("      null it returns carries no information.")

    out = RESULTS / "R6" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "T12c_lcz_variant_power.csv"
    r.to_csv(p, index=False)
    print(f"\n    {p.relative_to(RESULTS)}")
    best = r[r.side == "cool"].iloc[0]
    if best.theta80_pred <= 1:
        print(f"\n    VERDICT: {best.weight} reaches theta80 = "
              f"{best.theta80_pred:.2f}; the response model is worth fitting.")
    else:
        print(f"\n    VERDICT: the best candidate ({best.weight}) still needs "
              f"an effect {best.theta80_pred:.1f} times\n    the size of a "
              "complete change of weighting. No LCZ transformation rescues\n"
              "    the design; the limit is the exposure field, not the "
              "classification.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sweep",
                    choices=["extract", "sweep", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=300)
    args = ap.parse_args()
    print("=" * 78)
    print("LCZ weighting variants: is any of them powerful enough?")
    print("=" * 78)
    if args.stage in ("extract", "all"):
        stage_extract(args.limit, args.log_every)
    if args.stage in ("sweep", "all"):
        stage_sweep()


if __name__ == "__main__":
    main()
