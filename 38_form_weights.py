#!/usr/bin/env python
"""
38_form_weights.py  --  urban FORM weights, to sit beside the SCALE weight
===========================================================================

23_built_environment_weights.py gave R6 a scale measure: built-up surface and
built-up volume per city. R6's own moderator test then showed that volume
associates with the country cooling slope (t = 3.86) while surface does not
(t = 0.69), and volume over surface is a mean height. That points at FORM
rather than size, which neither GHS layer measures directly.

This script builds two form variables per functional urban area:

  height_m       building-area-weighted mean building height, from WSF3D V02
  compact_share  share of the built LCZ classes that are compact (LCZ 1-3),
                 from the global local climate zone map

UNITS: WSF3D BuildingHeight IS IN DECIMETRES
--------------------------------------------
The raster is int16 and a first read gave city means of 35-185, which cannot be
metres. Two independent checks fix the scale at one tenth: dividing by ten puts
Singapore at 18.5 m and Phoenix at 3.5 m, and the resulting distribution lines
up with GHS built_v / built_s, which is a height in metres by construction
(median 6.6 m). The factor is applied here and asserted in the validate stage,
because a silent factor-of-ten error would propagate into every weight.

WHY THE WEIGHTS ARE PRODUCTS, NOT THE FORM VARIABLES THEMSELVES
---------------------------------------------------------------
A weighting scheme aggregates city exposure to a national series:

    X_ct = sum_i w_i x_it / sum_i w_i

so w must be EXTENSIVE -- a quantity of city, not a property of it. Mean height
and compact share are intensive: using them directly would weight a small tall
city the same as a large one. The form weights are therefore products of the
scale a city has and the form it takes:

    w_tall     = built_v * height_m        volume held in tall stock
    w_compact  = built_v * compact_share   volume sitting in compact fabric

Each then generates a tilt against the population-weighted series, in the same
way built volume does in 37_built_tilt_response.py, and the tilts can be
entered together to ask whether form adds anything to scale.

STAGES
------
  extract   zonal statistics over the FUA polygons, chunked and resumable
  validate  scale check against GHS built_v / built_s, and coverage
  weights   assemble and write the form weights beside the existing schemes

USAGE
-----
  python 38_form_weights.py --stage extract --limit 400
  python 38_form_weights.py --stage extract          # resumes
  python 38_form_weights.py --stage validate
  python 38_form_weights.py --stage weights
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
OTHER = BASE / "Other_data"
FUA_SHP = BASE / "ShapefileData" / "FUA_cities_final.shp"
SCHEMES = BASE / "outputs" / "population" / "fua_weight_schemes.parquet"
OUT = BASE / "outputs" / "population" / "fua_form_weights.parquet"
CACHE = BASE / "outputs" / "_cache" / "form_zonal.parquet"

H_TIF = OTHER / "Global_WSF3D" / "WSF3D_V02_BuildingHeight.tif"
A_TIF = OTHER / "Global_WSF3D" / "WSF3D_V02_BuildingArea.tif"
LCZ_TIF = OTHER / "Global_LCZ" / "lcz_filter_v3.tif"

# WSF3D BuildingHeight is stored in decimetres; see the module docstring.
H_SCALE = 0.1
# LCZ 1-10 are the built types; 1-3 are the compact ones (compact high-rise,
# compact mid-rise, compact low-rise). 11-17 are the natural land-cover types
# and are excluded from the denominator so that compact_share is a statement
# about urban fabric, not about how much countryside a polygon happens to hold.
LCZ_BUILT = list(range(1, 11))
LCZ_COMPACT = [1, 2, 3]


def polygons():
    import geopandas as gpd
    g = gpd.read_file(FUA_SHP).to_crs("EPSG:4326")
    if "fuacode" not in g.columns:
        raise SystemExit("FUA shapefile has no fuacode column")
    return g[["fuacode", "city", "country", "geometry"]].reset_index(drop=True)


def stage_extract(limit: int, log_every: int) -> None:
    import rasterio
    from rasterio.mask import mask

    g = polygons()
    done = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(
        columns=["idx"])
    have = set(done["idx"]) if len(done) else set()
    todo = [i for i in range(len(g)) if i not in have]
    print(f"    polygons: {len(g)}   cached: {len(have)}   to do: {len(todo)}")
    if not todo:
        return
    todo = todo[:limit] if limit else todo

    rows, t0 = [], time.time()
    with rasterio.open(H_TIF) as sh, rasterio.open(A_TIF) as sa, \
            rasterio.open(LCZ_TIF) as sl:
        for n, i in enumerate(todo, 1):
            geom = [g.geometry.iloc[i]]
            r = {"idx": i, "fuacode": g.fuacode.iloc[i],
                 "city": g.city.iloc[i], "country": g.country.iloc[i]}
            try:
                h, _ = mask(sh, geom, crop=True, filled=True, nodata=sh.nodata)
                a, _ = mask(sa, geom, crop=True, filled=True, nodata=sa.nodata)
                hh = h[0].astype("float64")
                aa = a[0].astype("float64")
                ok = (hh > 0) & (aa > 0) & np.isfinite(hh) & np.isfinite(aa)
                # Building-area weighted, so a pixel holding one shed does not
                # count as much as a pixel of dense stock.
                r["bld_area_m2"] = float(aa[ok].sum())
                r["height_m"] = (float((hh[ok] * aa[ok]).sum() / aa[ok].sum())
                                 * H_SCALE) if ok.any() else np.nan
                r["n_px_wsf"] = int(ok.sum())
            except Exception as exc:                          # noqa: BLE001
                r["height_m"], r["bld_area_m2"], r["n_px_wsf"] = np.nan, np.nan, 0
                r["err_wsf"] = str(exc)[:80]
            try:
                z, _ = mask(sl, geom, crop=True, filled=True, nodata=0)
                zz = z[0].ravel()
                built = np.isin(zz, LCZ_BUILT)
                comp = np.isin(zz, LCZ_COMPACT)
                nb = int(built.sum())
                r["n_px_lcz_built"] = nb
                r["compact_share"] = float(comp.sum() / nb) if nb else np.nan
            except Exception as exc:                          # noqa: BLE001
                r["compact_share"], r["n_px_lcz_built"] = np.nan, 0
                r["err_lcz"] = str(exc)[:80]
            rows.append(r)
            if n % log_every == 0 or n == len(todo):
                print(f"      {n}/{len(todo)}  {time.time() - t0:6.1f}s",
                      flush=True)
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([done, pd.DataFrame(rows)],
                          ignore_index=True).to_parquet(CACHE, index=False)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([done, pd.DataFrame(rows)],
              ignore_index=True).to_parquet(CACHE, index=False)
    print(f"    wrote {len(rows)} rows -> {CACHE.relative_to(BASE)}")


def stage_validate() -> None:
    d = pd.read_parquet(CACHE)
    w = pd.read_parquet(SCHEMES)
    w["ghs_height_m"] = w.built_v / w.built_s
    d = d.drop_duplicates("fuacode")
    m = d.merge(w[["fuacode", "ghs_height_m", "built_v"]].drop_duplicates("fuacode"),
                on="fuacode", how="inner")
    m = m[m.height_m.notna() & m.ghs_height_m.notna()]
    print(f"    matched {len(m)} cities against the GHS height")
    print(f"    WSF3D height  median {m.height_m.median():5.2f} m   "
          f"5-95% {m.height_m.quantile(.05):5.2f}-{m.height_m.quantile(.95):5.2f}")
    print(f"    GHS   V/S     median {m.ghs_height_m.median():5.2f} m   "
          f"5-95% {m.ghs_height_m.quantile(.05):5.2f}-"
          f"{m.ghs_height_m.quantile(.95):5.2f}")
    r = float(m[["height_m", "ghs_height_m"]].corr(method="spearman").iloc[0, 1])
    print(f"    Spearman(WSF3D height, GHS V/S) = {r:+.3f}")
    ratio = float(m.height_m.median() / m.ghs_height_m.median())
    print(f"    median ratio WSF3D / GHS = {ratio:.2f}")
    # A factor-of-ten error is the failure this guards against; anything near
    # 10 or 0.1 means the decimetre assumption is wrong in one direction or the
    # other, and every weight built from it would be wrong too.
    if not 0.3 < ratio < 3.0:
        raise AssertionError(
            f"WSF3D height is {ratio:.2f} times the GHS height; the decimetre "
            "scale factor is wrong and the weights must not be built")
    if r < 0.3:
        raise AssertionError(
            "WSF3D height and GHS V/S rank cities differently (Spearman "
            f"{r:+.2f}); one of the two extractions is wrong")
    print(f"\n    compact share: median {d.compact_share.median():.3f}   "
          f"missing {int(d.compact_share.isna().sum())}")
    print("    scale and ranking both check out")


def stage_weights() -> None:
    d = pd.read_parquet(CACHE)
    w = pd.read_parquet(SCHEMES)
    d = d.drop_duplicates("fuacode")
    m = w.merge(d[["fuacode", "height_m", "compact_share", "bld_area_m2"]],
                on="fuacode", how="left")

    # Extensive form weights: how much stock a city has, times what form it is
    # in. See the module docstring for why the intensive variables cannot be
    # weights on their own.
    # NOTE ON THE SCHEMES TABLE. In fua_weight_schemes.parquet the columns
    # `built_s` and `built_v` hold the zonal sum for the whole FUA, repeated on
    # every city row that shares a fuacode; only `w_built_s` and `w_built_v` are
    # divided by n_share. `pop_weight_dedup` is already deduplicated. Summing or
    # weighting by the raw columns therefore double counts 34 cities in 5
    # countries -- 24 of them in China, which inflates its built volume per capita
    # by 31 % -- while leaving population correct. The deduplicated columns are
    # used everywhere below.
    m["w_tall"] = m.w_built_v * m.height_m
    m["w_compact"] = m.w_built_v * m.compact_share
    keep = ["city_id", "country", "pop_weight_dedup", "w_built_s",
            "w_built_v", "height_m", "compact_share", "w_tall", "w_compact"]
    out = m[keep]
    n_bad = int(out[["w_tall", "w_compact"]].isna().any(axis=1).sum())
    print(f"    {len(out)} cities, {n_bad} missing a form weight")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"    -> {OUT.relative_to(BASE)}")
    print("\n    how the four weights rank cities within country (Spearman):")
    for a, b in (("pop_weight_dedup", "w_built_v"),
                 ("w_built_v", "w_tall"),
                 ("w_built_v", "w_compact"), ("w_tall", "w_compact")):
        rs = []
        for _, s in out.groupby("country"):
            s2 = s[[a, b]].dropna()
            if len(s2) >= 5 and s2[a].std() > 0 and s2[b].std() > 0:
                rs.append(float(s2.corr(method="spearman").iloc[0, 1]))
        if rs:
            print(f"      {a:18s} vs {b:18s} median rho = {np.median(rs):+.3f}"
                  f"  ({len(rs)} countries)")
    print("    two weights that rank cities identically produce the same")
    print("    weighted mean, so a rho near 1 means the pair cannot be told")
    print("    apart by any national aggregate.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="extract",
                    choices=["extract", "validate", "weights", "all"])
    ap.add_argument("--limit", type=int, default=0,
                    help="polygons per run; 0 for all remaining")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    print("=" * 78)
    print("Urban form weights from WSF3D and LCZ")
    print("=" * 78)
    for p in (FUA_SHP, H_TIF, A_TIF, LCZ_TIF):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    if args.stage in ("extract", "all"):
        stage_extract(args.limit, args.log_every)
    if args.stage in ("validate", "all"):
        print("\n[validate]")
        stage_validate()
    if args.stage in ("weights", "all"):
        print("\n[weights]")
        stage_weights()


if __name__ == "__main__":
    main()
