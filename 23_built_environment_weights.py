#!/usr/bin/env python
"""
23_built_environment_weights.py  --  competing city weights for R6
===================================================================

Zonal-sums GHS-BUILT-S (built-up surface, m2) and GHS-BUILT-V (built-up volume,
m3) over the FUA polygons and assembles the alternative weighting schemes that
R6 puts in competition against population weighting.

WHAT THESE VARIABLES ARE, AND WHAT THEY ARE NOT
-----------------------------------------------
GHS_BUILT_S is the area of built-up SURFACE per cell and GHS_BUILT_V is the
gross built-up VOLUME per cell. Neither is the thermal envelope area A in
Q = U A dT, and this dataset contains no heat-transfer coefficient U, no indoor
set-point, and no operating hours. If a built-environment weight out-predicts a
population weight, the only claim available is that the result is CONSISTENT
WITH built-environment scale modulating heat-sensitive energy demand. The
script prints that phrasing next to the numbers so it cannot drift in the
writing.

A WEIGHTING SCHEME MUST VARY WITHIN A COUNTRY
---------------------------------------------
National exposure is a weighted mean over the cities of one country, so any
weight that is constant within a country cancels exactly:

    sum_i (k_c w_i x_i) / sum_i (k_c w_i)  ==  sum_i (w_i x_i) / sum_i (w_i)

Air-conditioning penetration is only available per country, so it CANNOT be a
weighting scheme, however physically relevant it is. It is therefore carried
forward to R6 as a country-level moderator (an interaction with exposure), not
as a weight, and `AuxData/ac_penetration_country.csv` is passed through
unchanged. Treating it as a weight would have produced a "fifth scheme" whose
results were algebraically identical to population weighting.

FIVE SCHEMES
------------
  equal        every sampled city counts once      no size information
  pop          GHS-POP 2020                        the study's baseline
  built_s      GHS-BUILT-S 2020                    built-up surface
  built_v      GHS-BUILT-V 2020                    built-up volume
  pop_vint     pop x (volume per capita, normalised within country)
               tilts weight toward capital-intensive cities without letting
               absolute city size enter twice

DEDUPLICATION
-------------
As in 13_fua_population.py: 1,583 emission records share 1,563 distinct
polygons (eight Carbon Monitor cities named Guangzhou1..8 share one FUA), so
zonal sums are computed per distinct `fuacode` and divided by the number of
records sharing it.

INPUTS
------
  ShapefileData/FUA_cities_final.shp
  Other_data/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000_V1_0.zip
  Other_data/GHS_BUILT_V_E2020_GLOBE_R2023A_54009_1000_V1_0.zip
  outputs/population/fua_city_weights.parquet
  AuxData/ac_penetration_country.csv

OUTPUTS
-------
  outputs/population/fua_built_environment.parquet
  outputs/population/fua_weight_schemes.parquet
  Results/R6/tables/T9_weight_schemes_summary.csv

USAGE
-----
  python 23_built_environment_weights.py
"""

from __future__ import annotations

import shutil
import time
import warnings
import zipfile
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
# Results root. Overridable with the HC_RESULTS_ROOT environment variable so an
# extended-period rerun can write a parallel result set instead of overwriting
# the published one. Unset, it resolves to the original BASE/"Results".
RESULTS_ROOT = Path(os.environ.get("HC_RESULTS_ROOT", str(BASE / "Results")))

FUA_SHP = BASE / "ShapefileData" / "FUA_cities_final.shp"
OTHER = BASE / "Other_data"
WEIGHTS = BASE / "outputs" / "population" / "fua_city_weights.parquet"
AC_CSV = BASE / "AuxData" / "ac_penetration_country.csv"
OUT_DIR = BASE / "outputs" / "population"
TAB_DIR = RESULTS_ROOT / "R6" / "tables"
CACHE = BASE / "outputs" / "_ghs_cache"

PRODUCTS = {
    "built_s": "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000_V1_0",
    "built_v": "GHS_BUILT_V_E2020_GLOBE_R2023A_54009_1000_V1_0",
}


def ensure_tif(key: str) -> Path:
    """Extract the GeoTIFF once, atomically, and cache it."""
    stem = PRODUCTS[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    tif = CACHE / f"{stem}.tif"
    zpath = OTHER / f"{stem}.zip"
    if not zpath.exists():
        raise FileNotFoundError(zpath)
    with zipfile.ZipFile(zpath) as z:
        info = z.getinfo(f"{stem}.tif")
        if tif.exists() and tif.stat().st_size == info.file_size:
            return tif
        print(f"    extracting {stem}.tif ({info.file_size / 1e6:.0f} MB) ...")
        tmp = tif.with_suffix(".extracting")
        with z.open(info) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
        tmp.replace(tif)
    return tif


def zonal_sum(src, geoms, log_every: int = 300) -> np.ndarray:
    out = np.full(len(geoms), np.nan, dtype="float64")
    nd = src.nodata
    t0 = time.time()
    for i, geom in enumerate(geoms):
        try:
            arr, _ = mask(src, [geom], crop=True, filled=True, nodata=nd,
                          all_touched=False)
            a = arr[0].astype("float64")
            a = a[np.isfinite(a)]
            if nd is not None:
                a = a[a != nd]
            if a.size == 0:
                arr, _ = mask(src, [geom], crop=True, filled=True, nodata=nd,
                              all_touched=True)
                a = arr[0].astype("float64")
                a = a[np.isfinite(a)]
                if nd is not None:
                    a = a[a != nd]
            out[i] = a.sum() if a.size else 0.0
        except Exception:
            out[i] = np.nan
        if (i + 1) % log_every == 0:
            print(f"      {i + 1:,}/{len(geoms):,}  ({time.time() - t0:.0f}s)")
    return out


def main() -> None:
    t0 = time.time()
    print("=" * 76)
    print("R6: built-environment weights over FUA polygons")
    print("=" * 76)

    print("\n[1] FUA polygons")
    g = gpd.read_file(FUA_SHP)
    g = (g.set_crs(4326) if g.crs is None else g.to_crs(4326)).reset_index(drop=True)
    g["city_id"] = g.index.astype(str).str.zfill(4)
    uniq = g.drop_duplicates("fuacode").reset_index(drop=True)
    n_share = g.groupby("fuacode").size().rename("n_share")
    print(f"    {len(g):,} records -> {len(uniq):,} distinct polygons")

    built = pd.DataFrame({"fuacode": uniq["fuacode"]})
    for key in PRODUCTS:
        print(f"\n[2] Zonal sums: {key}")
        tif = ensure_tif(key)
        with rasterio.open(tif) as src:
            print(f"    raster CRS {src.crs.to_string()}, "
                  f"{src.width:,} x {src.height:,}, nodata={src.nodata}")
            geoms = list(uniq.to_crs(src.crs).geometry)
            built[key] = zonal_sum(src, geoms)
        v = built[key]
        print(f"    total {v.sum():,.0f}   median {v.median():,.0f}   "
              f"NaN {int(v.isna().sum())}")

    print("\n[3] Assembling weight schemes")
    w = pd.read_parquet(WEIGHTS)
    w["city_id"] = w["city_id"].astype(str).str.zfill(4)
    if "pop_weight_dedup" not in w.columns:
        w["pop_weight_dedup"] = w["pop_mean"] / w["n_share"]
    keep = [c for c in ("city_id", "country", "fuacode", "n_share", "pop_mean",
                        "pop_weight_dedup") if c in w.columns]
    w = w[keep]
    if "fuacode" not in w.columns:
        w = w.merge(g[["city_id", "fuacode"]], on="city_id", how="left")
    if "n_share" not in w.columns:
        w = w.merge(n_share, on="fuacode", how="left")

    d = w.merge(built, on="fuacode", how="left")
    for key in PRODUCTS:
        d[key + "_weight"] = d[key] / d["n_share"]

    d["w_equal"] = 1.0
    d["w_pop"] = d["pop_weight_dedup"]
    d["w_built_s"] = d["built_s_weight"]
    d["w_built_v"] = d["built_v_weight"]

    # volume intensity: built volume per capita, normalised to mean 1 inside
    # each country so it re-ranks cities without re-introducing absolute size
    d["vol_per_cap"] = d["built_v_weight"] / d["pop_weight_dedup"].replace(0, np.nan)
    med = d.groupby("country")["vol_per_cap"].transform("median")
    d["vint"] = (d["vol_per_cap"] / med).clip(0.2, 5.0)
    d["w_pop_vint"] = d["w_pop"] * d["vint"]

    schemes = ["w_equal", "w_pop", "w_built_s", "w_built_v", "w_pop_vint"]
    bad = d[schemes].isna().any(axis=1) | (d[schemes] <= 0).all(axis=1)
    print(f"    {len(d):,} city records, {int(bad.sum())} with an unusable weight")
    for s in schemes:
        n_bad = int((d[s].isna() | (d[s] <= 0)).sum())
        if n_bad:
            print(f"      {s}: {n_bad} missing/zero -> filled from population")
            d[s] = d[s].where(d[s] > 0, d["w_pop"])

    print("\n[4] How different are the schemes really?")
    print("    within-country Spearman correlation with population weight")
    for s in schemes:
        if s == "w_pop":
            continue
        rs = []
        for c, grp in d.groupby("country"):
            if len(grp) < 4:
                continue
            rs.append(grp[["w_pop", s]].rank().corr().iloc[0, 1])
        if rs:
            print(f"      {s:14s} median rho = {np.median(rs):+.3f}  "
                  f"(n = {len(rs)} countries with >= 4 cities)")
    print("    a scheme correlated ~1.0 with population cannot produce a")
    print("    different national exposure series, whatever it measures")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    built.to_parquet(OUT_DIR / "fua_built_environment.parquet", index=False)
    out_cols = ["city_id", "country", "fuacode", "n_share", "pop_weight_dedup",
                "built_s", "built_v", "vol_per_cap", "vint"] + schemes
    d[[c for c in out_cols if c in d.columns]].to_parquet(
        OUT_DIR / "fua_weight_schemes.parquet", index=False)

    summ = d.groupby("country")[schemes].sum()
    summ["n_cities"] = d.groupby("country").size()
    summ.reset_index().to_csv(TAB_DIR / "T9_weight_schemes_summary.csv",
                              index=False)
    print(f"\n    outputs/population/fua_weight_schemes.parquet")
    print(f"    R6/tables/T9_weight_schemes_summary.csv")

    if AC_CSV.exists():
        ac = pd.read_csv(AC_CSV)
        print(f"\n[5] AC penetration available for {len(ac)} countries -- carried")
        print("    to R6 as a country-level MODERATOR, not a weight: a weight")
        print("    that is constant within a country cancels out of the")
        print("    weighted mean exactly.")

    print("\n    WORDING: a built-environment weight winning this comparison is")
    print("    consistent with built-environment scale modulating heat-sensitive")
    print("    energy demand. GHS_BUILT_S/V are not envelope area, and no U,")
    print("    set-point or operating hours exist in this data.")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
