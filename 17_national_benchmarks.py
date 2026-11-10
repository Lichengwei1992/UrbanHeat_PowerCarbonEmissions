#!/usr/bin/env python
"""
17_national_benchmarks.py  --  R2 benchmark exposures (needs country borders)
==============================================================================

Builds the two national-scale exposure benchmarks the review demanded, so the
city-weighted series is tested against a serious opponent rather than a straw
man:

  1. area-weighted national mean temperature      the naive benchmark
  2. population-weighted national mean temperature the established method
  3. FUA equal-weighted exposure                   already in 14's output
  4. FUA population-weighted exposure              already in 14's output

Population-weighted national temperature is a mature literature in its own
right, so the contribution of this study cannot be the weighting -- it has to
be the pairing of city-resolved exposure with high-frequency sectoral carbon.
Benchmarks 1 and 2 exist precisely so that claim can be tested rather than
asserted.

METHOD
------
Rasterising once is far cheaper than 1,096 daily zonal statistics. GADM level-0
polygons are burned onto the ERA5 0.25-degree grid, GHS-POP is aggregated from
1 km to the same grid, and each country-day mean becomes a single sparse
matrix product

    T_(country x day) = W_(country x cell) @ T_(cell x day)

with W holding either uniform (area) or population weights, normalised per
country. The whole 3-year global panel then collapses in seconds.

INPUTS
------
  gadm_410-levels/gadm_410-levels.gpkg          layer ADM_0
  ERA5/temperature/era5_temp_daily_{year}.nc    global daily t2m_max / t2m_min
  Other_data/Global Population/global_pop_2020_CN_1km_R2025A_UA_v1.tif

OUTPUT
------
  outputs/exposure/national_benchmark_exposure.parquet
      country | date | nat_area_tmax | nat_pop_tmax | nat_area_scdh24 | ...

USAGE
-----
  python 17_national_benchmarks.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
# Scratch cache. Kept inside the project rather than /tmp so the pipeline runs
# unchanged on Windows, where "/tmp" resolves to a non-existent <drive>\\tmp.
CACHE_DIR = BASE / "outputs" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GADM = CACHE_DIR / "gadm0.gpkg"                    # cached by the caller
GADM_SRC = BASE / "gadm_410-levels" / "gadm_410-levels.gpkg"
ERA5_DIR = BASE / "ERA5" / "temperature"
POP_TIF = BASE / "Other_data" / "Global Population" / "global_pop_2020_CN_1km_R2025A_UA_v1.tif"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
OUT = BASE / "outputs" / "exposure" / "national_benchmark_exposure.parquet"

YEARS = [2019, 2020, 2021]
BASE_C = 24.0            # cooling degree base, matches SCDH24
BASE_H = 15.0            # heating degree base, matches HDH15

# GADM COUNTRY -> Carbon Monitor country label
NAME_MAP = {
    "México": "Mexico", "South Korea": "Korea", "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Viet Nam": "Vietnam",
}


def load_grid(year: int):
    ds = xr.open_dataset(ERA5_DIR / f"era5_temp_daily_{year}.nc")
    lat = ds.latitude.values
    lon = ds.longitude.values                      # 0..360
    return ds, lat, lon


GRID_CACHE = CACHE_DIR / "gadm_grid.npy"
META_CACHE = CACHE_DIR / "gadm_meta.pkl"
SIMP_CACHE = CACHE_DIR / "gadm0_simp.gpkg"
SIMPLIFY_TOL = 0.05      # degrees; far below the 0.25 deg target grid


def build_country_raster(lat, lon):
    """Burn GADM level-0 onto the ERA5 grid; returns (code array, names, transform).

    GADM 4.1 geometries are extremely detailed -- rasterising them raw takes
    many minutes. They are simplified to 0.05 degrees first, which is invisible
    at the 0.25 degree target resolution, and both the simplified layer and the
    burned grid are cached so this runs once.
    """
    import pickle
    if GRID_CACHE.exists() and META_CACHE.exists():
        arr = np.load(GRID_CACHE)
        with META_CACHE.open("rb") as fh:
            meta = pickle.load(fh)
        print("    loaded cached country grid")
        return arr, meta["inv"], meta["transform"]

    import geopandas as gpd
    from shapely.affinity import translate

    if SIMP_CACHE.exists():
        g = gpd.read_file(SIMP_CACHE)
    else:
        g = gpd.read_file(GADM if GADM.exists() else GADM_SRC,
                          layer=None if GADM.exists() else "ADM_0").to_crs(4326)
        g["geometry"] = g.geometry.simplify(SIMPLIFY_TOL)
        g[["GID_0", "COUNTRY", "geometry"]].to_file(SIMP_CACHE, driver="GPKG")

    # ERA5 longitudes run 0..360; duplicate western geometries shifted by +360
    g2 = g.copy()
    west = g2.geometry.bounds.minx < 0
    g2.loc[west, "geometry"] = g2.loc[west, "geometry"].apply(
        lambda gm: translate(gm, xoff=360))
    both = pd.concat([g, g2], ignore_index=True)

    res_lat = abs(float(lat[1] - lat[0]))
    res_lon = abs(float(lon[1] - lon[0]))
    transform = from_origin(float(lon.min()) - res_lon / 2,
                            float(lat.max()) + res_lat / 2, res_lon, res_lat)
    codes = {name: i + 1 for i, name in enumerate(sorted(g.COUNTRY.unique()))}
    shapes = [(geom, codes[nm]) for geom, nm in zip(both.geometry, both.COUNTRY)
              if geom is not None and not geom.is_empty]
    arr = rasterize(shapes, out_shape=(len(lat), len(lon)), transform=transform,
                    fill=0, dtype="int32", all_touched=False)
    inv = {v: k for k, v in codes.items()}
    np.save(GRID_CACHE, arr)
    with META_CACHE.open("wb") as fh:
        pickle.dump({"inv": inv, "transform": transform}, fh)
    return arr, inv, transform


POP_CACHE = CACHE_DIR / "pop_on_era5_grid.npy"


def population_on_grid(lat, lon, transform):
    """Aggregate the 1 km population raster onto the ERA5 grid (sum-preserving).

    GHS-POP is stored on -180..180 while ERA5 runs 0..360. Reprojecting
    directly onto the ERA5 transform silently drops the entire western
    hemisphere -- the Americas came out with zero population and NaN weights.
    The raster is therefore reprojected onto a matching -180..180 grid and then
    rolled into ERA5's longitude order.
    """
    if POP_CACHE.exists():
        print("    loaded cached population grid")
        return np.load(POP_CACHE)

    res_lat = abs(float(lat[1] - lat[0]))
    res_lon = abs(float(lon[1] - lon[0]))
    n_lon = len(lon)
    tr180 = from_origin(-180.0 - res_lon / 2, float(lat.max()) + res_lat / 2,
                        res_lon, res_lat)
    with rasterio.open(POP_TIF) as src:
        dst = np.zeros((len(lat), n_lon), dtype="float64")
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=tr180, dst_crs="EPSG:4326",
                  resampling=Resampling.sum, src_nodata=src.nodata,
                  dst_nodata=0.0)
    dst[~np.isfinite(dst)] = 0.0
    dst[dst < 0] = 0.0
    # -180..180 -> 0..360: the cell holding lon 0 sits at index n_lon // 2
    dst = np.roll(dst, -(n_lon // 2), axis=1)
    np.save(POP_CACHE, dst)
    return dst


def main() -> None:
    # Study years are parameterised so the benchmark panel can be extended
    # alongside the city panel. The default reproduces the original run.
    global YEARS
    import argparse
    ap = argparse.ArgumentParser(
        description="Build area- and population-weighted national exposure.")
    ap.add_argument("--years", nargs="+", type=int, default=YEARS,
                    help="Years to build, e.g. --years 2019 2020 2021 2022")
    args = ap.parse_args()
    YEARS = sorted(args.years)

    t0 = time.time()
    print("=" * 76)
    print("R2 benchmarks: area- and population-weighted national exposure")
    print(f"  years: {YEARS[0]}-{YEARS[-1]}")
    print("=" * 76)

    ds, lat, lon = load_grid(YEARS[0])
    print(f"\n[1] ERA5 grid {len(lat)} x {len(lon)}")

    print("[2] Rasterising GADM level-0 onto the ERA5 grid")
    cty, inv, transform = build_country_raster(lat, lon)
    n_cells = int((cty > 0).sum())
    print(f"    {len(inv)} countries, {n_cells:,} land cells assigned")

    print("[3] Aggregating GHS-POP 1 km -> 0.25 deg")
    pop = population_on_grid(lat, lon, transform)
    print(f"    total {pop.sum()/1e9:.3f} billion on grid")

    # target countries only
    ex = pd.read_parquet(EXPOSURE)
    targets = sorted(ex.country.unique())
    rev = {}
    for code, nm in inv.items():
        cm = NAME_MAP.get(nm, nm)
        if cm in targets:
            rev.setdefault(cm, []).append(code)
    missing = [c for c in targets if c not in rev]
    print(f"[4] Matched {len(rev)}/{len(targets)} countries"
          + (f"  | unmatched: {missing}" if missing else ""))

    flat_cty = cty.ravel()
    flat_pop = pop.ravel()
    idx = {c: np.flatnonzero(np.isin(flat_cty, codes)) for c, codes in rev.items()}

    # Slice the global field down to the union of target cells once. Keeping
    # the full 721 x 1440 x 365 array in memory three times over (T, SCDH, HDH)
    # is ~4.5 GB and thrashes; the target countries cover a small fraction of it.
    needed = np.unique(np.concatenate([v for v in idx.values() if v.size]))
    pos = {cell: k for k, cell in enumerate(needed)}
    local = {c: np.array([pos[x] for x in ii], dtype=np.int64)
             for c, ii in idx.items() if ii.size}
    print(f"    reduced to {len(needed):,} cells "
          f"({100 * len(needed) / (len(lat) * len(lon)):.1f}% of the globe)")

    frames = []
    for year in YEARS:
        cache = CACHE_DIR / f"nat_bench_{year}.parquet"
        if cache.exists():
            print(f"[5] {year}: cached")
            frames.append(pd.read_parquet(cache))
            continue
        dsy, _, _ = load_grid(year)
        times = pd.to_datetime(dsy.valid_time.values)
        print(f"[5] {year}: {len(times)} days")
        # Read in time chunks: `.values` on the whole year would materialise
        # 1.5 GB per variable before any slicing takes place.
        nt = len(times)
        T = np.empty((nt, len(needed)), dtype="float32")
        N = np.empty((nt, len(needed)), dtype="float32")
        STEP = 30
        for s in range(0, nt, STEP):
            e = min(s + STEP, nt)
            T[s:e] = dsy["t2m_max"].isel(valid_time=slice(s, e)).values.reshape(
                e - s, -1)[:, needed]
            N[s:e] = dsy["t2m_min"].isel(valid_time=slice(s, e)).values.reshape(
                e - s, -1)[:, needed]
        dsy.close()
        if np.nanmean(T) > 150:
            T -= 273.15
            N -= 273.15
        scdh = np.clip(T - BASE_C, 0, None) * 24.0     # degree-hours proxy
        hdh = np.clip(BASE_H - N, 0, None) * 24.0
        pop_sub = flat_pop[needed]

        parts = []
        for c, jj in local.items():
            w_pop = pop_sub[jj]
            sw = w_pop.sum()
            rec = {"country": c, "date": times}
            for arr, nm in ((T, "tmax"), (scdh, "scdh24"), (hdh, "hdh15")):
                sub = arr[:, jj]
                rec[f"nat_area_{nm}"] = np.nanmean(sub, axis=1)
                rec[f"nat_pop_{nm}"] = ((sub @ w_pop) / sw if sw > 0
                                        else np.full(len(times), np.nan))
            parts.append(pd.DataFrame(rec))
        yr = pd.concat(parts, ignore_index=True)
        yr.to_parquet(cache, index=False)
        frames.append(yr)
        del T, N, scdh, hdh

    out = pd.concat(frames, ignore_index=True).sort_values(["country", "date"])
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(OUT)
    print(f"\n[6] Saved {OUT.name}: {len(out):,} country-days x {out.shape[1]} cols")

    print("\n[7] Sanity - 2020 annual means")
    s = out[out.date.dt.year == 2020].groupby("country").agg(
        area_tmax=("nat_area_tmax", "mean"), pop_tmax=("nat_pop_tmax", "mean"),
        area_scdh=("nat_area_scdh24", "mean"), pop_scdh=("nat_pop_scdh24", "mean"))
    s["pop_minus_area"] = s.pop_tmax - s.area_tmax
    for c in ["United States", "China", "India", "Russia", "Australia", "Japan",
              "France", "Brazil"]:
        if c in s.index:
            r = s.loc[c]
            print(f"    {c:16s} area {r.area_tmax:5.1f}C  pop {r.pop_tmax:5.1f}C  "
                  f"diff {r.pop_minus_area:+5.2f}C   SCDH area {r.area_scdh:6.1f} "
                  f"pop {r.pop_scdh:6.1f}")
    print(f"\n    population weighting shifts national Tmax by "
          f"{s.pop_minus_area.mean():+.2f} C on average "
          f"(range {s.pop_minus_area.min():+.2f} .. {s.pop_minus_area.max():+.2f})")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
