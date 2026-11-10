#!/usr/bin/env python
"""
13_fua_population.py  --  population weights for city-weighted national exposure
================================================================================

Step 1 of the revised pipeline: zonal-sum GHS-POP over the FUA polygons and
compute, per country, how much of the urban population the sample represents.

WHY DEDUPLICATION MATTERS
-------------------------
`FUA_cities_final.shp` holds 1,583 records but only 1,563 distinct polygons:
34 records share 14 geometries. The worst case is Guangzhou, where EIGHT
separate Carbon Monitor cities (Guangzhou1 ... Guangzhou8, each with its own
emission series) were all matched to a single GHSL FUA polygon. Others are
outright matching errors -- "Jian, Jinan" are two different Chinese cities on
one polygon, and "Cairo, Al Qahirah" is one city entered twice.

Summing population per RECORD would therefore count the Guangzhou polygon
eight times and inflate its weight in China's national exposure. Population is
a property of the place, not of the emission record, so weights are computed
per distinct polygon (keyed on `fuacode`) and the collapsed city_ids are kept
alongside for the later join.

COVERAGE RATIO
--------------
The denominator is not national total population, which would make every
country look badly covered, and not FUA population, which exceeds urban-centre
population by construction (an FUA is a core plus its commuting zone). Instead:

    Coverage_c = pop(UCDB urban centres inside a sampled FUA)
               / pop(all UCDB urban centres in country c)

Numerator and denominator use the same 2020 raster and the same spatial units,
so the ratio is bounded by 1 and answers the question that matters: of this
country's urban-centre population, how much sits inside a city we sample?

INPUTS
------
  ShapefileData/FUA_cities_final.shp
  Other_data/Global Population/global_pop_{2019,2020,2021}_CN_1km_R2025A_UA_v1.tif
  Other_data/GHS_UCDB_GLOBE_R2024A_V1_2.zip   (urban-centre polygons, optional)

OUTPUTS
-------
  outputs/population/fua_population.parquet       fuacode x year -> population
  outputs/population/fua_city_weights.parquet     city_id -> weight (dedup-safe)
  outputs/population/country_coverage.parquet     country -> coverage ratio
  outputs/population/tables/*.csv                 audit tables

USAGE
-----
  python 13_fua_population.py
  python 13_fua_population.py --skip-coverage      # FUA sums only (~2.5 min)
"""

from __future__ import annotations

import argparse
import shutil
import time
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
FUA_SHP = BASE / "ShapefileData" / "FUA_cities_final.shp"
POP_DIR = BASE / "Other_data" / "Global Population"
UCDB_ZIP = BASE / "Other_data" / "GHS_UCDB_GLOBE_R2024A_V1_2.zip"
UCDB_GPKG_NAME = "GHS_UCDB_GLOBE_R2024A.gpkg"
UCDB_LAYER = "GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
OUT_DIR = BASE / "outputs" / "population"
TABLE_DIR = OUT_DIR / "tables"

YEARS = [2019, 2020, 2021]
COVERAGE_YEAR = 2020


def pop_path(year: int) -> Path:
    return POP_DIR / f"global_pop_{year}_CN_1km_R2025A_UA_v1.tif"


def zonal_sum(src, geoms, labels, log_every: int = 300) -> np.ndarray:
    """Population sum per polygon, with an all_touched fallback for tiny areas."""
    out = np.full(len(geoms), np.nan, dtype="float64")
    nd = src.nodata
    t0 = time.time()
    for i, geom in enumerate(geoms):
        try:
            arr, _ = mask(src, [geom], crop=True, filled=True, nodata=nd,
                          all_touched=False)
            a = arr[0]
            a = a[(a != nd) & np.isfinite(a)]
            if a.size == 0:                       # polygon smaller than a cell
                arr, _ = mask(src, [geom], crop=True, filled=True, nodata=nd,
                              all_touched=True)
                a = arr[0]
                a = a[(a != nd) & np.isfinite(a)]
            out[i] = a.sum() if a.size else 0.0
        except Exception:
            out[i] = np.nan
        if (i + 1) % log_every == 0:
            print(f"      {i + 1:,}/{len(geoms):,}  ({time.time() - t0:.0f}s)")
    return out


def load_fua() -> gpd.GeoDataFrame:
    g = gpd.read_file(FUA_SHP)
    g = (g.set_crs(4326) if g.crs is None else g.to_crs(4326)).reset_index(drop=True)
    g["city_id"] = g.index.astype(str).str.zfill(4)
    return g


def ucdb_gpkg_is_valid(gpkg: Path, expected_size: int | None = None) -> bool:
    """Check whether the extracted UCDB GeoPackage is complete and readable."""
    if not gpkg.exists():
        return False
    if expected_size is not None and gpkg.stat().st_size != expected_size:
        return False
    try:
        gpd.read_file(gpkg, layer=UCDB_LAYER, rows=1)
    except Exception:
        return False
    return True


def ensure_ucdb_gpkg(zip_path: Path, out_dir: Path) -> Path:
    """Extract the UCDB GeoPackage atomically; re-extract partial/corrupt files."""
    if not zip_path.exists():
        raise FileNotFoundError(f"UCDB zip not found: {zip_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / UCDB_GPKG_NAME
    tmp_gpkg = gpkg.with_suffix(gpkg.suffix + ".extracting")

    with zipfile.ZipFile(zip_path) as z:
        info = z.getinfo(UCDB_GPKG_NAME)
        expected_size = int(info.file_size)
        if ucdb_gpkg_is_valid(gpkg, expected_size):
            return gpkg

        if gpkg.exists():
            print(
                f"    existing UCDB geopackage is incomplete/corrupt "
                f"({gpkg.stat().st_size:,} != {expected_size:,} bytes); re-extracting ..."
            )
        else:
            print("    extracting UCDB geopackage ...")

        if tmp_gpkg.exists():
            tmp_gpkg.unlink()
        with z.open(info) as src, tmp_gpkg.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        tmp_gpkg.replace(gpkg)

    if not ucdb_gpkg_is_valid(gpkg, expected_size):
        raise RuntimeError(f"Extracted UCDB geopackage is still unreadable: {gpkg}")
    return gpkg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-coverage", action="store_true",
                    help="Skip the UCDB-based coverage ratio (saves ~6 min).")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 74)
    print("FUA population weights and urban coverage")
    print("=" * 74)

    # ---- 1. polygons, deduplicated -------------------------------------
    print("\n[1] FUA polygons")
    g = load_fua()
    n_rec = len(g)
    dup_groups = g[g.duplicated("fuacode", keep=False)]
    uniq = g.drop_duplicates("fuacode").reset_index(drop=True)
    print(f"    {n_rec:,} records -> {len(uniq):,} distinct polygons "
          f"({len(dup_groups)} records share {dup_groups.fuacode.nunique()} geometries)")
    if len(dup_groups):
        worst = (dup_groups.groupby("fuacode")
                 .agg(n=("city", "size"), cities=("city", lambda s: ", ".join(s.head(8))),
                      country=("country", "first"))
                 .sort_values("n", ascending=False).head(5))
        for code, r in worst.iterrows():
            print(f"      {code}  x{r.n}  {r.country}: {r.cities}")

    # ---- 2. zonal sums --------------------------------------------------
    print("\n[2] Zonal population sums")
    geoms = list(uniq.geometry)
    pop = {}
    for year in YEARS:
        p = pop_path(year)
        if not p.exists():
            print(f"    MISSING {p.name}, skipped")
            continue
        print(f"    {year} ...")
        with rasterio.open(p) as src:
            pop[year] = zonal_sum(src, geoms, uniq.fuacode)
        print(f"      total {np.nansum(pop[year]) / 1e6:,.1f} million "
              f"in {len(geoms):,} FUAs")

    fua = uniq[["fuacode", "city", "country", "shp_source"]].copy()
    for year, v in pop.items():
        fua[f"pop_{year}"] = v
    fua["pop_mean"] = fua[[f"pop_{y}" for y in pop]].mean(axis=1)

    # ---- 3. record-level weights (dedup-safe) ---------------------------
    # Every city_id keeps its polygon's population, and n_share records how
    # many emission records share that polygon, so downstream code can either
    # divide by n_share or drop duplicates explicitly.
    share = g.groupby("fuacode")["city_id"].size().rename("n_share")
    weights = (g[["city_id", "city", "country", "fuacode"]]
               .merge(fua.drop(columns=["city", "country"]), on="fuacode", how="left")
               .merge(share, on="fuacode", how="left"))
    weights["pop_weight_dedup"] = weights["pop_mean"] / weights["n_share"]

    # ---- 4. country aggregates -----------------------------------------
    print("\n[3] Country aggregates (distinct polygons only)")
    ctry = (fua.groupby("country")
            .agg(n_fua=("fuacode", "size"),
                 **{f"pop_{y}": (f"pop_{y}", "sum") for y in pop})
            .reset_index())
    ctry["pop_share_of_sample"] = ctry[f"pop_{COVERAGE_YEAR}"] / ctry[f"pop_{COVERAGE_YEAR}"].sum()
    print(f"    {len(ctry)} countries, sample total "
          f"{ctry[f'pop_{COVERAGE_YEAR}'].sum() / 1e6:,.0f} million")
    top = ctry.nlargest(8, f"pop_{COVERAGE_YEAR}")[["country", "n_fua", f"pop_{COVERAGE_YEAR}"]]
    for _, r in top.iterrows():
        print(f"      {r.country:20s} {int(r.n_fua):4d} FUA  "
              f"{r[f'pop_{COVERAGE_YEAR}'] / 1e6:8,.1f} M")

    # ---- 5. coverage ratio via UCDB -------------------------------------
    if not args.skip_coverage and UCDB_ZIP.exists():
        print("\n[4] Urban coverage ratio (UCDB urban centres)")
        tmp = OUT_DIR / "_ucdb"
        gpkg = ensure_ucdb_gpkg(UCDB_ZIP, tmp)
        uc = gpd.read_file(gpkg, layer=UCDB_LAYER)
        uc = uc.to_crs(4326)[["ID_UC_G0", "GC_UCN_MAI_2025", "GC_CNT_GAD_2025", "geometry"]]
        uc.columns = ["uc_id", "uc_name", "uc_country", "geometry"]
        print(f"    {len(uc):,} urban centres")

        p = pop_path(COVERAGE_YEAR)
        print(f"    zonal sums for {COVERAGE_YEAR} ...")
        with rasterio.open(p) as src:
            uc["pop"] = zonal_sum(src, list(uc.geometry), uc.uc_id, log_every=2000)

        # which urban centres sit inside a sampled FUA
        inside = gpd.sjoin(uc[["uc_id", "geometry"]],
                           uniq[["fuacode", "geometry"]],
                           how="inner", predicate="intersects")
        uc["in_sample"] = uc.uc_id.isin(inside.uc_id).astype(int)

        cov = (uc.groupby("uc_country")
               .apply(lambda d: pd.Series({
                   "uc_pop_total": d["pop"].sum(),
                   "uc_pop_in_sample": d.loc[d.in_sample == 1, "pop"].sum(),
                   "n_uc": len(d),
                   "n_uc_in_sample": int(d.in_sample.sum())}))
               .reset_index().rename(columns={"uc_country": "country_gadm"}))
        cov["coverage"] = cov.uc_pop_in_sample / cov.uc_pop_total.replace(0, np.nan)
        cov.to_parquet(OUT_DIR / "country_coverage.parquet", index=False)

        cm = cov[cov.n_uc_in_sample > 0].nlargest(12, "uc_pop_total")
        print(f"\n    {'country':22s}{'UC pop (M)':>12s}{'in sample (M)':>14s}{'coverage':>10s}")
        for _, r in cm.iterrows():
            print(f"    {str(r.country_gadm)[:20]:22s}{r.uc_pop_total / 1e6:12,.1f}"
                  f"{r.uc_pop_in_sample / 1e6:14,.1f}{r.coverage:10.3f}")
        print(f"\n    countries with coverage >= 0.50: "
              f"{int((cov.coverage >= 0.5).sum())} / {int(cov.n_uc_in_sample.gt(0).sum())} sampled")
    else:
        print("\n[4] Coverage ratio skipped")

    # ---- 6. save ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fua.to_parquet(OUT_DIR / "fua_population.parquet", index=False)
    weights.to_parquet(OUT_DIR / "fua_city_weights.parquet", index=False)
    ctry.to_parquet(OUT_DIR / "country_population.parquet", index=False)
    fua.to_csv(TABLE_DIR / "fua_population.csv", index=False)
    ctry.to_csv(TABLE_DIR / "country_population.csv", index=False)

    print("\n[5] Saved")
    for f in ("fua_population", "fua_city_weights", "country_population"):
        pth = OUT_DIR / f"{f}.parquet"
        if pth.exists():
            print(f"    {f + '.parquet':30s} {pth.stat().st_size / 1024:7.0f} KB")

    print("\n[6] Sanity check - largest sampled FUAs by population")
    for _, r in fua.nlargest(8, "pop_mean")[["city", "country", "pop_mean"]].iterrows():
        print(f"    {r.city[:26]:28s}{r.country[:16]:18s}{r.pop_mean / 1e6:7.2f} M")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
