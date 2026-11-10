"""
03b_build_gadm_shapefiles.py
============================
Download GADM Level-2 boundaries for China (prefectures) and US (counties),
then match to Carbon Monitor Cities entries and output shapefiles.

Run AFTER 03_build_city_shapefiles.py.
Requires: pip install geopandas fiona pyogrio
"""

import os, sys, zipfile, urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

try:
    import geopandas as gpd
except ImportError:
    sys.exit("Install geopandas: pip install geopandas pyogrio fiona")

BASE     = Path(__file__).parent
CITY_DIR = BASE / "CarbonData" / "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide"
AUX_DIR  = BASE / "AuxData"
SHP_DIR  = BASE / "ShapefileData"
OUT_DIR  = BASE / "outputs"
for d in [AUX_DIR, SHP_DIR, OUT_DIR]:
    d.mkdir(exist_ok=True)


def download(url, dest, desc=""):
    if Path(dest).exists():
        print(f"  ✓ {Path(dest).name} already downloaded")
        return
    print(f"  ↓ Downloading {desc or Path(dest).name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"    → {dest}")


def fuzzy_best(name, candidates, cutoff=0.72):
    name_l = name.lower().strip()
    best_s, best_c = 0, None
    for c in candidates:
        s = SequenceMatcher(None, name_l, c.lower().strip()).ratio()
        if s > best_s:
            best_s, best_c = s, c
    return (best_c, best_s) if best_s >= cutoff else (None, 0)


# ═════════════════════════════════════════════════════════════════════════════
# A. CHINA GADM L2 — Prefecture-level cities
# ═════════════════════════════════════════════════════════════════════════════
def build_china_gadm():
    gadm_zip = AUX_DIR / "gadm41_CHN_shp.zip"
    gadm_dir = SHP_DIR / "GADM_CHN"
    gadm_dir.mkdir(exist_ok=True)

    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_CHN_shp.zip"
    download(url, gadm_zip, "GADM China L2 (~95 MB)")
    with zipfile.ZipFile(gadm_zip) as z:
        z.extractall(gadm_dir)

    shp = next(gadm_dir.rglob("*_2.shp"), None)
    if not shp:
        raise FileNotFoundError(
            "GADM China L2 shapefile not found.\n"
            "Download manually from https://gadm.org/download_country.html\n"
            f"Select China → Shapefile → extract to {gadm_dir}"
        )

    gadm = gpd.read_file(shp).to_crs(epsg=4326)
    print(f"  GADM China L2: {len(gadm)} prefectures")
    print(f"  Columns: {[c for c in gadm.columns if c != 'geometry']}")

    # NAME_2 = prefecture name in pinyin; VARNAME_2 = alternative names
    gadm['_name2_l']  = gadm['NAME_2'].str.lower().str.strip()
    gadm['_varnames'] = gadm.get('VARNAME_2', pd.Series([''] * len(gadm))).fillna('').str.lower()

    # Carbon Monitor China GADM file
    cn_file = CITY_DIR / "carbon-monitor-cities-China-GADM-prefecture-v0325.csv"
    cn = pd.read_csv(cn_file, encoding='utf-8-sig', usecols=['city', 'country']).drop_duplicates()
    cn['city'] = cn['city'].str.strip()
    print(f"  Carbon Monitor China GADM cities: {len(cn)}")

    matched, unmatched = [], []
    all_names = gadm['_name2_l'].tolist()

    for _, row in cn.iterrows():
        city = row['city']

        # 1. Exact match on NAME_2
        exact = gadm[gadm['_name2_l'] == city.lower().strip()]
        if not exact.empty:
            r = exact.iloc[0]
            matched.append({'city': city, 'country': 'China', 'state': r['NAME_1'],
                'boundary_t': 'GADM_prefecture', 'shp_source': 'GADM_CHN_L2',
                'gadm_l2': r['NAME_2'], 'gadm_l1': r['NAME_1'],
                'match_meth': 'exact', 'match_scor': 1.0, 'geometry': r['geometry']})
            continue

        # 2. Match against VARNAME_2 (alternative/English names)
        var_match = gadm[gadm['_varnames'].str.contains(city.lower(), regex=False, na=False)]
        if not var_match.empty:
            r = var_match.iloc[0]
            matched.append({'city': city, 'country': 'China', 'state': r['NAME_1'],
                'boundary_t': 'GADM_prefecture', 'shp_source': 'GADM_CHN_L2',
                'gadm_l2': r['NAME_2'], 'gadm_l1': r['NAME_1'],
                'match_meth': 'varname', 'match_scor': 0.95, 'geometry': r['geometry']})
            continue

        # 3. Fuzzy match on NAME_2
        best_name, score = fuzzy_best(city, all_names, cutoff=0.70)
        if best_name:
            r = gadm[gadm['_name2_l'] == best_name].iloc[0]
            matched.append({'city': city, 'country': 'China', 'state': r['NAME_1'],
                'boundary_t': 'GADM_prefecture', 'shp_source': 'GADM_CHN_L2',
                'gadm_l2': r['NAME_2'], 'gadm_l1': r['NAME_1'],
                'match_meth': 'fuzzy', 'match_scor': round(score, 3), 'geometry': r['geometry']})
        else:
            unmatched.append(city)

    pct = len(matched) / len(cn) * 100
    print(f"  Matched: {len(matched)}/{len(cn)} ({pct:.1f}%)")
    if unmatched:
        print(f"  Unmatched: {unmatched[:15]}")

    gdf = gpd.GeoDataFrame(matched, geometry='geometry', crs='EPSG:4326')
    out = SHP_DIR / "city_boundaries_China_GADM.shp"
    gdf.to_file(out)
    print(f"  ✓ Saved → {out}")
    pd.DataFrame({'city': unmatched}).to_csv(OUT_DIR / "unmatched_China_GADM.csv", index=False)
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# B. US GADM L2 — Counties
# ═════════════════════════════════════════════════════════════════════════════
def build_us_gadm():
    gadm_zip = AUX_DIR / "gadm41_USA_shp.zip"
    gadm_dir = SHP_DIR / "GADM_USA"
    gadm_dir.mkdir(exist_ok=True)

    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_USA_shp.zip"
    download(url, gadm_zip, "GADM USA L2 (~170 MB)")
    with zipfile.ZipFile(gadm_zip) as z:
        z.extractall(gadm_dir)

    shp = next(gadm_dir.rglob("*_2.shp"), None)
    if not shp:
        raise FileNotFoundError(
            "GADM USA L2 shapefile not found.\n"
            "Download manually from https://gadm.org/download_country.html\n"
            f"Select United States → Shapefile → extract to {gadm_dir}"
        )

    gadm = gpd.read_file(shp).to_crs(epsg=4326)
    print(f"  GADM USA L2: {len(gadm)} counties")
    gadm['_county_l'] = gadm['NAME_2'].str.lower().str.strip()
    gadm['_state_l']  = gadm['NAME_1'].str.lower().str.strip()

    # US state abbreviation → full name
    STATE_FULL = {
        'AL':'alabama','AK':'alaska','AZ':'arizona','AR':'arkansas','CA':'california',
        'CO':'colorado','CT':'connecticut','DE':'delaware','FL':'florida','GA':'georgia',
        'HI':'hawaii','ID':'idaho','IL':'illinois','IN':'indiana','IA':'iowa',
        'KS':'kansas','KY':'kentucky','LA':'louisiana','ME':'maine','MD':'maryland',
        'MA':'massachusetts','MI':'michigan','MN':'minnesota','MS':'mississippi',
        'MO':'missouri','MT':'montana','NE':'nebraska','NV':'nevada','NH':'new hampshire',
        'NJ':'new jersey','NM':'new mexico','NY':'new york','NC':'north carolina',
        'ND':'north dakota','OH':'ohio','OK':'oklahoma','OR':'oregon','PA':'pennsylvania',
        'RI':'rhode island','SC':'south carolina','SD':'south dakota','TN':'tennessee',
        'TX':'texas','UT':'utah','VT':'vermont','VA':'virginia','WA':'washington',
        'WV':'west virginia','WI':'wisconsin','WY':'wyoming','DC':'district of columbia',
    }

    us_file = CITY_DIR / "carbon-monitor-cities-US-Counties-v0325.csv"
    us = pd.read_csv(us_file, encoding='utf-8-sig',
                     usecols=['county', 'state']).drop_duplicates()
    us['county'] = us['county'].str.strip()
    us['state']  = us['state'].str.strip()
    print(f"  Carbon Monitor US counties: {len(us)}")

    matched, unmatched = [], []
    for _, row in us.iterrows():
        county     = row['county']
        state_abbr = row['state']
        state_full = STATE_FULL.get(state_abbr, state_abbr).lower()

        sub = gadm[gadm['_state_l'] == state_full]
        if sub.empty:
            # Try fuzzy state match
            best_state, _ = fuzzy_best(state_full, gadm['_state_l'].unique().tolist(), cutoff=0.85)
            if best_state:
                sub = gadm[gadm['_state_l'] == best_state]

        exact = sub[sub['_county_l'] == county.lower().strip()]
        if not exact.empty:
            r = exact.iloc[0]
            matched.append({'city': county, 'country': 'United States', 'state': state_abbr,
                'boundary_t': 'GADM_county', 'shp_source': 'GADM_USA_L2',
                'gadm_l2': r['NAME_2'], 'gadm_l1': r['NAME_1'],
                'match_meth': 'exact', 'match_scor': 1.0, 'geometry': r['geometry']})
            continue

        best_name, score = fuzzy_best(county, sub['_county_l'].tolist(), cutoff=0.75)
        if best_name:
            r = sub[sub['_county_l'] == best_name].iloc[0]
            matched.append({'city': county, 'country': 'United States', 'state': state_abbr,
                'boundary_t': 'GADM_county', 'shp_source': 'GADM_USA_L2',
                'gadm_l2': r['NAME_2'], 'gadm_l1': r['NAME_1'],
                'match_meth': 'fuzzy', 'match_scor': round(score, 3), 'geometry': r['geometry']})
        else:
            unmatched.append(f"{county}, {state_abbr}")

    pct = len(matched) / len(us) * 100
    print(f"  Matched: {len(matched)}/{len(us)} ({pct:.1f}%)")
    if unmatched:
        print(f"  Unmatched: {unmatched[:10]}")

    gdf = gpd.GeoDataFrame(matched, geometry='geometry', crs='EPSG:4326')
    out = SHP_DIR / "city_boundaries_US_Counties.shp"
    gdf.to_file(out)
    print(f"  ✓ Saved → {out}")
    pd.DataFrame({'county_state': unmatched}).to_csv(OUT_DIR / "unmatched_US_Counties.csv", index=False)
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# C. Merge all into single final GeoPackage
# ═════════════════════════════════════════════════════════════════════════════
def merge_all():
    """Merge FUA + China GADM + US Counties into city_boundaries_ALL.gpkg"""
    parts = []

    fua_shp = SHP_DIR / "FUA_cities_final.shp"
    if fua_shp.exists():
        fua = gpd.read_file(fua_shp)
        fua['gadm_l1'] = ''
        fua['gadm_l2'] = ''
        parts.append(fua)
        print(f"  FUA: {len(fua)} features")

    cn_shp = SHP_DIR / "city_boundaries_China_GADM.shp"
    if cn_shp.exists():
        cn = gpd.read_file(cn_shp)
        parts.append(cn)
        print(f"  China GADM: {len(cn)} features")

    us_shp = SHP_DIR / "city_boundaries_US_Counties.shp"
    if us_shp.exists():
        us = gpd.read_file(us_shp)
        parts.append(us)
        print(f"  US Counties: {len(us)} features")

    if not parts:
        print("No shapefiles found to merge.")
        return

    cols = ['city', 'country', 'state', 'boundary_t', 'shp_source',
            'fuacode', 'fuaname_en', 'gadm_l1', 'gadm_l2',
            'match_meth', 'match_scor', 'geometry']
    all_parts = []
    for p in parts:
        for c in cols:
            if c not in p.columns and c != 'geometry':
                p[c] = ''
        all_parts.append(p[cols])

    merged = gpd.GeoDataFrame(pd.concat(all_parts, ignore_index=True), crs='EPSG:4326')
    out = SHP_DIR / "city_boundaries_ALL.gpkg"
    merged.to_file(out, driver="GPKG")
    print(f"\n  ✓ Merged → {out}  ({len(merged)} total features)")

    # Summary
    print("\n  Summary by boundary type:")
    print(merged['boundary_t'].value_counts().to_string())
    print("\n  Match method breakdown:")
    print(merged['match_meth'].value_counts().to_string())
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("GADM SHAPEFILE BUILDER — China & US")
    print("=" * 60)

    print("\n[A] China GADM L2 prefectures …")
    try:
        build_china_gadm()
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n[B] US GADM L2 counties …")
    try:
        build_us_gadm()
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n[C] Merging all into city_boundaries_ALL.gpkg …")
    merge_all()

    print("\n" + "=" * 60)
    print("Output files in ShapefileData/:")
    print("  FUA_cities_final.shp          — 1,583 FUA cities (98.1%)")
    print("  city_boundaries_China_GADM.shp — China prefecture boundaries")
    print("  city_boundaries_US_Counties.shp — US county boundaries")
    print("  city_boundaries_ALL.gpkg       — All combined (GeoPackage)")
    print("=" * 60)
