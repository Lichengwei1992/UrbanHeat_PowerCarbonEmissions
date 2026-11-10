"""
03_build_city_shapefiles.py
===========================
Match Carbon Monitor Cities to boundary shapefiles and produce outputs.

NOTE: Boundary data already downloaded to City_Boundary/:
  City_Boundary/OECD_FUA/fuas/fuas.shp         — OECD FUA (iso3, fuaname_en)
  City_Boundary/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/*.gpkg — GHSL FUA

For China GADM and US Counties, run: python 03b_build_gadm_shapefiles.py

This script handles FUA cities (OECD + GHSL) only.

Three boundary sources (matching the dataset paper Fig. 2):
  A. OECD FUA      — OECD countries (FUA boundaries)
  B. GHSL FUA      — Non-OECD countries (FUA boundaries)
  C. GADM L2 China — Prefecture-level cities in China
  D. GADM L2 US    — County-level cities in the United States

Output (in ShapefileData/):
  city_boundaries_FUA.gpkg         — all FUA cities (A + B merged)
  city_boundaries_China_GADM.gpkg  — China prefecture boundaries (C)
  city_boundaries_US_Counties.gpkg — US county boundaries (D)
  city_boundaries_ALL.gpkg         — single merged file with all 2,273 entries
  city_boundaries_ALL.shp/.dbf/.prj/.shx  — shapefile version of above

Requires:
  pip install geopandas pandas numpy fiona pyogrio shapely requests tqdm
  pip install openpyxl  # for Excel reading of some OECD files

Run: python 03_build_city_shapefiles.py
"""

import os, sys, zipfile, shutil, json
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    sys.exit("Install geopandas first: pip install geopandas pyogrio fiona")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
CITY_DIR   = BASE / "CarbonData" / "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide"
AUX_DIR    = BASE / "AuxData"
SHP_DIR    = BASE / "ShapefileData"
OUT_DIR    = BASE / "outputs"
for d in [AUX_DIR, SHP_DIR, OUT_DIR]:
    d.mkdir(exist_ok=True)

CATALOG_FILE = OUT_DIR / "city_catalog.csv"

# ── Country lists ──────────────────────────────────────────────────────────────
OECD = {'Australia','Austria','Belgium','Canada','Chile','Colombia','Denmark',
        'Finland','France','Germany','Greece','Hungary','Italy','Japan','Korea',
        'Mexico','Netherlands','Norway','Poland','Portugal','Spain','Sweden',
        'Switzerland','Turkey','United Kingdom','United States'}

# ISO2 codes needed for GADM downloads
COUNTRY_ISO2 = {
    'Argentina':'ARG','Australia':'AUS','Austria':'AUT','Bangladesh':'BGD',
    'Belgium':'BEL','Brazil':'BRA','Canada':'CAN','Chile':'CHL','China':'CHN',
    'Colombia':'COL','Denmark':'DNK','Egypt':'EGY','Finland':'FIN','France':'FRA',
    'Germany':'DEU','Greece':'GRC','Hungary':'HUN','India':'IND','Indonesia':'IDN',
    'Iran':'IRN','Italy':'ITA','Japan':'JPN','Korea':'KOR','Malaysia':'MYS',
    'Mexico':'MEX','Myanmar':'MMR','Netherlands':'NLD','Nigeria':'NGA',
    'Norway':'NOR','Pakistan':'PAK','Peru':'PER','Philippines':'PHL',
    'Poland':'POL','Portugal':'PRT','Russia':'RUS','Singapore':'SGP',
    'South Africa':'ZAF','Spain':'ESP','Sweden':'SWE','Switzerland':'CHE',
    'Thailand':'THA','Turkey':'TUR','United Arab Emirates':'ARE',
    'United Kingdom':'GBR','United States':'USA','Vietnam':'VNM',
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: fuzzy name matching
# ─────────────────────────────────────────────────────────────────────────────
def fuzzy_match(name: str, candidates: list, cutoff: float = 0.75) -> str | None:
    """Return best fuzzy match from candidates list, or None if below cutoff."""
    name_l = name.lower().strip()
    best_score, best_match = 0, None
    for c in candidates:
        score = SequenceMatcher(None, name_l, c.lower().strip()).ratio()
        if score > best_score:
            best_score, best_match = score, c
    return best_match if best_score >= cutoff else None


def download(url: str, dest: Path, desc: str = "") -> Path:
    """Download url to dest if not already present."""
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return dest
    print(f"  ↓ Downloading {desc or dest.name} …")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"    Saved → {dest}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        raise
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# A. OECD FUA SHAPEFILE
# ═════════════════════════════════════════════════════════════════════════════
def get_oecd_fua() -> gpd.GeoDataFrame:
    """
    Download OECD Functional Urban Areas shapefile.

    Direct download from OECD (ZIP containing SHP):
    https://www.oecd.org/regional/regional-statistics/functional-urban-areas.htm

    The OECD FUA file contains ~1,100 FUAs across 33 OECD countries.
    Key attribute columns:
      METRO_ID  — unique FUA identifier
      METRO_NAME — FUA name (in English)
      COUNTRY   — ISO3 country code
    """
    oecd_zip  = AUX_DIR / "OECD_FUA.zip"
    oecd_dir  = SHP_DIR / "OECD_FUA"

    # Primary URL (OECD official)
    url = "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=a89b5e8c-c0ba-4b42-b69a-c7a73c47dc68"
    # Fallback: OECD figshare or alternative mirror
    url_alt = "https://zenodo.org/records/7803783/files/OECD_FUA_2020.zip"

    if not oecd_dir.exists() or not any(oecd_dir.rglob("*.shp")):
        oecd_dir.mkdir(exist_ok=True)
        try:
            download(url, oecd_zip, "OECD FUA shapefile")
        except Exception:
            print("  Trying alternative URL …")
            download(url_alt, oecd_zip, "OECD FUA shapefile (alt)")

        with zipfile.ZipFile(oecd_zip) as z:
            z.extractall(oecd_dir)
        print(f"  Extracted → {oecd_dir}")

    shp_files = list(oecd_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(
            f"No .shp found in {oecd_dir}.\n"
            "Manual download:\n"
            "  1. Go to: https://www.oecd.org/regional/regional-statistics/functional-urban-areas.htm\n"
            "  2. Download the FUA shapefile ZIP\n"
            f"  3. Extract to: {oecd_dir}"
        )

    gdf = gpd.read_file(shp_files[0])
    print(f"  OECD FUA: {len(gdf)} features, columns: {list(gdf.columns)}")
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# B. GHSL FUA SHAPEFILE (non-OECD countries)
# ═════════════════════════════════════════════════════════════════════════════
def get_ghsl_fua() -> gpd.GeoDataFrame:
    """
    Download GHSL Functional Urban Areas (GHS-FUA) shapefile.
    Source: JRC, R2019A release
    URL: https://ghsl.jrc.ec.europa.eu/download.php?ds=fuaV2

    Key columns:
      eFUA_name  — FUA name
      Cntry_ISO  — ISO3 country code
      FUA_p_2015 — population estimate
    """
    ghsl_zip = AUX_DIR / "GHS_FUA.zip"
    ghsl_dir = SHP_DIR / "GHSL_FUA"

    url = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
           "GHS_FUA_UCDB2015_GLOBE_R2019A/V1-0/"
           "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip")

    if not ghsl_dir.exists() or not any(ghsl_dir.rglob("*.shp")):
        ghsl_dir.mkdir(exist_ok=True)
        download(url, ghsl_zip, "GHSL FUA shapefile")
        with zipfile.ZipFile(ghsl_zip) as z:
            z.extractall(ghsl_dir)
        print(f"  Extracted → {ghsl_dir}")

    shp_files = list(ghsl_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(
            f"No .shp found in {ghsl_dir}.\n"
            "Manual download:\n"
            "  1. Go to: https://ghsl.jrc.ec.europa.eu/download.php?ds=fuaV2\n"
            "  2. Download GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip\n"
            f"  3. Extract to: {ghsl_dir}"
        )

    gdf = gpd.read_file(shp_files[0])
    print(f"  GHSL FUA: {len(gdf)} features, columns: {list(gdf.columns)}")
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# C. GADM LEVEL-2 — CHINA (prefecture-level cities)
# ═════════════════════════════════════════════════════════════════════════════
def get_gadm_china() -> gpd.GeoDataFrame:
    """
    Download GADM 4.1 level-2 administrative boundaries for China.
    Level 2 = prefecture-level cities (地级市).
    Key columns: NAME_1 (province), NAME_2 (prefecture), VARNAME_2
    """
    gadm_zip = AUX_DIR / "gadm41_CHN_shp.zip"
    gadm_dir = SHP_DIR / "GADM_CHN"
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_CHN_shp.zip"

    if not gadm_dir.exists() or not any(gadm_dir.rglob("*_2.shp")):
        gadm_dir.mkdir(exist_ok=True)
        download(url, gadm_zip, "GADM China L2 shapefile (~95MB)")
        with zipfile.ZipFile(gadm_zip) as z:
            z.extractall(gadm_dir)

    shp = next(gadm_dir.rglob("*_2.shp"), None)
    if not shp:
        raise FileNotFoundError(
            f"GADM China L2 shapefile not found in {gadm_dir}.\n"
            "Manual download:\n"
            "  1. Go to: https://gadm.org/download_country.html\n"
            "  2. Select China → Shapefile → Download\n"
            f"  3. Extract to: {gadm_dir}"
        )

    gdf = gpd.read_file(shp)
    print(f"  GADM China L2: {len(gdf)} features")
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# D. GADM LEVEL-2 — US (counties)
# ═════════════════════════════════════════════════════════════════════════════
def get_gadm_usa() -> gpd.GeoDataFrame:
    """
    Download GADM 4.1 level-2 administrative boundaries for the USA.
    Level 2 = counties.
    Key columns: NAME_1 (state), NAME_2 (county)
    """
    gadm_zip = AUX_DIR / "gadm41_USA_shp.zip"
    gadm_dir = SHP_DIR / "GADM_USA"
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_USA_shp.zip"

    if not gadm_dir.exists() or not any(gadm_dir.rglob("*_2.shp")):
        gadm_dir.mkdir(exist_ok=True)
        download(url, gadm_zip, "GADM USA L2 shapefile (~170MB)")
        with zipfile.ZipFile(gadm_zip) as z:
            z.extractall(gadm_dir)

    shp = next(gadm_dir.rglob("*_2.shp"), None)
    if not shp:
        raise FileNotFoundError(
            f"GADM USA L2 shapefile not found in {gadm_dir}.\n"
            "Manual download:\n"
            "  1. Go to: https://gadm.org/download_country.html\n"
            "  2. Select USA → Shapefile → Download\n"
            f"  3. Extract to: {gadm_dir}"
        )

    gdf = gpd.read_file(shp)
    print(f"  GADM USA L2: {len(gdf)} features")
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# MATCHING: Carbon Monitor city names → shapefile polygons
# ═════════════════════════════════════════════════════════════════════════════

def match_oecd_fua(catalog_oecd: pd.DataFrame, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Match OECD FUA cities to OECD FUA shapefile polygons.

    OECD FUA shapefile name fields (check actual columns after loading):
      Likely: 'eFUA_name' or 'METRO_NAME' or 'FUA_NAME'
    """
    # Detect name column
    name_col = None
    for candidate in ['eFUA_name','METRO_NAME','FUA_NAME','NAME','fua_name','metro_name']:
        if candidate in gdf.columns:
            name_col = candidate
            break
    if name_col is None:
        print(f"  ⚠ Could not find name column in OECD FUA. Columns: {list(gdf.columns)}")
        name_col = gdf.columns[0]

    # Detect country column
    country_col = None
    for candidate in ['Cntry_ISO','COUNTRY','ISO3','country_code','CNTR_ID']:
        if candidate in gdf.columns:
            country_col = candidate
            break

    gdf = gdf.to_crs(epsg=4326)
    gdf['_name_lower'] = gdf[name_col].str.lower().str.strip()

    matched, unmatched = [], []
    for _, row in catalog_oecd.iterrows():
        city = row['city']
        country = row['country']
        iso3 = COUNTRY_ISO2.get(country, '')

        # Filter by country if possible
        sub = gdf[gdf[country_col] == iso3] if country_col and iso3 else gdf

        # Try exact match first
        exact = sub[sub['_name_lower'] == city.lower().strip()]
        if not exact.empty:
            match_row = exact.iloc[0].copy()
            match_row['cm_city']        = city
            match_row['cm_country']     = country
            match_row['match_method']   = 'exact'
            match_row['match_score']    = 1.0
            matched.append(match_row)
            continue

        # Fuzzy match
        candidates = sub['_name_lower'].tolist()
        best = fuzzy_match(city, candidates)
        if best:
            best_row = sub[sub['_name_lower'] == best].iloc[0].copy()
            score = SequenceMatcher(None, city.lower(), best).ratio()
            best_row['cm_city']      = city
            best_row['cm_country']   = country
            best_row['match_method'] = 'fuzzy'
            best_row['match_score']  = round(score, 3)
            matched.append(best_row)
        else:
            unmatched.append({'city': city, 'country': country})

    print(f"  OECD FUA: matched {len(matched)}/{len(catalog_oecd)} "
          f"({len(matched)/len(catalog_oecd)*100:.1f}%); unmatched: {len(unmatched)}")
    if unmatched:
        print(f"    Unmatched sample: {[u['city'] for u in unmatched[:10]]}")

    out = gpd.GeoDataFrame(matched, crs="EPSG:4326")
    out['boundary_type'] = 'FUA'
    out['shp_source']    = 'OECD_FUA'
    return out


def match_ghsl_fua(catalog_ghsl: pd.DataFrame, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Match non-OECD FUA cities to GHSL FUA shapefile polygons.

    GHSL FUA columns: eFUA_name, Cntry_ISO, FUA_p_2015 (population)
    GHSL is in EPSG:54009 (Mollweide) — reproject to 4326.
    """
    gdf = gdf.to_crs(epsg=4326)

    name_col    = 'eFUA_name'   if 'eFUA_name'   in gdf.columns else gdf.columns[0]
    country_col = 'Cntry_ISO'   if 'Cntry_ISO'   in gdf.columns else None

    gdf['_name_lower'] = gdf[name_col].str.lower().str.strip()

    matched, unmatched = [], []
    for _, row in catalog_ghsl.iterrows():
        city    = row['city']
        country = row['country']
        iso3    = COUNTRY_ISO2.get(country, '')

        sub = gdf[gdf[country_col] == iso3] if country_col and iso3 else gdf

        exact = sub[sub['_name_lower'] == city.lower().strip()]
        if not exact.empty:
            best_row = exact.iloc[0].copy()
            best_row['cm_city']      = city
            best_row['cm_country']   = country
            best_row['match_method'] = 'exact'
            best_row['match_score']  = 1.0
            matched.append(best_row)
            continue

        candidates = sub['_name_lower'].tolist()
        best = fuzzy_match(city, candidates, cutoff=0.72)
        if best:
            best_row = sub[sub['_name_lower'] == best].iloc[0].copy()
            score = SequenceMatcher(None, city.lower(), best).ratio()
            best_row['cm_city']      = city
            best_row['cm_country']   = country
            best_row['match_method'] = 'fuzzy'
            best_row['match_score']  = round(score, 3)
            matched.append(best_row)
        else:
            unmatched.append({'city': city, 'country': country})

    print(f"  GHSL FUA: matched {len(matched)}/{len(catalog_ghsl)} "
          f"({len(matched)/len(catalog_ghsl)*100:.1f}%); unmatched: {len(unmatched)}")
    if unmatched:
        print(f"    Unmatched sample: {[u['city'] for u in unmatched[:10]]}")

    out = gpd.GeoDataFrame(matched, crs="EPSG:4326")
    out['boundary_type'] = 'FUA'
    out['shp_source']    = 'GHSL_FUA'
    return out


def match_gadm_china(catalog_cn: pd.DataFrame, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Match China Carbon Monitor prefecture names to GADM L2 polygons.
    GADM columns: NAME_1 (province), NAME_2 (prefecture in pinyin), VARNAME_2 (alternatives)
    """
    gdf = gdf.to_crs(epsg=4326)
    gdf['_name2_lower'] = gdf['NAME_2'].str.lower().str.strip()
    # Also try VARNAME_2 (alternative names, pipe-separated)
    gdf['_varnames'] = gdf.get('VARNAME_2', pd.Series([''] * len(gdf))).fillna('')

    matched, unmatched = [], []
    all_names = gdf['_name2_lower'].tolist()

    for _, row in catalog_cn.iterrows():
        city = row['city'].strip()

        # Exact on NAME_2
        exact = gdf[gdf['_name2_lower'] == city.lower()]
        if not exact.empty:
            best_row = exact.iloc[0].copy()
            best_row['cm_city']      = city
            best_row['cm_country']   = 'China'
            best_row['match_method'] = 'exact'
            best_row['match_score']  = 1.0
            matched.append(best_row)
            continue

        # Check VARNAME_2 (alternative pinyin / English names)
        var_match = gdf[gdf['_varnames'].str.lower().str.contains(city.lower(), regex=False, na=False)]
        if not var_match.empty:
            best_row = var_match.iloc[0].copy()
            best_row['cm_city']      = city
            best_row['cm_country']   = 'China'
            best_row['match_method'] = 'varname'
            best_row['match_score']  = 0.95
            matched.append(best_row)
            continue

        # Fuzzy on NAME_2
        best = fuzzy_match(city, all_names, cutoff=0.70)
        if best:
            best_row = gdf[gdf['_name2_lower'] == best].iloc[0].copy()
            score = SequenceMatcher(None, city.lower(), best).ratio()
            best_row['cm_city']      = city
            best_row['cm_country']   = 'China'
            best_row['match_method'] = 'fuzzy'
            best_row['match_score']  = round(score, 3)
            matched.append(best_row)
        else:
            unmatched.append(city)

    print(f"  GADM China L2: matched {len(matched)}/{len(catalog_cn)} "
          f"({len(matched)/len(catalog_cn)*100:.1f}%); unmatched: {len(unmatched)}")
    if unmatched:
        print(f"    Unmatched: {unmatched[:15]}")

    out = gpd.GeoDataFrame(matched, crs="EPSG:4326")
    out['boundary_type'] = 'GADM_prefecture'
    out['shp_source']    = 'GADM_CHN_L2'
    return out


def match_gadm_usa(catalog_us: pd.DataFrame, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Match US county names to GADM L2 polygons.
    Carbon Monitor has county + state → match on both NAME_2 (county) + NAME_1 (state).
    """
    gdf = gdf.to_crs(epsg=4326)
    gdf['_county_lower'] = gdf['NAME_2'].str.lower().str.strip()
    gdf['_state_lower']  = gdf['NAME_1'].str.lower().str.strip()

    # US state name → full name mapping
    state_abbrev = {
        'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California',
        'CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia',
        'HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa',
        'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland',
        'MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi',
        'MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire',
        'NJ':'New Jersey','NM':'New Mexico','NY':'New York','NC':'North Carolina',
        'ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania',
        'RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee',
        'TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington',
        'WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming','DC':'District of Columbia',
    }

    matched, unmatched = [], []
    for _, row in catalog_us.iterrows():
        county = row['city'].strip()
        state  = row['state'].strip()
        state_full = state_abbrev.get(state, state).lower()

        # Match by state first, then county name
        sub = gdf[gdf['_state_lower'] == state_full]
        if sub.empty:
            # Try fuzzy on state
            state_names = gdf['_state_lower'].unique().tolist()
            best_state = fuzzy_match(state_full, state_names, cutoff=0.85)
            if best_state:
                sub = gdf[gdf['_state_lower'] == best_state]

        exact = sub[sub['_county_lower'] == county.lower()]
        if not exact.empty:
            best_row = exact.iloc[0].copy()
            best_row['cm_city']      = county
            best_row['cm_country']   = 'United States'
            best_row['cm_state']     = state
            best_row['match_method'] = 'exact'
            best_row['match_score']  = 1.0
            matched.append(best_row)
            continue

        # Fuzzy on county within state
        candidates = sub['_county_lower'].tolist()
        best = fuzzy_match(county, candidates, cutoff=0.75)
        if best:
            best_row = sub[sub['_county_lower'] == best].iloc[0].copy()
            score = SequenceMatcher(None, county.lower(), best).ratio()
            best_row['cm_city']      = county
            best_row['cm_country']   = 'United States'
            best_row['cm_state']     = state
            best_row['match_method'] = 'fuzzy'
            best_row['match_score']  = round(score, 3)
            matched.append(best_row)
        else:
            unmatched.append(f"{county}, {state}")

    print(f"  GADM USA L2: matched {len(matched)}/{len(catalog_us)} "
          f"({len(matched)/len(catalog_us)*100:.1f}%); unmatched: {len(unmatched)}")
    if unmatched:
        print(f"    Unmatched sample: {unmatched[:10]}")

    out = gpd.GeoDataFrame(matched, crs="EPSG:4326")
    out['boundary_type'] = 'GADM_county'
    out['shp_source']    = 'GADM_USA_L2'
    if 'cm_state' not in out.columns:
        out['cm_state'] = ''
    return out


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("CARBON MONITOR CITIES — SHAPEFILE BUILDER")
    print("=" * 65)

    # Load city catalog
    catalog = pd.read_csv(CATALOG_FILE)
    cat_oecd  = catalog[catalog['fua_source'] == 'OECD_FUA'].copy()
    cat_ghsl  = catalog[catalog['fua_source'] == 'GHSL_FUA'].copy()
    cat_cn    = catalog[catalog['boundary_type'] == 'GADM_prefecture'].copy()
    cat_us    = catalog[catalog['boundary_type'] == 'GADM_county'].copy()

    print(f"\nCatalog loaded: {len(catalog)} total cities")
    print(f"  OECD FUA : {len(cat_oecd)}")
    print(f"  GHSL FUA : {len(cat_ghsl)}")
    print(f"  China L2 : {len(cat_cn)}")
    print(f"  US L2    : {len(cat_us)}")

    all_gdfs = []

    # ── A. OECD FUA ────────────────────────────────────────────────────────────
    print("\n[A] OECD FUA boundaries …")
    try:
        oecd_shp = get_oecd_fua()
        oecd_matched = match_oecd_fua(cat_oecd, oecd_shp)
        oecd_matched['cm_state'] = ''
        all_gdfs.append(oecd_matched[['cm_city','cm_country','cm_state',
                                       'boundary_type','shp_source','match_method',
                                       'match_score','geometry']])
        out_a = SHP_DIR / "city_boundaries_OECD_FUA.gpkg"
        oecd_matched.to_file(out_a, driver="GPKG")
        print(f"  Saved → {out_a}")
    except Exception as e:
        print(f"  ⚠ Skipped OECD FUA: {e}")

    # ── B. GHSL FUA ────────────────────────────────────────────────────────────
    print("\n[B] GHSL FUA boundaries …")
    try:
        ghsl_shp = get_ghsl_fua()
        ghsl_matched = match_ghsl_fua(cat_ghsl, ghsl_shp)
        ghsl_matched['cm_state'] = ''
        all_gdfs.append(ghsl_matched[['cm_city','cm_country','cm_state',
                                       'boundary_type','shp_source','match_method',
                                       'match_score','geometry']])
        out_b = SHP_DIR / "city_boundaries_GHSL_FUA.gpkg"
        ghsl_matched.to_file(out_b, driver="GPKG")
        print(f"  Saved → {out_b}")
    except Exception as e:
        print(f"  ⚠ Skipped GHSL FUA: {e}")

    # ── C. GADM China ──────────────────────────────────────────────────────────
    print("\n[C] GADM China L2 (prefectures) …")
    try:
        cn_shp = get_gadm_china()
        cn_matched = match_gadm_china(cat_cn, cn_shp)
        cn_matched['cm_state'] = ''
        all_gdfs.append(cn_matched[['cm_city','cm_country','cm_state',
                                     'boundary_type','shp_source','match_method',
                                     'match_score','geometry']])
        out_c = SHP_DIR / "city_boundaries_China_GADM.gpkg"
        cn_matched.to_file(out_c, driver="GPKG")
        print(f"  Saved → {out_c}")
    except Exception as e:
        print(f"  ⚠ Skipped GADM China: {e}")

    # ── D. GADM USA ────────────────────────────────────────────────────────────
    print("\n[D] GADM USA L2 (counties) …")
    try:
        us_shp = get_gadm_usa()
        us_matched = match_gadm_usa(cat_us, us_shp)
        all_gdfs.append(us_matched[['cm_city','cm_country','cm_state',
                                     'boundary_type','shp_source','match_method',
                                     'match_score','geometry']])
        out_d = SHP_DIR / "city_boundaries_US_Counties.gpkg"
        us_matched.to_file(out_d, driver="GPKG")
        print(f"  Saved → {out_d}")
    except Exception as e:
        print(f"  ⚠ Skipped GADM USA: {e}")

    # ── Merge all into single output ────────────────────────────────────────────
    if all_gdfs:
        print("\n[E] Merging all boundaries …")
        merged = gpd.GeoDataFrame(
            pd.concat(all_gdfs, ignore_index=True),
            crs="EPSG:4326"
        )
        merged = merged.rename(columns={
            'cm_city':    'city',
            'cm_country': 'country',
            'cm_state':   'state',
        })

        # Add city_id from catalog
        key = merged[['city','country','state','boundary_type']].copy()
        key_cat = catalog[['city','country','state','boundary_type','city_id']].copy()
        merged = merged.merge(key_cat, on=['city','country','state','boundary_type'], how='left')

        n = len(merged)
        print(f"  Total merged features: {n}")
        print(merged[['city','country','boundary_type','shp_source','match_method']].head(10).to_string(index=False))

        # GeoPackage (preferred — single file)
        out_all_gpkg = SHP_DIR / "city_boundaries_ALL.gpkg"
        merged.to_file(out_all_gpkg, driver="GPKG")
        print(f"\n  ✓ GeoPackage → {out_all_gpkg}")

        # Shapefile (for compatibility)
        out_all_shp = SHP_DIR / "city_boundaries_ALL.shp"
        # Truncate column names to ≤10 chars for shapefile format
        shp_out = merged.copy()
        shp_out.columns = [c[:10] for c in shp_out.columns]
        shp_out.to_file(out_all_shp)
        print(f"  ✓ Shapefile → {out_all_shp}")

        # Match report
        report = merged.groupby(['boundary_type','shp_source','match_method']).size().reset_index(name='count')
        print(f"\n  Match summary:")
        print(report.to_string(index=False))

        report_file = OUT_DIR / "shapefile_match_report.csv"
        report.to_csv(report_file, index=False)

        unmatched_cities = catalog[~catalog['city'].isin(merged['city'])][['city','country','boundary_type']]
        if len(unmatched_cities) > 0:
            print(f"\n  Unmatched cities: {len(unmatched_cities)}")
            um_file = OUT_DIR / "unmatched_cities.csv"
            unmatched_cities.to_csv(um_file, index=False)
            print(f"  Saved unmatched list → {um_file}")

    print("\n" + "=" * 65)
    print("DONE. Output files in:", SHP_DIR)
    print("=" * 65)
    print("""
Files produced:
  ShapefileData/
    city_boundaries_OECD_FUA.gpkg    — OECD FUA cities
    city_boundaries_GHSL_FUA.gpkg    — GHSL FUA cities (non-OECD)
    city_boundaries_China_GADM.gpkg  — China prefecture boundaries
    city_boundaries_US_Counties.gpkg — US county boundaries
    city_boundaries_ALL.gpkg         — All cities merged (recommended)
    city_boundaries_ALL.shp/dbf/prj  — Shapefile version

  outputs/
    city_catalog.csv                 — Master city list with metadata
    shapefile_match_report.csv       — Match quality summary
    unmatched_cities.csv             — Cities that could not be matched

Column schema (city_boundaries_ALL.gpkg):
  city_id       — unique integer ID (links to city_catalog.csv)
  city          — Carbon Monitor city name
  country       — country name
  state         — state/province (US counties only)
  boundary_type — FUA | GADM_prefecture | GADM_county
  shp_source    — OECD_FUA | GHSL_FUA | GADM_CHN_L2 | GADM_USA_L2
  match_method  — exact | fuzzy | varname
  match_score   — 0–1 fuzzy match confidence
  geometry      — polygon (EPSG:4326 / WGS84)
""")


if __name__ == "__main__":
    main()
