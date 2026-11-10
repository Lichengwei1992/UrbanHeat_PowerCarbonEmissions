"""
02_acquire_data.py
==================
Scripts to acquire all auxiliary data needed for the heat–carbon study:
  A. City coordinates (lat/lon) for ERA5 extraction
  B. ERA5 daily temperature (2m_temperature) via CDS API
  C. UTCI / heat stress from ERA5-HEAT (or computed via pythermalcomfort)
  D. Köppen–Geiger climate zone per city
  E. Oxford OxCGRT COVID stringency index
  F. City population from GHS-POP / UN data
  G. AC penetration rate (country-level, IEA)
  H. EM-DAT heatwave events catalogue

Run each section independently. ERA5 download (B) takes ~1–2 hours for 3 years.
"""

import os, json, time
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "CarbonData")
AUX_DIR  = os.path.join(BASE, "AuxData")
os.makedirs(AUX_DIR, exist_ok=True)

CITY_FILE = os.path.join(DATA_DIR,
    "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide",
    "carbon-monitor-cities-all-cities-FUA-v0325.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# A. EXTRACT CITY LIST WITH COORDINATES
# ═══════════════════════════════════════════════════════════════════════════════
def get_city_list():
    """
    Carbon Monitor Cities does not ship lat/lon directly.
    We match city names to coordinates using the GeoNames database
    (free, no API key needed for the dump).

    Steps:
    1. Download GeoNames city dump (~200MB):
       https://download.geonames.org/export/dump/cities15000.zip
       (cities with population > 15,000 — covers our ~1,597 cities)
    2. Fuzzy-match Carbon Monitor city names to GeoNames
    3. Output: city_coordinates.csv  [city, country, lat, lon, population]
    """
    import subprocess
    out_file = os.path.join(AUX_DIR, "city_coordinates.csv")
    if os.path.exists(out_file):
        print(f"  city_coordinates.csv already exists, skipping download.")
        return pd.read_csv(out_file)

    print("Step A: Extracting unique city+country combinations …")
    df = pd.read_csv(CITY_FILE, encoding="utf-8-sig",
                     usecols=["city", "country"]).drop_duplicates()
    df["city"]    = df["city"].str.strip()
    df["country"] = df["country"].str.strip()
    df = df[df["city"] != ""].reset_index(drop=True)
    print(f"  Found {len(df)} unique city-country pairs")

    # Download GeoNames if not present
    geonames_zip  = os.path.join(AUX_DIR, "cities15000.zip")
    geonames_txt  = os.path.join(AUX_DIR, "cities15000.txt")
    if not os.path.exists(geonames_txt):
        print("  Downloading GeoNames cities15000.zip (~35MB) …")
        import urllib.request
        url = "https://download.geonames.org/export/dump/cities15000.zip"
        urllib.request.urlretrieve(url, geonames_zip)
        import zipfile
        with zipfile.ZipFile(geonames_zip) as z:
            z.extractall(AUX_DIR)
        print("  Done.")

    # GeoNames columns (see readme.txt in the zip)
    gn_cols = ["geonameid","name","asciiname","alternatenames",
                "lat","lon","feature_class","feature_code",
                "country_code","cc2","admin1","admin2","admin3","admin4",
                "population","elevation","dem","timezone","modification_date"]
    gn = pd.read_csv(geonames_txt, sep="\t", names=gn_cols,
                     low_memory=False, encoding="utf-8")
    gn = gn[gn["feature_class"] == "P"]  # populated places only
    gn["asciiname_lower"] = gn["asciiname"].str.lower().str.strip()

    # Country code mapping (ISO2 → country name used in Carbon Monitor)
    country_iso2 = {
        "CN": "China", "US": "United States", "DE": "Germany",
        "MX": "Mexico", "GB": "United Kingdom", "FR": "France",
        "IT": "Italy", "ES": "Spain", "JP": "Japan", "IN": "India",
        "PL": "Poland", "CO": "Colombia", "RU": "Russia", "BR": "Brazil",
        "NL": "Netherlands", "AU": "Australia", "CA": "Canada",
        "KR": "Korea", "TR": "Turkey", "AR": "Argentina",
        "BE": "Belgium", "BD": "Bangladesh", "CL": "Chile",
        "DK": "Denmark", "EG": "Egypt", "FI": "Finland",
        "GR": "Greece", "HU": "Hungary", "ID": "Indonesia",
        "IR": "Iran", "MY": "Malaysia", "MM": "Myanmar",
        "NG": "Nigeria", "NO": "Norway", "PK": "Pakistan",
        "PE": "Peru", "PH": "Philippines", "PT": "Portugal",
        "SG": "Singapore", "ZA": "South Africa", "SE": "Sweden",
        "CH": "Switzerland", "TH": "Thailand", "AE": "United Arab Emirates",
        "VN": "Vietnam", "AT": "Austria",
    }
    gn["country_name"] = gn["country_code"].map(country_iso2)

    # Fuzzy match: simple lowercase strip first, then fuzz if needed
    from difflib import get_close_matches
    matched_rows = []
    for _, row in df.iterrows():
        city_lower   = row["city"].lower().strip()
        country_name = row["country"]
        subset = gn[gn["country_name"] == country_name].copy()
        if subset.empty:
            matched_rows.append({"city": row["city"], "country": country_name,
                                  "lat": np.nan, "lon": np.nan, "population": np.nan})
            continue
        # Exact match first
        exact = subset[subset["asciiname_lower"] == city_lower]
        if not exact.empty:
            best = exact.sort_values("population", ascending=False).iloc[0]
            matched_rows.append({"city": row["city"], "country": country_name,
                                  "lat": best["lat"], "lon": best["lon"],
                                  "population": best["population"]})
        else:
            # Fuzzy match
            names = subset["asciiname_lower"].tolist()
            close = get_close_matches(city_lower, names, n=1, cutoff=0.8)
            if close:
                best = subset[subset["asciiname_lower"] == close[0]].sort_values(
                    "population", ascending=False).iloc[0]
                matched_rows.append({"city": row["city"], "country": country_name,
                                      "lat": best["lat"], "lon": best["lon"],
                                      "population": best["population"]})
            else:
                matched_rows.append({"city": row["city"], "country": country_name,
                                      "lat": np.nan, "lon": np.nan, "population": np.nan})

    coords = pd.DataFrame(matched_rows)
    n_matched = coords["lat"].notna().sum()
    print(f"  Matched {n_matched}/{len(coords)} cities ({n_matched/len(coords)*100:.1f}%)")
    coords.to_csv(out_file, index=False)
    print(f"  Saved → {out_file}")
    return coords


# ═══════════════════════════════════════════════════════════════════════════════
# B. ERA5 DAILY TEMPERATURE DOWNLOAD (CDS API)
# ═══════════════════════════════════════════════════════════════════════════════
def download_era5_temperature():
    """
    Downloads ERA5 daily 2m temperature at 0.25° resolution for 2019–2021.
    Requires:
      pip install cdsapi
      CDS account: https://cds.climate.copernicus.eu/
      ~/.cdsapirc file with:
        url: https://cds.climate.copernicus.eu/api/v2
        key: <YOUR_UID>:<YOUR_API_KEY>

    Output: ERA5_temperature_2019_2021.nc  (~4GB, global)
    Tip: request one year at a time to avoid timeout.
    """
    try:
        import cdsapi
    except ImportError:
        print("  Install cdsapi first: pip install cdsapi")
        print("  Then create ~/.cdsapirc with your CDS credentials.")
        return

    c = cdsapi.Client()
    out_dir = os.path.join(AUX_DIR, "ERA5")
    os.makedirs(out_dir, exist_ok=True)

    for year in [2019, 2020, 2021]:
        out_file = os.path.join(out_dir, f"ERA5_2m_temperature_{year}.nc")
        if os.path.exists(out_file):
            print(f"  {out_file} already exists, skipping.")
            continue
        print(f"  Requesting ERA5 2m_temperature for {year} …")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "surface_solar_radiation_downwards",
                ],
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day":   [f"{d:02d}" for d in range(1, 32)],
                "time":  ["06:00", "12:00", "15:00", "18:00"],  # daily Tmax window
                "format": "netcdf",
            },
            out_file,
        )
        print(f"  Saved → {out_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# B2. EXTRACT CITY-LEVEL DAILY Tmax FROM ERA5 NetCDF
# ═══════════════════════════════════════════════════════════════════════════════
def extract_city_temperatures(coords_file=None):
    """
    For each city centroid, extract daily Tmax (°C) and Tmean from ERA5 NetCDF.
    Output: city_daily_temperature.parquet
    """
    try:
        import xarray as xr
    except ImportError:
        print("  pip install xarray netcdf4")
        return

    if coords_file is None:
        coords_file = os.path.join(AUX_DIR, "city_coordinates.csv")
    coords = pd.read_csv(coords_file).dropna(subset=["lat", "lon"])
    era5_dir = os.path.join(AUX_DIR, "ERA5")
    out_file = os.path.join(AUX_DIR, "city_daily_temperature.parquet")

    records = []
    for year in [2019, 2020, 2021]:
        nc_file = os.path.join(era5_dir, f"ERA5_2m_temperature_{year}.nc")
        if not os.path.exists(nc_file):
            print(f"  Missing: {nc_file} — run download_era5_temperature() first")
            continue
        print(f"  Processing {nc_file} …")
        ds = xr.open_dataset(nc_file)
        t2m = ds["t2m"] - 273.15  # K → °C

        for _, row in coords.iterrows():
            city, country, lat, lon = row["city"], row["country"], row["lat"], row["lon"]
            city_t = t2m.sel(latitude=lat, longitude=lon, method="nearest")
            # Daily max and mean across the 4 time steps
            daily = city_t.resample(time="1D").max().to_series()
            daily_mean = t2m.sel(latitude=lat, longitude=lon, method="nearest").resample(time="1D").mean().to_series()
            for date, tmax in daily.items():
                records.append({
                    "city": city, "country": country, "date": date,
                    "tmax_c": float(tmax), "tmean_c": float(daily_mean.get(date, np.nan)),
                })
        ds.close()
        print(f"  Year {year} done, {len(records):,} records so far")

    df_temp = pd.DataFrame(records)
    df_temp["date"] = pd.to_datetime(df_temp["date"]).dt.date
    df_temp.to_parquet(out_file, index=False)
    print(f"  Saved → {out_file}")
    return df_temp


# ═══════════════════════════════════════════════════════════════════════════════
# C. UTCI COMPUTATION (from ERA5 components)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_utci(city_era5_file=None):
    """
    Compute UTCI from ERA5 variables using pythermalcomfort.
    pip install pythermalcomfort

    Requires city-level ERA5 extracts with: tdb (Tair), tr (mean radiant temp),
    v (wind speed), rh (relative humidity).

    Shortcut: ERA5-HEAT dataset (precomputed UTCI, 0.25° daily):
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/derived-utci-historical
    Download via CDS API — dataset id: 'derived-utci-historical'
    """
    try:
        import cdsapi
    except ImportError:
        print("  pip install cdsapi")
        return

    c = cdsapi.Client()
    out_dir = os.path.join(AUX_DIR, "ERA5_HEAT")
    os.makedirs(out_dir, exist_ok=True)

    for year in [2019, 2020, 2021]:
        out_file = os.path.join(out_dir, f"UTCI_{year}.nc")
        if os.path.exists(out_file):
            print(f"  {out_file} exists, skipping.")
            continue
        print(f"  Requesting ERA5-HEAT UTCI for {year} …")
        c.retrieve(
            "derived-utci-historical",
            {
                "variable": "universal_thermal_climate_index",
                "product_type": "consolidated_dataset",
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["06:00", "12:00", "15:00", "18:00"],
                "format": "zip",
            },
            out_file,
        )
        print(f"  Saved → {out_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# D. KOPPEN–GEIGER CLIMATE ZONE PER CITY
# ═══════════════════════════════════════════════════════════════════════════════
def assign_koppen_geiger(coords_file=None):
    """
    Downloads Beck et al. (2018) 0.5° Köppen–Geiger raster and assigns
    climate zone to each city centroid.
    Source: http://www.gloh2o.org/koppen/
    File: KG_1991_2020.tif (~4MB)
    """
    try:
        import rasterio
        from rasterio.sample import sample_gen
    except ImportError:
        print("  pip install rasterio")
        return

    if coords_file is None:
        coords_file = os.path.join(AUX_DIR, "city_coordinates.csv")
    coords = pd.read_csv(coords_file).dropna(subset=["lat", "lon"])

    kg_file = os.path.join(AUX_DIR, "KG_1991_2020_0p5.tif")
    if not os.path.exists(kg_file):
        import urllib.request
        url = "https://figshare.com/ndownloader/files/45057352"  # Beck 2023 updated map
        print(f"  Downloading Köppen–Geiger raster …")
        urllib.request.urlretrieve(url, kg_file)

    # KG class codes (simplified)
    kg_classes = {
        1:"Af",2:"Am",3:"Aw",4:"BWh",5:"BWk",6:"BSh",7:"BSk",
        8:"Csa",9:"Csb",10:"Csc",11:"Cwa",12:"Cwb",13:"Cwc",
        14:"Cfa",15:"Cfb",16:"Cfc",17:"Dsa",18:"Dsb",19:"Dsc",
        20:"Dsd",21:"Dwa",22:"Dwb",23:"Dwc",24:"Dwd",
        25:"Dfa",26:"Dfb",27:"Dfc",28:"Dfd",29:"ET",30:"EF",
    }
    kg_major = {
        "A":"Tropical","B":"Arid","C":"Temperate","D":"Continental","E":"Polar"
    }

    with rasterio.open(kg_file) as src:
        xy = list(zip(coords["lon"], coords["lat"]))
        values = [v[0] for v in src.sample(xy)]

    coords = coords.copy()
    coords["kg_code_num"] = values
    coords["kg_code"]  = coords["kg_code_num"].map(kg_classes).fillna("Unknown")
    coords["kg_major"] = coords["kg_code"].str[0].map(kg_major).fillna("Unknown")

    out_file = os.path.join(AUX_DIR, "city_coordinates_kg.csv")
    coords.to_csv(out_file, index=False)
    print(f"  Saved → {out_file}")
    print(coords["kg_major"].value_counts())
    return coords


# ═══════════════════════════════════════════════════════════════════════════════
# E. OXFORD OxCGRT COVID STRINGENCY INDEX
# ═══════════════════════════════════════════════════════════════════════════════
def download_oxcgrt():
    """
    Downloads the Oxford COVID-19 Government Response Tracker stringency index.
    GitHub: https://github.com/OxCGRT/covid-policy-tracker
    Direct CSV: national-level daily stringency index, 2020–2021.
    """
    import urllib.request
    out_file = os.path.join(AUX_DIR, "OxCGRT_compact_national_v1.csv")
    if os.path.exists(out_file):
        print(f"  OxCGRT already downloaded.")
        return pd.read_csv(out_file)

    url = ("https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/"
           "master/data/OxCGRT_compact_national_v1.csv")
    print("  Downloading OxCGRT stringency index …")
    try:
        urllib.request.urlretrieve(url, out_file)
        print(f"  Saved → {out_file}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Manual download: https://github.com/OxCGRT/covid-policy-tracker/tree/master/data")
        return None

    df = pd.read_csv(out_file)
    # Keep relevant columns
    keep = [c for c in df.columns if c in
            ["CountryName","Date","StringencyIndex_Average"]]
    df = df[keep].rename(columns={
        "CountryName": "country",
        "Date": "date",
        "StringencyIndex_Average": "stringency_index"
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    out_slim = os.path.join(AUX_DIR, "oxcgrt_stringency.csv")
    df.to_csv(out_slim, index=False)
    print(f"  Slim file saved → {out_slim}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# F. CITY POPULATION (GHS-POP / UN)
# ═══════════════════════════════════════════════════════════════════════════════
def assign_population(coords_file=None):
    """
    GeoNames already provides population estimates for matched cities (see step A).
    For higher accuracy, use the Global Human Settlement Layer (GHS-POP):
      https://ghsl.jrc.ec.europa.eu/ghs_pop.php
      Resolution: 1km, Epoch 2020 (R2023A release)

    Simple approach: use GeoNames population from city_coordinates.csv.
    Advanced: rasterio point extraction from GHS-POP.tif.
    """
    if coords_file is None:
        coords_file = os.path.join(AUX_DIR, "city_coordinates_kg.csv")
        if not os.path.exists(coords_file):
            coords_file = os.path.join(AUX_DIR, "city_coordinates.csv")

    coords = pd.read_csv(coords_file)
    print(f"  Population coverage from GeoNames: "
          f"{coords['population'].notna().sum()}/{len(coords)} cities")

    # Classify by size
    def pop_class(p):
        if pd.isna(p): return "Unknown"
        if p >= 5_000_000: return "Megacity (≥5M)"
        if p >= 1_000_000: return "Large (1–5M)"
        if p >= 500_000:   return "Medium-large (500K–1M)"
        if p >= 100_000:   return "Medium (100K–500K)"
        return "Small (<100K)"

    coords["pop_class"] = coords["population"].apply(pop_class)
    print(coords["pop_class"].value_counts())
    return coords


# ═══════════════════════════════════════════════════════════════════════════════
# G. AC PENETRATION RATE (country-level, IEA proxy)
# ═══════════════════════════════════════════════════════════════════════════════
def build_ac_penetration_table():
    """
    From IEA (2018) 'The Future of Cooling' — Table A2.1
    AC ownership rates by country (~2018). Used as cross-sectional heterogeneity variable.
    Source: https://www.iea.org/reports/the-future-of-cooling
    """
    ac_data = {
        "country": [
            "United States","Japan","Korea","Australia","Saudi Arabia",
            "United Arab Emirates","China","Brazil","Mexico","Spain",
            "Italy","Argentina","Turkey","Thailand","Iran",
            "Malaysia","India","Indonesia","Germany","France",
            "United Kingdom","Russia","Pakistan","Nigeria","Bangladesh",
        ],
        "ac_ownership_pct": [
            90, 91, 86, 75, 97,
            97, 60, 16, 16, 35,
            52, 45, 72, 25, 35,
            65, 5, 9, 3, 7,
            5, 8, 8, 6, 3,
        ]
    }
    df = pd.DataFrame(ac_data)
    out_file = os.path.join(AUX_DIR, "ac_penetration_country.csv")
    df.to_csv(out_file, index=False)
    print(f"  AC penetration table saved → {out_file}")
    print(df.sort_values("ac_ownership_pct", ascending=False).to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# H. EM-DAT HEATWAVE EVENTS (manual download required)
# ═══════════════════════════════════════════════════════════════════════════════
def process_emdat(emdat_csv=None):
    """
    EM-DAT requires free registration at https://www.emdat.be/
    After downloading, filter for heat-wave events 2019–2021.

    Expected columns (EM-DAT 2023 format):
      DisNo, Country, ISO, DisasterType, DisasterSubtype,
      StartYear, StartMonth, StartDay, EndYear, EndMonth, EndDay,
      TotalAffected, TotalDeaths, TotalDamageAdjusted('000 US$)
    """
    if emdat_csv is None:
        emdat_csv = os.path.join(AUX_DIR, "emdat_heatwaves.csv")
    if not os.path.exists(emdat_csv):
        print("  EM-DAT not found. Please:")
        print("    1. Register at https://public.emdat.be/")
        print("    2. Download: Disaster Type = Heat wave, Period = 2019-2021")
        print(f"    3. Save as: {emdat_csv}")
        return None

    df = pd.read_csv(emdat_csv)
    heat = df[df["Disaster Subtype"].str.contains("Heat wave", na=False)].copy()
    heat["start_date"] = pd.to_datetime({
        "year": heat["Start Year"], "month": heat["Start Month"].fillna(1),
        "day": heat["Start Day"].fillna(1)
    }, errors="coerce")
    heat["end_date"] = pd.to_datetime({
        "year": heat["End Year"], "month": heat["End Month"].fillna(12),
        "day": heat["End Day"].fillna(28)
    }, errors="coerce")
    out = heat[["Country","ISO","start_date","end_date",
                "Total Deaths","Total Affected"]].dropna(subset=["start_date"])
    out_file = os.path.join(AUX_DIR, "emdat_heatwaves_clean.csv")
    out.to_csv(out_file, index=False)
    print(f"  Found {len(out)} heatwave events in EM-DAT → saved to {out_file}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: Run all available steps
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("DATA ACQUISITION PIPELINE")
    print("="*60)

    print("\n[A] City coordinates …")
    coords = get_city_list()

    print("\n[D] Köppen–Geiger climate zones …")
    try:
        coords_kg = assign_koppen_geiger()
    except Exception as e:
        print(f"  Skipped (need rasterio): {e}")

    print("\n[E] OxCGRT COVID stringency index …")
    try:
        oxcgrt = download_oxcgrt()
    except Exception as e:
        print(f"  Failed: {e}")

    print("\n[F] Population classification …")
    try:
        assign_population()
    except Exception as e:
        print(f"  Failed: {e}")

    print("\n[G] AC penetration rate table …")
    ac = build_ac_penetration_table()

    print("\n[H] EM-DAT heatwave catalogue …")
    emdat = process_emdat()

    print("\n" + "="*60)
    print("MANUAL STEPS STILL REQUIRED:")
    print("="*60)
    print("""
  [B] ERA5 temperature download:
      1. Register at https://cds.climate.copernicus.eu/
      2. Install: pip install cdsapi
      3. Create ~/.cdsapirc with your UID and API key
      4. Run: download_era5_temperature()  (in this script)
      5. Then run: extract_city_temperatures()

  [C] UTCI / ERA5-HEAT:
      1. Same CDS credentials as above
      2. Run: compute_utci()  (downloads derived-utci-historical)
      OR compute UTCI from ERA5 components:
         pip install pythermalcomfort

  [H] EM-DAT:
      1. Register at https://public.emdat.be/
      2. Download heatwave events 2019–2021
      3. Save to AuxData/emdat_heatwaves.csv
      4. Run: process_emdat()
""")
