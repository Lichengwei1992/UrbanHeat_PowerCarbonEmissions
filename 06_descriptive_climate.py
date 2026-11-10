"""
06_descriptive_climate.py
=========================
Comprehensive descriptive statistics for all 1,583 FUA cities, covering:

  1. Extreme temperature days  — hot days (Tmax > p90) and cold days (Tmin < p10),
                                  annual counts and fractions
  2. Heatwave events           — count, duration, peak Tmax, cumulative intensity,
                                  annual frequency; per-year breakdown
  3. Cold wave events          — same metrics for cold extremes
  4. UTCI thermal stress days  — days per ISO 15743 / Bröde et al. 2012 category
  5. Temperature climatology   — mean/max/min Tmax, Tmin, Tmean
  6. Koppen-Geiger climate class — sampled from 1991-2020 raster at city centroid

Outputs
-------
  outputs/06_descriptive/city_climate_stats.csv      — one row per city (full column names)
  outputs/06_descriptive/city_climate_stats.shp      — same joined to FUA geometry
  outputs/06_descriptive/event_catalog.csv           — one row per heatwave/coldwave event
                                                        (start date, end date, duration,
                                                         peak temp, intensity — for CO2 alignment)
  outputs/06_descriptive/extreme_days_panel.parquet  — daily extreme flags for all cities
                                                        (city_id × date; use for CO2 alignment)

Run after: 04_define_events.py, 05d_extract_era5_cities.py

Requirements:
    pip install geopandas rasterio pyarrow pandas numpy
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
SHP_DIR  = BASE / "ShapefileData"
OUT_DIR  = BASE / "outputs" / "06_descriptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FLAGS  = BASE / "outputs" / "04_events" / "event_flags.parquet"
CLIMATE_PAR  = BASE / "outputs" / "city_climate_2019_2021.parquet"
FUA_LOOKUP   = BASE / "outputs" / "fua_city_lookup.csv"
FUA_SHP      = SHP_DIR / "FUA_cities_final.shp"
KG_TIF       = BASE / "koppen_geiger_tif" / "1991_2020" / "koppen_geiger_0p1.tif"

# ─────────────────────────────────────────────────────────────────────────────
# KOPPEN-GEIGER LOOKUP  (Beck et al. 2023, 30-class scheme)
# ─────────────────────────────────────────────────────────────────────────────
KG_MAP = {
    1:  ("Af",  "A", "Tropical rainforest"),
    2:  ("Am",  "A", "Tropical monsoon"),
    3:  ("Aw",  "A", "Tropical savanna (dry winter)"),
    4:  ("As",  "A", "Tropical savanna (dry summer)"),
    5:  ("BWh", "B", "Arid desert (hot)"),
    6:  ("BWk", "B", "Arid desert (cold)"),
    7:  ("BSh", "B", "Arid steppe (hot)"),
    8:  ("BSk", "B", "Arid steppe (cold)"),
    9:  ("Csa", "C", "Temperate, dry & hot summer"),
    10: ("Csb", "C", "Temperate, dry & warm summer"),
    11: ("Csc", "C", "Temperate, dry & cold summer"),
    12: ("Cwa", "C", "Temperate, dry winter & hot summer"),
    13: ("Cwb", "C", "Temperate, dry winter & warm summer"),
    14: ("Cwc", "C", "Temperate, dry winter & cold summer"),
    15: ("Cfa", "C", "Temperate, humid & hot summer"),
    16: ("Cfb", "C", "Temperate, humid & warm summer"),
    17: ("Cfc", "C", "Temperate, humid & cold summer"),
    18: ("Dsa", "D", "Cold, dry & hot summer"),
    19: ("Dsb", "D", "Cold, dry & warm summer"),
    20: ("Dsc", "D", "Cold, dry & cold summer"),
    21: ("Dsd", "D", "Cold, dry & very cold winter"),
    22: ("Dwa", "D", "Cold, dry winter & hot summer"),
    23: ("Dwb", "D", "Cold, dry winter & warm summer"),
    24: ("Dwc", "D", "Cold, dry winter & cold summer"),
    25: ("Dwd", "D", "Cold, dry winter & very cold winter"),
    26: ("Dfa", "D", "Cold, humid & hot summer"),
    27: ("Dfb", "D", "Cold, humid & warm summer"),
    28: ("Dfc", "D", "Cold, humid & cold summer"),
    29: ("Dfd", "D", "Cold, humid & very cold winter"),
    30: ("ET",  "E", "Polar tundra"),
}

# UTCI thermal stress categories (Bröde et al. 2012 / ISO 15743)
UTCI_BINS   = [-99, -40, -27, -13, 0, 9, 26, 32, 38, 46, 999]
UTCI_LABELS = [
    "extreme_cold",     # < -40
    "very_strong_cold", # -40 to -27
    "strong_cold",      # -27 to -13
    "moderate_cold",    # -13 to  0
    "slight_cold",      #   0 to  9
    "no_stress",        #   9 to 26
    "moderate_heat",    #  26 to 32
    "strong_heat",      #  32 to 38
    "very_strong_heat", #  38 to 46
    "extreme_heat",     #  > 46
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    print("[1] Loading event flags and climate data …")
    ef = pd.read_parquet(EVENT_FLAGS)
    ef["city_id"] = ef["city_id"].astype(str).str.zfill(4)
    ef["date"]    = pd.to_datetime(ef["date"])
    ef["year"]    = ef["date"].dt.year
    print(f"    event_flags: {len(ef):,} rows  |  cities: {ef['city_id'].nunique()}")

    cl = pd.read_parquet(CLIMATE_PAR)
    cl["city_id"] = cl["city_id"].astype(str).str.zfill(4)
    cl["date"]    = pd.to_datetime(cl["date"])
    print(f"    climate:     {len(cl):,} rows  |  utci_mean: "
          f"{cl['utci_mean'].min():.1f} – {cl['utci_mean'].max():.1f} °C")

    lookup = pd.read_csv(FUA_LOOKUP, dtype={"city_id": str})
    lookup["city_id"] = lookup["city_id"].str.zfill(4)
    print(f"    lookup: {len(lookup)} cities")
    return ef, cl, lookup


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Per-city temperature climatology
# ─────────────────────────────────────────────────────────────────────────────
def city_temp_stats(ef: pd.DataFrame) -> pd.DataFrame:
    print("[2] Computing temperature climatology …")
    g = ef.groupby("city_id").agg(
        tmax_mean    = ("tmax",  "mean"),
        tmax_max     = ("tmax",  "max"),
        tmin_mean    = ("tmin",  "mean"),
        tmin_min     = ("tmin",  "min"),
        tmean_mean   = ("tmean", "mean"),
        tmax_p90_med = ("tmax_p90", "median"),  # median DOY threshold (city summary)
        tmin_p10_med = ("tmin_p10", "median"),
        n_study_days = ("date",  "count"),
    ).reset_index()
    return g


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Extreme single-day counts (threshold exceedance, not consecutive)
# ─────────────────────────────────────────────────────────────────────────────
def extreme_day_stats(ef: pd.DataFrame) -> pd.DataFrame:
    """
    Hot day  = Tmax > DOY-specific p90 (even if only 1 day, not a full heatwave).
    Cold day = Tmin < DOY-specific p10.
    """
    print("[3] Computing extreme single-day counts …")
    ef = ef.copy()
    ef["is_hot_day"]  = (ef["tmax"] > ef["tmax_p90"]).astype(int)
    ef["is_cold_day"] = (ef["tmin"] < ef["tmin_p10"]).astype(int)

    # Annual breakdown
    annual = ef.groupby(["city_id", "year"]).agg(
        n_hot  = ("is_hot_day",  "sum"),
        n_cold = ("is_cold_day", "sum"),
    ).reset_index()

    # Pivot years
    hot_yr  = annual.pivot(index="city_id", columns="year", values="n_hot").rename(
        columns={2019: "n_hot_2019", 2020: "n_hot_2020", 2021: "n_hot_2021"})
    cold_yr = annual.pivot(index="city_id", columns="year", values="n_cold").rename(
        columns={2019: "n_cold_2019", 2020: "n_cold_2020", 2021: "n_cold_2021"})

    # Totals
    total = ef.groupby("city_id").agg(
        n_hot_days  = ("is_hot_day",  "sum"),
        n_cold_days = ("is_cold_day", "sum"),
        n_days_total = ("date",       "count"),
    ).reset_index()
    total["hot_day_pct"]  = total["n_hot_days"]  / total["n_days_total"] * 100
    total["cold_day_pct"] = total["n_cold_days"] / total["n_days_total"] * 100

    out = total.merge(hot_yr.reset_index(),  on="city_id", how="left")
    out = out.merge(cold_yr.reset_index(), on="city_id", how="left")
    out = out.drop(columns=["n_days_total"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Heatwave and cold wave event statistics
# ─────────────────────────────────────────────────────────────────────────────
def event_stats(ef: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        city_ev_stats  — per-city aggregated event statistics
        event_catalog  — one row per event (heatwave + cold wave)
    """
    print("[4] Computing heatwave and cold wave statistics …")

    catalog_parts = []

    # ── Heatwave catalog ──────────────────────────────────────────────────
    hw_days = ef[ef["heat_event_id"] > 0].copy()
    hw_days["intensity"] = hw_days["tmax"] - hw_days["tmax_p90"]

    hw_cat = hw_days.groupby(["city_id", "heat_event_id"]).agg(
        start_date      = ("date",       "min"),
        end_date        = ("date",       "max"),
        duration_days   = ("heat_day_n", "max"),
        peak_tmax       = ("tmax",       "max"),
        mean_tmax       = ("tmax",       "mean"),
        tmax_p90_ref    = ("tmax_p90",   "first"),  # first-day threshold as reference
        cumul_intensity = ("intensity",  "sum"),     # Σ(Tmax − p90) °C·days
        mean_intensity  = ("intensity",  "mean"),
    ).reset_index().rename(columns={"heat_event_id": "event_id"})
    hw_cat["event_type"] = "heatwave"
    hw_cat["year"]       = hw_cat["start_date"].dt.year
    hw_cat = hw_cat.rename(columns={"tmax_p90_ref": "threshold_ref",
                                    "peak_tmax": "peak_temp",
                                    "mean_tmax":  "mean_temp"})
    catalog_parts.append(hw_cat)

    # Per-city heatwave summaries
    hw_city = hw_cat.groupby("city_id").agg(
        hw_n_events   = ("event_id",       "count"),
        hw_n_days     = ("duration_days",  "sum"),
        hw_mean_dur   = ("duration_days",  "mean"),
        hw_max_dur    = ("duration_days",  "max"),
        hw_peak_tmax  = ("peak_temp",      "max"),
        hw_cumul_int  = ("cumul_intensity","sum"),
        hw_mean_int   = ("mean_intensity", "mean"),
    ).reset_index()
    hw_city["hw_annual_freq"] = hw_city["hw_n_events"] / 3.0

    # Per-year counts
    hw_yr = hw_cat.groupby(["city_id", "year"])["event_id"].count().reset_index().rename(
        columns={"event_id": "count"})
    for yr in [2019, 2020, 2021]:
        col = f"hw_n_{yr}"
        hw_city[col] = hw_city["city_id"].map(
            hw_yr[hw_yr["year"] == yr].set_index("city_id")["count"]
        ).fillna(0).astype(int)

    # ── Cold wave catalog ──────────────────────────────────────────────────
    cw_days = ef[ef["cold_event_id"] > 0].copy()
    cw_days["intensity"] = cw_days["tmin_p10"] - cw_days["tmin"]  # positive = colder than threshold

    cw_cat = cw_days.groupby(["city_id", "cold_event_id"]).agg(
        start_date      = ("date",       "min"),
        end_date        = ("date",       "max"),
        duration_days   = ("cold_day_n", "max"),
        peak_tmin       = ("tmin",       "min"),
        mean_tmin       = ("tmin",       "mean"),
        tmin_p10_ref    = ("tmin_p10",   "first"),
        cumul_intensity = ("intensity",  "sum"),
        mean_intensity  = ("intensity",  "mean"),
    ).reset_index().rename(columns={"cold_event_id": "event_id"})
    cw_cat["event_type"] = "coldwave"
    cw_cat["year"]       = cw_cat["start_date"].dt.year
    cw_cat = cw_cat.rename(columns={"tmin_p10_ref": "threshold_ref",
                                    "peak_tmin": "peak_temp",
                                    "mean_tmin":  "mean_temp"})
    catalog_parts.append(cw_cat)

    # Per-city cold wave summaries
    cw_city = cw_cat.groupby("city_id").agg(
        cw_n_events   = ("event_id",       "count"),
        cw_n_days     = ("duration_days",  "sum"),
        cw_mean_dur   = ("duration_days",  "mean"),
        cw_max_dur    = ("duration_days",  "max"),
        cw_peak_tmin  = ("peak_temp",      "min"),
        cw_cumul_int  = ("cumul_intensity","sum"),
        cw_mean_int   = ("mean_intensity", "mean"),
    ).reset_index()
    cw_city["cw_annual_freq"] = cw_city["cw_n_events"] / 3.0

    for yr in [2019, 2020, 2021]:
        col = f"cw_n_{yr}"
        cw_city[col] = cw_city["city_id"].map(
            cw_cat[cw_cat["year"] == yr].groupby("city_id")["event_id"].count()
        ).fillna(0).astype(int)

    # ── Combine catalogs ──────────────────────────────────────────────────
    event_catalog = pd.concat(catalog_parts, ignore_index=True)
    # Keep consistent columns
    keep_cols = ["city_id", "event_type", "event_id", "year",
                 "start_date", "end_date", "duration_days",
                 "peak_temp", "mean_temp", "threshold_ref",
                 "cumul_intensity", "mean_intensity"]
    event_catalog = event_catalog[[c for c in keep_cols if c in event_catalog.columns]]
    event_catalog = event_catalog.sort_values(["city_id", "event_type", "start_date"])

    # Fill cities with zero events (all 1,583 cities should appear in city stats)
    city_ev_stats = hw_city.merge(cw_city, on="city_id", how="outer")
    for col in [c for c in city_ev_stats.columns if c != "city_id"]:
        city_ev_stats[col] = city_ev_stats[col].fillna(0)

    print(f"    Heatwave events: {len(hw_cat):,}  |  "
          f"Cold wave events: {len(cw_cat):,}")
    return city_ev_stats, event_catalog


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: UTCI thermal stress days
# ─────────────────────────────────────────────────────────────────────────────
def utci_stats(cl: pd.DataFrame) -> pd.DataFrame:
    """
    Count days per UTCI thermal stress category (Bröde et al. 2012).
    UTCI in cl is in °C.

    Two sets of category counts are produced:
      utci_<cat>      — based on daily MEAN UTCI   (average thermal load across the day)
      utci_mx_<cat>   — based on daily MAX  UTCI   (peak daytime thermal stress)

    The max-based counts (utci_mx_*) better represent the worst-case human heat exposure
    during peak hours and are recommended for heat-health impact studies. The mean-based
    counts are retained for comparability with studies that use daily-mean UTCI.
    """
    print("[5] Computing UTCI thermal stress day counts …")

    cl = cl.copy()

    def _bin_utci(series: pd.Series, prefix: str, df_base: pd.DataFrame) -> pd.DataFrame:
        """Bin a UTCI series into stress categories; return wide city×category table."""
        cat = pd.cut(series, bins=UTCI_BINS, labels=UTCI_LABELS, right=True)
        df_tmp = df_base[["city_id", "date"]].copy()
        df_tmp["utci_cat"] = cat
        counts = df_tmp.groupby(["city_id", "utci_cat"], observed=False)["date"].count()
        wide   = counts.unstack("utci_cat", fill_value=0).reset_index()
        wide.columns = ["city_id"] + [f"{prefix}{str(c)}" for c in wide.columns[1:]]

        heat_cols = [f"{prefix}{c}" for c in
                     ["moderate_heat", "strong_heat", "very_strong_heat", "extreme_heat"]]
        cold_cols = [f"{prefix}{c}" for c in
                     ["extreme_cold", "very_strong_cold", "strong_cold",
                      "moderate_cold", "slight_cold"]]
        wide[f"{prefix}heat_days"] = wide[[c for c in heat_cols if c in wide.columns]].sum(axis=1)
        wide[f"{prefix}cold_days"] = wide[[c for c in cold_cols if c in wide.columns]].sum(axis=1)
        return wide

    # ── Mean-UTCI bins (existing, daily-average thermal load) ─────────────────
    mean_wide = _bin_utci(cl["utci_mean"], "utci_", cl)

    # ── Max-UTCI bins (new, peak daytime thermal stress) ──────────────────────
    # utci_max column already in city_climate_2019_2021.parquet from 05d extraction
    if "utci_max" in cl.columns and cl["utci_max"].notna().any():
        max_wide = _bin_utci(cl["utci_max"], "utci_mx_", cl)
    else:
        print("    Note: utci_max column missing or all-NaN — skipping peak-UTCI bins")
        max_wide = pd.DataFrame({"city_id": cl["city_id"].unique()})

    # ── Summary annual stats ───────────────────────────────────────────────────
    utci_summ = cl.groupby("city_id").agg(
        utci_mean_ann = ("utci_mean", "mean"),
        utci_max_ann  = ("utci_max",  "max"),   # annual max of daily-max UTCI (peak stress)
        utci_min_ann  = ("utci_mean", "min"),   # annual min of daily-mean UTCI (coldest day)
    ).reset_index()

    out = utci_summ.merge(mean_wide, on="city_id", how="left")
    out = out.merge(max_wide,  on="city_id", how="left")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Sample Koppen-Geiger at city centroids
# ─────────────────────────────────────────────────────────────────────────────
def sample_koppen(lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Sample the 0.1° Koppen-Geiger TIF at each city centroid.
    Returns DataFrame: city_id | kg_value | kg_code | kg_class | kg_desc
    """
    print("[6] Sampling Koppen-Geiger climate class …")
    try:
        import rasterio
    except ImportError:
        print("  rasterio not installed — skipping KG classification")
        return pd.DataFrame({"city_id": lookup["city_id"],
                             "kg_code": "?", "kg_class": "?", "kg_desc": "?"})

    coords = list(zip(lookup["lon"], lookup["lat"]))

    with rasterio.open(KG_TIF) as src:
        # sample() returns an iterator of arrays, one per point
        values = np.array([v[0] for v in src.sample(coords)], dtype=np.uint8)

    kg_codes  = [KG_MAP.get(int(v), ("?", "?", "Unknown"))[0] for v in values]
    kg_classes = [KG_MAP.get(int(v), ("?", "?", "Unknown"))[1] for v in values]
    kg_descs   = [KG_MAP.get(int(v), ("?", "?", "Unknown"))[2] for v in values]

    result = pd.DataFrame({
        "city_id":  lookup["city_id"].values,
        "kg_value": values,
        "kg_code":  kg_codes,
        "kg_class": kg_classes,
        "kg_desc":  kg_descs,
    })

    print(f"    KG class distribution:")
    print(result.groupby("kg_class")["city_id"].count().sort_values(ascending=False).to_string())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Assemble city-level stats table
# ─────────────────────────────────────────────────────────────────────────────
def assemble_city_stats(temp_stats, exday_stats, ev_stats, utci_stats_df,
                        kg_df, lookup) -> pd.DataFrame:
    print("[7] Assembling city statistics table …")
    df = lookup.copy()
    df = df.merge(temp_stats,   on="city_id", how="left")
    df = df.merge(exday_stats,  on="city_id", how="left")
    df = df.merge(ev_stats,     on="city_id", how="left")
    df = df.merge(utci_stats_df,on="city_id", how="left")
    df = df.merge(kg_df,        on="city_id", how="left")

    # Fill NaN event counts with 0
    event_count_cols = [c for c in df.columns
                        if any(c.startswith(p) for p in
                               ("hw_n","cw_n","n_hot","n_cold","utci_","hw_annual","cw_annual"))]
    df[event_count_cols] = df[event_count_cols].fillna(0)

    print(f"    Table shape: {df.shape}  |  cities: {df['city_id'].nunique()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Add city names to event catalog
# ─────────────────────────────────────────────────────────────────────────────
def enrich_catalog(event_catalog: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Add city name and country to event catalog for readability."""
    cat = event_catalog.merge(
        lookup[["city_id", "city", "country"]],
        on="city_id", how="left"
    )
    # Reorder: identifiers first
    front = ["city_id", "city", "country", "event_type", "event_id", "year"]
    rest  = [c for c in cat.columns if c not in front]
    return cat[front + rest]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Export CSV + Shapefile
# ─────────────────────────────────────────────────────────────────────────────
def export_outputs(city_stats: pd.DataFrame, event_catalog: pd.DataFrame,
                   ef: pd.DataFrame):
    print("[8] Exporting outputs …")

    # ── CSV (full column names) ───────────────────────────────────────────
    csv_path = OUT_DIR / "city_climate_stats.csv"
    city_stats.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path.name}  ({city_stats.shape[0]} rows × {city_stats.shape[1]} cols)")

    # ── Shapefile (geometry from FUA shapefile) ───────────────────────────
    try:
        import geopandas as gpd
        gdf = gpd.read_file(FUA_SHP)
        gdf["city_id"] = gdf.index.astype(str).str.zfill(4)

        # Bring in only stats columns that don't already exist in gdf
        # (avoids duplicate city_id and any other name collisions)
        gdf_existing = set(gdf.columns)  # city, country, fuacode, fuaname_en, city_id, geometry, …
        exclude_from_stats = {"city", "country", "lat", "lon"} | gdf_existing
        new_cols = [c for c in city_stats.columns if c not in exclude_from_stats]

        stats_sub = city_stats[["city_id"] + new_cols]  # city_id used as join key only
        gdf_out = gdf.merge(stats_sub, on="city_id", how="left")

        # Defensive: drop any accidental duplicate columns (keeps first occurrence)
        if not gdf_out.columns.is_unique:
            dups = gdf_out.columns[gdf_out.columns.duplicated()].tolist()
            print(f"  Dropping duplicate columns: {dups}")
            gdf_out = gdf_out.loc[:, ~gdf_out.columns.duplicated()]

        # Ensure it is a proper GeoDataFrame with CRS
        gdf_out = gpd.GeoDataFrame(gdf_out, geometry="geometry", crs=gdf.crs)

        # ── Truncate column names to ≤10 chars for ESRI Shapefile ──────────
        # Build rename map; resolve collisions against ALL existing short names
        short_names: dict[str, str] = {}   # orig → short (for columns being renamed)
        reserved = {c for c in gdf_out.columns if len(c) <= 10 and c != "geometry"}

        for orig in gdf_out.columns:
            if orig == "geometry" or len(orig) <= 10:
                continue
            candidate = orig[:10]
            # Resolve collision with a simple incrementing counter
            if candidate in reserved or candidate in short_names.values():
                base    = orig[:8]   # 8-char base leaves room for 2-digit suffix
                counter = 0
                candidate = base + f"{counter:02d}"
                while candidate in reserved or candidate in short_names.values():
                    counter += 1
                    candidate = base + f"{counter:02d}"
            short_names[orig] = candidate
            reserved.add(candidate)

        if short_names:
            gdf_out = gdf_out.rename(columns=short_names)

        shp_path = OUT_DIR / "city_climate_stats.shp"
        gdf_out.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        print(f"  ✓ {shp_path.name}  ({len(gdf_out)} features × {len(gdf_out.columns)} fields)")

        # Save column name mapping for reference
        if short_names:
            mapping_df = pd.DataFrame(
                [(short, orig) for orig, short in short_names.items()],
                columns=["shp_field", "full_name"]
            )
            mapping_df.to_csv(OUT_DIR / "shp_column_mapping.csv", index=False)
            print(f"  ✓ shp_column_mapping.csv  (field name key)")

    except Exception as e:
        import traceback
        print(f"  Warning: shapefile export failed — {e}")
        traceback.print_exc()
        print("  CSV still saved.")

    # ── Event catalog CSV ─────────────────────────────────────────────────
    cat_path = OUT_DIR / "event_catalog.csv"
    event_catalog.to_csv(cat_path, index=False)
    print(f"  ✓ {cat_path.name}  ({len(event_catalog):,} events: "
          f"{(event_catalog['event_type']=='heatwave').sum():,} HW + "
          f"{(event_catalog['event_type']=='coldwave').sum():,} CW)")

    # ── Extreme-days panel parquet (for CO2 alignment) ────────────────────
    # Slim version: essential columns only, one row per city × date
    keep = ["city_id", "date", "year",
            "tmax", "tmin", "tmean",
            "tmax_p90", "tmin_p10",
            "heat_event", "heat_event_id", "heat_day_n",
            "cold_event", "cold_event_id", "cold_day_n",
            "any_extreme", "days_since_heat", "days_since_cold"]
    slim = ef[[c for c in keep if c in ef.columns]].copy()
    slim_path = OUT_DIR / "extreme_days_panel.parquet"
    slim.to_parquet(slim_path, index=False)
    print(f"  ✓ {slim_path.name}  ({len(slim):,} rows — daily panel for CO2 alignment)")

    # ── Quick summary printout ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("SUMMARY STATISTICS ACROSS 1,583 CITIES")
    print("─" * 60)
    num_cols = {
        "Total heatwave events": "hw_n_events",
        "Total cold wave events": "cw_n_events",
        "Mean HW events/city":   "hw_n_events",
        "Mean HW duration (days)": "hw_mean_dur",
        "Mean cold wave dur (days)": "cw_mean_dur",
        "Median UTCI heat days":  "utci_heat_days",
        "Median UTCI cold days":  "utci_cold_days",
    }
    for label, col in num_cols.items():
        if col in city_stats.columns:
            val = city_stats[col].sum() if "Total" in label else city_stats[col].mean()
            print(f"  {label:<35} {val:,.1f}")

    if "kg_code" in city_stats.columns:
        print("\n  Koppen-Geiger class breakdown:")
        kg_cnt = city_stats.groupby("kg_class")["city_id"].count().sort_values(ascending=False)
        for cls, cnt in kg_cnt.items():
            print(f"    {cls}: {cnt} cities")
    print("─" * 60)

    # ── QC: Sanity-check extreme-day fractions ────────────────────────────────
    # By p90/p10 definition, ~10% of days in the baseline period are "extreme".
    # In 2019-2021 (climate change era), ~10-20% hot days and ~5-15% cold days
    # is plausible. Fractions above 30% almost certainly indicate a pipeline bug
    # (longitude wrapping, resolution mismatch, or corrupted thresholds).
    print("\n  QC: Extreme-day fraction anomalies")
    print("  (Expected ~10-20% hot, ~5-15% cold; flag >30% as suspicious)")
    if "hot_day_pct" in city_stats.columns and "cold_day_pct" in city_stats.columns:
        hot_sus  = city_stats[city_stats["hot_day_pct"]  > 30].sort_values("hot_day_pct",  ascending=False)
        cold_sus = city_stats[city_stats["cold_day_pct"] > 30].sort_values("cold_day_pct", ascending=False)
        both_sus = city_stats[
            (city_stats["hot_day_pct"] > 20) & (city_stats["cold_day_pct"] > 20)
        ]
        print(f"  Cities with hot_day_pct  > 30%: {len(hot_sus)}")
        if len(hot_sus) > 0:
            for _, r in hot_sus.head(5).iterrows():
                print(f"    {r['city_id']}  {r.get('city','?'):30s}  {r['hot_day_pct']:.1f}%")
        print(f"  Cities with cold_day_pct > 30%: {len(cold_sus)}")
        if len(cold_sus) > 0:
            for _, r in cold_sus.head(5).iterrows():
                print(f"    {r['city_id']}  {r.get('city','?'):30s}  {r['cold_day_pct']:.1f}%")
        print(f"  Cities simultaneously >20% hot AND >20% cold: {len(both_sus)}  "
              f"← highly suspicious / likely a pipeline error")
        if len(both_sus) > 0:
            for _, r in both_sus.head(5).iterrows():
                print(f"    {r['city_id']}  {r.get('city','?'):30s}  "
                      f"hot={r['hot_day_pct']:.1f}%  cold={r['cold_day_pct']:.1f}%")
        if len(hot_sus) == 0 and len(cold_sus) == 0 and len(both_sus) == 0:
            print("  ✓ All cities within expected range — no pipeline anomalies detected")
        elif len(hot_sus) + len(cold_sus) > 0:
            print(f"\n  ⚠ WARNING: {len(hot_sus)+len(cold_sus)} suspicious cities detected.")
            print("    Check: longitude wrapping in 05c/05d, resolution match, threshold table.")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("DESCRIPTIVE CLIMATE EVENT STATISTICS")
    print("1,583 FUA cities  |  2019–2021")
    print("=" * 65 + "\n")

    ef, cl, lookup = load_data()

    temp_s  = city_temp_stats(ef)
    exday_s = extreme_day_stats(ef)
    ev_s, event_catalog = event_stats(ef)
    utci_s  = utci_stats(cl)
    kg_s    = sample_koppen(lookup)

    city_stats = assemble_city_stats(temp_s, exday_s, ev_s, utci_s, kg_s, lookup)
    event_catalog = enrich_catalog(event_catalog, lookup)

    export_outputs(city_stats, event_catalog, ef)

    print("\n" + "=" * 65)
    print("Outputs saved to: outputs/06_descriptive/")
    print("  city_climate_stats.csv / .shp  — per-city statistics + geometry")
    print("  event_catalog.csv              — each event with start/end dates")
    print("  extreme_days_panel.parquet     — daily flags for CO2 alignment")
    print("  shp_column_mapping.csv         — shapefile field name key")
    print("=" * 65)


if __name__ == "__main__":
    main()
