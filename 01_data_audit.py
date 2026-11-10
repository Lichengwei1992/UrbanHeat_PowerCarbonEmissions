"""
01_data_audit.py
================
Profile all Carbon Monitor Cities data and produce a summary report.
Run from the project root: python 01_data_audit.py
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
CITY_DIR = os.path.join(BASE, "CarbonData",
    "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide")
ALL_CITIES_FILE = os.path.join(CITY_DIR, "carbon-monitor-cities-all-cities-FUA-v0325.csv")
GLOBAL_FILE = os.path.join(BASE, "CarbonData",
    "Global_daily_sectoral_CO2_emissions_from_1970_2024_updated.csv")
OUT_DIR = os.path.join(BASE, "outputs", "01_data_audit")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all-cities file ───────────────────────────────────────────────────────
print("Loading all-cities file …")
df = pd.read_csv(ALL_CITIES_FILE, encoding="utf-8-sig", parse_dates=["date"])
df.columns = df.columns.str.strip()
df.rename(columns={"value (KtCO2 per day)": "value_ktco2"}, inplace=True)
df["city"] = df["city"].str.strip()
df["country"] = df["country"].str.strip()
df["sector"] = df["sector"].str.strip()

print(f"  Rows: {len(df):,}")

# ── Basic statistics ───────────────────────────────────────────────────────────
n_cities    = df["city"].nunique()
n_countries = df["country"].nunique()
n_sectors   = df["sector"].nunique()
date_min    = df["date"].min()
date_max    = df["date"].max()
n_days      = (date_max - date_min).days + 1

print(f"\n{'='*60}")
print("CARBON MONITOR CITIES — DATA AUDIT")
print(f"{'='*60}")
print(f"  Cities       : {n_cities:,}")
print(f"  Countries    : {n_countries}")
print(f"  Sectors      : {n_sectors}  →  {sorted(df['sector'].unique())}")
print(f"  Date range   : {date_min.date()} to {date_max.date()} ({n_days} days)")
print(f"  Total rows   : {len(df):,}")
print(f"  Missing vals : {df['value_ktco2'].isna().sum():,} ({df['value_ktco2'].isna().mean()*100:.2f}%)")
print(f"  Zero vals    : {(df['value_ktco2']==0).sum():,} ({(df['value_ktco2']==0).mean()*100:.2f}%)")
print(f"  Negative vals: {(df['value_ktco2']<0).sum():,}")

# ── Cities per country ─────────────────────────────────────────────────────────
cities_per_country = (
    df.groupby("country")["city"].nunique()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"city": "n_cities"})
)
print(f"\n  Top 15 countries by city count:")
print(cities_per_country.head(15).to_string(index=False))
cities_per_country.to_csv(os.path.join(OUT_DIR, "cities_per_country.csv"), index=False)

# ── Temporal coverage per city ─────────────────────────────────────────────────
print("\nChecking temporal coverage per city …")
expected_days_per_sector = n_days  # each city-sector should have this many rows
coverage = (
    df.groupby(["city", "country", "sector"])
    .agg(
        obs=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        missing_pct=("value_ktco2", lambda x: x.isna().mean() * 100),
        zero_pct=("value_ktco2", lambda x: (x == 0).mean() * 100),
        mean_val=("value_ktco2", "mean"),
    )
    .reset_index()
)
coverage["completeness_pct"] = (coverage["obs"] / expected_days_per_sector * 100).clip(upper=100)
coverage.to_csv(os.path.join(OUT_DIR, "coverage_by_city_sector.csv"), index=False)

# Summary of completeness
print(f"  Median completeness (all city-sector pairs): {coverage['completeness_pct'].median():.1f}%")
print(f"  City-sector pairs with <90% completeness : {(coverage['completeness_pct']<90).sum()}")
print(f"  City-sector pairs with 100% completeness : {(coverage['completeness_pct']==100).sum()}")

# ── Sector-level statistics ────────────────────────────────────────────────────
sector_stats = (
    df.groupby("sector")["value_ktco2"]
    .agg(["count", "mean", "std", "min", "max",
          lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
    .round(4)
)
print(f"\n  Sector statistics (KtCO2/day):")
print(sector_stats.to_string())
sector_stats.to_csv(os.path.join(OUT_DIR, "sector_stats.csv"))

# ── Annual totals per sector ───────────────────────────────────────────────────
df["year"] = df["date"].dt.year
annual = (
    df.groupby(["year", "sector"])["value_ktco2"]
    .sum()
    .unstack("sector")
    .round(1)
)
print(f"\n  Annual totals by sector (KtCO2):")
print(annual.to_string())
annual.to_csv(os.path.join(OUT_DIR, "annual_totals_by_sector.csv"))

# ── Country × year totals ──────────────────────────────────────────────────────
country_year = (
    df.groupby(["country", "year"])["value_ktco2"]
    .sum()
    .unstack("year")
    .round(1)
    .sort_values(2019, ascending=False)
)
print(f"\n  Top 10 countries by total city CO2 (KtCO2):")
print(country_year.head(10).to_string())
country_year.to_csv(os.path.join(OUT_DIR, "country_year_totals.csv"))

# ── Identify candidate heatwave months (data availability check) ───────────────
# Key events we'll study: Jun-Jul 2019 (EU), May-Jun 2019 (India), Jun-Jul 2021 (PNW)
heatwave_windows = {
    "2019 EU heatwave": ("2019-06-01", "2019-08-31"),
    "2019 India heatwave": ("2019-05-01", "2019-06-30"),
    "2021 Pacific NW heat dome": ("2021-06-20", "2021-07-10"),
    "2021 Western US / Mediterranean": ("2021-07-01", "2021-08-31"),
}

print(f"\n{'='*60}")
print("DATA AVAILABILITY FOR KEY HEATWAVE EVENTS")
print(f"{'='*60}")
for event, (start, end) in heatwave_windows.items():
    mask = (df["date"] >= start) & (df["date"] <= end)
    sub = df[mask]
    n_c = sub["city"].nunique()
    n_rows = len(sub)
    print(f"\n  {event} ({start} to {end})")
    print(f"    Cities with data: {n_c}")
    print(f"    Rows: {n_rows:,}")

# ── Global file audit ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("GLOBAL SECTORAL FILE AUDIT")
print(f"{'='*60}")
gdf = pd.read_csv(GLOBAL_FILE, parse_dates=["date"])
print(f"  Rows        : {len(gdf):,}")
print(f"  Countries   : {sorted(gdf['country'].unique())}")
print(f"  Sectors     : {sorted(gdf['sector'].unique())}")
print(f"  Date range  : {gdf['date'].min().date()} to {gdf['date'].max().date()}")
print(f"  Missing vals: {gdf['value'].isna().sum()}")

# 2019–2021 subset for alignment check
gdf_sub = gdf[gdf["date"].dt.year.isin([2019, 2020, 2021])]
print(f"\n  2019–2021 subset: {len(gdf_sub):,} rows")
print(f"  Countries present: {gdf_sub['country'].nunique()}")

gdf_sub.groupby(["country", "sector"])["value"].mean().unstack("sector").round(2).to_csv(
    os.path.join(OUT_DIR, "global_mean_2019_2021.csv")
)

print(f"\n{'='*60}")
print(f"Audit complete. Outputs saved to: {OUT_DIR}")
print(f"{'='*60}")
