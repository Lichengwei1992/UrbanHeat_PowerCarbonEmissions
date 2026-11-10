#!/usr/bin/env python
"""
11_event_statistics.py  --  study_design_v2.md, Part IV
=======================================================

Turn the daily event flags from 10_define_events.py into the Part IV
statistical variable system: intensity, frequency, duration, timing/context,
and anomaly -- at event level, city-year level, and city level.

Supersedes the event-summary half of `06_descriptive_climate.py`.

WHAT PART IV ASKS FOR
---------------------
  IV.1 Intensity   peak / mean / cumulative exceedance, HWMId, peak heat-stress
                   index values, cooling degree-hours accumulated in-event
  IV.2 Frequency   events and event-days per city-year, by event type
  IV.3 Duration    per-event length, annual max/mean, count of long events
  IV.4 Timing      onset DOY, season-first flag, days since previous event,
                   events already elapsed this season, spatial synchrony
  IV.5 Anomaly     deviation from 1991-2020 city-DOY climatology

HWMId IS COMPUTED CANONICALLY
-----------------------------
Russo et al. (2015) normalise daily exceedance by the interquartile range of
the series of ANNUAL MAXIMA of daily Tmax over the reference period -- not by
the day-of-year IQR. This script derives T25p/T75p from the 30 annual maxima
per city in the 1991-2020 baseline, so the values are comparable with the
published index rather than a look-alike.

    Md(Td) = (Td - T25p) / (T75p - T25p)   if Td > T25p, else 0
    HWMId  = sum of Md over the days of one event

SEASON YEAR
-----------
"First event of the season" is meaningless on calendar years in the southern
hemisphere, where summer straddles New Year. Cities with lat < 0 are assigned
a July-June season year so that season-first is defined consistently on both
hemispheres.

INPUTS
------
  outputs/04_events/city_events_daily_v2.parquet      from 10_define_events.py
  ERA5/baseline/city_baseline_all_years.parquet       for the HWMId normaliser
  outputs/fua_city_lookup.csv                         lat/lon, country

OUTPUTS
-------
  outputs/04_events/event_catalog_heat.parquet    one row per heatwave
  outputs/04_events/event_catalog_cold.parquet    one row per cold wave
  outputs/04_events/city_year_event_stats.parquet city x calendar-year
  outputs/04_events/city_event_summary.parquet    city (3-year averages)

USAGE
-----
  python 11_event_statistics.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
BASELINE = BASE / "ERA5" / "baseline" / "city_baseline_all_years.parquet"
LOOKUP = BASE / "outputs" / "fua_city_lookup.csv"
OUT_DIR = BASE / "outputs" / "04_events"
TABLE_DIR = OUT_DIR / "tables"

LONG_EVENT_DAYS = 5      # "long" heatwave for IV.3
MIN_COMPLETE_MOIST_DAYS = 360  # annual coverage rule for hourly/local-day metrics


# --------------------------------------------------------------------------
# IV.1 HWMId normaliser
# --------------------------------------------------------------------------
def hwmid_normaliser() -> pd.DataFrame:
    """T25p / T75p of the 30 annual maxima of daily Tmax per city (Russo 2015)."""
    b = pd.read_parquet(BASELINE, columns=["city_id", "date", "tmax_c"])
    b["city_id"] = b["city_id"].astype(str).str.zfill(4)
    ann = b.groupby(["city_id", b.date.dt.year])["tmax_c"].max().reset_index(name="ann_max")
    q = ann.groupby("city_id")["ann_max"].quantile([0.25, 0.75]).unstack()
    q.columns = ["T25p", "T75p"]
    q["hwmid_iqr"] = (q.T75p - q.T25p).replace(0, np.nan)
    return q.reset_index()


# --------------------------------------------------------------------------
# IV.4 season year
# --------------------------------------------------------------------------
def add_time_context(df: pd.DataFrame, lat: pd.Series) -> pd.DataFrame:
    """Add calendar year and hemisphere-correct heat/cold season years."""
    south = df["city_id"].map(lat) < 0
    year = df["date"].dt.year
    month = df["date"].dt.month
    df["calendar_year"] = year
    # Heat season: calendar year in the north; July-June in the south.
    df["heat_season_year"] = year.where(
        ~south, np.where(month >= 7, year, year - 1))
    # Cold season: July-June in the north; calendar year in the south.
    df["cold_season_year"] = year.where(
        south, np.where(month >= 7, year, year - 1))
    df["is_south"] = south.astype("int8")
    return df


# --------------------------------------------------------------------------
# Event catalogue
# --------------------------------------------------------------------------
def build_catalog(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One row per event. kind is 'heat' or 'cold'."""
    if kind == "heat":
        sid, excess, season_col = "hw_spell_id", "tmax_excess", "heat_season_year"
        peak_cols = {
            "peak_tmax": ("event_tmax", "max"),
            "peak_tmax_z": ("tmax_z", "max"),
            "peak_twb": ("Twb_max", "max"),
            "peak_utci": ("utci_max", "max"),
            "peak_twbg": ("Twbg_max", "max"),
            "peak_at": ("AT_max", "max"),
            "peak_hi": ("HI_max", "max"),
            "mean_wbd_at_tmax": ("WBD_at_tmax", "mean"),
            "min_wbd_at_tmax": ("WBD_at_tmax", "min"),
            "cum_scdh24": ("SCDH24", "sum"),
            "cum_cdh_twb26": ("CDH_Twb26", "sum"),
            "cum_edh_h": ("EDH_h", "sum"),
            "cum_ldh_w": ("LDH_w", "sum"),
            "n_hot_nights": ("HN", "sum"),
            "n_compound_daynight": ("CDNH", "sum"),
            "n_humid_days": ("CHHE", "sum"),
        }
    else:
        sid, excess, season_col = "cw_spell_id", "tmin_deficit", "cold_season_year"
        peak_cols = {
            "peak_tmin": ("event_tmin", "min"),
            "peak_twc": ("Twc_min", "min"),
            "peak_utci": ("utci_min", "min"),
            "cum_hdh15": ("HDH15", "sum"),
        }

    ev = df[df[sid] >= 0]
    if ev.empty:
        return pd.DataFrame()

    agg = {
        "city_id": ("city_id", "first"),
        "event_season_year": (season_col, "first"),
        "start_date": ("date", "min"),
        "end_date": ("date", "max"),
        "duration": ("date", "size"),
        "peak_excess": (excess, "max"),
        "mean_excess": (excess, "mean"),
        "cum_excess": (excess, "sum"),
        "mean_anom": ("tmax_anom" if kind == "heat" else "tmin_anom", "mean"),
        "peak_anom": ("tmax_anom" if kind == "heat" else "tmin_anom",
                      "max" if kind == "heat" else "min"),
        "mean_spatial_sync": ("spatial_sync", "mean"),
        "max_spatial_sync": ("spatial_sync", "max"),
        "mean_spatial_sync_loo": ("spatial_sync_loo", "mean"),
        "max_spatial_sync_loo": ("spatial_sync_loo", "max"),
    }
    agg |= {k: v for k, v in peak_cols.items() if v[0] in df.columns}
    cat = (ev.groupby(sid, sort=True).agg(**agg).reset_index()
             .rename(columns={sid: "event_id"}))

    # IV.1 HWMId (heat only)
    if kind == "heat" and {"T25p", "hwmid_iqr"}.issubset(df.columns):
        md = ((ev["event_tmax"] - ev["T25p"]) / ev["hwmid_iqr"]).clip(lower=0)
        hwmid = md.groupby(ev[sid]).sum()
        cat["hwmid"] = cat["event_id"].map(hwmid)

    # IV.4 timing and context
    cat["onset_doy"] = cat.start_date.dt.dayofyear
    cat["calendar_year"] = cat.start_date.dt.year
    cat = cat.sort_values(["city_id", "start_date"]).reset_index(drop=True)
    g = cat.groupby(["city_id", "event_season_year"])
    cat["event_seq_in_season"] = g.cumcount() + 1
    cat["is_season_first"] = (cat.event_seq_in_season == 1).astype("int8")
    cat["n_prior_events_in_season"] = cat.event_seq_in_season - 1
    prev_end = cat.groupby("city_id")["end_date"].shift(1)
    cat["days_since_prev_event"] = (cat.start_date - prev_end).dt.days
    next_start = cat.groupby("city_id")["start_date"].shift(-1)
    cat["days_to_next_event"] = (next_start - cat.end_date).dt.days
    cat["is_long_event"] = (cat.duration >= LONG_EVENT_DAYS).astype("int8")
    if kind == "heat" and {"n_humid_days", "duration"}.issubset(cat.columns):
        cat["humid_day_fraction"] = cat.n_humid_days / cat.duration
        if "n_compound_daynight" in cat.columns:
            cat["compound_daynight_fraction"] = cat.n_compound_daynight / cat.duration
    cat["event_type"] = kind
    return cat


# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("Part IV  |  event statistics: intensity, frequency, duration, timing")
    print("=" * 72)

    if not EVENTS.exists():
        raise SystemExit(f"Missing {EVENTS}. Run 10_define_events.py first.")

    lk = pd.read_csv(LOOKUP, dtype={"city_id": str})
    lk["city_id"] = lk["city_id"].str.zfill(4)
    lat = lk.set_index("city_id")["lat"]
    country = lk.set_index("city_id")["country"]

    print("\n[1] Loading daily event panel")
    df = pd.read_parquet(EVENTS)
    df["city_id"] = df["city_id"].astype(str).str.zfill(4)
    df["country"] = df["city_id"].map(country)
    df = add_time_context(df, lat)
    print(f"    {len(df):,} city-days | {df.city_id.nunique():,} cities "
          f"| {int(df.is_south.sum() / 1096):,} southern-hemisphere cities")

    print("\n[2] HWMId normaliser from 1991-2020 annual maxima")
    norm = hwmid_normaliser()
    df = df.merge(norm, on="city_id", how="left")
    print(f"    T25p median {norm.T25p.median():.1f} C, "
          f"T75p median {norm.T75p.median():.1f} C, "
          f"IQR median {norm.hwmid_iqr.median():.2f} K")

    # IV.4 spatial synchrony: raw and leave-one-out country shares. LOO avoids
    # mechanically correlating a city's own EHD flag with its synchrony value.
    print("\n[3] Spatial synchrony (share of same-country cities in extreme heat)")
    cg = df.groupby(["country", "date"])["EHD"]
    hot_sum = cg.transform("sum")
    n_city = cg.transform("size")
    df["spatial_sync"] = hot_sum / n_city
    df["spatial_sync_loo"] = np.where(
        n_city > 1, (hot_sum - df.EHD) / (n_city - 1), np.nan
    ).astype("float32")
    print(f"    median {df.spatial_sync.median():.3f}, "
          f"p95 {df.spatial_sync.quantile(0.95):.3f}")

    print("\n[4] Building event catalogues")
    heat = build_catalog(df, "heat")
    cold = build_catalog(df, "cold")
    print(f"    heatwaves : {len(heat):,} events, "
          f"mean duration {heat.duration.mean():.2f} d, "
          f"max {int(heat.duration.max())} d")
    print(f"    cold waves: {len(cold):,} events, "
          f"mean duration {cold.duration.mean():.2f} d, "
          f"max {int(cold.duration.max())} d")
    if "hwmid" in heat.columns:
        print(f"    HWMId     : median {heat.hwmid.median():.2f}, "
              f"p99 {heat.hwmid.quantile(0.99):.2f}, max {heat.hwmid.max():.1f}")

    # ---- IV.2 / IV.3 city x calendar-year -----------------------------
    print("\n[5] City x calendar-year aggregates")
    day_flags = ["EHD", "ECD", "HW", "CW", "HN", "CD", "CDNH",
                 "HEH", "DEH", "CHHE", "CHHE_plus",
                 "utci_strong", "utci_verystrong", "twbg_danger", "twb_stress"]
    day_flags = [c for c in day_flags if c in df.columns]
    cy = df.groupby(["city_id", "calendar_year"]).agg(
        n_days=("date", "size"),
        n_complete_local_days=("local_day_complete", "sum"),
        **{f"n_{c}_days": (c, "sum") for c in day_flags},
        cum_tmax_excess=("tmax_excess", "sum"),
        cum_tmin_deficit=("tmin_deficit", "sum"),
        cum_scdh24=("SCDH24", "sum"),
        cum_cdh_twb26=("CDH_Twb26", "sum"),
        cum_hdh15=("HDH15", "sum"),
        mean_tmax_anom=("tmax_anom", "mean"),
        max_tmax_z=("tmax_z", "max"),
    ).reset_index()

    for kind, cat in (("hw", heat), ("cw", cold)):
        if cat.empty:
            continue
        a = cat.groupby(["city_id", "calendar_year"]).agg(
            **{f"n_{kind}_events": ("duration", "size"),
               f"{kind}_mean_duration": ("duration", "mean"),
               f"{kind}_max_duration": ("duration", "max"),
               f"n_{kind}_long_events": ("is_long_event", "sum"),
               f"{kind}_cum_excess": ("cum_excess", "sum"),
               f"{kind}_peak_excess": ("peak_excess", "max"),
               f"{kind}_first_onset_doy": ("onset_doy", "min")}).reset_index()
        if kind == "hw" and "hwmid" in cat.columns:
            a = a.merge(cat.groupby(["city_id", "calendar_year"])["hwmid"]
                        .agg(hw_hwmid_sum="sum", hw_hwmid_max="max").reset_index(),
                        on=["city_id", "calendar_year"], how="left")
        cy = cy.merge(a, on=["city_id", "calendar_year"], how="left")

    count_cols = [c for c in cy.columns if c.startswith("n_") and c != "n_days"]
    cy[count_cols] = cy[count_cols].fillna(0)

    expected_days = cy.calendar_year.map(
        lambda y: 366 if pd.Timestamp(y, 12, 31).is_leap_year else 365)
    cy["complete_calendar_year"] = (cy.n_days == expected_days).astype("int8")
    cy["moist_coverage_pct"] = (100 * cy.n_complete_local_days / expected_days).astype("float32")
    cy["complete_moist_year_exact"] = (
        cy.n_complete_local_days == expected_days).astype("int8")
    cy["sufficient_moist_coverage"] = (
        cy.n_complete_local_days >= MIN_COMPLETE_MOIST_DAYS).astype("int8")
    complete = cy.complete_calendar_year.eq(1)
    print(f"    {len(cy):,} city-years, {int(complete.sum()):,} complete panel years; "
          f"{int(cy.sufficient_moist_coverage.sum()):,} with >= "
          f"{MIN_COMPLETE_MOIST_DAYS} complete local-moist days")
    print(f"    exact 365/366-day moist coverage: "
          f"{int(cy.complete_moist_year_exact.sum()):,} city-years (audit only)")

    print("\n[6] City-level summary (mean over complete calendar years)")
    num = cy.loc[complete].select_dtypes(include=[np.number]).columns.drop("calendar_year")
    summ = cy.loc[complete].groupby("city_id")[list(num)].mean().reset_index()
    n_complete = cy.loc[complete].groupby("city_id").size().rename("n_complete_years")
    summ = summ.merge(n_complete, on="city_id", how="left")
    summ = summ.merge(lk[["city_id", "city", "country", "lat", "lon"]], on="city_id", how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    heat = heat.merge(lk[["city_id", "city", "country"]], on="city_id", how="left")
    cold = cold.merge(lk[["city_id", "city", "country"]], on="city_id", how="left")
    heat.to_parquet(OUT_DIR / "event_catalog_heat.parquet", index=False)
    cold.to_parquet(OUT_DIR / "event_catalog_cold.parquet", index=False)
    cy.to_parquet(OUT_DIR / "city_year_event_stats.parquet", index=False)
    summ.to_parquet(OUT_DIR / "city_event_summary.parquet", index=False)

    # Full readable result tables.
    heat.to_csv(TABLE_DIR / "event_catalog_heat.csv", index=False)
    cold.to_csv(TABLE_DIR / "event_catalog_cold.csv", index=False)
    cy.merge(lk[["city_id", "city", "country"]], on="city_id", how="left").to_csv(
        TABLE_DIR / "city_year_event_stats.csv", index=False)
    summ.to_csv(TABLE_DIR / "city_event_summary.csv", index=False)

    moist_coverage = cy.groupby("calendar_year").agg(
        n_city_years=("city_id", "size"),
        min_complete_local_days=("n_complete_local_days", "min"),
        mean_complete_local_days=("n_complete_local_days", "mean"),
        n_exact_complete=("complete_moist_year_exact", "sum"),
        n_sufficient_coverage=("sufficient_moist_coverage", "sum"),
        mean_coverage_pct=("moist_coverage_pct", "mean"),
    ).reset_index()
    moist_coverage["minimum_required_days"] = MIN_COMPLETE_MOIST_DAYS
    moist_coverage.to_csv(
        TABLE_DIR / "table_03b_moist_year_coverage.csv", index=False)

    annual_days = df.groupby("calendar_year")[day_flags].sum().add_prefix("n_").add_suffix("_days")
    annual_events = pd.DataFrame(index=sorted(df.calendar_year.unique()))
    annual_events["n_hw_events"] = heat.groupby("calendar_year").size()
    annual_events["n_cw_events"] = cold.groupby("calendar_year").size()
    annual = annual_days.join(annual_events).fillna(0).reset_index()
    annual.to_csv(TABLE_DIR / "table_04_global_annual_event_counts.csv", index=False)

    intensity_rows = []
    for event_type, cat in (("heat", heat), ("cold", cold)):
        for var in ["duration", "peak_excess", "mean_excess", "cum_excess",
                    "hwmid", "peak_twb", "peak_utci", "peak_twbg",
                    "cum_scdh24", "cum_cdh_twb26", "cum_hdh15"]:
            if var not in cat.columns:
                continue
            s = cat[var].dropna()
            intensity_rows.append({
                "event_type": event_type, "variable": var, "n": len(s),
                "mean": s.mean(), "sd": s.std(), "p05": s.quantile(.05),
                "p25": s.quantile(.25), "p50": s.median(),
                "p75": s.quantile(.75), "p95": s.quantile(.95), "max": s.max(),
            })
    pd.DataFrame(intensity_rows).to_csv(
        TABLE_DIR / "table_05_event_intensity_summary.csv", index=False)

    dist_rows = []
    dist_vars = [c for c in cy.columns if c.startswith(("n_", "cum_", "hw_", "cw_"))]
    for var in dist_vars:
        s = cy.loc[complete, var].dropna()
        if not len(s):
            continue
        dist_rows.append({
            "variable": var, "n_city_years": len(s), "mean": s.mean(),
            "sd": s.std(), "p05": s.quantile(.05), "p25": s.quantile(.25),
            "p50": s.median(), "p75": s.quantile(.75), "p95": s.quantile(.95),
            "min": s.min(), "max": s.max(),
        })
    pd.DataFrame(dist_rows).to_csv(
        TABLE_DIR / "table_06_city_year_distribution.csv", index=False)

    cy_country = cy.loc[complete].merge(
        lk[["city_id", "country"]], on="city_id", how="left")
    country_vars = [c for c in ["n_hw_events", "n_cw_events", "n_EHD_days",
                                "n_ECD_days", "n_CDNH_days", "n_HEH_days",
                                "n_DEH_days", "n_CHHE_days", "cum_scdh24",
                                "cum_hdh15"] if c in cy_country.columns]
    country_table = cy_country.groupby("country").agg(
        n_cities=("city_id", "nunique"), n_city_years=("city_id", "size"),
        **{f"mean_{c}": (c, "mean") for c in country_vars},
    ).reset_index()
    country_table.to_csv(TABLE_DIR / "table_07_country_event_summary.csv", index=False)

    timing_rows = []
    for event_type, cat in (("heat", heat), ("cold", cold)):
        timing_rows.append({
            "event_type": event_type, "n_events": len(cat),
            "share_season_first_pct": 100 * cat.is_season_first.mean(),
            "onset_doy_p05": cat.onset_doy.quantile(.05),
            "onset_doy_p50": cat.onset_doy.median(),
            "onset_doy_p95": cat.onset_doy.quantile(.95),
            "days_since_prev_p50": cat.days_since_prev_event.median(),
            "days_since_prev_p95": cat.days_since_prev_event.quantile(.95),
        })
    pd.DataFrame(timing_rows).to_csv(
        TABLE_DIR / "table_08_event_timing_summary.csv", index=False)

    rank_rows = []
    rank_metrics = [c for c in ["n_hw_events", "n_cw_events", "n_CHHE_days",
                                "n_CDNH_days", "hw_hwmid_max"] if c in summ.columns]
    for metric in rank_metrics:
        top = summ.nlargest(25, metric)
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            rank_rows.append({"metric": metric, "rank": rank,
                              "city_id": row.city_id, "city": row.city,
                              "country": row.country, "value": getattr(row, metric)})
    pd.DataFrame(rank_rows).to_csv(
        TABLE_DIR / "table_09_top_cities_event_burden.csv", index=False)

    print("\n[7] Saved")
    for f in ("event_catalog_heat", "event_catalog_cold",
              "city_year_event_stats", "city_event_summary"):
        p = OUT_DIR / f"{f}.parquet"
        print(f"    {f + '.parquet':34s} {p.stat().st_size / 1024**2:7.2f} MB")

    print("\n[8] Sanity check - global averages per city-year")
    for c in ["n_hw_events", "hw_mean_duration", "n_EHD_days", "n_CDNH_days",
              "n_CHHE_days", "hw_hwmid_sum", "n_cw_events", "cum_scdh24"]:
        if c in summ.columns:
            print(f"    {c:20s} mean {summ[c].mean():9.2f}   "
                  f"p50 {summ[c].median():8.2f}   p95 {summ[c].quantile(.95):9.2f}")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
