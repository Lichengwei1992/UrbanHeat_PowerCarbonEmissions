#!/usr/bin/env python
"""
14_national_exposure.py  --  city-weighted national thermal exposure
=====================================================================

Step 2 of the revised pipeline: collapse the 1,583-city daily exposure panel
into a country-day panel using population weights, so it can be paired with
national daily sectoral CO2 in step 3.

WEIGHTING IS DEDUPLICATION-SAFE
-------------------------------
34 emission records share 14 FUA polygons (Guangzhou1...Guangzhou8 are eight
Carbon Monitor cities on one polygon). Each record therefore carries
`pop_weight_dedup = polygon population / n_share`, so the eight Guangzhou
records together contribute the polygon's population exactly once. Because all
records on a shared polygon have identical weather, this is algebraically the
same as aggregating over the 1,563 distinct polygons.

WHAT IS PRODUCED
----------------
For each country and day:

  * `w_*`   population-weighted mean of an exposure variable
  * `e_*`   equal-weighted mean (internal contrast: does weighting matter?)
  * `hhi_*` Herfindahl concentration of the exposure across cities -- two
            countries can share a total exposure while one has it concentrated
            in a few cities and the other spread out
  * `n_cities`, `pop_total`, `pop_in_event`

Event flags aggregate to a share of urban population under that event, which
is the natural national analogue of a city-level indicator.

INPUTS
------
  outputs/04_events/city_events_daily_v2.parquet     from 10_define_events.py
  outputs/population/fua_city_weights.parquet        from 13_fua_population.py
  outputs/population/country_coverage.parquet        for the Tier flag
  outputs/fua_city_lookup.csv

OUTPUT
------
  outputs/exposure/country_daily_exposure.parquet    country x date
  outputs/exposure/tables/*.csv

USAGE
-----
  python 14_national_exposure.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
WEIGHTS = BASE / "outputs" / "population" / "fua_city_weights.parquet"
COVERAGE = BASE / "outputs" / "population" / "country_coverage.parquet"
LOOKUP = BASE / "outputs" / "fua_city_lookup.csv"
OUT_DIR = BASE / "outputs" / "exposure"
TABLE_DIR = OUT_DIR / "tables"

# UCDB country labels arrive mojibake'd (UTF-8 bytes read as latin-1)
COUNTRY_FIX = {"MÃ©xico": "Mexico", "México": "Mexico", "South Korea": "Korea"}

# Event indicators -> share of urban population under that condition
EVENT_VARS = ["EHD", "ECD", "HW", "CW", "HN", "CD", "CDNH",
              "HEH", "DEH", "CHHE", "CHHE_plus",
              "utci_strong", "utci_verystrong", "twbg_danger", "twb_stress"]

# Continuous exposure -> population-weighted mean
CONT_VARS = ["tmax_anom", "tmax_z", "tmin_anom", "utci_max", "Twb_max",
             "WBD_at_tmax", "SCDH18", "SCDH24", "CDH_Twb26", "LDH_w", "EDH_h",
             "HDH15", "SCDH24_night", "SCDH24_day",
             "CDH_Twb26_night", "CDH_Twb26_day"]

# Concentration is only meaningful for non-negative intensity measures
HHI_VARS = ["SCDH24", "CDH_Twb26", "HDH15"]

TIER_CUTS = [(0.80, "A"), (0.50, "B")]      # else C


def tier_of(coverage: float, n_cities: int) -> str:
    if not np.isfinite(coverage):
        return "C"
    for cut, lab in TIER_CUTS:
        if coverage >= cut and n_cities >= 3:
            return lab
    return "C"


def main() -> None:
    t0 = time.time()
    print("=" * 74)
    print("City-weighted national thermal exposure")
    print("=" * 74)

    # ---- 1. weights ------------------------------------------------------
    print("\n[1] Population weights")
    w = pd.read_parquet(WEIGHTS)
    w["city_id"] = w["city_id"].astype(str).str.zfill(4)
    if "pop_weight_dedup" not in w.columns:
        w["pop_weight_dedup"] = w["pop_mean"] / w["n_share"]
    w = w[["city_id", "country", "fuacode", "n_share", "pop_mean", "pop_weight_dedup"]]
    bad = w["pop_weight_dedup"].isna() | (w["pop_weight_dedup"] <= 0)
    if bad.any():
        print(f"    dropping {int(bad.sum())} records with missing/zero population")
        w = w[~bad]
    print(f"    {len(w):,} records | {w.fuacode.nunique():,} polygons | "
          f"{w.pop_weight_dedup.sum() / 1e6:,.0f} M total (deduplicated)")

    # ---- 2. city-day exposure -------------------------------------------
    print("\n[2] City-day exposure panel")
    keep = ["city_id", "date"] + EVENT_VARS + CONT_VARS
    ev = pd.read_parquet(EVENTS)
    ev["city_id"] = ev["city_id"].astype(str).str.zfill(4)
    missing = [c for c in keep if c not in ev.columns]
    if missing:
        print(f"    NOTE: absent columns skipped -> {missing}")
        keep = [c for c in keep if c in ev.columns]
    ev = ev[keep]
    print(f"    {len(ev):,} city-days x {len(keep) - 2} exposure variables")

    d = ev.merge(w, on="city_id", how="inner")
    print(f"    matched {d.city_id.nunique():,} cities in {d.country.nunique()} countries")

    # ---- 3. aggregate ----------------------------------------------------
    print("\n[3] Aggregating to country-day")
    d["wt"] = d["pop_weight_dedup"]
    g = d.groupby(["country", "date"], sort=False)
    base = g.agg(n_cities=("city_id", "size"), pop_total=("wt", "sum")).reset_index()

    val_cols = [c for c in EVENT_VARS + CONT_VARS if c in d.columns]
    # weighted sums in one pass
    wsum = pd.DataFrame({c: d[c].to_numpy(dtype="float64") * d["wt"].to_numpy()
                         for c in val_cols})
    wsum["country"] = d["country"].to_numpy()
    wsum["date"] = d["date"].to_numpy()
    W = wsum.groupby(["country", "date"], sort=False)[val_cols].sum().reset_index()
    E = d.groupby(["country", "date"], sort=False)[val_cols].mean().reset_index()

    out = base.merge(W, on=["country", "date"]).merge(
        E, on=["country", "date"], suffixes=("_wsum", "_emean"))
    for c in val_cols:
        out[f"w_{c}"] = out[f"{c}_wsum"] / out["pop_total"]
        out[f"e_{c}"] = out[f"{c}_emean"]
    out = out.drop(columns=[f"{c}_wsum" for c in val_cols] +
                           [f"{c}_emean" for c in val_cols])

    # population actually under a heat / cold event
    for f in ("HW", "CW", "EHD", "CHHE"):
        if f"w_{f}" in out.columns:
            out[f"pop_in_{f}"] = out[f"w_{f}"] * out["pop_total"]

    # ---- 4. exposure concentration (HHI) --------------------------------
    print("[4] Exposure concentration")
    for c in [v for v in HHI_VARS if v in d.columns]:
        num = d[c].to_numpy(dtype="float64") * d["wt"].to_numpy()
        tmp = pd.DataFrame({"country": d["country"].to_numpy(),
                            "date": d["date"].to_numpy(), "x": num})
        tot = tmp.groupby(["country", "date"], sort=False)["x"].transform("sum")
        tmp["share2"] = np.where(tot > 0, (tmp["x"] / tot) ** 2, np.nan)
        h = tmp.groupby(["country", "date"], sort=False)["share2"].sum().rename(f"hhi_{c}")
        out = out.merge(h.reset_index(), on=["country", "date"], how="left")

    # ---- 5. coverage tier ------------------------------------------------
    print("[5] Coverage tier")
    cov = pd.read_parquet(COVERAGE)
    cov["country_gadm"] = cov["country_gadm"].astype(str).replace(COUNTRY_FIX)
    cov = cov[["country_gadm", "coverage", "uc_pop_total"]].rename(
        columns={"country_gadm": "country"})
    ncity = w.groupby("country")["fuacode"].nunique().rename("n_fua").reset_index()
    ctab = ncity.merge(cov, on="country", how="left")
    ctab["tier"] = [tier_of(c, n) for c, n in zip(ctab.coverage, ctab.n_fua)]
    print(ctab.tier.value_counts().rename_axis("tier").to_frame("n_countries")
          .to_string())
    out = out.merge(ctab[["country", "coverage", "tier", "n_fua"]],
                    on="country", how="left")

    out = out.sort_values(["country", "date"]).reset_index(drop=True)
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")

    # ---- 6. save + sanity -----------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "country_daily_exposure.parquet"
    tmp = p.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(p)
    ctab.to_csv(TABLE_DIR / "country_tier.csv", index=False)

    print(f"\n[6] Saved {p.name}: {len(out):,} country-days x {out.shape[1]} cols "
          f"({p.stat().st_size / 1024**2:.1f} MB)")
    print(f"    {out.country.nunique()} countries, "
          f"{out.date.min().date()} -> {out.date.max().date()}")

    print("\n[7] Sanity - population share under heatwave (w_HW)")
    s = out.groupby("country")["w_HW"].agg(["mean", "max"]).nlargest(8, "mean")
    for c, r in s.iterrows():
        print(f"    {c[:22]:24s} mean {100 * r['mean']:5.2f}%   max {100 * r['max']:6.2f}%")

    print("\n    Weighted vs equal-weighted divergence (does weighting matter?)")
    for v in ("HW", "SCDH24", "CDH_Twb26"):
        if f"w_{v}" in out and f"e_{v}" in out:
            a, b = out[f"w_{v}"], out[f"e_{v}"]
            ok = a.notna() & b.notna() & (b.abs() > 1e-9)
            print(f"      {v:12s} corr {np.corrcoef(a[ok], b[ok])[0,1]:.4f}   "
                  f"median |w/e - 1| {np.nanmedian((a[ok] / b[ok] - 1).abs()):.3f}")

    print(f"\nDone in {time.time() - t0:.0f}s.  Next: 15_power_response.py")


if __name__ == "__main__":
    main()
