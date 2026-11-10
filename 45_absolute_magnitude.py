#!/usr/bin/env python
"""
45_absolute_magnitude.py  --  how much CO2, not how many per cent
=================================================================

Every response number in the paper is a percentage. That is the right unit for
estimation and the wrong one for a closing statement: a reader cannot tell
whether a 2.96 % response is a rounding error or a national policy problem.

This script converts the estimated responses into mass of CO2, using only
quantities already established:

    dCO2  =  beta  x  sigma_c  x  Ebar_c  x  N_c

  beta     the cumulative response per SD of load (T3, Tier A+B)
  sigma_c  country c's within-country-month SD of the exposure series, which is
           the scale beta is expressed in
  Ebar_c   country c's mean daily sectoral CO2 (Carbon Monitor Cities)
  N_c      the number of country-days in a stated exposure band

Three magnitudes are reported, in increasing scope:

  1. per day    the CO2 on a day one SD above the country-month norm
  2. per event  the same, accumulated over a mean-duration spell
  3. per year   summed over the days that actually occur in the panel,
                relative to each country-month's own mean

WHAT THIS IS AND IS NOT
-----------------------
It is a linear rescaling of a within-country-month estimate into mass units. It
is NOT a counterfactual: it does not claim what emissions would have been in a
world without thermal extremes, because the fixed effects absorb the level and
only the daily deviation is identified. The correct reading is "CO2 associated
with the deviation of thermal load from its country-month norm".

Because beta is per SD, the country's own sigma enters, so a country with a
large exposure SD converts the same beta into more mass. That is intended: it
is what makes the aggregate reflect exposure as well as sensitivity.

USAGE
-----
  python 45_absolute_magnitude.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
TAB = RESULTS / "SI" / "tables"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
# Mirror the pipeline exactly: ONE directory, not a recursive glob, and skip
# the alternative disaggregations. A recursive glob over CarbonData/ counts
# China three times (the per-country file, the GADM-prefecture variant, and the
# 2019-2022 merge) and the United States twice.
CARBON = Path(os.environ.get(
    "HC_CARBON_DIR", str(BASE / "CarbonData" / "merged_2019_2022")))
SKIP = ("all-cities", "GADM", "Counties")

COOL, HEAT = "w_SCDH24", "w_HDH15"
SECTORS = ["Power", "Industry"]
MEAN_DUR = {"heat": 4.35, "cold": 3.86}          # Table T1e


# --------------------------------------------------------------------------
def national_daily_co2() -> pd.DataFrame:
    """Mean daily sectoral CO2 per country, kt/day, over the study period.

    The merged files carry no `country` column -- the country is in the file
    name, exactly as 16_r3_dynamic_response.py reads it. Mirroring that loader
    is deliberate: the magnitudes must rest on the same emission series the
    coefficients were estimated on.
    """
    rows = []
    for f in sorted(glob.glob(str(CARBON / "carbon-monitor-cities-*-v0325.csv"))):
        name = Path(f).stem.replace("carbon-monitor-cities-", "").replace(
            "-v0325", "")
        if any(k in name for k in SKIP):
            continue
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip().lower().replace("\ufeff", "") for c in d.columns]
        val = [c for c in d.columns if c.startswith("value")]
        if not val or "sector" not in d.columns or "date" not in d.columns:
            continue
        d = d[d.sector.isin(SECTORS)]
        if not len(d):
            continue
        d["date"] = pd.to_datetime(d.date, errors="coerce")
        d = d[(d.date >= "2019-01-01") & (d.date <= "2022-12-31")]
        nat = d.groupby(["sector", "date"])[val[0]].sum().reset_index()
        g = nat.groupby("sector")[val[0]].mean().rename(
            "kt_per_day").reset_index()
        g["country"] = name
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    dup = out.duplicated(["country", "sector"]).sum()
    if dup:
        raise AssertionError(
            f"{dup} duplicate country-sector rows: the emission files are "
            "being double counted")
    print(f"    {out.country.nunique()} countries loaded from {CARBON.name}")
    return out


def exposure_sd() -> pd.DataFrame:
    """Within-country-month SD of each exposure series -- the scale of beta."""
    ex = pd.read_parquet(EXPOSURE, columns=["country", "date", "tier",
                                            COOL, HEAT])
    ex["date"] = pd.to_datetime(ex.date)
    ex = ex[ex.tier.isin(["A", "B"])]
    ex["ym"] = ex.date.dt.to_period("M")
    out = []
    for side, col in (("heat", COOL), ("cold", HEAT)):
        dev = ex[col] - ex.groupby(["country", "ym"])[col].transform("mean")
        g = dev.groupby(ex.country).std().rename("sd").reset_index()
        g["side"] = side
        # days on which the deviation exceeds one SD, per country
        n_hi = (dev > dev.groupby(ex.country).transform("std")).groupby(
            ex.country).sum().rename("n_days_above_1sd").reset_index()
        out.append(g.merge(n_hi, on="country"))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    beta = pd.read_csv(RESULTS / "R3" / "tables" / "T3_sectoral_response.csv")
    beta = beta[(beta.form == "continuous") & beta.sector.isin(SECTORS)]
    beta = beta[["sector", "exposure", "cum", "cum_se"]].rename(
        columns={"exposure": "side"})

    co2 = national_daily_co2()
    sd = exposure_sd()
    d = sd.merge(co2, on="country").merge(beta, on=["sector", "side"])
    d = d[d.kt_per_day.notna() & (d.kt_per_day > 0)]

    # 1. a day one SD above the country-month norm
    d["kt_per_1sd_day"] = d.cum * d.kt_per_day
    # 2. accumulated over one mean-duration spell
    d["kt_per_event"] = d.kt_per_1sd_day * d.side.map(MEAN_DUR)
    # 3. summed over the >1 SD days that actually occur, per year
    d["mt_per_year"] = d.kt_per_1sd_day * d.n_days_above_1sd / 1e3 / 4

    tot = d.groupby(["sector", "side"]).agg(
        countries=("country", "nunique"),
        mean_daily_co2_kt=("kt_per_day", "sum"),
        kt_per_1sd_day=("kt_per_1sd_day", "sum"),
        kt_per_event=("kt_per_event", "sum"),
        mt_per_year=("mt_per_year", "sum")).round(1).reset_index()

    TAB.mkdir(parents=True, exist_ok=True)
    d.round(4).to_csv(TAB / "TS21_absolute_magnitude_by_country.csv",
                      index=False)
    tot.to_csv(TAB / "TS21b_absolute_magnitude_totals.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nTier A+B totals\n" + "-" * 74)
    print(tot.to_string(index=False))

    print("\nAnnual sectoral emissions in the response sample (Mt CO2/yr)")
    ann = d.groupby(["sector", "side"]).kt_per_day.sum() * 365 / 1e3
    print(ann.groupby("sector").first().round(0).to_string())

    print("\nLargest contributors, Power (Mt CO2/yr associated with the "
          "deviation)\n" + "-" * 74)
    for side in ("heat", "cold"):
        top = d[(d.sector == "Power") & (d.side == side)].nlargest(
            6, "mt_per_year")[["country", "sd", "n_days_above_1sd",
                               "kt_per_day", "mt_per_year"]]
        print(f"\n  {side}:")
        print(top.round(1).to_string(index=False))

    print(f"\n    wrote {TAB / 'TS21_absolute_magnitude_by_country.csv'}")
    print(f"    wrote {TAB / 'TS21b_absolute_magnitude_totals.csv'}")


if __name__ == "__main__":
    main()
