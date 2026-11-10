#!/usr/bin/env python
"""
08_build_moist_air.py
=====================

Combine hourly ERA5 dry-bulb temperature (from 05f) with the existing hourly
CHI wet-bulb temperature to recover the full moist-air state, then build the
exposure variables the analysis needs.

WHAT THIS SOLVES
----------------
Dry-bulb T alone cannot separate sensible from latent cooling load. Wet-bulb
Twb alone confounds the two. Together they pin down the psychrometric state:

    e   = actual vapour pressure          -> humidity
    RH  = relative humidity
    w   = humidity ratio    (kg/kg)       -> LATENT load driver
    h   = moist-air enthalpy (kJ/kg)      -> TOTAL load driver
    WBD = T - Twb                         -> evaporative cooling potential

WBD IS COMPUTED HOURLY, THEN AGGREGATED
---------------------------------------
`Tmax_daily - Twb_max_daily` is not a physical quantity: the two maxima
generally occur at different hours. This script forms WBD_h = T_h - Twb_h at
each hour and only then aggregates.

INVERSION MODE: USE stull (SETTLED FROM THE CHI DATA DESCRIPTOR)
---------------------------------------------------------------
Malik et al. (2026), Sci Data 13:200, Table 3 row 5 and Eq. (7):

    Twb | input variables: T2m, rh | method: Stull | code: thermofeel

CHI Twb is the EMPIRICAL Stull (2011) wet-bulb, a function of (T2m, rh) only.
It is not a thermodynamic wet-bulb, so:

  stull           (default, correct) Invert Stull for rh given (T2m, Twb).
                  Exact -- round-trip error ~1e-12 pp -- and pressure-free.

  psychrometric   e = e_s(Twb) - A*P*(T - Twb). Correct for a THERMODYNAMIC
                  wet-bulb, which this is not. Kept only as a cross-check;
                  it disagrees with Stull by up to ~4.5 pp of rh.

  both            Runs both and reports the round-trip residual, so the
                  choice is re-verifiable rather than taken on trust.

Note the same paper computes Tnwb and Twbg with the physically-based
Liljegren model. That applies to Tnwb/Twbg, NOT to Twb.

Pressure enters only the humidity ratio w and enthalpy h. rh, vapour pressure
and WBD are all pressure-free.

LOCAL TIME
----------
CHI and ERA5 hourly data are UTC. Day/night partitions are meaningless in UTC
for East Asia and the Americas, so hours are shifted to local solar time
(timezonefinder if installed, otherwise longitude/15) before splitting.

INPUTS
------
  ERA5/processed_hourly_polygon/t2m/{year}/era5_t2m_{yyyy}-{mm}_city_hour.parquet
  CHI_Data/processed_hourly_polygon/Twb/{year}/chi_Twb_{yyyy}-{mm}_city_hour.parquet
  outputs/fua_city_lookup.csv                      (for longitude)
  ERA5/city_surface_pressure.parquet               (optional; see --pressure)

OUTPUT
------
  outputs/moist_air/city_day/moist_{yyyy}-{mm}.parquet

USAGE
-----
  python 08_build_moist_air.py --mode both --years 2019 --months 7   # diagnose
  python 08_build_moist_air.py                                       # full run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ERA5_HOURLY = BASE / "ERA5" / "processed_hourly_polygon" / "t2m"
CHI_HOURLY = BASE / "CHI_Data" / "processed_hourly_polygon" / "Twb"
LOOKUP = BASE / "outputs" / "fua_city_lookup.csv"
PRESSURE_TABLE = BASE / "ERA5" / "city_surface_pressure.parquet"
OUT_DIR = BASE / "outputs" / "moist_air" / "city_day"

# ---- Pre-registered thresholds (vary these in the sensitivity analysis) ----
BASE_SENSIBLE_C = 18.0     # conventional CDD base
BASE_COMFORT_C = 24.0      # AC activation / comfort limit
BASE_TWB_C = 26.0          # significant latent load
BASE_W = 0.0120            # kg/kg, ASHRAE 55 upper humidity ratio of comfort zone
BASE_H = 56.8              # kJ/kg, enthalpy at the (26 degC, 0.012 kg/kg) corner
BASE_HEATING_C = 15.0      # heating degree-hour base

A_PSYCHRO = 6.6e-4         # K^-1, aspirated psychrometer coefficient
P_STANDARD = 1013.25       # hPa


# --------------------------------------------------------------------------
# Psychrometrics
# --------------------------------------------------------------------------
def sat_vapour_pressure(t_c):
    """Saturation vapour pressure over water, hPa (Bolton 1980)."""
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def rh_from_psychrometric(t_c, twb_c, p_hpa):
    """RH (%) from dry-bulb, thermodynamic wet-bulb and station pressure."""
    e = sat_vapour_pressure(twb_c) - A_PSYCHRO * p_hpa * (t_c - twb_c)
    e = np.clip(e, 1e-6, None)
    rh = 100.0 * e / sat_vapour_pressure(t_c)
    return np.clip(rh, 0.1, 100.0)


def stull_twb(t_c, rh_pct):
    """Stull (2011) empirical wet-bulb temperature from dry-bulb and RH."""
    rh = np.clip(rh_pct, 0.1, 100.0)
    return (
        t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def rh_from_stull(t_c, twb_c, iters: int = 45):
    """Invert Stull (2011) for RH by vectorised bisection. Pressure-free."""
    lo = np.full_like(t_c, 0.1, dtype="float64")
    hi = np.full_like(t_c, 100.0, dtype="float64")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_low = stull_twb(t_c, mid) < twb_c      # Twb increases with RH
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def moist_air_state(t_c, rh_pct, p_hpa):
    """Return vapour pressure (hPa), humidity ratio (kg/kg), enthalpy (kJ/kg)."""
    e = np.clip(sat_vapour_pressure(t_c) * rh_pct / 100.0, 1e-6, None)
    w = 0.62198 * e / np.clip(p_hpa - e, 1e-3, None)
    h = 1.006 * t_c + w * (2501.0 + 1.86 * t_c)
    return e, w, h


# --------------------------------------------------------------------------
# Local time
# --------------------------------------------------------------------------
def build_utc_offsets(lookup: pd.DataFrame) -> pd.Series:
    """Whole-hour UTC offset per city; timezonefinder when available."""
    try:
        from timezonefinder import TimezoneFinder
        import pytz
        from datetime import datetime

        tf = TimezoneFinder()
        ref = datetime(2020, 1, 15)
        offsets = {}
        for row in lookup.itertuples():
            name = tf.timezone_at(lat=float(row.lat), lng=float(row.lon))
            if name is None:
                offsets[row.city_id] = int(round(float(row.lon) / 15.0))
            else:
                off = pytz.timezone(name).utcoffset(ref)
                offsets[row.city_id] = int(round(off.total_seconds() / 3600.0))
        print("  local time: timezonefinder (IANA zones, standard time)")
        return pd.Series(offsets, name="utc_offset")
    except ImportError:
        print("  local time: longitude/15 approximation "
              "(install timezonefinder+pytz for IANA zones)")
        return pd.Series(
            (lookup["lon"] / 15.0).round().astype(int).values,
            index=lookup["city_id"].values,
            name="utc_offset",
        )


# --------------------------------------------------------------------------
# Per-month build
# --------------------------------------------------------------------------
def positive_part_sum(series, base):
    return np.maximum(series - base, 0.0).sum()


def build_month(year: int, month: int, offsets: pd.Series,
                pressure: pd.Series | float, mode: str) -> pd.DataFrame | None:
    t_path = ERA5_HOURLY / f"{year}" / f"era5_t2m_{year}-{month:02d}_city_hour.parquet"
    w_path = CHI_HOURLY / f"{year}" / f"chi_Twb_{year}-{month:02d}_city_hour.parquet"
    if not t_path.exists():
        print(f"  missing ERA5 hourly t2m: {t_path.name} - run 05f first")
        return None
    if not w_path.exists():
        print(f"  missing CHI hourly Twb: {w_path.name}")
        return None

    t = pd.read_parquet(t_path)[["city_id", "datetime", "t2m"]]
    wb = pd.read_parquet(w_path)[["city_id", "datetime", "Twb"]]
    df = t.merge(wb, on=["city_id", "datetime"], how="inner")
    if df.empty:
        print("  no overlapping city-hours after merge")
        return None

    df["p_hpa"] = (
        float(pressure) if np.isscalar(pressure)
        else df["city_id"].map(pressure).fillna(P_STANDARD).to_numpy()
    )

    t_c = df["t2m"].to_numpy(dtype="float64")
    twb_c = np.minimum(df["Twb"].to_numpy(dtype="float64"), t_c)   # Twb <= T
    p = df["p_hpa"].to_numpy(dtype="float64")

    if mode in ("psychrometric", "both"):
        rh_psy = rh_from_psychrometric(t_c, twb_c, p)
    if mode in ("stull", "both"):
        rh_stu = rh_from_stull(t_c, twb_c)

    if mode == "both":
        # Round-trip residual: which definition reproduces CHI's Twb?
        res_stu = stull_twb(t_c, rh_stu) - twb_c
        e_psy = np.clip(
            sat_vapour_pressure(twb_c) - A_PSYCHRO * p * (t_c - twb_c), 1e-6, None
        )
        res_psy = stull_twb(t_c, 100.0 * e_psy / sat_vapour_pressure(t_c)) - twb_c
        print(
            f"  inversion diagnostic | median |Twb residual|: "
            f"stull {np.nanmedian(np.abs(res_stu)):.4f} K, "
            f"psychrometric {np.nanmedian(np.abs(res_psy)):.4f} K, "
            f"median RH gap {np.nanmedian(np.abs(rh_stu - rh_psy)):.2f} pp"
        )
        rh = rh_stu          # Stull is the definition CHI actually used
    else:
        rh = rh_psy if mode == "psychrometric" else rh_stu

    e, w, h = moist_air_state(t_c, rh, p)
    df["RH"] = rh.astype("float32")
    df["w"] = w.astype("float32")
    df["h"] = h.astype("float32")
    df["WBD"] = (t_c - twb_c).astype("float32")

    # ---- local time ----
    off = df["city_id"].map(offsets).fillna(0).astype(int)
    df["local_dt"] = df["datetime"] + pd.to_timedelta(off, unit="h")
    df["date"] = df["local_dt"].dt.floor("D")
    hour = df["local_dt"].dt.hour
    df["is_night"] = (hour >= 18) | (hour < 6)

    # ---- daily aggregation ----
    g = df.groupby(["city_id", "date"], sort=False)
    daily = g.agg(
        t2m_mean=("t2m", "mean"), t2m_max=("t2m", "max"), t2m_min=("t2m", "min"),
        Twb_mean=("Twb", "mean"), Twb_max=("Twb", "max"), Twb_min=("Twb", "min"),
        RH_mean=("RH", "mean"), RH_min=("RH", "min"),
        w_mean=("w", "mean"), w_max=("w", "max"),
        h_mean=("h", "mean"), h_max=("h", "max"),
        WBD_mean=("WBD", "mean"), WBD_min=("WBD", "min"),
        n_hours=("t2m", "size"),
    ).reset_index()

    # WBD at the hour of peak dry-bulb: the moment AC load peaks
    peak = df.loc[g["t2m"].idxmax(), ["city_id", "date", "WBD", "RH", "w"]]
    peak = peak.rename(columns={"WBD": "WBD_at_tmax", "RH": "RH_at_tmax", "w": "w_at_tmax"})
    daily = daily.merge(peak, on=["city_id", "date"], how="left")

    # ---- degree-hours (total, day, night) ----
    dh_specs = {
        "SCDH18": ("t2m", BASE_SENSIBLE_C, 1),
        "SCDH24": ("t2m", BASE_COMFORT_C, 1),
        "CDH_Twb26": ("Twb", BASE_TWB_C, 1),
        "LDH_w": ("w", BASE_W, 1),
        "EDH_h": ("h", BASE_H, 1),
        "HDH15": ("t2m", BASE_HEATING_C, -1),
    }
    for name, (col, base, sign) in dh_specs.items():
        excess = np.maximum(sign * (df[col].to_numpy(dtype="float64") - base), 0.0)
        tmp = pd.DataFrame(
            {"city_id": df["city_id"].values, "date": df["date"].values,
             "x": excess, "night": df["is_night"].values}
        )
        tot = tmp.groupby(["city_id", "date"], sort=False)["x"].sum().rename(name)
        night = (
            tmp[tmp["night"]].groupby(["city_id", "date"], sort=False)["x"]
            .sum().rename(f"{name}_night")
        )
        daily = daily.merge(tot, on=["city_id", "date"], how="left")
        daily = daily.merge(night, on=["city_id", "date"], how="left")
        daily[f"{name}_night"] = daily[f"{name}_night"].fillna(0.0)
        daily[f"{name}_day"] = daily[name] - daily[f"{name}_night"]

    for col in daily.columns:
        if daily[col].dtype == "float64":
            daily[col] = daily[col].astype("float32")
    return daily


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive moist-air state and degree-hours.")
    ap.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021])
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--mode", choices=["psychrometric", "stull", "both"],
                    default="stull",
                    help="Humidity inversion. 'stull' is correct for CHI Twb "
                         "(Malik et al. 2026, Table 3) and is exact and "
                         "pressure-free. 'both' additionally prints the "
                         "round-trip residual as a check (default: stull).")
    ap.add_argument("--pressure", default="auto",
                    help="'auto' uses ERA5/city_surface_pressure.parquet when present "
                         "and falls back to 1013.25 hPa; or pass a number in hPa.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    lookup = pd.read_csv(LOOKUP, dtype={"city_id": str})
    lookup["city_id"] = lookup["city_id"].str.zfill(4)
    print(f"Cities in lookup: {len(lookup):,}")
    offsets = build_utc_offsets(lookup)

    if args.pressure == "auto":
        if PRESSURE_TABLE.exists():
            pt = pd.read_parquet(PRESSURE_TABLE)
            pressure = pt.set_index("city_id")["p_hpa"]
            print(f"  pressure: per-city table ({len(pressure):,} cities)")
        else:
            pressure = P_STANDARD
            print(f"  pressure: CONSTANT {P_STANDARD} hPa - no per-city table found.\n"
                  "    Under --mode stull this affects ONLY humidity ratio w and\n"
                  "    enthalpy h (LDH_w, EDH_h). rh, vapour pressure and WBD are\n"
                  "    pressure-free and exact. For high-altitude cities (Mexico\n"
                  "    City, Bogota, Denver, Nairobi) build a per-city table from\n"
                  "    ERA5-Land monthly-mean surface_pressure before using LDH/EDH.")
    else:
        pressure = float(args.pressure)
        print(f"  pressure: constant {pressure} hPa")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = 0
    for year in args.years:
        for month in args.months:
            out = OUT_DIR / f"moist_{year}-{month:02d}.parquet"
            if out.exists() and not args.overwrite:
                continue
            print(f"\n{year}-{month:02d}")
            daily = build_month(year, month, offsets, pressure, args.mode)
            if daily is None:
                continue
            tmp = out.with_suffix(".part.parquet")
            daily.to_parquet(tmp, index=False)
            tmp.replace(out)
            print(f"  wrote {len(daily):,} city-days -> {out.name}")
            built += 1

    print(f"\nBuilt {built} monthly files in {OUT_DIR}")


if __name__ == "__main__":
    main()
