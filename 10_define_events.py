#!/usr/bin/env python
"""
10_define_events.py  --  study_design_v2.md, Part III
=====================================================

Build the full extreme-event taxonomy for 1,583 FUA cities, 2019-2021.

Supersedes the event-flagging half of `04_define_events.py`, which knew only
`tmax_p90` / `tmin_p10` and had no humidity typing or physiological grading.

THREE ORTHOGONAL CRITERIA (Part III.1)
--------------------------------------
  A  Relative  (percentile vs 1991-2020 local climatology)  -> IS it extreme?
  B  State-variable difference (wet-bulb depression)        -> humid or dry?
  C  Absolute physiological thresholds                      -> how severe?

Only A needs a 30-year baseline, and that baseline already exists. B and C are
computed from same-period data, which is why no extra baseline download for
UTCI / Twb / Twbg / Twc was ever required.

WET-BULB DEPRESSION IS HOURLY
-----------------------------
`daily Tmax - daily Twb_max` is not a physical quantity: the two maxima occur
at different hours. WBD is formed hourly in 08_build_moist_air.py and only
then aggregated. This script uses `WBD_at_tmax`, the depression at the hour of
peak dry-bulb -- the moment cooling load peaks.

MONTH-BOUNDARY RECOMBINATION
----------------------------
08 writes one file per UTC month, but aggregates to LOCAL calendar days, so a
city-day at a month boundary is split across two files. This script recombines
them with type-correct rules (sum degree-hours, max the maxima, min the
minima, n_hours-weighted mean the means) before flagging anything.

INPUTS
------
  outputs/city_climate_2019_2021.parquet          Tmax/Tmin per city-day
  ERA5/baseline/threshold_table_doy_full.parquet  from 09_extend_thresholds.py
  outputs/moist_air/city_day/moist_YYYY-MM.parquet
  outputs/utci/city_utci_daily_2019_2021.parquet
  CHI_Data/processed_daily_polygon/{AT,HI,Twbg,Twc}/YYYY/*.parquet

OUTPUT
------
  outputs/04_events/city_events_daily_v2.parquet   one row per city-day

USAGE
-----
  python 10_define_events.py
  python 10_define_events.py --min-spell 3 --heat-pct 90
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CLIMATE = BASE / "outputs" / "city_climate_2019_2021.parquet"
THRESH = BASE / "ERA5" / "baseline" / "threshold_table_doy_full.parquet"
MOIST_DIR = BASE / "outputs" / "moist_air" / "city_day"
UTCI = BASE / "outputs" / "utci" / "city_utci_daily_2019_2021.parquet"
CHI_DIR = BASE / "CHI_Data" / "processed_daily_polygon"
OUT_DIR = BASE / "outputs" / "04_events"
OUT = OUT_DIR / "city_events_daily_v2.parquet"
TABLE_DIR = OUT_DIR / "tables"

STUDY_START, STUDY_END = "2019-01-01", "2021-12-31"

# ---- Absolute physiological thresholds (Part III.2 criterion C) ------------
UTCI_CUTS = {"utci_moderate": 26.0, "utci_strong": 32.0, "utci_verystrong": 38.0}
UTCI_COLD_CUTS = {"utci_coldstress": 0.0, "utci_strongcold": -13.0}
TWBG_CUTS = {"twbg_danger": 28.0, "twbg_highrisk": 32.0}
TWB_CUTS = {"twb_stress": 28.0, "twb_severe": 31.0}
TWC_CUTS = {"twc_danger": -10.0, "twc_frostbite": -27.0}
TWB_HUMID_ABS = 26.0        # CHHE
TWB_HUMID_ABS_PLUS = 28.0   # CHHE+

LEAP_MONTH_START = np.array(
    [0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
    dtype=np.int16,
)


def climatological_doy(dates: pd.Series) -> np.ndarray:
    """Fixed month-day DOY matching 09_extend_thresholds.py."""
    month = dates.dt.month.to_numpy(dtype=np.int16)
    day = dates.dt.day.to_numpy(dtype=np.int16)
    return LEAP_MONTH_START[month] + day


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_moist() -> pd.DataFrame:
    """Load all monthly moist-air files and recombine split local days."""
    files = sorted(glob.glob(str(MOIST_DIR / "moist_*.parquet")))
    if not files:
        raise SystemExit(f"No moist-air files in {MOIST_DIR}. Run 08_build_moist_air.py.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["city_id"] = df["city_id"].astype(str).str.zfill(4)

    dup = df.duplicated(["city_id", "date"], keep=False)
    n_split = int(dup.sum())
    if n_split == 0:
        return df

    sum_cols = [c for c in df.columns
                if c.startswith(("SCDH", "CDH_", "LDH", "EDH", "HDH")) or c == "n_hours"]
    max_cols = [c for c in df.columns if c.endswith("_max")]
    min_cols = [c for c in df.columns if c.endswith("_min")]
    at_cols = [c for c in df.columns if c.endswith("_at_tmax")]
    mean_cols = [c for c in df.columns
                 if c.endswith("_mean") or c in ("RH_mean", "WBD_mean")]
    known = set(sum_cols + max_cols + min_cols + at_cols + mean_cols + ["city_id", "date"])
    other = [c for c in df.columns if c not in known]

    clean = df[~dup]
    part = df[dup].copy()

    # n_hours-weighted means keep the daily mean unbiased across the split
    for c in mean_cols:
        part[c] = part[c] * part["n_hours"]

    agg = {c: "sum" for c in sum_cols}
    agg |= {c: "max" for c in max_cols}
    agg |= {c: "min" for c in min_cols}
    agg |= {c: "sum" for c in mean_cols}
    agg |= {c: "first" for c in other}
    fixed = part.groupby(["city_id", "date"], as_index=False).agg(agg)
    for c in mean_cols:
        fixed[c] = fixed[c] / fixed["n_hours"]

    # *_at_tmax must come from whichever fragment held the higher dry-bulb peak
    if at_cols:
        pick = part.loc[part.groupby(["city_id", "date"])["t2m_max"].idxmax(),
                        ["city_id", "date"] + at_cols]
        fixed = fixed.drop(columns=at_cols, errors="ignore").merge(
            pick, on=["city_id", "date"], how="left")

    out = pd.concat([clean, fixed], ignore_index=True)
    print(f"    recombined {n_split:,} split rows -> {len(fixed):,} local days")
    return out


def load_chi_daily(index: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(CHI_DIR / index / "*" / "*.parquet")))
    if not files:
        print(f"    WARNING: no CHI daily files for {index}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["city_id"] = df["city_id"].astype(str).str.zfill(4)
    return df.drop(columns=[c for c in ("n_grid_cells",) if c in df.columns])


# --------------------------------------------------------------------------
# Spells
# --------------------------------------------------------------------------
def spell_flags(flag: np.ndarray, city: np.ndarray, dates: np.ndarray, min_len: int):
    """Return (is_in_spell, spell_id) for runs of >= min_len consecutive True.

    Rows must be sorted by (city, date). Date gaps always break a spell.
    """
    flag = flag.astype(bool)
    new_block = np.empty(len(flag), dtype=bool)
    new_block[0] = True
    day_gap = (dates[1:] - dates[:-1]) != np.timedelta64(1, "D")
    new_block[1:] = ((flag[1:] != flag[:-1]) | (city[1:] != city[:-1])
                     | day_gap)
    block = np.cumsum(new_block) - 1
    length = np.bincount(block)[block]
    keep = flag & (length >= min_len)

    spell_id = np.full(len(flag), -1, dtype=np.int32)
    if keep.any():
        # A spell starts when the previous row is not a kept day OF THE SAME
        # CITY. Without the city check, a spell ending on the last day of one
        # city and one starting on the first day of the next would be merged
        # into a single event spanning two cities.
        prev_same_city = np.r_[False, keep[:-1] & (city[1:] == city[:-1])]
        starts = keep & ~prev_same_city
        spell_id[keep] = (np.cumsum(starts)[keep] - 1)
    return keep, spell_id


def main() -> None:
    # Declared up front: the argparse defaults below READ these module-level
    # names, and Python forbids `global` after a name is used in the scope.
    global STUDY_START, STUDY_END, CLIMATE, UTCI

    ap = argparse.ArgumentParser(description="Build the Part III event taxonomy.")
    ap.add_argument("--min-spell", type=int, default=3,
                    help="Consecutive days required for a heatwave/cold wave (default 3).")
    ap.add_argument("--heat-pct", type=int, default=90, choices=[90, 95, 99],
                    help="Percentile defining an extreme heat day (default 90).")
    ap.add_argument("--cold-pct", type=int, default=10, choices=[1, 5, 10],
                    help="Percentile defining an extreme cold day (default 10).")
    ap.add_argument(
        "--day-basis", choices=["baseline-utc", "local"], default="baseline-utc",
        help="Daily Tmax/Tmin used for event flags. 'baseline-utc' uses the "
             "0.25-degree centroid UTC-day series that exactly matches the "
             "1991-2020 threshold construction. 'local' uses the 0.1-degree "
             "FUA-polygon ERA5-Land series and is diagnostic only until a matched "
             "30-year polygon baseline exists. Both versions are retained as "
             "sensitivity flags (default: baseline-utc).",
    )
    # Study window and the two year-stamped inputs are parameterised so the
    # panel can be extended. Defaults reproduce the original 2019-2021 run
    # exactly, including the original file names.
    ap.add_argument("--start", default=STUDY_START,
                    help="First day of the study window (default 2019-01-01).")
    ap.add_argument("--end", default=STUDY_END,
                    help="Last day of the study window (default 2021-12-31).")
    ap.add_argument("--climate", type=Path, default=CLIMATE,
                    help="City daily climate parquet from 05d.")
    ap.add_argument("--utci", type=Path, default=UTCI,
                    help="City daily UTCI parquet.")
    args = ap.parse_args()

    STUDY_START, STUDY_END = args.start, args.end
    CLIMATE, UTCI = args.climate, args.utci

    t0 = time.time()
    print("=" * 72)
    print(f"Part III  |  extreme-event taxonomy, FUA cities, "
          f"{STUDY_START[:4]}-{STUDY_END[:4]}")
    print(f"  climate: {CLIMATE.name}")
    print(f"  utci   : {UTCI.name}")
    print("=" * 72)

    # ---- 1. climate + thresholds ----------------------------------------
    print("\n[1] Climate and DOY thresholds")
    clim = pd.read_parquet(CLIMATE)
    clim["city_id"] = clim["city_id"].astype(str).str.zfill(4)

    # Column-name compatibility. The three-year file used downstream so far
    # carries `tmax_c` / `tmin_c`; the native output of 05d_extract_era5_cities
    # names the same 0.25-degree ERA5 centroid series `tmax` / `tmin`. They are
    # the same quantity -- verified over the 1,734,968 overlapping city-days:
    # median |difference| 0.000000 K, and only two cities differ at all
    # (Castellon and Brighton, both within 0.06 deg of the prime meridian, where
    # the current longitude-wrap handling picks a different nearest cell than the
    # version that wrote the older file). Accepting both names keeps the script
    # runnable against either file without silently reading the wrong column.
    for src, dst in (("tmax", "tmax_c"), ("tmin", "tmin_c")):
        if dst not in clim.columns and src in clim.columns:
            clim = clim.rename(columns={src: dst})
    missing = [c for c in ("tmax_c", "tmin_c") if c not in clim.columns]
    if missing:
        raise SystemExit(
            f"{CLIMATE.name} lacks {missing} (and no tmax/tmin alias). "
            "Rerun 05d_extract_era5_cities.py.")

    clim = clim[(clim.date >= STUDY_START) & (clim.date <= STUDY_END)]
    clim["clim_doy"] = climatological_doy(clim["date"])
    print(f"    {len(clim):,} city-days | {clim.city_id.nunique():,} cities")

    if not THRESH.exists():
        raise SystemExit(f"Missing {THRESH}. Run 09_extend_thresholds.py first.")
    thr = pd.read_parquet(THRESH)
    thr["city_id"] = thr["city_id"].astype(str).str.zfill(4)
    hot, cold = f"tmax_p{args.heat_pct}", f"tmin_p{args.cold_pct:02d}"
    keep_thr = ["city_id", "doy", hot, cold, "tmin_p90", "tmax_p10",
                "tmax_mean", "tmax_sd", "tmin_mean", "tmin_sd",
                "tmax_p25", "tmax_p75"]
    thr = thr.rename(columns={"doy": "clim_doy"})
    df = clim.merge(thr[["city_id", "clim_doy"] + keep_thr[2:]],
                    on=["city_id", "clim_doy"], how="left")
    n_miss = int(df[hot].isna().sum())
    if n_miss:
        print(f"    WARNING: {n_miss:,} rows without a threshold match")
    print(f"    using {hot} for heat, {cold} for cold, spell >= {args.min_spell} days")

    # ---- 2. moist air ----------------------------------------------------
    print("\n[2] Moist-air state and degree-hours (08 output)")
    moist = load_moist()
    mcols = ["city_id", "date", "t2m_mean", "t2m_max", "t2m_min",
             "Twb_max", "Twb_mean", "Twb_min",
             "RH_mean", "RH_at_tmax", "w_max", "h_max",
             "WBD_mean", "WBD_min", "WBD_at_tmax",
             "SCDH18", "SCDH24", "CDH_Twb26", "LDH_w", "EDH_h", "HDH15",
             "SCDH24_night", "CDH_Twb26_night", "n_hours"]
    mcols = [c for c in mcols if c in moist.columns]
    df = df.merge(moist[mcols], on=["city_id", "date"], how="left")
    print(f"    merged {len(mcols) - 2} moist-air variables")
    df["local_day_complete"] = (df["n_hours"] == 24).astype("int8")
    n_partial = int((df["local_day_complete"] == 0).sum())
    print(f"    complete local days: {len(df) - n_partial:,}/{len(df):,}; "
          f"partial boundary days: {n_partial:,}")

    # ---- 3. UTCI + remaining CHI indices ---------------------------------
    print("\n[3] UTCI and CHI heat-stress indices")
    if UTCI.exists():
        u = pd.read_parquet(UTCI)
        u["city_id"] = u["city_id"].astype(str).str.zfill(4)
        # The dedicated UTCI file is authoritative. Newer climate files from
        # 05d_extract_era5_cities also carry utci_mean/max/min, and merging both
        # would produce utci_max_x / utci_max_y -- after which grade("utci_max")
        # finds no such column and skips, losing utci_strong / utci_verystrong
        # without any error. Drop the climate-side copies first.
        dupes = [c for c in df.columns
                 if c.startswith("utci_") and c in u.columns]
        if dupes:
            print(f"    dropping duplicate UTCI columns from the climate file: "
                  f"{dupes}")
            df = df.drop(columns=dupes)
        df = df.merge(u, on=["city_id", "date"], how="left")
        print("    UTCI merged")
    else:
        print(f"    WARNING: {UTCI} not found - UTCI grading will be skipped")

    for idx in ("AT", "HI", "Twbg", "Twc"):
        sub = load_chi_daily(idx)
        if sub is None:
            continue
        cols = ["city_id", "date"] + [c for c in sub.columns
                                      if c.startswith(idx) and c.endswith(("_max", "_min"))]
        df = df.merge(sub[cols], on=["city_id", "date"], how="left")
    print(f"    panel now {df.shape[1]} columns")

    df = df.sort_values(["city_id", "date"]).reset_index(drop=True)
    city = df["city_id"].to_numpy()
    dates = df["date"].to_numpy(dtype="datetime64[D]")

    # Keep both products, but do not confuse them as interchangeable time bases.
    # The baseline and UTC series are 0.25-degree ERA5 centroid values. The local
    # series is a 0.1-degree ERA5-Land FUA-polygon mean. It is valuable for local
    # degree-hours and day/night mechanisms, but its relative-event flags remain
    # diagnostic until a spatially matched 1991-2020 polygon baseline is built.
    df["tmax_utc"] = df["tmax_c"]
    df["tmin_utc"] = df["tmin_c"]
    local_ok = df["local_day_complete"].eq(1)
    df["tmax_local"] = df["t2m_max"].where(local_ok)
    df["tmin_local"] = df["t2m_min"].where(local_ok)
    if args.day_basis == "local":
        event_tmax, event_tmin = df["tmax_local"], df["tmin_local"]
    else:
        event_tmax, event_tmin = df["tmax_utc"], df["tmin_utc"]
    df["event_tmax"] = event_tmax.astype("float32")
    df["event_tmin"] = event_tmin.astype("float32")
    df["event_day_basis"] = pd.Categorical(
        np.repeat(args.day_basis, len(df)), categories=["local", "baseline-utc"]
    )

    # ---- 4. criterion A: relative threshold events -----------------------
    print("\n[4] Criterion A - relative (percentile) events")
    df["EHD"] = (event_tmax > df[hot]).astype("int8")            # extreme heat day
    df["ECD"] = (event_tmin < df[cold]).astype("int8")           # extreme cold day
    df["HN"] = (event_tmin > df.tmin_p90).astype("int8")         # hot-minimum proxy
    df["CD"] = (event_tmax < df.tmax_p10).astype("int8")         # cold day
    df["CDNH"] = (df.EHD & df.HN).astype("int8")                # compound day-night heat

    # Explicit sensitivity flags preserve the alternative time basis.
    df["EHD_utc_sens"] = (df.tmax_utc > df[hot]).astype("int8")
    df["ECD_utc_sens"] = (df.tmin_utc < df[cold]).astype("int8")
    df["EHD_local_sens"] = (local_ok & (df.tmax_local > df[hot])).astype("int8")
    df["ECD_local_sens"] = (local_ok & (df.tmin_local < df[cold])).astype("int8")

    hw, hw_id = spell_flags(df.EHD.to_numpy(), city, dates, args.min_spell)
    cw, cw_id = spell_flags(df.ECD.to_numpy(), city, dates, args.min_spell)
    df["HW"], df["hw_spell_id"] = hw.astype("int8"), hw_id
    df["CW"], df["cw_spell_id"] = cw.astype("int8"), cw_id
    for lab in ("HW", "CW"):
        prev = df.groupby("city_id")[lab].shift(1).fillna(0)
        df[f"{lab}_onset"] = ((df[lab] == 1) & (prev == 0)).astype("int8")

    # exceedance magnitudes, used by 11_event_statistics.py
    df["tmax_excess"] = (event_tmax - df[hot]).clip(lower=0).astype("float32")
    df["tmin_deficit"] = (df[cold] - event_tmin).clip(lower=0).astype("float32")
    df["tmax_anom"] = (event_tmax - df.tmax_mean).astype("float32")
    df["tmin_anom"] = (event_tmin - df.tmin_mean).astype("float32")
    df["tmax_z"] = ((event_tmax - df.tmax_mean) /
                    df.tmax_sd.replace(0, np.nan)).astype("float32")

    for c in ("EHD", "ECD", "HW", "CW", "HN", "CD", "CDNH"):
        print(f"    {c:5s} {100 * df[c].mean():5.2f}% of city-days"
              + (f"   ({int(df[c + '_onset'].sum()):,} events)" if c in ("HW", "CW") else ""))

    # ---- 5. criterion B: humid vs dry typing -----------------------------
    print("\n[5] Criterion B - humid vs dry heat (hourly wet-bulb depression)")
    if "WBD_at_tmax" in df.columns:
        hot_days = df[df.EHD == 1]
        q = (hot_days.groupby("city_id")["WBD_at_tmax"]
             .quantile([0.33, 0.67]).unstack())
        q.columns = ["wbd_p33", "wbd_p67"]
        df = df.merge(q, on="city_id", how="left")
        df["HEH"] = ((df.EHD == 1) & (df.WBD_at_tmax < df.wbd_p33)).astype("int8")
        df["DEH"] = ((df.EHD == 1) & (df.WBD_at_tmax > df.wbd_p67)).astype("int8")
        df["CHHE"] = ((df.EHD == 1) & (df.Twb_max >= TWB_HUMID_ABS)).astype("int8")
        df["CHHE_plus"] = ((df.EHD == 1) & (df.Twb_max >= TWB_HUMID_ABS_PLUS)).astype("int8")
        for c in ("HEH", "DEH", "CHHE", "CHHE_plus"):
            share = 100 * df.loc[df.EHD == 1, c].mean()
            print(f"    {c:10s} {int(df[c].sum()):7,} days  ({share:5.1f}% of extreme-heat days)")
    else:
        print("    SKIPPED - WBD_at_tmax absent; run 08_build_moist_air.py")

    # ---- 6. criterion C: absolute physiological grading -------------------
    print("\n[6] Criterion C - absolute physiological thresholds")
    def grade(col, cuts, above=True):
        if col not in df.columns:
            # Loud on purpose. A silent skip here once dropped the whole UTCI
            # grade family from the panel because a merge had renamed the source
            # column, and nothing downstream failed -- 14_national_exposure just
            # listed them as "absent columns skipped" and R3.2 lost two of its
            # seven constructs.
            near = [c for c in df.columns if c.startswith(col.split("_")[0])]
            print(f"    WARNING: '{col}' absent - skipping {list(cuts)}. "
                  f"Similar columns present: {near}")
            return
        for name, cut in cuts.items():
            df[name] = ((df[col] > cut) if above else (df[col] < cut)).astype("int8")
            print(f"    {name:18s} {100 * df[name].mean():5.2f}%  ({col} "
                  f"{'>' if above else '<'} {cut} C)")
    grade("utci_max", UTCI_CUTS, True)
    grade("utci_min", UTCI_COLD_CUTS, False)
    grade("Twbg_max", TWBG_CUTS, True)
    grade("Twb_max", TWB_CUTS, True)
    grade("Twc_min", TWC_CUTS, False)

    # ---- 7. save ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT)

    # ---- human-readable Part III tables ---------------------------------
    definitions = [
        ("EHD", "extreme heat day", f"event_tmax > {hot}", "relative"),
        ("ECD", "extreme cold day", f"event_tmin < {cold}", "relative"),
        ("HW", "heatwave day", f">= {args.min_spell} consecutive EHD days", "spell"),
        ("CW", "cold-wave day", f">= {args.min_spell} consecutive ECD days", "spell"),
        ("HN", "hot daily-minimum proxy", "event_tmin > tmin_p90", "relative"),
        ("CD", "cold day", "event_tmax < tmax_p10", "relative"),
        ("CDNH", "compound hot day and hot minimum", "EHD & HN", "compound"),
        ("HEH", "humid extreme-heat day", "EHD & WBD_at_tmax < city EHD p33", "typing"),
        ("DEH", "dry extreme-heat day", "EHD & WBD_at_tmax > city EHD p67", "typing"),
        ("CHHE", "compound humid-heat day", "EHD & Twb_max >= 26 C", "compound"),
        ("CHHE_plus", "severe compound humid-heat day", "EHD & Twb_max >= 28 C", "compound"),
    ]
    for name, cut in UTCI_CUTS.items():
        definitions.append((name, name.replace("_", " "), f"utci_max > {cut} C", "absolute"))
    for name, cut in UTCI_COLD_CUTS.items():
        definitions.append((name, name.replace("_", " "), f"utci_min < {cut} C", "absolute"))
    for name, cut in TWBG_CUTS.items():
        definitions.append((name, name.replace("_", " "), f"Twbg_max > {cut} C", "absolute"))
    for name, cut in TWB_CUTS.items():
        definitions.append((name, name.replace("_", " "), f"Twb_max > {cut} C", "absolute"))
    for name, cut in TWC_CUTS.items():
        definitions.append((name, name.replace("_", " "), f"Twc_min < {cut} C", "absolute"))
    def_df = pd.DataFrame(definitions, columns=["code", "name", "definition", "criterion"])
    def_df["primary_day_basis"] = args.day_basis
    def_df["baseline_day_basis"] = "ERA5 daily UTC statistics, 1991-2020"
    def_df["relative_event_support"] = "ERA5 0.25-degree city centroid"
    def_df["moist_air_support"] = "ERA5-Land/CHI 0.1-degree FUA polygon mean, local day"
    def_df.to_csv(TABLE_DIR / "table_01_event_definitions.csv", index=False)

    valid_source = {
        **{c: None for c in ("EHD", "ECD", "HW", "CW", "HN", "CD", "CDNH", "HEH", "DEH", "CHHE", "CHHE_plus")},
        **{c: "utci_max" for c in UTCI_CUTS},
        **{c: "utci_min" for c in UTCI_COLD_CUTS},
        **{c: "Twbg_max" for c in TWBG_CUTS},
        **{c: "Twb_max" for c in TWB_CUTS},
        **{c: "Twc_min" for c in TWC_CUTS},
    }
    rows = []
    for c in def_df.code:
        if c not in df.columns:
            continue
        src = valid_source.get(c)
        valid = df[src].notna() if src else pd.Series(True, index=df.index)
        rows.append({
            "event": c,
            "n_valid_city_days": int(valid.sum()),
            "n_event_days": int(df.loc[valid, c].sum()),
            "share_of_valid_days_pct": float(100 * df.loc[valid, c].mean()),
            "n_cities_with_event": int(df.loc[df[c].eq(1), "city_id"].nunique()),
            "mean_days_per_city_year": float(df[c].sum() / (df.city_id.nunique() * 3)),
        })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table_02_event_day_counts.csv", index=False)

    coverage = []
    for c in [hot, cold, "event_tmax", "event_tmin", "Twb_max", "WBD_at_tmax",
              "utci_max", "Twbg_max", "Twc_min", "SCDH24", "LDH_w", "EDH_h"]:
        if c in df.columns:
            coverage.append({"variable": c, "n_rows": len(df),
                             "n_missing": int(df[c].isna().sum()),
                             "coverage_pct": float(100 * df[c].notna().mean())})
    coverage.append({"variable": "complete_local_day_24h", "n_rows": len(df),
                     "n_missing": int((df.local_day_complete == 0).sum()),
                     "coverage_pct": float(100 * df.local_day_complete.mean())})
    pd.DataFrame(coverage).to_csv(TABLE_DIR / "table_00_data_coverage.csv", index=False)

    agreement = []
    for kind in ("EHD", "ECD"):
        a, b = f"{kind}_utc_sens", f"{kind}_local_sens"
        valid = local_ok
        agreement.append({
            "event": kind,
            "n_valid": int(valid.sum()),
            "agreement_pct": float(100 * (df.loc[valid, a] == df.loc[valid, b]).mean()),
            "both_event": int((df.loc[valid, a].eq(1) & df.loc[valid, b].eq(1)).sum()),
            "utc_only": int((df.loc[valid, a].eq(1) & df.loc[valid, b].eq(0)).sum()),
            "local_only": int((df.loc[valid, a].eq(0) & df.loc[valid, b].eq(1)).sum()),
        })
    pd.DataFrame(agreement).to_csv(TABLE_DIR / "table_03_day_basis_agreement.csv", index=False)
    print(f"\n[7] saved {OUT}")
    print(f"    {len(df):,} city-days x {df.shape[1]} columns "
          f"({OUT.stat().st_size / 1024**2:.1f} MB)")
    print(f"\nDone in {time.time() - t0:.0f}s.  Next: python 11_event_statistics.py")


if __name__ == "__main__":
    main()
