#!/usr/bin/env python
"""
12_event_windows.py  --  event-time windows for the before/during/after design
==============================================================================

Builds everything the dynamic event-study needs that depends ONLY on climate.
Emissions enter later as the dependent variable; nothing here requires them.

WHY BUILD THIS BEFORE TOUCHING EMISSIONS
----------------------------------------
1. Overlap diagnostics. With repeated events and a 36-day window
   (tau = -14 ... +21), windows can collide. The diagnostics quantify how much
   nearest-onset profiles differ from clean, isolated-event profiles and guide
   the later choice between conventional event-time and distributed-lag models.
2. The exposure response profile -- how Tmax, UTCI, Twb and cooling
   degree-hours themselves evolve before, during and after onset -- is a
   climate result in its own right and needs no emissions data.
3. Once emissions arrive the merge is a single join.

TWO REPRESENTATIONS OF EVENT TIME
---------------------------------
  (a) tau labelling            each city-day gets the signed offset to its
                               NEAREST onset. Intuitive, good for plots, but
                               ambiguous when windows overlap.

  (b) distributed lead-lag     one indicator per offset s,
                               L_s(i,t) = HW_onset(i, t-s).
                               Overlaps superpose linearly instead of forcing
                               a choice, which is why the conceptual framework
                               writes the model as sum_k beta_k * Event_(i,t+k).
                               THIS is the estimation form.

  Sign convention: s > 0 means an onset occurred s days BEFORE t (post-event
  response); s < 0 means an onset occurs |s| days AFTER t (pre-event window,
  used for the parallel-trends check).

INPUT
-----
  outputs/04_events/city_events_daily_v2.parquet    from 10_define_events.py

OUTPUTS
-------
  outputs/04_events/city_event_windows.parquet
      city_id | date | tau_hw | tau_cw | in_hw_window | n_hw_onsets_in_window
      | clean_hw_window | hw_Lm14 ... hw_Lp21 | cw_Lm14 ... cw_Lp21
  outputs/04_events/exposure_by_tau.parquet
      mean exposure by tau, raw and city-month demeaned

USAGE
-----
  python 12_event_windows.py
  python 12_event_windows.py --leads 14 --lags 21
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
OUT_DIR = BASE / "outputs" / "04_events"
TABLE_DIR = OUT_DIR / "tables"

# Exposure variables profiled around onset (all climate-side)
PROFILE_VARS = ["tmax_anom", "tmax_z", "tmin_anom", "utci_max", "Twb_max",
                "WBD_at_tmax", "SCDH24", "CDH_Twb26", "EDH_h", "HDH15",
                "SCDH24_night", "CDH_Twb26_night", "RH_at_tmax"]


def offset_label(s: int) -> str:
    return f"m{abs(s)}" if s < 0 else f"p{s}"


def build_leadlag(onset: np.ndarray, city: np.ndarray,
                  leads: int, lags: int) -> tuple[np.ndarray, list[int]]:
    """L[:, k] = onset shifted by s_k, blanked across city boundaries.

    Rows must be sorted by (city_id, date) with no gaps inside a city.
    """
    offsets = list(range(-leads, lags + 1))
    n = len(onset)
    L = np.zeros((n, len(offsets)), dtype=np.int8)

    # Row index of the first record of each city, to stop shifts leaking across
    city_codes, first_idx = np.unique(city, return_index=True)
    city_start = np.zeros(n, dtype=np.int64)
    city_start[first_idx] = 1
    city_start = np.cumsum(city_start) - 1              # city ordinal per row
    counts = np.bincount(city_start)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    pos_in_city = np.arange(n) - starts[city_start]      # 0-based day index

    for k, s in enumerate(offsets):
        if s == 0:
            L[:, k] = onset
            continue
        if s > 0:                       # onset s days earlier
            shifted = np.r_[np.zeros(s, dtype=np.int8), onset[:-s]]
            valid = pos_in_city >= s
        else:                           # onset |s| days later
            a = -s
            shifted = np.r_[onset[a:], np.zeros(a, dtype=np.int8)]
            valid = pos_in_city < (counts[city_start] - a)
        L[:, k] = np.where(valid, shifted, 0)
    return L, offsets


def nearest_tau(L: np.ndarray, offsets: list[int]) -> np.ndarray:
    """Signed offset to the nearest onset; NaN when no onset is in window."""
    tau = np.full(L.shape[0], np.nan, dtype="float32")
    order = sorted(range(len(offsets)), key=lambda k: (abs(offsets[k]), offsets[k]))
    unfilled = np.ones(L.shape[0], dtype=bool)
    for k in order:
        hit = unfilled & (L[:, k] == 1)
        if hit.any():
            tau[hit] = offsets[k]
            unfilled &= ~hit
        if not unfilled.any():
            break
    return tau


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet through a temporary file so interrupted runs are safe."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, default=14, help="Days before onset (default 14).")
    ap.add_argument("--lags", type=int, default=21, help="Days after onset (default 21).")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 72)
    print(f"Event windows  |  tau in [-{args.leads}, +{args.lags}]")
    print("=" * 72)

    if not EVENTS.exists():
        raise SystemExit(f"Missing {EVENTS}. Run 10_define_events.py first.")

    df = pd.read_parquet(EVENTS)
    df["city_id"] = df["city_id"].astype(str).str.zfill(4)
    df = df.sort_values(["city_id", "date"]).reset_index(drop=True)
    city = df["city_id"].to_numpy()
    print(f"\n[1] {len(df):,} city-days | {df.city_id.nunique():,} cities")

    # continuity check: shifts assume no missing days inside a city
    gaps = df.groupby("city_id")["date"].diff().dt.days
    n_gap = int((gaps.dropna() != 1).sum())
    print(f"    date continuity: {n_gap} non-unit gaps inside cities"
          + ("  (OK)" if n_gap == 0 else "  <-- CHECK, shifts assume daily continuity"))

    out = df[["city_id", "date"]].copy()

    print("\n[2] Distributed lead-lag matrices")
    for kind, col in (("hw", "HW_onset"), ("cw", "CW_onset")):
        onset = df[col].to_numpy(dtype=np.int8)
        L, offsets = build_leadlag(onset, city, args.leads, args.lags)
        for k, s in enumerate(offsets):
            out[f"{kind}_L{offset_label(s)}"] = L[:, k]
        out[f"tau_{kind}"] = nearest_tau(L, offsets)
        n_in = L.sum(axis=1)
        out[f"n_{kind}_onsets_in_window"] = n_in.astype("int8")
        out[f"in_{kind}_window"] = (n_in > 0).astype("int8")
        out[f"clean_{kind}_window"] = (n_in == 1).astype("int8")
        print(f"    {kind}: {int(onset.sum()):,} onsets -> "
              f"{len(offsets)} indicator columns")

    # ---- overlap diagnostics -------------------------------------------
    print("\n[3] Overlap diagnostics (this decides the estimation form)")
    for kind in ("hw", "cw"):
        n = out[f"n_{kind}_onsets_in_window"]
        inw = n > 0
        print(f"    {kind.upper()}: {100 * inw.mean():5.1f}% of city-days lie in a window")
        vc = n[inw].value_counts().sort_index()
        for k, v in vc.head(6).items():
            print(f"        {k} onset(s) in window: {100 * v / inw.sum():5.1f}%")
        print(f"        -> single-event ('clean') share: "
              f"{100 * (n[inw] == 1).mean():5.1f}%")

    out["strict_clean_hw_window"] = (
        (out["n_hw_onsets_in_window"] == 1) &
        (out["n_cw_onsets_in_window"] == 0)
    ).astype("int8")
    out["strict_clean_cw_window"] = (
        (out["n_cw_onsets_in_window"] == 1) &
        (out["n_hw_onsets_in_window"] == 0)
    ).astype("int8")

    overlap_rows = []
    for kind, other in (("hw", "cw"), ("cw", "hw")):
        inw = out[f"in_{kind}_window"].eq(1)
        denom = int(inw.sum())
        overlap_rows.append({
            "event_type": kind.upper(),
            "n_onsets": int(df[f"{kind.upper()}_onset"].sum()),
            "n_city_days_in_window": denom,
            "share_all_city_days_in_window": float(inw.mean()),
            "share_window_days_single_same_type": (
                float(out.loc[inw, f"clean_{kind}_window"].mean()) if denom else np.nan
            ),
            "share_window_days_with_opposite_event": (
                float(out.loc[inw, f"in_{other}_window"].mean()) if denom else np.nan
            ),
            "share_window_days_strict_clean": (
                float(out.loc[inw, f"strict_clean_{kind}_window"].mean()) if denom else np.nan
            ),
        })
    overlap = pd.DataFrame(overlap_rows)

    print("\n    Reading: low clean shares do not invalidate event-time analysis,")
    print("    but they require explicit overlap handling and comparison with")
    print("    distributed lead-lag estimates.")

    # ---- exposure profile by tau ---------------------------------------
    print("\n[4] Exposure response profile by tau (climate-side, no emissions)")
    prof_vars = [v for v in PROFILE_VARS if v in df.columns]
    work = df[["city_id", "date"] + prof_vars].copy()
    work["tau_hw"] = out["tau_hw"].to_numpy()
    work["clean_hw_window"] = out["clean_hw_window"].to_numpy()
    work["strict_clean_hw_window"] = out["strict_clean_hw_window"].to_numpy()
    work["ym"] = work["date"].dt.to_period("M").astype(str)

    # demean within city x calendar-month so the profile is not just the
    # seasonal cycle showing through (events cluster in summer)
    dem = work[prof_vars] - work.groupby(["city_id", work["date"].dt.month])[prof_vars].transform("mean")
    dem.columns = [f"{c}_dm" for c in prof_vars]
    work = pd.concat([work, dem], axis=1)

    profile_parts = []
    profile_masks = {
        "nearest_all": work["tau_hw"].notna(),
        "single_heat_event": work["tau_hw"].notna() & work["clean_hw_window"].eq(1),
        "strict_clean": work["tau_hw"].notna() & work["strict_clean_hw_window"].eq(1),
    }
    profile_cols = prof_vars + list(dem.columns)
    for sample, mask in profile_masks.items():
        part = (work.loc[mask]
                    .groupby("tau_hw")[profile_cols]
                    .mean().reset_index())
        n_obs = work.loc[mask].groupby("tau_hw").size().rename("n_obs").reset_index()
        part = part.merge(n_obs, on="tau_hw", how="left")
        part.insert(0, "sample", sample)
        profile_parts.append(part)
    prof = pd.concat(profile_parts, ignore_index=True)

    show = [c for c in ("tmax_anom_dm", "utci_max_dm", "Twb_max_dm",
                        "SCDH24_dm", "CDH_Twb26_dm", "WBD_at_tmax_dm") if c in prof.columns]
    print(f"\n    {'tau':>5} " + " ".join(f"{c.replace('_dm',''):>13}" for c in show))
    for _, r in prof.loc[prof["sample"].eq("strict_clean")].iterrows():
        if int(r.tau_hw) in (-14, -10, -7, -3, -1, 0, 1, 2, 3, 5, 7, 10, 14, 21):
            print(f"    {int(r.tau_hw):>5} " + " ".join(f"{r[c]:13.3f}" for c in show))

    # ---- save ------------------------------------------------------------
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    p1 = OUT_DIR / "city_event_windows.parquet"
    p2 = OUT_DIR / "exposure_by_tau.parquet"
    atomic_parquet(out, p1)
    atomic_parquet(prof, p2)
    overlap.to_csv(TABLE_DIR / "table_10_event_window_overlap.csv", index=False)
    prof.to_csv(TABLE_DIR / "table_11_exposure_by_tau.csv", index=False)
    print(f"\n[5] Saved")
    print(f"    {p1.name:30s} {len(out):,} rows x {out.shape[1]} cols "
          f"({p1.stat().st_size / 1024**2:.1f} MB)")
    print(f"    {p2.name:30s} {len(prof):,} rows")
    print(f"    tables/table_10_event_window_overlap.csv")
    print(f"    tables/table_11_exposure_by_tau.csv")

    print("\n    Estimation form once emissions are joined:")
    print("      ln(E_ist) = a_i + d_c,t + sum_s beta_s * hw_L{s} "
          "+ sum_s gamma_s * cw_L{s} + controls")
    print("      beta_s for s<0 is the parallel-trends check; s=0 the onset")
    print("      jump; s>0 persistence and rebound.")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
