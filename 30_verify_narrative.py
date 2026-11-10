#!/usr/bin/env python
"""
30_verify_narrative.py  --  check that the Results narrative matches the tables
==============================================================================

WHY THIS EXISTS
---------------
Numbers in the prose are typed by hand. Two ways they go wrong were both hit in
this project:

  1. A rerun changes a coefficient in the third significant figure and the text
     is not updated. Nothing errors and the mismatch is invisible on reading.
  2. A cached intermediate silently serves results fitted on a superseded panel
     (this is what happened to T4 after the country-label fix), so the text is
     internally consistent with a table that is itself stale.

Case 2 is why the check reads the CSVs rather than a saved summary, and why it
also asserts the sample sizes: a coefficient that matches but at the wrong n is
the failure mode that is easiest to miss.

USAGE
-----
  python 30_verify_narrative.py
  python 30_verify_narrative.py --root Results          # the 3-year tree
Exit code is 1 if any claim fails, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# claim extraction
# --------------------------------------------------------------------------
def pick(df: pd.DataFrame, **eq) -> pd.Series:
    """Single row matching all equality conditions; raises if not unique."""
    m = pd.Series(True, index=df.index)
    for k, v in eq.items():
        m &= df[k] == v
    hit = df[m]
    if len(hit) != 1:
        raise LookupError(f"{eq} matched {len(hit)} rows, expected 1")
    return hit.iloc[0]


def build_claims(root: Path) -> list[tuple[str, float, float, float]]:
    """(label, value found in tables, value asserted in prose, tolerance)."""
    T = lambda p: pd.read_csv(root / p)  # noqa: E731

    t3 = T("R3/tables/T3_sectoral_response.csv")
    t3 = t3[t3.form == "continuous"]
    t4 = T("R3/tables/T4_exposure_form_comparison.csv")
    t4ab = t4[t4.tier == "AB"]
    t5 = T("SI/tables/TS9_tier_comparison.csv")
    t6 = T("R4/tables/T5_humidity_tests_tierAB.csv")
    t8 = T("R5/tables/T7_sector_comparison_tierAB.csv")
    t9 = T("R5/tables/T8_covid_window_tierAB.csv")
    t11 = T("SI/tables/TS13_weight_race_tierAB.csv")
    t12 = T("SI/tables/TS14_moderators_tierAB.csv")
    ts1 = T("SI/tables/TS1_benchmark_comparison.csv")
    ts2 = T("SI/tables/TS2_chhe_threshold_scan_tierAB.csv")

    c: list[tuple[str, float, float, float]] = []
    add = lambda lab, got, want, tol=0.005: c.append((lab, got, want, tol))  # noqa: E731

    # ---- sample sizes: the check that catches a stale cache ---------------
    add("R3 n (Power)", pick(t3, sector="Power", exposure="heat").n, 40320, 0.5)
    add("R3.2 n (Tier AB)", t4ab.n.iloc[0], 40516, 0.5)
    add("R4/R5 n (Power)", pick(t6, sector="Power", test="T1_same_load",
                                term="scdh").n, 40712, 0.5)
    add("R2 benchmark n", ts1[ts1.tier == "AB"].n.iloc[0], 40908, 0.5)

    # ---- R3.1 -------------------------------------------------------------
    p_h = pick(t3, sector="Power", exposure="heat")
    p_c = pick(t3, sector="Power", exposure="cold")
    add("R3.1 Power heat %", 100 * p_h.cum, 2.96)
    add("R3.1 Power heat t", p_h.cum_t, 3.02)
    add("R3.1 Power heat wcb p", p_h.cum_p_wcb, 0.011, 0.0005)
    add("R3.1 Power heat pre-p", p_h.pretrend_p, 0.181)
    add("R3.1 Power cold %", 100 * p_c.cum, 9.91)
    add("R3.1 Power cold t", p_c.cum_t, 3.74)
    add("R3.1 cold/heat ratio", p_c.cum / p_h.cum, 3.3, 0.05)
    add("R3.1 Residential heat pre-p",
        pick(t3, sector="Residential", exposure="heat").pretrend_p, 0.024)

    # ---- R3.2: the pre-trend column that was dropped once -----------------
    for lab, want_pct, want_t, want_pre in [
        ("Extreme heat day share", -0.68, -1.37, 0.561),
        ("Heatwave share", -0.12, -0.33, 0.265),
        ("Compound humid heat share", 1.15, 4.44, 0.0000),
        ("WBGT > 28 °C share", 1.47, 4.67, 0.311),
        ("UTCI > 32 °C share", 3.02, 3.32, 0.054),
        ("Wet-bulb degree-hours", 0.66, 3.45, 0.0001),
        ("Cooling degree-hours", 2.63, 2.88, 0.202),
    ]:
        r = pick(t4ab, label=lab)
        add(f"R3.2 {lab} %", 100 * r.cum, want_pct)
        add(f"R3.2 {lab} t", r.t, want_t)
        add(f"R3.2 {lab} pre-p", r.pretrend_p, want_pre, 0.0006)

    # the interpretive claim: exactly two constructs are significant AND pass
    sig_pass = t4ab[(t4ab.t.abs() > 1.96) & (t4ab.pretrend_p > 0.10)]
    add("R3.2 # constructs significant and passing pre-trend",
        len(sig_pass), 2, 0.5)

    # ---- R3.3 -------------------------------------------------------------
    add("R3.3 Tier C Power %", 100 * pick(t5, tier="C", sector="Power").cum, 0.62)

    # ---- R4 ---------------------------------------------------------------
    add("R4.1 Power humid-at-load %",
        100 * pick(t6, sector="Power", test="T1b_orth_humidity",
                   term="humid").cum, 0.81)
    add("R4.1 Power humid-at-load t",
        pick(t6, sector="Power", test="T1b_orth_humidity", term="humid").t, 1.67)
    add("R4.2 Power CHHE %",
        100 * pick(t6, sector="Power", test="T2b_absolute_chhe",
                   term="chhe").cum, 1.33)
    add("R4.2 Power CHHE t",
        pick(t6, sector="Power", test="T2b_absolute_chhe", term="chhe").t, 4.80)
    add("R4.2 Aviation CHHE t",
        pick(t6, sector="Aviation", test="T2b_absolute_chhe", term="chhe").t,
        0.08, 0.006)
    add("R4.2 Power CHHE+ t",
        pick(t6, sector="Power", test="T2b_absolute_chhe_plus",
             term="chhe_plus").t, 15.73)
    add("R4.2 Aviation CHHE+ t",
        pick(t6, sector="Aviation", test="T2b_absolute_chhe_plus",
             term="chhe_plus").t, 8.77)
    add("R4.3 Power peak-load humid %",
        100 * pick(t6, sector="Power", test="T1c_peak_load", term="humid").cum,
        -0.37)
    add("R4.4 Power day %",
        100 * pick(t6, sector="Power", test="T3_day_night", term="day").cum, 2.14)
    add("R4.4 Residential night-minus-day %",
        100 * pick(t6, sector="Residential", test="T3_day_night",
                   term="night_minus_day").cum, 11.36)
    add("R4.5 Power humid-minus-dry %",
        100 * pick(t6, sector="Power", test="T2_humid_vs_dry_relative",
                   term="humid_minus_dry").cum, -4.21)

    # ---- R5 ---------------------------------------------------------------
    pw = pick(t8, sector="Power")
    add("R5 Power cooling", pw.heat, 16.72, 0.01)
    add("R5 Power heating", pw.cold, 9.32, 0.01)
    add("R5 Power difference", pw["diff"], 6.21, 0.01)
    add("R5 Power difference z", pw.diff_z, 1.43)
    add("R5 Aviation difference (detection floor)",
        pick(t8, sector="Aviation")["diff"], 1.56, 0.01)
    # T8 now carries all five sectors, so the window alone no longer picks a
    # row. The Power rows are the ones the text quotes.
    t9pw = t9[t9.sector == "Power"] if "sector" in t9.columns else t9
    add("R5 drop-2020 cooling",
        pick(t9pw, window="drop 2020").heat, 20.18, 0.01)
    # T8's first row must carry the window label, not an inherited one
    if str(t9pw.window.iloc[0]) != "2019-2022":
        raise AssertionError(
            f"T8 first row is labelled {t9pw.window.iloc[0]!r}; it holds the "
            "full-panel estimate and must be labelled 2019-2022")
    # The full window must reproduce the headline pooled estimate; if it does
    # not, T8 was written by a different specification than T7.
    if abs(float(pick(t9pw, window="2019-2022")["diff"])
           - float(pw["diff"])) > 0.02:
        raise AssertionError(
            "T8's full window disagrees with T7's pooled Power difference")

    # per-country counts behind 'no universal asymmetry'
    t7 = T("R5/tables/T6_thermal_asymmetry_Power_tierAB.csv")
    b = t7[t7.both]
    add("R5 countries with both sides", t7.both.sum(), 18, 0.5)
    add("R5 heat-side identified", t7.heat_ok.sum(), 20, 0.5)
    add("R5 cooling > heating", (b.diff_eb_lo > 0).sum(), 7, 0.5)
    add("R5 heating > cooling", (b.diff_eb_hi < 0).sum(), 2, 0.5)

    # ---- SI.8: weight invariance, the one result kept from the retired R6 --
    add("SI.8 OOS R2 spread",
        t11.oos_r2_loco.max() - t11.oos_r2_loco.min(), 0.00020, 0.000006)
    add("SI.8 coefficient span across the five schemes",
        t11.pct_per_sd.max() - t11.pct_per_sd.min(), 0.17, 0.006)
    add("SI.8 lowest correlation with population weighting",
        t11.corr_with_pop.min(), 0.9686, 0.00006)
    # The main text says the choice of weight decides nothing, so the spread
    # being small is a rule, not a number to be read off.
    if t11.pct_per_sd.max() - t11.pct_per_sd.min() > 0.5:
        raise AssertionError(
            "SI.8 and the Figure S14 caption say the five weighting schemes "
            "give the same coefficient; they no longer do")
    if t11.t.min() < 2.5 or t11.t.max() > 3.3:
        raise AssertionError(
            "SI.8 says every t lies between 2.87 and 2.96; it no longer does")

    # ---- SI.7: the moderator regression, reported WITH its fragility -------
    add("SI.7 volume per capita slope",
        pick(t12, column="vol_per_cap").slope_per_sd, 18.52, 0.01)
    add("SI.7 volume per capita t", pick(t12, column="vol_per_cap").t, 3.55)
    add("SI.7 surface per capita t", pick(t12, column="surf_per_cap").t, 0.64)
    add("SI.7 AC penetration t", pick(t12, column="ac_ownership_pct").t, 2.20)
    add("SI.7 Aviation negative control t",
        pick(t12, column="vol_per_cap|aviation").t, -0.73)
    # SI.7.1 dismantles that t statistic, and every number it uses to do so is
    # recomputed here rather than quoted, because the whole point of the
    # subsection is that the headline number does not survive them.
    r5p = T("R5/tables/T6_thermal_asymmetry_Power_tierAB.csv")
    r5p = r5p[r5p.heat_ok.fillna(False) & r5p["heat"].notna()]
    sch = T("SI/tables/TS13b_weight_schemes_summary.csv")
    if {"w_built_v", "w_pop"}.issubset(sch.columns):
        # the deduplicated columns; see the note in SI.7.1
        sch = sch.assign(vpc=sch.w_built_v / sch.w_pop)[["country", "vpc"]]
        m = r5p.merge(sch, on="country", how="inner")
        yv = 100 * m["heat"].to_numpy(float)
        sev = 100 * m["heat_se"].to_numpy(float)
        xv = (m.vpc.to_numpy(float) - m.vpc.mean()) / m.vpc.std()
        wt = 1.0 / np.maximum(sev ** 2, 1e-9)

        def _wls(w_):
            Xd = np.column_stack([np.ones(len(xv)), xv])
            XtWX = Xd.T @ (Xd * w_[:, None])
            bb = np.linalg.solve(XtWX, Xd.T @ (w_ * yv))
            rr = yv - Xd @ bb
            s2 = float((w_ * rr ** 2).sum() / (len(xv) - 2))
            return bb[1], bb[1] / np.sqrt(s2 * np.linalg.inv(XtWX)[1, 1])

        add("SI.7 countries in the moderator regression", len(m), 20, 0.5)
        add("SI.7 unweighted t", _wls(np.ones(len(xv)))[1], 1.04, 0.06)
        add("SI.7 raw Pearson correlation",
            float(np.corrcoef(xv, yv)[0, 1]), 0.239, 0.006)
        add("SI.7 effective sample size of the precision weights",
            wt.sum() ** 2 / (wt ** 2).sum(), 6.2, 0.06)
        add("SI.7 largest single weight share (%)",
            100 * wt.max() / wt.sum(), 33.7, 0.06)
        # Dropping a country removes its row from X, y and w alike, so the
        # leave-one-out fit is rebuilt rather than reweighted.
        loo = []
        for i in range(len(m)):
            k = [j for j in range(len(m)) if j != i]
            Xd = np.column_stack([np.ones(len(k)), xv[k]])
            w_ = wt[k]
            XtWX = Xd.T @ (Xd * w_[:, None])
            bb = np.linalg.solve(XtWX, Xd.T @ (w_ * yv[k]))
            rr = yv[k] - Xd @ bb
            s2 = float((w_ * rr ** 2).sum() / (len(k) - 2))
            loo.append(bb[1] / np.sqrt(s2 * np.linalg.inv(XtWX)[1, 1]))
        add("SI.7 minimum leave-one-out t", float(np.min(loo)), 1.62, 0.06)
        if float(np.min(loo)) >= 1.96:
            raise AssertionError(
                "SI.7 says the moderator loses significance when one country "
                "is dropped; it no longer does, and the section needs "
                "rewriting rather than the number updating")

    # ---- SI.7.2: the tilt intervals ---------------------------------------
    tilt = T("SI/tables/TS16_built_tilt_response_tierAB.csv")
    tilt = tilt[tilt.spec.astype(str).str.startswith("solo_")]
    for tag, want_lo, want_hi in [("bv", -10.19, 5.90), ("cp4", -0.67, 0.77)]:
        r = tilt[(tilt.sector == "Power") & (tilt.spec == f"solo_{tag}")]
        if not len(r):
            continue
        r = r.iloc[0]
        beta = float(r["X_cool_cum"])
        g, se_ = float(r[f"D{tag}_cool_cum"]), float(r[f"D{tag}_cool_se"])
        lo, hi = (g - 1.96 * se_) / beta, (g + 1.96 * se_) / beta
        if beta < 0:
            lo, hi = hi, lo
        add(f"SI.7 theta low, {tag}", lo, want_lo, 0.06)
        add(f"SI.7 theta high, {tag}", hi, want_hi, 0.06)
    # The section's structure rests on scale being untestable and form being
    # testable. The right criterion is whether the interval EXCLUDES theta = 1,
    # not how wide it is: a wide interval centred far from zero could exclude 1
    # and a narrow one centred on 1 could fail to.
    def _excludes_one(tag):
        r = tilt[(tilt.sector == "Power") & (tilt.spec == f"solo_{tag}")]
        if not len(r):
            return None
        r = r.iloc[0]
        beta = float(r["X_cool_cum"])
        g, se_ = float(r[f"D{tag}_cool_cum"]), float(r[f"D{tag}_cool_se"])
        lo, hi = (g - 1.96 * se_) / beta, (g + 1.96 * se_) / beta
        if beta < 0:
            lo, hi = hi, lo
        return not (lo <= 1.0 <= hi)
    if _excludes_one("bv") is not False:
        raise AssertionError(
            "SI.7.2 says built volume cannot reject theta = 1; its interval "
            "now does, and the section's claim that scale is untestable would "
            "have to be rewritten")
    if _excludes_one("cp4") is not True:
        raise AssertionError(
            "SI.7.2 says the sharpened LCZ weight rejects theta = 1; its "
            "interval no longer excludes it")

    # ---- R1.3: the Figure 1 contrast ---------------------------------------
    # Fig 5c: the class means behind the 11.5 K inversion. These were computed
    # inside the plotting function and nowhere else until T4d was added, so
    # nothing backed the four numbers the Results and the caption quote.
    t4d_p = root / "SI" / "tables" / "T4d_definition_class_means.csv"
    if t4d_p.exists():
        t4d = pd.read_csv(t4d_p).set_index("definition")
        add("V2 relative humid mean Tmax",
            t4d.loc["relative humid (HEH)", "mean_tmax_c"], 19.43, 0.006)
        add("V2 relative dry mean Tmax",
            t4d.loc["relative dry (DEH)", "mean_tmax_c"], 30.96, 0.006)
        add("V2 CHHE mean Tmax",
            t4d.loc["absolute humid (CHHE)", "mean_tmax_c"], 35.16, 0.006)
        add("V2 CHHE mean Twb",
            t4d.loc["absolute humid (CHHE)", "mean_twb_max_c"], 27.23, 0.006)
        add("V2 humid-minus-dry Tmax inversion (K)",
            t4d.loc["relative dry (DEH)", "mean_tmax_c"]
            - t4d.loc["relative humid (HEH)", "mean_tmax_c"], 11.53, 0.006)
        # HEH and DEH are a tercile split, so their day counts must be equal
        if (t4d.loc["relative humid (HEH)", "n_city_days"]
                != t4d.loc["relative dry (DEH)", "n_city_days"]):
            raise AssertionError(
                "HEH and DEH are p33/p67 terciles of the same set and must "
                "have equal day counts; they no longer do")

    t1b = T("R1/tables/T1b_metric_geography.csv")
    rel = t1b[t1b.criterion == "relative"]
    ab = t1b[t1b.criterion != "relative"]
    add("R1.3 relative panels", len(rel), 3, 0.5)
    add("R1.3 absolute/continuous panels", len(ab), 6, 0.5)
    add("R1.3 max |r| among relative", rel.r_abs_lat.abs().max(), 0.15)
    add("R1.3 UTCI > 32 C r", pick(t1b, variable="n_utci_strong_days").r_abs_lat,
        -0.71)
    add("R1.3 UTCI < 0 C r",
        pick(t1b, variable="n_utci_coldstress_days").r_abs_lat, 0.88)
    add("R1.3 CHHE cities", pick(t1b, variable="n_CHHE_days").n_cities_nonzero,
        630, 0.5)
    # Each column of rows 2-3 is the threshold and the continuous form of one
    # physical quantity; the text quotes all three correlations.
    summ0 = pd.read_parquet(BASE / "outputs" / "_cache" /
                            "city_constructs.parquet")
    for a, b, want in [("n_utci_strong_days", "cum_scdh24", 0.94),
                       ("n_CHHE_days", "cum_cdh_twb26", 0.98),
                       ("n_utci_coldstress_days", "cum_hdh15", 0.92)]:
        add(f"R1.3 rho({a}, {b})",
            summ0[[a, b]].corr(method="spearman").iloc[0, 1], want)
    # The claim "relative and absolute are near-orthogonal across cities" is the
    # load-bearing sentence of R1.3, so it is recomputed rather than quoted.
    cross = summ0[list(rel.variable) + list(ab.variable)].corr(
        method="spearman").loc[list(rel.variable), list(ab.variable)].abs()
    add("R1.3 max |cross-family Spearman|", cross.max().max(), 0.20)
    add("R1.3 median |cross-family Spearman|", cross.stack().median(), 0.09)

    # ---- R1.2 / R1.3 / R1.7: the Figure 2 panels ---------------------------
    t1e = T("R1/tables/T1e_event_duration_survival.csv")
    t1d = T("R1/tables/T1d_criteria_conditional_probability.csv")
    t1c = T("R1/tables/T1c_country_exposure_profile.csv")

    def sv(ev, k):
        return pick(t1e, event=ev, duration_days=k).p_at_least

    for k, wh, wc in [(5, 0.323, 0.191), (7, 0.110, 0.056),
                      (10, 0.024, 0.009), (14, 0.005, 0.001)]:
        add(f"R1.2 P(heat >= {k})", sv("Heatwaves", k), wh, 0.0006)
        add(f"R1.2 P(cold >= {k})", sv("Cold waves", k), wc, 0.0006)
    add("R1.2 heat/cold persistence ratio at D=7",
        sv("Heatwaves", 7) / sv("Cold waves", 7), 1.97, 0.01)
    add("R1.2 heat events", pick(t1e, event="Heatwaves", duration_days=5)
        .n_events, 36795, 0.5)
    add("R1.2 cold events", pick(t1e, event="Cold waves", duration_days=5)
        .n_events, 14775, 0.5)
    add("R1.2 mean heat duration", pick(t1e, event="Heatwaves",
                                        duration_days=5).mean_duration, 4.35)
    add("R1.2 mean cold duration", pick(t1e, event="Cold waves",
                                        duration_days=5).mean_duration, 3.86)

    def cp(then, given):
        return pick(t1d, **{"then": then, "given": given}).p_row_given_col

    add("R1.3 P(UTCI>32 | EHD)", cp("utci_strong", "EHD"), 0.411)
    add("R1.3 P(EHD | UTCI>32)", cp("EHD", "utci_strong"), 0.240)
    add("R1.3 P(WBGT>28 | EHD)", cp("twbg_danger", "EHD"), 0.128)
    add("R1.3 P(EHD | WBGT>28)", cp("EHD", "twbg_danger"), 0.247)
    add("R1.3 P(UTCI>32 | CHHE)", cp("utci_strong", "CHHE"), 0.997, 0.0006)
    add("R1.3 P(WBGT>28 | CHHE)", cp("twbg_danger", "CHHE"), 0.997, 0.0006)
    add("R1.3 P(UTCI>32 | WBGT>28)", cp("utci_strong", "twbg_danger"), 0.995,
        0.0006)
    add("R1.3 P(WBGT>28 | UTCI>32)", cp("twbg_danger", "utci_strong"), 0.300)
    add("R1.3 P(CHHE | HEH)", cp("CHHE", "HEH"), 0.072)
    add("R1.3 P(HEH | CHHE)", cp("HEH", "CHHE"), 0.369)
    add("R1.3 criteria in Fig 2b", int(t1d["then"].nunique()), 13, 0.5)
    add("R1.3 cells in Fig 2b", len(t1d), 169, 0.5)
    add("R1.3 definitional cells", int(t1d.definitional.sum()), 13, 0.5)
    # heat and cold are almost disjoint at day level
    add("R1.3 P(ECD | EHD)", cp("ECD", "EHD"), 0.006, 0.0006)
    add("R1.3 P(EHD | ECD)", cp("EHD", "ECD"), 0.013, 0.0006)
    add("R1.3 P(CW | HW)", cp("CW", "HW"), 0.001, 0.0006)
    # compound humid heat is a no-recovery event
    add("R1.3 P(hot night | CHHE)", cp("HN", "CHHE"), 0.643)
    add("R1.3 P(hot night | CHHE+)", cp("HN", "CHHE_plus"), 0.702)
    add("R1.3 P(CHHE | CDNH)", cp("CHHE", "CDNH"), 0.098)
    add("R1.3 P(CHHE | EHD)", cp("CHHE", "EHD"), 0.064)
    add("R1.3 P(CHHE+ | CHHE)", cp("CHHE_plus", "CHHE"), 0.213)
    add("R1.3 hot-night base rate",
        pick(t1d, **{"then": "HN", "given": "HN"}).base_rate_row, 0.123, 0.0006)

    # ---- SI.5: the full 21-code matrix -------------------------------------
    ts8 = T("SI/tables/TS8_criteria_conditional_probability.csv")
    add("SI.5 codes", int(ts8["then"].nunique()), 21, 0.5)
    add("SI.5 cells", len(ts8), 441, 0.5)
    add("SI.5 definitional cells", int(ts8.definitional.sum()), 19, 0.5)
    off8 = ts8[ts8["then"] != ts8["given"]]
    add("SI.5 off-diagonal cells", len(off8), 420, 0.5)
    add("SI.5 off-diagonal cells below 0.005",
        int((off8.p_row_given_col < 0.005).sum()), 140, 0.5)
    hc = off8[off8.then_family.isin(["heat", "humid"])
              & (off8.given_family == "cold")]
    add("SI.5 heat/humid given relative-cold pairs", len(hc), 24, 0.5)
    add("SI.5 heat/humid given relative-cold, max", hc.p_row_given_col.max(),
        0.013, 0.0006)
    add("SI.5 heat/humid given relative-cold, median",
        hc.p_row_given_col.median(), 0.00004, 0.00002)

    def cp8(then, given):
        return pick(ts8, **{"then": then, "given": given}).p_row_given_col

    add("SI.5 P(UTCI>26 | DEH)", cp8("utci_moderate", "DEH"), 0.888)
    add("SI.5 P(UTCI>26 | HEH)", cp8("utci_moderate", "HEH"), 0.349)
    # twb_severe must stay out: conditioning on ~2 city-days is meaningless
    if "twb_severe" in set(ts8["then"]):
        raise AssertionError(
            "twb_severe is in the matrix; it fires on 0.0001 % of city-days "
            "and every probability conditioned on it rests on a handful of "
            "observations")
    # The Jaccard ceiling quoted as the reason for switching statistic.
    add("R1.3 Jaccard ceiling, WBGT>28 within EHD",
        pick(t1d, **{"then": "twbg_danger",
                     "given": "twbg_danger"}).base_rate_row
        / pick(t1d, **{"then": "EHD", "given": "EHD"}).base_rate_row,
        0.52, 0.006)

    add("R1.7 countries in the profile", len(t1c), 46, 0.5)
    add("R1.7 Spearman(HW, cooling degree-hours), 46",
        t1c[["HW", "SCDH24"]].corr(method="spearman").iloc[0, 1], -0.098)
    ab7 = t1c[t1c.tier.isin(["A", "B"])]
    add("R1.7 Spearman(HW, cooling degree-hours), Tier A+B",
        ab7[["HW", "SCDH24"]].corr(method="spearman").iloc[0, 1], -0.266)
    add("R1.7 cooling degree-hour range factor",
        t1c.SCDH24.max() / t1c.SCDH24.min(), 839, 1.0)
    add("R1.7 heatwave-day range factor", t1c.HW.max() / t1c.HW.min(), 54, 0.5)
    ci = t1c.set_index("country")
    # `c` is the claims list in this scope; shadowing it here raised
    # AttributeError deep inside add(). Country loop variables are `cn`.
    for cn, hw, scdh in [("France", 33.7, 2229), ("United Kingdom", 27.5, 229),
                         ("Germany", 25.8, 1032), ("United States", 23.9, 8307),
                         ("China", 25.8, 9114), ("India", 21.7, 26505)]:
        add(f"R1.7 {cn} heatwave days", ci.HW[cn], hw, 0.06)
        add(f"R1.7 {cn} cooling degree-hours", ci.SCDH24[cn], scdh, 1.0)
    add("R1.7 France/India heatwave days", ci.HW["France"] / ci.HW["India"],
        1.55, 0.006)
    add("R1.7 India/France cooling degree-hours",
        ci.SCDH24["India"] / ci.SCDH24["France"], 11.9, 0.06)
    add("R1.7 India/UK cooling degree-hours",
        ci.SCDH24["India"] / ci.SCDH24["United Kingdom"], 116, 0.6)
    add("R1.7 countries with a night share", int(t1c.ns_cool.notna().sum()),
        39, 0.5)
    add("R1.7 night share min", t1c.ns_cool.min(), 6.2, 0.06)
    add("R1.7 night share max", t1c.ns_cool.max(), 39.7, 0.06)
    for cn, w in [("India", 29.8), ("China", 29.1), ("United States", 25.1),
                  ("Germany", 19.1), ("France", 18.1)]:
        add(f"R1.7 night share {cn}", ci.ns_cool[cn], w, 0.06)
    add("R1.7 countries with a wet-bulb night share",
        int(t1c.ns_wb.notna().sum()), 13, 0.5)

    # ---- SI.4: why the country profile is supplementary ---------------------
    # The reason given for demoting the heatmap is quantitative, so it is
    # checked. If a rerun made the panel informative, this would fail and the
    # SI text would need rewriting rather than quietly staying wrong.
    ts7 = T("SI/tables/TS7_country_profile_structure.csv")
    add("SI.4 constructs tracking the row order at |rho| > 0.6",
        int((ts7.rho_with_row_order.abs() > 0.6).sum()), 6, 0.5)
    add("SI.4 PC1 share", 100 * ts7.pc1_share.iloc[0], 54, 0.5)
    add("SI.4 PC1+PC2 share",
        100 * (ts7.pc1_share.iloc[0] + ts7.pc2_share.iloc[0]), 77, 0.5)
    for var, want in [("SCDH24", 1.00), ("utci_strong", 0.93),
                      ("twbg_danger", 0.87), ("CDH_Twb26", 0.80),
                      ("HDH15", -0.80), ("CHHE", 0.79),
                      ("HW", -0.10), ("EHD", -0.03), ("CDNH", 0.03)]:
        add(f"SI.4 rho with row order, {var}",
            pick(ts7, construct=var).rho_with_row_order, want)

    # ---- the base-temperature scan (unit dependence of the asymmetry) ------
    scan = T("R5/tables/T8b_base_scan_tierAB.csv")
    add("scan rows", len(scan), 270, 0.5)
    add("scan sectors", scan.sector.nunique(), 5, 0.5)
    add("scan base pairs per cell",
        len(scan[(scan.sector == "Power") &
                 (scan.normalisation == "per100")]), 18, 0.5)
    pw_scan = scan[scan.sector == "Power"]
    for nm, lo, hi, neg, pos in [("per100", -6.66, 17.32, 2, 2),
                                 ("perSD", -11.41, -9.13, 18, 0),
                                 ("perP95", -30.52, -24.32, 18, 0)]:
        g = pw_scan[pw_scan.normalisation == nm]
        add(f"scan Power {nm} min", g.difference.min(), lo, 0.006)
        add(f"scan Power {nm} max", g.difference.max(), hi, 0.006)
        add(f"scan Power {nm} significantly negative",
            int((g.diff_z < -1.96).sum()), neg, 0.5)
        add(f"scan Power {nm} significantly positive",
            int((g.diff_z > 1.96).sum()), pos, 0.5)
    add("scan Power perSD heating/cooling ratio, median",
        (pw_scan[pw_scan.normalisation == "perSD"].heating
         / pw_scan[pw_scan.normalisation == "perSD"].cooling).median(),
        3.22, 0.006)
    # The load-bearing sentence: the side coefficients are unit-invariant and
    # only their difference moves. If a rerun broke that, every word of the
    # section would be wrong, so it is asserted rather than described.
    pub = pw_scan[pw_scan.published_pair]
    if pub.cooling_z.round(3).nunique() != 1 or pub.heating_z.round(3).nunique() != 1:
        raise AssertionError(
            "the cooling and heating z statistics differ across units; "
            "rescaling a regressor cannot change its t, so something is wrong")
    add("scan published-pair cooling z", pub.cooling_z.iloc[0], 3.525)
    add("scan published-pair heating z", pub.heating_z.iloc[0], 5.711)

    # ---- heterogeneity, covariance-consistent -----------------------------
    # These are the per-100 K.h values quoted in the R5 prose. They moved when
    # Cochran's Q was corrected to its GLS form: the country slopes come from
    # one joint regression and are not independent, so the earlier statistic
    # was computed under an assumption the estimator explicitly rejects.
    for sec, want_i2, want_k in [("Power", 92.74, 18), ("Industry", 91.37, 18)]:
        r = pick(t8, sector=sec)
        add(f"heterogeneity {sec} difference I2", r.diff_I2, want_i2, 0.06)
        add(f"heterogeneity {sec} difference k", r.diff_k, want_k, 0.5)
        if not (r.diff_Q_p < 1e-10):
            raise AssertionError(
                f"{sec}: Cochran Q no longer rejects a shared slope "
                f"(p = {r.diff_Q_p}); R5's headline claim would need rewriting")
    add("heterogeneity Power cooling I2", pick(t8, sector="Power").heat_I2,
        95.5, 0.06)
    add("heterogeneity Power heating I2", pick(t8, sector="Power").cold_I2,
        93.62, 0.06)

    # ---- the same asymmetry under the two scale-free units ----------------
    for nm, want in [("perSD", -10.59), ("perP95", -27.62)]:
        tu = T(f"R5/tables/T7_sector_comparison_tierAB_{nm}.csv")
        add(f"R5 Power difference, {nm}", pick(tu, sector="Power")["diff"],
            want, 0.01)
        add(f"R5 Aviation difference, {nm}",
            pick(tu, sector="Aviation")["diff"],
            1.58 if nm == "perSD" else 4.53, 0.01)

    # ---- resolution and humidity encompassing tests ------------------------
    t4c = T("R3/tables/T4c_resolution_and_humidity_tierAB.csv")

    def enc(spec, term):
        return pick(t4c, specification=spec, term=term)

    add("R3.4 national temperature alone",
        enc("national temperature alone", "nat").cum_pct, 3.43)
    add("R3.4 city temperature alone",
        enc("city temperature alone", "city").cum_pct, 3.09)
    add("R3.4 matched, national", enc("national pop-weighted alone",
                                      "nat").cum_pct, 3.43)
    add("R3.4 matched, FUA", enc("FUA pop-weighted alone", "fua").cum_pct, 3.32)
    add("R3.4 matched, FUA equal-weighted",
        enc("FUA equal-weighted alone", "fuae").cum_pct, 3.54)
    add("R3.4 r(national, FUA) matched",
        enc("within-country correlation", "nat_pop_cdd vs fua_pop_cdd").t, 0.990)
    add("R3.4 r(national, city) unmatched",
        enc("within-country correlation", "nat_pop_scdh24 vs w_SCDH24").t, 0.943)

    # ---- SI.6: why the response is estimated at country scale --------------
    ts10 = T("SI/tables/TS10_city_panel_rank.csv")
    ts11 = T("SI/tables/TS11_city_share_dynamics.csv")
    ts12 = T("SI/tables/TS12_city_month_response.csv")
    add("SI.6 rank cells", len(ts10), 21, 0.5)
    add("SI.6 cells rank one to machine precision",
        int((ts10.sv2_over_sv1 < 1e-10).sum()), 16, 0.5)
    add("SI.6 largest departure from rank one", ts10.sv2_over_sv1.max(),
        0.0103, 0.0002)
    add("SI.6 median lag-1 autocorrelation", ts11.lag1_autocorr.median(), 0.907)
    add("SI.6 median lag-12 autocorrelation", ts11.lag12_autocorr.median(),
        0.001, 0.002)
    add("SI.6 median linear-trend R2", ts11.linear_trend_r2.median(), 0.479)
    add("SI.6 month-of-year, Power and Transport",
        ts11[ts11.sector != "Residential"].month_of_year_pct.median(), 1.1, 0.06)
    add("SI.6 month-of-year, Residential",
        ts11[ts11.sector == "Residential"].month_of_year_pct.median(), 13.1, 0.06)
    cool = pick(ts12, term="cooling degree-hours")
    heat = pick(ts12, term="heating degree-hours")
    add("SI.6 city-month cooling t", cool.t, 1.01)
    add("SI.6 city-month heating t", heat.t, -1.13)
    add("SI.6 city-month cooling exposure variance surviving",
        cool.exposure_var_surviving_pct, 22.3, 0.06)
    add("SI.6 city-months", cool.n_city_months, 38133, 0.5)
    add("SI.6 cities", cool.n_cities, 854, 0.5)
    # The null only means something if the regressor still varies.
    if cool.exposure_var_surviving_pct < 10:
        raise AssertionError(
            "less than a tenth of the cooling exposure variance survives the "
            "fixed effects; the city-scale null would then be a power problem, "
            "not a statement about the data")

    # ---- SI.0: the map atlas ----------------------------------------------
    ts6 = T("SI/tables/TS6_map_atlas_geography.csv")
    add("SI.0 atlas panels", len(ts6), 36, 0.5)
    add("SI.0 atlas plates", ts6.figure.nunique(), 4, 0.5)
    for lab, var, want in [
            ("hot nights", "n_HN_days", -0.48),
            ("cold days", "n_CD_days", -0.45),
            ("extreme heat days", "n_EHD_days", -0.09),
            ("heatwave days", "n_HW_days", 0.05),
            ("compound day-night", "n_CDNH_days", 0.13),
            ("peak heatwave excess", "hw_peak_excess", 0.71),
            ("cumulative heatwave excess", "hw_cum_excess", 0.53),
            ("HWMId max", "hw_hwmid_max", 0.10),
            ("peak standardised anomaly", "max_tmax_z", 0.37),
            ("UTCI > 26 C", "n_utci_moderate_days", -0.87),
            ("UTCI > 38 C", "n_utci_verystrong_days", -0.48),
            ("WBGT > 32 C", "n_twbg_highrisk_days", -0.14)]:
        add(f"SI.0 r(|lat|) {lab}", pick(ts6, variable=var).r_abs_lat, want)
    add("SI.0 WBGT > 32 C cities", pick(ts6, variable="n_twbg_highrisk_days")
        .n_cities_nonzero, 205, 0.5)
    add("SI.0 WBGT > 32 C CV",
        pick(ts6, variable="n_twbg_highrisk_days").cv, 5.48)
    add("SI.0 Twb > 28 C cities",
        pick(ts6, variable="n_twb_stress_days").n_cities_nonzero, 285, 0.5)

    # SI.0.4 claims the latitude correlation weakens monotonically as the UTCI
    # threshold is raised. Asserted as an ordering, so a rerun that breaks it
    # fails here instead of leaving the sentence quietly wrong.
    utci = [abs(pick(ts6, variable=v).r_abs_lat) for v in
            ("n_utci_moderate_days", "n_utci_strong_days",
             "n_utci_verystrong_days")]
    if not (utci[0] > utci[1] > utci[2]):
        raise AssertionError(
            f"SI.0.4 says |r| falls as the UTCI threshold rises; got {utci}")

    # ---- figure captions ---------------------------------------------------
    # The captions restate numbers from both tables and add a few of their own.
    # They are checked here because a caption is the last thing anyone rereads
    # after a rerun and the easiest place for a superseded value to survive --
    # the median cold-wave onset in the S7 caption was a three-year value.
    cc = summ0
    add("caption S4/S7 cities with no cold wave",
        int(cc.cw_first_onset_doy.isna().sum()), 20, 0.5)
    add("caption S7 southern-hemisphere cities", int((cc.lat < 0).sum()), 92, 0.5)
    add("caption S7 median first cold-wave onset (DOY)",
        cc.cw_first_onset_doy.median(), 108, 0.5)
    add("caption S4 longest heatwave r", pick(ts6, variable="hw_max_duration")
        .r_abs_lat, -0.04)
    add("caption S4 dry extreme heat r",
        pick(ts6, variable="n_DEH_days").r_abs_lat, -0.09)
    add("caption S5 UTCI > 26 C CV",
        pick(ts6, variable="n_utci_moderate_days").cv, 0.65)
    add("caption S5 UTCI > 38 C CV",
        pick(ts6, variable="n_utci_verystrong_days").cv, 1.85)
    add("caption S5 UTCI > 38 C cities",
        pick(ts6, variable="n_utci_verystrong_days").n_cities_nonzero,
        1199, 0.5)
    add("caption S6 cooling deg-h base 18 r",
        pick(ts6, variable="cum_scdh18").r_abs_lat, -0.73)
    add("caption S6 night-only cooling r",
        pick(ts6, variable="cum_scdh24_night").r_abs_lat, -0.54)
    add("caption S6 latent degree-hours r",
        pick(ts6, variable="cum_ldh_w").r_abs_lat, -0.66)
    add("caption S6 enthalpy degree-hours r",
        pick(ts6, variable="cum_edh_h").r_abs_lat, -0.64)
    add("caption S6 wet-bulb degree-hours CV",
        pick(ts6, variable="cum_cdh_twb26").cv, 2.97)
    add("caption S6 wet-bulb degree-hours cities",
        pick(ts6, variable="cum_cdh_twb26").n_cities_nonzero, 649, 0.5)
    add("caption S7 long heatwaves r",
        pick(ts6, variable="n_hw_long_events").r_abs_lat, 0.11)

    # ---- captions for Figures 3-7 and S10 ---------------------------------
    # Every number a reader can see on one of these figures is asserted here.
    # A figure caption is prose like any other, and the numbers in it are the
    # ones most likely to go stale, because they are typed once and then the
    # figure is regenerated many times.
    c3 = t3[t3.form == "continuous"]
    for lab, sec, exp, cum, tt, pv in [
            ("Power heat", "Power", "heat", 2.96, 3.02, 0.180),
            ("Power cold", "Power", "cold", 9.91, 3.74, 0.343),
            ("Industry heat", "Industry", "heat", 2.74, 2.89, 0.213),
            ("Industry cold", "Industry", "cold", 9.89, 3.65, 0.530),
            ("Ground Transport heat", "Ground Transport", "heat",
             -0.21, -0.91, 0.277),
            ("Aviation heat", "Aviation", "heat", -0.08, -0.38, 0.303),
            ("Aviation cold", "Aviation", "cold", -1.29, -2.66, 0.573),
            ("Residential heat", "Residential", "heat", -0.94, -1.01, 0.024),
    ]:
        r = pick(c3, sector=sec, exposure=exp)
        add(f"caption 3 {lab} cum", 100 * r.cum, cum, 0.006)
        add(f"caption 3 {lab} t", r.cum_t, tt, 0.006)
        add(f"caption 3 {lab} pre-trend p", r.pretrend_p, pv, 0.0006)
    add("caption 3 heating/cooling ratio in Power",
        pick(c3, sector="Power", exposure="cold").cum
        / pick(c3, sector="Power", exposure="heat").cum, 3.35, 0.006)
    add("caption 3 countries", c3.n_country.max(), 28, 0.5)
    add("caption 3 country-days",
        pick(c3, sector="Power", exposure="heat").n, 40320, 0.5)
    add("caption 3 Aviation country-days",
        pick(c3, sector="Aviation", exposure="heat").n, 39589, 0.5)

    t4ab = t4[t4.tier == "AB"]
    for lab, cum, tt, pv in [
            ("extreme heat day share", -0.68, -1.37, 0.561),
            ("heatwave share", -0.12, -0.33, 0.265),
            ("WBGT > 28 C share", 1.47, 4.67, 0.311),
            ("UTCI > 32 C share", 3.02, 3.32, 0.054),
            ("compound humid heat share", 1.15, 4.44, 0.0005),
            ("cooling degree-hours", 2.63, 2.88, 0.202),
            ("wet-bulb degree-hours", 0.66, 3.45, 0.0005),
    ]:
        key = {"extreme heat day share": "Extreme heat day share",
               "heatwave share": "Heatwave share",
               "WBGT > 28 C share": "WBGT > 28 °C share",
               "UTCI > 32 C share": "UTCI > 32 °C share",
               "compound humid heat share": "Compound humid heat share",
               "cooling degree-hours": "Cooling degree-hours",
               "wet-bulb degree-hours": "Wet-bulb degree-hours"}[lab]
        r = pick(t4ab, label=key)
        add(f"caption 4 {lab} cum", 100 * r.cum, cum, 0.006)
        add(f"caption 4 {lab} t", r.t, tt, 0.006)
        if pv > 0.001:
            add(f"caption 4 {lab} pre-trend p", r.pretrend_p, pv, 0.0006)
        elif r.pretrend_p >= 0.001:
            raise AssertionError(
                f"the caption says {lab} fails the pre-trend test at p < 0.001 "
                f"but the table gives {r.pretrend_p:.4f}")
    # The Tier C claim in the caption is a rule, not a number.
    if t4[t4.tier == "C"].t.abs().max() >= 1.1:
        raise AssertionError(
            "the Figure 4 caption says every Tier C metric has |t| below 1.1; "
            f"the largest is {t4[t4.tier == 'C'].t.abs().max():.2f}")
    # Figure 4b. Recomputed here from the exposure panel rather than read from
    # a table, because the panel is drawn straight from the parquet. The
    # earlier version of this panel filtered on w_HW > 0.5 and a minimum count,
    # which is a filter on spatial synchrony: it left the United States with 2
    # qualifying days and Australia with 3, so the figure described small
    # compact countries. The exposure-weighted mean asserted here uses every
    # country-day and every country, and that coverage is itself asserted.
    exp = pd.read_parquet(BASE / "outputs" / "exposure"
                          / "country_daily_exposure.parquet")
    ab = exp[exp.tier.isin(["A", "B"])]
    hwl = (ab.groupby("country")
             .apply(lambda d: np.average(d.w_SCDH24, weights=d.w_HW)
                    if d.w_HW.sum() > 0 else np.nan)
             .dropna().sort_values())
    add("caption 4b countries", len(hwl), 28, 0.5)
    add("caption 4b countries in Tier A+B", ab.country.nunique(), 28, 0.5)
    add("caption 4b Greece", hwl.max(), 45.4, 0.06)
    add("caption 4b Norway", hwl.min(), 1.8, 0.06)
    add("caption 4b fold range", hwl.max() / hwl.min(), 26, 0.5)
    if hwl.idxmax() != "Greece" or hwl.idxmin() != "Norway":
        raise AssertionError(
            "the Figure 4b caption names Greece and Norway as the extremes; "
            f"the panel now has {hwl.idxmax()} and {hwl.idxmin()}")

    # ---- Figure 5a: the latent/sensible split, the figure's spine --------
    LS = "T5_latent_sensible"
    for lab, sec, term, cum, tt in [
            ("Power sensible", "Power", "sensible", 2.42, 2.18),
            ("Power latent", "Power", "latent", 1.80, 2.24),
            ("Industry sensible", "Industry", "sensible", 2.43, 2.20),
            ("Industry latent", "Industry", "latent", 1.50, 1.95),
            ("Aviation latent", "Aviation", "latent", 0.06, 0.23),
            ("Ground Transport latent", "Ground Transport", "latent",
             0.25, 1.57),
    ]:
        r = pick(t6, sector=sec, test=LS, term=term)
        add(f"caption 5a {lab} cum", 100 * r.cum, cum, 0.006)
        add(f"caption 5a {lab} t", r.t, tt, 0.006)
    add("caption 5a latent as a share of sensible",
        pick(t6, sector="Power", test=LS, term="latent").cum
        / pick(t6, sector="Power", test=LS, term="sensible").cum, 0.74, 0.006)
    # The panel's claim is that neither term is a common factor. That is a
    # rule about the two no-cooling sectors, not a number.
    for sec in ("Aviation", "Ground Transport"):
        for term in ("latent", "sensible"):
            if abs(float(pick(t6, sector=sec, test=LS, term=term).t)) >= 1.96:
                raise AssertionError(
                    f"the Figure 5a caption says the negative controls are "
                    f"flat on both terms; {sec} {term} is not")

    # ---- Figure 5d: the two windows, and the two excluded terms ----------
    t6_3y = pd.read_csv(BASE / "Results" / "R4" / "tables"
                        / "T5_humidity_tests_tierAB.csv")
    for lab, test, term, t3, t4 in [
            ("orthogonalised humidity", "T1b_orth_humidity", "humid",
             2.24, 1.67),
            ("latent", LS, "latent", 2.87, 2.24),
            ("sensible", LS, "sensible", 1.77, 2.18),
            ("enthalpy", "T5b_enthalpy", "enthalpy", 2.43, 1.80),
    ]:
        add(f"caption 5d {lab} t, 2019-2021",
            pick(t6_3y, sector="Power", test=test, term=term).t, t3, 0.006)
        add(f"caption 5d {lab} t, 2019-2022",
            pick(t6, sector="Power", test=test, term=term).t, t4, 0.006)
    add("caption 5d orthogonalised humidity cum, 2019-2021",
        100 * pick(t6_3y, sector="Power", test="T1b_orth_humidity",
                   term="humid").cum, 1.24, 0.006)
    add("caption 5d orthogonalised humidity cum, 2019-2022",
        100 * pick(t6, sector="Power", test="T1b_orth_humidity",
                   term="humid").cum, 0.81, 0.006)
    # The two exclusions are the load-bearing editorial decisions of the
    # figure, so both are asserted as rules.
    if pick(t4ab, label="Compound humid heat share").pretrend_p >= 0.001:
        raise AssertionError(
            "Figure 5d marks humid heat at Twb >= 26 C as failing the "
            "pre-trend test at p < 0.001; the table no longer agrees")
    add("caption 5d CHHE+ Aviation t",
        pick(t6, sector="Aviation", test="T2b_absolute_chhe_plus",
             term="chhe_plus").t, 8.77, 0.006)
    add("caption 5d CHHE+ Ground Transport t",
        pick(t6, sector="Ground Transport", test="T2b_absolute_chhe_plus",
             term="chhe_plus").t, 2.86, 0.006)
    add("caption 5d CHHE+ Power t",
        pick(t6, sector="Power", test="T2b_absolute_chhe_plus",
             term="chhe_plus").t, 15.73, 0.006)

    # The day/night numbers moved to the supplement but are still quoted
    # there, so they stay under assertion.
    add("SI Residential day",
        100 * pick(t6, sector="Residential", test="T3_day_night",
                   term="day").cum, -6.52, 0.006)
    add("SI Residential night",
        100 * pick(t6, sector="Residential", test="T3_day_night",
                   term="night").cum, 4.84, 0.006)
    add("SI Residential night-minus-day t",
        pick(t6, sector="Residential", test="T3_day_night",
             term="night_minus_day").t, 3.44, 0.006)
    if abs(float(pick(t6, sector="Power", test="T3_day_night",
                      term="night_minus_day").t)) >= 1.96:
        raise AssertionError(
            "the Figure 5 caption says no day-night difference is significant "
            "in Power; it now is")
    for lab, spec, term, cum, tt in [
            ("national T alone", "national temperature alone", "nat",
             3.43, 3.01),
            ("FUA pop alone", "FUA pop-weighted alone", "fua", 3.32, 3.08),
            ("FUA equal alone", "FUA equal-weighted alone", "fuae",
             3.54, 3.15),
            ("city T alone", "city temperature alone", "city", 3.09, 2.87),
            ("national T with FUA", "national + FUA, both pop-weighted",
             "nat", 2.35, 0.56),
            ("FUA with national", "national + FUA, both pop-weighted",
             "fua", 1.04, 0.29),
            ("national T with WBGT", "national temperature + city WBGT>28",
             "nat", 2.94, 2.51),
            ("WBGT with national T", "national temperature + city WBGT>28",
             "wbgt", 0.91, 2.52),
    ]:
        r = pick(t4c, specification=spec, term=term)
        add(f"caption 5 {lab} cum", r.cum_pct, cum, 0.006)
        add(f"caption 5 {lab} t", r.t, tt, 0.006)
    rr = t4c[t4c.specification == "within-country correlation"]
    add("caption 5 r national vs FUA",
        pick(rr, term="nat_pop_cdd vs fua_pop_cdd").t, 0.99, 0.006)
    add("caption 5 r national T vs WBGT",
        pick(rr, term="nat_pop_scdh24 vs w_twbg_danger").t, 0.49, 0.006)
    add("caption 5 country-days", t4c.n.max(), 40908, 0.5)

    # Figure S12: the spread and the sign counts printed in every panel.
    for sec, unit, lo, hi, nneg, npos in [
            ("Power", "per100", -6.66, 17.32, 2, 2),
            ("Industry", "per100", -6.53, 16.76, 3, 3),
            ("Power", "perSD", -11.41, -9.13, 18, 0),
            ("Power", "perP95", -30.52, -24.32, 18, 0),
    ]:
        g = scan[(scan.sector == sec) & (scan.normalisation == unit)]
        add(f"caption S12 {sec} {unit} min", g.difference.min(), lo, 0.006)
        add(f"caption S12 {sec} {unit} max", g.difference.max(), hi, 0.006)
        add(f"caption S12 {sec} {unit} cells", len(g), 18, 0.5)
        add(f"caption S12 {sec} {unit} significantly negative",
            (g.diff_z < -1.96).sum(), nneg, 0.5)
        add(f"caption S12 {sec} {unit} significantly positive",
            (g.diff_z > 1.96).sum(), npos, 0.5)
    add("caption S12 Power per-SD spread",
        scan[(scan.sector == "Power") & (scan.normalisation == "perSD")]
        .difference.pipe(lambda s: s.max() - s.min()), 2.29, 0.006)
    add("caption S12 Power per-p95 spread",
        scan[(scan.sector == "Power") & (scan.normalisation == "perP95")]
        .difference.pipe(lambda s: s.max() - s.min()), 6.20, 0.006)
    pub = scan[(scan.sector == "Power") & (scan.normalisation == "per100")
               & (scan.cool_base_c == 24.0)]
    add("caption S12 Power at 24/15", pick(pub, heat_base_c=15.0).difference,
        6.20, 0.006)
    add("caption S12 Power at 24/15 z", pick(pub, heat_base_c=15.0).diff_z,
        1.43, 0.006)
    add("caption S12 Power at 24/18", pick(pub, heat_base_c=18.0).difference,
        10.37, 0.006)
    add("caption S12 Power at 24/18 z", pick(pub, heat_base_c=18.0).diff_z,
        2.36, 0.006)

    # Figure 6.
    sd = T("R5/tables/T6_thermal_asymmetry_Power_tierAB_perSD.csv")
    sd = sd[sd.both.fillna(False)].dropna(subset=["diff_eb_pct"])
    add("caption 6 countries with both sides", len(sd), 18, 0.5)
    add("caption 6 heating dominates", (sd.diff_eb_hi < 0).sum(), 12, 0.5)
    add("caption 6 cooling dominates", (sd.diff_eb_lo > 0).sum(), 1, 0.5)
    fr = pick(sd, country="France")
    add("caption 6 France heating", fr.cold_pct, 53.31, 0.006)
    add("caption 6 France cooling", fr.heat_pct, 3.89, 0.006)
    ch = pick(sd, country="Switzerland")
    if not ch.heat_pct > ch.cold_pct:
        raise AssertionError(
            "the Figure 6 caption says Switzerland's cooling slope exceeds its "
            "heating slope, which the per-SD table no longer supports")
    t8sd = T("R5/tables/T7_sector_comparison_tierAB_perSD.csv")
    t8p95 = T("R5/tables/T7_sector_comparison_tierAB_perP95.csv")
    add("caption 6 pooled Power per SD",
        pick(t8sd, sector="Power")["diff"], -10.59, 0.006)
    add("caption 6 pooled Power per SD z",
        pick(t8sd, sector="Power").diff_z, -4.06, 0.006)
    add("caption 6 detection floor",
        max(abs(float(pick(t8sd, sector=s)["diff"]))
            for s in ("Aviation", "Ground Transport")), 1.58, 0.006)
    # Cochran's Q is now the GLS statistic using the full cross-country
    # covariance. The independence version is carried in the table so the size
    # of the correction can be asserted rather than described.
    add("caption 6 Power difference Q", pick(t8sd, sector="Power").diff_Q,
        239.81, 0.06)
    add("caption 6 Power difference Q, independence version",
        pick(t8sd, sector="Power").diff_Q_indep, 226.70, 0.06)
    add("caption 6 Residential difference Q",
        pick(t8sd, sector="Residential").diff_Q, 545.99, 0.06)
    add("caption 6 Residential difference Q, independence version",
        pick(t8sd, sector="Residential").diff_Q_indep, 1021.72, 0.06)
    add("caption 6 Power difference Q df",
        pick(t8sd, sector="Power").diff_Q_df, 17, 0.5)
    add("caption 6 Power difference I2", pick(t8sd, sector="Power").diff_I2,
        92.91, 0.06)
    add("caption 6 Power cooling I2", pick(t8sd, sector="Power").heat_I2,
        95.54, 0.06)
    add("caption 6 Power heating I2", pick(t8sd, sector="Power").cold_I2,
        93.62, 0.06)
    add("caption 6 Aviation difference I2",
        pick(t8sd, sector="Aviation").diff_I2, 61.18, 0.06)
    # Every heterogeneity number must come from the covariance-consistent test;
    # a table written by an older version would silently reinstate the
    # independence assumption.
    for sec_ in ("Power", "Industry", "Ground Transport", "Aviation",
                 "Residential"):
        for q in ("heat", "cold", "diff"):
            if not bool(pick(t8sd, sector=sec_)[f"{q}_cov_consistent"]):
                raise AssertionError(
                    f"{sec_} {q} heterogeneity was computed without the "
                    "cross-country covariance")
    # The caption's claim about the I2 = 75 % line is a rule about which
    # sectors fall on which side of it, so it is asserted as one.
    for sec_ in ("Power", "Industry", "Residential"):
        for q in ("heat", "cold", "diff"):
            if pick(t8sd, sector=sec_)[f"{q}_I2"] <= 75:
                raise AssertionError(
                    f"the Figure 6d caption says every quantity in {sec_} is "
                    f"above I2 = 75 %; {q} is not")
    for sec_ in ("Ground Transport", "Aviation"):
        for q in ("heat", "cold", "diff"):
            if pick(t8sd, sector=sec_)[f"{q}_I2"] >= 75:
                raise AssertionError(
                    f"the Figure 6d caption says every quantity in {sec_} is "
                    f"below I2 = 75 %; {q} is not")
    add("caption 6 Power per 100 K.h", pick(t8, sector="Power")["diff"],
        6.21, 0.006)
    add("caption 6 Power per 100 K.h z", pick(t8, sector="Power").diff_z,
        1.43, 0.006)
    add("caption 6 Industry per 100 K.h", pick(t8, sector="Industry")["diff"],
        5.93, 0.006)
    add("caption 6 Industry per 100 K.h z",
        pick(t8, sector="Industry").diff_z, 1.57, 0.006)
    add("caption 6 Power per p95", pick(t8p95, sector="Power")["diff"],
        -27.62, 0.006)
    add("caption 6 Power per p95 z", pick(t8p95, sector="Power").diff_z,
        -3.79, 0.006)
    add("caption 6 Residential per p95",
        pick(t8p95, sector="Residential")["diff"], -77.70, 0.006)
    # The caption says the two control sectors keep their sign under all three
    # units. That is the load-bearing claim of panel d, so it is a rule.
    for s, want in (("Ground Transport", -1), ("Aviation", +1)):
        signs = {float(pick(tb, sector=s)["diff"]) > 0
                 for tb in (t8, t8sd, t8p95)}
        if signs != {want > 0}:
            raise AssertionError(
                f"the Figure 6 caption says {s} keeps one sign under all three "
                "units; it no longer does")

    # Figure 6, panels a, b and f -- the panels brought back from the original.
    sha = cty_sd = T("R5/tables/T6_thermal_asymmetry_Power_tierAB_perSD.csv")
    sha = sha[sha.heat_ok.fillna(False)].dropna(subset=["heat_eb_pct"])
    add("caption 6 countries with a cooling side", len(sha), 20, 0.5)
    add("caption 6 median cooling shrinkage", sha.heat_shrink.median(),
        0.09, 0.006)
    # Panel b's message is a rule: at least one country must sit on each side
    # of the 45-degree line, or "country-specific" is not what the panel shows.
    above = int((sd.heat_eb_pct > sd.cold_eb_pct).sum())
    if not 0 < above < len(sd):
        raise AssertionError(
            "the Figure 6b caption says countries fall on both sides of the "
            f"45-degree line; {above} of {len(sd)} are above it")
    cov = T(f"R5/tables/T8_covid_window_tierAB_perSD.csv")
    full = cov[cov.window == "2019-2022"].set_index("sector")
    drop = cov[cov.window == "drop 2020"].set_index("sector")
    only = cov[cov.window == "2020 only"].set_index("sector")
    add("caption 6 COVID sectors", full.shape[0], 5, 0.5)
    same = sum(np.sign(float(full.loc[s, "diff"]))
               == np.sign(float(drop.loc[s, "diff"]))
               for s in full.index)
    add("caption 6 sectors keeping their sign", same, 5, 0.5)
    add("caption 6 Power full window", full.loc["Power", "diff"],
        -10.59, 0.006)
    add("caption 6 Power drop 2020", drop.loc["Power", "diff"], -12.62, 0.006)
    add("caption 6 Power drop 2020 z", drop.loc["Power", "diff_z"],
        -4.66, 0.006)
    add("caption 6 Power 2020 only", only.loc["Power", "diff"], -5.68, 0.006)
    add("caption 6 Power 2020 only z", only.loc["Power", "diff_z"],
        -1.29, 0.006)
    # The 2020-only window must be the least precise of the three, or the
    # caption's reading of it as a power problem is wrong.
    for s in full.index:
        if not (float(only.loc[s, "diff_se"]) > float(full.loc[s, "diff_se"])):
            raise AssertionError(
                f"the Figure 6f caption treats the 2020-only window as the "
                f"least precise; for {s} it is not")

    # Figure S11: the ratio the paper declines to use.
    rt = t7[t7.both.fillna(False) & t7.ratio.notna()]
    add("caption S11 ratio minimum", rt.ratio.min(), -32.1, 0.06)
    add("caption S11 ratio maximum", rt.ratio.max(), 10.9, 0.06)
    if str(rt.loc[rt.ratio.idxmin(), "country"]) != "Greece" or \
            str(rt.loc[rt.ratio.idxmax(), "country"]) != "Korea":
        raise AssertionError(
            "the Figure S11 caption names Greece and Korea as the extremes of "
            "the ratio; the table no longer agrees")

    # Figure 6e: the base-pair whisker. Its whole purpose is that the per-100
    # range crosses zero and the two scale-free ranges do not, so that is
    # asserted as a rule rather than left to the eye.
    for sec_, unit_, lo_, hi_ in [("Power", "per100", -6.66, 17.32),
                                  ("Industry", "per100", -6.53, 16.76),
                                  ("Power", "perSD", -11.41, -9.13),
                                  ("Power", "perP95", -30.52, -24.32)]:
        g = scan[(scan.sector == sec_) & (scan.normalisation == unit_)]
        add(f"caption 6e {sec_} {unit_} whisker low", g.difference.min(),
            lo_, 0.006)
        add(f"caption 6e {sec_} {unit_} whisker high", g.difference.max(),
            hi_, 0.006)
    for sec_ in ("Power", "Industry", "Ground Transport", "Aviation"):
        rng = {}
        for unit_ in ("per100", "perSD", "perP95"):
            g = scan[(scan.sector == sec_) & (scan.normalisation == unit_)]
            rng[unit_] = (float(g.difference.min()), float(g.difference.max()))
        for unit_ in ("perSD", "perP95"):
            if rng[unit_][0] < 0 < rng[unit_][1]:
                raise AssertionError(
                    f"the Figure 6e caption says the scale-free whiskers do "
                    f"not cross zero; {sec_} {unit_} does")
    if not (scan[(scan.sector == "Power")
                 & (scan.normalisation == "per100")].difference.min() < 0
            < scan[(scan.sector == "Power")
                   & (scan.normalisation == "per100")].difference.max()):
        raise AssertionError(
            "the Figure 6e caption says the per-100 K.h whisker for Power "
            "crosses zero; it no longer does")

    # Figure S10.
    for sec, ab, cc_ in [("Power", 2.96, 0.62), ("Industry", 2.74, 0.57)]:
        add(f"caption S10 {sec} Tier A+B",
            100 * pick(t5, tier="AB", sector=sec).cum, ab, 0.006)
        add(f"caption S10 {sec} Tier C",
            100 * pick(t5, tier="C", sector=sec).cum, cc_, 0.006)

    # ---- SI.1 -------------------------------------------------------------
    a = ts1[(ts1.tier == "AB") & (ts1.set == "matched (daily Tmax CDD)")]
    add("SI.1 national pop LOCO",
        pick(a, benchmark="2 nat_pop").oos_r2_loco, 0.00451, 0.000006)
    add("SI.1 FUA pop LOCO",
        pick(a, benchmark="4 fua_pop").oos_r2_loco, 0.00415, 0.000006)
    add("SI.1 gap (FUA pop - national pop)",
        pick(a, benchmark="4 fua_pop").oos_r2_loco
        - pick(a, benchmark="2 nat_pop").oos_r2_loco, -0.00036, 0.000006)

    # ---- SI.2: the boundary claim, asserted as a rule not a number --------
    for th, want_p, want_a in [(24.0, 4.26, -0.62), (25.0, 4.58, -0.72),
                               (26.0, 4.80, 0.08), (27.0, 7.10, 4.72),
                               (28.0, 15.73, 8.77)]:
        add(f"SI.2 Power t at {th:.0f} C",
            pick(ts2, sector="Power", family="cond", threshold_c=th).t, want_p)
        add(f"SI.2 Aviation t at {th:.0f} C",
            pick(ts2, sector="Aviation", family="cond", threshold_c=th).t,
            want_a, 0.006)

    # "the interpretable range is theta <= 26" must follow from the table,
    # not from the sentence. A future rerun that moves the boundary should
    # break here rather than leave the prose quietly wrong.
    av = ts2[(ts2.sector == "Aviation") & (ts2.family == "cond")]
    clean = set(av.loc[av.t.abs() < 2, "threshold_c"])
    if clean != {24.0, 25.0, 26.0}:
        raise AssertionError(
            f"SI.2 says the interpretable range is theta <= 26; the table has "
            f"a clean Aviation placebo at {sorted(clean)}")

    ts3 = T("SI/tables/TS3_chhe_threshold_prevalence.csv")
    add("SI.2 prevalence at 26 C (%)",
        pick(ts3, threshold_c=26.0).conditional_share_pct, 0.824)
    add("SI.2 countries at 28 C",
        pick(ts3, threshold_c=28.0).conditional_n_countries, 14, 0.5)

    # ---- SI.3 -------------------------------------------------------------
    ts4 = T("SI/tables/TS4_oxcgrt_robustness_tierAB.csv")
    add("SI.3 max relative change (%)", 100 * ts4.rel_change.abs().max(), 3.0, 0.05)
    add("SI.3 sign flips",
        int(((ts4.base_pct * ts4.ctrl_pct) < 0).sum()), 0, 0.5)
    add("SI.3 significance changes",
        int(((ts4.base_t.abs() > 1.96) != (ts4.ctrl_t.abs() > 1.96)).sum()),
        0, 0.5)
    add("SI.3 R5 difference, controlled",
        pick(ts4, test="R5 pooled: cooling - heating").ctrl_pct, 6.267, 0.01)
    add("SI.3 R4 humidity, controlled",
        pick(ts4, test="R4 Power: humidity at same load").ctrl_pct, 0.837)
    ts5 = T("SI/tables/TS5_oxcgrt_country_match.csv")
    add("SI.3 countries matched", int(ts5.matched.sum()), 28, 0.5)

    # ---- captions for Supplementary Figures S1-S3 -------------------------
    # These three captions quote numbers that SI.1-SI.3 do not, so they are
    # asserted here rather than trusted.

    # S1: both comparison sets, so the caption's "the gain is temporal, not
    # spatial" reading cannot drift away from the table.
    for bench, want in [("1 nat_area", 0.00273), ("2 nat_pop", 0.00451),
                        ("3 fua_equal", 0.00438), ("4 fua_pop", 0.00415)]:
        add(f"caption S1 matched {bench}",
            pick(a, benchmark=bench).oos_r2_loco, want, 0.000006)
    nat = ts1[(ts1.tier == "AB") & (ts1.set == "native (best available)")]
    for bench, want in [("3 fua_equal", 0.00462), ("4 fua_pop", 0.00449)]:
        add(f"caption S1 native {bench}",
            pick(nat, benchmark=bench).oos_r2_loco, want, 0.000006)
    # the caption calls S1 a null: no FUA benchmark may beat national pop.
    best_fua = nat[nat.benchmark.str.startswith(("3", "4"))].oos_r2_loco.max()
    nat_pop = pick(nat, benchmark="2 nat_pop").oos_r2_loco
    if best_fua - nat_pop > 0.0005:
        raise AssertionError(
            f"caption S1 reports a null, but the best FUA benchmark now beats "
            f"national pop by {best_fua - nat_pop:.5f}")

    # S2: the coefficient profile the caption calls "a smooth decline, not a
    # threshold effect", plus the prevalence collapse it rests on.
    for th, want in [(24.0, 1.53), (25.0, 1.51), (26.0, 1.33),
                     (27.0, 0.89), (28.0, 0.65)]:
        add(f"caption S2 Power %/SD at {th:.0f} C",
            pick(ts2, sector="Power", family="cond",
                 threshold_c=th).pct_per_sd, want, 0.006)
    prof = [pick(ts2, sector="Power", family="cond",
                 threshold_c=th).pct_per_sd for th in (24., 25., 26., 27., 28.)]
    if any(b > a_ + 1e-9 for a_, b in zip(prof, prof[1:])):
        raise AssertionError(
            f"caption S2 calls the Power profile monotone decreasing; it is "
            f"now {['%.3f' % v for v in prof]}")
    add("caption S2 prevalence at 24 C (%)",
        pick(ts3, threshold_c=24.0).conditional_share_pct, 1.86, 0.006)
    add("caption S2 countries at 24 C",
        pick(ts3, threshold_c=24.0).conditional_n_countries, 38, 0.5)
    # the placebo must break in the pure family too, or the caption overclaims.
    for th, want in [(26.0, 0.88), (27.0, 9.30), (28.0, 14.29)]:
        add(f"caption S2 Aviation pure t at {th:.0f} C",
            pick(ts2, sector="Aviation", family="pure", threshold_c=th).t,
            want, 0.02)
    ts2c = T("SI/tables/TS2_chhe_threshold_scan_tierC.csv")
    add("caption S2 Tier C max |t|", ts2c.t.abs().max(), 1.56, 0.006)
    if ts2c.t.abs().max() >= 1.96:
        raise AssertionError(
            "caption S2 says Tier C returns nothing at any threshold, but a "
            "cell is now significant")

    # S3: the four coefficients the caption prints as base -> controlled.
    for test, term, wb, wc in [
        ("R3 Power: cooling load", "heat", 3.087, 3.078),
        ("R3 Power: heating load", "cold", 10.349, 10.384),
        ("R4 Power: humidity at same load", "humid", 0.813, 0.837),
        ("R5 pooled: cooling - heating", "diff", 6.207, 6.267),
    ]:
        r = pick(ts4, test=test, term=term)
        add(f"caption S3 {test} base", r.base_pct, wb, 0.006)
        add(f"caption S3 {test} controlled", r.ctrl_pct, wc, 0.006)
    add("caption S3 max relative change (%)",
        100 * ts4.rel_change.abs().max(), 2.95, 0.006)
    # the caption's "identical n" claim is the reason nothing moves.
    if not (ts4.n_base == ts4.n_ctrl).all():
        raise AssertionError(
            "caption S3 says base and controlled fits use the identical "
            "sample; TS4 now has rows where n_base != n_ctrl")
    add("caption S3 n", ts4.n_ctrl.iloc[0], 40712, 0.5)

    # ---- V2 / Fig 7f: the corrected Power-only mass equivalents -----------
    #
    # This block used to read TS21/TS21b (45_absolute_magnitude.py). Those
    # numbers were withdrawn: they stacked Industry, whose daily profile is
    # downscaled from national electricity activity; they priced a pooled-SD
    # move while counting days on a within-country SD (a x2.16 / x2.89
    # mismatch that differed between the two sides); and the mid scope
    # multiplied a continuous-load coefficient by a percentile event's mean
    # duration. The "28 of 28 countries" claim that sat here was a tautology,
    # not a finding: with one pooled beta per side the country's own mean
    # emissions cancel, so the comparison reduced to N_cold/N_heat > 0.2987
    # against an observed minimum of 0.530. Its assertion has been deleted
    # rather than reworded -- it guarded an identity.
    mag_p = root / "R5" / "tables" / "T22b_corrected_power_mass_totals.csv"
    if mag_p.exists():
        m = pd.read_csv(mag_p).set_index(["scope", "side"])
        con = pd.read_csv(
            root / "R5" / "tables"
            / "T22d_corrected_power_mass_contrasts.csv").set_index("scope")
        DAY = "simultaneous_one_day_pooled_1sd_cumulative_lag_0_14"
        SPELL = "one_typical_observed_high_load_spell_per_country"
        YEAR = "observed_high_load_anomalies_average_annual"

        for tag, scope, k, want in [
                ("1-SD day", DAY, 1.0, (436.7, 128.8)),
                ("spell", SPELL, 1.0, (726.9, 443.3)),
                ("year", YEAR, 1e-3, (11.33, 4.72))]:
            for side, lab, w in (("cold", "heating", want[0]),
                                 ("heat", "cooling", want[1])):
                add(f"Fig 7f {lab} {tag}", m.loc[(scope, side), "estimate_kt"] * k,
                    w, max(0.02, abs(w) * 0.002))
            add(f"Fig 7f ratio {tag}", con.loc[scope, "cold_to_heat_ratio"],
                {DAY: 3.4, SPELL: 1.6, YEAR: 2.4}[scope], 0.05)

        # The caption prints the intervals; they have to be the ones in the
        # table, not rounded from memory.
        for scope, side, lo, hi in [(DAY, "cold", 204, 669),
                                    (DAY, "heat", 45, 213),
                                    (SPELL, "cold", 338, 1116),
                                    (SPELL, "heat", 153, 734)]:
            add(f"Fig 7f CI lo {scope[:9]} {side}",
                m.loc[(scope, side), "ci95_lo_kt"], lo, 1.0)
            add(f"Fig 7f CI hi {scope[:9]} {side}",
                m.loc[(scope, side), "ci95_hi_kt"], hi, 1.0)

        # Structural: the ordering must hold at every scope, and the spell
        # scope must remain the one whose contrast straddles zero -- the text
        # and the panel both say so in as many words.
        for scope in (DAY, SPELL, YEAR):
            if not (m.loc[(scope, "cold"), "estimate_kt"]
                    > m.loc[(scope, "heat"), "estimate_kt"]):
                raise AssertionError(
                    f"Fig 7f and the Results say the heating side is the "
                    f"larger in mass at every scope; that fails at {scope}")
        ns = {s: bool(con.loc[s, "ci95_lo_kt"] < 0 < con.loc[s, "ci95_hi_kt"])
              for s in (DAY, SPELL, YEAR)}
        if ns != {DAY: False, SPELL: True, YEAR: False}:
            raise AssertionError(
                "the caption marks the spell scope, and only the spell scope, "
                f"as not significant; the table now says {ns}")

        val = pd.read_csv(root / "R5" / "tables"
                          / "T22_corrected_power_mass_refit_validation.csv")
        if float(val.absolute_difference.max()) > 1e-10:
            raise AssertionError(
                "Fig 7f claims its refit reproduces T3 to 1e-10; the "
                "validation table no longer supports that")

    # ---- Methods: the design constants the Methods section states -----------
    #
    # These are quoted in prose rather than in a figure, which is exactly the
    # kind of number that goes stale in silence. They are asserted here so a
    # change in the sample or the lag window fails loudly.
    tier_p = (BASE / "outputs" / "exposure" / "tables" / "country_tier.csv")
    if tier_p.exists():
        tt = pd.read_csv(tier_p)
        add("Methods Tier A countries", int((tt.tier == "A").sum()), 25, 0.5)
        add("Methods Tier B countries", int((tt.tier == "B").sum()), 3, 0.5)
        add("Methods Tier C countries", int((tt.tier == "C").sum()), 18, 0.5)
        add("Methods Tier A+B countries",
            int(tt.tier.isin(["A", "B"]).sum()), 28, 0.5)

    t3n = T("R3/tables/T3_sectoral_response.csv")
    t3n = t3n[t3n.form == "continuous"]
    add("Methods n, main response",
        pick(t3n, sector="Power", exposure="heat").n, 40320, 0.5)
    add("Methods n, aviation",
        pick(t3n, sector="Aviation", exposure="heat").n, 39589, 0.5)

    # ---- V2 section 1: where each load dimension is highest ------------------
    #
    # The section says the three dimensions are highest in different places and
    # supports it with counts of a country's sampled FUAs in the highest global
    # decile of each metric. Counts are asserted, not shares: the count is the
    # datum, the share was an interpretation and has been dropped.
    sat_p = root / "SI" / "tables" / "TS22_load_saturation_by_country.csv"
    if sat_p.exists():
        sat = pd.read_csv(sat_p)

        def cell(var, country):
            r = sat[(sat.variable == var) & (sat.country == country)]
            return (int(r.cities_in_global_top_decile.iloc[0]),
                    int(r.cities_sampled.iloc[0])) if len(r) else (0, 0)

        for var, country, want_k, want_n in [
                ("cum_scdh24", "India", 55, 60),
                ("n_utci_strong_days", "India", 54, 60),
                ("n_utci_strong_days", "Brazil", 19, 40),
                ("n_utci_strong_days", "Colombia", 25, 53),
                ("n_utci_strong_days", "Mexico", 36, 92),
                ("n_CHHE_days", "India", 32, 60),
                ("n_CHHE_days", "China", 92, 281),
                ("cum_cdh_twb26", "India", 31, 60),
                ("cum_cdh_twb26", "China", 111, 281),
                ("cum_hdh15", "Finland", 7, 7),
                ("cum_hdh15", "Norway", 4, 4),
                ("cum_hdh15", "Russia", 41, 50),
                ("cum_hdh15", "Canada", 18, 26)]:
            k, n_ = cell(var, country)
            add(f"V2.1 {country} {var} top-decile cities", k, want_k, 0.5)
            add(f"V2.1 {country} {var} cities sampled", n_, want_n, 0.5)

        # Structural: the four cold-belt countries must carry no heat-side
        # load at all -- the section says "not one city".
        cold = {"Finland", "Norway", "Russia", "Canada"}
        for var in ("cum_scdh24", "n_utci_strong_days"):
            bad = sorted(cold & set(sat[sat.variable == var].country))
            if bad:
                raise AssertionError(
                    f"section 1 says no city of Finland, Norway, Russia or "
                    f"Canada reaches the highest decile of either heat metric; "
                    f"{bad} now do on {var}")

    # ---- V2 section 1: dynamic range and structural zeros -------------------
    # On the 1,583-row city basis, matching the published 630 / 649 / 205
    # counts. TS22's shares are on the 1,563-polygon basis; the two are stated
    # separately in the text and must not be mixed.
    zr = root / "SI" / "tables" / "TS23_metric_range_and_zeros.csv"
    if zr.exists():
        z = pd.read_csv(zr).set_index("variable")
        for var, med, mx, nzero in [
                ("n_EHD_days", 46.8, 109.5, 0),
                ("cum_scdh24", 3307.0, 51529.2, 16),
                ("n_CHHE_days", 0.0, 42.8, 953),
                ("cum_cdh_twb26", 0.0, 3324.6, 934)]:
            add(f"V2.1 {var} median", z.loc[var, "median"], med, 0.15)
            add(f"V2.1 {var} max", z.loc[var, "max"], mx, 0.15)
            add(f"V2.1 {var} cities at zero", z.loc[var, "n_zero"], nzero, 0.5)
        add("V2.1 EHD max/median", z.loc["n_EHD_days", "max"]
            / z.loc["n_EHD_days", "median"], 2.3, 0.05)
        add("V2.1 cooling DH max/median", z.loc["cum_scdh24", "max"]
            / z.loc["cum_scdh24", "median"], 15.6, 0.05)
        if int(z.loc["n_EHD_days", "n_zero"]) != 0:
            raise AssertionError(
                "section 1 says the relative criterion is non-zero in every "
                "city; that is no longer true")

    # ---- V2: the dynamic-shape indicators, previously computed and unused --
    t3c = T("R3/tables/T3_sectoral_response.csv")
    t3c = t3c[t3c.form == "continuous"]
    for sec_, side, want_j, want_hl, want_peak in [
            ("Power", "cold", 5.007, 1.0, 0), ("Power", "heat", 0.324, 1.0, 0),
            ("Industry", "cold", 5.062, 1.0, 0),
            ("Ground Transport", "heat", 0.185, 1.0, 13),
            ("Aviation", "heat", 0.005, None, 14)]:
        r = pick(t3c, sector=sec_, exposure=side)
        add(f"V2 {sec_} {side} onset jump (%)", 100 * r.onset_jump, want_j,
            0.006)
        add(f"V2 {sec_} {side} peak lag", r.peak_lag, want_peak, 0.5)
        if want_hl is not None:
            add(f"V2 {sec_} {side} half-life (d)", r.half_life, want_hl, 0.06)
    pc = pick(t3c, sector="Power", exposure="cold")
    ph = pick(t3c, sector="Power", exposure="heat")
    add("V2 onset jump ratio cold/heat",
        pc.onset_jump / ph.onset_jump, 15.5, 0.06)
    add("V2 cold onset as share of cumulative (%)",
        100 * pc.onset_jump / pc.cum, 50.5, 0.6)
    add("V2 heat onset as share of cumulative (%)",
        100 * ph.onset_jump / ph.cum, 10.9, 0.6)
    # the lead and day-0 coefficients themselves: the claim is about their
    # RATIO, and quoting only the jump previously reversed the heat-side reading
    lp = pd.read_csv(BASE / "outputs" / "response" /
                     "tables" / "r3_lag_profiles_tierAB.csv")
    lp = lp[lp.form == "continuous"]
    for side, want_m1, want_0, want_share in [("cold", 0.068, 5.075, 51.2),
                                              ("heat", 0.621, 0.945, 31.9)]:
        g = lp[(lp.sector == "Power") & (lp.exposure == side)].set_index("lag")
        add(f"V2 Power {side} beta(-1) (%)", 100 * g.beta[-1], want_m1, 0.006)
        add(f"V2 Power {side} beta(0) (%)", 100 * g.beta[0], want_0, 0.006)
        post = g[g.index >= 0]
        add(f"V2 Power {side} day-0 share of cumulative (%)",
            100 * g.beta[0] / post.beta.sum(), want_share, 0.06)
    gh = lp[(lp.sector == "Power") & (lp.exposure == "heat")].set_index("lag")
    gc = lp[(lp.sector == "Power") & (lp.exposure == "cold")].set_index("lag")
    if not (gh.beta[-1] / gh.beta[0] > 0.5 > gc.beta[-1] / gc.beta[0]):
        raise AssertionError(
            "V2 says the heat side is already elevated the day before while "
            "the cold side is not; that contrast no longer holds")
    add("V2 Power cold rebound (%)", 100 * pc.rebound, 1.015, 0.006)
    add("V2 Power heat rebound (%)", 100 * ph.rebound, 0.345, 0.006)
    # the section's two structural claims
    resp = t3c[t3c.sector.isin(["Power", "Industry"])]
    ctrl = t3c[t3c.sector.isin(["Ground Transport", "Aviation"])]
    if not (resp.peak_lag == 0).all():
        raise AssertionError(
            "V2 says every responding series peaks at lag 0; it no longer does")
    if not (ctrl.peak_lag.max() >= 13):
        raise AssertionError(
            "V2 says the negative controls peak at the far edge of the window; "
            f"the largest control peak lag is now {ctrl.peak_lag.max()}")
    if not (resp.half_life.dropna() == 1.0).all():
        raise AssertionError(
            "V2 says the half-life is one day in every responding sector; "
            f"it is now {sorted(resp.half_life.dropna().unique())}")

    # ---- numbers the V2 Results quotes that no other section asserts ------
    t1e = T("R1/tables/T1e_event_duration_survival.csv")
    hw = t1e[t1e.event == "Heatwaves"].set_index("duration_days")
    cw = t1e[t1e.event == "Cold waves"].set_index("duration_days")
    add("V2 heat P(D>=10)", hw.loc[10, "p_at_least"], 0.0237, 0.0006)
    add("V2 cold P(D>=10)", cw.loc[10, "p_at_least"], 0.0093, 0.0006)
    add("V2 ten-day persistence ratio",
        hw.loc[10, "p_at_least"] / cw.loc[10, "p_at_least"], 2.5, 0.06)
    # the claim is "widens with length": the ratio must grow from 5 to 10 days
    r5 = hw.loc[5, "p_at_least"] / cw.loc[5, "p_at_least"]
    r10 = hw.loc[10, "p_at_least"] / cw.loc[10, "p_at_least"]
    if not r10 > r5:
        raise AssertionError(
            f"V2 says the heat/cold persistence gap widens with length; the "
            f"ratio now falls from {r5:.2f} at 5 days to {r10:.2f} at 10")

    # Fig 5c: the class means behind the 11.5 K inversion. These were computed
    # inside the plotting function and nowhere else until T4d was added, so
    # nothing backed the four numbers the Results and the caption quote.
    t4d_p = root / "SI" / "tables" / "T4d_definition_class_means.csv"
    if t4d_p.exists():
        t4d = pd.read_csv(t4d_p).set_index("definition")
        add("V2 relative humid mean Tmax",
            t4d.loc["relative humid (HEH)", "mean_tmax_c"], 19.43, 0.006)
        add("V2 relative dry mean Tmax",
            t4d.loc["relative dry (DEH)", "mean_tmax_c"], 30.96, 0.006)
        add("V2 CHHE mean Tmax",
            t4d.loc["absolute humid (CHHE)", "mean_tmax_c"], 35.16, 0.006)
        add("V2 CHHE mean Twb",
            t4d.loc["absolute humid (CHHE)", "mean_twb_max_c"], 27.23, 0.006)
        add("V2 humid-minus-dry Tmax inversion (K)",
            t4d.loc["relative dry (DEH)", "mean_tmax_c"]
            - t4d.loc["relative humid (HEH)", "mean_tmax_c"], 11.53, 0.006)
        # HEH and DEH are a tercile split, so their day counts must be equal
        if (t4d.loc["relative humid (HEH)", "n_city_days"]
                != t4d.loc["relative dry (DEH)", "n_city_days"]):
            raise AssertionError(
                "HEH and DEH are p33/p67 terciles of the same set and must "
                "have equal day counts; they no longer do")

    t1b = T("R1/tables/T1b_metric_geography.csv")
    if "cv" in t1b.columns and "variable" in t1b.columns:
        for var, want in [("n_CW_days", 0.50)]:
            hit = t1b[t1b.variable == var]
            if len(hit) == 1:
                add(f"V2 CV {var}", hit.cv.iloc[0], want, 0.006)

    # ---- R1.5: the named-city table --------------------------------------
    ts20_p = root / "SI" / "tables" / "TS20_megacity_exposure.csv"
    if ts20_p.exists():
        g = pd.read_csv(ts20_p)
        add("R1.5 megacities listed", len(g), 20, 0.5)
        for city, col, want in [
                ("Mexico City", "n_EHD_days", 58.0),
                ("Bangkok", "n_EHD_days", 25.2),
                ("Mexico City", "cum_scdh24", 0.4),
                ("Bangkok", "cum_scdh24", 35.6),
                ("Lagos", "n_EHD_days", 89.5),
                ("Moscow", "n_utci_strong_days", 11.5),
                ("Bangkok", "n_utci_strong_days", 346.0),
                ("Kolkata", "n_CHHE_days", 33.0),
                ("Moscow", "cum_hdh15", 84.8),
                ("Tokyo", "n_EHD_days", 48.5),
                ("Kolkata", "n_EHD_days", 47.8),
                ("Tokyo", "cum_scdh24", 3.5),
                ("Kolkata", "cum_scdh24", 28.4)]:
            add(f"R1.5 {city} {col}", pick(g, city=city)[col], want, 0.06)
        # the section's structure, not just its numbers
        rel = g.n_EHD_days
        add("R1.5 relative spread (x)", rel.max() / rel.min(), 3.55, 0.02)
        add("R1.5 cooling spread (x)",
            g.cum_scdh24.max() / g.cum_scdh24.min(), 89.0, 0.6)
        add("R1.5 megacities with zero compound humid heat",
            int((g.n_CHHE_days == 0).sum()), 6, 0.5)
        add("R1.5 megacities with zero heating degree-hours",
            int((g.cum_hdh15 == 0).sum()), 3, 0.5)
        if not (g.cum_scdh24.max() / g.cum_scdh24.min()
                > 5 * rel.max() / rel.min()):
            raise AssertionError(
                "R1.5 claims the physical metrics spread these cities far more "
                "than the relative ones; that ordering no longer holds")

    # ---- Figure 6: every number printed on the new unit figure ------------
    scan = T("R5/tables/T8b_base_scan_tierAB.csv")
    pw = scan[scan.sector == "Power"]

    # panel a: the coefficient triples while its z does not move. Both halves
    # are asserted, because the panel's whole claim is the contrast.
    a = pw[(pw.normalisation == "per100") & (pw.heat_base_c == 15.0)]
    a = a.sort_values("cool_base_c")
    for base, want_b, want_z in [(16.0, 8.25, 3.63), (20.0, 10.77, 3.61),
                                 (24.0, 16.73, 3.52), (26.0, 22.65, 3.30)]:
        r = pick(a, cool_base_c=base)
        add(f"caption 6a cooling coef at base {base:.0f}", r.cooling, want_b,
            0.006)
        add(f"caption 6a cooling z at base {base:.0f}", r.cooling_z, want_z,
            0.006)
    add("caption 6a coefficient ratio",
        a.cooling.iloc[-1] / a.cooling.iloc[0], 2.7, 0.05)
    # the panel says "rescaling changes the coefficient, not the evidence".
    # If the z ever moved as much as the coefficient, that reading would fail.
    z_ratio = a.cooling_z.max() / a.cooling_z.min()
    b_ratio = a.cooling.max() / a.cooling.min()
    if z_ratio > 1.25 or b_ratio < 2.0:
        raise AssertionError(
            f"caption 6a contrasts a coefficient ratio of {b_ratio:.2f} with a "
            f"z ratio of {z_ratio:.2f}; the contrast no longer holds")

    # panels b and c: the grids, the published cell, and the sign structure
    for norm, lo, hi, n_neg, n_pos in [("per100", -6.66, 17.32, 2, 2),
                                       ("perSD", -11.41, -9.13, 18, 0),
                                       ("perP95", -30.52, -24.32, 18, 0)]:
        g = pw[pw.normalisation == norm]
        add(f"caption 6 {norm} min", g.difference.min(), lo, 0.006)
        add(f"caption 6 {norm} max", g.difference.max(), hi, 0.006)
        add(f"caption 6 {norm} cells", len(g), 18, 0.5)
        add(f"caption 6 {norm} significantly negative",
            int(((g.difference < 0) & (g.diff_z.abs() > 1.96)).sum()),
            n_neg, 0.5)
        add(f"caption 6 {norm} significantly positive",
            int(((g.difference > 0) & (g.diff_z.abs() > 1.96)).sum()),
            n_pos, 0.5)
    # the figure's central visual claim: a zero contour exists in b and not c
    for norm, want_cross in [("per100", True), ("perSD", False),
                             ("perP95", False)]:
        g = pw[pw.normalisation == norm]
        crosses = bool(g.difference.min() < 0 < g.difference.max())
        if crosses != want_cross:
            raise AssertionError(
                f"Figure 6 draws a zero contour for {norm} only if the grid "
                f"spans zero; it now {'does' if crosses else 'does not'}")
    # the two cells the caption quotes by name
    add("caption 6b published pair",
        pick(pw, normalisation="per100", cool_base_c=24.0,
             heat_base_c=15.0).difference, 6.20, 0.006)
    add("caption 6b published pair z",
        pick(pw, normalisation="per100", cool_base_c=24.0,
             heat_base_c=15.0).diff_z, 1.43, 0.006)
    add("caption 6b heating base 18 cell",
        pick(pw, normalisation="per100", cool_base_c=24.0,
             heat_base_c=18.0).difference, 10.37, 0.006)
    add("caption 6b heating base 18 cell z",
        pick(pw, normalisation="per100", cool_base_c=24.0,
             heat_base_c=18.0).diff_z, 2.36, 0.006)
    add("caption 6c spread (pp)",
        pw[pw.normalisation == "perSD"].difference.max()
        - pw[pw.normalisation == "perSD"].difference.min(), 2.29, 0.006)

    # panel d: the sector bars, and the claim that the controls do not move
    for sec_, u, want in [("Power", "per100", 6.21), ("Power", "perSD", -10.59),
                          ("Industry", "per100", 5.93)]:
        t7 = T(f"R5/tables/T7_sector_comparison_tierAB"
               f"{'' if u == 'per100' else '_' + u}.csv")
        add(f"caption 6d {sec_} {u}", pick(t7, sector=sec_)["diff"], want,
            0.006)
    signs = {}
    for u in ("per100", "perSD", "perP95"):
        t7 = T(f"R5/tables/T7_sector_comparison_tierAB"
               f"{'' if u == 'per100' else '_' + u}.csv").set_index("sector")
        for sec_ in ("Ground Transport", "Aviation"):
            signs.setdefault(sec_, set()).add(np.sign(t7.loc[sec_, "diff"]))
    for sec_, ss in signs.items():
        if len(ss) != 1:
            raise AssertionError(
                f"caption 6d says the negative controls do not move with the "
                f"unit; {sec_} now changes sign across the three units")

    # ---- panels the S1 and S3 captions describe but no table carries ------
    # These are read from source rather than from a saved summary, for the
    # reason given at the top of this file.
    nb_p = BASE / "outputs" / "exposure" / "national_benchmark_exposure.parquet"
    if nb_p.exists():
        nb = pd.read_parquet(nb_p)
        nb["date"] = pd.to_datetime(nb.date)
        g = nb[nb.date.dt.year == 2020].groupby("country").agg(
            area=("nat_area_tmax", "mean"), popw=("nat_pop_tmax", "mean"))
        g["diff"] = g.popw - g.area
        add("caption S1b Canada pop - area Tmax (K)", g.loc["Canada", "diff"],
            13.10, 0.006)
        add("caption S1b Australia pop - area Tmax (K)",
            g.loc["Australia", "diff"], -6.79, 0.006)
        add("caption S1b spread (K)", g["diff"].max() - g["diff"].min(),
            19.9, 0.05)
        add("caption S1b countries", len(g), 45, 0.5)

    tier_p = BASE / "outputs" / "exposure" / "tables" / "country_tier.csv"
    if tier_p.exists():
        tt = pd.read_csv(tier_p)
        for tier, n, cov, fua in [("A", 25, 0.98, 26), ("B", 3, 0.65, 50),
                                  ("C", 18, 0.48, 1)]:
            sub = tt[tt.tier == tier]
            add(f"caption S1c Tier {tier} n", len(sub), n, 0.5)
            add(f"caption S1c Tier {tier} median coverage",
                sub.coverage.median(), cov, 0.006)
            add(f"caption S1c Tier {tier} median FUAs",
                sub.n_fua.median(), fua, 0.5)

    ox = BASE / "Other_data" / "OxCGRT_compact_national_v1.csv"
    if ox.exists():
        use = ["CountryName", "Date", "StringencyIndex_Average"]
        o = pd.read_csv(ox, usecols=lambda k: k in use, low_memory=False)
        o["Date"] = pd.to_datetime(o.Date, format="%Y%m%d", errors="coerce")
        o = o[o.CountryName.isin(ts5.oxcgrt)]
        o = o[(o.Date >= "2019-01-01") & (o.Date <= "2022-12-31")]
        o = o.dropna(subset=["StringencyIndex_Average"])
        yr = o.groupby(o.Date.dt.year).StringencyIndex_Average.mean()
        for y, want in [(2020, 52.3), (2021, 56.2), (2022, 24.6)]:
            add(f"caption S3a mean stringency {y}", yr.loc[y], want, 0.06)
        wi = o.groupby([o.CountryName,
                        o.Date.dt.to_period("M")]).StringencyIndex_Average.std()
        add("caption S3b median within-month SD", wi.median(), 1.20, 0.006)
        add("caption S3b share of zero-variation country-months (%)",
            100 * (wi == 0).mean(), 32.0, 0.06)
        add("caption S3b country-months", len(wi), 1008, 0.5)
        # the caption reads panel b as a bound on what the control can do;
        # that reading fails if the residual variation is ever large.
        if wi.median() > 5:
            raise AssertionError(
                "caption S3b says the within-month residual is small; the "
                f"median is now {wi.median():.2f} index points")

    # ---- criterion alignment: why the cold percentile criterion works -----
    # Table 1 carries a significant relative-cold result (+1.03 %, t = 2.03,
    # WCB p = 0.028) that the text has to explain rather than omit. The
    # explanation is that the cold threshold tracks physical load and the heat
    # threshold does not; T4e measures that. These assertions guard the
    # numbers the section quotes AND the two structural facts the argument
    # rests on, because either one failing would invalidate the paragraph even
    # if every individual number still matched.
    fa = root / "R3" / "tables" / "T4e_criterion_alignment_tierAB.csv"
    if fa.exists():
        al = pd.read_csv(fa)

        def av(fact, side, contains=None):
            m = al[(al.fact == fact) & (al.side == side)]
            if contains:
                m = m[m.definition.str.contains(contains, regex=False)]
            return float(m.iloc[0].value)

        # Exposure-weighted over all 28 countries. The thresholded variant is
        # still written to T4e but is NOT asserted, because the w_HW > 0.5
        # screen filters on spatial synchrony and drops the United States and
        # Australia; see spread() in 50_criterion_alignment.py.
        add("R3.2 load-free heatwave country-days (%)",
            av("load-free event days", "heat", "0.10 x positive median"),
            48.0, 0.06)
        add("R3.2 load-free cold-wave country-days (%)",
            av("load-free event days", "cold", "0.10 x positive median"),
            7.5, 0.06)
        add("R3.2 load spread, heatwave days (fold)",
            av("spread of load across event days", "heat"), 25.6, 0.06)
        add("R3.2 load spread, cold-wave days (fold)",
            av("spread of load across event days", "cold"), 21.9, 0.06)
        add("R3.2 attenuation of the aligned percentile criterion",
            av("attenuation of the aligned percentile criterion", "cold"),
            9.7, 0.06)

        # Structural assertion 1. The contrast must survive the whole 6 x 3
        # base grid. If the worst heat-side base ever falls below the best
        # cold-side base, the mechanism is a property of the 24/15 C
        # convention rather than of the criterion, and the paragraph has to be
        # rewritten, not renumbered.
        sweep = al[al.fact == "load-free event days, base sweep"]
        hmin = sweep[sweep.side == "heat"].value.min()
        cmax = sweep[sweep.side == "cold"].value.max()
        if not hmin > cmax:
            raise AssertionError(
                "T4e: the heat/cold load-free contrast does not survive the "
                f"base grid (worst heat base {hmin:.1f} % vs best cold base "
                f"{cmax:.1f} %); the criterion-alignment claim must be "
                "qualified to the published 24/15 C convention")
        add("R3.2 base sweep, worst heat-side base (%)", hmin, 28.7, 0.06)
        add("R3.2 base sweep, best cold-side base (%)", cmax, 15.3, 0.06)

        # Structural assertion 2. The alignment argument only works if the
        # relative criteria retain MORE identifying variation than the
        # continuous ones and still under-perform them. If that ordering ever
        # reverses, the null becomes a precision problem and the whole
        # measurement thesis changes.
        var = al[al.fact == "variance surviving fixed effects"]
        keep = dict(zip(var.definition.str.split().str[0], var.value))
        for rel, cont in (("w_HW", "w_SCDH24"), ("w_CW", "w_HDH15")):
            if rel in keep and cont in keep and not keep[rel] > keep[cont]:
                raise AssertionError(
                    f"T4e: {rel} no longer retains more variance than {cont} "
                    f"({keep[rel]:.1f} % vs {keep[cont]:.1f} %); the paper's "
                    "'not a power problem' argument no longer holds")
        if "w_HDH15" in keep:
            add("R3.2 variance kept, w_HDH15 (%)", keep["w_HDH15"], 7.7, 0.06)
        if "w_CW" in keep:
            add("R3.2 variance kept, w_CW (%)", keep["w_CW"], 70.0, 0.06)

    # ---- the cold-side metric race (Fig 4 extended, panel b) --------------
    fc = root / "R3" / "tables" / "T4f_cold_metric_race_tierAB.csv"
    if fc.exists():
        cr = pd.read_csv(fc).set_index("column")
        for col, want_cum, want_t in (("w_CW", 0.966, 2.36),
                                      ("w_ECD", 1.091, 2.29),
                                      ("w_CD", 1.129, 2.62),
                                      ("w_utci_coldstress", 2.570, 2.43),
                                      ("w_twc_danger", 1.130, 3.07),
                                      ("w_HDH15", 9.281, 3.58)):
            if col in cr.index:
                add(f"Fig4ext(b) {col} cum (%)",
                    100 * float(cr.loc[col, "cum"]), want_cum, 0.006)
                add(f"Fig4ext(b) {col} t", float(cr.loc[col, "t"]),
                    want_t, 0.006)
        add("Fig4ext(b) n matches the heat-side race",
            float(cr["n"].iloc[0]), 40516, 0.5)

        # The panel exists to make one structural point: on the cold side every
        # criterion family recovers a signal, whereas on the heat side the two
        # relative metrics do not. If that ever stops being true the panel
        # argues the opposite of its own caption, so it is asserted rather
        # than eyeballed.
        weak = cr[cr["t"].abs() < 1.96]
        if len(weak):
            raise AssertionError(
                "T4f: the cold-side panel claims every criterion family "
                "recovers a signal, but "
                f"{', '.join(weak.index)} now has |t| < 1.96")
        bad_pt = cr[cr["pretrend_p"] < 0.05]
        if len(bad_pt):
            raise AssertionError(
                "T4f: a cold-side metric now fails its joint lead test "
                f"({', '.join(bad_pt.index)}); the panel would have to mark "
                "it red and the 'the cold side is clean' reading no longer "
                "holds")
        rel_heat = t4ab[t4ab.column.isin(["w_EHD", "w_HW"])]
        if (rel_heat["t"].abs() >= 1.96).any():
            raise AssertionError(
                "T4/T4f: a relative HEAT metric is now significant; the "
                "heat/cold contrast that panels a and b are built on has "
                "changed")

    # ---- the metric race under the wild cluster bootstrap ----------------
    # Table 1 makes the restricted country-cluster bootstrap the headline
    # inference standard, so the Figure 4 panels have to clear it too. Five of
    # the six cold-side metrics sit between t = 2.29 and 3.07, which with 28
    # clusters is where the analytical two-way covariance over-rejects.
    fw = root / "R3" / "tables" / "T4g_metric_race_wcb_tierAB.csv"
    if fw.exists():
        wb = pd.read_csv(fw).set_index("column")
        for col, want in (("w_HDH15", 0.001), ("w_utci_coldstress", 0.017),
                          ("w_twc_danger", 0.003), ("w_CD", 0.011),
                          ("w_ECD", 0.025), ("w_CW", 0.010)):
            if col in wb.index:
                add(f"Fig4ext(b) {col} WCB p", float(wb.loc[col, "wcb_p"]),
                    want, 0.0006)
        cold_fail = wb[(wb.side == "cold") & (wb.wcb_p >= 0.05)]
        if len(cold_fail):
            raise AssertionError(
                "T4g: the text states that every cold-side metric survives the "
                f"wild cluster bootstrap, but {', '.join(cold_fail.index)} "
                "now has WCB p >= 0.05")
        rel = wb[wb.index.isin(["w_EHD", "w_HW"])]
        if (rel["wcb_p"] < 0.05).any():
            raise AssertionError(
                "T4g: a relative HEAT metric is now significant under the "
                "bootstrap; the null that the measurement claim rests on has "
                "changed")

    return c


# --------------------------------------------------------------------------
# cross-check: does the prose contain any superseded figure?
# --------------------------------------------------------------------------
STALE = {
    "39,590": "pre-fix R3 sample size",
    "40,178": "pre-fix panel size",
    "30,100": "three-year sample size",
}


def scan_prose(md: Path) -> list[str]:
    """Flag superseded numbers, ignoring the block that documents them."""
    bad = []
    for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith(">"):   # the correction note quotes them
            continue
        for tok, why in STALE.items():
            if tok in line:
                bad.append(f"line {i}: {tok} ({why})")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get(
        "HC_RESULTS_ROOT", "Results_2019_2022"))
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = BASE / root

    print("=" * 78)
    print(f"Verifying narrative against tables in {root.name}")
    print("=" * 78)

    claims = build_claims(root)
    bad = []
    for lab, got, want, tol in claims:
        got, want = float(got), float(want)
        ok = abs(got - want) <= tol
        if not ok:
            bad.append((lab, got, want))
        print(f"  {'ok ' if ok else 'BAD'}  {lab:48s} table {got:12.5f}   "
              f"text {want:12.5f}")

    stale = []
    for md in (root / "Results_R1_R5.md", root / "Results_V2.md",
               root / "SI" / "Results_SI.md", root / "Figure_captions.md"):
        if md.exists():
            stale += [f"{md.name} {s}" for s in scan_prose(md)]
        else:
            stale.append(f"{md.name} is missing")

    print("\n" + "-" * 78)
    if bad:
        print(f"{len(bad)} claim(s) in the text do not match the tables:")
        for lab, got, want in bad:
            print(f"  {lab}: table {got:.5f} vs text {want:.5f}")
    else:
        print(f"all {len(claims)} numeric claims match the source tables")

    if stale:
        print(f"\n{len(stale)} superseded number(s) still in the prose:")
        for s in stale:
            print(f"  {s}")
    else:
        print("no superseded numbers found in the prose")

    sys.exit(1 if (bad or stale) else 0)


if __name__ == "__main__":
    main()
