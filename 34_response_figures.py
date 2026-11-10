#!/usr/bin/env python
"""
34_response_figures.py  --  figures for the response half (R3-R6)
==================================================================

Reads the tables written by 22 / 31 / 32 and draws, so a figure can be redrawn
without refitting anything. Numbering follows the agreed structure in
`Results_2019_2022/Response_section_plan.md`:

  Fig 6   the response is heterogeneous              (this file)
  FigS12  the base-temperature scan behind panel e  (from 32)

WHY FIGURE 7 WAS REBUILT
------------------------
The previous Figure 7 was drawn entirely in per-100 K.h units, and its headline
panel read "pooled mean is not distinguishable from zero". That statement is
true only in those units: the base-temperature scan shows the same difference
ranging from -6.66 to +17.32 across eighteen equally conventional base pairs,
significantly negative in two cells and significantly positive in two others.
Under either scale-free unit the answer is stable and signed. A figure whose
title asserts the unit-dependent version of the result would now contradict the
section it illustrates.

What the rebuilt figure shows instead is the finding that survives every unit:
the response is not one number. Cochran's Q rejects a shared slope at
p < 10^-40 and I-squared is 93-96 % in the three interpreted
sectors, against 53-75 % in the two negative controls.

WHAT WAS KEPT FROM THE OLD FIGURE
---------------------------------
Four of the old panels earned their place and are back, re-estimated in per SD:
shrinkage (a), the two sides plotted against each other with the 45-degree line
(b), the per-country difference (c), and the COVID sub-windows (f). The last is
now run for all five sectors rather than for Power alone, by
35_covid_windows.py; a shift confined to the two responsive sectors is a
different fact from a shift shared by the two with no cooling channel.

Two old panels did not come back. The exploding heat/cold ratio is a reason for
choosing the difference, not a result, and lives in the supplementary
diagnostic plate FigS11 written by 22_r5_asymmetry.py. The old sector
comparison is subsumed by panel (e), which shows the same five sectors under
all three units.

USAGE
-----
  python 34_response_figures.py                 # every figure it owns
  python 34_response_figures.py --only 7
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
TIER = "AB"

SECTORS = ["Power", "Industry", "Ground Transport", "Aviation", "Residential"]
UNITS = ["per100", "perSD", "perP95"]
ULAB = {"per100": "per 100 K·h", "perSD": "per SD", "perP95": "per p95-day"}
# The unit R5 now reports in. per-SD rather than per-p95 because a standard
# deviation is the unit R3.1 already uses, so the two sections stay comparable.
MAIN_UNIT = "perSD"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def save(fig, name: str, section: str) -> None:
    """Write png/pdf/svg, but never silently destroy a hand-edited SVG.

    Fig7_thermal_asymmetry.svg has been reworked in Inkscape -- labels reworded,
    font sizes raised, panel f scaled and moved -- and panel f itself was later
    replaced by 48_panel_f_svg_patch.py. None of that can be regenerated from
    here. If the target SVG carries editor metadata, it is left alone and the
    fresh one is written beside it with a _regen suffix.
    """
    out = RESULTS / section / "figures"
    out.mkdir(parents=True, exist_ok=True)

    svg = out / f"{name}.svg"
    edited = svg.exists() and any(
        k in svg.read_text(encoding="utf-8", errors="ignore")[:4000]
        for k in ("inkscape", "sodipodi"))
    if edited:
        # The SVG is the master for this figure and the png/pdf beside it are
        # exports of it, so none of the three may be replaced by a fresh
        # render. Everything goes to _regen for comparison instead.
        print(f"    ! {name}.svg is hand-edited -- {name}.{{png,pdf,svg}} left "
              f"untouched; wrote {name}_regen.* for comparison")

    written = []
    for ext in ("png", "pdf", "svg"):
        target = out / f"{name}{'_regen' if edited else ''}.{ext}"
        fig.savefig(target, bbox_inches="tight", dpi=600)
        written.append(target.suffix.lstrip("."))
    plt.close(fig)
    print(f"    {section}/figures/{name}  [{', '.join(written)}]  600 dpi")


def tab(rel: str) -> pd.DataFrame:
    p = RESULTS / rel
    if not p.exists():
        raise SystemExit(f"{p} not found -- run the upstream script first")
    return pd.read_csv(p)


# --------------------------------------------------------------------------
def fig7() -> None:
    """The response is heterogeneous; heating dominates once units agree.

    Six panels. The unit comparison that used to be panel e moved to Figure 6
    (43_base_convention_figure.py), where it sits beside the base-temperature
    grids that its whiskers summarise. What is left here is one argument in
    two registers: the response is a distribution across countries, it is not
    an artefact of the pandemic (a-e, per SD), and it is large in mass (f).

    Panel f was Supplementary Figure S15a. It belongs with the asymmetry
    rather than with the normalisation figure: the mass numbers are the per-SD
    coefficients rescaled by each country's own sigma and emission level, so
    they inherit the per-SD answer by construction and cannot arbitrate the
    per-100 K.h versus per-SD question that Figure 6 poses. Placing them in
    Figure 6 would have implied an independence they do not have.

    Requires 45_absolute_magnitude.py to have been run (TS21b).
    """
    suf = "" if MAIN_UNIT == "per100" else f"_{MAIN_UNIT}"
    cty = tab(f"R5/tables/T6_thermal_asymmetry_Power_tier{TIER}{suf}.csv")
    sec = {u: tab(f"R5/tables/T7_sector_comparison_tier{TIER}"
                  f"{'' if u == 'per100' else '_' + u}.csv") for u in UNITS}

    covid = tab(f"R5/tables/T8_covid_window_tier{TIER}{suf}.csv")
    # Written by 47_corrected_power_mass.py. Panel f is the only place in the
    # paper where a response is quoted in mass rather than per cent.
    #
    # It used to read TS21b (45_absolute_magnitude.py). Those numbers were
    # withdrawn on audit: Industry was stacked in although its daily profile is
    # downscaled from national electricity activity; the coefficient's "1 SD"
    # was pooled while the day count used a within-country SD, and the two
    # sides were inflated by different factors (x2.16 against x2.89); and the
    # mid scope multiplied a continuous-load coefficient by a percentile
    # event's mean duration. 47 refits R3 and asserts the cumulative beta
    # reproduces T3 to 1e-10 before converting.
    mass = tab("R5/tables/T22b_corrected_power_mass_totals.csv")
    mass_ci = tab("R5/tables/T22d_corrected_power_mass_contrasts.csv")

    both = cty[cty.both.fillna(False)].dropna(subset=["diff_eb_pct"])
    both = both.sort_values("diff_eb_pct")
    y = np.arange(len(both))

    fig = plt.figure(figsize=(16.2, 9.4), layout="constrained")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.92])
    axes = {(0, 0): fig.add_subplot(gs[0, 0:2]),
            (0, 1): fig.add_subplot(gs[0, 2:4]),
            (0, 2): fig.add_subplot(gs[0, 4:6]),
            (1, 0): fig.add_subplot(gs[1, 0:2]),
            (1, 1): fig.add_subplot(gs[1, 2:4])}
    # Panel f is three small axes rather than one: the three scopes differ by
    # three orders of magnitude and carry different units, and they are meant
    # to be read cold-against-heat within each scope, never across scopes. A
    # shared y axis would render the first two invisible.
    gs_f = gs[1, 4:6].subgridspec(1, 3, wspace=0.75)
    axes_f = [fig.add_subplot(gs_f[0, k]) for k in range(3)]

    # ---- (a) how much of each estimate is data, how much is prior --------
    #
    # With 18 countries and four years the per-country slopes are noisy, and a
    # reader is owed a view of how far shrinkage moves them before being shown
    # the shrunk estimates in (b) and (c). Cooling is the side where it matters
    # most: several countries have few genuinely hot days.
    ax = axes[0, 0]
    sh = cty[cty.heat_ok.fillna(False)].dropna(subset=["heat_eb_pct"])
    sh = sh.sort_values("heat_pct")
    ys = np.arange(len(sh))
    for k, r in enumerate(sh.itertuples()):
        ax.plot([r.heat_pct, r.heat_eb_pct], [k, k], color="#c0392b", lw=0.7,
                alpha=0.55, zorder=1)
    ax.scatter(sh.heat_pct, ys, s=18, facecolors="none", edgecolors="#c0392b",
               lw=0.8, zorder=3, label="raw")
    ax.scatter(sh.heat_eb_pct, ys, s=20, color="#c0392b", zorder=3,
               label="shrunk")
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(sh.country, fontsize=6)
    ax.set_xlabel(f"cooling-load response {ULAB[MAIN_UNIT]} (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    med_shrink = float(sh.heat_shrink.median()) if "heat_shrink" in sh else \
        float("nan")
    ax.set_title("a  Raw and empirical-Bayes country slopes, cooling side",
                 loc="left", fontweight="bold", fontsize=9.5)
    ax.text(0.03, 0.97, f"median shrinkage {med_shrink:.2f}\n"
            f"{len(sh)} countries identify a cooling side",
            transform=ax.transAxes, fontsize=6.4, color="0.3", va="top",
            linespacing=1.4)

    # ---- (b) the two sides against each other ----------------------------
    ax = axes[0, 1]
    ax.errorbar(both.cold_eb_pct, both.heat_eb_pct,
                xerr=196 * both.cold_eb_se, yerr=196 * both.heat_eb_se,
                fmt="o", ms=4, lw=0.7, color="#6a3d9a", ecolor="#c8b6d6",
                capsize=0, zorder=2)
    allv = np.concatenate([both.cold_eb_pct.to_numpy(),
                           both.heat_eb_pct.to_numpy()])
    span = float(allv.max() - allv.min())
    lim = [float(allv.min()) - 0.10 * span, float(allv.max()) + 0.10 * span]
    ax.plot(lim, lim, ls="--", color="0.35", lw=0.8, zorder=1)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    cx = float(both.cold_eb_pct.mean())
    cy = float(both.heat_eb_pct.mean())
    for r in both.itertuples():
        dx, dy = r.cold_eb_pct - cx, r.heat_eb_pct - cy
        n = float(np.hypot(dx, dy)) or 1.0
        ax.annotate(r.country, (r.cold_eb_pct, r.heat_eb_pct), fontsize=5.4,
                    xytext=(6.5 * dx / n, 6.5 * dy / n),
                    textcoords="offset points", color="0.2",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"heating-load response {ULAB[MAIN_UNIT]} (%)", fontsize=8)
    ax.set_ylabel(f"cooling-load response {ULAB[MAIN_UNIT]} (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("b  Cooling against heating slope, by country",
                 loc="left", fontweight="bold", fontsize=9.5)
    ax.text(0.97, 0.05, "below the dashed line:\nheating dominates",
            transform=ax.transAxes, fontsize=6.4, color="0.35", ha="right",
            va="bottom", linespacing=1.4)

    # ---- (c) the difference, with the detection floor --------------------
    ax = axes[0, 2]
    # Floor from the two sectors with no cooling channel. Aviation carries a
    # real winter-disruption channel, so it bounds how large a difference can
    # appear without a cooling mechanism behind it.
    floor = max(abs(float(sec[MAIN_UNIT].set_index("sector")
                          .loc[s, "diff"])) for s in
                ("Aviation", "Ground Transport"))
    ax.axvspan(-floor, floor, color="0.88", zorder=0)
    for k, r in enumerate(both.itertuples()):
        sig = (r.diff_eb_lo > 0) or (r.diff_eb_hi < 0)
        c = "#6a3d9a" if sig else "0.6"
        ax.plot([r.diff_eb_lo, r.diff_eb_hi], [k, k], color=c, lw=1.6,
                zorder=2)
        ax.scatter([r.diff_eb_pct], [k], s=26, color=c, zorder=3)
    pooled = float(sec[MAIN_UNIT].set_index("sector").loc["Power", "diff"])
    ax.axvline(pooled, color="#6a3d9a", ls="--", lw=1.1, zorder=1)
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(both.country, fontsize=7)
    ax.set_xlabel(f"cooling − heating {ULAB[MAIN_UNIT]} (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    neg = int((both.diff_eb_hi < 0).sum())
    pos = int((both.diff_eb_lo > 0).sum())
    ax.set_title("c  Cooling − heating difference, empirical-Bayes "
                 "intervals", loc="left", fontweight="bold", fontsize=9.5)
    ax.text(0.03, 0.30,
            f"dashed line: pooled {pooled:+.1f}\n"
            f"grey band: detection floor from the\n"
            f"negative controls (\u00b1{floor:.1f})\n"
            f"heating dominates in {neg} of {len(both)}; "
            f"cooling in {pos}",
            transform=ax.transAxes, fontsize=6.4, color="0.25", va="top")

    # ---- (d) heterogeneity ------------------------------------------------
    ax = axes[1, 0]
    t8 = sec[MAIN_UNIT].set_index("sector")
    keys = [("heat_I2", "cooling", "#c0392b"),
            ("cold_I2", "heating", "#2471a3"),
            ("diff_I2", "difference", "#6a3d9a")]
    w, order = 0.26, [s for s in SECTORS if s in t8.index]
    xs = np.arange(len(order))
    for m, (col, lab, c) in enumerate(keys):
        if col not in t8.columns:
            continue
        ax.bar(xs + (m - 1) * w, t8.loc[order, col], width=w, color=c,
               label=lab)
    ax.axhline(75, color="0.4", lw=0.8, ls=":")
    # In the gap between the two control sectors, the only place above the
    # line that no bar reaches.
    ax.text(2.5, 78, "I² = 75 %", fontsize=6.4, color="0.35", ha="center",
            va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace(" ", "\n") for s in order], fontsize=7)
    ax.set_ylabel("I² (%)", fontsize=8)
    ax.set_ylim(0, 118)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center")
    q = float(t8.loc["Power", "diff_Q"])
    ax.set_title("d  Between-country heterogeneity (I²)",
                 loc="left", fontweight="bold", fontsize=9.5)
    # The exponent used to be hardcoded at 10^-35. It is read from the table
    # now, because the covariance-consistent Q changed it and a hardcoded
    # exponent is exactly the kind of number that goes stale in silence.
    qp = float(t8.loc["Power", "diff_Q_p"])
    # Below the axes rather than inside it: at a third of the figure width the
    # legend already occupies the headroom above the bars.
    ax.text(0.0, -0.145,
            f"Power difference: Cochran Q = {q:.0f} on "
            f"{int(t8.loc['Power', 'diff_Q_df'])} df, p = {qp:.0e}   ·   "
            "GLS form, using the full cross-country covariance",
            transform=ax.transAxes, fontsize=6.6, color="0.32", va="top")

    # ---- (e) is the pandemic driving this? -------------------------------
    #
    # 2019-2022 contains a demand collapse that is not weather. Running the
    # three windows for all five sectors is what makes the answer readable:
    # every sector's difference keeps its sign when 2020 is dropped, and the
    # 2020-only window is simply less precise rather than pointing the other
    # way. Only the difference is drawn -- the two sides separately would be
    # forty-five points in a quarter panel.
    ax = axes[1, 1]
    wins = list(dict.fromkeys(covid.window))
    wcol = {wins[0]: "#6a3d9a", wins[1]: "#2471a3", wins[2]: "0.55"}
    order_f = [s for s in SECTORS if s in set(covid.sector)]
    yb = np.arange(len(order_f))
    for m, w in enumerate(wins):
        d = covid[covid.window == w].set_index("sector")
        vals = [float(d.loc[s, "diff"]) for s in order_f]
        errs = [1.96 * float(d.loc[s, "diff_se"]) for s in order_f]
        yy = yb + (1 - m) * 0.26
        ax.errorbar(vals, yy, xerr=errs, fmt="o", ms=4, lw=1.0,
                    color=wcol[w], ecolor=wcol[w], capsize=0, label=w)
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(yb)
    ax.set_yticklabels(order_f, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"pooled cooling − heating {ULAB[MAIN_UNIT]} (%)",
                  fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left", title=None)
    # State the claim as a count rather than leaving it to the eye.
    full = covid[covid.window == wins[0]].set_index("sector")
    drop = covid[covid.window == wins[1]].set_index("sector")
    same = sum(np.sign(float(full.loc[s, "diff"]))
               == np.sign(float(drop.loc[s, "diff"])) for s in order_f)
    ax.set_title("e  Difference by estimation window",
                 loc="left", fontweight="bold", fontsize=9.5)
    ax.text(0.98, 0.03, f"{same} of {len(order_f)} sectors keep their sign\n"
            "when 2020 is removed",
            transform=ax.transAxes, fontsize=6.4, color="0.3", ha="right",
            va="bottom", linespacing=1.4)

    # ---- (f) the same asymmetry in mass ----------------------------------
    #
    # Every other number in this figure is a percentage, and a percentage does
    # not tell a reader whether the asymmetry is a rounding error or a policy
    # problem. This panel rescales the per-SD coefficients by each country's
    # own exposure SD and its own mean daily sectoral CO2, so it is downstream
    # of panel c rather than independent evidence for it: it answers "how
    # much", not "which way". It is NOT a counterfactual -- the fixed effects
    # absorb the level, so what is priced is the deviation of thermal load
    # from each country-month's own mean.
    DAY = "simultaneous_one_day_pooled_1sd_cumulative_lag_0_14"
    SPELL = "one_typical_observed_high_load_spell_per_country"
    YEAR = "observed_high_load_anomalies_average_annual"
    scopes = [(DAY, "one day, +1 pooled SD\ncumulated over lags 0–14",
               1.0, "kt CO₂", 0),
              (SPELL, "one observed spell per\ncountry, mean 2.6–2.8 d",
               1.0, "kt CO₂", 0),
              (YEAR, "per year, days above one\nwithin-country SD",
               1e-3, "Mt CO₂ yr⁻¹", 2)]
    # `side` in T22b is named for the weather, not for the load: side "cold"
    # carries the HEATING-load coefficient and side "heat" the COOLING-load
    # one. The rest of this figure speaks in loads, so the panel is labelled
    # in loads and coloured to match panel d -- heating blue, cooling red.
    for axf, (scope, lab, k_, unit, dec) in zip(axes_f, scopes):
        t = mass[mass.scope == scope].set_index("side")
        cc = mass_ci[mass_ci.scope == scope].iloc[0]
        sides = (("cold", "#2471a3"), ("heat", "#c0392b"))
        val = np.array([float(t.loc[s, "estimate_kt"]) for s, _ in sides]) * k_
        lo = np.array([float(t.loc[s, "ci95_lo_kt"]) for s, _ in sides]) * k_
        hi = np.array([float(t.loc[s, "ci95_hi_kt"]) for s, _ in sides]) * k_
        axf.bar([0, 1], val, 0.66, color=[c_ for _, c_ in sides],
                edgecolor="white", lw=0.7, zorder=2)
        axf.errorbar([0, 1], val, yerr=np.vstack([val - lo, hi - val]),
                     fmt="none", ecolor="0.15", elinewidth=0.9, capsize=2.2,
                     capthick=0.9, zorder=3)
        for x_, v_, h_, (_, colr) in zip((0, 1), val, hi, sides):
            axf.text(x_, h_ + hi.max() * 0.03, f"{v_:,.{dec}f}", ha="center",
                     va="bottom", fontsize=7.0, fontweight="bold", color=colr)
        # Inside the axes, not above it. Above, the three ratio labels and the
        # panel title share a band that constrained_layout does not reserve --
        # the title rode up into panel c's x label. The spell scope is the
        # honest weak point: its contrast straddles zero, and that is marked
        # on the panel rather than left in the table.
        ns = bool(cc.ci95_lo_kt < 0 < cc.ci95_hi_kt)
        axf.text(0.5, 0.985,
                 f"heating ×{cc.cold_to_heat_ratio:.1f}"
                 + ("\ndifference n.s." if ns else ""),
                 transform=axf.transAxes, ha="center", va="top",
                 fontsize=7.4 if not ns else 6.9, fontweight="bold",
                 color="0.2", linespacing=1.35)
        axf.set_xticks([0, 1])
        axf.set_xticklabels(["heating\nload", "cooling\nload"], fontsize=6.6)
        axf.set_ylabel(unit, fontsize=7.2, labelpad=1.5)
        axf.set_xlabel(lab, fontsize=6.2, labelpad=2)
        axf.tick_params(axis="y", labelsize=6.4)
        axf.tick_params(axis="x", length=0)
        axf.grid(axis="y", lw=0.3, alpha=0.4)
        axf.set_axisbelow(True)
        axf.set_ylim(0, hi.max() * 1.26)
        for s in ("top", "right"):
            axf.spines[s].set_visible(False)
        axf.set_xlim(-0.66, 1.66)
    axes_f[0].set_title("f  Asymmetry in mass of CO₂", loc="left",
                        fontweight="bold", fontsize=9.5, pad=6)
    axes_f[1].text(0.5, -0.36,
                   "Power only; one pooled sensitivity applied to every "
                   "country.\n"
                   "Exact exp(η)−1; 95 % CI. Model-implied, not attributable.",
                   transform=axes_f[1].transAxes, ha="center", va="top",
                   fontsize=6.2, color="0.32", linespacing=1.5)

    fig.suptitle(
        "Country-specific thermal response of Power CO₂, Tier A+B, 2019–2022. "
        f"Panels a–e are in {ULAB[MAIN_UNIT]} of load, which puts the two "
        "sides on a common footing; the unit comparison itself is Figure 6. "
        "Panel f converts the same estimates to mass at three scopes.",
        fontsize=10)
    save(fig, "Fig7_thermal_asymmetry", "R5")

    for scope, _, k_, unit, _ in scopes:
        cc = mass_ci[mass_ci.scope == scope].iloc[0]
        ns = "  (n.s.)" if cc.ci95_lo_kt < 0 < cc.ci95_hi_kt else ""
        print(f"      mass, {scope[:34]:36s} heating/cooling = "
              f"×{cc.cold_to_heat_ratio:.2f}{ns}")

    print(f"      {neg} of {len(both)} countries: heating > cooling; "
          f"{pos} the other way")
    print(f"      pooled {pooled:+.2f}, detection floor ±{floor:.2f}, "
          f"Power difference I² = {t8.loc['Power', 'diff_I2']:.1f} %")
    print(f"      COVID windows: {same} of {len(order_f)} sectors keep their "
          f"sign when 2020 is dropped")


# --------------------------------------------------------------------------
def figS14() -> None:
    """Aggregation weight does not change the answer.

    This is what survives of the retired R6. The section was planned as the
    explanatory one -- use built-environment variables to say why the response
    is larger in some countries -- and neither of the two designs that could
    have supported that survived scrutiny (see SI.7). What does survive is the
    negative result underneath: five ways of weighting a country's cities give
    the same national exposure series and the same coefficient. That defends
    the population weighting used throughout the paper, and it is the direct
    answer to the obvious question of whether the weighting drives the result.

    It sits beside Figure 4 and Figure 6 as the third member of a set: metric
    choice matters enormously, unit choice flips a sign, and weight choice does
    nothing. A paper that found every choice mattered would be reporting noise.
    """
    t = tab("SI/tables/TS13_weight_race_tierAB.csv")
    order = ["equal", "population", "built surface", "built volume",
             "pop x volume intensity"]
    t = t.set_index("scheme").reindex([o for o in order if o in set(t.scheme)]
                                      if "scheme" in t.columns else order)
    t = t.reset_index()
    lab = [x.replace(" x ", " × ") for x in t.scheme]
    y = np.arange(len(t))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6), layout="constrained")

    # ---- (a) how different are the national series at all? ---------------
    ax = axes[0]
    ax.barh(y, t.corr_with_pop, height=0.6, color="#2c3e50", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.90, 1.005)
    ax.axvline(0.99, color="#c0392b", lw=0.8, ls=":")
    for k, r in enumerate(t.itertuples()):
        ax.text(r.corr_with_pop - 0.002, k,
                f"mean |Δ| {r.mean_abs_diff_Kh:.2f} K·h", ha="right",
                va="center", fontsize=6.4, color="white")
    ax.set_xlabel("correlation with the population-weighted national series",
                  fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("a  Agreement of the five weighted exposure series",
                 loc="left", fontweight="bold", fontsize=9.5)

    # ---- (b) and does the coefficient move? ------------------------------
    ax = axes[1]
    se = t.pct_per_sd / t.t
    ax.errorbar(t.pct_per_sd, y, xerr=1.96 * se, fmt="o", ms=5, lw=1.3,
                color="#2471a3", ecolor="#2471a3", capsize=0)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.4", lw=0.8)
    for k, r in enumerate(t.itertuples()):
        ax.text(r.pct_per_sd + 1.96 * (r.pct_per_sd / r.t) + 0.06, k,
                f"t = {r.t:.2f}", va="center", fontsize=6.6, color="0.3")
    ax.set_xlabel("cumulative power CO₂ response per SD of exposure (%), "
                  "Tier A+B", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("b  Cumulative response under each weighting", loc="left",
                 fontweight="bold", fontsize=9.5)
    lo, hi = float(t.pct_per_sd.min()), float(t.pct_per_sd.max())
    ax.text(0.02, 0.06,
            f"the five point estimates span {hi - lo:.2f} percentage points "
            f"({lo:.2f} to {hi:.2f});\nout-of-sample R² differs in the fourth "
            "decimal place",
            transform=ax.transAxes, fontsize=6.4, color="0.3", va="bottom",
            linespacing=1.4)

    fig.suptitle("Aggregation weight does not change the national exposure "
                 "series or the response it produces", fontsize=10)
    save(fig, "FigS14_weight_invariance", "SI")
    print(f"      span {hi - lo:.2f} pp; min corr with population "
          f"{t.corr_with_pop.min():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", type=int, default=[7, 14])
    args = ap.parse_args()
    print("=" * 78)
    print("Response-half figures")
    print("=" * 78)
    if 7 in args.only:
        print("\n[Fig 6]")
        fig7()
    if 14 in args.only:
        print("\n[Fig S14]")
        figS14()


if __name__ == "__main__":
    main()
