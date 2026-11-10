#!/usr/bin/env python
"""
42_threshold_ladder_figure.py  --  candidate main-text figure from FigS5
========================================================================

FigS5 is a nine-panel atlas of absolute physiological grades. Six of its
panels carry an argument the main text needs and three (the cold row) do not:
the paper's cold-side story is heating degree-hours, not UTCI cold stress.

This script rebuilds the useful part as an argument rather than an atlas:

    a-f   the two heat-side threshold ladders, mapped on the Figure 1 rules
          UTCI  > 26 / 32 / 38 C        (a, b, c)
          WBGT  > 28 / 32 C, Twb > 28 C (d, e, f)

    g-h   what happens as the metric becomes rarer
          g  |r(|lat|)|  -- how much latitudinal gradient survives
          h  CV          -- across-city dispersion

The x axis is the share of cities where the metric ever occurs, NOT the
threshold in degrees. That choice is deliberate: 28 C of WBGT and 28 C of UTCI
are not the same physiological state, so plotting against degrees invites a
cross-family comparison that means nothing. Rarity is comparable across every
index and across both tails, which lets the cold grades join the same axis
instead of being quoted in a footnote.

The claim the bottom row makes, and that nine maps only imply: past some
strictness an absolute threshold stops describing a climate gradient and
becomes an indicator for a small, geographically clustered set of cities.
That is the exposure-side counterpart of the Aviation negative control
breaking at 27-28 C in the wet-bulb threshold scan (FigS2), and it is visible
here with no emissions data at all.

Diagnostics in g-i are computed by the same `panel_stats` used for the map
annotations, so the lines and the numbers printed on the maps cannot diverge.

USAGE
-----
  python 42_threshold_ladder_figure.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors                        # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
OUT = RESULTS / "R1" / "figures"
NAME = "FigX_absolute_threshold_ladder"

# 19_results_figures.py cannot be imported by name (leading digit)
spec = importlib.util.spec_from_file_location(
    "rf", BASE / "19_results_figures.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)

# --------------------------------------------------------------------------
# the six maps: two heat-side ladders
# (column, title, cmap, (q_lo, q_hi), unit, family)
# --------------------------------------------------------------------------
PANELS = [
    ("n_utci_moderate_days",   "UTCI > 26 °C  (moderate heat stress)",
     "YlOrRd", (0.0, 0.98), "days yr$^{-1}$", "absolute"),
    ("n_utci_strong_days",     "UTCI > 32 °C  (strong)",
     "YlOrRd", (0.0, 0.98), "days yr$^{-1}$", "absolute"),
    ("n_utci_verystrong_days", "UTCI > 38 °C  (very strong)",
     "Reds",   (0.0, 0.98), "days yr$^{-1}$", "absolute"),
    ("n_twbg_danger_days",     "WBGT > 28 °C  (ISO 7243 danger)",
     "YlOrBr", (0.0, 0.98), "days yr$^{-1}$", "absolute"),
    ("n_twbg_highrisk_days",   "WBGT > 32 °C  (high risk)",
     "OrRd",   (0.0, 0.99), "days yr$^{-1}$", "absolute"),
    ("n_twb_stress_days",      "T$_{wb}$ > 28 °C",
     "GnBu",   (0.0, 0.99), "days yr$^{-1}$", "absolute"),
]

# which panels form a ladder, and at what threshold
LADDERS = [
    ("UTCI heat  26 → 38 °C", "#C0392B", "o", "-",
     [("n_utci_moderate_days", "26"), ("n_utci_strong_days", "32"),
      ("n_utci_verystrong_days", "38")]),
    ("WBGT  28 → 32 °C", "#B9770E", "s", "-",
     [("n_twbg_danger_days", "28"), ("n_twbg_highrisk_days", "32")]),
    ("T$_{wb}$ > 28 °C", "#1F6FB2", "^", "-",
     [("n_twb_stress_days", "28")]),
    # the cold grades are mapped in Fig. S5, not here, but rarity is a
    # comparable axis so they belong on the same plot: the collapse is not a
    # heat-side artefact.
    ("UTCI cold  0 → −13 °C, wind chill", "#5D6D7E", "D", "--",
     [("n_utci_coldstress_days", "0"), ("n_utci_strongcold_days", "−13"),
      ("n_twc_danger_days", "wc −10")]),
]

COLD = [("n_utci_coldstress_days", "UTCI < 0 °C"),
        ("n_utci_strongcold_days", "UTCI < −13 °C"),
        ("n_twc_danger_days", "wind chill < −10 °C")]


# --------------------------------------------------------------------------
def draw_map(ax, d, spec, letter, stats):
    """One city point map, on the conventions of Figure 1."""
    col, ptitle, cmap, (qlo, qhi), unit, fam = spec
    rf.world_outline(ax)
    ax.set_aspect("equal", adjustable="box")

    x = d[col].astype(float)
    bad = x.isna()
    if bad.any():
        ax.scatter(d.lon[bad], d.lat[bad], s=3.0, facecolors="none",
                   edgecolors="0.75", lw=0.25, rasterized=True, zorder=2)
    lo, hi = float(x.quantile(qlo)), float(x.quantile(qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x)) or 1.0
        if hi <= lo:
            hi = lo + 1.0

    norm = mcolors.PowerNorm(gamma=0.5, vmin=lo, vmax=hi)
    zero = (x <= 0).fillna(False)
    if zero.any():
        ax.scatter(d.lon[zero], d.lat[zero], s=2.4, color="0.82", lw=0,
                   rasterized=True, zorder=2)
    show = ~zero & ~bad
    sc = ax.scatter(d.lon[show], d.lat[show], c=x[show], s=3.8, cmap=cmap,
                    norm=norm, lw=0, alpha=0.92, rasterized=True, zorder=3)
    ax.set_title(f"{letter}  {ptitle}", loc="left", fontweight="bold",
                 fontsize=8.2, pad=3)

    cax = ax.inset_axes([1.025, 0.0, 0.028, 1.0])
    cb = ax.figure.colorbar(sc, cax=cax, extend="max")
    cb.ax.tick_params(labelsize=5.6, length=1.8, pad=1.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label(unit + r"  ($\sqrt{\,}$)", fontsize=5.8)

    s = stats.loc[stats.variable == col].iloc[0]
    ax.text(0.01, 0.115, f"r(|lat|) = {s.r_abs_lat:+.2f}",
            transform=ax.transAxes, fontsize=6.4, color="0.12",
            fontweight="bold")
    ax.text(0.01, 0.035,
            f"CV = {s.cv:.2f}   {s.n_cities_nonzero:,}/{len(d):,} > 0",
            transform=ax.transAxes, fontsize=5.9, color="0.38")


def rarity_panel(ax, stats, field, letter, title, ylab, n_city,
                 log=False, note_at="left"):
    """One diagnostic against RARITY, one line per criterion family.

    x is the share of cities where the metric ever occurs, on a reversed log
    axis so that moving right means tightening the threshold.
    """
    pts = []
    for lab, colr, mk, ls, items in LADDERS:
        xs, ys, tags = [], [], []
        for var, tag in items:
            r = stats.loc[stats.variable == var].iloc[0]
            v = float(r[field])
            xs.append(100 * r.n_cities_nonzero / n_city)
            ys.append(abs(v) if field == "r_abs_lat" else v)
            tags.append(tag)
        ax.plot(xs, ys, marker=mk, color=colr, lw=1.7, ms=6.2, ls=ls,
                label=lab, zorder=3, clip_on=False,
                alpha=0.95 if ls == "-" else 0.75)
        for xx, yy, tg in zip(xs, ys, tags):
            ax.annotate(tg, (xx, yy), textcoords="offset points",
                        xytext=(0, -11), ha="center", fontsize=6.2,
                        color=colr, fontweight="bold")
        pts += list(zip(xs, ys))

    # one number for the whole panel: does rarity order the diagnostic?
    xs, ys = zip(*pts)
    rho = stats_spearman(xs, ys)
    ax.text(0.035 if note_at == "left" else 0.975, 0.055,
            f"Spearman ρ across all {len(pts)} absolute grades = {rho:+.2f}",
            transform=ax.transAxes, fontsize=7.2, color="0.2",
            fontweight="bold", ha="left" if note_at == "left" else "right")

    ax.set_xscale("log")
    ax.set_xlim(126, 9)                       # reversed: stricter to the right
    ax.set_xticks([100, 50, 25, 12])
    ax.set_xticklabels(["100%", "50%", "25%", "12%"])
    ax.minorticks_off()
    if log:
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 5])
        ax.set_yticklabels(["0.5", "1", "2", "5"])
    ax.set_xlabel("cities where the metric ever occurs  "
                  "(% of 1,583)   →  stricter", fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold",
                 fontsize=8.2, pad=3)
    ax.tick_params(labelsize=7)
    ax.grid(lw=0.4, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def stats_spearman(x, y) -> float:
    from scipy import stats as st
    return float(st.spearmanr(x, y).statistic)


# --------------------------------------------------------------------------
def main() -> None:
    d = rf.city_metrics()
    n_city = len(d)
    stats = rf.panel_stats(d, PANELS + [
        (c, lab, "Blues", (0.0, 0.98), "days yr$^{-1}$", "absolute")
        for c, lab in COLD], NAME)

    fig = plt.figure(figsize=(11.6, 6.5), layout="constrained")
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.75])

    for k, spec in enumerate(PANELS):
        ax = fig.add_subplot(gs[k // 3, 2 * (k % 3):2 * (k % 3) + 2])
        draw_map(ax, d, spec, chr(97 + k), stats)

    axg = fig.add_subplot(gs[2, 0:3])
    rarity_panel(axg, stats, "r_abs_lat", "g",
                 "Latitudinal gradient retained", "|r(|lat|)|", n_city)
    axh = fig.add_subplot(gs[2, 3:6])
    rarity_panel(axh, stats, "cv", "h", "Across-city dispersion",
                 "coefficient of variation", n_city, log=True,
                 note_at="right")
    axh.legend(fontsize=7.2, frameon=False, loc="upper left",
               title="criterion family  (point labels are thresholds, °C)",
               title_fontsize=7.2, handletextpad=0.5, labelspacing=0.35)

    fig.text(0.5, -0.014,
             "Panels g–h plot rarity, not degrees: 28 °C of WBGT and 28 °C of "
             "UTCI are not the same physiological state, so a common "
             "temperature axis would invite a comparison that means nothing. "
             "Cold grades (dashed) are mapped in Fig. S5.",
             ha="center", fontsize=6.9, color="0.32")

    fig.suptitle(
        "Tightening an absolute threshold trades climatological reach for "
        "physiological severity\n"
        f"{n_city:,} functional urban areas, 46 countries, 2019–2022, "
        "annual means; conventions as in Figure 1",
        fontsize=10.2)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{NAME}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUT / NAME}.{{png,pdf,svg}}")

    keep = [p[0] for p in PANELS] + [c for c, _ in COLD]
    tab = stats[stats.variable.isin(keep)][
        ["panel", "variable", "label", "r_abs_lat", "cv",
         "n_cities_nonzero"]]
    print("\n" + tab.to_string(index=False))


if __name__ == "__main__":
    main()
