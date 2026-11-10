#!/usr/bin/env python
"""
43_base_convention_figure.py  --  the base-temperature result, redrawn
======================================================================

FigS12 presents the base-temperature scan as fifteen line panels (3 units x 5
sectors), and the chart type fights the message twice:

  * a line plot's most salient feature is its slope, and here the slope IS the
    artefact the reader is meant to discount;
  * under the scale-free units the lines are flat, which reads visually as
    "nothing here" when flatness is precisely the result.

It also never shows the mechanism. The scan moves the difference by rescaling
one regressor, and the giveaway is that the coefficient triples while its z
statistic does not move. FigS12 plots only the difference, so that is invisible.

This rebuild is three panels, Power only (the five-sector version stays in the
supplement):

  a  the mechanism -- cooling and heating coefficients against the cooling
     base on the left axis, their z statistics on the right axis. The
     coefficients diverge; the z statistics are flat. Rescaling a regressor
     changes its coefficient and not its evidence.

  b  per 100 K.h -- the 6 x 3 base-pair grid as a heat map, with the ZERO
     CONTOUR drawn. The contour crosses the grid: the sign of the published
     result is set by where the convention happens to sit.

  c  per SD -- the identical grid. No zero contour exists to draw.

  d  the cross-sector summary, moved here from the old Fig. 6e. The pooled
     difference for every sector under all three normalisations, each bar
     carrying the range over the eighteen base pairs. It generalises a-c
     beyond Power and beyond two units, and its whiskers are now checkable:
     for Power they are exactly the spread laid out cell by cell in b and c.

Reads T7_sector_comparison_tierAB*.csv and T8b_base_scan_tierAB.csv; no model
is refitted.

USAGE
-----
  python 43_base_convention_figure.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# The first SVG of this figure opened in Illustrator as ~1,700 glyph outlines
# with the two heat maps embedded as PNGs. Two settings and one drawing change
# fix that:
#   svg.fonttype "none"  -> real <text> elements, so the labels stay editable
#                           text rather than becoming paths;
#   pdf/ps fonttype 42   -> TrueType embedding, editable in the PDF too;
#   Rectangle patches    -> the grids are 18 vector cells each, not a raster
#                           QuadMesh (see panel_grid).
matplotlib.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})
import matplotlib.colors as mcolors                        # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
SCAN = RESULTS / "R5" / "tables" / "T8b_base_scan_tierAB.csv"
OUT = RESULTS / "R5" / "figures"
NAME = "Fig6_unit_convention"

UNITS = ["per100", "perSD", "perP95"]
ULAB = {"per100": "per 100 K·h", "perSD": "per SD",
        "perP95": "per p95-day"}
SECTORS = ["Power", "Industry", "Ground Transport", "Aviation"]

SECTOR = "Power"
PUB_COOL, PUB_HEAT = 24.0, 15.0
COOL_C, HEAT_C = "#C0392B", "#1F6FB2"


# --------------------------------------------------------------------------
def panel_mechanism(ax, d):
    """Coefficients move, evidence does not."""
    s = d[(d.normalisation == "per100") & (d.heat_base_c == PUB_HEAT)]
    s = s.sort_values("cool_base_c")
    x = s.cool_base_c.to_numpy()

    ax.plot(x, s.cooling, "o-", color=COOL_C, lw=2.0, ms=6,
            label="cooling coefficient", zorder=4)
    ax.plot(x, s.heating, "s-", color=HEAT_C, lw=2.0, ms=6,
            label="heating coefficient", zorder=4)
    ax.set_ylabel("cumulative response  (% per 100 K·h)", fontsize=9)
    ax.set_xlabel("cooling base temperature (°C)", fontsize=9)
    ax.set_ylim(0, 27)

    az = ax.twinx()
    az.plot(x, s.cooling_z, "o--", color=COOL_C, lw=1.5, ms=4.5, alpha=0.55,
            label="cooling z", zorder=3)
    az.plot(x, s.heating_z, "s--", color=HEAT_C, lw=1.5, ms=4.5, alpha=0.55,
            label="heating z", zorder=3)
    az.axhline(1.96, color="0.55", lw=0.8, ls=":")
    az.text(x[0] - 0.15, 1.96, "z = 1.96 ", fontsize=6.8, color="0.45",
            va="bottom", ha="left")
    az.set_ylabel("z statistic  (dashed)", fontsize=9, color="0.35")
    az.set_ylim(0, 9.2)
    az.tick_params(labelsize=7.5, colors="0.35")
    az.spines["top"].set_visible(False)

    # the two numbers that carry the panel
    lo, hi = s.cooling.iloc[0], s.cooling.iloc[-1]
    zl, zh = s.cooling_z.iloc[0], s.cooling_z.iloc[-1]
    ax.annotate(f"cooling coefficient ×{hi / lo:.1f}  ({lo:.1f} → {hi:.1f})",
                xy=(x[-1], hi), xytext=(-6, 9), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.6, color=COOL_C,
                fontweight="bold")
    ax.text(0.30, 0.055,
            f"the cooling z barely moves: {zl:.2f} → {zh:.2f}   ·   "
            f"rescaling a regressor changes its coefficient, not its evidence",
            transform=ax.transAxes, fontsize=8.2, color="0.2",
            fontweight="bold")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = az.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.0, frameon=False, ncol=4,
              loc="upper left", handletextpad=0.5, columnspacing=1.6)
    ax.set_title("a  Why the difference moves: the cooling regressor is being "
                 "rescaled", loc="left", fontweight="bold", fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.tick_params(labelsize=7.5)
    ax.grid(lw=0.4, alpha=0.4)
    ax.spines[["top"]].set_visible(False)
    ax.set_zorder(az.get_zorder() + 1)
    ax.patch.set_visible(False)


def panel_grid(ax, d, norm_key, letter, title, unit_lab, cbar=False):
    """The 6 x 3 base-pair grid as a heat map, with the zero contour."""
    s = d[d.normalisation == norm_key]
    piv = s.pivot(index="heat_base_c", columns="cool_base_c",
                  values="difference").sort_index()
    zed = s.pivot(index="heat_base_c", columns="cool_base_c",
                  values="diff_z").sort_index()
    M, Z = piv.to_numpy(), zed.to_numpy()
    xs, ys = piv.columns.to_numpy(float), piv.index.to_numpy(float)

    lim = float(np.nanmax(np.abs(M)))
    norm = mcolors.Normalize(-lim, lim)
    cmap = plt.get_cmap("RdBu_r")
    # One Rectangle per cell keeps every element vector and individually
    # selectable. pcolormesh would have exported the whole grid as a raster.
    for i in range(len(ys)):
        for j in range(len(xs)):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(norm(M[i, j])),
                                       edgecolor="white", lw=0.8, zorder=1))
    ax.set_xlim(0, len(xs))
    im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    im.set_array(M)

    # the zero contour: the whole point of the panel
    xc, yc = np.arange(len(xs)) + 0.5, np.arange(len(ys)) + 0.5
    crosses = np.nanmin(M) < 0 < np.nanmax(M)
    if crosses:
        ax.contour(xc, yc, M, levels=[0.0], colors="k", linewidths=2.4,
                   zorder=5)
        ax.text(0.04, len(ys) + 0.16, "— the sign flips here",
                fontsize=7.4, fontweight="bold", color="#111111",
                va="bottom", ha="left")

    for i in range(len(ys)):
        for j in range(len(xs)):
            sig = abs(Z[i, j]) > 1.96
            ax.text(j + 0.5, i + 0.5, f"{M[i, j]:+.1f}", ha="center",
                    va="center", fontsize=8.4,
                    fontweight="bold" if sig else "normal",
                    color="white" if abs(M[i, j]) > 0.62 * lim else "0.15")
            if sig:
                ax.add_patch(plt.Rectangle((j + 0.06, i + 0.06), 0.88, 0.88,
                                           fill=False, ec="0.15", lw=1.0,
                                           zorder=4))
    # the published convention
    jp = int(np.where(xs == PUB_COOL)[0][0])
    ip = int(np.where(ys == PUB_HEAT)[0][0])
    ax.add_patch(plt.Rectangle((jp, ip), 1, 1, fill=False, ec="#111111",
                               lw=2.6, zorder=6))
    ax.text(len(xs) - 0.04, len(ys) + 0.16, "bold = published pair",
            ha="right", va="bottom", fontsize=7.4, fontweight="bold",
            color="#111111")

    ax.set_xticks(np.arange(len(xs)) + 0.5)
    ax.set_xticklabels([f"{v:.0f}" for v in xs], fontsize=7.5)
    ax.set_yticks(np.arange(len(ys)) + 0.5)
    ax.set_yticklabels([f"{v:.0f}" for v in ys], fontsize=7.5)
    ax.set_xlabel("cooling base (°C)", fontsize=8.5, labelpad=1)
    ax.set_yticks(np.arange(len(ys)) + 0.5)
    ax.set_ylabel("heating base (°C)", fontsize=8.5)
    ax.set_ylim(-1.55, len(ys) + 0.70)
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold",
                 fontsize=9.4, pad=5)

    n_neg = int(((M < 0) & (np.abs(Z) > 1.96)).sum())
    n_pos = int(((M > 0) & (np.abs(Z) > 1.96)).sum())
    ax.text(0.0, -0.70,
            f"range {np.nanmin(M):+.2f} to {np.nanmax(M):+.2f} {unit_lab}\n"
            f"{n_neg} significantly negative, {n_pos} positive"
            + ("\nno zero contour exists" if not crosses else ""),
            ha="left", va="top", fontsize=7.8, color="0.2",
            fontweight="bold", linespacing=1.5)
    if cbar:
        cax = ax.inset_axes([1.018, 0.0, 0.022, 1.0])
        cb = ax.figure.colorbar(im, cax=cax)
        cb.solids.set_rasterized(False)      # keep the gradient vector too
        cb.set_label("difference", fontsize=7.2)
        cb.ax.tick_params(labelsize=6.8, length=2)
        cb.outline.set_linewidth(0.5)
    return im


def panel_sectors(ax, scan):
    """The old Fig. 6e: pooled difference by sector under all three units.

    Residential is left out. Its allocation is built from heating degree days,
    so it is circular, and its per-p95 difference of about -78 would compress
    every other sector into invisibility.
    """
    ucol = {"per100": "0.45", "perSD": "#6a3d9a", "perP95": "#1a7a5e"}
    sec = {u: pd.read_csv(
        RESULTS / f"R5/tables/T7_sector_comparison_tier{'AB'}"
        f"{'' if u == 'per100' else '_' + u}.csv") for u in UNITS}
    order = [x for x in SECTORS if x in set(sec["per100"].sector)]

    for m, u in enumerate(UNITS):
        t = sec[u].set_index("sector")
        vals = [float(t.loc[x, "diff"]) for x in order]
        zs = [float(t.loc[x, "diff_z"]) for x in order]
        yy = np.arange(len(order)) + (m - 1) * 0.24
        ax.barh(yy, vals, height=0.22, color=ucol[u], label=ULAB[u])
        for v, ypos, x_ in zip(vals, yy, order):
            g = scan[(scan.sector == x_) & (scan.normalisation == u)]
            if not len(g):
                continue
            lo, hi = float(g.difference.min()), float(g.difference.max())
            ax.plot([lo, hi], [ypos, ypos], color="0.15", lw=0.9, zorder=4,
                    solid_capstyle="butt")
            for xe in (lo, hi):
                ax.plot([xe, xe], [ypos - 0.075, ypos + 0.075], color="0.15",
                        lw=0.9, zorder=4)
        for v, z, ypos in zip(vals, zs, yy):
            if abs(z) > 1.96:
                ax.text(v + (1.2 if v >= 0 else -1.2), ypos, "*", fontsize=9,
                        color=ucol[u], va="center",
                        ha="left" if v >= 0 else "right")

    ax.axvline(0, color="0.35", lw=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("pooled cooling − heating (%)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.4, frameon=False, loc="lower left")
    ax.set_title("d  The same contrast across sectors and all three units",
                 loc="left", fontweight="bold", fontsize=9.4, pad=5)
    ax.text(0.0, -0.175,
            "* |z| > 1.96   ·   whisker: range over the 18 base pairs;\n"
            "for Power these are the cells of b and c\n"
            "Residential omitted (circular allocation)",
            transform=ax.transAxes, fontsize=7.0, color="0.32", ha="left",
            va="top", linespacing=1.45)


# --------------------------------------------------------------------------
def main() -> None:
    d = pd.read_csv(SCAN)
    d = d[d.sector == SECTOR]

    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.70, 1.0],
                          width_ratios=[1.0, 1.0, 1.06],
                          left=0.052, right=0.955, top=0.895, bottom=0.105,
                          wspace=0.46, hspace=0.40)

    panel_mechanism(fig.add_subplot(gs[0, :]), d)
    panel_grid(fig.add_subplot(gs[1, 0]), d, "per100", "b",
               "Per 100 K·h: the convention decides the sign",
               "%/100 K·h", cbar=True)
    panel_grid(fig.add_subplot(gs[1, 1]), d, "perSD", "c",
               "Per SD: the same 18 fits, one answer", "% per SD", cbar=True)
    panel_sectors(fig.add_subplot(gs[1, 2]), d)

    fig.suptitle(
        "The cooling-minus-heating difference in power-sector CO₂ under "
        "eighteen equally conventional base-temperature pairs\n"
        "Tier A+B, 28 countries, 2019–2022; degree-hours re-integrated from "
        "hourly city temperature at every pair, sample and country set held "
        "fixed. Bold cells and outlines are |z| > 1.96.",
        fontsize=9.6)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{NAME}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUT / NAME}.{{png,pdf,svg}}")

    for k in ("per100", "perSD", "perP95"):
        s = d[d.normalisation == k]
        print(f"    {k:7s} range {s.difference.min():+7.2f} .. "
              f"{s.difference.max():+7.2f}   sig-neg "
              f"{int(((s.difference < 0) & (s.diff_z.abs() > 1.96)).sum()):2d}"
              f"   sig-pos "
              f"{int(((s.difference > 0) & (s.diff_z.abs() > 1.96)).sum()):2d}")


if __name__ == "__main__":
    main()
