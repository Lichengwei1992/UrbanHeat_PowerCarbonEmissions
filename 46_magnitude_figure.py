#!/usr/bin/env python
"""
46_magnitude_figure.py  --  SUPERSEDED, kept for provenance
===========================================================

*** THIS SCRIPT IS NO LONGER PART OF THE FIGURE SET. Do not run it as part of
*** a rebuild; its output, Supplementary Figure S15, has been withdrawn.
***
*** Panel a became Figure 7f, drawn by 34_response_figures.py. It belongs with
*** the asymmetry rather than in a supplement of its own: the mass numbers are
*** the per-SD coefficients rescaled by each country's sigma and emission
*** level, so they restate panel 7c in another unit.
***
*** Panel b was dropped. Its claim -- that all 28 countries carry a larger
*** heating-side than cooling-side mass -- survives as a sentence in the
*** Results and the Figure 7 caption, sourced to Table TS21, and is still
*** enforced by an assertion in 30_verify_narrative.py.
***
*** TS21 / TS21b, written by 45_absolute_magnitude.py, remain live: Figure 7f
*** and the verifier both read them. Only this drawing step is retired.

Original rationale, for the record
----------------------------------
The mass result added to Section 5 has no figure. Two things need showing that
a table does not carry well:

  a  the asymmetry survives the change of units. Cold against heat, in three
     scopes (a 1-SD day, a mean-duration spell, a year), for Power and
     Industry. The point is that the cold side is the larger one in mass as
     well as per SD -- a factor of 4.5 a year.

  b  it is not one country. The per-country contribution, cold against heat,
     with the diagonal drawn. If the aggregate rested on China or the United
     States alone the cloud would be a single point far from the rest.

Reads TS21 / TS21b, written by 45_absolute_magnitude.py. No model is refitted.

USAGE
-----
  python 46_magnitude_figure.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
TAB = RESULTS / "SI" / "tables"
OUT = RESULTS / "SI" / "figures"
NAME = "FigS15_absolute_magnitude"

COLD, HEAT = "#1F6FB2", "#C0392B"


def panel_scopes(axes, tot):
    """Cold against heat at three scopes.

    Each scope gets its own axes and its own unit. Sharing one y axis across
    kt/day and Mt/yr made the last two scopes invisible -- the quantities
    differ by three orders of magnitude and are not meant to be compared with
    one another, only cold against heat within each.
    """
    scopes = [("kt_per_1sd_day", "a day 1 SD above\nthe country-month norm",
               1.0, "kt CO₂", "{:,.0f}"),
              ("kt_per_event", "one spell of\nmean duration",
               1e-3, "Mt CO₂", "{:.2f}"),
              ("mt_per_year", "per year, summed over\nall days above 1 SD",
               1.0, "Mt CO₂ yr⁻¹", "{:.1f}")]

    for ax, (col, lab, k_, unit, fmt) in zip(axes, scopes):
        for m, (side, colr) in enumerate((("cold", COLD), ("heat", HEAT))):
            t = tot[tot.side == side].set_index("sector")
            base = t.loc["Power", col] * k_
            ax.bar(m, base, 0.74, color=colr, edgecolor="white", lw=0.8,
                   label="Power" if m == 0 else None)
            ax.bar(m, t.loc["Industry", col] * k_, 0.74, bottom=base,
                   color=colr, alpha=0.45, edgecolor="white", lw=0.8,
                   label="Industry" if m == 0 else None)
            tot_v = base + t.loc["Industry", col] * k_
            ax.text(m, tot_v, fmt.format(tot_v), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=colr)

        r = (tot[tot.side == "cold"][col].sum()
             / tot[tot.side == "heat"][col].sum())
        ax.text(0.5, 1.005, f"cold is ×{r:.1f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9.4,
                fontweight="bold", color="0.2")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["cold", "heat"], fontsize=8.8)
        ax.set_ylabel(unit, fontsize=8.4)
        ax.set_xlabel(lab, fontsize=8.2, labelpad=2)
        ax.tick_params(axis="y", labelsize=7.6)
        ax.grid(axis="y", lw=0.4, alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.26)
        ax.set_xlim(-0.62, 1.62)   # after margins: it autoscales x

    axes[-1].legend(fontsize=7.6, frameon=False, loc="upper right",
                    handlelength=1.1, title="stacked", title_fontsize=7.2)
    axes[0].set_title("a  The asymmetry in mass, at three scopes",
                      loc="left", fontweight="bold", fontsize=9.6, pad=16)


def panel_countries(ax, d):
    """Per-country Mt/yr, cold against heat, Power + Industry."""
    g = d.groupby(["country", "side"]).mt_per_year.sum().unstack("side")
    g = g.dropna()
    hi = float(np.nanmax(g.values)) * 1.12
    ax.plot([0, hi], [0, hi], ls="--", lw=1.0, color="0.55", zorder=1)
    ax.scatter(g.heat, g.cold, s=34, color="#34495e", alpha=0.85, lw=0,
               zorder=3)
    for c in g.index:
        if g.loc[c].max() > 2.0:   # the cluster below 2 Mt is unlabellable
            ax.annotate(c[:13], (g.loc[c, "heat"], g.loc[c, "cold"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=7.2, color="0.2")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("heat side  (Mt CO₂ yr⁻¹)", fontsize=8.6)
    ax.set_ylabel("cold side  (Mt CO₂ yr⁻¹)", fontsize=8.6)
    ax.set_title("b  Every country sits above the diagonal",
                 loc="left", fontweight="bold", fontsize=9.6, pad=6)

    above = int((g.cold > g.heat).sum())
    top3 = g.sum(axis=1).nlargest(3).sum() / g.values.sum()
    ax.text(0.97, 0.06,
            f"{above} of {len(g)} countries above the diagonal\n"
            f"the three largest carry {100 * top3:.0f} % of the total",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            color="0.2", fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(lw=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    d = pd.read_csv(TAB / "TS21_absolute_magnitude_by_country.csv")
    tot = pd.read_csv(TAB / "TS21b_absolute_magnitude_totals.csv")

    # Explicit layout. constrained_layout collapsed the three nested scope
    # axes to 6 px wide -- it cannot solve a nested subgridspec next to an
    # aspect-locked scatter.
    fig = plt.figure(figsize=(13.4, 4.8))
    scope_ax = [fig.add_axes([0.055 + k * 0.155, 0.20, 0.105, 0.60])
                for k in range(3)]
    panel_scopes(scope_ax, tot)
    panel_countries(fig.add_axes([0.585, 0.145, 0.335, 0.700]), d)
    fig.suptitle(
        "CO₂ associated with the deviation of thermal load from each "
        "country-month's own mean\n"
        "Power and industry, 28 Tier A+B countries, 1,583 sampled urban areas, "
        "2019–2022. Not a counterfactual without thermal extremes.",
        fontsize=9.6, y=0.995)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{NAME}.{ext}", dpi=400)
    plt.close(fig)
    print(f"    wrote {OUT / NAME}.{{png,pdf,svg}}")

    g = d.groupby(["country", "side"]).mt_per_year.sum().unstack("side")
    print(f"    {int((g.cold > g.heat).sum())} of {len(g)} countries have a "
          f"larger cold-side mass")
    print(f"    three largest countries carry "
          f"{100 * g.sum(axis=1).nlargest(3).sum() / g.values.sum():.0f} % "
          f"of the total")


if __name__ == "__main__":
    main()
