#!/usr/bin/env python
"""
19_results_figures.py  --  figures and tables for Results sections R1-R3
========================================================================

Produces every display item for the first three results sections into
`Results/figures` and `Results/tables`. Model fitting for R3 is cached so the
script can be rerun cheaply while figures are being tuned.

FIGURES
-------
  Fig1  global pattern of urban thermal extremes           (R1)
  Fig2  orthogonality of the three criteria + event-aligned exposure  (R1)
  Fig3  city-resolved vs national exposure benchmarks      (R2)
  Fig4  sectoral dynamic response                          (R3, from 16's output)
  Fig5  which exposure metric carries the carbon signal (R3)

TABLES
------
  T1 event taxonomy      T2 coverage and tier      T3 benchmark comparison
  T4 sectoral response   T5 exposure-form comparison

USAGE
-----
  python 19_results_figures.py                # everything
  python 19_results_figures.py --only 1 2     # selected figures
"""

from __future__ import annotations

import os
import argparse
import glob
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
# Results root. Overridable with the HC_RESULTS_ROOT environment variable so an
# extended-period rerun can write a parallel result set instead of overwriting
# the published one. Unset, it resolves to the original BASE/"Results".
RESULTS_ROOT = Path(os.environ.get("HC_RESULTS_ROOT", str(BASE / "Results")))

RES = RESULTS_ROOT
# Everything is filed per results section by section_dir(): <root>/R1/figures,
# <root>/R3/tables and so on. The flat <root>/figures and <root>/tables that an
# earlier layout used are no longer written to -- creating them would leave an
# empty stray directory that does not exist in the published Results tree.
# Emissions directory. Every loader in this project globs
# carbon-monitor-cities-<Country>-v0325.csv from here. HC_CARBON_DIR points the
# whole pipeline at an alternative directory -- e.g. a 2019-2022 merge built by
# 28_merge_emissions.py -- without touching the published 2019-2021 files.
CD = Path(os.environ.get(
    "HC_CARBON_DIR",
    str(BASE / "CarbonData" /
        "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide")))
# Scratch cache. Kept inside the project rather than /tmp so the pipeline runs
# unchanged on Windows, where "/tmp" resolves to a non-existent <drive>\\tmp.
CACHE_DIR = BASE / "outputs" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE = CACHE_DIR / "r3_form_cache.parquet"

LOOKUP = BASE / "outputs" / "fua_city_lookup.csv"
CITY_SUMMARY = BASE / "outputs" / "04_events" / "city_event_summary.parquet"
EVENTS = BASE / "outputs" / "04_events" / "city_events_daily_v2.parquet"
TAU = BASE / "outputs" / "04_events" / "exposure_by_tau.parquet"
CAT_HEAT = BASE / "outputs" / "04_events" / "event_catalog_heat.parquet"
CAT_COLD = BASE / "outputs" / "04_events" / "event_catalog_cold.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
TIER = BASE / "outputs" / "exposure" / "tables" / "country_tier.csv"
BENCH_FMT = str(BASE / "outputs" / "response" / "tables" / "r2_benchmarks_tier{t}.csv")
LAGPROF_FMT = str(BASE / "outputs" / "response" / "tables" / "r3_lag_profiles_tier{t}.csv")
SHAPES_FMT = str(BASE / "outputs" / "response" / "tables" / "r3_shape_indicators_tier{t}.csv")
NATBENCH = BASE / "outputs" / "exposure" / "national_benchmark_exposure.parquet"
WORLD = CACHE_DIR / "gadm0_simp.gpkg"

LAGS = 7
SAVE_FORMATS = ("png", "pdf", "svg")
SAVE_DPI = 600
TIER_MAIN = "AB"          # main sample: Tier A + B (28 countries, incl. China)
TIER_ALT = "C"            # low-coverage contrast (18 countries)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "savefig.dpi": SAVE_DPI, "figure.dpi": 110,
                     "svg.fonttype": "none"})


# DISPLAY NUMBERING
# -----------------
# The R2 benchmark comparison returns a null -- city-resolved exposure does not
# out-predict population-weighted national exposure -- so on the reviewer's
# advice it is demoted to the Supplementary Information as a validation check
# rather than carried as a main-text result. The functions below keep their
# original names (fig3 builds the benchmark figure, fig4 the lag profiles) so
# that the code stays readable against the analysis history, but the FILES they
# write are renumbered to a gapless main-text sequence:
#
#     fig3 -> Figure S1      fig4..fig7 -> Figures 3..6
#     T3   -> Table  S1      T4..T7     -> Tables  3..6
#
# Renaming at write time rather than renaming the functions means there is
# exactly one place to change if R2 is ever promoted back into the main text.
# The coverage-tier comparison (internal fig6/T6) moved to the supplement in
# the agreed restructure: it answers "is the Tier A result an artefact of who
# is in the sample", which is a robustness question, not one of the four core
# results. Humidity takes the vacated Fig 5 slot, and Fig 6 is now the base
# scan written directly by 32_base_temperature_scan.py.
RENUMBER = {"Fig3": "FigS1", "Fig4": "Fig3", "Fig5": "Fig4",
            "Fig6": "FigS10", "Fig7": "Fig5",
            "T3": "TS1", "T4": "T3", "T5": "T4", "T6": "TS9", "T7": "T5"}

# Each display item belongs to one results section, so outputs are filed under
# Results/<section>/figures and Results/<section>/tables rather than in one flat
# directory. Keys are the RENUMBERED identifiers.
SECTION_OF = {"Fig1": "R1", "Fig2": "R1", "Fig3": "R3", "Fig4": "R3",
              "Fig5": "R4", "FigS1": "SI",
              "FigS4": "SI", "FigS5": "SI", "FigS6": "SI", "FigS7": "SI",
              "FigS8": "SI", "FigS9": "SI", "FigS10": "SI",
              "FigS12": "SI", "FigS13": "SI",
              "TS7": "SI", "TS8": "SI", "TS9": "SI",
              "T1": "R1", "T1b": "R1", "T1c": "R1", "T1d": "R1", "T1e": "R1",
              "T2": "R2", "T3": "R3", "T4": "R3",
              "T5": "R4", "TS1": "SI", "TS6": "SI"}


def tier_label(t: str) -> str:
    """"AB" is the internal key for the pooled tier; readers see "A+B"."""
    return "A+B" if t == "AB" else t


def display_name(name: str) -> str:
    """Map an internal item name onto its published figure/table number."""
    key, _, rest = name.partition("_")
    return f"{RENUMBER.get(key, key)}_{rest}" if rest else RENUMBER.get(key, key)


def section_dir(name: str, kind: str) -> Path:
    key = name.split("_")[0]
    # An unlisted key used to fall through to SI without comment, so a new or
    # mistyped table quietly appeared in the supplement instead of the section
    # it belongs to. The fallback is kept -- it must not stop a long run -- but
    # it now announces itself.
    if key not in SECTION_OF:
        print(f"    WARNING: '{key}' has no section mapping; filing under SI")
    d = RES / SECTION_OF.get(key, "SI") / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name: str):
    """Write every figure as PNG, PDF and SVG into its results section."""
    name = display_name(name)
    out = section_dir(name, "figures")
    for ext in SAVE_FORMATS:
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)
    print(f"    {out.parent.name}/figures/{name}  "
          f"[{', '.join(SAVE_FORMATS)}]  {SAVE_DPI} dpi")


def save_table(df, name: str):
    name = display_name(name)
    out = section_dir(name, "tables") / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"    {out.parent.parent.name}/tables/{name}.csv")


_WORLD_CACHE = {}


def world_outline(ax):
    """Draw country outlines, rasterised.

    At 600 dpi a fully vectorised GADM outline plus 1,583 scatter points per
    panel produced a 107 MB SVG. Heavy geometry is therefore rasterised while
    all text and axes stay vector, which is the normal publication compromise.
    The layer is also cached and further simplified, since reading it once per
    panel dominated runtime.
    """
    try:
        import geopandas as gpd
        if "w" not in _WORLD_CACHE:
            w = gpd.read_file(WORLD)
            w["geometry"] = w.geometry.simplify(0.25)
            _WORLD_CACHE["w"] = w.boundary
        _WORLD_CACHE["w"].plot(ax=ax, lw=0.18, color="0.72", rasterized=True)
    except Exception:
        pass
    ax.set_xlim(-170, 180)
    ax.set_ylim(-58, 78)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ==========================================================================
# City-level exposure metric table and the generic map grid
# ==========================================================================
# Extra per-city annual means that city_event_summary.parquet does not carry.
# The published summary stops at the heat-side counts; the cold-side absolute
# grades and the hourly demand integrals only exist in the daily file, and both
# are needed for Figure 1 and the supplementary map atlas.
# Only exposure metrics the summary does NOT already carry. HEH, DEH,
# utci_verystrong and twb_stress are in city_event_summary.parquet; adding them
# here again produced n_*_days_x / _y on the merge and a KeyError downstream.
_EXTRA_BINARY = ["utci_moderate", "utci_coldstress", "utci_strongcold",
                 "twbg_highrisk", "twc_danger", "twc_frostbite"]
_EXTRA_SUM = ["SCDH18", "SCDH24_night", "CDH_Twb26_night", "LDH_w", "EDH_h"]
_EXTRA_MEAN = ["WBD_at_tmax", "Twb_max", "utci_max", "RH_at_tmax"]

# Set from --plates; empty means render the whole atlas.
_ATLAS_PLATES: list[str] = []


def city_metrics() -> pd.DataFrame:
    """Per-city annual means of every exposure metric used in a map.

    Cached on the daily file's fingerprint. Aggregating 2.3 M city-days is
    cheap once and wasteful eleven times over, and keying the cache on the
    input rather than on its own existence is the lesson from the R3.2 form
    cache, which quietly served coefficients from a superseded panel.
    """
    cache = CACHE_DIR / "city_metrics.parquet"
    st = EVENTS.stat()
    stamp = f"{EVENTS.name}:{st.st_size}:{int(st.st_mtime)}"
    if cache.exists():
        got = pd.read_parquet(cache)
        if got.attrs.get("stamp") == stamp or (
                "_stamp" in got.columns and got["_stamp"].iloc[0] == stamp):
            return got.drop(columns=["_stamp"], errors="ignore")

    s = pd.read_parquet(CITY_SUMMARY)
    s["city_id"] = s.city_id.astype(str).str.zfill(4)
    if not {"lat", "lon"}.issubset(s.columns):   # older summaries lack them
        lk = pd.read_csv(LOOKUP, dtype={"city_id": str})
        lk["city_id"] = lk.city_id.str.zfill(4)
        s = s.merge(lk[["city_id", "lat", "lon"]], on="city_id", how="left")

    cols = ["city_id"] + _EXTRA_BINARY + _EXTRA_SUM + _EXTRA_MEAN
    ev = pd.read_parquet(EVENTS, columns=cols)
    ev["city_id"] = ev.city_id.astype(str).str.zfill(4)
    g = ev.groupby("city_id")
    agg = pd.concat([
        g[_EXTRA_BINARY + _EXTRA_SUM].sum(),
        g[_EXTRA_MEAN].mean(),
    ], axis=1).reset_index()
    agg = agg.rename(columns={c: f"n_{c}_days" for c in _EXTRA_BINARY}
                     | {c: f"cum_{c.lower()}" for c in _EXTRA_SUM}
                     | {c: f"mean_{c.lower()}" for c in _EXTRA_MEAN})

    # Guard against a future column appearing on both sides: a silent _x/_y
    # suffix pair is far harder to diagnose than a dropped duplicate.
    dup = [c for c in agg.columns if c != "city_id" and c in s.columns]
    if dup:
        print(f"      already in the summary, not recomputed: {', '.join(dup)}")
        agg = agg.drop(columns=dup)
    d = s.merge(agg, on="city_id", how="left")
    # n_days is already 365.25 for every city in this sample, but dividing
    # explicitly keeps the table correct if a future run has partial cities.
    years = (d.n_days / 365.25).replace(0, np.nan)
    for c in [f"n_{x}_days" for x in _EXTRA_BINARY] + \
             [f"cum_{x.lower()}" for x in _EXTRA_SUM]:
        if c in d.columns:
            d[c] = d[c] / years
    d = d.dropna(subset=["lat", "lon"])
    out = d.copy()
    out["_stamp"] = stamp
    out.to_parquet(cache, index=False)
    print(f"      built city exposure metric table: {len(d):,} cities, "
          f"{len(agg.columns) - 1} extran exposure metrics")
    return d


def panel_stats(d, panels, name) -> pd.DataFrame:
    """Per-panel diagnostics for a map grid, without drawing anything.

    Split out from `map_grid` so the summary table is complete whether or not
    every plate was rendered in this run. Tying the table to the rendering meant
    a partial run silently produced a partial table.
    """
    abslat = d.lat.abs()
    rows = []
    for k, (col, ptitle, _cmap, (qlo, qhi), unit, fam) in enumerate(panels):
        if col not in d.columns:
            continue
        x = d[col].astype(float)
        ok = x.notna()
        lo, hi = float(x.quantile(qlo)), float(x.quantile(qhi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
            if hi <= lo:
                hi = lo + 1.0
        rows.append({
            "figure": display_name(name), "panel": chr(97 + k),
            "variable": col, "label": ptitle, "criterion": fam,
            "unit": unit.replace("$", ""),
            "colour_scale": "sqrt" if fam not in ("relative", "linear")
                            else "linear",
            "scale_lo": lo, "scale_hi": hi,
            "r_abs_lat": (float(np.corrcoef(abslat[ok], x[ok])[0, 1])
                          if ok.sum() > 2 else np.nan),
            "cv": float(x.std() / x.mean()) if x.mean() else np.nan,
            "n_cities_defined": int(ok.sum()),
            "n_cities_nonzero": int((x > 0).sum()),
            "p10": x.quantile(0.10), "median": x.median(),
            "p90": x.quantile(0.90)})
    return pd.DataFrame(rows)


def map_grid(d, panels, nrow, ncol, title, name, row_labels=None,
             panel_width=3.35):
    """Render a grid of city point maps and return per-panel diagnostics.

    Every map figure in the paper goes through here so that the scale rules,
    the zero handling and the two annotated statistics are identical across
    the main figure and the supplementary atlas. `panels` entries are
    (column, title, cmap, (q_lo, q_hi), unit, family).
    """
    stats = panel_stats(d, panels, name)
    # Height is set from the map aspect (350 deg x 136 deg) so the aspect
    # constraint binds on height; a taller figure only produces colour bars
    # that overhang the maps they belong to.
    fig, axes = plt.subplots(
        nrow, ncol, layout="constrained",
        figsize=(panel_width * ncol, nrow * (panel_width / 2.35) + 0.55))
    axes = np.atleast_2d(axes)
    for k, (ax, spec) in enumerate(zip(axes.ravel(), panels)):
        col, ptitle, cmap, (qlo, qhi), unit, fam = spec
        world_outline(ax)
        ax.set_aspect("equal", adjustable="box")
        if col not in d.columns:
            ax.text(0.5, 0.5, f"{col}\nnot available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="0.5")
            continue

        x = d[col].astype(float)
        # Undefined is not zero. A city with no cold wave in four years has no
        # first-onset day of year; drawing it at the bottom of the colour scale
        # would invent a January event. Those cities get an open grey marker.
        bad = x.isna()
        if bad.any():
            ax.scatter(d.lon[bad], d.lat[bad], s=3.0, facecolors="none",
                       edgecolors="0.75", lw=0.25, rasterized=True, zorder=2)
        lo, hi = float(x.quantile(qlo)), float(x.quantile(qhi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x)) or 1.0
            if hi <= lo:
                hi = lo + 1.0

        # Right-skewed exposure metrics -- WBGT > 28 C has a median of 1 day and a
        # p90 of 90 -- render almost entirely white on a linear scale, hiding
        # the gradient the panel exists to show. Those use a square-root norm,
        # stated on the colour bar. Near-symmetric exposure metrics stay linear,
        # which also keeps the criterion families visually distinguishable.
        skewed = fam not in ("relative", "linear")
        norm = (mcolors.PowerNorm(gamma=0.5, vmin=lo, vmax=hi) if skewed
                else mcolors.Normalize(vmin=lo, vmax=hi))

        # Cities where an exposure metric is defined but never fires are grey, not the
        # lowest colour: "zero in four years" and "just above zero" are
        # different statements.
        zero = (x <= 0).fillna(False)
        if skewed and zero.any():
            ax.scatter(d.lon[zero], d.lat[zero], s=2.4, color="0.82", lw=0,
                       rasterized=True, zorder=2)
        show = ~zero & ~bad if skewed else ~bad
        sc = ax.scatter(d.lon[show], d.lat[show], c=x[show], s=3.8,
                        cmap=cmap, norm=norm, lw=0, alpha=0.92,
                        rasterized=True, zorder=3)
        ax.set_title(f"{chr(97 + k)}  {ptitle}", loc="left", fontweight="bold",
                     fontsize=8.2, pad=3)

        # The colour bar is an inset of the map axes. With aspect-locked axes
        # `shrink` applies to the gridspec cell rather than to the shrunken
        # map, so a gridspec colour bar stood about 40 % taller than its map.
        cax = ax.inset_axes([1.025, 0.0, 0.028, 1.0])
        cb = fig.colorbar(sc, cax=cax, extend="max")
        cb.ax.tick_params(labelsize=5.6, length=1.8, pad=1.4)
        cb.outline.set_linewidth(0.4)
        cb.set_label(unit + (r"  ($\sqrt{\,}$)" if skewed else ""), fontsize=5.8)

        srow = stats.loc[stats.variable == col].iloc[0]
        ax.text(0.01, 0.115, f"r(|lat|) = {srow.r_abs_lat:+.2f}",
                transform=ax.transAxes, fontsize=6.4, color="0.12",
                fontweight="bold")
        ax.text(0.01, 0.035,
                f"CV = {srow.cv:.2f}   {srow.n_cities_nonzero:,}/{len(d):,} > 0",
                transform=ax.transAxes, fontsize=5.9, color="0.38")

    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=10)
    if row_labels:
        # Kept short: the label is rotated, so its length is measured against
        # the height of an aspect-locked map (about 1.1 in). Longer wording
        # overflowed and was clipped at the figure edge.
        for row, lab in enumerate(row_labels):
            axes[row, 0].set_ylabel(lab, fontsize=6.9, fontweight="bold",
                                    color="0.25", labelpad=3)
    save(fig, name)
    return stats


# ==========================================================================
# Supplementary map atlas  (FigS4-FigS7)
# ==========================================================================
def si_map_atlas():
    """Every remaining city-level exposure metric, mapped on the Figure 1 rules.

    Figure 1 shows nine exposure metrics chosen to make one argument. The codebook
    defines many more, and a reader who wants to know what an exposure metric this
    paper did not use looks like should not have to take that on trust. These
    four plates map the rest under identical scale, zero and annotation
    conventions, so a panel here is directly comparable with a panel there.

    They are also where the known artefacts are put on display rather than
    described: HWMId inflates in the tropics because it is normalised by the
    IQR of annual maxima, and the relative cold-wave definition fires in late
    spring because its threshold is calendar-day specific. Both are visible in
    FigS7 and both are stated in the SI text.
    """
    d = city_metrics()
    span = pd.read_parquet(EVENTS, columns=["date"]).date
    y0, y1 = span.dt.year.min(), span.dt.year.max()
    tail = f"{len(d):,} functional urban areas, {y0}–{y1}, annual means"

    plates = [
        ("FigS4_atlas_relative",
         f"Relative criteria: percentile thresholds against each city's own "
         f"1991–2020 climatology. {tail}",
         ["Day counts\n(heat)", "Day counts\n(cold)", "Spells and\ncompounds"],
         [
             ("n_EHD_days", "Extreme heat days (Tmax > p90)", "Reds",
              (0.02, 0.98), "days yr$^{-1}$", "relative"),
             ("n_HN_days", "Hot nights (Tmin > p90)", "OrRd", (0.02, 0.98),
              "days yr$^{-1}$", "relative"),
             ("n_HW_days", "Days inside a heatwave", "Oranges", (0.02, 0.98),
              "days yr$^{-1}$", "relative"),
             ("n_ECD_days", "Extreme cold days (Tmin < p10)", "Blues",
              (0.02, 0.98), "days yr$^{-1}$", "relative"),
             ("n_CD_days", "Cold days (Tmax < p10)", "PuBu", (0.02, 0.98),
              "days yr$^{-1}$", "relative"),
             ("n_CW_days", "Days inside a cold wave", "BuPu", (0.02, 0.98),
              "days yr$^{-1}$", "relative"),
             ("n_CDNH_days", "Compound day–night heat (EHD and hot night)",
              "PuRd", (0.02, 0.98), "days yr$^{-1}$", "relative"),
             ("n_DEH_days", "Dry extreme heat (WBD > city p67)", "YlOrBr",
              (0.02, 0.98), "days yr$^{-1}$", "relative"),
             ("hw_max_duration", "Longest heatwave", "Oranges", (0.02, 0.98),
              "days", "relative"),
         ]),

        ("FigS5_atlas_absolute",
         f"Absolute physiological grades: fixed international thresholds, no "
         f"local baseline. {tail}",
         ["Heat stress\n(UTCI)", "Heat stress\n(WBGT, T$_{wb}$)",
          "Cold stress"],
         [
             ("n_utci_moderate_days", "UTCI > 26 °C (moderate)", "YlOrRd",
              (0.0, 0.98), "days yr$^{-1}$", "absolute"),
             ("n_utci_strong_days", "UTCI > 32 °C (strong)", "YlOrRd",
              (0.0, 0.98), "days yr$^{-1}$", "absolute"),
             ("n_utci_verystrong_days", "UTCI > 38 °C (very strong)", "Reds",
              (0.0, 0.98), "days yr$^{-1}$", "absolute"),
             ("n_twbg_danger_days", "WBGT > 28 °C (ISO 7243 danger)",
              "YlOrBr", (0.0, 0.98), "days yr$^{-1}$", "absolute"),
             ("n_twbg_highrisk_days", "WBGT > 32 °C (high risk)", "OrRd",
              (0.0, 0.99), "days yr$^{-1}$", "absolute"),
             ("n_twb_stress_days", "T$_{wb}$ > 28 °C", "GnBu", (0.0, 0.99),
              "days yr$^{-1}$", "absolute"),
             ("n_utci_coldstress_days", "UTCI < 0 °C (cold stress)", "PuBu",
              (0.0, 0.98), "days yr$^{-1}$", "absolute"),
             ("n_utci_strongcold_days", "UTCI < −13 °C (strong cold)",
              "Blues", (0.0, 0.98), "days yr$^{-1}$", "absolute"),
             ("n_twc_danger_days", "Wind chill < −10 °C", "BuPu",
              (0.0, 0.98), "days yr$^{-1}$", "absolute"),
         ]),

        ("FigS6_atlas_demand",
         f"Continuous demand integrals and mean humid-air state, formed hourly "
         f"and then aggregated. {tail}",
         ["Sensible\ncooling", "Latent and\nenthalpy", "Mean state"],
         [
             ("cum_scdh18", "Cooling degree-hours, base 18 °C", "OrRd",
              (0.0, 0.98), "K h yr$^{-1}$", "continuous"),
             ("cum_scdh24", "Cooling degree-hours, base 24 °C", "OrRd",
              (0.0, 0.98), "K h yr$^{-1}$", "continuous"),
             ("cum_scdh24_night", "Cooling degree-hours, night only",
              "PuRd", (0.0, 0.98), "K h yr$^{-1}$", "continuous"),
             ("cum_cdh_twb26", "Wet-bulb degree-hours, base 26 °C", "GnBu",
              (0.0, 0.99), "K h yr$^{-1}$", "continuous"),
             ("cum_ldh_w", "Latent degree-hours (humidity ratio)", "BuGn",
              (0.0, 0.98), "g kg$^{-1}$ h yr$^{-1}$", "continuous"),
             ("cum_edh_h", "Enthalpy degree-hours", "YlGnBu", (0.0, 0.98),
              "kJ kg$^{-1}$ h yr$^{-1}$", "continuous"),
             ("mean_wbd_at_tmax", "Wet-bulb depression at T$_{max}$",
              "BrBG_r", (0.02, 0.98), "K", "linear"),
             ("mean_rh_at_tmax", "Relative humidity at T$_{max}$", "BrBG",
              (0.02, 0.98), "%", "linear"),
             ("mean_utci_max", "Mean daily maximum UTCI", "RdYlBu_r",
              (0.02, 0.98), "°C", "linear"),
         ]),

        ("FigS7_atlas_intensity",
         f"Event intensity, duration and timing, from the heat and cold event "
         f"catalogues. {tail}",
         ["Heat intensity", "Heat duration\nand timing", "Cold events"],
         [
             ("hw_cum_excess", "Cumulative heatwave excess", "Reds",
              (0.02, 0.98), "K day yr$^{-1}$", "continuous"),
             ("hw_peak_excess", "Peak heatwave excess above p90", "OrRd",
              (0.02, 0.98), "K", "linear"),
             ("max_tmax_z", "Peak standardised T$_{max}$ anomaly", "YlOrRd",
              (0.02, 0.98), "SD", "linear"),
             ("n_hw_long_events", "Heatwaves lasting ≥ 5 days", "Oranges",
              (0.0, 0.98), "events yr$^{-1}$", "continuous"),
             ("hw_first_onset_doy", "First heatwave of the season",
              "twilight", (0.02, 0.98), "day of year", "linear"),
             ("hw_hwmid_max", "HWMId, seasonal maximum", "PuRd",
              (0.02, 0.98), "index", "continuous"),
             ("cw_cum_excess", "Cumulative cold-wave deficit", "Blues",
              (0.02, 0.98), "K day yr$^{-1}$", "continuous"),
             ("cw_max_duration", "Longest cold wave", "PuBu", (0.02, 0.98),
              "days", "continuous"),
             ("cw_first_onset_doy", "First cold wave of the season",
              "twilight", (0.02, 0.98), "day of year", "linear"),
         ]),
    ]

    # The table is assembled from every plate, whether or not this run rendered
    # every plate: `--plates` exists so one figure can be iterated on without
    # waiting for the other three, and a partial table would be worse than no
    # table at all.
    atlas = pd.concat([panel_stats(d, panels, name)
                       for name, _t, _r, panels in plates], ignore_index=True)
    save_table(atlas.round(4), "TS6_map_atlas_geography")

    wanted = _ATLAS_PLATES or None
    for name, title, rows, panels in plates:
        if wanted and not any(w in name for w in wanted):
            continue
        map_grid(d, panels, 3, 3, title, name, row_labels=rows)

    # The atlas is where the criterion contrast can be checked against many
    # more exposure metrics than Figure 1 shows, so the summary is printed for the
    # SI text rather than left to be read off the panels.
    for fam, g in atlas.groupby("criterion"):
        print(f"      {fam:11s} n={len(g):2d}  |r(|lat|)| "
              f"{g.r_abs_lat.abs().min():.2f}-{g.r_abs_lat.abs().max():.2f}"
              f"   CV {g.cv.min():.2f}-{g.cv.max():.2f}")


# ==========================================================================
# Fig 1  global pattern of urban thermal extremes
# ==========================================================================
def fig1():
    d = city_metrics()

    # ---------------------------------------------------------------------
    # Panel selection: rows are the three ways this paper defines exposure,
    # columns are the three physical dimensions it defines them over.
    #
    #                heat            humid heat          cold
    #   relative     heatwave freq   humid extreme heat  cold-wave freq
    #   absolute     UTCI > 32 C     CHHE Twb >= 26 C    UTCI cold stress < 0
    #   continuous   cooling deg-h   wet-bulb deg-h      heating deg-h
    #
    # This is not a gallery. Two readings are built into the layout:
    #
    #   ACROSS ROWS -- row 1 is near-uniform because a percentile scored
    #   against local climatology gives every city about the same number of
    #   extreme days however hot it is (|r| with |lat| <= 0.15). Rows 2 and 3
    #   are strongly latitudinal. That is the exposure metric mismatch R3.2 tests
    #   against emissions, visible here in the exposure field alone.
    #
    #   DOWN COLUMNS -- rows 2 and 3 are the threshold and the continuous form
    #   of the same physical quantity, so each column is a redundancy check.
    #   UTCI > 32 C correlates 0.94 with cooling degree-hours and CHHE 0.98
    #   with wet-bulb degree-hours. Rows 2 and 3 agree with each other and both
    #   disagree with row 1, which is the point: the relative row is the odd
    #   one out, not one of three equally different views.
    #
    # The cold column is in the main figure because the cold-side Power
    # response is 3.3 times the heat side and is identified in 26 countries
    # against 20, so a heat-only Figure 1 would put the study's firmest results
    # outside its opening figure.
    # ---------------------------------------------------------------------
    panels = [
        # col, title, cmap, (q_lo, q_hi) colour-scale quantiles, unit, family
        ("n_hw_events", "Heatwave frequency", "Reds", (0.02, 0.98),
         "events yr$^{-1}$", "relative"),
        ("n_HEH_days", "Humid extreme heat (WBD < city p33)", "BuGn",
         (0.02, 0.98), "days yr$^{-1}$", "relative"),
        ("n_cw_events", "Cold-wave frequency", "Blues", (0.02, 0.98),
         "events yr$^{-1}$", "relative"),

        ("n_utci_strong_days", "UTCI > 32 °C (strong heat stress)", "YlOrRd",
         (0.0, 0.98), "days yr$^{-1}$", "absolute"),
        ("n_CHHE_days", "Compound humid heat, T$_{wb}$ ≥ 26 °C", "GnBu",
         (0.0, 0.99), "days yr$^{-1}$", "absolute"),
        ("n_utci_coldstress_days", "UTCI < 0 °C (cold stress)", "PuBu",
         (0.0, 0.98), "days yr$^{-1}$", "absolute"),

        ("cum_scdh24", "Cooling degree-hours, base 24 °C", "OrRd",
         (0.0, 0.98), "K h yr$^{-1}$", "continuous"),
        ("cum_cdh_twb26", "Wet-bulb degree-hours, base 26 °C", "GnBu",
         (0.0, 0.99), "K h yr$^{-1}$", "continuous"),
        ("cum_hdh15", "Heating degree-hours, base 15 °C", "Blues",
         (0.0, 0.98), "K h yr$^{-1}$", "continuous"),
    ]

    # Read the window from the data. A hardcoded "2019-2021" survived the
    # four-year rebuild in the title of this very figure, which is exactly the
    # kind of error nobody catches by reading the caption.
    span = pd.read_parquet(EVENTS, columns=["date"]).date
    y0, y1 = span.dt.year.min(), span.dt.year.max()

    sdf = map_grid(
        d, panels, 3, 3,
        f"Global pattern of urban thermal extremes: {len(d):,} functional "
        f"urban areas, {y0}\u2013{y1}, annual means",
        "Fig1_global_event_pattern",
        row_labels=["Relative\n(local baseline)",
                    "Absolute\n(fixed thresholds)",
                    "Continuous\n(degree-hours)"])
    save_table(sdf.round(4), "T1b_metric_geography")

    # The claims R1.3 makes about this figure are computed here rather than
    # asserted in prose, so they move with the data.
    rel = [p[0] for p in panels if p[5] == "relative"]
    oth = [p[0] for p in panels if p[5] != "relative"]
    cross = d[rel + oth].corr(method="spearman").loc[rel, oth].abs()
    r_rel = sdf[sdf.criterion == "relative"]
    r_oth = sdf[sdf.criterion != "relative"]
    print(f"      row 1 relative    |r(|lat|)| <= {r_rel.r_abs_lat.abs().max():.2f}"
          f", CV {r_rel.cv.min():.2f}-{r_rel.cv.max():.2f}")
    print(f"      rows 2-3          |r(|lat|)| >= {r_oth.r_abs_lat.abs().min():.2f}"
          f", CV {r_oth.cv.min():.2f}-{r_oth.cv.max():.2f}")
    print(f"      between-family |Spearman|  max {cross.max().max():.2f}, "
          f"median {cross.stack().median():.2f}")
    for a, b, lab in [("n_utci_strong_days", "cum_scdh24", "heat column"),
                      ("n_CHHE_days", "cum_cdh_twb26", "humid-heat column"),
                      ("n_utci_coldstress_days", "cum_hdh15", "cold column")]:
        rho = d[[a, b]].corr(method="spearman").iloc[0, 1]
        print(f"      {lab:18s} threshold vs continuous  rho = {rho:+.2f}")

    # Table 1
    ev = pd.read_parquet(EVENTS, columns=[
        "EHD", "ECD", "HW", "CW", "HN", "CD", "CDNH", "HEH", "DEH",
        "CHHE", "CHHE_plus", "utci_strong", "utci_verystrong",
        "twbg_danger", "twb_stress"])
    rows = []
    labels = {
        "EHD": ("relative", "Tmax > local DOY p90"),
        "ECD": ("relative", "Tmin < local DOY p10"),
        "HW": ("relative", ">= 3 consecutive EHD"),
        "CW": ("relative", ">= 3 consecutive ECD"),
        "HN": ("relative", "Tmin > local DOY p90"),
        "CD": ("relative", "Tmax < local DOY p10"),
        "CDNH": ("compound", "EHD and HN on the same day"),
        "HEH": ("humidity", "EHD and WBD below city p33"),
        "DEH": ("humidity", "EHD and WBD above city p67"),
        "CHHE": ("absolute", "EHD and Twb_max >= 26 C"),
        "CHHE_plus": ("absolute", "EHD and Twb_max >= 28 C"),
        "utci_strong": ("absolute", "UTCI_max > 32 C"),
        "utci_verystrong": ("absolute", "UTCI_max > 38 C"),
        "twbg_danger": ("absolute", "WBGT_max > 28 C"),
        "twb_stress": ("absolute", "Twb_max > 28 C"),
    }
    for k, (crit, defn) in labels.items():
        if k in ev.columns:
            rows.append({"code": k, "criterion": crit, "definition": defn,
                         "share_of_city_days_pct": 100 * ev[k].mean(),
                         "n_city_days": int(ev[k].sum())})
    t1 = pd.DataFrame(rows)
    save_table(t1, "T1_event_taxonomy")


# ==========================================================================
# Fig 2  criteria orthogonality + event-aligned exposure
# ==========================================================================
# Floors for the two night-share ratios. A ratio needs a denominator: cold
# countries have essentially no wet-bulb degree-hours, and dividing their
# near-zero night component by their near-zero total returned values such as
# 42 % for the Netherlands on a denominator of 0.03 K.h.
MIN_COOL, MIN_WB = 500.0, 50.0
HILITE = ["India", "China", "United States", "Germany", "France",
          "United Kingdom", "Japan", "Brazil", "Russia", "Spain",
          "Singapore", "Nigeria"]
PROF = ["HW", "CW", "EHD", "ECD", "HN", "CDNH", "CHHE", "utci_strong",
        "twbg_danger", "SCDH24", "CDH_Twb26", "HDH15"]
PROF_LAB = ["HW", "CW", "EHD", "ECD", "Hot\nnight", "CDNH", "CHHE",
            "UTCI\n>32", "WBGT\n>28", "Cool\ndeg-h", "Twb\ndeg-h",
            "Heat\ndeg-h"]


# --------------------------------------------------------------------------
# Part III event codes and the criterion co-occurrence matrix
# --------------------------------------------------------------------------
# (code, label, family). Families drive the axis colouring; order drives the
# block structure, so codes are listed family by family.
PART3 = [
    ("EHD", "EHD", "heat"), ("HW", "HW", "heat"),
    ("HN", "Hot night", "heat"), ("CDNH", "CDNH", "heat"),
    ("ECD", "ECD", "cold"), ("CW", "CW", "cold"), ("CD", "Cold day", "cold"),
    ("HEH", "HEH", "humid"), ("DEH", "DEH", "humid"),
    ("CHHE", "CHHE", "humid"), ("CHHE_plus", "CHHE+", "humid"),
    ("utci_moderate", "UTCI>26", "absolute"),
    ("utci_strong", "UTCI>32", "absolute"),
    ("utci_verystrong", "UTCI>38", "absolute"),
    ("twbg_danger", "WBGT>28", "absolute"),
    ("twbg_highrisk", "WBGT>32", "absolute"),
    ("twb_stress", "Twb>28", "absolute"),
    ("utci_coldstress", "UTCI<0", "absolute"),
    ("utci_strongcold", "UTCI<-13", "absolute"),
    ("twc_danger", "Windchill<-10", "absolute"),
    ("twc_frostbite", "Windchill<-27", "absolute"),
]
# twb_severe (Twb > 31 C) is deliberately absent: it fires on 0.0001 % of
# city-days, so every probability conditioned on it would rest on a couple of
# observations. Prevalence that low is reported in Table T1, not conditioned on.

FAM_TEXT = {"heat": "#c0392b", "cold": "#2471a3",
            "humid": "#1f4e79", "absolute": "#8a6d1f"}
FAM_BAND = {"heat": "#f6cfc6", "cold": "#cfe2f3",
            "humid": "#aebfd4", "absolute": "#f0e3bc"}
FAM_NAME = {"heat": "relative, heat", "cold": "relative, cold",
            "humid": "humidity typing", "absolute": "absolute grade"}

# Cells whose value is fixed by the definitions rather than by the climate.
# Declared once so the main figure and the full supplementary matrix cannot
# disagree about which cells are findings.
#   NESTED    (a, b): b is a subset of a, so P(a | b) = 1 exactly
#   DISJOINT  (a, b): mutually exclusive by construction, so P = 0 both ways
#   FIXED     (a, b): P(a | b) is set by the definition, e.g. a tercile split
NESTED = [("EHD", "HW"), ("EHD", "CDNH"), ("EHD", "HEH"), ("EHD", "DEH"),
          ("EHD", "CHHE"), ("EHD", "CHHE_plus"), ("HN", "CDNH"),
          ("CHHE", "CHHE_plus"), ("ECD", "CW"),
          ("utci_moderate", "utci_strong"), ("utci_strong", "utci_verystrong"),
          ("utci_moderate", "utci_verystrong"),
          ("utci_coldstress", "utci_strongcold"),
          ("twbg_danger", "twbg_highrisk"), ("twc_danger", "twc_frostbite")]
DISJOINT = [("HEH", "DEH")]
FIXED = [("HEH", "EHD"), ("DEH", "EHD")]


def cond_matrix(ev: pd.DataFrame, codes: list[str]):
    """P(row | column) among city-days, plus the definitional-cell mask."""
    n = len(codes)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            den = int((ev[b] == 1).sum())
            if den:
                M[i, j] = int(((ev[a] == 1) & (ev[b] == 1)).sum()) / den
    mask = np.zeros((n, n), dtype=bool)
    idx = {c: k for k, c in enumerate(codes)}
    for a, b in NESTED + FIXED:
        if a in idx and b in idx:
            mask[idx[a], idx[b]] = True
    for a, b in DISJOINT:
        if a in idx and b in idx:
            mask[idx[a], idx[b]] = mask[idx[b], idx[a]] = True
    return M, mask


def criteria_matrix(fig, ax, ev, spec, show_values=True, label_size=6.4,
                    value_size=5.6):
    """Draw a conditional co-occurrence matrix with family-coloured axes.

    Family is encoded on the tick labels and on a band beside each axis, never
    in the cell fill: the fill carries the probability, and colouring cells by
    family would make magnitudes incomparable across blocks -- which is exactly
    where the near-zero heat-cold cells need to be read.
    """
    codes = [c for c, _, _ in spec]
    labs = [l for _, l, _ in spec]
    fams = [f for _, _, f in spec]
    n = len(codes)
    M, mask = cond_matrix(ev, codes)

    im = ax.imshow(np.where(mask, np.nan, M), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=label_size)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labs, fontsize=label_size)
    for tick, f in zip(ax.get_xticklabels(), fams):
        tick.set_color(FAM_TEXT[f])
    for tick, f in zip(ax.get_yticklabels(), fams):
        tick.set_color(FAM_TEXT[f])
    ax.tick_params(length=1.5)

    # family bands just outside the axes
    for k, f in enumerate(fams):
        ax.add_patch(plt.Rectangle((k - .5, n - .5), 1, 0.42,
                                   color=FAM_BAND[f], clip_on=False, lw=0))
        ax.add_patch(plt.Rectangle((-1.42, k - .5), 0.42, 1,
                                   color=FAM_BAND[f], clip_on=False, lw=0))

    # square border around the humidity-typing block
    hs = [k for k, f in enumerate(fams) if f == "humid"]
    if hs:
        lo, hi = min(hs), max(hs)
        ax.add_patch(plt.Rectangle((lo - .5, lo - .5), hi - lo + 1,
                                   hi - lo + 1, fill=False,
                                   edgecolor=FAM_TEXT["humid"], lw=1.4,
                                   zorder=5))

    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           color="0.87", zorder=2))
                if show_values:
                    ax.text(j, i, "def.", ha="center", va="center",
                            fontsize=value_size - 0.4, color="0.45",
                            style="italic", zorder=3)
            elif show_values and np.isfinite(M[i, j]):
                # P(UTCI>32 | CHHE) is 0.997, not 1. Printing "1.00" would turn
                # a near-certain physical implication into an apparent
                # definitional one, which is the distinction this panel exists
                # to keep straight.
                v = M[i, j]
                ax.text(j, i, f"{v:.3f}" if 0.99 < v < 1 else f"{v:.2f}",
                        ha="center", va="center", fontsize=value_size,
                        color="white" if v > 0.55 else "0.15")
    return im, M, mask


def country_profile() -> pd.DataFrame:
    """Annual totals of the population-weighted country-day exposure series.

    These are the regressors of R3-R6, not city means, which is the whole point
    of describing them: a reader is entitled to see the variable before seeing
    what it does.
    """
    cols = ["country", "tier"] + [f"w_{c}" for c in PROF] + \
           ["w_SCDH24_night", "w_CDH_Twb26_night"]
    cd = pd.read_parquet(EXPOSURE, columns=cols)
    C = cd.groupby(["country", "tier"], as_index=False).mean(numeric_only=True)
    for c in C.columns[2:]:
        C[c] *= 365.25                  # per-day mean -> per-year total
    C = C.rename(columns=lambda c: c[2:] if c.startswith("w_") else c)
    C["ns_cool"] = np.where(C.SCDH24 >= MIN_COOL,
                            100 * C.SCDH24_night / C.SCDH24, np.nan)
    C["ns_wb"] = np.where(C.CDH_Twb26 >= MIN_WB,
                          100 * C.CDH_Twb26_night / C.CDH_Twb26, np.nan)
    return C


def fig2():
    """Quantitative structure of the exposure field, in four panels.

    Figure 1 is geography; this figure is structure. Two panels describe the
    taxonomy at city-day level -- how long events last and how the criteria
    relate to one another -- one shows how exposure evolves around an onset,
    and the last moves to the country-day series that R3-R6 regress on.

    Two further country panels were drafted and moved to the supplement. A
    46 x 12 percentile heatmap turned out to be mostly an artefact of its own
    row ordering: six of its twelve columns correlate with the sort key at
    |rho| > 0.79 and are gradients by construction, three are the noise that
    panel d already quantifies with a single number, and the matrix is close to
    two-dimensional anyway (PC1 54 %, PC1+PC2 77 %). The day-night split is a
    real descriptive fact but nothing downstream uses it, and R4.4 already
    treats day against night as a response question with coefficients. Both are
    Supplementary Figure S8.
    """
    # Panel b covers every Part III relative and humidity-typing code, plus the
    # two absolute grades used elsewhere in the paper. Seven codes was an
    # arbitrary selection a reader would rightly query; "all of families A and
    # B" is a rule. The full 21-code matrix including every absolute grade is
    # Supplementary Figure S9 -- at that size the cells cannot carry numbers.
    spec = [s for s in PART3 if s[2] != "absolute"] + \
           [s for s in PART3 if s[0] in ("utci_strong", "twbg_danger")]
    crit = [c for c, _, _ in spec]
    ev = pd.read_parquet(EVENTS, columns=crit)
    C = country_profile()

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.6), layout="constrained")

    # ------------------------------------------------------------------
    # (a) survival curves of event duration
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    surv_rows = []
    ps = {}
    for path, lab, col in [(CAT_HEAT, "Heatwaves", "#c0392b"),
                           (CAT_COLD, "Cold waves", "#2471a3")]:
        dur = pd.read_parquet(path, columns=["duration"]).duration
        ks = np.arange(3, int(dur.max()) + 1)
        ax.step(ks, [(dur >= k).mean() for k in ks], where="post", lw=1.9,
                color=col, label=f"{lab} (n = {len(dur):,})")
        ps[lab] = float((dur >= 7).mean())
        for k in (5, 7, 10, 14):
            surv_rows.append({"event": lab, "duration_days": int(k),
                              "p_at_least": float((dur >= k).mean()),
                              "n_events": int(len(dur)),
                              "mean_duration": float(dur.mean()),
                              "median_duration": float(dur.median()),
                              "max_duration": int(dur.max())})
    ax.set_yscale("log")
    ax.set_xlim(3, 20)
    ax.set_ylim(2e-4, 1.4)
    ax.set_xlabel("event duration D (days)", fontsize=8)
    ax.set_ylabel("P(duration \u2265 D)", fontsize=8)
    ax.set_title("a  How long events last", loc="left", fontweight="bold",
                 fontsize=9)
    ax.legend(fontsize=6.8, frameon=False, loc="upper right")
    ax.grid(lw=0.35, alpha=0.45)
    ax.tick_params(labelsize=7)
    # Both curves start at 1 by construction: a spell needs >= 3 days to be an
    # event at all. Everything to the right of D = 3 is data.
    ax.axvline(3, color="0.6", lw=0.7, ls=":")
    ax.text(3.12, 2.6e-4, "definition floor\n(\u22653 consecutive days)",
            fontsize=6.0, color="0.4", va="bottom")
    ratio = ps["Heatwaves"] / ps["Cold waves"]
    ax.annotate(f"at D = 7 heat is {ratio:.1f}\u00d7 more likely",
                xy=(7, ps["Heatwaves"]), xytext=(9.0, 0.32), fontsize=6.6,
                color="0.2",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))

    # ------------------------------------------------------------------
    # (b) conditional co-occurrence P(row | col)
    # ------------------------------------------------------------------
    # Jaccard was replaced here. It is symmetric, so it answered a question
    # nobody asks, and its ceiling is set by the base rates -- WBGT>28 occurs on
    # 6.6 % of city-days against 12.8 % for EHD, so even perfect nesting caps
    # Jaccard at 0.52 and a value of 0.09 cannot be read on its own. P(row|col)
    # is asymmetric, fills the whole square, and reads directly as "if a day is
    # flagged by the column criterion, how often is it also flagged by the row
    # criterion".
    ax = axes[0, 1]
    im, M, mask = criteria_matrix(fig, ax, ev, spec, label_size=5.6,
                                  value_size=4.2)
    ax.set_xlabel("given this criterion (column)", fontsize=7.5)
    ax.set_ylabel("probability of this one (row)", fontsize=7.5)
    ax.set_title("b  P(row | column) among city-days", loc="left",
                 fontweight="bold", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    handles = [plt.Line2D([], [], marker="s", ls="", markersize=5,
                          color=FAM_BAND[f], markeredgecolor=FAM_TEXT[f],
                          label=FAM_NAME[f])
               for f in ("heat", "cold", "humid", "absolute")]
    # Anchored to the axes, not floating: at ncol=2 and a left anchor the
    # legend drifted into the gap between panels b and d and read as if it
    # belonged to neither.
    ax.legend(handles=handles, fontsize=5.6, frameon=False, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.26),
              handletextpad=0.4, columnspacing=1.0)

    # ------------------------------------------------------------------
    # (c) event-aligned exposure trajectory
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    tau = pd.read_parquet(TAU)
    samp = ("strict_clean" if "strict_clean" in
            tau.get("sample", pd.Series(dtype=object)).unique()
            else tau["sample"].unique()[0])
    tj = tau[tau["sample"] == samp].sort_values("tau_hw")
    for c, lab, col in [("tmax_anom_dm", "Tmax anomaly (K)", "#c0392b"),
                        ("utci_max_dm", "UTCI max (K)", "#e67e22"),
                        ("Twb_max_dm", "Twb max (K)", "#2471a3")]:
        if c in tj.columns:
            ax.plot(tj.tau_hw, tj[c], lw=1.7, color=col, label=lab)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.axvline(0, color="0.5", lw=0.7, ls="--")
    ax2 = ax.twinx()
    if "SCDH24_dm" in tj.columns:
        ax2.plot(tj.tau_hw, tj["SCDH24_dm"], lw=1.4, color="0.35", ls=":",
                 label="SCDH24 (K\u00b7h, right)")
        ax2.set_ylabel("cooling degree-hours", fontsize=8)
        ax2.tick_params(labelsize=7)
    ax.set_xlabel("event time \u03c4 (days from heatwave onset)", fontsize=8)
    ax.set_ylabel("deseasonalised anomaly (K)", fontsize=8)
    ax.set_title("c  Exposure around heatwave onset", loc="left",
                 fontweight="bold", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left", frameon=False)
    pk = tj.loc[tj.tmax_anom_dm.idxmax(), "tau_hw"]
    ax.annotate(f"peak \u03c4 = +{int(pk)}", xy=(pk, tj.tmax_anom_dm.max()),
                xytext=(pk + 4, tj.tmax_anom_dm.max() * 0.90), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
    ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # (d) heatwave days against cooling demand, in levels
    # ------------------------------------------------------------------
    # A rank slopegraph was tried first and abandoned: with 46 crossing lines
    # the country labels collide and nothing is legible. Plotting the levels is
    # also strictly more informative, because it shows the two facts at once --
    # there is no relationship, AND the two axes have completely different
    # dynamic ranges.
    ax = axes[1, 1]
    rho_all = C[["HW", "SCDH24"]].corr(method="spearman").iloc[0, 1]
    ab = C[C.tier.isin(["A", "B"])]
    rho_ab = ab[["HW", "SCDH24"]].corr(method="spearman").iloc[0, 1]
    for tier, fc in [("AB", "#c0392b"), ("C", "none")]:
        sub = ab if tier == "AB" else C[C.tier == "C"]
        ax.scatter(sub.HW, sub.SCDH24, s=26, facecolors=fc,
                   edgecolors="#c0392b", lw=0.9, alpha=0.85, zorder=3,
                   label="Tier A+B (in the response sample)" if tier == "AB"
                   else "Tier C (outside it)")
    ax.set_yscale("log")
    ax.set_xlabel("heatwave days per year (relative criterion)", fontsize=8)
    ax.set_ylabel("cooling degree-hours per year (K\u00b7h)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(lw=0.35, alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    off = {"India": (7, -3), "France": (-7, -11), "United Kingdom": (-7, -11),
           "Germany": (7, -3), "China": (7, 5), "United States": (-8, -12),
           "Japan": (-8, -10), "Brazil": (-8, 7), "Russia": (7, 4),
           "Spain": (7, 3), "Singapore": (-11, 6), "Nigeria": (-12, -12)}
    for _, r in C[C.country.isin(HILITE)].iterrows():
        dx, dy = off.get(r.country, (6, 6))
        ax.annotate(r.country, (r.HW, r.SCDH24), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.2, color="0.12",
                    ha="left" if dx > 0 else "right")
    ax.legend(fontsize=6.2, frameon=False, loc="lower left")
    ax.set_title(f"d  Heatwave days do not rank cooling demand\n"
                 f"    \u03c1 = {rho_all:+.2f} (46), {rho_ab:+.2f} (Tier A+B)",
                 loc="left", fontweight="bold", fontsize=9)

    fig.suptitle(
        "Quantitative structure of the exposure field: city-day taxonomy "
        "(a\u2013c) and the country-day series used in R3\u2013R6 (d)",
        fontsize=9.5)
    save(fig, "Fig2_criteria_and_trajectory")

    # ------------------------------------------------------------------
    # tables behind the panels
    # ------------------------------------------------------------------
    save_table(C.sort_values("SCDH24", ascending=False).round(3),
               "T1c_country_exposure_profile")
    base = {c: float(ev[c].mean()) for c in crit}
    save_table(pd.DataFrame(
        [{"panel": "b", "given": b, "then": a,
          "p_row_given_col": M[crit.index(a), crit.index(b)],
          "definitional": bool(mask[crit.index(a), crit.index(b)]),
          "base_rate_row": base[a], "base_rate_col": base[b]}
         for a in crit for b in crit]).round(5),
        "T1d_criteria_conditional_probability")
    save_table(pd.DataFrame(surv_rows).round(4), "T1e_event_duration_survival")

    print(f"      a  heat n=36,795 cold n=14,775  P(>=7) "
          f"{ps['Heatwaves']:.3f} vs {ps['Cold waves']:.3f}  "
          f"ratio {ratio:.2f}x")
    print(f"      b  {len(crit)} criteria, {int(mask.sum())} of "
          f"{len(crit) ** 2} cells fixed by definition")
    print(f"      d  Spearman(HW, SCDH24) = {rho_all:+.3f} (46), "
          f"{rho_ab:+.3f} (Tier A+B)")
    fr = C.set_index("country")
    print(f"         France {fr.HW['France']:.1f} d/yr and "
          f"{fr.SCDH24['France']:,.0f} K.h;  India {fr.HW['India']:.1f} d/yr "
          f"and {fr.SCDH24['India']:,.0f} K.h")


# ==========================================================================
# Fig S9  the complete Part III criterion matrix
# ==========================================================================
def si_criteria_matrix():
    """Every Part III event code against every other, at full size.

    Figure 2b shows thirteen codes because the cells there still have to carry
    printed numbers. This is the same statistic over all twenty-one usable
    codes, with the values in Table TS8 rather than in the cells, so nothing in
    the codebook is left to the reader's imagination.
    """
    spec = PART3
    codes = [c for c, _, _ in spec]
    ev = pd.read_parquet(EVENTS, columns=codes)

    fig, ax = plt.subplots(figsize=(8.6, 8.0), layout="constrained")
    im, M, mask = criteria_matrix(fig, ax, ev, spec, show_values=False,
                                  label_size=7.0)
    ax.set_xlabel("given this criterion (column)", fontsize=8.5)
    ax.set_ylabel("probability of this one (row)", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("P(row | column) among 2,312,763 city-days", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], marker="s", ls="", markersize=6,
                          color=FAM_BAND[f], markeredgecolor=FAM_TEXT[f],
                          label=FAM_NAME[f])
               for f in ("heat", "cold", "humid", "absolute")]
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=4,
              loc="upper left", bbox_to_anchor=(0.0, -0.13))
    fig.suptitle("Conditional co-occurrence of every Part III event code, "
                 "1,583 cities, 2019\u20132022.\nGrey cells are fixed by the "
                 "definitions rather than by the climate.", fontsize=9)
    save(fig, "FigS9_criteria_matrix")

    base = {c: float(ev[c].mean()) for c in codes}
    lab = {c: l for c, l, _ in spec}
    fam = {c: f for c, _, f in spec}
    save_table(pd.DataFrame(
        [{"given": b, "given_label": lab[b], "given_family": fam[b],
          "then": a, "then_label": lab[a], "then_family": fam[a],
          "p_row_given_col": M[codes.index(a), codes.index(b)],
          "definitional": bool(mask[codes.index(a), codes.index(b)]),
          "base_rate_row": base[a], "base_rate_col": base[b]}
         for a in codes for b in codes]).round(5),
        "TS8_criteria_conditional_probability")

    off = ~np.eye(len(codes), dtype=bool)
    hi = [i for i, c in enumerate(codes) if fam[c] in ("heat", "humid")]
    ci = [i for i, c in enumerate(codes) if fam[c] == "cold"]
    cross = M[np.ix_(hi, ci)]
    print(f"      {len(codes)} codes, {int(mask.sum())} of "
          f"{len(codes) ** 2} cells fixed by definition")
    print(f"      off-diagonal cells below 0.005: "
          f"{int((M[off] < 0.005).sum())} of {int(off.sum())}")
    print(f"      heat/humid vs relative-cold block: max "
          f"{np.nanmax(cross):.3f}, median {np.nanmedian(cross):.4f}")


# ==========================================================================
# Fig S8  country exposure profile and the day-night split
# ==========================================================================
def si_country_profile():
    """The two country panels that did not earn a place in Figure 2.

    Kept because both are genuine descriptive statistics of the regressor, and
    dropped from the main figure because neither carries an argument the main
    text needs. The heatmap is largely determined by its own row ordering and
    the day-night split is superseded by R4.4, which asks the same question of
    the response rather than of the exposure.
    """
    C = country_profile()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.4), layout="constrained",
                             width_ratios=[1.0, 1.15])

    # (a) percentile-rank profile
    ax = axes[0]
    P = C.sort_values("SCDH24", ascending=False).reset_index(drop=True)
    R = P[PROF].rank(pct=True) * 100
    im = ax.imshow(R.values, cmap="RdYlBu_r", vmin=0, vmax=100,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(PROF)))
    ax.set_xticklabels(PROF_LAB, fontsize=6.2)
    TC = "\u25e6"          # open circle: Tier C, outside the response sample
    ax.set_yticks(range(len(P)))
    ax.set_yticklabels([c + (" " + TC if tt == "C" else "")
                        for c, tt in zip(P.country, P.tier)], fontsize=5.2)
    ax.tick_params(length=1.5)
    ax.set_title("a  Country exposure profile, percentile rank within each "
                 "metric\n    rows ordered by cooling degree-hours",
                 loc="left", fontweight="bold", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("percentile among 46 countries", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)

    # (b) night share
    ax = axes[1]
    F = C[C.SCDH24 >= MIN_COOL].sort_values("ns_cool").reset_index(drop=True)
    y = np.arange(len(F))
    ax.barh(y, F.ns_cool, color="#e67e22", height=0.72,
            label="cooling degree-hours")
    ax.scatter(F.ns_wb, y, s=16, color="#2471a3", zorder=3, lw=0,
               label=f"wet-bulb degree-hours ({int(F.ns_wb.notna().sum())} "
                     f"countries above the floor)")
    ax.set_yticks(y)
    ax.set_yticklabels([c + (" " + TC if tt == "C" else "")
                        for c, tt in zip(F.country, F.tier)], fontsize=5.4)
    ax.set_xlabel("share of the annual total accruing at night (%)",
                  fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", lw=0.35, alpha=0.45)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    ax.set_title("b  How much of the load is nocturnal", loc="left",
                 fontweight="bold", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Country-level description of the population-weighted "
                 "exposure series, 46 countries, 2019\u20132022. "
                 f"{TC} = Tier C, outside the response sample.", fontsize=9)
    save(fig, "FigS8_country_exposure")

    # How much of the heatmap is its own row ordering? This is why the panel is
    # in the supplement rather than the main figure, so it is measured rather
    # than asserted.
    key = R["SCDH24"]
    align = pd.DataFrame({"metric": PROF,
                          "rho_with_row_order": [R[c].corr(key, method="spearman")
                                                 for c in PROF]})
    X = ((R - R.mean()) / R.std()).values
    ev = np.linalg.svd(X, compute_uv=False) ** 2
    ev = ev / ev.sum()
    align["pc1_share"] = ev[0]
    align["pc2_share"] = ev[1]
    save_table(align.round(4), "TS7_country_profile_structure")
    print(f"      profile: {int((align.rho_with_row_order.abs() > 0.6).sum())}"
          f"/{len(PROF)} exposure metrics track the row ordering at |rho| > 0.6; "
          f"PC1 {100 * ev[0]:.0f}%, PC1+PC2 {100 * (ev[0] + ev[1]):.0f}%")
    print(f"      night share of cooling load {F.ns_cool.min():.1f}% "
          f"({F.country.iloc[0]}) to {F.ns_cool.max():.1f}% "
          f"({F.country.iloc[-1]}), {len(F)} countries above "
          f"{MIN_COOL:.0f} K.h/yr")


# ==========================================================================
# Fig 3  benchmark comparison (R2)
# ==========================================================================
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    # (a) out-of-sample R2 of the four benchmarks
    ax = axes[0]
    b = pd.read_csv(BENCH_FMT.format(t=TIER_MAIN))
    b["tier"] = TIER_MAIN
    try:
        bc = pd.read_csv(BENCH_FMT.format(t=TIER_ALT))
        bc["tier"] = TIER_ALT
        b = pd.concat([b, bc], ignore_index=True)
    except FileNotFoundError:
        pass
    m = b[(b["set"].str.startswith("matched")) & (b.tier == TIER_MAIN)].copy()
    lab = {"1 nat_area": "national\narea-wtd", "2 nat_pop": "national\npop-wtd",
           "3 fua_equal": "FUA\nequal-wtd", "4 fua_pop": "FUA\npop-wtd"}
    m["lbl"] = m.benchmark.map(lab)
    x = np.arange(len(m))
    ax.bar(x - 0.19, m.oos_r2_loco * 1e3, 0.36, label="leave-one-country-out",
           color="#4c72b0")
    ax.bar(x + 0.19, m.oos_r2_loyo * 1e3, 0.36, label="leave-one-year-out",
           color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(m.lbl, fontsize=7)
    ax.set_ylabel("out-of-sample R²  (×10⁻³)")
    ax.set_title(f"a  Which exposure predicts power CO₂?  (Tier {tier_label(TIER_MAIN)})",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    # (b) population vs area weighting shifts national temperature
    ax = axes[1]
    nb = pd.read_parquet(NATBENCH)
    # NB: do not name a column "pop" -- it collides with DataFrame.pop
    s = nb[nb.date.dt.year == 2020].groupby("country").agg(
        area=("nat_area_tmax", "mean"), popw=("nat_pop_tmax", "mean"))
    s["diff"] = s["popw"] - s["area"]
    s = s.sort_values("diff")
    show = pd.concat([s.head(6), s.tail(6)])
    col = ["#2471a3" if v < 0 else "#c0392b" for v in show["diff"]]
    ax.barh(range(len(show)), show["diff"], color=col)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([c[:16] for c in show.index], fontsize=7)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("population-weighted − area-weighted Tmax (K)")
    ax.set_title("b  Where a country's people actually live", loc="left",
                 fontweight="bold")
    ax.grid(axis="x", lw=0.4, alpha=0.5)

    # (c) sample coverage and tier
    ax = axes[2]
    t = pd.read_csv(TIER)
    colors = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#d62728"}
    for tier in ["A", "B", "C"]:
        sub = t[t.tier == tier]
        ax.scatter(sub.n_fua, sub.coverage, s=26, color=colors[tier],
                   label=f"Tier {tier_label(tier)} (n={len(sub)})", alpha=0.85, lw=0)
    ax.set_xscale("log")
    ax.axhline(0.8, color="0.5", lw=0.7, ls="--")
    ax.axhline(0.5, color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("sampled FUAs per country (log)")
    ax.set_ylabel("urban-population coverage")
    ax.set_title("c  Sample representativeness", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.grid(lw=0.4, alpha=0.5)
    for c in ["China", "India", "United States", "Japan"]:
        r = t[t.country == c]
        if len(r):
            ax.annotate(c[:6], (r.n_fua.iloc[0], r.coverage.iloc[0]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    fig.tight_layout()
    save(fig, "Fig3_exposure_benchmarks")

    save_table(t, "T2_coverage_tier")
    save_table(b, "T3_benchmark_comparison")


# ==========================================================================
# Fig 4  sectoral dynamic response (R3)
# ==========================================================================
def fig4():
    prof = pd.read_csv(LAGPROF_FMT.format(t=TIER_MAIN))
    shp = pd.read_csv(SHAPES_FMT.format(t=TIER_MAIN))
    prof = prof[prof.form == "continuous"]
    shp_c = shp[shp.form == "continuous"]
    secs = ["Power", "Industry", "Ground Transport", "Aviation", "Residential"]
    secs = [s for s in secs if s in prof.sector.unique()]

    fig, axes = plt.subplots(2, len(secs), figsize=(2.55 * len(secs), 5.1),
                             sharex=True)
    for j, sec in enumerate(secs):
        for i, exp in enumerate(["heat", "cold"]):
            ax = axes[i, j]
            p = prof[(prof.sector == sec) & (prof.exposure == exp)].sort_values("lag")
            c = "#c0392b" if exp == "heat" else "#2471a3"
            ax.fill_between(p.lag, 100 * (p.beta - 1.96 * p.se),
                            100 * (p.beta + 1.96 * p.se), color=c, alpha=0.2, lw=0)
            ax.plot(p.lag, 100 * p.beta, lw=1.6, color=c)
            ax.axhline(0, color="0.45", lw=0.7)
            ax.axvline(-0.5, color="0.45", lw=0.7, ls="--")
            r = shp_c[(shp_c.sector == sec) & (shp_c.exposure == exp)]
            if len(r):
                r = r.iloc[0]
                # A tick told the reader the pre-trend test had been run but
                # not how close it came. UTCI > 32 °C sits at p = 0.054, which
                # a tick reports as a clean pass and a reader would rightly
                # want to see. The p value is printed instead, and coloured so
                # a failure or a near miss is visible at a glance.
                pv = float(r.pretrend_p)
                pc = "#a93226" if pv <= 0.05 else ("#b9770e" if pv <= 0.10
                                                   else "0.35")
                # beta(-1) and beta(0) are printed because the paper's
                # dynamic-shape claim is about their RATIO: the cold side is a
                # step (nothing the day before), the heat side is already
                # elevated. Quoting only the jump reverses that reading.
                try:
                    bm1 = 100 * float(p.loc[p.lag == -1, "beta"].iloc[0])
                    b0 = 100 * float(p.loc[p.lag == 0, "beta"].iloc[0])
                    lead = f"\nβ₋₁={bm1:+.2f}  β₀={b0:+.2f}"
                except (IndexError, KeyError):
                    lead = ""
                ax.text(0.03, 0.95,
                        f"Σβ={100*r.cum:+.2f}%  t={r.cum_t:+.1f}\n"
                        f"peak k={int(r.peak_lag)}" + lead,
                        transform=ax.transAxes, va="top", fontsize=6.4)
                ptxt = "p<0.001" if pv < 0.001 else f"p={pv:.3f}"
                ax.text(0.03, 0.78, f"pre-trend {ptxt}",
                        transform=ax.transAxes, va="top", fontsize=6.4,
                        color=pc)
            if i == 0:
                ax.set_title(sec, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{'Heat' if exp=='heat' else 'Cold'} response (%)",
                              fontsize=8)
            if i == 1:
                ax.set_xlabel("lag k (days)", fontsize=8)
            ax.tick_params(labelsize=7)
    n_c = int(shp.n_country.max()) if "n_country" in shp.columns else 28
    tier_lab = tier_label(TIER_MAIN)
    fig.suptitle("Sectoral CO₂ response to city-weighted thermal exposure "
                 f"(per SD, Tier {tier_lab}, {n_c} countries)",
                 fontsize=10, y=0.985)
    # Residential fails the pre-trend test on both sides, and that is expected
    # rather than surprising: in Carbon Monitor Cities the residential series is
    # allocated with a heating-degree-day model, so regressing it on heating
    # degree-hours partly recovers the allocation. Saying so on the figure
    # stops the red p values from reading as an unexplained defect.
    fig.text(0.5, 0.005,
             "Residential is not an independent test: its city allocation is "
             "built from a heating degree-day model, which is why its "
             "pre-trend fails on both sides.",
             ha="center", fontsize=6.6, color="0.3")
    fig.tight_layout(rect=[0, 0.028, 1, 0.965])
    save(fig, "Fig4_sectoral_dynamic_response")

    save_table(shp, "T4_sectoral_response")


# ==========================================================================
# Fig 5  which exposure metric carries the carbon signal (R3)
# ==========================================================================
def load_power():
    rows = []
    for f in sorted(glob.glob(str(CD / "carbon-monitor-cities-*-v0325.csv"))):
        n = Path(f).stem.replace("carbon-monitor-cities-", "").replace("-v0325", "")
        if "all-cities" in n or "GADM" in n or "Counties" in n:
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip().lower().replace("﻿", "") for c in d.columns]
        if "city" not in d.columns:
            continue
        d = d.rename(columns={"value (ktco2 per day)": "v"})
        d = d[d.sector == "Power"]
        d["country"] = n
        rows.append(d[["country", "date", "v"]])
    e = pd.concat(rows, ignore_index=True)
    e["date"] = pd.to_datetime(e["date"])
    return e.groupby(["country", "date"], as_index=False)["v"].sum()


def demean(X, facs, it=250, tol=1e-11):
    X = np.asarray(X, dtype="float64").copy()
    meta = [(f, np.bincount(f).astype(float), f.max() + 1) for f in facs]
    for _ in range(it):
        p = X.copy()
        for f, c, n in meta:
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(f, weights=X[:, j], minlength=n) / c)[f]
        if np.max(np.abs(X - p)) < tol:
            break
    return X


def fit_form(p: pd.DataFrame, col: str) -> dict:
    d = p.sort_values(["country", "date"]).reset_index(drop=True).copy()
    d["x"] = d[col] / d[col].std()
    g = d.groupby("country")["x"]
    nm = []
    for s in range(-LAGS, LAGS + 1):
        k = f"b{s:+d}"
        d[k] = g.shift(s)
        nm.append(k)
    d = d.dropna(subset=nm + ["y"]).reset_index(drop=True)
    f1 = pd.factorize(d.country + "_" + d.date.dt.to_period("M").astype(str))[0]
    f2 = pd.factorize(d.date)[0]
    f3 = pd.factorize(d.country + "_" + d.date.dt.dayofweek.astype(str))[0]
    Z = demean(np.column_stack([d[nm].values, d.y.values]), [f1, f2, f3])
    X, y = Z[:, :-1], Z[:, -1]
    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-10
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    inv = np.linalg.inv(XtX)

    def meat(gg):
        n = gg.max() + 1
        Xe = X * e[:, None]
        S = np.zeros((n, X.shape[1]))
        for j in range(X.shape[1]):
            S[:, j] = np.bincount(gg, weights=Xe[:, j], minlength=n)
        return S.T @ S

    g1 = pd.factorize(d.country)[0]
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(f2).astype(str))[0]
    M = meat(g1) + meat(f2) - meat(g12)
    w, Q = np.linalg.eigh(0.5 * (M + M.T))
    M = (Q * np.clip(w, 0, None)) @ Q.T
    V = inv @ M @ inv
    post = slice(LAGS, len(nm))
    cum = b[post].sum()
    cse = float(np.sqrt(V[post, post].sum()))
    lead = np.arange(0, LAGS)
    bl = b[lead]
    # After the PSD projection the lead block can be rank deficient (it is in
    # Tier C, where several countries contribute almost no variation). Use the
    # Moore-Penrose inverse and test against the actual rank rather than
    # letting a singular solve abort the run.
    Vl = V[np.ix_(lead, lead)]
    Vinv = np.linalg.pinv(Vl, rcond=1e-10)
    dof = int(np.linalg.matrix_rank(Vl, tol=1e-10))
    wald = float(bl @ Vinv @ bl)
    pv = float(1 - stats.chi2.cdf(wald, dof)) if dof > 0 else np.nan
    return {"cum": cum, "se": cse, "t": cum / cse,
            "pretrend_p": pv, "pretrend_dof": dof, "n": len(d)}


def fig5_one(tier: str, force: bool = False) -> pd.DataFrame:
    forms = [
        ("w_EHD", "Extreme heat day share", "relative"),
        ("w_HW", "Heatwave share", "relative"),
        ("w_CHHE", "Compound humid heat share", "absolute"),
        ("w_twbg_danger", "WBGT > 28 °C share", "absolute"),
        ("w_utci_strong", "UTCI > 32 °C share", "absolute"),
        ("w_CDH_Twb26", "Wet-bulb degree-hours", "continuous"),
        ("w_SCDH24", "Cooling degree-hours", "continuous"),
    ]
    cache = CACHE.with_name(f"r3_form_cache_{tier}.parquet")
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"]}[tier]

    # Cache key over the INPUTS, not just the file's existence. Keyed only on
    # existence, this cache silently served coefficients fitted on a superseded
    # panel: after the country-label fix added 730 country-days, T4 still
    # carried the pre-fix numbers and nothing in the log said so. The stamp
    # covers the exposure panel and every emissions file, so any change to
    # either invalidates the cache without anyone having to remember --force.
    def input_stamp() -> str:
        parts = []
        for f in [EXPOSURE] + sorted(Path(CD).glob("carbon-monitor-cities-*-v0325.csv")):
            try:
                st = f.stat()
                parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                parts.append(f"{f.name}:missing")
        return f"{tier}|" + "|".join(parts)

    stamp = input_stamp()
    if cache.exists() and not force:
        res = pd.read_parquet(cache)
        cached = res.attrs.get("input_stamp") if hasattr(res, "attrs") else None
        if cached is None and "input_stamp" in res.columns:
            cached = res["input_stamp"].iloc[0]
        if cached != stamp:
            print("      inputs changed since the cache was written — refitting")
            res = None
    else:
        res = None
    if res is None:
        pw = load_power()
        ex = pd.read_parquet(EXPOSURE)
        p = pw.merge(ex[ex.tier.isin(keep)], on=["country", "date"])
        p = p[p.v > 0].copy()
        p["y"] = np.log(p.v)
        rows = []
        for col, lab, kind in forms:
            if col not in p.columns:
                continue
            r = fit_form(p, col)
            rows.append({"tier": tier, "column": col, "label": lab,
                         "kind": kind, **r})
            print(f"      {lab:32s} Σβ={100*r['cum']:+7.3f}%  t={r['t']:+5.2f}")
        res = pd.DataFrame(rows)
        res["input_stamp"] = stamp
        res.to_parquet(cache, index=False)

    order = {"relative": 0, "absolute": 1, "continuous": 2}
    res = res.sort_values(["kind", "cum"], key=lambda s: s.map(order) if s.name == "kind" else s)
    colors = {"relative": "#9e9e9e", "absolute": "#c0392b", "continuous": "#2471a3"}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    y = np.arange(len(res))
    ax.barh(y, 100 * res.cum, xerr=196 * res.se, color=[colors[k] for k in res.kind],
            error_kw=dict(lw=0.9, capsize=2.5, ecolor="0.3"))
    ax.set_yticks(y)
    ax.set_yticklabels(res.label, fontsize=7.5)
    ax.axvline(0, color="0.4", lw=0.8)
    tl = tier_label(tier)
    ax.set_xlabel("cumulative power CO₂ response per SD of exposure (%), "
                  f"Tier {tl}", fontsize=8.5)
    ax.set_title("a  Cumulative response per exposure metric, by criterion "
                 "family", loc="left", fontweight="bold", fontsize=9.5)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    # The label used to sit a fixed 0.16 from the bar end, which put it on top
    # of the whisker whenever the interval was wide. It is offset from the end
    # of the error bar instead, and the axis is widened to make room, so no
    # label can overlap the interval it annotates.
    # Every label is placed to the right of the right-hand whisker, including
    # the ones on negative bars. Mirroring them to the left put them on top of
    # the metric names in the y axis.
    pad = 0.03 * max(1.0, float((100 * res.cum.abs() + 196 * res.se).max()))
    for i, r in enumerate(res.itertuples()):
        right = 100 * r.cum + 196 * r.se
        pv = float(r.pretrend_p)
        pc = "#a93226" if pv <= 0.05 else ("#b9770e" if pv <= 0.10 else "0.35")
        ptxt = "p<0.001" if pv < 0.001 else f"p={pv:.3f}"
        note = f"t={r.t:+.1f}   pre-trend {ptxt}"
        if 0.05 < pv <= 0.10:
            note += " (borderline)"
        ax.text(right + pad, i, note, va="center", fontsize=6.6, color=pc,
                ha="left")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.62 * (hi - lo))
    hs = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in
          ["relative", "absolute", "continuous"]]
    ax.legend(hs, ["relative threshold", "absolute threshold", "continuous demand"],
              fontsize=6.8, frameon=False, loc="lower right")

    # (b) why the relative metrics fail: the same label, different loads
    #
    # An earlier version of this panel selected country-days with w_HW > 0.5 --
    # more than half the country's city weight in a heatwave at once -- and
    # then kept countries with at least twenty such days. That is a filter on
    # SPATIAL SYNCHRONY, not on heatwave incidence, and it silently dropped the
    # two largest and most climatically diverse countries in the sample: the
    # United States had 2 qualifying days and Australia 3, against 121 for
    # Norway. The panel therefore described small compact countries.
    #
    # The exposure-weighted mean below asks the same question without a
    # threshold: averaging cooling degree-hours with w_HW as the weight gives
    # the expected cooling load on a heatwave city-day, uses every country-day,
    # and covers all of Tier A+B.
    ax = axes[1]
    ex = pd.read_parquet(EXPOSURE)
    a = ex[ex.tier.isin(keep)]
    g = (a.groupby("country")
          .apply(lambda d: np.average(d.w_SCDH24, weights=d.w_HW)
                 if d.w_HW.sum() > 0 else np.nan, include_groups=False)
          .dropna().sort_values())
    ax.barh(range(len(g)), g.values, color="#c0392b", alpha=0.85)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([c[:15] for c in g.index], fontsize=6.2)
    ax.set_xlabel("cooling degree-hours on a heatwave city-day (K·h), "
                  "exposure-weighted mean", fontsize=8.5)
    ratio = float(g.max() / g.min())
    ax.set_title("b  Exposure-weighted cooling load on a heatwave "
                 "city-day, by country", loc="left", fontweight="bold",
                 fontsize=9.5)
    ax.text(0.97, 0.05,
            f"{g.index[-1]} {g.max():.0f} K·h vs {g.index[0]} {g.min():.1f}: "
            f"{ratio:.0f}-fold\nall {len(g)} Tier {tl} countries, no threshold "
            "on synchrony",
            transform=ax.transAxes, fontsize=6.4, color="0.3", ha="right",
            va="bottom", linespacing=1.4)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    # Two of the metrics carry a pre-trend, and the figure should say which and
    # what that leaves standing rather than leave two red p values unexplained.
    fig.text(0.005, 0.005,
             "Red: the metric fails the joint pre-trend test on lags −7…−1, so "
             "its coefficient is not read as a response. WBGT > 28 °C and "
             "cooling degree-hours pass, so the absolute-threshold and "
             "continuous-demand conclusions do not rest on a metric that fails.",
             ha="left", fontsize=6.2, color="0.3")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    # Tier A+B is Figure 4; the Tier C version is supplementary. It is a
    # uniform null across all seven metrics, so it cannot rank them -- it is a
    # statement about coverage, and coverage already has FigS10.
    save(fig, "Fig5_exposure_metric_tierAB" if tier == TIER_MAIN
         else "FigS13_exposure_metric_tierC")
    return res


def fig5(force: bool = False):
    parts = [fig5_one(t, force) for t in (TIER_MAIN, TIER_ALT)]
    res = pd.concat(parts, ignore_index=True)
    save_table(res, "T5_exposure_form_comparison")




# ==========================================================================
# Fig 6  does sample coverage decide whether the signal is detectable?
# ==========================================================================
def fig6():
    """Tier A+B versus Tier C.

    Tier C countries contribute one or two sampled cities each, so their
    national exposure series is measured with large error. If the coverage
    tiers mean what they are meant to mean, the response should be visible in
    Tier A+B and attenuated towards zero in Tier C -- an internal validation of
    the sampling design rather than a separate finding.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    # (a) Power lag profile, both tiers
    ax = axes[0]
    for tier, col in ((TIER_MAIN, "#c0392b"), (TIER_ALT, "#7f8c8d")):
        try:
            pr = pd.read_csv(LAGPROF_FMT.format(t=tier))
        except FileNotFoundError:
            continue
        p = pr[(pr.sector == "Power") & (pr.exposure == "heat")
               & (pr.form == "continuous")].sort_values("lag")
        ax.fill_between(p.lag, 100 * (p.beta - 1.96 * p.se),
                        100 * (p.beta + 1.96 * p.se), color=col, alpha=0.16, lw=0)
        ax.plot(p.lag, 100 * p.beta, lw=1.7, color=col, label=f"Tier {tier_label(tier)}")
    ax.axhline(0, color="0.45", lw=0.7)
    ax.axvline(-0.5, color="0.45", lw=0.7, ls="--")
    ax.set_xlabel("lag k (days)")
    ax.set_ylabel("power CO₂ response (%)")
    ax.set_title("a  Distributed lead-lag profile by coverage tier, Power",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(lw=0.4, alpha=0.5)

    # (b) cumulative response by sector and tier
    ax = axes[1]
    rows = []
    for tier in (TIER_MAIN, TIER_ALT):
        try:
            sh = pd.read_csv(SHAPES_FMT.format(t=tier))
        except FileNotFoundError:
            continue
        sh = sh[(sh.form == "continuous") & (sh.exposure == "heat")]
        for _, r in sh.iterrows():
            rows.append({"tier": tier, "sector": r.sector,
                         "cum": r.cum, "se": r.cum / r.cum_t if r.cum_t else np.nan})
    d = pd.DataFrame(rows)
    secs = ["Power", "Industry", "Ground Transport", "Aviation", "Residential"]
    secs = [s for s in secs if s in d.sector.unique()]
    x = np.arange(len(secs))
    for i, (tier, col) in enumerate(((TIER_MAIN, "#c0392b"), (TIER_ALT, "#7f8c8d"))):
        sub = d[d.tier == tier].set_index("sector").reindex(secs)
        ax.bar(x + (i - 0.5) * 0.38, 100 * sub.cum, 0.36,
               yerr=196 * sub.se.abs(), color=col, label=f"Tier {tier_label(tier)}",
               error_kw=dict(lw=0.8, capsize=2, ecolor="0.35"))
    ax.axhline(0, color="0.45", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in secs], fontsize=7)
    ax.set_ylabel("cumulative heat response (%)")
    ax.set_title("b  Cumulative heat response by sector and coverage tier",
                 loc="left",
                 fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    # (c) benchmark out-of-sample R2, both tiers
    ax = axes[2]
    frames = []
    for tier in (TIER_MAIN, TIER_ALT):
        try:
            b = pd.read_csv(BENCH_FMT.format(t=tier))
        except FileNotFoundError:
            continue
        b = b[b["set"].str.startswith("matched")].copy()
        b["tier"] = tier
        frames.append(b)
    if frames:
        bb = pd.concat(frames, ignore_index=True)
        lab = {"1 nat_area": "nat\narea", "2 nat_pop": "nat\npop",
               "3 fua_equal": "FUA\nequal", "4 fua_pop": "FUA\npop"}
        order = list(lab)
        x = np.arange(len(order))
        for i, (tier, col) in enumerate(((TIER_MAIN, "#4c72b0"), (TIER_ALT, "#b0b0b0"))):
            sub = bb[bb.tier == tier].set_index("benchmark").reindex(order)
            ax.bar(x + (i - 0.5) * 0.38, sub.oos_r2_loco * 1e3, 0.36,
                   color=col, label=f"Tier {tier_label(tier)}")
        ax.set_xticks(x)
        ax.set_xticklabels([lab[o] for o in order], fontsize=7)
        ax.axhline(0, color="0.45", lw=0.8)
        ax.set_ylabel("out-of-sample R²  (×10⁻³)")
        ax.set_title("c  Out-of-sample R² by weighting scheme and tier",
                     loc="left",
                     fontweight="bold")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    save(fig, "Fig6_tier_comparison")

    if len(d):
        save_table(d, "T6_tier_comparison")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--force", action="store_true")
    # Layout iteration at 600 dpi in three formats costs minutes per pass,
    # nearly all of it in the PDF and SVG writers. --fast renders PNG only at
    # screen resolution so a panel arrangement can be judged in seconds; it is
    # never used for the figures that go into the paper.
    ap.add_argument("--fast", action="store_true",
                    help="PNG only at 150 dpi, for checking layout")
    ap.add_argument("--plates", nargs="+", default=[],
                    help="Render only these atlas plates, e.g. S4 S7. "
                         "The TS6 summary is always written in full.")
    args = ap.parse_args()

    if args.plates:
        global _ATLAS_PLATES
        _ATLAS_PLATES = list(args.plates)

    if args.fast:
        global SAVE_FORMATS, SAVE_DPI
        SAVE_FORMATS, SAVE_DPI = ("png",), 150
        print("    --fast: PNG only at 150 dpi (not publication output)")
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    fns = {1: fig1, 2: fig2, 3: fig3, 4: fig4,
           5: lambda: fig5(force=args.force), 6: fig6, 7: fig7,
           8: si_map_atlas, 9: si_country_profile,
           10: si_criteria_matrix}
    for k in args.only:
        print(f"\n[Fig {k}]")
        fns[k]()
    print(f"\nDone in {time.time() - t0:.0f}s -> {RES}")




# ==========================================================================
# Fig 5  latent heat load as a distinct driver of the response (R4)
# ==========================================================================
# Path to the three-year result tree, used only by panel (d). The humidity
# terms are re-read from there rather than refitted: the two trees were
# produced by the same code on different emissions windows, which is exactly
# the comparison panel (d) is for.
RES_3Y = BASE / "Results"

# Terms carried in the main figure, and the reason any humidity term is kept
# out of it. Stating the exclusions in one place keeps the figure and the
# caption from drifting apart.
#
#   CHHE  (Twb >= 26 C) is significant in Power (t = 4.80) but fails the joint
#         pre-trend test at p < 0.001, so by the rule applied in Figure 3 its
#         coefficient is not read as a response.
#   CHHE+ (Twb >= 28 C) moves Aviation at t = 8.77 and Ground Transport at
#         t = 2.86. When both no-cooling sectors move with it, the term is
#         picking up a common factor, not a humidity channel.
#
# Both remain in T5 and in the supplement; they are not drawn as evidence.
HUMID_TERMS = [
    ("T5_latent_sensible", "latent", "latent degree-hours"),
    ("T5_latent_sensible", "sensible", "sensible degree-hours"),
    ("T1b_orth_humidity", "humid", "humidity at the same thermal load"),
    ("T5b_enthalpy", "enthalpy", "enthalpy (total heat)"),
    ("T2b_absolute_chhe", "chhe", "humid heat, Twb ≥ 26 °C"),
    ("T2b_absolute_chhe_plus", "chhe_plus", "humid heat, Twb ≥ 28 °C"),
]
EXCLUDED = {"chhe": "pre-trend p < 0.001",
            "chhe_plus": "negative controls move (Aviation t = 8.8)"}


def _humid_table(root: Path, tier: str = "AB") -> pd.DataFrame | None:
    f = root / "R4" / "tables" / f"T5_humidity_tests_tier{tier}.csv"
    return pd.read_csv(f) if f.exists() else None


def fig7():
    """Latent heat load, not humid-heat labels, is what the response tracks.

    WHY THE FIGURE IS BUILT THIS WAY
    --------------------------------
    An earlier version was titled "humidity amplification" and gave equal
    visual weight to five humidity metrics. That overstated what the tests
    support. The most direct test of amplification -- a humidity term entered
    beside cooling degree-hours, orthogonalised -- is +1.24 % (t = 2.24) over
    2019-2021 but +0.81 % (t = 1.67) over 2019-2022, so it is window-dependent
    in the same way the thermal asymmetry is base-temperature-dependent. It is
    reported as a sensitivity result, not as the finding.

    What survives both windows with clean negative controls is the split of
    thermal load into its latent and sensible parts. That is also the quantity
    Li et al. (Nature Cities, 2026) project forward when they report a 47 %
    rise in global urban latent load and warn that extrapolating from a
    temperature-only response relationship will misstate it. This figure
    supplies the observed response term that projection needs.
    """
    ab4 = _humid_table(RES)
    ab3 = None if RES.resolve() == RES_3Y.resolve() else _humid_table(RES_3Y)
    if ab4 is None:
        print("    run 20_r4_humidity.py first")
        return
    secs = [s for s in ["Power", "Industry", "Ground Transport", "Aviation",
                        "Residential"] if s in set(ab4.sector)]

    fig = plt.figure(figsize=(12.4, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.34,
                          wspace=0.26)

    # ---- (a) cooling and latent degree-hours entered jointly -------------
    #
    # Not a decomposition: w_SCDH24 and w_LDH_w are two separately constructed
    # series entered in the same distributed-lag regression. Calling it a
    # "split" would imply the two sum to a total, which they do not.
    ax = fig.add_subplot(gs[0, 0])
    cols = {"sensible": "#e67e22", "latent": "#2471a3"}
    w = 0.36
    xs = np.arange(len(secs))
    for i, term in enumerate(("sensible", "latent")):
        vals, errs, ts = [], [], []
        for s in secs:
            r = ab4[(ab4.sector == s) & (ab4.test == "T5_latent_sensible")
                    & (ab4.term == term)]
            vals.append(100 * r.cum.iloc[0] if len(r) else np.nan)
            errs.append(196 * r.se.iloc[0] if len(r) else np.nan)
            ts.append(r.t.iloc[0] if len(r) else np.nan)
        ax.bar(xs + (i - 0.5) * w, vals, w * 0.9, yerr=errs, color=cols[term],
               label=term, error_kw=dict(lw=0.7, capsize=2, ecolor="0.35"))
        for x0, v, t in zip(xs + (i - 0.5) * w, vals, ts):
            if np.isfinite(t) and abs(t) > 1.96:
                ax.text(x0, v + (0.35 if v >= 0 else -0.9), "*", ha="center",
                        fontsize=9, color=cols[term])
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace(" ", "\n") for s in secs], fontsize=7)
    ax.set_ylabel("cumulative CO₂ response per SD (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    ax.set_title("a  Cooling and latent degree-hours entered jointly, "
                 "by sector", loc="left", fontweight="bold", fontsize=9.5)
    pw_l = ab4[(ab4.sector == "Power") & (ab4.test == "T5_latent_sensible")
               & (ab4.term == "latent")].iloc[0]
    pw_s = ab4[(ab4.sector == "Power") & (ab4.test == "T5_latent_sensible")
               & (ab4.term == "sensible")].iloc[0]
    ax.text(0.02, 0.03,
            f"Power: latent {100 * pw_l.cum:+.2f} % (t = {pw_l.t:.2f}), "
            f"sensible {100 * pw_s.cum:+.2f} % (t = {pw_s.t:.2f})\n"
            f"latent is {100 * pw_l.cum / (100 * pw_s.cum):.0%} of sensible; "
            "* marks |t| > 1.96",
            transform=ax.transAxes, fontsize=6.2, color="0.3", va="bottom",
            linespacing=1.4)

    # ---- (b) the encompassing test ---------------------------------------
    #
    # The first objection to any humidity term is that it is temperature
    # measured on a finer grid. Three temperature series that differ only in
    # spatial resolution correlate at 0.94-0.99 within country and none
    # survives when two are entered together; the humidity term correlates at
    # 0.49 and does survive.
    ax = fig.add_subplot(gs[0, 1])
    enc = RES / "R3" / "tables" / "T4c_resolution_and_humidity_tierAB.csv"
    if not enc.exists():
        ax.text(0.5, 0.5, "run 31_r3r5_diagnostics.py first", ha="center",
                transform=ax.transAxes, fontsize=7)
    else:
        e = pd.read_csv(enc)
        ROWS = [("national temperature alone", "nat", "national T", "t",
                 "each entered alone"),
                ("FUA pop-weighted alone", "fua", "FUA T, pop", "t",
                 "each entered alone"),
                ("FUA equal-weighted alone", "fuae", "FUA T, equal", "t",
                 "each entered alone"),
                ("city temperature alone", "city", "city T", "t",
                 "each entered alone"),
                ("city WBGT>28 alone", "wbgt", "WBGT > 28 °C", "h",
                 "each entered alone"),
                ("national + FUA, both pop-weighted", "nat", "national T", "t",
                 "two temperatures together"),
                ("national + FUA, both pop-weighted", "fua", "FUA T, pop", "t",
                 "two temperatures together"),
                ("national temperature + city WBGT>28", "nat", "national T",
                 "t", "temperature with humidity"),
                ("national temperature + city WBGT>28", "wbgt",
                 "WBGT > 28 °C", "h", "temperature with humidity")]
        col = {"t": "0.45", "h": "#2471a3"}
        keep_r, labs, blocks = [], [], []
        for spec, term, lab, kind, blk in ROWS:
            r = e[(e.specification == spec) & (e.term == term)]
            if not len(r):
                continue
            r = r.iloc[0]
            keep_r.append((float(r.cum_pct), float(r.se_pct), float(r.t), kind))
            labs.append(lab)
            blocks.append(blk)
        seen: list[str] = []
        for b in blocks:
            if b not in seen:
                seen.append(b)
        ypos, head_at, cur = [], {}, 0.0
        for i, b in enumerate(blocks):
            if i == 0 or b != blocks[i - 1]:
                head_at[b] = cur
                cur += 1.0
            ypos.append(cur)
            cur += 1.0
        top = cur
        ypos = [top - v for v in ypos]
        head_at = {b: top - v for b, v in head_at.items()}
        for (v, se, t, kind), yy in zip(keep_r, ypos):
            c = col[kind]
            ax.plot([v - 1.96 * se, v + 1.96 * se], [yy, yy], color=c, lw=1.5,
                    zorder=2)
            ax.scatter([v], [yy], s=26, zorder=3, linewidths=1.0,
                       facecolors=c if abs(t) >= 1.96 else "white",
                       edgecolors=c)
            ax.text(v + 1.96 * se + 0.22, yy, f"t={t:+.2f}", fontsize=5.8,
                    va="center", color=c)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labs, fontsize=6.4)
        ax.set_ylim(-0.6, top + 0.6)
        for b in seen:
            ax.text(0.0, head_at[b], b, transform=ax.get_yaxis_transform(),
                    fontsize=6.2, color="0.3", style="italic", va="center",
                    ha="left")
        rr = e[e.specification == "within-country correlation"]
        rn = {str(x.term): float(x.t) for x in rr.itertuples()}
        ax.set_xlabel("Power response per SD (%), Tier A+B", fontsize=8)
        ax.tick_params(axis="x", labelsize=6.6)
        ax.set_title("b  Encompassing regressions: exposure terms alone "
                     "and in pairs", loc="left", fontweight="bold",
                     fontsize=9.5)
        ax.text(0.015, 0.015,
                "hollow marker: |t| < 1.96      within-country r: national vs "
                f"FUA temperature {rn.get('nat_pop_cdd vs fua_pop_cdd', np.nan):.2f}, "
                f"national temperature vs WBGT "
                f"{rn.get('nat_pop_scdh24 vs w_twbg_danger', np.nan):.2f}",
                transform=ax.transAxes, fontsize=5.8, color="0.35",
                va="bottom")
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.20 * (hi - lo))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---- (c) what the relative typing actually selects --------------------
    ax = fig.add_subplot(gs[1, 0])
    try:
        ev = pd.read_parquet(EVENTS, columns=["HEH", "DEH", "CHHE", "tmax_c",
                                              "Twb_max"])
        grp = [('relative\n"humid" (HEH)', ev[ev.HEH == 1]),
               ('relative\n"dry" (DEH)', ev[ev.DEH == 1]),
               ("absolute\nhumid (CHHE)", ev[ev.CHHE == 1])]
        pos = np.arange(len(grp))
        tm = [g.tmax_c.mean() for _, g in grp]
        tw = [g.Twb_max.mean() for _, g in grp]
        # These four means are quoted in the Results and in the caption. They
        # were previously computed here and nowhere else, so no table backed
        # them and the verifier could not check them. Persist them.
        save_table(pd.DataFrame({
            "definition": ["relative humid (HEH)", "relative dry (DEH)",
                           "absolute humid (CHHE)"],
            "n_city_days": [int(len(g)) for _, g in grp],
            "mean_tmax_c": np.round(tm, 3),
            "mean_twb_max_c": np.round(tw, 3)}),
            "T4d_definition_class_means")
        ax.bar(pos - 0.19, tm, 0.36, color="#e67e22", label="Tmax")
        ax.bar(pos + 0.19, tw, 0.36, color="#2471a3", label="Twb max")
        ax.set_xticks(pos)
        ax.set_xticklabels([g[0] for g in grp], fontsize=7)
        ax.set_ylabel("°C", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title("c  Mean Tmax and Twb of the selected city-days",
                     loc="left", fontweight="bold", fontsize=9.5)
        ax.legend(fontsize=6.8, frameon=False, loc="upper right",
                  bbox_to_anchor=(1.0, 0.90))
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.22 * (hi - lo))
        ax.text(0.5, 0.94,
                f'the relative "humid" class is {tm[1] - tm[0]:.1f} K COOLER '
                f'than the relative "dry" class',
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.0, color="#a93226")
        ax.text(0.5, -0.20,
                "a wet-bulb-depression tercile among hot days selects damp "
                "overcast days, not humid heat;\nthe absolute definition "
                "selects days that are both hot and humid",
                transform=ax.transAxes, fontsize=6.2, color="0.3",
                ha="center", va="top", linespacing=1.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    except Exception as exc:                                # noqa: BLE001
        ax.text(0.5, 0.5, str(exc), ha="center", transform=ax.transAxes)

    # ---- (d) which humidity terms replicate across the two windows --------
    #
    # Window length is an analyst choice of the same kind as the base
    # temperature in Figure 6, and it has to be shown rather than chosen. The
    # two trees differ only in the emissions window, so the comparison is
    # like for like.
    ax = fig.add_subplot(gs[1, 1])
    if ab3 is None:
        ax.text(0.5, 0.5, "panel (d) compares the 2019-2021 and 2019-2022\n"
                "result trees; it is drawn only for the latter",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=7, color="0.4")
        ax.set_axis_off()
    else:
        rows = []
        for test, term, lab in HUMID_TERMS:
            a = ab4[(ab4.sector == "Power") & (ab4.test == test)
                    & (ab4.term == term)]
            b = ab3[(ab3.sector == "Power") & (ab3.test == test)
                    & (ab3.term == term)]
            if len(a) and len(b):
                rows.append({"lab": lab, "term": term,
                             "t4": float(a.t.iloc[0]),
                             "t3": float(b.t.iloc[0])})
        rr = pd.DataFrame(rows).iloc[::-1]
        yy = np.arange(len(rr))
        for k, r in enumerate(rr.itertuples()):
            ax.plot([r.t3, r.t4], [k, k], color="0.75", lw=1.1, zorder=1)
        ax.scatter(rr.t3, yy, s=26, facecolors="white", edgecolors="#7d3c98",
                   lw=1.1, zorder=3, label="2019–2021")
        ax.scatter(rr.t4, yy, s=26, color="#7d3c98", zorder=3,
                   label="2019–2022")
        for v in (-1.96, 1.96):
            ax.axvline(v, color="0.55", lw=0.7, ls="--")
        ax.axvline(0, color="0.4", lw=0.8)
        # CHHE+ reaches t = 15.7, which would compress every term that the
        # panel is actually about into a thumbnail. The axis is clipped to the
        # range where the decision is made and the off-scale term is labelled
        # with its values, which is the honest version of not plotting it.
        inr = rr[~rr.term.isin(EXCLUDED) | (rr.t4.abs() < 6)]
        hi_x = float(max(6.0, inr[["t3", "t4"]].to_numpy().max() + 1.2))
        lo_x = float(min(-2.6, inr[["t3", "t4"]].to_numpy().min() - 0.8))
        for k, r in enumerate(rr.itertuples()):
            if r.t4 > hi_x or r.t3 > hi_x:
                ax.annotate(f"t = {r.t3:.1f} → {r.t4:.1f}",
                            xy=(hi_x, k), xytext=(-4, 0),
                            textcoords="offset points", fontsize=6.2,
                            color="#7d3c98", ha="right", va="center")
        ax.set_xlim(lo_x, hi_x)
        ax.set_yticks(yy)
        ax.set_yticklabels(
            [f"{r.lab}  ✗" if r.term in EXCLUDED else r.lab
             for r in rr.itertuples()], fontsize=6.6)
        ax.set_xlabel("Power t statistic", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        ax.legend(fontsize=6.8, frameon=False, loc="upper left",
                  bbox_to_anchor=(0.005, 0.995))
        ax.set_title("d  Power t statistic per term, two estimation windows",
                     loc="left", fontweight="bold", fontsize=9.5)
        excl = "; ".join(f"{v}" for v in EXCLUDED.values())
        ax.text(0.5, -0.16, "✗ excluded from the main claim: " + excl,
                transform=ax.transAxes, fontsize=6.0, color="0.35",
                ha="center", va="top")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Latent heat load as a component of the carbon response to "
                 "thermal exposure, Tier A+B", fontsize=10.5, y=0.965)
    save(fig, "Fig7_humidity_amplification")
    # No table written here: 20_r4_humidity.py already writes
    # T5_humidity_tests_tier{AB,C}.csv, and this was their concat.


if __name__ == "__main__":
    main()
