#!/usr/bin/env python
"""
32_base_temperature_scan.py  --  is the cooling/heating asymmetry a real
                                 quantity, or a property of two chosen bases?
=============================================================================

R5 reports a pooled cooling-minus-heating difference in %/100 K.h. Cooling
degree-hours use base 24 C and heating degree-hours base 15 C. Both bases are
conventions carried in from the buildings literature, and the diagnostic run in
31_r3r5_diagnostics.py showed that moving the cooling base from 24 C to 18 C
takes the difference from +6.21 to -1.38 -- a sign change.

That first check was not conclusive, for two reasons this script removes.

  1. ONLY TWO COOLING BASES EXISTED. 08_build_moist_air.py stores degree-hours
     at fixed bases, so the check could compare 18 against 24 and nothing else.
     Here the hourly city temperature series is re-integrated at every base in
     the scan, which is the only way to see where a sign change actually sits.

  2. THE COUNTRY SET MOVED WITH THE BASE. R5 admits a country to a side only if
     it has at least 100 days above a load threshold. Lowering the cooling base
     raises every value, so more countries qualify (368 country-slopes at base
     24, 432 at base 18). Coefficients and sample were both changing. The scan
     pins the country set to the one the published specification uses and holds
     it fixed across every cell.

WHAT IT DOES NOT SETTLE
-----------------------
A base is not only a scale factor: it changes which hours enter the integral,
so cooling degree-hours on base 16 C include mild evenings that carry no
air-conditioning load in most of the sample. The scan shows how sensitive the
published number is to a choice; it does not identify a correct base, and no
cell of it should be quoted as the true asymmetry.

STAGES
------
  build     re-integrate hourly city temperature at every base, weight to
            country-days, cache per year               (heavy; ~1.2 M city-hours
                                                        per month, 48 months)
  validate  reproduce the published SCDH18 / SCDH24 / HDH15 columns from the
            rebuild and refuse to continue if they disagree
  scan      fit R5's country-specific-slope model on every base pair with the
            country set held fixed

USAGE
-----
  python 32_base_temperature_scan.py                     # all stages
  python 32_base_temperature_scan.py --stage build --years 2019 2020
  python 32_base_temperature_scan.py --stage scan
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
HOURLY = BASE / "ERA5" / "processed_hourly_polygon" / "t2m"
LOOKUP = BASE / "outputs" / "fua_city_lookup.csv"
WEIGHTS = BASE / "outputs" / "population" / "fua_city_weights.parquet"
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
CACHE = BASE / "outputs" / "_cache" / "base_scan"

# Bases to scan. 24 and 15 are the published pair and must stay in the grid so
# the scan contains the number the paper reports.
COOL_BASES = [16.0, 18.0, 20.0, 22.0, 24.0, 26.0]
HEAT_BASES = [12.0, 15.0, 18.0]
YEARS = [2019, 2020, 2021, 2022]

# The published pair, used to pin the country set.
PUB_COOL, PUB_HEAT = 24.0, 15.0


def cool_col(b: float) -> str:
    return f"cdh{b:g}"


def heat_col(b: float) -> str:
    return f"hdh{b:g}"


# --------------------------------------------------------------------------
def utc_offsets(lookup: pd.DataFrame) -> pd.Series:
    """Whole-hour UTC offset per city, exactly as 08_build_moist_air.py does.

    The local day boundary decides which hours fall in which day, so a
    different offset rule would produce degree-hours that are not comparable
    with the published columns. The validate stage checks that it did not.
    """
    try:
        from datetime import datetime

        import pytz
        from timezonefinder import TimezoneFinder

        tf = TimezoneFinder()
        ref = datetime(2020, 1, 15)
        off = {}
        for row in lookup.itertuples():
            name = tf.timezone_at(lat=float(row.lat), lng=float(row.lon))
            if name is None:
                off[row.city_id] = int(round(float(row.lon) / 15.0))
            else:
                off[row.city_id] = int(round(
                    pytz.timezone(name).utcoffset(ref).total_seconds() / 3600))
        print("    local time: timezonefinder (IANA zones, standard time)")
        return pd.Series(off, name="utc_offset")
    except ImportError:
        print("    local time: longitude/15 fallback -- install timezonefinder "
              "and pytz, or the rebuild will not match the published columns")
        return pd.Series((lookup["lon"] / 15.0).round().astype(int).values,
                         index=lookup["city_id"].values, name="utc_offset")


def build_year(year: int, offsets: pd.Series, w: pd.DataFrame) -> pd.DataFrame:
    """Country-day population-weighted degree-hours at every base, one year."""
    # The adjacent boundary months are read as well. A city at UTC+7 takes the
    # first hours of its local 1 January from 31 December UTC, which lives in
    # the previous year's file; building strictly year by year left those local
    # days short and the validate stage caught it as a spike on 1 January in
    # single-city tropical countries.
    files = sorted(glob.glob(str(HOURLY / str(year) /
                                 f"era5_t2m_{year}-*_city_hour.parquet")))
    if not files:
        raise SystemExit(f"no hourly files for {year} under {HOURLY}")
    for adj in (HOURLY / str(year - 1) / f"era5_t2m_{year - 1}-12_city_hour.parquet",
                HOURLY / str(year + 1) / f"era5_t2m_{year + 1}-01_city_hour.parquet"):
        if adj.exists():
            files.append(str(adj))
    out = []
    for f in files:
        d = pd.read_parquet(f, columns=["city_id", "datetime", "t2m"])
        off = d["city_id"].map(offsets).fillna(0).astype(int)
        d["date"] = (d["datetime"] + pd.to_timedelta(off, unit="h")).dt.floor("D")
        t = d["t2m"].to_numpy(dtype="float64")
        acc = {}
        for b in COOL_BASES:
            acc[cool_col(b)] = np.maximum(t - b, 0.0)
        for b in HEAT_BASES:
            acc[heat_col(b)] = np.maximum(b - t, 0.0)
        city_day = pd.DataFrame(acc)
        city_day["city_id"] = d["city_id"].to_numpy()
        city_day["date"] = d["date"].to_numpy()
        val = [c for c in city_day.columns if c not in ("city_id", "date")]
        city_day = city_day.groupby(["city_id", "date"], sort=False)[val].sum()
        out.append(city_day.reset_index())
        print(f"      {Path(f).name}: {len(d):,} city-hours", flush=True)

    cd = pd.concat(out, ignore_index=True)
    cd = cd.groupby(["city_id", "date"], as_index=False).sum()   # month seams
    cd = cd[cd["date"].dt.year == year]          # drop the borrowed edge days

    # population-weighted country mean, matching 14_national_exposure.py:
    #   w_X = sum(pop_weight_dedup * X) / sum(pop_weight_dedup)
    m = cd.merge(w, on="city_id", how="inner")
    val = [c for c in cd.columns if c not in ("city_id", "date")]
    num = pd.DataFrame({c: m[c].to_numpy() * m["wt"].to_numpy() for c in val})
    num["country"] = m["country"].to_numpy()
    num["date"] = m["date"].to_numpy()
    S = num.groupby(["country", "date"], sort=False)[val].sum().reset_index()
    D = m.groupby(["country", "date"], sort=False)["wt"].sum().reset_index()
    S = S.merge(D, on=["country", "date"])
    for c in val:
        S[c] = S[c] / S["wt"]
    return S.drop(columns=["wt"])


def stage_build(years: list[int]) -> None:
    print("\n[build] re-integrating hourly temperature at every base")
    CACHE.mkdir(parents=True, exist_ok=True)
    lk = pd.read_csv(LOOKUP, dtype={"city_id": str})
    lk["city_id"] = lk.city_id.str.zfill(4)
    offsets = utc_offsets(lk)

    w = pd.read_parquet(WEIGHTS)
    if "pop_weight_dedup" not in w.columns:
        w["pop_weight_dedup"] = w["pop_mean"] / w["n_share"]
    w["city_id"] = w.city_id.astype(str).str.zfill(4)
    w = w.rename(columns={"pop_weight_dedup": "wt"})[["city_id", "country", "wt"]]
    w = w[w.wt > 0]

    for y in years:
        f = CACHE / f"country_day_bases_{y}.parquet"
        if f.exists():
            print(f"    {y}: cached")
            continue
        print(f"    {y}:")
        build_year(y, offsets, w).to_parquet(f, index=False)
        print(f"    wrote {f.name}")


def load_scan_panel() -> pd.DataFrame:
    f = sorted(glob.glob(str(CACHE / "country_day_bases_*.parquet")))
    if not f:
        raise SystemExit("run --stage build first")
    d = pd.concat([pd.read_parquet(x) for x in f], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d


# --------------------------------------------------------------------------
def stage_validate(tier: str = "AB") -> bool:
    """Does the rebuild reproduce the published degree-hour columns?

    Judged on the estimation sample, not on all 46 countries, and in relative
    terms. The rebuild is not bit-identical and should not be expected to be:
    08_build_moist_air.py inner-joins ERA5 temperature against CHI wet-bulb, so
    it silently drops the 0.06 % of city-hours with no wet-bulb value, while a
    temperature-only rebuild keeps them. The effect is invisible where a country
    averages many cities and visible where it has one -- and every one-city
    country is Tier C, outside the R5 sample.
    """
    print(f"\n[validate] rebuild against the published panel, Tier {tier}")
    new = load_scan_panel()
    old = pd.read_parquet(EXPOSURE,
                          columns=["country", "date", "tier", "w_SCDH18",
                                   "w_SCDH24", "w_HDH15"])
    old["date"] = pd.to_datetime(old["date"])
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"]}[tier]
    old = old[old.tier.isin(keep)]
    m = new.merge(old, on=["country", "date"], how="inner")
    print(f"    {len(m):,} country-days matched")
    ok = True
    for newc, oldc in [(cool_col(18), "w_SCDH18"), (cool_col(24), "w_SCDH24"),
                       (heat_col(15), "w_HDH15")]:
        if newc not in m.columns:
            continue
        a, b = m[newc].to_numpy(float), m[oldc].to_numpy(float)
        good = np.isfinite(a) & np.isfinite(b)
        e = np.abs(a[good] - b[good])
        scale = max(float(np.mean(b[good])), 1e-9)
        r = float(np.corrcoef(a[good], b[good])[0, 1])
        rel99 = float(np.quantile(e, 0.99)) / scale
        flag = "ok " if (r > 0.9999 and rel99 < 0.10) else "BAD"
        print(f"    {flag} {newc:>7s} vs {oldc:<10s} r = {r:.7f}   "
              f"mean level {scale:8.2f} K.h   p99 |diff| "
              f"{100 * rel99:5.2f} % of the mean")
        ok &= flag == "ok "
    return ok


def validate_estimate(ox, p: pd.DataFrame) -> None:
    """The check that matters: does the rebuild reproduce R5's published cell?

    Column-by-column agreement is necessary but not sufficient. What the scan
    needs is that the rebuilt series, fed through R5's own estimator at the
    published bases, lands on the published coefficient.
    """
    q = p.copy()
    q["w_SCDH24"], q["w_HDH15"] = q[cool_col(PUB_COOL)], q[heat_col(PUB_HEAT)]
    r = ox.fit_r5(q)
    if r is None or "diff" not in r:
        print("    published cell did not converge on the rebuild")
        return
    g = {k: 100 * r[k]["cum"] for k in ("heat", "cold", "diff")}
    print(f"    rebuilt at base {PUB_COOL:g}/{PUB_HEAT:g}:  cooling "
          f"{g['heat']:+7.2f}  heating {g['cold']:+7.2f}  difference "
          f"{g['diff']:+7.2f} (z = {r['diff']['t']:+5.2f})")
    print(f"    published            :  cooling  +16.72  heating   +9.32  "
          f"difference   +6.21 (z = +1.43)")


# --------------------------------------------------------------------------
def load_25():
    """Reuse R5's estimator rather than restate it."""
    spec = importlib.util.spec_from_file_location(
        "ox", BASE / "25_oxcgrt_robustness.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ox"] = m
    argv = sys.argv
    sys.argv = ["25_oxcgrt_robustness.py"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


SECTORS = ["Power", "Industry", "Ground Transport", "Aviation",
           "Residential"]

# Three ways to define "one unit of thermal load". The first is what R5
# publishes; the other two exist because the first turned out to be the source
# of the base dependence.
#
#   per100   one unit = 100 K.h. Absolute and physical, but how rare 100 K.h is
#            depends entirely on the base, so the two sides are not on a common
#            footing and their ratio inherits the base pair.
#   perSD    one unit = one standard deviation of that side's own daily series.
#            Removes the pure scale effect of the base; anything left is
#            composition, i.e. the base changing which hours enter the integral.
#   perP95   one unit = that side's 95th percentile among days when it is
#            positive. Puts the two sides on a common PROBABILITY footing
#            rather than a common physical one -- which is the relative
#            convention this paper criticises elsewhere, and is reported for
#            exactly that reason: neither footing is privileged, and saying so
#            is the point.
NORMS = ["per100", "perSD", "perP95"]


def unit_scale(x: np.ndarray, norm: str) -> float:
    if norm == "per100":
        return 100.0
    if norm == "perSD":
        s = float(np.nanstd(x))
        return s if s > 0 else np.nan
    pos = x[np.isfinite(x) & (x > 0)]
    if pos.size < 100:
        return np.nan
    return float(np.quantile(pos, 0.95))


def scan_sector(ox, scan: pd.DataFrame, exmeta: pd.DataFrame, sector: str,
                tier: str, pairs=None) -> pd.DataFrame:
    em = ox.load_sector(sector)
    p = em.merge(exmeta, on=["country", "date"]).merge(
        scan, on=["country", "date"], how="inner")
    p = p[p.v > 0].copy()
    p["y"] = np.log(p.v)
    p = p.sort_values(["country", "date"]).reset_index(drop=True)

    # Country set pinned at the published base pair and per-100 units, then
    # held fixed across every base and every normalisation for this sector.
    q = p.copy()
    q["w_SCDH24"], q["w_HDH15"] = q[cool_col(PUB_COOL)], q[heat_col(PUB_HEAT)]
    ref = ox.fit_r5(q)
    if ref is None or "diff" not in ref:
        print(f"    {sector}: published specification did not converge")
        return pd.DataFrame()
    fh, fc = set(ref["countries_h"]), set(ref["countries_c"])
    print(f"    {sector}: {len(p):,} country-days, set pinned at "
          f"{len(fh)} cooling / {len(fc)} heating / {len(fh & fc)} both",
          flush=True)

    grid = pairs or [(cb, hb) for cb in COOL_BASES for hb in HEAT_BASES]
    rows = []
    for cb, hb in grid:
        rc = p[cool_col(cb)].to_numpy(float)
        rh = p[heat_col(hb)].to_numpy(float)
        for norm in NORMS:
            sc, sh = unit_scale(rc, norm), unit_scale(rh, norm)
            if not (np.isfinite(sc) and np.isfinite(sh)):
                continue
            q = p.copy()
            # fit_r5 divides by 100 internally, so pre-scale to land on raw/s
            q["w_SCDH24"] = rc * (100.0 / sc)
            q["w_HDH15"] = rh * (100.0 / sh)
            r = ox.fit_r5(q, fixed_h=fh, fixed_c=fc)
            if r is None or "diff" not in r:
                continue
            g = {k: 100 * r[k]["cum"] for k in ("heat", "cold", "diff")}
            rows.append({"sector": sector, "cool_base_c": cb,
                         "heat_base_c": hb, "normalisation": norm,
                         "cooling_unit": sc, "heating_unit": sh,
                         "cooling": g["heat"], "heating": g["cold"],
                         "difference": g["diff"], "diff_z": r["diff"]["t"],
                         "cooling_z": r["heat"]["t"], "heating_z": r["cold"]["t"],
                         "n_slope": r["n_slope"], "n": r["n"],
                         "published_pair": cb == PUB_COOL and hb == PUB_HEAT})
        print(f"      cool {cb:g} / heat {hb:g} done", flush=True)
    return pd.DataFrame(rows)


def stage_scan(tier: str, sectors: list[str], quick: bool) -> pd.DataFrame:
    print(f"\n[scan] Tier {tier}: base pairs x normalisations x sector")
    ox = load_25()
    scan = load_scan_panel()
    ex = pd.read_parquet(EXPOSURE)
    keep = {"A": ["A"], "AB": ["A", "B"], "C": ["C"]}[tier]
    exmeta = ex[ex.tier.isin(keep)][["country", "date", "tier"]]

    pairs = [(PUB_COOL, PUB_HEAT), (18.0, 18.0)] if quick else None
    if quick:
        print("    --quick: two base pairs only, for checking the machinery")

    out = []
    for s in sectors:
        r = scan_sector(ox, scan, exmeta, s, tier, pairs)
        if len(r):
            out.append(r)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(res):
        summarise(res)
    return res


def summarise(res: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("Range of the cooling-minus-heating difference across base pairs")
    print("=" * 78)
    print(f"    {'sector':18s}{'norm':8s}{'min':>9s}{'max':>9s}"
          f"{'spread':>9s}{'sig -':>7s}{'sig +':>7s}{'n':>5s}")
    for (s, nm), g in res.groupby(["sector", "normalisation"], sort=False):
        neg = int(((g.diff_z < -1.96)).sum())
        pos = int(((g.diff_z > 1.96)).sum())
        print(f"    {s:18s}{nm:8s}{g.difference.min():9.2f}"
              f"{g.difference.max():9.2f}{g.difference.max() - g.difference.min():9.2f}"
              f"{neg:7d}{pos:7d}{len(g):5d}")
    print("\n    A normalisation that removes the base dependence should show a")
    print("    small spread and no cells of opposite significant sign.")



# --------------------------------------------------------------------------
def make_figure(res: pd.DataFrame, tier: str) -> None:
    """One row per unit, one column per sector.

    The figure has to make a single comparison legible: under per-degree-hour
    units the difference crosses zero as the bases move, and under either
    scale-free unit it does not. Rows are therefore the units and the zero line
    is drawn in every panel, since "does this cross zero" is the question.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    sectors = [s for s in SECTORS if s in set(res.sector)]
    units = [u for u in NORMS if u in set(res.normalisation)]
    ulab = {"per100": "per 100 K·h", "perSD": "per SD",
            "perP95": "per p95-day"}
    hcol = {12.0: "#2471a3", 15.0: "#c0392b", 18.0: "#e67e22"}

    fig, axes = plt.subplots(len(units), len(sectors), figsize=(14.4, 7.2),
                             sharex=True, layout="constrained")
    axes = np.atleast_2d(axes)
    for i, u in enumerate(units):
        for j, s in enumerate(sectors):
            ax = axes[i, j]
            g = res[(res.normalisation == u) & (res.sector == s)]
            for hb, gh in g.groupby("heat_base_c"):
                gh = gh.sort_values("cool_base_c")
                c = hcol.get(float(hb), "0.4")
                ax.plot(gh.cool_base_c, gh.difference, lw=1.4, color=c,
                        zorder=2, label=f"heating base {hb:g} °C")
                sig = gh[gh.diff_z.abs() > 1.96]
                ns = gh[gh.diff_z.abs() <= 1.96]
                ax.scatter(sig.cool_base_c, sig.difference, s=22, color=c,
                           zorder=3)
                ax.scatter(ns.cool_base_c, ns.difference, s=22,
                           facecolors="white", edgecolors=c, lw=1.0, zorder=3)
            ax.axhline(0, color="0.35", lw=0.8)
            ax.grid(lw=0.3, alpha=0.4)
            ax.tick_params(labelsize=7)
            # The claim the figure has to support is arithmetic, not visual, so
            # the two numbers that carry it are printed in the panel: how far
            # the difference travels across the eighteen cells, and whether any
            # of that travel crosses zero significantly in both directions.
            if len(g):
                spread = float(g.difference.max() - g.difference.min())
                n_neg = int(((g.diff_z < -1.96)).sum())
                n_pos = int(((g.diff_z > 1.96)).sum())
                flips = "sign flips" if (n_neg and n_pos) else "one sign"
                # Rather than hunt for an empty corner in each of fifteen
                # panels, make one: the y limits are opened downwards so the
                # bottom fifth of every panel is guaranteed to be free.
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo - 0.26 * (hi - lo), hi)
                ax.text(0.03, 0.03,
                        f"spread {spread:.1f}\n{n_neg}−  {n_pos}+  ({flips})",
                        transform=ax.transAxes, fontsize=6.0, color="0.3",
                        va="bottom", linespacing=1.35)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(s, fontsize=8.6, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{ulab.get(u, u)}\ncooling − heating",
                              fontsize=8)
            if i == len(units) - 1:
                ax.set_xlabel("cooling base (°C)", fontsize=8)
            # the published cell, marked once
            if u == "per100" and s == "Power":
                pub = g[(g.cool_base_c == PUB_COOL)
                        & (g.heat_base_c == PUB_HEAT)]
                if len(pub):
                    y = float(pub.difference.iloc[0])
                    ax.annotate("published\n24 / 15 °C", (PUB_COOL, y),
                                textcoords="offset points", xytext=(-38, 14),
                                fontsize=6.4, color="0.15",
                                arrowprops=dict(arrowstyle="->", lw=0.6,
                                                color="0.4"))
    axes[0, 0].legend(fontsize=6.4, frameon=False, loc="upper left")
    fig.suptitle(
        "The cooling-minus-heating difference under 18 base pairs and three "
        "units, Tier A+B.\nFilled markers are significant at |z| > 1.96; the "
        "country set is held fixed at the published base pair.", fontsize=9)

    out = RESULTS / "SI" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    # Supplementary. The base dependence is the argument behind the unit
    # panel of Figure 6, not a result the main text has to carry: fifteen
    # panels on three different y scales cost a reader more than the single
    # range whisker that now sits on Figure 6e conveys.
    name = f"FigS12_base_scan_tier{tier}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"    SI/figures/{name}  [png, pdf, svg]  600 dpi")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "build", "validate", "scan", "figure"])
    ap.add_argument("--years", nargs="+", type=int, default=YEARS)
    ap.add_argument("--sector", default="Power")
    ap.add_argument("--tier", default="AB", choices=["A", "AB", "C"])
    ap.add_argument("--force", action="store_true",
                    help="scan even if validation fails")
    ap.add_argument("--sectors", nargs="+", default=SECTORS)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="two base pairs only, to check the machinery")
    args = ap.parse_args()

    print("=" * 78)
    print("Base-temperature scan of the cooling/heating asymmetry")
    print("=" * 78)

    if args.stage == "figure":
        f = RESULTS / "R5" / "tables" / f"T8b_base_scan_tier{args.tier}.csv"
        if not f.exists():
            raise SystemExit(f"{f} not found; run --stage scan first")
        make_figure(pd.read_csv(f), args.tier)
        return

    if args.stage in ("all", "build"):
        stage_build(args.years)
    if args.stage in ("all", "validate", "scan"):
        ok = stage_validate(args.tier)
        if not ok and not args.force:
            raise SystemExit("\nvalidation failed; rerun with --force to scan "
                             "anyway, but the numbers will not be comparable "
                             "with the published ones")
    if args.stage in ("all", "scan"):
        res = stage_scan(args.tier, args.sectors, args.quick)
        if len(res):
            if args.quick:
                out = RESULTS / "_diagnostics"
                out.mkdir(parents=True, exist_ok=True)
                f = out / "D2b_base_scan_quick.csv"
            else:
                # A full scan is a result, not a diagnostic: it is what shows
                # that the sign of the asymmetry is set by the base pair. The
                # T-number is provisional until the response sections are
                # renumbered.
                out = RESULTS / "R5" / "tables"
                out.mkdir(parents=True, exist_ok=True)
                f = out / f"T8b_base_scan_tier{args.tier}.csv"
            res.round(4).to_csv(f, index=False)
            print(f"\n    wrote {f}")
            if not args.quick and not args.no_figure:
                make_figure(res, args.tier)


if __name__ == "__main__":
    main()
