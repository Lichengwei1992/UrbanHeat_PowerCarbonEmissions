#!/usr/bin/env python
"""
28_merge_emissions.py  --  splice the 2022 Carbon Monitor Cities release onto 2019-2021
=======================================================================================

Writes a 2019-2022 emissions directory in the same layout every downstream
loader already expects (`carbon-monitor-cities-<Country>-v0325.csv`), so the
whole pipeline can be pointed at it with HC_CARBON_DIR and nothing else changes.

WHY THIS SCRIPT IS NEEDED AT ALL
--------------------------------
Extending the CLIMATE panel to 2022 does not extend the RESPONSE analyses. Every
loader globs the v0325 per-country files, which stop at 2021-12-31, so a merge
against a four-year exposure panel silently yields a three-year regression
sample -- same n, same coefficients, no error. That failure mode is invisible in
the output, which is why the merge is made explicit here.

TWO RELEASES, NOT ONE SERIES
----------------------------
The 2022 data come from `CMC-2023-03-06-merged.csv` (figshare
10.6084/m9.figshare.22223596), a later release of Carbon Monitor Cities. It is
not a continuation of the v0325 files:

  * city coverage differs -- 1,207 cities appear in both, 390 only in v0325,
    99 only in the 2022 release;
  * levels jump at the seam. The January/December ratio of city means, which
    averages 1.02 for Power inside the old release, is 0.45 across the seam,
    0.83 for Industry and 1.35 for Residential.

A LEVEL SHIFT IS ABSORBED; A DYNAMICS SHIFT IS NOT
--------------------------------------------------
All response models carry country x year-month fixed effects, so a multiplicative
break that is constant within a country-month drops out of ln E exactly. What
those effects cannot absorb is a change in DAY-TO-DAY structure. The within-
country-month coefficient of variation is therefore reported per sector on both
sides of the seam: Power 0.098 (2021) against 0.121 (2022), Ground Transport
0.095 against 0.097. Read those before trusting a four-year coefficient, and run
16_r3_dynamic_response.py on 2019-2021 and 2019-2022 to confirm stability.

CITY SAMPLE
-----------
Restricted to the 1,207 cities present in both releases. Letting cities enter or
leave at 2022-01-01 would make the national aggregate jump for a compositional
reason on exactly the date the release changes, confounding the two.

INPUTS
------
  CarbonData/Near-real-time .../carbon-monitor-cities-<Country>-v0325.csv
  CarbonData/CMC-2023-03-06-merged.csv

OUTPUT
------
  CarbonData/merged_2019_2022/carbon-monitor-cities-<Country>-v0325.csv
  CarbonData/merged_2019_2022/_merge_report.csv

USAGE
-----
  python 28_merge_emissions.py
  python 28_merge_emissions.py --dry-run
then
  set HC_CARBON_DIR=D:\\Paper_Carbon and Heat\\CarbonData\\merged_2019_2022
"""

from __future__ import annotations

import argparse
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
CD_OLD = BASE / "CarbonData" / "Near-real-time daily estimates of CO2 emissions from 1500 cities worldwide"
NEW_FILE = BASE / "CarbonData" / "CMC-2023-03-06-merged.csv"
OUT_DIR = BASE / "CarbonData" / "merged_2019_2022"

SECTORS = ["Power", "Industry", "Ground Transport", "Aviation", "Residential"]


def load_old() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(CD_OLD / "carbon-monitor-cities-*-v0325.csv"))):
        n = Path(f).stem.replace("carbon-monitor-cities-", "").replace("-v0325", "")
        if "all-cities" in n or "GADM" in n or "Counties" in n:
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip().lower().replace("\ufeff", "") for c in d.columns]
        if "city" not in d.columns:
            continue
        d = d.rename(columns={"value (ktco2 per day)": "v"})
        d["country"] = n
        rows.append(d[["city", "country", "date", "sector", "v"]])
    e = pd.concat(rows, ignore_index=True)
    e["date"] = pd.to_datetime(e["date"])
    return clean_names(e, "v0325")


def clean_names(d: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalise city and country labels before anything is matched on them.

    The 2022 release ships two labels with invisible whitespace -- 'Belgium '
    with a trailing ASCII space and 'France\\xa0' with a non-breaking space.
    Left alone they do not match the exposure panel, so those two countries
    silently lose their 2022 emissions AND gain a phantom country file named
    after the corrupted string. Nothing errors; the panel just comes out 730
    country-days short. Every label is therefore normalised here, and any
    change is printed so a future release with new anomalies is visible rather
    than absorbed.
    """
    for col in ("city", "country"):
        if col not in d.columns:
            continue
        raw = d[col].astype(str)
        # \xa0 non-breaking,   figure,   narrow no-break, ​ zero width
        fixed = (raw.str.replace(r"[   ​]", " ", regex=True)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True))
        changed = sorted(set(raw[raw != fixed]))
        if changed:
            print(f"    {label}: normalised {len(changed)} {col} label(s):")
            for c in changed[:10]:
                print(f"      {c!r} -> {fixed[raw == c].iloc[0]!r}")
        d[col] = fixed
    return d


def load_new() -> pd.DataFrame:
    d = pd.read_csv(NEW_FILE, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={"value (KtCO2 per day)": "v"})
    d["date"] = pd.to_datetime(d["date"], format="%d/%m/%Y", errors="coerce")
    d = clean_names(d, "2022 release")
    return d[["city", "country", "date", "sector", "v"]].dropna(subset=["date"])


def seam_report(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Level jump and day-to-day variability on both sides of 2022-01-01."""
    rows = []
    for sec in SECTORS:
        o = old[old.sector == sec]
        n = new[new.sector == sec]
        dec21 = o[(o.date.dt.year == 2021) & (o.date.dt.month == 12)] \
            .groupby("city")["v"].mean()
        jan22 = n[(n.date.dt.year == 2022) & (n.date.dt.month == 1)] \
            .groupby("city")["v"].mean()
        dec20 = o[(o.date.dt.year == 2020) & (o.date.dt.month == 12)] \
            .groupby("city")["v"].mean()
        jan21 = o[(o.date.dt.year == 2021) & (o.date.dt.month == 1)] \
            .groupby("city")["v"].mean()
        within = (jan21 / dec20).replace([np.inf, -np.inf], np.nan).dropna()
        across = (jan22 / dec21).replace([np.inf, -np.inf], np.nan).dropna()

        def cv(df):
            g = df.copy()
            g["ym"] = g.date.dt.to_period("M")
            return g.groupby(["city", "ym"])["v"].apply(
                lambda x: x.std() / x.mean() if x.mean() else np.nan).median()

        rows.append({
            "sector": sec,
            "jan_dec_ratio_within_old": within.median(),
            "jan_dec_ratio_across_seam": across.median(),
            "within_month_cv_2021": cv(o[o.date.dt.year == 2021]),
            "within_month_cv_2022": cv(n),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("Merging the 2022 Carbon Monitor Cities release onto 2019-2021")
    print("=" * 78)

    print("\n[1] Loading")
    old, new = load_old(), load_new()
    print(f"    v0325       : {len(old):,} rows, {old.city.nunique():,} cities, "
          f"{old.date.min().date()} .. {old.date.max().date()}")
    print(f"    2022 release: {len(new):,} rows, {new.city.nunique():,} cities, "
          f"{new.date.min().date()} .. {new.date.max().date()}")

    print("\n[2] City sample")
    co, cn = set(old.city), set(new.city)
    common = sorted(co & cn)
    print(f"    in both      : {len(common):,}")
    print(f"    only in v0325: {len(co - cn):,}   (dropped)")
    print(f"    only in 2022 : {len(cn - co):,}   (dropped)")
    print("    the panel is restricted to the intersection so that no city")
    print("    enters or leaves on the same date the release changes")

    print("\n[3] Seam diagnostics")
    rep = seam_report(old[old.city.isin(common)], new[new.city.isin(common)])
    print(f"    {'sector':18s}{'Jan/Dec within old':>20s}{'across seam':>14s}"
          f"{'CV 2021':>10s}{'CV 2022':>10s}")
    for _, r in rep.iterrows():
        print(f"    {r.sector:18s}{r.jan_dec_ratio_within_old:20.3f}"
              f"{r.jan_dec_ratio_across_seam:14.3f}"
              f"{r.within_month_cv_2021:10.4f}{r.within_month_cv_2022:10.4f}")
    print("    level jumps are absorbed by country x year-month effects;")
    print("    a large CV change would NOT be, and would invalidate splicing")

    if args.dry_run:
        print("\n    dry run: nothing written")
        return

    print("\n[4] Writing merged per-country files")
    old_k = old[old.city.isin(common)].copy()
    new_k = new[new.city.isin(common)].copy()
    new_k = new_k[new_k.sector.isin(old_k.sector.unique())]
    merged = pd.concat([old_k, new_k], ignore_index=True)
    merged = merged.sort_values(["country", "city", "sector", "date"])

    # Stale files from an earlier run would otherwise survive: the phantom
    # 'Belgium ' and 'France\xa0' files stay on disk and are picked up by the
    # glob in every downstream loader.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("carbon-monitor-cities-*-v0325.csv"):
        old.unlink()

    n_files = 0
    for country, g in merged.groupby("country"):
        out = g[["city", "date", "sector", "v"]].rename(
            columns={"v": "value (KtCO2 per day)"})
        out = out.assign(date=out.date.dt.strftime("%Y-%m-%d"))
        safe = str(country).replace("/", "-")
        if safe != safe.strip() or "  " in safe:
            raise RuntimeError(f"country label still not clean: {safe!r}")
        out.to_csv(OUT_DIR / f"carbon-monitor-cities-{safe}-v0325.csv",
                   index=False)
        n_files += 1
    rep.to_csv(OUT_DIR / "_merge_report.csv", index=False)
    print(f"    {n_files} country files -> {OUT_DIR}")
    print(f"    {len(merged):,} rows, {merged.city.nunique():,} cities, "
          f"{merged.date.min().date()} .. {merged.date.max().date()}")

    # Completeness assertion. A country that silently loses one year of
    # emissions still produces a plausible-looking file, and the shortfall only
    # surfaces as a smaller n in the regression -- which is easy to miss. Every
    # country must now cover the full window on both sides of the seam.
    print("\n[4b] Per-country coverage")
    yrs = sorted(merged.date.dt.year.unique())
    cov = (merged.assign(y=merged.date.dt.year)
                 .groupby(["country", "y"]).size().unstack(fill_value=0))
    incomplete = cov[(cov == 0).any(axis=1)]
    print(f"    {len(cov)} countries x {len(yrs)} years "
          f"({yrs[0]}-{yrs[-1]})")
    if len(incomplete):
        print("    COUNTRIES MISSING A YEAR:")
        print(incomplete.to_string())
        raise RuntimeError(
            f"{len(incomplete)} country/countries do not span the full window. "
            "This is what a label mismatch looks like -- check for stray "
            "whitespace or spelling differences between the two releases.")
    print("    every country spans every year")

    print("\n[5] Point the pipeline at it")
    print(f'    set HC_CARBON_DIR={OUT_DIR}')
    print("    then rerun 27_run_2019_2022.py --from 8")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
