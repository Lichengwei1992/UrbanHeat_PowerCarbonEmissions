#!/usr/bin/env python
"""
44_megacity_table.py  --  named-city exposure values for R1.1
=============================================================

Figure 1 and Figures S4-S7 map the exposure field, but a map cannot be read
off: a reader cannot recover Tokyo's or Lagos's numbers from a coloured dot.
R1.1 claims that the three criterion families rank cities differently, and the
cleanest way to make that checkable is a table of named cities.

WHY THE ROWS ARE DEDUPLICATED BY FUACODE
----------------------------------------
The city list carries 1,583 rows over 1,563 FUA polygons: 14 polygons appear
under more than one name. Two of them matter for a population ranking.

  * Cairo and "Al Qahirah" are one polygon; ranking city rows shows each at
    13.7 M instead of the FUA's 27.4 M.
  * Guangzhou is split across EIGHT rows, each carrying one eighth of the
    population, which drops the 20.9 M FUA out of a top-20 list entirely.

`pop_mean` is the FUA total repeated on every shared row, so one row per
fuacode is the right unit here. This affects presentation only: the response
pipeline weights by `pop_weight_dedup`, which is already divided by n_share.

OUTPUT
------
  SI/tables/TS20_megacity_exposure.csv    the 20 most populous FUAs
  printed: the contrasts quoted in R1.1

USAGE
-----
  python 44_megacity_table.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
TAB = RESULTS / "SI" / "tables"
N_TOP = 20

# The city list keeps the source gazetteer's labels. These are the ones a
# reader would not recognise, or would read as a different place.
DISPLAY = {"Guangzhou1": "Guangzhou", "Al Qahirah": "Cairo",
           "New York (Greater)": "New York", "Los Angeles (Greater)": "Los Angeles",
           "Sao Paulo": "São Paulo"}

spec = importlib.util.spec_from_file_location("rf", BASE / "19_results_figures.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)

# criterion family -> (column, printed name, scale)
COLS = [
    ("relative", "n_EHD_days", "Extreme heat days", 1.0),
    ("relative", "n_HW_days", "Heatwave days", 1.0),
    ("relative", "n_CW_days", "Cold-wave days", 1.0),
    ("absolute", "n_utci_strong_days", "UTCI > 32 °C days", 1.0),
    ("absolute", "n_CHHE_days", "Compound humid-heat days", 1.0),
    ("continuous", "cum_scdh24", "Cooling degree-hours (10³ K·h)", 1e-3),
    ("continuous", "cum_cdh_twb26", "Wet-bulb degree-hours (10³ K·h)", 1e-3),
    ("continuous", "cum_hdh15", "Heating degree-hours (10³ K·h)", 1e-3),
]


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = rf.city_metrics()
    w = pd.read_parquet(BASE / "outputs/population/fua_city_weights.parquet")
    tier = pd.read_csv(BASE / "outputs/exposure/tables/country_tier.csv")

    m = d.merge(w[["city_id", "fuacode", "pop_mean"]], on="city_id", how="left")
    m = m.merge(tier[["country", "tier"]], on="country", how="left")
    n_rows = len(m)
    m = m.sort_values("city").drop_duplicates("fuacode", keep="first")
    print(f"    {n_rows} city rows -> {len(m)} FUA polygons "
          f"({n_rows - len(m)} duplicate names removed)")

    m["pop_M"] = m.pop_mean / 1e6
    m["city"] = m.city.replace(DISPLAY)
    out = m.nlargest(N_TOP, "pop_M")[
        ["city", "country", "pop_M", "tier"] + [c for _, c, _, _ in COLS]].copy()
    for _, c, _, k in COLS:
        out[c] = (out[c] * k).round(1)
    out["pop_M"] = out.pop_M.round(1)
    return out, m


def main() -> None:
    out, m = build()
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB / "TS20_megacity_exposure.csv", index=False)
    print(f"    wrote {TAB / 'TS20_megacity_exposure.csv'}\n")
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))

    # ---- the contrasts R1.1 quotes --------------------------------------
    print("\n    CONTRASTS ACROSS THE 20 MOST POPULOUS FUAs")
    for _, c, name, k in COLS:
        v = out[c]
        lo_c, hi_c = out.loc[v.idxmin(), "city"], out.loc[v.idxmax(), "city"]
        ratio = (v.max() / v.min()) if v.min() > 0 else float("inf")
        print(f"      {name:32s} {v.min():8.1f} ({lo_c}) .. {v.max():8.1f} "
              f"({hi_c})   spread {'inf' if ratio == float('inf') else f'{ratio:.0f}x'}"
              f"   zeros {int((v == 0).sum())}")

    # the inversion that carries the section
    a = out[out.city == "Mexico City"].iloc[0]
    b = out[out.city == "Bangkok"].iloc[0]
    print(f"\n      Mexico City vs Bangkok — the relative criterion inverts "
          f"the physical one:")
    print(f"        extreme heat days   {a.n_EHD_days:6.1f}  vs {b.n_EHD_days:6.1f}"
          f"   ({a.n_EHD_days / b.n_EHD_days:.1f}x the other way)")
    print(f"        cooling degree-hours{a.cum_scdh24:6.1f}  vs {b.cum_scdh24:6.1f}"
          f"   ({b.cum_scdh24 / a.cum_scdh24:.0f}x)")

    print("\n    SAMPLE CONTEXT (1,563 FUA polygons)")
    for _, c, name, k in COLS:
        s = m[c] * k
        print(f"      {name:32s} p10 {s.quantile(.1):8.1f}   median "
              f"{s.median():8.1f}   p90 {s.quantile(.9):8.1f}")


if __name__ == "__main__":
    main()
