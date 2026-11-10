#!/usr/bin/env python
"""
41_conceptual_framework_svg.py  --  the corrected conceptual framework figure
=============================================================================

Emits an editable SVG (plain shapes and text, no paths-as-text, no embedded
raster) so it can be opened in Illustrator / Inkscape and restyled to a
journal's template without redrawing.

WHY THIS REPLACES `Conceptual framework.png`
--------------------------------------------
The earlier drawing is a picture of the study as planned. Six of its claims
conflict with the finalised results; see `Conceptual_framework_review.md`. The
three that matter most:

  * it labels the panel as city-sector, but the dependent variable is national
    daily sectoral CO2 -- city-resolved emissions were ruled out because, with
    country x date fixed effects absorbed, a per-city slope identifies that
    city's meteorological representativeness of its country;
  * it groups Industry with Aviation as a null, when Industry is one of the two
    responding sectors (+2.74 % heat, +9.89 % cold per SD);
  * it draws four estimated effect modifiers, none of which this design can
    identify -- which is why R6 was retired.

It also omitted the paper's actual thesis: the three definitional choices.
Those are the spine of this version.

STRUCTURE
---------
  band 1   the pipeline:  exposure -> aggregation -> response -> what survives
  band 2   the three definitional forks, each anchored to the stage it acts on
  band 3   what the design cannot identify, drawn detached and without arrows
  band 4   the note bar (data, model, sample)

Every number printed on the figure is asserted in 30_verify_narrative.py.

USAGE
-----
  python 41_conceptual_framework_svg.py            # writes SVG (+ PNG if able)
"""
from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import escape

BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
OUT_DIR = RESULTS / "R1" / "figures"

W, H = 1760, 1050

# --------------------------------------------------------------------------
# palette -- kept close to the original drawing so the visual language carries
# --------------------------------------------------------------------------
BLU_D, BLU_M, BLU_F, BLU_S = "#1F4E79", "#2E75B6", "#DEEBF7", "#9DC3E6"
GRN_D, GRN_M, GRN_F, GRN_S = "#375623", "#548235", "#E2F0D9", "#A9D18E"
AMB_D, AMB_F, AMB_S = "#9C6500", "#FFF2CC", "#FFD966"
GRY_D, GRY_F, GRY_S = "#5A5A5A", "#F2F2F2", "#BFBFBF"
INK = "#222222"
FONT = "Inter, Helvetica Neue, Helvetica, Arial, sans-serif"

out: list[str] = []


# --------------------------------------------------------------------------
def rect(x, y, w, h, fill, stroke, rx=9, sw=1.4, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
               f'fill="{fill}" fill-opacity="{op}" stroke="{stroke}" '
               f'stroke-width="{sw}"{d}/>')


def text(x, y, lines, size=11, fill=INK, weight="normal", anchor="start",
         lh=14.5, style=""):
    """lines: str, or list of str / (str, dict-of-overrides)."""
    if isinstance(lines, str):
        lines = [lines]
    st = f' font-style="{style}"' if style else ""
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
               f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"'
               f'{st}>')
    for i, ln in enumerate(lines):
        ov = {}
        if isinstance(ln, tuple):
            ln, ov = ln
        a = "".join(f' {k}="{v}"' for k, v in ov.items())
        dy = 0 if i == 0 else lh
        out.append(f'<tspan x="{x}" dy="{dy}"{a}>{escape(ln)}</tspan>')
    out.append("</text>")


def arrow(x1, y1, x2, y2, color=BLU_D, sw=2.2, marker="ah", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
               f'stroke="{color}" stroke-width="{sw}"{d} '
               f'marker-end="url(#{marker})"/>')


def circle_num(cx, cy, n, fill, r=13):
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>')
    out.append(f'<text x="{cx}" y="{cy + 4.6}" font-family="{FONT}" '
               f'font-size="13" font-weight="700" fill="#FFFFFF" '
               f'text-anchor="middle">{n}</text>')


def band_label(x, y, n, title, color):
    circle_num(x + 13, y, n, color)
    text(x + 34, y + 5, title, size=14.5, weight="700", fill=color)


# ==========================================================================
def build() -> str:
    out.clear()
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="{FONT}">')
    out.append('<defs>')
    for name, col in (("ah", BLU_D), ("ahg", GRN_D), ("ahk", GRY_D),
                      ("aha", AMB_D)):
        out.append(
            f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>')
    out.append('</defs>')
    out.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # ======================================================================
    # BAND 1 -- the pipeline
    # ======================================================================
    Y0 = 34

    # ---------- stage A: defining exposure --------------------------------
    ax, aw = 30, 452
    rect(ax, Y0, aw, 362, "#FFFFFF", BLU_S, rx=12, sw=1.6)
    band_label(ax + 14, Y0 + 26, "a", "Defining exposure", BLU_D)
    text(ax + 16, Y0 + 52, "Three criterion families × three physical dimensions;",
         size=10.5, fill="#555555")
    text(ax + 16, Y0 + 66, "all combinations constructed, none privileged a priori.",
         size=10.5, fill="#555555")

    gx, gy, cw, ch, rlw = ax + 128, Y0 + 84, 100, 60, 112
    cols = ["Heat", "Humid heat", "Cold"]
    rows = [("Relative", "city p90 / p10 vs\n1991–2020", BLU_F, BLU_S),
            ("Absolute", "UTCI, WBGT, Twb\nfixed thresholds", BLU_F, BLU_S),
            ("Continuous", "degree-hours,\nformed hourly", BLU_F, BLU_S)]
    for j, c in enumerate(cols):
        text(gx + j * (cw + 6) + cw / 2, gy - 8, c, size=11, weight="700",
             fill=BLU_D, anchor="middle")
    cell = {
        (0, 0): "heatwave\nfrequency", (0, 1): "humid EHD\n(WBD tercile)",
        (0, 2): "cold-wave\nfrequency",
        (1, 0): "UTCI > 32 °C", (1, 1): "Twb ≥ 26 °C\ncompound",
        (1, 2): "UTCI < 0 °C",
        (2, 0): "cooling\ndegree-hours", (2, 1): "wet-bulb\ndegree-hours",
        (2, 2): "heating\ndegree-hours"}
    for i, (rl, rs, f, s) in enumerate(rows):
        yy = gy + i * (ch + 8)
        text(ax + 16, yy + 22, rl, size=11, weight="700", fill=BLU_D)
        text(ax + 16, yy + 36, rs.split("\n")[0], size=8.6, fill="#666666")
        text(ax + 16, yy + 46, rs.split("\n")[1], size=8.6, fill="#666666")
        for j in range(3):
            xx = gx + j * (cw + 6)
            rect(xx, yy, cw, ch, f, s, rx=6, sw=1.1)
            ls = cell[(i, j)].split("\n")
            text(xx + cw / 2, yy + (26 if len(ls) > 1 else 34), ls, size=9.4,
                 anchor="middle", lh=12, fill=BLU_D)
    text(ax + 16, Y0 + 300, "1,583 functional urban areas · 46 countries · 2019–2022",
         size=10, weight="700", fill=BLU_D)
    text(ax + 16, Y0 + 316,
         "Hourly ERA5 and CHI fields, averaged over FUA polygons.",
         size=9.4, fill="#666666")
    text(ax + 16, Y0 + 330,
         "Relative criteria are scored against each city's own climatology.",
         size=9.4, fill="#666666")

    # ---------- stage B: aggregation --------------------------------------
    bx, bw = 512, 300
    rect(bx, Y0, bw, 362, "#FFFFFF", BLU_S, rx=12, sw=1.6)
    band_label(bx + 14, Y0 + 26, "b", "Aggregating", BLU_D)
    text(bx + 16, Y0 + 52, "City exposure → the series the models regress on.",
         size=10.5, fill="#555555")

    rect(bx + 16, Y0 + 70, bw - 32, 54, BLU_F, BLU_S, rx=7)
    text(bx + bw / 2, Y0 + 92, ["Population-weighted mean",
                                "over functional urban areas"],
         size=11, anchor="middle", fill=BLU_D, lh=15)
    arrow(bx + bw / 2, Y0 + 126, bx + bw / 2, Y0 + 146)
    rect(bx + 16, Y0 + 148, bw - 32, 46, BLU_F, BLU_S, rx=7)
    text(bx + bw / 2, Y0 + 176, "National daily exposure series",
         size=11, anchor="middle", fill=BLU_D)
    arrow(bx + bw / 2, Y0 + 196, bx + bw / 2, Y0 + 216)

    rect(bx + 16, Y0 + 218, bw - 32, 80, "#FFFFFF", BLU_D, rx=7, sw=2.0)
    text(bx + bw / 2, Y0 + 240,
         ["The dependent variable is",
          "NATIONAL daily sectoral CO₂"],
         size=11, weight="700", anchor="middle", fill=BLU_D, lh=15)
    text(bx + bw / 2, Y0 + 272,
         ["With country × date effects absorbed, a per-city",
          "slope identifies that city's weather, not its",
          "energy system — so city emissions cannot be used."],
         size=8.8, anchor="middle", fill="#666666", lh=11)
    text(bx + 16, Y0 + 318, "Response sample — Tier A+B:",
         size=9.4, weight="700", fill=BLU_D)
    text(bx + 16, Y0 + 331, "28 countries, 40,908 country-days",
         size=9.4, weight="700", fill=BLU_D)
    text(bx + 16, Y0 + 345,
         "Tiers: share of national urban-centre population the cities capture.",
         size=8.8, fill="#666666")

    # ---------- stage C: response -----------------------------------------
    cx0, cwid = 842, 496
    rect(cx0, Y0, cwid, 362, "#FFFFFF", BLU_S, rx=12, sw=1.6)
    band_label(cx0 + 14, Y0 + 26, "c", "Estimating the response", BLU_D)

    rect(cx0 + 16, Y0 + 44, cwid - 32, 52, "#F7FAFD", BLU_S, rx=7, sw=1.1)
    text(cx0 + cwid / 2, Y0 + 64,
         ["Distributed lead–lag panel regression, lags −7 to +14",
          "FE: country × year-month  +  global date  +  country × day-of-week"],
         size=9.6, anchor="middle", fill=BLU_D, lh=13)

    sy = Y0 + 110
    # responders
    rect(cx0 + 16, sy, 232, 128, BLU_F, BLU_S, rx=8)
    text(cx0 + 28, sy + 20, "Thermal-comfort channel", size=10.5,
         weight="700", fill=BLU_D)
    for k, (nm, h, cd) in enumerate([
            ("Power", "heat +2.96 %  (t = 3.02)", "cold +9.91 %  (t = 3.74)"),
            ("Industry", "heat +2.74 %  (t = 2.89)", "cold +9.89 %  (t = 3.65)")]):
        yy = sy + 34 + k * 46
        text(cx0 + 28, yy + 12, nm, size=11, weight="700", fill=BLU_D)
        text(cx0 + 96, yy + 6, h, size=9.0, fill="#555555")
        text(cx0 + 96, yy + 18, cd, size=9.0, fill="#555555")
    text(cx0 + 28, sy + 122, "per SD of load", size=8.4, style="italic",
         fill="#888888")

    # negative controls
    rect(cx0 + 258, sy, 222, 84, GRN_F, GRN_S, rx=8)
    text(cx0 + 270, sy + 20, "Negative controls", size=10.5, weight="700",
         fill=GRN_D)
    text(cx0 + 270, sy + 38, "Ground Transport   −0.21 %  (t = −0.91)",
         size=9.0, fill="#4a5a3a")
    text(cx0 + 270, sy + 52, "Aviation   −0.08 %  (t = −0.38)",
         size=9.0, fill="#4a5a3a")
    text(cx0 + 270, sy + 70,
         "No cooling channel — they calibrate the null.",
         size=8.6, style="italic", fill="#6b7a5a")

    rect(cx0 + 258, sy + 92, 222, 52, GRY_F, GRY_S, rx=8, dash="5 3")
    text(cx0 + 369, sy + 110, "Residential — not interpreted",
         size=9.2, weight="700", anchor="middle", fill=GRY_D)
    text(cx0 + 369, sy + 123,
         ["its city allocation is built from a",
          "heating degree-day model"],
         size=8.2, anchor="middle", fill="#7a7a7a", lh=11)

    text(cx0 + 16, Y0 + 320,
         "Reported effect: cumulative Σβ over lags 0 to +14; leads are a",
         size=9.0, fill="#666666")
    text(cx0 + 16, Y0 + 333,
         "parallel-trends check (Wald χ²) and are never interpreted as effects.",
         size=9.0, fill="#666666")

    # ---------- stage D: what survives ------------------------------------
    dx, dwid = 1368, 362
    rect(dx, Y0, dwid, 362, BLU_D, BLU_D, rx=12)
    circle_num(dx + 27, Y0 + 26, "d", "#FFFFFF")
    out.append(f'<text x="{dx+27}" y="{Y0+31}" font-family="{FONT}" '
               f'font-size="13" font-weight="700" fill="{BLU_D}" '
               f'text-anchor="middle">d</text>')
    text(dx + dwid / 2, Y0 + 52, ["What survives", "every choice"],
         size=19, weight="700", anchor="middle", fill="#FFFFFF", lh=24)
    out.append(f'<line x1="{dx+40}" y1="{Y0+88}" x2="{dx+dwid-40}" y2="{Y0+88}" '
               f'stroke="#7FA6CC" stroke-width="1.2"/>')
    text(dx + dwid / 2, Y0 + 112, "The response is not one number.",
         size=13, weight="700", anchor="middle", fill="#FFFFFF")
    text(dx + dwid / 2, Y0 + 142,
         ["Between-country heterogeneity in the",
          "cooling−heating difference, Power, per SD:",
          "Cochran's Q = 240 on 17 df, I² = 92.9 %"],
         size=10.4, anchor="middle", fill="#D6E4F2", lh=15)
    rect(dx + 30, Y0 + 196, dwid - 60, 78, "#2A5E8C", "#5C8CB8", rx=7, sw=1.0)
    text(dx + dwid / 2, Y0 + 218,
         ["I² = 92.9–95.5 % in Power and 93.6–94.8 %",
          "in Industry, against 53.2–74.9 % in the two",
          "negative controls: heterogeneity is a property",
          "of the response, not of the estimator."],
         size=9.4, anchor="middle", fill="#E7F0F8", lh=13)
    text(dx + dwid / 2, Y0 + 300,
         ["A pooled slope is therefore a summary of a",
          "distribution, not an estimate of a common",
          "parameter."],
         size=10, anchor="middle", fill="#FFFFFF", lh=14)

    # ---------- pipeline arrows -------------------------------------------
    my = Y0 + 181
    arrow(ax + aw + 4, my, bx - 6, my, sw=2.6)
    arrow(bx + bw + 4, my, cx0 - 6, my, sw=2.6)
    arrow(cx0 + cwid + 4, my, dx - 6, my, sw=2.6)

    # ======================================================================
    # BAND 2 -- the three definitional forks
    # ======================================================================
    FY = 432
    circle_num(43, FY - 15, "e", AMB_D)
    text(64, FY - 10,
         "Three definitional choices — each changes the conclusion, "
         "and neither arm of any fork is correct in general",
         size=13, weight="700", fill=AMB_D)

    forks = [
        (30, 552, "1", "Defining exposure",
         "relative percentile", "absolute / continuous",
         "extreme-heat-day share\n−0.68 %  (t = −1.37)",
         "WBGT > 28 °C  +1.47 %  (t = 4.67)\ncooling degree-hours  +2.63 %  (t = 2.88)",
         "decides whether a carbon signal appears at all", "a"),
        (606, 552, "2", "Treating humidity",
         "a relative type", "a latent load",
         "within-city WBD tercile: the\n“humid” class is 11.5 K COOLER",
         "latent degree-hours  +1.80 %  (t = 2.24)\nbeside sensible  +2.42 %  (t = 2.18)",
         "decides whether humidity is visible as a dimension", "a and c"),
        (1182, 548, "3", "Normalising the two loads",
         "physical units (per 100 K·h)", "scale-free (per SD)",
         "cooling − heating  = +6.21\n(z = 1.43, not significant)",
         "cooling − heating  = −10.59\n(z = −4.06, significant, sign flips)",
         "decides the sign of the thermal asymmetry", "c"),
    ]
    for fx, fw, n, title, oa, ob, ea, eb, conseq, anchor_x in forks:
        rect(fx, FY, fw, 238, AMB_F, AMB_S, rx=10, sw=1.5)
        circle_num(fx + 26, FY + 26, n, AMB_D)
        text(fx + 48, FY + 31, title, size=13.5, weight="700", fill=AMB_D)
        text(fx + fw - 16, FY + 31, f"acts on panel {anchor_x}",
             size=9.4, style="italic", anchor="end", fill="#8a6a2a")
        half = (fw - 46) / 2
        for k, (lab, ev, col) in enumerate(
                [(oa, ea, "#B25C00"), (ob, eb, GRN_D)]):
            xx = fx + 16 + k * (half + 14)
            rect(xx, FY + 52, half, 84, "#FFFFFF", AMB_S, rx=7, sw=1.2)
            text(xx + half / 2, FY + 74, lab, size=11.5, weight="700",
                 anchor="middle", fill=col)
            out.append(f'<line x1="{xx+18}" y1="{FY+86}" x2="{xx+half-18}" '
                       f'y2="{FY+86}" stroke="{AMB_S}" stroke-width="1"/>')
            text(xx + half / 2, FY + 106, ev.split("\n"), size=9.4,
                 anchor="middle", fill="#555555", lh=13)
        arrow(fx + fw / 2, FY + 140, fx + fw / 2, FY + 162, color=AMB_D,
              sw=2.0, marker="aha")
        rect(fx + 16, FY + 166, fw - 32, 56, "#FFFFFF", AMB_D, rx=7, sw=1.8)
        text(fx + fw / 2, FY + 192, conseq, size=11.5, weight="700",
             anchor="middle", fill=AMB_D)
        text(fx + fw / 2, FY + 210,
             "the definition has to match the question",
             size=9.0, style="italic", anchor="middle", fill="#8a6a2a")

    # ======================================================================
    # BAND 3 -- what the design cannot identify (detached, no arrows)
    # ======================================================================
    NY = 700
    rect(30, NY, 1700, 128, GRY_F, GRY_S, rx=10, sw=1.5, dash="7 4")
    circle_num(59, NY + 21, "f", GRY_D)
    text(80, NY + 26,
         "Candidate effect modifiers — tested, and NOT identified in this design",
         size=12.5, weight="700", fill=GRY_D)
    text(80, NY + 44,
         "Drawn without arrows on purpose: a documented null is a result, "
         "not an omission (Supplementary Information 7).",
         size=9.6, style="italic", fill="#7a7a7a")
    chips = ["Urban form / built volume", "Cooling capacity",
             "Grid carbon intensity", "Income"]
    for k, c in enumerate(chips):
        xx = 46 + k * 196
        rect(xx, NY + 58, 184, 34, "#FFFFFF", GRY_S, rx=17, sw=1.2, dash="4 3")
        text(xx + 92, NY + 79, c, size=9.8, anchor="middle", fill=GRY_D)
    out.append(f'<line x1="{46+4*196-8}" y1="{NY+22}" x2="{46+4*196-8}" '
               f'y2="{NY+106}" stroke="{GRY_S}" stroke-width="1.2"/>')
    text(838, NY + 30,
         "Whether a city attribute can be tested in an aggregated design depends on "
         "how differently it ranks cities from population,",
         size=10, fill="#5a5a5a")
    text(838, NY + 46,
         "not on how physically interpretable it is. The four most physically "
         "motivated variants — sky view factor, building surface fraction,",
         size=10, fill="#5a5a5a")
    text(838, NY + 62,
         "impervious fraction and LCZ-implied height — all rank cities as "
         "population does (ρ ≈ 0.94–0.95) and admit no test.",
         size=10, fill="#5a5a5a")
    text(838, NY + 84,
         "The one variant that is testable, the sharpened LCZ compact share, "
         "returns θ = [−0.67, +0.77]: centred on zero, and it excludes",
         size=10, fill="#5a5a5a")
    text(838, NY + 100,
         "θ = 1, so the alternative weighting is rejected rather than merely "
         "unsupported. Urban scale cannot be tested; urban form shows no effect.",
         size=10, fill="#5a5a5a")

    # ======================================================================
    # BAND 4 -- note bar
    # ======================================================================
    NB = 852
    rect(30, NB, 1700, 172, "#FFFFFF", BLU_D, rx=10, sw=1.6)
    text(50, NB + 26, "Data and model", size=12, weight="700", fill=BLU_D)
    rows_note = [
        ("Exposure", "Hourly ERA5-Land (0.1°) and the CHI heat-indices dataset "
                     "(Malik et al. 2026, same 0.1° grid), averaged over 1,583 "
                     "FUA polygons; wet-bulb after Stull (2011)."),
        ("Baseline", "Daily Tmax/Tmin and the 1991–2020 climatology are ERA5 at "
                     "0.25°; they define the city- and calendar-day-specific "
                     "percentiles used by every relative criterion."),
        ("Emissions", "Carbon Monitor Cities, aggregated to national daily "
                      "sectoral CO₂ for 2019–2022; five sectors, of which "
                      "two carry no thermal-comfort channel."),
        ("Estimation", "Distributed lead–lag regression with three sets of "
                       "fixed effects; two-way cluster-robust covariance "
                       "(country and date) in R3–R4, Driscoll–Kraay "
                       "(bandwidth 14 days) in R5."),
        ("Weighting", "Population weighting throughout; five candidate weights "
                      "give cumulative responses spanning 0.17 percentage points, "
                      "so the choice is shown to be inconsequential."),
    ]
    for k, (h, b) in enumerate(rows_note):
        yy = NB + 48 + k * 24
        text(50, yy, h, size=10, weight="700", fill=BLU_M)
        text(140, yy, b, size=10, fill="#444444")

    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build()
    p = OUT_DIR / "Fig1_conceptual_framework.svg"
    p.write_text(svg, encoding="utf-8")
    print(f"    wrote {p}  ({len(svg)/1024:.0f} kB, {W}×{H})")
    try:
        import cairosvg
        for ext, kw in (("png", {"scale": 2}), ("pdf", {})):
            cairosvg.__dict__[f"svg2{ext}"](
                bytestring=svg.encode(), write_to=str(p.with_suffix(f".{ext}")),
                **kw)
            print(f"    wrote {p.with_suffix('.' + ext)}")
    except Exception as e:                                   # noqa: BLE001
        print(f"    (raster export skipped: {e})")


if __name__ == "__main__":
    main()
