#!/usr/bin/env python
"""
29_check_results_parity.py  --  does the new result tree match the published one?
=================================================================================

Compares a newly generated results root against `Results/` item by item and
reports anything missing, extra, or written in the wrong format.

WHY THIS EXISTS
---------------
A rerun that silently drops a figure produces a tree that still looks complete
at a glance -- the section folders are all there and full of files. The failure
modes that actually happen are specific and quiet:

  * a script omitted from the driver, so a whole item never appears
    (25_oxcgrt_robustness was missing at first: no FigS3, TS4, TS5);
  * a figure written in fewer than three formats;
  * a figure written below 600 dpi;
  * a stray flat `figures/` directory left by an older layout.

Each is checked explicitly rather than inferred from file counts.

WHAT IS AND IS NOT COMPARED
---------------------------
Only the SET of items and their output format. Values are expected to differ --
that is the point of the rerun -- so no numerical comparison is made here. Use
16_r3_dynamic_response.py on both windows for that.

USAGE
-----
  python 29_check_results_parity.py
  python 29_check_results_parity.py --new Results_2019_2022 --ref Results
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
FIG_EXT = {"png", "pdf", "svg"}
MIN_DPI = 600


def inventory(root: Path) -> dict[str, dict[str, set[str]]]:
    """section/kind -> stem -> set of extensions."""
    inv: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if not root.exists():
        return inv
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if len(rel.parts) < 3:
            continue                      # top-level notes such as the .md
        section, kind = rel.parts[0], rel.parts[1]
        inv[f"{section}/{kind}"][p.stem].add(p.suffix.lstrip(".").lower())
    return inv


def png_dpi(p: Path) -> float | None:
    """Read pHYs from a PNG; returns dots per inch, or None if absent."""
    try:
        with p.open("rb") as fh:
            if fh.read(8) != b"\x89PNG\r\n\x1a\n":
                return None
            while True:
                head = fh.read(8)
                if len(head) < 8:
                    return None
                length = int.from_bytes(head[:4], "big")
                ctype = head[4:8]
                data = fh.read(length)
                fh.read(4)                                  # CRC
                if ctype == b"pHYs":
                    ppux = int.from_bytes(data[0:4], "big")
                    unit = data[8]
                    return ppux * 0.0254 if unit == 1 else None
                if ctype == b"IDAT":
                    return None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="Results")
    ap.add_argument("--new", default="Results_2019_2022")
    args = ap.parse_args()

    ref_root, new_root = BASE / args.ref, BASE / args.new
    print("=" * 78)
    print(f"Results parity:  {args.new}  vs  {args.ref}")
    print("=" * 78)
    if not new_root.exists():
        raise SystemExit(f"\n{new_root} does not exist -- run 27_run_2019_2022.py first")

    ref, new = inventory(ref_root), inventory(new_root)
    problems = 0

    # Items that exist only in the reference tree BY DESIGN. R5 originally wrote
    # a single Power-only asymmetry table; it now writes one per sector, so the
    # unsuffixed name survives in the three-year tree as a superseded artefact.
    # Listing it here keeps the check at zero problems when nothing is wrong,
    # which is the only state in which a "1 problem" line means anything.
    EXPECTED_REF_ONLY = {
        ("R5/tables", "T6_thermal_asymmetry_tierAB"):
            "superseded by the per-sector T6_thermal_asymmetry_<sector>_tierAB",
    }

    print("\n[1] Item coverage by section")
    print(f"    {'section':16s}{'reference':>11s}{'new':>7s}   status")
    for sec in sorted(set(ref) | set(new)):
        r, n = set(ref.get(sec, {})), set(new.get(sec, {}))
        missing, extra = sorted(r - n), sorted(n - r)
        waived = [m for m in missing if (sec, m) in EXPECTED_REF_ONLY]
        missing = [m for m in missing if m not in waived]
        status = "ok"
        if missing:
            status = f"MISSING {len(missing)}"
            problems += len(missing)
        elif waived:
            status = f"ok ({len(waived)} superseded)"
        elif extra:
            status = f"extra {len(extra)}"
        print(f"    {sec:16s}{len(r):11d}{len(n):7d}   {status}")
        for m in missing:
            print(f"        missing: {m}")
        for m in waived:
            print(f"        reference only, expected: {m}")
            print(f"          {EXPECTED_REF_ONLY[(sec, m)]}")
        for e in extra:
            print(f"        extra  : {e}")

    print("\n[2] Figure formats (expect png + pdf + svg)")
    # Figures are also opened and re-exported by hand -- the SVGs in this tree
    # carry an Inkscape signature and JPGs appear beside them. A hand export is
    # not a pipeline failure and should not be counted as one, but a pipeline
    # figure that has lost a required format is a real gap. The two are
    # separated here so the problem count keeps meaning something.
    bad_fmt, hand = [], []
    for sec, items in sorted(new.items()):
        if not sec.endswith("/figures"):
            continue
        for stem, exts in sorted(items.items()):
            if FIG_EXT.issubset(exts):
                continue
            if not (exts & FIG_EXT):
                hand.append((sec, stem, sorted(exts)))   # export only
            else:
                bad_fmt.append((sec, stem, sorted(exts)))
    if bad_fmt:
        problems += len(bad_fmt)
        for sec, stem, exts in bad_fmt:
            print(f"    INCOMPLETE {sec}/{stem}: only {exts}")
            if "jpg" in exts or "jpeg" in exts:
                print("        a JPG sits beside it, so a manual export has "
                      "probably replaced the PNG")
    if hand:
        for sec, stem, exts in hand:
            print(f"    export only, not a pipeline figure: {sec}/{stem} "
                  f"{exts}")
    if not bad_fmt:
        print("    every pipeline figure is present in all three formats")

    print(f"\n[3] Raster resolution (expect >= {MIN_DPI} dpi)")
    low = []
    for p in sorted(new_root.rglob("*.png")):
        d = png_dpi(p)
        if d is not None and d < MIN_DPI - 1:
            low.append((p.relative_to(new_root), d))
    if low:
        problems += len(low)
        for rel, d in low:
            print(f"    LOW DPI {rel}: {d:.0f}")
    else:
        print("    no PNG below the threshold")

    print("\n[4] Layout")
    stray = [d for d in (new_root / "figures", new_root / "tables") if d.exists()]
    if stray:
        for d in stray:
            n = len(list(d.iterdir()))
            print(f"    stray flat directory {d.name}/ ({n} files) -- the published "
                  f"tree files everything under R1..R6/SI")
            if n == 0:
                d.rmdir()
                print("      (was empty, removed)")
            else:
                problems += 1
    else:
        print("    no stray flat directories")

    print("\n" + "=" * 78)
    if problems:
        print(f"{problems} problem(s) found -- see above")
        raise SystemExit(1)
    print("Parity OK: the new tree matches the published layout")


if __name__ == "__main__":
    main()
