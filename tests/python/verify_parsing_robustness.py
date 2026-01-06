"""
Strict robustness verification for image parsing.

Requirement: All p4_* variants must parse IDENTICALLY to p4.png.
             All p5_* variants must parse IDENTICALLY to p5.png.

This checks:
- grid dimensions
- horse position
- per-cell labels (water/grass/cherry/horse)
- cherry cell list
- summary counts

Run:
  source venv/bin/activate
  python tests/python/verify_parsing_robustness.py
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIXTURES_DIR = os.path.join(ROOT_DIR, "tests", "fixtures")
sys.path.insert(0, ROOT_DIR)

from hp_image import parse_image_to_grid


def _diff_grids(a: List[List[str]], b: List[List[str]]) -> List[Tuple[int, int, str, str]]:
    diffs: List[Tuple[int, int, str, str]] = []
    h = len(a)
    w = len(a[0]) if h else 0
    for y in range(h):
        for x in range(w):
            if a[y][x] != b[y][x]:
                diffs.append((x, y, a[y][x], b[y][x]))
    return diffs


def check_family(prefix: str, base_path: str) -> None:
    base = parse_image_to_grid(base_path)
    base_cherries = sorted(base.cherry_cells)
    base_counts = (base.debug["water_count"], base.debug["grass_count"], len(base.cherry_cells))

    files = sorted(glob.glob(os.path.join(FIXTURES_DIR, f"{prefix}_*.png")))
    if not files:
        print(f"[WARN] No files found for {prefix}_*.png in {FIXTURES_DIR}")
        return

    print(f"\n=== {prefix} family ===")
    print(f"base={base_path} grid={base.width}x{base.height} horse={base.horse} water/grass/cherries={base_counts}")

    failed = 0
    for path in files:
        pg = parse_image_to_grid(path)

        ok = True
        if (pg.width, pg.height) != (base.width, base.height):
            ok = False
            print(f"[FAIL] {path}: grid {pg.width}x{pg.height} != {base.width}x{base.height}")
        if pg.horse != base.horse:
            ok = False
            print(f"[FAIL] {path}: horse {pg.horse} != {base.horse}")

        diffs = _diff_grids(base.grid, pg.grid)
        if diffs:
            ok = False
            print(f"[FAIL] {path}: {len(diffs)} cell diffs (showing up to 30):")
            for d in diffs[:30]:
                print("   ", d)

        cherries = sorted(pg.cherry_cells)
        if cherries != base_cherries:
            ok = False
            # show symmetric diff
            a = set(base_cherries)
            b = set(cherries)
            only_a = sorted(a - b)[:20]
            only_b = sorted(b - a)[:20]
            print(f"[FAIL] {path}: cherry cells differ (base={len(base_cherries)} vs {len(cherries)})")
            if only_a:
                print("   only in base (up to 20):", only_a)
            if only_b:
                print("   only in variant (up to 20):", only_b)

        counts = (pg.debug["water_count"], pg.debug["grass_count"], len(pg.cherry_cells))
        if counts != base_counts:
            ok = False
            print(f"[FAIL] {path}: counts water/grass/cherries={counts} != {base_counts}")

        if ok:
            print(f"[OK]   {path}")
        else:
            failed += 1

    if failed:
        raise SystemExit(f"{prefix} family failed: {failed}/{len(files)} variants mismatched")


def main() -> None:
    check_family("p4", os.path.join(FIXTURES_DIR, "p4.png"))
    check_family("p5", os.path.join(FIXTURES_DIR, "p5.png"))
    print("\n✅ All parsing robustness checks passed.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"\n❌ {e}")
        sys.exit(1)

