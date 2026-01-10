"""
Portal detection + solver correctness checks.

Run:
  source venv/bin/activate
  python tests/python/test_portals.py
"""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from hp_image import parse_image_to_grid
from milp_solver import solve_milp


def _fixtures_path(name: str) -> str:
    return os.path.join(ROOT_DIR, "tests", "fixtures", name)


def test_portal2_pairing() -> None:
    pg = parse_image_to_grid(_fixtures_path("portal2.png"))
    pairs = pg.debug.get("portal_pairs") or []
    assert len(pairs) == 8, f"expected 8 portal pairs in portal2.png, got {len(pairs)}"

    used = set()
    for a, b, _h in pairs:
        assert isinstance(a, tuple) and isinstance(b, tuple)
        ax, ay = a
        bx, by = b
        d = abs(ax - bx) + abs(ay - by)
        assert d == 2, f"expected manhattan distance 2, got {d} for pair {a} <-> {b}"
        assert a not in used, f"portal cell reused across pairs: {a}"
        assert b not in used, f"portal cell reused across pairs: {b}"
        used.add(a)
        used.add(b)


def test_portal_png_score() -> None:
    pg = parse_image_to_grid(_fixtures_path("portal.png"))
    status, score, area, cherries, walls = solve_milp(pg, k=12, time_limit_s=20.0, msg=False)
    assert status in ("Optimal", "Feasible"), f"unexpected MILP status: {status}"
    assert score == 7, f"expected score=7 on portal.png with k=12, got score={score} (area={area}, cherries={cherries})"
    assert len(walls) <= 12, f"expected <=12 walls, got {len(walls)}"


def test_portal_cherries_not_portals() -> None:
    pg = parse_image_to_grid(_fixtures_path("portal_cherries.png"), cols=12, rows=12)
    assert len(pg.cherry_cells) == 2, f"expected 2 cherries in portal_cherries.png, got {len(pg.cherry_cells)}"
    assert (9, 7) in pg.cherry_cells and (5, 10) in pg.cherry_cells, f"unexpected cherries: {pg.cherry_cells}"
    # Ensure those cells are not portals.
    assert pg.grid[7][9] == "cherry" and pg.grid[10][5] == "cherry"


def main() -> None:
    test_portal2_pairing()
    print("✓ portal2.png pairing OK")
    test_portal_png_score()
    print("✓ portal.png MILP score OK (k=12, score=7)")
    test_portal_cherries_not_portals()
    print("✓ portal_cherries.png cherries not misclassified as portals")


if __name__ == "__main__":
    main()


