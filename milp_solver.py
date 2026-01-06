"""
MILP solver for the Horse Pen problem (verification / ground truth).

This matches the browser MILP formulation (GLPK) and uses the same image parsing logic as JS:
- auto grid dimension detection
- water/horse/cherry detection

Requires:
  - `pip install pulp`
  - CBC (bundled with PuLP on macOS in this environment)
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import time
from typing import Dict, List, Tuple, Optional

import pulp

from hp_image import CHERRY_BONUS, ParsedGrid, parse_image_to_grid


def solve_milp(
    pg: ParsedGrid,
    k: int,
    time_limit_s: float = 10.0,
    cherry_bonus: int = CHERRY_BONUS,
    msg: bool = False,
) -> Tuple[str, Optional[int], Optional[int], Optional[int], List[Tuple[int, int]]]:
    """
    Solve the MILP and return (status, score, area, cherries, walls_xy).
    Score = area + cherry_bonus * cherries
    """

    w, h = pg.width, pg.height

    # Build vertex ids for passable cells
    vid: Dict[Tuple[int, int], int] = {}
    coords: List[Tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if pg.grid[y][x] != "water":
                vid[(x, y)] = len(coords)
                coords.append((x, y))

    n = len(coords)
    horse = vid[pg.horse]

    # Build adjacency
    out_neigh = defaultdict(list)
    in_neigh = defaultdict(list)
    edges: List[Tuple[int, int]] = []
    und_edges = set()

    for (x, y), u in vid.items():
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in vid:
                v = vid[(nx, ny)]
                edges.append((u, v))
                out_neigh[u].append(v)
                in_neigh[v].append(u)
                und_edges.add((u, v) if u < v else (v, u))

    boundary = set()
    for (x, y), u in vid.items():
        if x == 0 or x == w - 1 or y == 0 or y == h - 1:
            boundary.add(u)

    m = n

    prob = pulp.LpProblem("horse_pen", pulp.LpMaximize)

    r = pulp.LpVariable.dicts("r", range(n), lowBound=0, upBound=1, cat=pulp.LpBinary)  # reachable
    wall = pulp.LpVariable.dicts("w", range(n), lowBound=0, upBound=1, cat=pulp.LpBinary)  # wall
    f = pulp.LpVariable.dicts("f", edges, lowBound=0, cat=pulp.LpContinuous)  # flow

    # Objective: maximize sum r_i (+ cherry_bonus if cell is a cherry)
    weights = []
    for i, (x, y) in enumerate(coords):
        weights.append(1 + (cherry_bonus if pg.grid[y][x] == "cherry" else 0))
    prob += pulp.lpSum(weights[i] * r[i] for i in range(n))

    # Wall budget
    prob += pulp.lpSum(wall[i] for i in range(n)) <= k

    # Horse fixed
    prob += wall[horse] == 0
    prob += r[horse] == 1

    # Cherries cannot have walls placed on them
    cherry_ids = [vid[(x, y)] for x, y in pg.cherry_cells if (x, y) in vid]
    for c_id in cherry_ids:
        prob += wall[c_id] == 0

    # Reachable implies not a wall
    for i in range(n):
        prob += r[i] <= 1 - wall[i]

    # Boundary cannot be reachable (else escape)
    for b in boundary:
        if b != horse:
            prob += r[b] == 0

    # Closure: if u reachable and v isn't walled, v must be reachable (prevents under-approx)
    for u, v in und_edges:
        prob += r[v] >= r[u] - wall[v]
        prob += r[u] >= r[v] - wall[u]

    # Flow: enforce connectivity to the horse
    for u, v in edges:
        prob += f[(u, v)] <= m * r[u]
        prob += f[(u, v)] <= m * r[v]

    for v in range(n):
        inflow = pulp.lpSum(f[(u, v)] for u in in_neigh[v])
        outflow = pulp.lpSum(f[(v, u)] for u in out_neigh[v])
        if v == horse:
            prob += outflow - inflow == pulp.lpSum(r[i] for i in range(n)) - 1
        else:
            prob += inflow - outflow == r[v]

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=float(time_limit_s))
    prob.solve(solver)

    status = pulp.LpStatus.get(prob.status, str(prob.status))

    # If the solver didn't produce a solution, return empty walls + None metrics.
    obj = pulp.value(prob.objective)
    if obj is None:
        return status, None, None, None, []

    area = sum(1 for i in range(n) if (pulp.value(r[i]) or 0.0) > 0.5)
    walls = [coords[i] for i in range(n) if (pulp.value(wall[i]) or 0.0) > 0.5]
    walls.sort()
    cherries = sum(1 for i in range(n) if (pulp.value(r[i]) or 0.0) > 0.5 and weights[i] > 1)
    score = int(round(float(obj)))

    return status, score, area, cherries, walls


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Solve Horse Pen MILP from a screenshot image.")
    ap.add_argument("image", help="Path to screenshot (png/webp/jpg).")
    ap.add_argument("--walls", "-k", type=int, default=13, help="Wall budget (max number of walls to place).")
    ap.add_argument("--time", type=float, default=10.0, help="CBC time limit (seconds).")
    ap.add_argument("--cols", type=int, default=None, help="Override detected grid columns.")
    ap.add_argument("--rows", type=int, default=None, help="Override detected grid rows.")
    ap.add_argument("--msg", action="store_true", help="Enable CBC solver output.")
    args = ap.parse_args()

    pg = parse_image_to_grid(args.image, cols=args.cols, rows=args.rows)
    t0 = time.time()
    status, score, area, cherries, walls = solve_milp(
        pg, k=int(args.walls), time_limit_s=float(args.time), msg=bool(args.msg)
    )
    dt = time.time() - t0

    print(
        f"detected grid={pg.width}x{pg.height} horse={pg.horse} cherries={len(pg.cherry_cells)} "
        f"(method={pg.debug.get('horse_method')})"
    )
    print(
        f"status={status} score={score} area={area} cherries_enclosed={cherries} "
        f"walls={len(walls)} time={dt:.2f}s"
    )
    print("walls:", walls)


