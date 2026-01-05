"""
MILP solver for the Horse Pen problem (used to verify optimality on preview.webp).

This formulation solved preview.webp at k=13 with area=95 in ~2-3 seconds on my machine.

Requires:
  - `pip install pulp`
  - CBC (bundled with PuLP on macOS in this environment)
"""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Dict, List, Tuple

import pulp

from exact_solver import parse_preview_webp


def solve_preview_k13(time_limit_s: int = 10) -> Tuple[int, List[Tuple[int, int]]]:
    pg = parse_preview_webp(cols=19, rows=23, water_thresh=50, horse_thresh=160)

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

    k = 13
    m = n

    prob = pulp.LpProblem("horse_pen", pulp.LpMaximize)

    r = pulp.LpVariable.dicts("r", range(n), lowBound=0, upBound=1, cat=pulp.LpBinary)  # reachable
    wall = pulp.LpVariable.dicts("w", range(n), lowBound=0, upBound=1, cat=pulp.LpBinary)  # wall
    f = pulp.LpVariable.dicts("f", edges, lowBound=0, cat=pulp.LpContinuous)  # flow

    # Maximize reachable tiles
    prob += pulp.lpSum(r[i] for i in range(n))

    # Wall budget
    prob += pulp.lpSum(wall[i] for i in range(n)) <= k

    # Horse fixed
    prob += wall[horse] == 0
    prob += r[horse] == 1

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

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)

    area = sum(1 for i in range(n) if pulp.value(r[i]) > 0.5)
    walls = [coords[i] for i in range(n) if pulp.value(wall[i]) > 0.5]
    walls.sort()

    return area, walls


if __name__ == "__main__":
    t0 = time.time()
    area, walls = solve_preview_k13(time_limit_s=10)
    dt = time.time() - t0
    print(f"area={area} walls={len(walls)} time={dt:.2f}s")
    print("walls:", walls)


