"""
Exact solver (parameterized by k) using IMPORTANT SEPARATORS.

Goal: pick <=k wall cells (vertex removals) so the horse cannot reach the map edge,
maximizing the number of reachable cells (enclosed area).

This is practical for small k (e.g. k=13) on grids like 19x23.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from itertools import combinations
import time
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIXTURES_DIR = os.path.join(ROOT_DIR, "tests", "fixtures")
PREVIEW_PATH = os.path.join(FIXTURES_DIR, "preview.webp")


INF = 10**9


class Dinic:
    class Edge:
        __slots__ = ("to", "rev", "cap")

        def __init__(self, to: int, rev: int, cap: int):
            self.to = to
            self.rev = rev
            self.cap = cap

    def __init__(self, n: int):
        self.n = n
        self.g: List[List[Dinic.Edge]] = [[] for _ in range(n)]
        self.level = [0] * n
        self.it = [0] * n

    def add_edge(self, fr: int, to: int, cap: int) -> None:
        fwd = Dinic.Edge(to, len(self.g[to]), cap)
        rev = Dinic.Edge(fr, len(self.g[fr]), 0)
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def bfs(self, s: int, t: int) -> bool:
        self.level = [-1] * self.n
        q = deque([s])
        self.level[s] = 0
        while q:
            v = q.popleft()
            for e in self.g[v]:
                if e.cap > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[v] + 1
                    q.append(e.to)
        return self.level[t] >= 0

    def dfs(self, v: int, t: int, f: int) -> int:
        if v == t:
            return f
        for i in range(self.it[v], len(self.g[v])):
            self.it[v] = i
            e = self.g[v][i]
            if e.cap <= 0 or self.level[e.to] != self.level[v] + 1:
                continue
            pushed = self.dfs(e.to, t, min(f, e.cap))
            if pushed:
                e.cap -= pushed
                self.g[e.to][e.rev].cap += pushed
                return pushed
        return 0

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                pushed = self.dfs(s, t, INF)
                if not pushed:
                    break
                flow += pushed
        return flow

    def reachable_in_residual(self, s: int) -> List[bool]:
        seen = [False] * self.n
        q = deque([s])
        seen[s] = True
        while q:
            v = q.popleft()
            for e in self.g[v]:
                if e.cap > 0 and not seen[e.to]:
                    seen[e.to] = True
                    q.append(e.to)
        return seen


@dataclass(frozen=True)
class ParsedGrid:
    grid: List[List[str]]
    width: int
    height: int
    horse: Tuple[int, int]


def parse_preview_webp(cols: int = 19, rows: int = 23, water_thresh: int = 50, horse_thresh: int = 160) -> ParsedGrid:
    img = Image.open(PREVIEW_PATH)
    pixels = np.array(img, dtype=np.float32)
    img_h, img_w = pixels.shape[:2]

    cell_w = img_w / cols
    cell_h = img_h / rows

    grid: List[List[str]] = []
    horse: Optional[Tuple[int, int]] = None

    for r in range(rows):
        row: List[str] = []
        for c in range(cols):
            samples = []
            for fx, fy in [(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]:
                px = int((c + fx) * cell_w)
                py = int((r + fy) * cell_h)
                samples.append(pixels[py, px, :3])
            avg = np.mean(samples, axis=0)
            avg_r, avg_g, avg_b = float(avg[0]), float(avg[1]), float(avg[2])
            brightness = (avg_r + avg_g + avg_b) / 3.0
            is_whitish = abs(avg_r - avg_g) < 30 and abs(avg_g - avg_b) < 30

            if brightness > horse_thresh and is_whitish:
                row.append("horse")
                horse = (c, r)
            elif avg_b > avg_g and avg_b > avg_r and avg_b > water_thresh and avg_r < 80 and avg_g < 110:
                row.append("water")
            else:
                row.append("grass")
        grid.append(row)

    if horse is None:
        raise RuntimeError("Horse not detected in preview.webp")

    return ParsedGrid(grid=grid, width=cols, height=rows, horse=horse)


def parse_preview_webp_with_fallback(cols: int = 19, rows: int = 23, water_thresh: int = 50, horse_thresh: int = 160) -> ParsedGrid:
    """
    Same as parse_preview_webp, but if horse is not detected via thresholds, picks the brightest cell.
    """
    img = Image.open(PREVIEW_PATH)
    pixels = np.array(img, dtype=np.float32)
    img_h, img_w = pixels.shape[:2]

    cell_w = img_w / cols
    cell_h = img_h / rows

    grid: List[List[str]] = []
    horse: Optional[Tuple[int, int]] = None

    best_brightness = -1.0
    best_cell: Optional[Tuple[int, int]] = None

    for r in range(rows):
        row: List[str] = []
        for c in range(cols):
            samples = []
            for fx, fy in [(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]:
                px = int((c + fx) * cell_w)
                py = int((r + fy) * cell_h)
                samples.append(pixels[py, px, :3])
            avg = np.mean(samples, axis=0)
            avg_r, avg_g, avg_b = float(avg[0]), float(avg[1]), float(avg[2])
            brightness = (avg_r + avg_g + avg_b) / 3.0

            if brightness > best_brightness:
                best_brightness = brightness
                best_cell = (c, r)

            is_whitish = abs(avg_r - avg_g) < 30 and abs(avg_g - avg_b) < 30

            if brightness > horse_thresh and is_whitish:
                row.append("horse")
                horse = (c, r)
            elif avg_b > avg_g and avg_b > avg_r and avg_b > water_thresh and avg_r < 80 and avg_g < 110:
                row.append("water")
            else:
                row.append("grass")
        grid.append(row)

    if horse is None:
        if best_cell is None:
            raise RuntimeError("Could not detect horse or brightest cell")
        horse = best_cell
        # Ensure we don't label water as horse; but for preview.webp this should be fine.
        grid[horse[1]][horse[0]] = "horse"

    return ParsedGrid(grid=grid, width=cols, height=rows, horse=horse)


def build_graph(pg: ParsedGrid) -> Tuple[List[List[int]], Dict[Tuple[int, int], int], List[Tuple[int, int]], int, Set[int]]:
    """
    Returns:
      - adj: adjacency list for passable vertices + sink
      - id_of[(x,y)] => vertex id (0..V-2) for passable cells
      - coord_of[id] => (x,y)
      - sink_id => V-1
      - boundary_ids => set of vertex ids that are passable AND on map edge
    """
    id_of: Dict[Tuple[int, int], int] = {}
    coord_of: List[Tuple[int, int]] = []

    for y in range(pg.height):
        for x in range(pg.width):
            if pg.grid[y][x] != "water":
                id_of[(x, y)] = len(coord_of)
                coord_of.append((x, y))

    sink_id = len(coord_of)
    adj: List[List[int]] = [[] for _ in range(sink_id + 1)]

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < pg.width and 0 <= y < pg.height

    boundary_ids: Set[int] = set()

    for vid, (x, y) in enumerate(coord_of):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            if pg.grid[ny][nx] == "water":
                continue
            nid = id_of[(nx, ny)]
            adj[vid].append(nid)

        if x == 0 or x == pg.width - 1 or y == 0 or y == pg.height - 1:
            boundary_ids.add(vid)

    # connect sink to all boundary vertices
    for b in boundary_ids:
        adj[b].append(sink_id)
        adj[sink_id].append(b)

    return adj, id_of, coord_of, sink_id, boundary_ids


def reachable_vertices(adj: List[List[int]], s: int, blocked: Set[int]) -> Set[int]:
    if s in blocked:
        return set()
    seen = {s}
    q = deque([s])
    while q:
        v = q.popleft()
        for to in adj[v]:
            if to in blocked or to in seen:
                continue
            seen.add(to)
            q.append(to)
    return seen


def min_vertex_separator(
    adj: List[List[int]],
    horse: int,
    source_set: Set[int],
    sink: int,
    deleted: Set[int],
) -> Set[int]:
    """
    Minimum vertex separator separating source_set from sink, in the graph with deleted vertices removed.
    Vertex cut costs: 1 for normal vertices, INF for vertices in source_set or sink.
    """
    n = len(adj)
    # split nodes: in=2*v, out=2*v+1
    SRC = 2 * n
    din = Dinic(2 * n + 1)

    def vin(v: int) -> int:
        return 2 * v

    def vout(v: int) -> int:
        return 2 * v + 1

    # split edges
    for v in range(n):
        if v in deleted:
            cap = 0
        elif v in source_set or v == sink:
            cap = INF
        else:
            cap = 1
        din.add_edge(vin(v), vout(v), cap)

    # adjacency edges
    for v in range(n):
        if v in deleted:
            continue
        for to in adj[v]:
            if to in deleted:
                continue
            # undirected represented as directed both ways, but adj already includes both directions
            din.add_edge(vout(v), vin(to), INF)

    # connect SRC to all sources
    for s in source_set:
        if s in deleted:
            continue
        din.add_edge(SRC, vin(s), INF)

    # compute maxflow to sink_in
    flow = din.max_flow(SRC, vin(sink))
    # If the minimum cut is effectively "infinite", then with the current forced sources
    # there's no finite vertex separator (e.g., forcing a boundary vertex to be reachable).
    if flow >= INF // 2:
        return set()
    reach = din.reachable_in_residual(SRC)

    sep: Set[int] = set()
    for v in range(n):
        if v in deleted or v in source_set or v == sink:
            continue
        if reach[vin(v)] and not reach[vout(v)]:
            sep.add(v)
    # Sanity check: sep must actually disconnect horse-side sources from sink in the original graph.
    # (If not, we extracted the wrong thing, or the cut went through INF edges.)
    blocked = set(deleted) | set(sep)
    if horse in blocked:
        return set()
    if sink in reachable_vertices(adj, horse, blocked):
        # Not a valid separator
        return set()

    return sep


@dataclass
class BestSolution:
    area: int
    walls: Set[int]
    enclosed: Set[int]


def best_separator_by_important_separators(
    adj: List[List[int]],
    horse: int,
    sink: int,
    k: int,
    time_budget_s: float = 10.0,
) -> BestSolution:
    """
    Enumerate important separators (size <= k) and return the one maximizing reachable area from horse.
    """
    t0 = time.time()
    best = BestSolution(area=0, walls=set(), enclosed=set())

    memo: Set[Tuple[Tuple[int, ...], Tuple[int, ...], int]] = set()

    def rec(deleted: Set[int], forced: Set[int], k_rem: int) -> None:
        nonlocal best
        if time.time() - t0 > time_budget_s:
            return
        if k_rem < 0:
            return

        key = (tuple(sorted(deleted)), tuple(sorted(forced)), k_rem)
        if key in memo:
            return
        memo.add(key)

        source_set = set(forced)
        source_set.add(horse)

        sep = min_vertex_separator(adj, horse, source_set, sink, deleted)
        if not sep and sink in reachable_vertices(adj, horse, set(deleted)):  # infeasible state (forced boundary etc)
            return
        if len(sep) > k_rem:
            return

        full_sep = set(deleted) | set(sep)
        enclosed = reachable_vertices(adj, horse, set(full_sep))
        if sink in enclosed:
            # Shouldn't happen if sep is valid; extra guard.
            return
        if sink in full_sep:
            return
        # Do not count sink in the area if it somehow appeared (it shouldn't).
        enclosed.discard(sink)

        area = len(enclosed)
        if area > best.area:
            best = BestSolution(area=area, walls=full_sep, enclosed=enclosed)

        # Branch over the current minimum separator
        for v in sep:
            if time.time() - t0 > time_budget_s:
                return

            # Branch A: include v in separator (consume 1 wall)
            if v not in deleted:
                deleted2 = set(deleted)
                deleted2.add(v)
                rec(deleted2, forced, k_rem - 1)

            # Branch B: force v reachable (so v cannot be in separator)
            if v not in forced:
                forced2 = set(forced)
                forced2.add(v)
                rec(deleted, forced2, k_rem)

    rec(set(), set(), k)
    return best


def solve_preview_exact(k: int = 13, time_budget_s: float = 10.0) -> None:
    pg = parse_preview_webp()
    adj, id_of, coord_of, sink_id, _boundary = build_graph(pg)
    horse_id = id_of[pg.horse]

    best = best_separator_by_important_separators(adj, horse_id, sink_id, k=k, time_budget_s=time_budget_s)

    walls_xy = sorted([coord_of[v] for v in best.walls if v != sink_id])
    print(f"Best area={best.area} with |walls|={len(best.walls)} (k={k})")
    print("Walls:", walls_xy)


# ============================
# Experimental: contraction-based important separator search (single source)
# ============================


def best_separator_by_important_separators_contracted(
    base_adj: List[List[int]],
    horse: int,
    sink: int,
    k: int,
    time_budget_s: float = 10.0,
) -> BestSolution:
    """
    Variant that keeps a SINGLE source by contracting chosen vertices into `horse`.
    This avoids the multi-source pitfall for our objective (reachable-from-horse only).
    """

    t0 = time.time()
    n = len(base_adj)

    best = BestSolution(area=0, walls=set(), enclosed=set())
    memo: Set[Tuple[Tuple[int, ...], Tuple[int, ...], int]] = set()

    def rep(v: int, contracted: Set[int]) -> int:
        return horse if v in contracted else v

    def effective_neighbors(v: int, deleted: Set[int], contracted: Set[int]) -> List[int]:
        # Vertex v is assumed not deleted and not contracted (unless v==horse)
        out: Set[int] = set()
        if v == horse:
            # horse represents the whole contracted supernode
            sources = {horse} | set(contracted)
            for s in sources:
                if s in deleted:
                    continue
                for to in base_adj[s]:
                    if to in deleted:
                        continue
                    to2 = rep(to, contracted)
                    if to2 == horse:
                        continue
                    out.add(to2)
            return list(out)

        for to in base_adj[v]:
            if to in deleted:
                continue
            to2 = rep(to, contracted)
            if to2 == v:
                continue
            out.add(to2)
        return list(out)

    def build_flow_and_minsep(deleted: Set[int], contracted: Set[int]) -> Set[int]:
        """
        Build a vertex-split flow network for the current contracted graph and return a minimum separator.
        Returns empty set if already disconnected OR infeasible (we'll validate later).
        """

        # We'll keep original vertex ids, but skip deleted/contracted vertices (except horse).
        active = [True] * n
        for v in deleted:
            active[v] = False
        for v in contracted:
            active[v] = False
        active[horse] = True
        active[sink] = True

        SRC = 2 * n
        din = Dinic(2 * n + 1)

        def vin(v: int) -> int:
            return 2 * v

        def vout(v: int) -> int:
            return 2 * v + 1

        # split edges
        for v in range(n):
            if not active[v]:
                cap = 0
            elif v == horse or v == sink:
                cap = INF
            else:
                cap = 1
            din.add_edge(vin(v), vout(v), cap)

        # adjacency edges for active vertices
        for v in range(n):
            if not active[v]:
                continue
            for to in effective_neighbors(v, deleted, contracted):
                if not active[to]:
                    continue
                din.add_edge(vout(v), vin(to), INF)

        # super source to horse
        din.add_edge(SRC, vin(horse), INF)

        flow = din.max_flow(SRC, vin(sink))
        if flow >= INF // 2:
            return set()  # infeasible

        reach = din.reachable_in_residual(SRC)
        sep: Set[int] = set()
        for v in range(n):
            if not active[v] or v == horse or v == sink:
                continue
            if reach[vin(v)] and not reach[vout(v)]:
                sep.add(v)
        return sep

    def reachable_area(walls: Set[int], deleted: Set[int], contracted: Set[int]) -> int:
        # reachable in the effective graph, counting contracted vertices too
        blocked = set(walls) | set(deleted)
        q = deque([horse])
        seen = {horse}
        while q:
            v = q.popleft()
            for to in effective_neighbors(v, blocked, contracted):
                if to == sink:
                    continue
                if to in blocked or to in seen:
                    continue
                seen.add(to)
                q.append(to)
        # exclude sink if present (it won't be)
        seen.discard(sink)
        # add contracted vertex count (they're merged into horse)
        return len(seen) + len(contracted)

    def sink_reachable(walls: Set[int], deleted: Set[int], contracted: Set[int]) -> bool:
        blocked = set(walls) | set(deleted)
        q = deque([horse])
        seen = {horse}
        while q:
            v = q.popleft()
            for to in effective_neighbors(v, blocked, contracted):
                if to in blocked or to in seen:
                    continue
                if to == sink:
                    return True
                seen.add(to)
                q.append(to)
        return False

    def rec(deleted: Set[int], contracted: Set[int], k_rem: int) -> None:
        nonlocal best
        if time.time() - t0 > time_budget_s:
            return
        if k_rem < 0:
            return

        key = (tuple(sorted(deleted)), tuple(sorted(contracted)), k_rem)
        if key in memo:
            return
        memo.add(key)

        sep = build_flow_and_minsep(deleted, contracted)

        # if sep empty, either already disconnected OR infeasible; validate by reachability
        if not sep:
            if sink_reachable(set(), deleted, contracted):
                return

        if len(sep) > k_rem:
            return

        walls = set(sep)
        if sink_reachable(walls, deleted, contracted):
            # not a valid enclosure
            return

        area = reachable_area(walls, deleted, contracted)
        if area > best.area:
            best = BestSolution(area=area, walls=set(deleted) | walls, enclosed=set())

        # branch
        for v in sep:
            if time.time() - t0 > time_budget_s:
                return
            # include v as a wall
            deleted2 = set(deleted)
            deleted2.add(v)
            rec(deleted2, contracted, k_rem - 1)

            # contract v into horse (force reachable)
            contracted2 = set(contracted)
            contracted2.add(v)
            rec(deleted, contracted2, k_rem)

    rec(set(), set(), k)
    return best


# ============================
# Correct contraction-based search:
# - compute min separator in contracted graph
# - evaluate candidate cut in ORIGINAL graph
# ============================


def best_cut_by_contraction_search(
    base_adj: List[List[int]],
    horse: int,
    sink: int,
    k: int,
    time_budget_s: float = 10.0,
) -> BestSolution:
    """
    Exact search (FPT in k) inspired by important-separator enumeration.

    State:
      - deleted: vertices chosen as walls
      - contracted: vertices forced to be on the horse-side (not walled)

    At each state we compute a minimum (horse,sink)-vertex cut S in the contracted graph.
    Candidate solution is deleted ∪ S (evaluated in the ORIGINAL graph).
    Then branch on each v in S: delete it OR contract it.
    """

    t0 = time.time()
    n = len(base_adj)

    # Precompute original evaluation BFS
    def eval_original(walls: Set[int]) -> Optional[int]:
        if horse in walls or sink in walls:
            return None
        blocked = set(walls)
        q = deque([horse])
        seen = {horse}
        while q:
            v = q.popleft()
            for to in base_adj[v]:
                if to in blocked or to in seen:
                    continue
                if to == sink:
                    return None  # escape
                seen.add(to)
                q.append(to)
        return len(seen)

    # contracted mincut helper (single source)
    def minsep_contracted(deleted: Set[int], contracted: Set[int]) -> Set[int]:
        active = [True] * n
        for v in deleted:
            active[v] = False
        for v in contracted:
            active[v] = False
        active[horse] = True
        active[sink] = True

        def rep(v: int) -> int:
            return horse if v in contracted else v

        def neighbors(v: int) -> List[int]:
            out: Set[int] = set()
            if v == horse:
                sources = {horse} | set(contracted)
                for s in sources:
                    if s in deleted:
                        continue
                    for to in base_adj[s]:
                        if to in deleted:
                            continue
                        to2 = rep(to)
                        if to2 == horse:
                            continue
                        if to2 != sink and not active[to2]:
                            continue
                        out.add(to2)
                return list(out)

            for to in base_adj[v]:
                if to in deleted:
                    continue
                to2 = rep(to)
                if to2 == v:
                    continue
                if to2 != horse and to2 != sink and not active[to2]:
                    continue
                out.add(to2)
            return list(out)

        SRC = 2 * n
        din = Dinic(2 * n + 1)

        def vin(v: int) -> int:
            return 2 * v

        def vout(v: int) -> int:
            return 2 * v + 1

        for v in range(n):
            if not active[v]:
                cap = 0
            elif v == horse or v == sink:
                cap = INF
            else:
                cap = 1
            din.add_edge(vin(v), vout(v), cap)

        for v in range(n):
            if not active[v]:
                continue
            for to in neighbors(v):
                if not active[to]:
                    continue
                din.add_edge(vout(v), vin(to), INF)

        din.add_edge(SRC, vin(horse), INF)

        flow = din.max_flow(SRC, vin(sink))
        if flow >= INF // 2:
            return set()

        reach = din.reachable_in_residual(SRC)
        sep: Set[int] = set()
        for v in range(n):
            if not active[v] or v == horse or v == sink:
                continue
            if reach[vin(v)] and not reach[vout(v)]:
                sep.add(v)
        return sep

    best = BestSolution(area=0, walls=set(), enclosed=set())
    memo: Set[Tuple[Tuple[int, ...], Tuple[int, ...], int]] = set()

    def rec(deleted: Set[int], contracted: Set[int], k_rem: int) -> None:
        nonlocal best
        if time.time() - t0 > time_budget_s:
            return
        if k_rem < 0:
            return

        state = (tuple(sorted(deleted)), tuple(sorted(contracted)), k_rem)
        if state in memo:
            return
        memo.add(state)

        sep = minsep_contracted(deleted, contracted)
        if not sep:
            # already disconnected in contracted graph; try candidate = deleted
            area = eval_original(deleted)
            if area is not None and area > best.area:
                best = BestSolution(area=area, walls=set(deleted), enclosed=set())
            return

        if len(sep) > k_rem:
            return

        candidate = set(deleted) | set(sep)
        area = eval_original(candidate)
        if area is not None and area > best.area:
            best = BestSolution(area=area, walls=set(candidate), enclosed=set())

        # Branch on separator vertices
        for v in sep:
            if time.time() - t0 > time_budget_s:
                return

            # delete v
            if v not in deleted and v not in contracted:
                d2 = set(deleted)
                d2.add(v)
                rec(d2, contracted, k_rem - 1)

            # contract v
            if v not in contracted and v not in deleted:
                c2 = set(contracted)
                c2.add(v)
                rec(deleted, c2, k_rem)

    rec(set(), set(), k)
    return best


if __name__ == "__main__":
    solve_preview_exact(k=13, time_budget_s=10.0)


