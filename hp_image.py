"""
Image parsing utilities for Horse Pen screenshots.

This module mirrors the browser implementation in `src/image.js`:
- auto-detect grid dimensions using gradient autocorrelation
- classify tiles as water/grass/cherry/horse

The output is intended to feed the Python MILP solver (PuLP/CBC) for verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# Keep these aligned with `src/constants.js` + `src/image.js`.
WATER_BLUE = 50
HORSE_BRIGHTNESS = 160
CHERRY_BONUS = 3


@dataclass(frozen=True)
class ParsedGrid:
    grid: List[List[str]]  # "water" | "grass" | "cherry" | "horse"
    width: int
    height: int
    horse: Tuple[int, int]
    cherry_cells: List[Tuple[int, int]]
    debug: Dict[str, object]


def load_image_rgb(path: str) -> np.ndarray:
    """Return uint8 RGB pixels with shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _score_periods(signal: np.ndarray, min_period: int, max_period: int) -> List[Tuple[int, float]]:
    """
    Mirrors `scorePeriods` in `src/image.js`.
    Returns a list of (period, score), sorted descending by score.
    """
    signal = signal.astype(np.float32, copy=False)
    n = int(signal.shape[0])
    if n <= 0:
        return []

    mean = float(signal.mean())
    std = float(signal.std())
    if std == 0.0:
        return []

    normalized = (signal - mean) / std
    upper = min(int(max_period), int(n // 3))
    if upper < int(min_period):
        return []

    out: List[Tuple[int, float]] = []
    for p in range(int(min_period), upper + 1):
        # correlation sum_{i=0}^{n-p-1} norm[i] * norm[i+p]
        s = float(np.dot(normalized[:-p], normalized[p:]))
        out.append((p, s))

    out.sort(key=lambda t: t[1], reverse=True)
    return out


def detect_grid_dimensions_from_pixels(pixels_rgb: np.ndarray) -> Tuple[int, int, Dict[str, object]]:
    """
    Auto-detect grid (cols, rows) for a screenshot.
    Mirrors `HP.detectGridDimensions` from `src/image.js`.
    """
    h, w = pixels_rgb.shape[:2]

    gray = pixels_rgb.astype(np.float32).mean(axis=2)  # (H, W)

    # Horizontal gradient profile (per-column)
    col_profile = np.abs(gray[:, 1:] - gray[:, :-1]).sum(axis=0)  # (W-1,)
    # Vertical gradient profile (per-row)
    row_profile = np.abs(gray[1:, :] - gray[:-1, :]).sum(axis=1)  # (H-1,)

    min_period = 10
    max_period = 140
    col_periods = _score_periods(col_profile, min_period, max_period)
    row_periods = _score_periods(row_profile, min_period, max_period)

    if not col_periods or not row_periods:
        dbg = {"fallback": True, "reason": "no_periods"}
        return 19, 23, dbg

    top_k = 10
    col_top = col_periods[:top_k]
    row_top = row_periods[:top_k]

    best_score = float("-inf")
    best_cols, best_rows = 19, 23

    for cw_p, cw_score in col_top:
        for rh_p, rh_score in row_top:
            cols = int(round(w / cw_p))
            rows = int(round(h / rh_p))
            cols = max(5, min(50, cols))
            rows = max(5, min(50, rows))

            cell_w = w / cols
            cell_h = h / rows
            aspect = cell_w / cell_h if cell_h != 0 else 0.0

            if aspect < 0.8 or aspect > 1.25:
                continue

            err_w = abs(cell_w - cw_p) / float(cw_p)
            err_h = abs(cell_h - rh_p) / float(rh_p)
            aspect_penalty = abs(aspect - 1.0)

            score = float(cw_score + rh_score - 1000.0 * aspect_penalty - 500.0 * (err_w + err_h))
            if score > best_score:
                best_score = score
                best_cols, best_rows = cols, rows

    if not np.isfinite(best_score):
        cols = max(5, min(50, int(round(w / col_periods[0][0]))))
        rows = max(5, min(50, int(round(h / row_periods[0][0]))))
        dbg = {
            "fallback": True,
            "reason": "cross_scoring_failed",
            "cols": cols,
            "rows": rows,
            "best_col_period": col_periods[0][0],
            "best_row_period": row_periods[0][0],
        }
        return cols, rows, dbg

    dbg = {
        "fallback": False,
        "cols": best_cols,
        "rows": best_rows,
        "best_score": best_score,
        "best_col_period": col_top[0][0] if col_top else None,
        "best_row_period": row_top[0][0] if row_top else None,
    }
    return best_cols, best_rows, dbg


def _is_cherry_pixel(r: int, g: int, b: int) -> bool:
    # Mirrors `isCherryPixel` in JS.
    # Cast to int to avoid uint8 overflow in comparisons
    r, g, b = int(r), int(g), int(b)
    return r >= 150 and (r - g) >= 60 and (r - b) >= 60 and g <= 170 and b <= 170


def _detect_cherry_cells_by_pixels(
    pixels_rgb: np.ndarray,
    cell_w: float,
    cell_h: float,
    rows: int,
    cols: int,
    grid: List[List[str]],
) -> np.ndarray:
    """
    Mirrors `detectCherryCellsByPixels` in JS.
    Returns uint8 array length rows*cols with 1 if cherry-like, else 0.
    """
    h, w = pixels_rgb.shape[:2]
    counts = np.zeros((rows, cols), dtype=np.uint16)

    step = 2
    sub = pixels_rgb[::step, ::step, :]  # (H/2, W/2, 3)
    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)

    mask = (r >= 150) & ((r - g) >= 60) & ((r - b) >= 60) & (g <= 170) & (b <= 170)
    if not mask.any():
        return np.zeros(rows * cols, dtype=np.uint8)

    ys, xs = np.nonzero(mask)
    # Map back to original pixel coordinates
    xs = xs.astype(np.float32) * step
    ys = ys.astype(np.float32) * step

    cols_idx = np.floor(xs / cell_w).astype(np.int32)
    rows_idx = np.floor(ys / cell_h).astype(np.int32)
    cols_idx = np.clip(cols_idx, 0, cols - 1)
    rows_idx = np.clip(rows_idx, 0, rows - 1)

    # Skip water cells (grid is already water/grass at this stage)
    keep = np.array([grid[int(ry)][int(cx)] != "water" for ry, cx in zip(rows_idx, cols_idx)], dtype=bool)
    rows_idx = rows_idx[keep]
    cols_idx = cols_idx[keep]

    if rows_idx.size == 0:
        return np.zeros(rows * cols, dtype=np.uint8)

    np.add.at(counts, (rows_idx, cols_idx), 1)

    max_count = int(counts.max())
    out = np.zeros((rows, cols), dtype=np.uint8)
    if max_count == 0:
        return out.reshape(-1)

    threshold = max(6, int(np.floor(max_count * 0.2)))
    out[counts >= threshold] = 1
    return out.reshape(-1)


def _find_horse_by_brightest_square(
    pixels_rgb: np.ndarray,
    cell_w: float,
    cell_h: float,
    rows: int,
    cols: int,
) -> Optional[Dict[str, float]]:
    """
    Find the N×N square region with highest MEAN brightness (where N ≈ cell size).
    The horse sprite is white, so its cell will have the highest mean brightness.
    Returns dict {x,y,brightness,whiteness} in grid coords.
    """
    h, w = pixels_rgb.shape[:2]
    
    # Use average cell size for the square
    cell_size = int((cell_w + cell_h) / 2)
    
    # Compute brightness for the entire image
    gray = pixels_rgb.astype(np.float32).mean(axis=2)  # (H, W)
    
    # Use a sliding window to find the square with highest mean brightness
    # Step by half a cell to ensure we don't miss the horse
    step = max(1, cell_size // 4)
    best_brightness = 0.0
    best_y, best_x = 0, 0
    
    for y in range(0, h - cell_size + 1, step):
        for x in range(0, w - cell_size + 1, step):
            # Compute mean brightness of this square
            square = gray[y:y+cell_size, x:x+cell_size]
            if square.size == 0:
                continue
            mean_brightness = float(square.mean())
            if mean_brightness > best_brightness:
                best_brightness = mean_brightness
                best_y, best_x = y, x
    
    # No minimum threshold - just find the brightest square
    if best_brightness < 50:  # Sanity check: reject if entire image is dark
        return None
    
    # Map the center of the square to grid coordinates
    center_x = best_x + cell_size // 2
    center_y = best_y + cell_size // 2
    
    col = int(np.floor(center_x / cell_w))
    row = int(np.floor(center_y / cell_h))
    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))
    
    # Compute whiteness at center
    if 0 <= center_y < h and 0 <= center_x < w:
        r0, g0, b0 = pixels_rgb[center_y, center_x, :].astype(np.float32)
        wh = float(abs(r0 - g0) + abs(g0 - b0))
    else:
        wh = 0.0
    
    return {"x": float(col), "y": float(row), "brightness": best_brightness, "whiteness": wh}


def parse_image_to_grid(
    image_path: str,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
) -> ParsedGrid:
    """
    Parse an image file into a logical grid.
    If cols/rows are not provided, they are auto-detected (JS algorithm).
    """
    pixels = load_image_rgb(image_path)
    h, w = pixels.shape[:2]

    auto_dbg: Dict[str, object] = {}
    if cols is None or rows is None:
        cols2, rows2, dbg = detect_grid_dimensions_from_pixels(pixels)
        cols = cols if cols is not None else cols2
        rows = rows if rows is not None else rows2
        auto_dbg = dbg

    assert cols is not None and rows is not None

    cell_w = w / float(cols)
    cell_h = h / float(rows)

    grid: List[List[str]] = [["grass"] * cols for _ in range(rows)]
    water_count = 0
    grass_count = 0

    cherry_candidates = np.zeros(rows * cols, dtype=np.uint8)

    sample_points = [(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]

    for r in range(rows):
        for c in range(cols):
            samples: List[Tuple[int, int, int]] = []
            for fx, fy in sample_points:
                px = int(np.floor((c + fx) * cell_w))
                py = int(np.floor((r + fy) * cell_h))
                if 0 <= px < w and 0 <= py < h:
                    rr, gg, bb = pixels[py, px, :]
                    samples.append((int(rr), int(gg), int(bb)))

            if not samples:
                grid[r][c] = "grass"
                grass_count += 1
                continue

            arr = np.asarray(samples, dtype=np.float32)
            avg = arr.mean(axis=0)
            avg_r, avg_g, avg_b = float(avg[0]), float(avg[1]), float(avg[2])

            cherry_hit = False
            for rr, gg, bb in samples:
                if (not cherry_hit) and _is_cherry_pixel(rr, gg, bb):
                    cherry_hit = True
                    break

            is_water = (
                avg_b > avg_g
                and avg_b > avg_r
                and avg_b > WATER_BLUE
                and avg_r < 80
                and avg_g < 110
            )

            if is_water:
                grid[r][c] = "water"
                water_count += 1
            else:
                grid[r][c] = "grass"
                grass_count += 1

            if (not is_water) and cherry_hit:
                cherry_candidates[r * cols + c] = 1

    # Find horse using brightest square algorithm (most reliable)
    horse_method = "brightest_square"
    horse_stats = _find_horse_by_brightest_square(pixels, cell_w, cell_h, rows, cols)

    if horse_stats is None:
        raise RuntimeError("Could not detect horse position")

    hx = int(horse_stats["x"])
    hy = int(horse_stats["y"])
    hx = max(0, min(cols - 1, hx))
    hy = max(0, min(rows - 1, hy))

    prev = grid[hy][hx]
    if prev == "water":
        water_count = max(0, water_count - 1)
    if prev == "grass":
        grass_count = max(0, grass_count - 1)
    grid[hy][hx] = "horse"

    # Cherries: pixel-scan detection (priority: reliability) + per-cell candidates.
    cherry_by_pixels = _detect_cherry_cells_by_pixels(pixels, cell_w, cell_h, rows, cols, grid)
    cherry_cells: List[Tuple[int, int]] = []
    cherry_count = 0
    for y in range(rows):
        for x in range(cols):
            idx = y * cols + x
            if grid[y][x] != "grass":
                continue
            if cherry_candidates[idx] == 0 and cherry_by_pixels[idx] == 0:
                continue
            grid[y][x] = "cherry"
            grass_count = max(0, grass_count - 1)
            cherry_count += 1
            cherry_cells.append((x, y))

    debug: Dict[str, object] = {
        "img_w": w,
        "img_h": h,
        "auto_grid": auto_dbg,
        "water_count": water_count,
        "grass_count": grass_count,
        "cherry_count": cherry_count,
        "horse": (hx, hy),
        "horse_method": horse_method,
        "horse_brightness": horse_stats.get("brightness"),
        "horse_whiteness": horse_stats.get("whiteness"),
    }

    return ParsedGrid(
        grid=grid,
        width=cols,
        height=rows,
        horse=(hx, hy),
        cherry_cells=cherry_cells,
        debug=debug,
    )


