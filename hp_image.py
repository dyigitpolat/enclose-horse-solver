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

import math
import numpy as np
from PIL import Image, ImageFilter


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

    raw: Dict[int, float] = {}
    for p in range(int(min_period), upper + 1):
        # correlation sum_{i=0}^{n-p-1} norm[i] * norm[i+p]
        s = float(np.dot(normalized[:-p], normalized[p:]))
        raw[p] = s

    # Harmonic aggregation: if a fundamental period exists, its harmonics (2p, 3p, ...)
    # often also score well. Summing harmonics helps prefer the true cell size over
    # larger multiples that can dominate after downscaling.
    #
    # IMPORTANT: allow a small tolerance when matching harmonics because resampling/JPEG
    # can shift the dominant period by ±1–2 px.
    scores: Dict[int, float] = {}
    max_harm = 6
    tol = 2
    min_p = int(min_period)
    for p, sp in raw.items():
        if sp <= 0:
            scores[p] = sp
            continue
        agg = sp
        for k in range(2, max_harm + 1):
            pk = p * k
            if pk > upper:
                break
            best_sk = 0.0
            for d in range(-tol, tol + 1):
                idx = pk + d
                if idx < min_p or idx > upper:
                    continue
                v = raw.get(idx, 0.0)
                if v > best_sk:
                    best_sk = v
            if best_sk > 0:
                agg += best_sk / float(k)
        scores[p] = agg

    out = list(scores.items())
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _find_best_grid_from_periods(
    col_periods: List[Tuple[int, float]],
    row_periods: List[Tuple[int, float]],
    w: int,
    h: int,
    top_k: int = 10,
) -> Tuple[int, int, float]:
    """
    Find the best grid dimensions from period candidates.
    Returns (cols, rows, score).
    """
    col_top = col_periods[:top_k]
    row_top = row_periods[:top_k]

    best_score = float("-inf")
    best_cols, best_rows = 19, 23

    cover_weight = 200.0
    for cw_p, cw_score in col_top:
        col_est = float(w) / float(cw_p) if cw_p else 0.0
        col_cands = {
            int(round(col_est)),
            int(math.floor(col_est)),
            int(math.ceil(col_est)),
        }
        for rh_p, rh_score in row_top:
            row_est = float(h) / float(rh_p) if rh_p else 0.0
            row_cands = {
                int(round(row_est)),
                int(math.floor(row_est)),
                int(math.ceil(row_est)),
            }
            for cols0 in col_cands:
                cols = max(5, min(50, int(cols0)))
                for rows0 in row_cands:
                    rows = max(5, min(50, int(rows0)))

                    cell_w = w / cols
                    cell_h = h / rows
                    aspect = cell_w / cell_h if cell_h != 0 else 0.0

                    if aspect < 0.8 or aspect > 1.25:
                        continue

                    err_w = abs(cell_w - cw_p) / float(cw_p)
                    err_h = abs(cell_h - rh_p) / float(rh_p)
                    aspect_penalty = abs(aspect - 1.0)

                    # Coverage penalty: prefer periods that tile the trimmed width/height well.
                    # This helps avoid off-by-one cell counts after downscaling (e.g. 16 vs 17).
                    cover_w = abs(float(w) - float(cols) * float(cw_p)) / float(cw_p)
                    cover_h = abs(float(h) - float(rows) * float(rh_p)) / float(rh_p)

                    score = float(
                        cw_score
                        + rh_score
                        - 1000.0 * aspect_penalty
                        - 500.0 * (err_w + err_h)
                        - cover_weight * (cover_w + cover_h)
                    )
                    if score > best_score:
                        best_score = score
                        best_cols, best_rows = cols, rows

    return best_cols, best_rows, best_score


def detect_grid_dimensions_from_pixels(pixels_rgb: np.ndarray, use_multiscale: bool = True, trim_edges: Optional[int] = None) -> Tuple[int, int, Dict[str, object]]:
    """
    Auto-detect grid (cols, rows) for a screenshot with improved robustness.
    Uses multi-scale gradient analysis with Gaussian blur and edge trimming.
    
    Args:
        pixels_rgb: RGB image as numpy array
        use_multiscale: If True, uses multiple blur scales and votes (more robust)
        trim_edges: Number of pixels to trim from each edge (None = auto based on image size)
    """
    h, w = pixels_rgb.shape[:2]
    
    # Adaptive edge trimming: trim 8 pixels by default (good balance)
    if trim_edges is None:
        trim_edges = 8
    
    # Trim edges to reduce border artifacts from cropping/extending
    if trim_edges > 0 and h > 2*trim_edges and w > 2*trim_edges:
        pixels_trimmed = pixels_rgb[trim_edges:-trim_edges, trim_edges:-trim_edges, :]
    else:
        pixels_trimmed = pixels_rgb
    
    h_trim, w_trim = pixels_trimmed.shape[:2]

    # JPEG 8x8 block artifact detector (cheap): compare gradient energy at 8px boundaries vs others.
    gray0 = pixels_trimmed.astype(np.float32).mean(axis=2)
    col0 = np.abs(gray0[:, 1:] - gray0[:, :-1]).sum(axis=0)  # (W-1,)
    row0 = np.abs(gray0[1:, :] - gray0[:-1, :]).sum(axis=1)  # (H-1,)
    if col0.size > 0:
        idx = np.arange(col0.size, dtype=np.int32)
        mask = ((idx + 1) % 8) == 0
        col_ratio = float(col0[mask].mean() / (col0[~mask].mean() + 1e-6)) if np.any(mask) and np.any(~mask) else 1.0
    else:
        col_ratio = 1.0
    if row0.size > 0:
        idx = np.arange(row0.size, dtype=np.int32)
        mask = ((idx + 1) % 8) == 0
        row_ratio = float(row0[mask].mean() / (row0[~mask].mean() + 1e-6)) if np.any(mask) and np.any(~mask) else 1.0
    else:
        row_ratio = 1.0
    blockiness = max(col_ratio, row_ratio)

    if use_multiscale:
        # Multi-scale detection: test with different blur levels and vote
        # Use more diverse scales to be more robust
        if blockiness > 1.6:
            # JPEG-like inputs: heavier blur is necessary to suppress 8x8 block boundaries.
            scales = [1.2, 1.8, 2.5, 3.5]
            scale_weights = [1.0, 1.0, 0.9, 0.8]
        else:
            scales = [0.0, 0.3, 0.7, 1.2, 1.8]  # From no blur to heavy blur
            scale_weights = [2.0, 1.5, 1.0, 0.8, 0.5]  # Prefer less blurred results

        scale_results = []
        
        for blur_sigma, weight in zip(scales, scale_weights):
            if blur_sigma > 0:
                gray = _apply_gaussian_blur(pixels_trimmed, radius=blur_sigma).astype(np.float32).mean(axis=2)
            else:
                gray = pixels_trimmed.astype(np.float32).mean(axis=2)
            
            # Horizontal gradient profile (per-column)
            col_profile = np.abs(gray[:, 1:] - gray[:, :-1]).sum(axis=0)  # (W-1,)
            # Vertical gradient profile (per-row)
            row_profile = np.abs(gray[1:, :] - gray[:-1, :]).sum(axis=1)  # (H-1,)
            
            # Allow smaller cells for heavily downscaled inputs, but also avoid JPEG 8x8 blocking artifacts:
            # derive a lower bound from the maximum plausible grid size.
            max_grid_dim = 50
            min_cell_from_max = int(np.floor(min(w_trim, h_trim) / float(max_grid_dim)))
            min_period = max(4, int(np.floor(min_cell_from_max * 0.7)))
            max_period = 200
            col_periods = _score_periods(col_profile, min_period, max_period)
            row_periods = _score_periods(row_profile, min_period, max_period)
            
            if col_periods and row_periods:
                # Get best grid for this scale on the trimmed region.
                # Trimming removes border pixels but does NOT change the number of cells.
                cols, rows, score = _find_best_grid_from_periods(col_periods, row_periods, w_trim, h_trim)
                scale_results.append((cols, rows, score, blur_sigma, weight))
        
        if not scale_results:
            return 19, 23, {"fallback": True, "reason": "no_periods_multiscale"}
        
        # Vote: group by (cols, rows) and sum weighted scores
        from collections import defaultdict
        grid_votes = defaultdict(float)
        for cols, rows, score, _, weight in scale_results:
            grid_votes[(cols, rows)] += score * weight
        
        # Pick the grid with highest total score
        best_cols, best_rows = max(grid_votes.items(), key=lambda x: x[1])[0]
        
        dbg = {
            "fallback": False,
            "multiscale": True,
            "blockiness": blockiness,
            "scales": scales,
            "scale_results": [(c, r, s, sig, w) for c, r, s, sig, w in scale_results],
            "votes": dict(grid_votes),
            "best_cols": best_cols,
            "best_rows": best_rows,
        }
        return best_cols, best_rows, dbg
    
    else:
        # Original single-scale detection
        gray = pixels_trimmed.astype(np.float32).mean(axis=2)  # (H, W)

        # Horizontal gradient profile (per-column)
        col_profile = np.abs(gray[:, 1:] - gray[:, :-1]).sum(axis=0)  # (W-1,)
        # Vertical gradient profile (per-row)
        row_profile = np.abs(gray[1:, :] - gray[:-1, :]).sum(axis=1)  # (H-1,)

        # Allow smaller cells for heavily downscaled inputs, but also avoid JPEG 8x8 blocking artifacts:
        # derive a lower bound from the maximum plausible grid size.
        max_grid_dim = 50
        min_cell_from_max = int(np.floor(min(w_trim, h_trim) / float(max_grid_dim)))
        min_period = max(4, int(np.floor(min_cell_from_max * 0.7)))
        max_period = 200
        col_periods = _score_periods(col_profile, min_period, max_period)
        row_periods = _score_periods(row_profile, min_period, max_period)

        if not col_periods or not row_periods:
            dbg = {"fallback": True, "reason": "no_periods"}
            return 19, 23, dbg

        best_cols, best_rows, best_score = _find_best_grid_from_periods(col_periods, row_periods, w_trim, h_trim)

        if not np.isfinite(best_score):
            cols = max(5, min(50, int(round(w_trim / col_periods[0][0]))))
            rows = max(5, min(50, int(round(h_trim / row_periods[0][0]))))
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
            "multiscale": False,
            "cols": best_cols,
            "rows": best_rows,
            "best_score": best_score,
            "best_col_period": w_trim / best_cols if best_cols > 0 else None,
            "best_row_period": h_trim / best_rows if best_rows > 0 else None,
        }
        return best_cols, best_rows, dbg


def _smooth_1d(x: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    if window % 2 == 0:
        window += 1
    if x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x.astype(np.float32, copy=False), kernel, mode="same")


def _fit_grid_axis(profile: np.ndarray, n_cells: int, target_cell: float) -> Tuple[float, float, Dict[str, float]]:
    """
    Given a 1D gradient profile (length ~pixels-1), fit a grid axis:
    find (start, period) such that profile peaks align with start + k*period for k=0..n_cells.

    Returns (start, period, debug).
    """
    prof = profile.astype(np.float32, copy=False)
    prof_s = _smooth_1d(prof, window=3)

    # Search period in a range around the naive estimate.
    target = float(target_cell)
    p_min = max(4, int(math.floor(target * 0.7)))
    p_max = min(200, int(math.ceil(target * 1.3)))

    best_score = float("-inf")
    best_p = None
    best_s = None

    n = int(prof_s.shape[0])
    if n <= 0:
        return 0.0, target, {"fallback": 1.0, "reason": 1.0}

    # Important: crop/extend can remove the outermost grid line, so don't require the full
    # (n_cells+1) set of lines to fit inside the profile. Instead score using the lines
    # that do fit, and normalize by count (so smaller periods don't win by having more lines).
    ks = np.arange(n_cells + 1, dtype=np.int32)
    cap = float(np.percentile(prof_s, 90.0)) if prof_s.size > 0 else 0.0
    min_lines = max(6, n_cells - 2)  # allow missing a couple of outer lines

    for p in range(p_min, p_max + 1):
        for s in range(0, p):
            idxs = s + p * ks
            valid = idxs <= (n - 1)
            if not np.any(valid):
                continue
            vals = prof_s[idxs[valid]]
            if int(vals.size) < min_lines:
                continue
            # Clip to reduce domination by shoreline/terrain edges; gridlines are consistent but weaker.
            if cap > 0:
                vals = np.minimum(vals, cap)
            score = float(vals.mean())
            if score > best_score or (score == best_score and best_s is not None and s < best_s):
                best_score = score
                best_p = p
                best_s = s

    if best_p is None or best_s is None:
        # Fallback: just use naive dimensions (no offset)
        return 0.0, target, {"fallback": 1.0, "reason": 2.0}

    # Refine by snapping each predicted line to the strongest local peak, then least-squares fit.
    p0 = float(best_p)
    s0 = float(best_s)
    win = max(2, int(round(p0 * 0.35)))
    obs: List[float] = []
    kk_used: List[float] = []
    for k in range(n_cells + 1):
        t = s0 + p0 * float(k)
        # Skip lines that are far outside the profile range (outermost line may be missing due to crop)
        if t < -win or t > (n - 1) + win:
            continue
        lo = max(0, int(math.floor(t - win)))
        hi = min(n - 1, int(math.ceil(t + win)))
        if hi <= lo:
            continue
        j = lo + int(np.argmax(prof[lo : hi + 1]))
        obs.append(float(j))
        kk_used.append(float(k))

    if len(obs) >= 2:
        kk_arr = np.asarray(kk_used, dtype=np.float32)
        yy = np.asarray(obs, dtype=np.float32)
        # Fit yy ≈ a + b*kk
        b, a = np.polyfit(kk_arr, yy, 1)
        start = float(a)
        period = float(b)
    else:
        start = s0
        period = p0

    # Sanity clamp
    if not (p0 * 0.7 <= period <= p0 * 1.3):
        start = s0
        period = p0

    # Canonicalize the phase to the most plausible absolute start by shifting by ±period.
    # When the screenshot is cropped, the true grid start can be slightly negative, but
    # the phase search returns an equivalent start in [0, period). We pick the shift that
    # keeps the most cell centers inside the image width/height.
    if period > 0:
        pixel_size = float(n + 1)  # profile length is pixels-1
        centers = (np.arange(n_cells, dtype=np.float32) + 0.5) * float(period)

        best_start = start
        best_in = -1
        best_abs = float("inf")
        for m in [-2, -1, 0, 1, 2]:
            s_cand = float(start + m * period)
            c = s_cand + centers
            in_count = int(np.logical_and(c >= 0.0, c < pixel_size).sum())
            abs_s = abs(s_cand)
            if in_count > best_in or (in_count == best_in and abs_s < best_abs):
                best_in = in_count
                best_abs = abs_s
                best_start = s_cand
        start = float(best_start)

    dbg = {
        "coarse_period": float(best_p),
        "coarse_start": float(best_s),
        "refined_period": float(period),
        "refined_start": float(start),
        "score": float(best_score),
    }
    return start, period, dbg


def _compute_grid_geometry(pixels_rgb: np.ndarray, cols: int, rows: int) -> Dict[str, object]:
    """
    Compute grid geometry (top-left offset and cell sizes) by aligning to grid-line peaks.
    This is critical for robustness to cropping/extension: we must not assume grid spans the full image.
    """
    h, w = pixels_rgb.shape[:2]
    gray = pixels_rgb.astype(np.float32).mean(axis=2)

    # Gradient profiles (same as dimension detection)
    col_profile = np.abs(gray[:, 1:] - gray[:, :-1]).sum(axis=0)  # (W-1,)
    row_profile = np.abs(gray[1:, :] - gray[:-1, :]).sum(axis=1)  # (H-1,)

    x0, cell_w, dbg_x = _fit_grid_axis(col_profile, cols, w / float(cols))
    y0, cell_h, dbg_y = _fit_grid_axis(row_profile, rows, h / float(rows))

    return {
        "x0": float(x0),
        "y0": float(y0),
        "cell_w": float(cell_w),
        "cell_h": float(cell_h),
        "axis_x": dbg_x,
        "axis_y": dbg_y,
    }


def _is_cherry_pixel(r: int, g: int, b: int) -> bool:
    # Mirrors `isCherryPixel` in JS.
    # Cast to int to avoid uint8 overflow in comparisons
    r, g, b = int(r), int(g), int(b)
    # Cherries are sprite-like strong red pixels. This must be robust to:
    # - saturation reduction (r can drop below 150)
    # - modest hue/contrast/brightness shifts
    #
    # Use a mix of absolute and relative dominance checks.
    if r < 130:
        return False
    if (r - g) < 50 or (r - b) < 50:
        return False
    # Ratio check prevents brown/gray false positives when saturation is low.
    if r < 1.7 * (g + 1) or r < 1.7 * (b + 1):
        return False
    return True


def _quantize_hue_deg(h_deg: float, bin_deg: int = 15) -> int:
    h = float(h_deg) % 360.0
    q = round(h / float(bin_deg)) * float(bin_deg)
    return int(round(q)) % 360


def _detect_portal_cells_by_pixels(
    pixels_rgb: np.ndarray,
    x0: float,
    y0: float,
    cell_w: float,
    cell_h: float,
    rows: int,
    cols: int,
    grid: List[List[str]],
) -> Dict[str, object]:
    """
    Detect portal sprites by scanning each cell's center region for a dense cluster of
    vivid + bright pixels. This is designed to be robust even when:
    - blue portals are misclassified as water
    - red/orange portals are misclassified as cherries

    Returns:
      {
        "portal_cells": [{"x":int,"y":int,"hue_key":int,"hue_deg":float,"count":int}, ...],
        "portal_pairs": [((ax,ay),(bx,by),hue_key), ...],
      }
    """
    h, w = pixels_rgb.shape[:2]
    min_cell_px = float(min(cell_w, cell_h))
    # For smaller cells, use a denser scan so portals still have enough supporting pixels.
    step = 1 if min_cell_px < 60.0 else 2

    fx0, fx1 = 0.15, 0.85
    fy0, fy1 = 0.15, 0.85

    portal_cells: List[Dict[str, object]] = []

    eps = 1e-9
    for gy in range(rows):
        for gx in range(cols):
            if grid[gy][gx] == "horse":
                continue

            left = int(math.floor(x0 + (gx + fx0) * cell_w))
            right = int(math.ceil(x0 + (gx + fx1) * cell_w))
            top = int(math.floor(y0 + (gy + fy0) * cell_h))
            bottom = int(math.ceil(y0 + (gy + fy1) * cell_h))
            left = max(0, left)
            right = min(w, right)
            top = max(0, top)
            bottom = min(h, bottom)
            if right <= left or bottom <= top:
                continue

            region = pixels_rgb[top:bottom:step, left:right:step, :].astype(np.float32) / 255.0
            if region.size == 0:
                continue

            rf = region[:, :, 0]
            gf = region[:, :, 1]
            bf = region[:, :, 2]
            mx = np.maximum(np.maximum(rf, gf), bf)
            mn = np.minimum(np.minimum(rf, gf), bf)
            chroma = mx - mn
            v = mx
            s = np.where(mx > eps, chroma / (mx + eps), 0.0)

            # Thresholds tuned on fixtures (`portal.png`, `portal2.png`):
            # - require high saturation & value so grass doesn't trigger
            # - require significant chroma so the white horse doesn't trigger
            mask = (s > 0.55) & (v > 0.55) & (chroma > 0.25)
            count = int(mask.sum())

            sample_total = int(region.shape[0] * region.shape[1])
            # Portals occupy a large fraction of the center region; background terrain should not.
            # Use a conservative absolute floor to avoid false positives in water/terrain artifacts.
            # Note: 180 floor allows smaller portals like (1,5) in portal_cherries.png (count~198) to be detected.
            min_count = max(180, int(sample_total * 0.12))
            if count < min_count:
                continue

            # Hue computation (degrees)
            h6 = np.zeros_like(mx, dtype=np.float32)
            valid = chroma > eps
            mask_r = valid & (mx == rf)
            mask_g = valid & (mx == gf)
            mask_b = valid & (mx == bf)
            h6[mask_r] = np.mod((gf[mask_r] - bf[mask_r]) / (chroma[mask_r] + eps), 6.0)
            h6[mask_g] = (bf[mask_g] - rf[mask_g]) / (chroma[mask_g] + eps) + 2.0
            h6[mask_b] = (rf[mask_b] - gf[mask_b]) / (chroma[mask_b] + eps) + 4.0
            hue_deg = (h6 * 60.0) % 360.0

            hh = hue_deg[mask]
            ww = s[mask]
            if hh.size == 0:
                continue

            ang = hh * (math.pi / 180.0)
            sum_cos = float(np.cos(ang).dot(ww))
            sum_sin = float(np.sin(ang).dot(ww))
            mean_ang = math.atan2(sum_sin, sum_cos)
            mean_h_deg = (mean_ang * 180.0 / math.pi) % 360.0

            # Cherries are vivid red, but are much smaller than portals. In some layouts (e.g. portal_cherries.png),
            # the cherry sprite can look "portal-like" by our generic vivid-pixel test. Disambiguate by requiring
            # near-pure-red candidates to also have a high fill ratio in the cell center region.
            fill_ratio = float(count) / float(sample_total) if sample_total > 0 else 0.0
            red_dist = min(float(mean_h_deg), 360.0 - float(mean_h_deg))
            if red_dist <= 8.0 and fill_ratio < 0.28:
                continue

            hue_key = _quantize_hue_deg(mean_h_deg, 15)

            portal_cells.append(
                {"x": int(gx), "y": int(gy), "hue_key": int(hue_key), "hue_deg": float(mean_h_deg), "count": int(count)}
            )

    # Pair portals by quantized hue (greedy by Manhattan distance).
    by_hue: Dict[int, List[Tuple[int, int]]] = {}
    for c in portal_cells:
        k = int(c["hue_key"])
        by_hue.setdefault(k, []).append((int(c["x"]), int(c["y"])))

    portal_pairs: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []
    for hue_key, cells in by_hue.items():
        if len(cells) < 2:
            continue
        remaining = list(cells)
        while len(remaining) >= 2:
            a = remaining.pop(0)
            best_i = 0
            best_d = 10**9
            for i, b in enumerate(remaining):
                d = abs(a[0] - b[0]) + abs(a[1] - b[1])
                if d < best_d:
                    best_d = d
                    best_i = i
            b = remaining.pop(best_i)
            portal_pairs.append((a, b, int(hue_key)))

    return {"portal_cells": portal_cells, "portal_pairs": portal_pairs}


def _detect_cherry_cells_by_pixels(
    pixels_rgb: np.ndarray,
    x0: float,
    y0: float,
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

    # For very small images (downscaled), scan every pixel to avoid missing cherries.
    step = 1 if min(cell_w, cell_h) < 20 else 2
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

    # Only consider pixels inside the inferred grid bbox
    grid_left = float(x0)
    grid_top = float(y0)
    grid_right = float(x0 + cols * cell_w)
    grid_bottom = float(y0 + rows * cell_h)
    in_bbox = (xs >= grid_left) & (xs < grid_right) & (ys >= grid_top) & (ys < grid_bottom)
    xs = xs[in_bbox]
    ys = ys[in_bbox]
    if xs.size == 0:
        return np.zeros(rows * cols, dtype=np.uint8)

    cols_idx = np.floor((xs - grid_left) / float(cell_w)).astype(np.int32)
    rows_idx = np.floor((ys - grid_top) / float(cell_h)).astype(np.int32)
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


def _apply_gaussian_blur(pixels_rgb: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to reduce noise before processing."""
    img = Image.fromarray(pixels_rgb)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred, dtype=np.uint8)


def _analyze_color_clusters(
    pixels_rgb: np.ndarray,
    x0: float,
    y0: float,
    cell_w: float,
    cell_h: float,
    rows: int,
    cols: int,
    sample_points: List[Tuple[float, float]],
) -> Dict[str, object]:
    """
    Analyze the color distribution across all cells to adaptively determine
    water (blue) and grass (green) thresholds.
    
    Returns a dict with adaptive thresholds and cluster info.
    """
    h, w = pixels_rgb.shape[:2]
    
    # Sample colors from all cells
    all_samples = []
    
    for r in range(rows):
        for c in range(cols):
            for fx, fy in sample_points:
                px = int(np.floor(x0 + (c + fx) * cell_w))
                py = int(np.floor(y0 + (r + fy) * cell_h))
                if 0 <= px < w and 0 <= py < h:
                    rr, gg, bb = pixels_rgb[py, px, :]
                    all_samples.append([int(rr), int(gg), int(bb)])
    
    if not all_samples:
        # Fallback: no samples
        return {
            "water_threshold": {"b_min": 50, "r_max": 80, "g_max": 110},
            "method": "fallback",
        }
    
    samples_arr = np.array(all_samples, dtype=np.float32)  # (N, 3)
    
    # Identify blue-dominant pixels (potential water)
    # Use a more lenient criterion: B is highest OR B is close to highest
    blue_dominant = (
        ((samples_arr[:, 2] > samples_arr[:, 1]) & (samples_arr[:, 2] > samples_arr[:, 0])) |  # B is max
        ((samples_arr[:, 2] >= samples_arr[:, 1] - 15) & (samples_arr[:, 2] >= samples_arr[:, 0] - 15) & (samples_arr[:, 2] > 40))  # B is close to max and not too dark
    )
    blue_samples = samples_arr[blue_dominant]
    
    # Identify green-dominant pixels (potential grass)
    green_dominant = (samples_arr[:, 1] > samples_arr[:, 2]) & (samples_arr[:, 1] > samples_arr[:, 0])
    green_samples = samples_arr[green_dominant]
    
    # Adaptive thresholds for water
    if len(blue_samples) > 10:
        # We have blue pixels - compute adaptive threshold
        blue_mean = blue_samples.mean(axis=0)
        blue_std = blue_samples.std(axis=0)
        blue_percentile_25 = np.percentile(blue_samples, 25, axis=0)
        blue_percentile_90 = np.percentile(blue_samples, 90, axis=0)
        
        # Water threshold: B should be significantly higher than R and G
        # Use percentile-based approach to be more robust
        b_min = max(35, float(blue_percentile_25[2]) - 0.5 * float(blue_std[2]))
        
        # For R and G, use 90th percentile to be more lenient (handles hue shifts)
        r_max = max(80, float(blue_percentile_90[0]))
        g_max = max(100, float(blue_percentile_90[1]))
        
        # Clamp to reasonable ranges
        b_min = float(np.clip(b_min, 30, 80))
        r_max = float(np.clip(r_max, 70, 180))  # Very lenient for hue shifts
        g_max = float(np.clip(g_max, 90, 200))  # Very lenient for hue shifts
    else:
        # No clear blue cluster - try to find any blue-ish pixels
        # This handles extreme hue shifts where "blue" might shift to cyan/purple
        blue_ish = samples_arr[:, 2] > (samples_arr[:, 0] + samples_arr[:, 1]) / 2.5
        if blue_ish.sum() > 5:
            blue_ish_samples = samples_arr[blue_ish]
            b_min = max(30, float(np.percentile(blue_ish_samples[:, 2], 15)))
            r_max = max(80, float(np.percentile(blue_ish_samples[:, 0], 75)))
            g_max = max(110, float(np.percentile(blue_ish_samples[:, 1], 75)))
        else:
            # Absolute fallback
            b_min = 50.0
            r_max = 80.0
            g_max = 110.0
    
    # Adaptive thresholds for grass (for future use)
    grass_info = {}
    if len(green_samples) > 10:
        grass_mean = green_samples.mean(axis=0)
        grass_std = green_samples.std(axis=0)
        grass_info = {
            "mean_rgb": grass_mean.tolist(),
            "std_rgb": grass_std.tolist(),
        }
    
    return {
        "water_threshold": {
            "b_min": float(b_min),
            "r_max": float(r_max),
            "g_max": float(g_max),
        },
        "grass_info": grass_info,
        "method": "adaptive",
        "n_blue_samples": int(blue_dominant.sum()),
        "n_green_samples": int(green_dominant.sum()),
        "total_samples": len(all_samples),
    }


def _find_horse_by_brightest_square(
    pixels_rgb: np.ndarray,
    x0: float,
    y0: float,
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

    # Restrict search to the inferred grid bbox
    x_min = max(0, int(math.floor(x0)))
    y_min = max(0, int(math.floor(y0)))
    x_max = min(w, int(math.ceil(x0 + cols * cell_w)))
    y_max = min(h, int(math.ceil(y0 + rows * cell_h)))

    for y in range(y_min, max(y_min, y_max - cell_size + 1), step):
        for x in range(x_min, max(x_min, x_max - cell_size + 1), step):
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

    col = int(np.floor((center_x - x0) / cell_w))
    row = int(np.floor((center_y - y0) / cell_h))
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
    blur_radius: float = 0.0,
) -> ParsedGrid:
    """
    Parse an image file into a logical grid with adaptive color detection and offset calibration.
    If cols/rows are not provided, they are auto-detected (JS algorithm).
    
    Args:
        image_path: Path to the image file
        cols: Optional manual column count
        rows: Optional manual row count
        blur_radius: Gaussian blur radius for noise reduction (default 1.0)
    """
    pixels_original = load_image_rgb(image_path)
    h, w = pixels_original.shape[:2]

    # Auto-detect grid dimensions on ORIGINAL (unblurred) image
    auto_dbg: Dict[str, object] = {}
    if cols is None or rows is None:
        cols2, rows2, dbg = detect_grid_dimensions_from_pixels(pixels_original)
        cols = cols if cols is not None else cols2
        rows = rows if rows is not None else rows2
        auto_dbg = dbg

    assert cols is not None and rows is not None

    # Compute grid geometry by aligning to grid-line peaks (robust to crop/extend)
    geom = _compute_grid_geometry(pixels_original, cols, rows)
    x0 = float(geom["x0"])
    y0 = float(geom["y0"])
    cell_w = float(geom["cell_w"])
    cell_h = float(geom["cell_h"])

    # Apply Gaussian blur AFTER geometry for color sampling (optional; default off for JS parity)
    if blur_radius > 0:
        pixels = _apply_gaussian_blur(pixels_original, radius=blur_radius)
    else:
        pixels = pixels_original

    # Use 25 sample points for maximum robustness (5x5 grid).
    # For very small cell sizes (after downscaling), avoid sampling too close to edges
    # where interpolation/JPEG ringing blends colors across tiles.
    min_cell_px = float(min(cell_w, cell_h))
    coords = [0.3, 0.4, 0.5, 0.6, 0.7] if min_cell_px < 35.0 else [0.2, 0.35, 0.5, 0.65, 0.8]
    sample_points: List[Tuple[float, float]] = []
    for fy in coords:
        for fx in coords:
            sample_points.append((fx, fy))
    
    # Analyze color clusters adaptively
    cluster_info = _analyze_color_clusters(pixels, x0, y0, cell_w, cell_h, rows, cols, sample_points)
    water_thresh = cluster_info["water_threshold"]

    # JPEG/downscale robustness: when the image has strong 8x8 block artifacts and cells are small,
    # water edges can get slightly darkened and lose a couple of water votes. A tiny slack on b_min
    # fixes rare 1-cell flips (e.g. p4_scale_0.5_jpg_q20) while keeping non-JPEG inputs unchanged.
    blockiness = float(auto_dbg.get("blockiness", 1.0) or 1.0)
    if blockiness > 1.6 and min_cell_px < 35.0:
        water_thresh["b_min"] = float(max(30.0, float(water_thresh["b_min"]) - 2.0))

    grid: List[List[str]] = [["grass"] * cols for _ in range(rows)]
    water_count = 0
    grass_count = 0

    cherry_candidates = np.zeros(rows * cols, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            samples: List[Tuple[int, int, int]] = []
            for fx, fy in sample_points:
                px = int(np.floor(x0 + (c + fx) * cell_w))
                py = int(np.floor(y0 + (r + fy) * cell_h))
                if 0 <= px < w and 0 <= py < h:
                    rr, gg, bb = pixels[py, px, :]
                    samples.append((int(rr), int(gg), int(bb)))

            if not samples:
                grid[r][c] = "grass"
                grass_count += 1
                continue

            # Use MAJORITY VOTING for robust classification
            water_votes = 0
            grass_votes = 0
            cherry_hit = False
            
            for rr, gg, bb in samples:
                # Check for cherry first
                if (not cherry_hit) and _is_cherry_pixel(rr, gg, bb):
                    cherry_hit = True
                
                # Vote: is this sample point water or grass?
                # Use a lenient \"blue-ish\" rule so hue shifts (toward purple/magenta) still count as water.
                is_water_sample = (
                    bb > water_thresh["b_min"]
                    and rr <= water_thresh["r_max"]
                    and gg <= water_thresh["g_max"]
                    and bb >= (gg - 15)
                    and bb >= (rr - 15)
                )
                
                if is_water_sample:
                    water_votes += 1
                else:
                    grass_votes += 1
            
            # Majority wins
            is_water = water_votes > grass_votes

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
    horse_stats = _find_horse_by_brightest_square(pixels_original, x0, y0, cell_w, cell_h, rows, cols)

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

    # Detect portals BEFORE cherries and override misclassifications.
    portal_info = _detect_portal_cells_by_pixels(pixels_original, x0, y0, cell_w, cell_h, rows, cols, grid)
    portal_cells = portal_info.get("portal_cells", []) or []
    portal_pairs = portal_info.get("portal_pairs", []) or []
    portal_count = 0
    for c in portal_cells:
        px = int(c["x"])
        py = int(c["y"])
        if (px, py) == (hx, hy):
            continue
        prev = grid[py][px]
        if prev == "water":
            water_count = max(0, water_count - 1)
        if prev == "grass":
            grass_count = max(0, grass_count - 1)
        grid[py][px] = "portal"
        portal_count += 1

    # Cherries: pixel-scan detection (priority: reliability) + per-cell candidates.
    cherry_by_pixels = _detect_cherry_cells_by_pixels(pixels_original, x0, y0, cell_w, cell_h, rows, cols, grid)
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
        "grid_geom": geom,
        "water_count": water_count,
        "grass_count": grass_count,
        "cherry_count": cherry_count,
        "horse": (hx, hy),
        "horse_method": horse_method,
        "horse_brightness": horse_stats.get("brightness"),
        "horse_whiteness": horse_stats.get("whiteness"),
        "portal_count": portal_count,
        "portal_cells": portal_cells,
        "portal_pairs": portal_pairs,
        "color_clusters": cluster_info,
        "blur_radius": blur_radius,
    }

    return ParsedGrid(
        grid=grid,
        width=cols,
        height=rows,
        horse=(hx, hy),
        cherry_cells=cherry_cells,
        debug=debug,
    )


