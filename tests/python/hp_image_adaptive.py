"""
Adaptive color detection for Horse Pen image parsing.

This version:
- Applies Gaussian blur to reduce noise
- Analyzes color histogram to adaptively detect water/grass
- Uses clustering to identify blue (water) and green (grass) regions
- Tests robustness with color jittering
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# Ensure repo root is on sys.path so we can import `hp_image`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

# Import existing utilities
from hp_image import (
    CHERRY_BONUS,
    ParsedGrid,
    detect_grid_dimensions_from_pixels,
    _find_horse_by_brightest_square,
    _is_cherry_pixel,
    _detect_cherry_cells_by_pixels,
)


def apply_gaussian_blur(pixels_rgb: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to reduce noise before processing."""
    img = Image.fromarray(pixels_rgb)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred, dtype=np.uint8)


def apply_color_jitter(
    pixels_rgb: np.ndarray,
    hue_shift: float = 0.0,
    saturation_scale: float = 1.0,
    contrast_scale: float = 1.0,
    brightness_scale: float = 1.0,
) -> np.ndarray:
    """
    Apply color jittering for testing robustness.
    
    Args:
        hue_shift: Hue shift in degrees (-180 to 180)
        saturation_scale: Saturation multiplier (0.5 = less saturated, 1.5 = more saturated)
        contrast_scale: Contrast multiplier
        brightness_scale: Brightness multiplier
    """
    img = Image.fromarray(pixels_rgb)
    
    # Apply brightness
    if brightness_scale != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_scale)
    
    # Apply contrast
    if contrast_scale != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_scale)
    
    # Apply saturation
    if saturation_scale != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_scale)
    
    # Hue shift (convert to HSV, shift H, convert back)
    if hue_shift != 0.0:
        arr = np.array(img, dtype=np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # RGB to HSV
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c
        
        v = max_c
        s = np.where(max_c > 0, delta / max_c, 0)
        
        h = np.zeros_like(max_c)
        mask_r = (max_c == r) & (delta > 0)
        mask_g = (max_c == g) & (delta > 0)
        mask_b = (max_c == b) & (delta > 0)
        
        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
        h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
        h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
        h = h / 6.0  # Normalize to [0, 1]
        
        # Shift hue
        h = (h + hue_shift / 360.0) % 1.0
        
        # HSV to RGB
        h6 = h * 6.0
        c = v * s
        x = c * (1 - np.abs(h6 % 2 - 1))
        m = v - c
        
        r_new = np.zeros_like(h)
        g_new = np.zeros_like(h)
        b_new = np.zeros_like(h)
        
        mask0 = (h6 >= 0) & (h6 < 1)
        mask1 = (h6 >= 1) & (h6 < 2)
        mask2 = (h6 >= 2) & (h6 < 3)
        mask3 = (h6 >= 3) & (h6 < 4)
        mask4 = (h6 >= 4) & (h6 < 5)
        mask5 = (h6 >= 5) & (h6 < 6)
        
        r_new[mask0], g_new[mask0], b_new[mask0] = c[mask0], x[mask0], 0
        r_new[mask1], g_new[mask1], b_new[mask1] = x[mask1], c[mask1], 0
        r_new[mask2], g_new[mask2], b_new[mask2] = 0, c[mask2], x[mask2]
        r_new[mask3], g_new[mask3], b_new[mask3] = 0, x[mask3], c[mask3]
        r_new[mask4], g_new[mask4], b_new[mask4] = x[mask4], 0, c[mask4]
        r_new[mask5], g_new[mask5], b_new[mask5] = c[mask5], 0, x[mask5]
        
        arr[:, :, 0] = np.clip((r_new + m) * 255, 0, 255)
        arr[:, :, 1] = np.clip((g_new + m) * 255, 0, 255)
        arr[:, :, 2] = np.clip((b_new + m) * 255, 0, 255)
        
        img = Image.fromarray(arr.astype(np.uint8))
    
    return np.array(img, dtype=np.uint8)


def analyze_color_clusters(
    pixels_rgb: np.ndarray,
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
                px = int(np.floor((c + fx) * cell_w))
                py = int(np.floor((r + fy) * cell_h))
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
    
    # Compute statistics
    mean_rgb = samples_arr.mean(axis=0)
    std_rgb = samples_arr.std(axis=0)
    
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
            b_min = 50
            r_max = 80
            g_max = 110
    
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


def parse_image_to_grid_adaptive(
    image_path: str,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    blur_radius: float = 1.0,
    color_jitter: Optional[Dict[str, float]] = None,
) -> ParsedGrid:
    """
    Parse an image with adaptive color detection.
    
    Args:
        image_path: Path to the image
        cols: Optional manual column count
        rows: Optional manual row count
        blur_radius: Gaussian blur radius (0 = no blur)
        color_jitter: Optional dict with hue_shift, saturation_scale, contrast_scale, brightness_scale
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img, dtype=np.uint8)
    
    # Apply color jittering (for testing)
    if color_jitter:
        pixels = apply_color_jitter(
            pixels,
            hue_shift=color_jitter.get("hue_shift", 0.0),
            saturation_scale=color_jitter.get("saturation_scale", 1.0),
            contrast_scale=color_jitter.get("contrast_scale", 1.0),
            brightness_scale=color_jitter.get("brightness_scale", 1.0),
        )
    
    # Apply Gaussian blur
    if blur_radius > 0:
        pixels = apply_gaussian_blur(pixels, radius=blur_radius)
    
    h, w = pixels.shape[:2]
    
    # Auto-detect grid dimensions
    auto_dbg: Dict[str, object] = {}
    if cols is None or rows is None:
        cols2, rows2, dbg = detect_grid_dimensions_from_pixels(pixels)
        cols = cols if cols is not None else cols2
        rows = rows if rows is not None else rows2
        auto_dbg = dbg
    
    assert cols is not None and rows is not None
    
    cell_w = w / float(cols)
    cell_h = h / float(rows)
    
    # Sample points for each cell
    sample_points = [(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    
    # Analyze color clusters adaptively
    cluster_info = analyze_color_clusters(pixels, cell_w, cell_h, rows, cols, sample_points)
    water_thresh = cluster_info["water_threshold"]
    
    # Parse grid with adaptive thresholds
    grid: List[List[str]] = [["grass"] * cols for _ in range(rows)]
    water_count = 0
    grass_count = 0
    cherry_candidates = np.zeros(rows * cols, dtype=np.uint8)
    
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
            
            # Check for cherries
            cherry_hit = False
            for rr, gg, bb in samples:
                if (not cherry_hit) and _is_cherry_pixel(rr, gg, bb):
                    cherry_hit = True
                    break
            
            # Adaptive water detection
            is_water = (
                avg_b > avg_g
                and avg_b > avg_r
                and avg_b > water_thresh["b_min"]
                and avg_r < water_thresh["r_max"]
                and avg_g < water_thresh["g_max"]
            )
            
            if is_water:
                grid[r][c] = "water"
                water_count += 1
            else:
                grid[r][c] = "grass"
                grass_count += 1
            
            if (not is_water) and cherry_hit:
                cherry_candidates[r * cols + c] = 1
    
    # Find horse using brightest square algorithm
    horse_method = "brightest_square"
    horse_stats = _find_horse_by_brightest_square(pixels, cell_w, cell_h, rows, cols)
    
    if horse_stats is None:
        raise RuntimeError("Could not detect horse position")
    
    hx = int(horse_stats["x"])
    hy = int(horse_stats["y"])
    hx = max(0, min(cols - 1, hx))
    hy = max(0, min(rows - 1, hy))
    
    # Adjust counts if we overwrite water/grass
    prev = grid[hy][hx]
    if prev == "water":
        water_count = max(0, water_count - 1)
    if prev == "grass":
        grass_count = max(0, grass_count - 1)
    grid[hy][hx] = "horse"
    
    # Detect cherries
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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hp_image_adaptive.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Testing adaptive color detection on: {image_path}")
    print("=" * 60)
    
    # Test 1: Original image
    print("\n1. Original (no jitter, blur=1.0)")
    pg = parse_image_to_grid_adaptive(image_path, blur_radius=1.0)
    print(f"   Grid: {pg.width}x{pg.height}, Horse: {pg.horse}, Cherries: {len(pg.cherry_cells)}")
    print(f"   Water: {pg.debug['water_count']}, Grass: {pg.debug['grass_count']}")
    print(f"   Color clusters: {pg.debug['color_clusters']['method']}")
    print(f"   Water thresholds: {pg.debug['color_clusters']['water_threshold']}")
    
    # Test 2: Hue shift
    print("\n2. Hue shift +20°")
    pg2 = parse_image_to_grid_adaptive(
        image_path, blur_radius=1.0, color_jitter={"hue_shift": 20}
    )
    print(f"   Grid: {pg2.width}x{pg2.height}, Horse: {pg2.horse}, Cherries: {len(pg2.cherry_cells)}")
    print(f"   Water: {pg2.debug['water_count']}, Grass: {pg2.debug['grass_count']}")
    
    # Test 3: Saturation scale
    print("\n3. Saturation 0.7x")
    pg3 = parse_image_to_grid_adaptive(
        image_path, blur_radius=1.0, color_jitter={"saturation_scale": 0.7}
    )
    print(f"   Grid: {pg3.width}x{pg3.height}, Horse: {pg3.horse}, Cherries: {len(pg3.cherry_cells)}")
    print(f"   Water: {pg3.debug['water_count']}, Grass: {pg3.debug['grass_count']}")
    
    # Test 4: Contrast scale
    print("\n4. Contrast 1.3x")
    pg4 = parse_image_to_grid_adaptive(
        image_path, blur_radius=1.0, color_jitter={"contrast_scale": 1.3}
    )
    print(f"   Grid: {pg4.width}x{pg4.height}, Horse: {pg4.horse}, Cherries: {len(pg4.cherry_cells)}")
    print(f"   Water: {pg4.debug['water_count']}, Grass: {pg4.debug['grass_count']}")
    
    # Test 5: Brightness scale
    print("\n5. Brightness 0.9x")
    pg5 = parse_image_to_grid_adaptive(
        image_path, blur_radius=1.0, color_jitter={"brightness_scale": 0.9}
    )
    print(f"   Grid: {pg5.width}x{pg5.height}, Horse: {pg5.horse}, Cherries: {len(pg5.cherry_cells)}")
    print(f"   Water: {pg5.debug['water_count']}, Grass: {pg5.debug['grass_count']}")
    
    # Test 6: Combined jitter
    print("\n6. Combined (hue+20, sat=0.8, contrast=1.2, brightness=0.95)")
    pg6 = parse_image_to_grid_adaptive(
        image_path,
        blur_radius=1.0,
        color_jitter={
            "hue_shift": 20,
            "saturation_scale": 0.8,
            "contrast_scale": 1.2,
            "brightness_scale": 0.95,
        },
    )
    print(f"   Grid: {pg6.width}x{pg6.height}, Horse: {pg6.horse}, Cherries: {len(pg6.cherry_cells)}")
    print(f"   Water: {pg6.debug['water_count']}, Grass: {pg6.debug['grass_count']}")
    
    print("\n" + "=" * 60)
    print("✓ Adaptive color detection test complete!")

