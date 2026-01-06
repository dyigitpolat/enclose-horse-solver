"""
Test grid dimension detection robustness against cropping and extending.
"""

import os
import sys
import numpy as np
from PIL import Image
from typing import Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIXTURES_DIR = os.path.join(ROOT_DIR, "tests", "fixtures")
sys.path.insert(0, ROOT_DIR)

from hp_image import detect_grid_dimensions_from_pixels


def crop_or_extend_image(pixels: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """
    Crop or extend an image.
    Positive values = crop (remove pixels)
    Negative values = extend (mirror pixels)
    """
    h, w = pixels.shape[:2]
    
    # Handle top
    if top > 0:
        pixels = pixels[top:, :, :]
    elif top < 0:
        mirror_top = pixels[:abs(top), :, :][::-1, :, :]
        pixels = np.vstack([mirror_top, pixels])
    
    # Handle bottom
    if bottom > 0:
        pixels = pixels[:-bottom, :, :]
    elif bottom < 0:
        mirror_bottom = pixels[-abs(bottom):, :, :][::-1, :, :]
        pixels = np.vstack([pixels, mirror_bottom])
    
    # Handle left
    if left > 0:
        pixels = pixels[:, left:, :]
    elif left < 0:
        mirror_left = pixels[:, :abs(left), :][:, ::-1, :]
        pixels = np.hstack([mirror_left, pixels])
    
    # Handle right
    if right > 0:
        pixels = pixels[:, :-right, :]
    elif right < 0:
        mirror_right = pixels[:, -abs(right):, :][:, ::-1, :]
        pixels = np.hstack([pixels, mirror_right])
    
    return pixels


def test_image_grid_robustness(image_path: str, num_tests: int = 10):
    """Test grid detection with random cropping/extending."""
    
    print(f"\n{'='*70}")
    print(f"Testing grid detection: {image_path}")
    print('='*70)
    
    # Load original image
    img = Image.open(image_path).convert("RGB")
    pixels_original = np.array(img, dtype=np.uint8)
    
    # Detect baseline grid
    print("\n1. Baseline (no modifications)")
    cols_baseline, rows_baseline, dbg = detect_grid_dimensions_from_pixels(pixels_original)
    print(f"   Grid: {cols_baseline}x{rows_baseline}")
    print(f"   Fallback: {dbg.get('fallback', False)}")
    if not dbg.get('fallback', False):
        print(f"   Best periods: col={dbg.get('best_col_period')}, row={dbg.get('best_row_period')}")
    
    # Test with random crops/extends
    print(f"\n2. Testing {num_tests} random crop/extend scenarios...")
    
    all_passed = True
    matches = 0
    
    for i in range(num_tests):
        # Random crop/extend amounts (10-15 pixels, can be positive or negative)
        top = np.random.randint(-15, 16)
        bottom = np.random.randint(-15, 16)
        left = np.random.randint(-15, 16)
        right = np.random.randint(-15, 16)
        
        # Apply transformation
        pixels_modified = crop_or_extend_image(pixels_original, top, bottom, left, right)
        
        # Detect grid
        cols, rows, _ = detect_grid_dimensions_from_pixels(pixels_modified)
        
        match = (cols == cols_baseline) and (rows == rows_baseline)
        if match:
            matches += 1
            status = "✓"
        else:
            status = "✗"
            all_passed = False
        
        print(f"   Test {i+1:2d}: {status} crop({top:+3d},{bottom:+3d},{left:+3d},{right:+3d}) -> {cols}x{rows} (expected {cols_baseline}x{rows_baseline})")
    
    print(f"\n{'='*70}")
    print(f"Results: {matches}/{num_tests} passed ({100*matches/num_tests:.1f}%)")
    if all_passed:
        print(f"✓ ALL TESTS PASSED")
    else:
        print(f"✗ {num_tests - matches} TESTS FAILED")
    print('='*70)
    
    return matches == num_tests


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/python/test_grid_robustness.py <image1> [image2] ...")
        print(f"Tip: test images live in: {FIXTURES_DIR}")
        sys.exit(1)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    all_images_passed = True
    
    for arg in sys.argv[1:]:
        image_path = arg
        if not os.path.exists(image_path):
            cand = os.path.join(FIXTURES_DIR, arg)
            if os.path.exists(cand):
                image_path = cand
        passed = test_image_grid_robustness(image_path, num_tests=20)
        if not passed:
            all_images_passed = False
    
    print(f"\n\n{'='*70}")
    if all_images_passed:
        print("✓✓✓ ALL IMAGES PASSED ALL TESTS ✓✓✓")
    else:
        print("✗✗✗ SOME IMAGES FAILED TESTS ✗✗✗")
    print('='*70)

