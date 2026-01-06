"""
Test color robustness of the adaptive color detection algorithm.
"""

import os
import sys
from hp_image_adaptive import parse_image_to_grid_adaptive

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIXTURES_DIR = os.path.join(ROOT_DIR, "tests", "fixtures")

def test_image_robustness(image_path: str):
    """Test an image with various color jittering scenarios."""
    
    print(f"\n{'='*70}")
    print(f"Testing: {image_path}")
    print('='*70)
    
    # Baseline
    print("\n1. Baseline (blur=1.0, no jitter)")
    pg_baseline = parse_image_to_grid_adaptive(image_path, blur_radius=1.0)
    print(f"   Grid: {pg_baseline.width}x{pg_baseline.height}")
    print(f"   Horse: {pg_baseline.horse}")
    print(f"   Water: {pg_baseline.debug['water_count']}, Grass: {pg_baseline.debug['grass_count']}, Cherries: {len(pg_baseline.cherry_cells)}")
    print(f"   Water thresholds: b_min={pg_baseline.debug['color_clusters']['water_threshold']['b_min']:.1f}")
    
    # Test scenarios
    scenarios = [
        ("Hue +30°", {"hue_shift": 30}),
        ("Hue -30°", {"hue_shift": -30}),
        ("Saturation 0.6x", {"saturation_scale": 0.6}),
        ("Saturation 1.4x", {"saturation_scale": 1.4}),
        ("Contrast 0.8x", {"contrast_scale": 0.8}),
        ("Contrast 1.3x", {"contrast_scale": 1.3}),
        ("Brightness 0.85x", {"brightness_scale": 0.85}),
        ("Brightness 1.15x", {"brightness_scale": 1.15}),
        ("Combined extreme", {
            "hue_shift": 25,
            "saturation_scale": 0.7,
            "contrast_scale": 1.25,
            "brightness_scale": 0.9,
        }),
    ]
    
    all_passed = True
    
    for i, (name, jitter) in enumerate(scenarios, start=2):
        print(f"\n{i}. {name}")
        try:
            pg = parse_image_to_grid_adaptive(image_path, blur_radius=1.0, color_jitter=jitter)
            
            # Check if results are consistent
            grid_match = pg.width == pg_baseline.width and pg.height == pg_baseline.height
            horse_match = pg.horse == pg_baseline.horse
            water_tolerance = abs(pg.debug['water_count'] - pg_baseline.debug['water_count']) <= 5
            cherry_match = len(pg.cherry_cells) == len(pg_baseline.cherry_cells)
            
            status = "✓" if (grid_match and horse_match and water_tolerance and cherry_match) else "✗"
            
            print(f"   {status} Grid: {pg.width}x{pg.height} (expected {pg_baseline.width}x{pg_baseline.height})")
            print(f"   {status} Horse: {pg.horse} (expected {pg_baseline.horse})")
            print(f"   {status} Water: {pg.debug['water_count']} (expected ~{pg_baseline.debug['water_count']})")
            print(f"   {status} Cherries: {len(pg.cherry_cells)} (expected {len(pg_baseline.cherry_cells)})")
            
            if not (grid_match and horse_match and water_tolerance and cherry_match):
                all_passed = False
                print(f"   ⚠️  MISMATCH DETECTED")
        
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            all_passed = False
    
    print(f"\n{'='*70}")
    if all_passed:
        print(f"✓ ALL TESTS PASSED for {image_path}")
    else:
        print(f"✗ SOME TESTS FAILED for {image_path}")
    print('='*70)
    
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/python/test_color_robustness.py <image1> [image2] ...")
        print(f"Tip: test images live in: {FIXTURES_DIR}")
        sys.exit(1)
    
    all_images_passed = True
    
    for arg in sys.argv[1:]:
        image_path = arg
        if not os.path.exists(image_path):
            cand = os.path.join(FIXTURES_DIR, arg)
            if os.path.exists(cand):
                image_path = cand
        passed = test_image_robustness(image_path)
        if not passed:
            all_images_passed = False
    
    print(f"\n\n{'='*70}")
    if all_images_passed:
        print("✓✓✓ ALL IMAGES PASSED ALL TESTS ✓✓✓")
    else:
        print("✗✗✗ SOME IMAGES FAILED TESTS ✗✗✗")
    print('='*70)

