"""
Create test versions of an image with color jittering and crop/extend.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIXTURES_DIR = os.path.join(ROOT_DIR, "tests", "fixtures")

def apply_color_jitter(img, hue_shift=0, saturation_scale=1.0, contrast_scale=1.0, brightness_scale=1.0):
    """Apply color jittering transformations."""
    
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
    
    return img


def crop_or_extend(img, top, bottom, left, right):
    """
    Crop or extend an image.
    Positive = crop (remove pixels)
    Negative = extend (mirror pixels)
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    
    # Handle top
    if top > 0:
        arr = arr[top:, :, :]
    elif top < 0:
        mirror_top = arr[:abs(top), :, :][::-1, :, :]
        arr = np.vstack([mirror_top, arr])
    
    # Handle bottom
    if bottom > 0:
        arr = arr[:-bottom, :, :]
    elif bottom < 0:
        mirror_bottom = arr[-abs(bottom):, :, :][::-1, :, :]
        arr = np.vstack([arr, mirror_bottom])
    
    # Handle left
    if left > 0:
        arr = arr[:, left:, :]
    elif left < 0:
        mirror_left = arr[:, :abs(left), :][:, ::-1, :]
        arr = np.hstack([mirror_left, arr])
    
    # Handle right
    if right > 0:
        arr = arr[:, :-right, :]
    elif right < 0:
        mirror_right = arr[:, -abs(right):, :][:, ::-1, :]
        arr = np.hstack([arr, mirror_right])
    
    return Image.fromarray(arr.astype(np.uint8))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/python/create_test_images.py <image_path>")
        print(f"Tip: test images live in: {FIXTURES_DIR}")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        cand = os.path.join(FIXTURES_DIR, image_path)
        if os.path.exists(cand):
            image_path = cand
    base_name = image_path.rsplit('.', 1)[0]
    
    img = Image.open(image_path).convert("RGB")
    
    # Test variations
    variations = [
        {
            "name": "hue_shift_20",
            "desc": "Hue shift +20°",
            "jitter": {"hue_shift": 20},
            "crop": (0, 0, 0, 0),
        },
        {
            "name": "hue_shift_m20",
            "desc": "Hue shift -20°",
            "jitter": {"hue_shift": -20},
            "crop": (0, 0, 0, 0),
        },
        {
            "name": "saturation_low",
            "desc": "Saturation 0.7x",
            "jitter": {"saturation_scale": 0.7},
            "crop": (0, 0, 0, 0),
        },
        {
            "name": "contrast_high",
            "desc": "Contrast 1.3x",
            "jitter": {"contrast_scale": 1.3},
            "crop": (0, 0, 0, 0),
        },
        {
            "name": "crop_random_1",
            "desc": "Random crop/extend #1",
            "jitter": {},
            "crop": (-10, 12, -8, 15),  # top, bottom, left, right
        },
        {
            "name": "crop_random_2",
            "desc": "Random crop/extend #2",
            "jitter": {},
            "crop": (14, -9, 11, -13),
        },
        {
            "name": "combined_1",
            "desc": "Combined: hue+15, sat=0.8, crop",
            "jitter": {"hue_shift": 15, "saturation_scale": 0.8},
            "crop": (-12, 10, 8, -11),
        },
        {
            "name": "combined_2",
            "desc": "Combined: hue-25, contrast=1.2, brightness=0.9, crop",
            "jitter": {"hue_shift": -25, "contrast_scale": 1.2, "brightness_scale": 0.9},
            "crop": (9, -14, -7, 13),
        },
    ]
    
    print(f"Creating test variations of {image_path}...")
    print("=" * 70)
    
    for var in variations:
        print(f"\n{var['name']}: {var['desc']}")
        
        # Apply jittering
        img_var = img
        if var["jitter"]:
            img_var = apply_color_jitter(img, **var["jitter"])
        
        # Apply crop/extend
        top, bottom, left, right = var["crop"]
        if any(var["crop"]):
            img_var = crop_or_extend(img_var, top, bottom, left, right)
            print(f"  Crop: top={top:+3d}, bottom={bottom:+3d}, left={left:+3d}, right={right:+3d}")
        
        # Save
        output_path = f"{base_name}_{var['name']}.png"
        img_var.save(output_path)
        print(f"  Saved: {output_path} ({img_var.width}x{img_var.height})")
    
    print("\n" + "=" * 70)
    print(f"✓ Created {len(variations)} test variations")

