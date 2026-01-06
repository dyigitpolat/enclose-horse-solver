"""
Generate robustness fixtures (bounded to realistic limits):
- JPEG round-trip (moderate compression)
- Downscale to smaller resolutions (interpolation blur)
- Combined downscale + JPEG round-trip

Outputs PNG files into: tests/fixtures/
Naming:
  <base>_jpg_qXX.png
  <base>_scale_0.5.png
  <base>_scale_0.5_jpg_q20.png

These are intended to be picked up by:
- tests/python/verify_parsing_robustness.py
- tests/js/test_js_parsing.js

Run:
  source venv/bin/activate
  python tests/python/generate_jpg_scale_fixtures.py
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from PIL import Image
from PIL import ImageEnhance
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIXTURES_DIR = os.path.join(ROOT_DIR, "tests", "fixtures")


def _png_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return out


def downscale(img: Image.Image, scale: float) -> Image.Image:
    s = float(scale)
    w, h = img.size
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    # Bilinear is common for resizes and creates challenging blur.
    return img.resize((nw, nh), resample=Image.BILINEAR)


def apply_color_jitter(
    img: Image.Image,
    hue_shift: float = 0.0,
    saturation_scale: float = 1.0,
    contrast_scale: float = 1.0,
    brightness_scale: float = 1.0,
) -> Image.Image:
    """Apply deterministic color jittering transformations (same logic as create_test_images.py)."""

    if brightness_scale != 1.0:
        img = ImageEnhance.Brightness(img).enhance(float(brightness_scale))
    if contrast_scale != 1.0:
        img = ImageEnhance.Contrast(img).enhance(float(contrast_scale))
    if saturation_scale != 1.0:
        img = ImageEnhance.Color(img).enhance(float(saturation_scale))

    # Hue shift (convert to HSV, shift H, convert back)
    if hue_shift != 0.0:
        arr = np.array(img, dtype=np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # RGB -> HSV
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
        h = h / 6.0

        # Shift hue
        h = (h + float(hue_shift) / 360.0) % 1.0

        # HSV -> RGB
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


def crop_or_extend(img: Image.Image, top: int, bottom: int, left: int, right: int) -> Image.Image:
    """
    Crop or extend an image.
    Positive = crop (remove pixels)
    Negative = extend (mirror pixels)
    """
    arr = np.array(img, dtype=np.uint8)

    if top > 0:
        arr = arr[top:, :, :]
    elif top < 0:
        mirror_top = arr[: abs(top), :, :][::-1, :, :]
        arr = np.vstack([mirror_top, arr])

    if bottom > 0:
        arr = arr[: -bottom, :, :]
    elif bottom < 0:
        mirror_bottom = arr[-abs(bottom) :, :, :][::-1, :, :]
        arr = np.vstack([arr, mirror_bottom])

    if left > 0:
        arr = arr[:, left:, :]
    elif left < 0:
        mirror_left = arr[:, : abs(left), :][:, ::-1, :]
        arr = np.hstack([mirror_left, arr])

    if right > 0:
        arr = arr[:, : -right, :]
    elif right < 0:
        mirror_right = arr[:, -abs(right) :, :][:, ::-1, :]
        arr = np.hstack([arr, mirror_right])

    return Image.fromarray(arr.astype(np.uint8))


@dataclass(frozen=True)
class VariantSpec:
    suffix: str
    build: callable  # (img) -> img


def generate_for_base(base_name: str, qualities: List[int], scales: List[float]) -> List[str]:
    base_path = _png_path(base_name)
    if not os.path.exists(base_path):
        raise FileNotFoundError(base_path)

    img0 = Image.open(base_path).convert("RGB")
    created: List[str] = []

    # JPEG-only
    for q in qualities:
        out_name = f"{os.path.splitext(base_name)[0]}_jpg_q{q:02d}.png"
        out_path = _png_path(out_name)
        if os.path.exists(out_path):
            continue
        img = jpeg_roundtrip(img0, q)
        img.save(out_path, format="PNG")
        created.append(out_name)

    # Downscale-only
    for s in scales:
        tag = f"{s:.2f}".rstrip("0").rstrip(".")
        out_name = f"{os.path.splitext(base_name)[0]}_scale_{tag}.png"
        out_path = _png_path(out_name)
        if os.path.exists(out_path):
            continue
        img = downscale(img0, s)
        img.save(out_path, format="PNG")
        created.append(out_name)

    # Combined: downscale + JPEG (use a representative low quality)
    if qualities:
        q = min(qualities)
        for s in scales:
            tag = f"{s:.2f}".rstrip("0").rstrip(".")
            out_name = f"{os.path.splitext(base_name)[0]}_scale_{tag}_jpg_q{q:02d}.png"
            out_path = _png_path(out_name)
            if os.path.exists(out_path):
                continue
            img = downscale(img0, s)
            img = jpeg_roundtrip(img, q)
            img.save(out_path, format="PNG")
            created.append(out_name)

    # Combined: (color jitter / crop variants) + downscale + JPEG
    # Apply color/crop first, then downscale, then JPEG (most realistic pipeline).
    if qualities and scales:
        q = min(qualities)
        s = min(scales)
        tag = f"{s:.2f}".rstrip("0").rstrip(".")
        stem = os.path.splitext(base_name)[0]

        variants: List[Tuple[str, dict, Tuple[int, int, int, int]]] = [
            ("hue_shift_20", {"hue_shift": 20.0}, (0, 0, 0, 0)),
            ("hue_shift_m20", {"hue_shift": -20.0}, (0, 0, 0, 0)),
            ("saturation_low", {"saturation_scale": 0.7}, (0, 0, 0, 0)),
            ("contrast_high", {"contrast_scale": 1.3}, (0, 0, 0, 0)),
            # same as create_test_images.py
            ("combined_1", {"hue_shift": 15.0, "saturation_scale": 0.8}, (-12, 10, 8, -11)),
            ("combined_2", {"hue_shift": -25.0, "contrast_scale": 1.2, "brightness_scale": 0.9}, (9, -14, -7, 13)),
        ]

        for name, jitter, crop in variants:
            out_name = f"{stem}_scale_{tag}_jpg_q{q:02d}_{name}.png"
            out_path = _png_path(out_name)
            if os.path.exists(out_path):
                continue
            img = img0
            if jitter:
                img = apply_color_jitter(img, **jitter)
            top, bottom, left, right = crop
            if any(crop):
                img = crop_or_extend(img, top, bottom, left, right)
            img = downscale(img, s)
            img = jpeg_roundtrip(img, q)
            img.save(out_path, format="PNG")
            created.append(out_name)

    return created


def main() -> None:
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    bases = ["p2.png", "p3.png", "p4.png", "p5.png"]
    # Keep this bounded: scale in [0.5, 1.0] and JPEG quality >= 20.
    # (The verifier will also ignore any older/out-of-policy fixtures still present.)
    qualities = [20]
    scales = [0.5]

    print(f"fixtures: {FIXTURES_DIR}")
    total_created: List[str] = []
    for b in bases:
        print(f"\nBase: {b}")
        created = generate_for_base(b, qualities=qualities, scales=scales)
        if created:
            for name in created:
                print("  +", name)
        else:
            print("  (no new files; already generated)")
        total_created.extend(created)

    print(f"\nDone. Created {len(total_created)} new fixture(s).")


if __name__ == "__main__":
    main()


