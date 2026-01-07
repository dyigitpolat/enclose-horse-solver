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


