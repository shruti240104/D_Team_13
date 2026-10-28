from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.detection import order_corners

PAGE_W, PAGE_H = 1240, 1754   # A4 at 150 dpi
DEFAULT_LINES = [
    "DOCUMENT SCANNER TEST PAGE",
    "",
    "The quick brown fox jumps over the lazy dog.",
    "Computer vision turns photographs into scans.",
    "Invoice number 20481 dated 12 March 2025.",
    "Total amount due is 1250 dollars.",
    "Perspective correction makes the page flat.",
    "Adaptive thresholding handles uneven light.",
    "Tesseract reads the cleaned image.",
    "",
    "Student project report section one.",
    "All measurements were taken on site.",
]
BACKGROUNDS = ("dark", "wood", "cluttered", "lowcontrast")


@dataclass
class SyntheticSample:
    image: np.ndarray            # BGR photo
    corners: np.ndarray          # (4,2) float32 ground truth TL,TR,BR,BL
    page: np.ndarray             # the flat page that was rendered
    text: str                    # ground-truth text (lines joined by \n, blank lines removed)


def render_text_page(lines: Optional[Sequence[str]] = None, size: Tuple[int, int] = (PAGE_W, PAGE_H),
                     font_scale: float = 1.5, thickness: int = 2) -> np.ndarray:
    """White-ish page with dark text lines (Hershey font; deterministic)."""
    w, h = size
    page = np.full((h, w, 3), (244, 244, 240), np.uint8)
    y = 160
    for line in (lines if lines is not None else DEFAULT_LINES):
        if line:
            cv2.putText(page, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), thickness, cv2.LINE_AA)
        y += 110
    return page


def make_background(kind: str, size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    w, h = size
    if kind == "dark":
        bg = np.full((h, w, 3), (40, 45, 50), np.uint8)
    elif kind == "wood":
        bg = np.full((h, w, 3), (70, 110, 160), np.uint8)
        streak = (20 * np.sin(np.arange(h) / 6.0) + rng.normal(0, 6, h))[:, None, None]
        bg = np.clip(bg + streak, 0, 255).astype(np.uint8)
    elif kind == "cluttered":
        bg = np.full((h, w, 3), (90, 100, 70), np.uint8)
        for _ in range(25):
            colour = tuple(int(c) for c in rng.integers(20, 200, 3))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            if rng.random() < 0.5:
                cv2.rectangle(bg, p1, p2, colour, -1)
            else:
                cv2.line(bg, p1, p2, colour, int(rng.integers(2, 12)))
    elif kind == "lowcontrast":
        bg = np.full((h, w, 3), (228, 230, 232), np.uint8)   # almost as bright as the paper
    else:
        raise ValueError(f"Unknown background '{kind}'")
    return np.clip(bg + rng.normal(0, 3, bg.shape), 0, 255).astype(np.uint8)


def make_photo(page: Optional[np.ndarray] = None, canvas: Tuple[int, int] = (1200, 900), background: str = "dark",
               rotation_deg: float = 0.0, perspective: float = 0.0, shadow: float = 0.0, blur: float = 0.0,
               noise: float = 0.0, seed: int = 0, lines: Optional[Sequence[str]] = None) -> SyntheticSample:
    """Project ``page`` onto a background with a known perspective.

    rotation_deg  in-plane rotation of the page
    perspective   0..1 amount of random corner displacement (keystone effect)
    shadow        0..0.6 brightness fall-off across the photo (uneven light)
    blur / noise  Gaussian blur sigma / noise std (camera defects)
    """
    rng = np.random.default_rng(seed)
    text_lines = list(lines) if lines is not None else DEFAULT_LINES
    page = render_text_page(text_lines) if page is None else page
    ph, pw = page.shape[:2]
    cw, ch = canvas
    height = ch * 0.78
    width = height * pw / ph
    half = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                     [width / 2, height / 2], [-width / 2, height / 2]])
    a = np.radians(rotation_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    quad = half @ rot.T + rng.uniform(-1, 1, (4, 2)) * perspective * min(width, height) * 0.12
    quad += np.array([cw / 2, ch / 2])
    quad[:, 0] = np.clip(quad[:, 0], 4, cw - 5)
    quad[:, 1] = np.clip(quad[:, 1], 4, ch - 5)
    quad = quad.astype(np.float32)

    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(page, matrix, (cw, ch), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), matrix, (cw, ch), flags=cv2.INTER_NEAREST)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255.0
    img = make_background(background, canvas, rng).astype(np.float32) * (1 - mask) + warped.astype(np.float32) * mask

    if shadow > 0:
        ramp = np.linspace(1.0, 1.0 - shadow, cw, dtype=np.float32)[None, :, None]
        img = img * ramp
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    text = "\n".join(l for l in text_lines if l)
    return SyntheticSample(image=img, corners=order_corners(quad), page=page, text=text)


def save_sample(sample: SyntheticSample, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{name}.jpg"), sample.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (folder / f"{name}.json").write_text(
        json.dumps({"corners": sample.corners.round(2).tolist(), "text": sample.text, "synthetic": True}, indent=2),
        encoding="utf-8")


def build_dataset(out_dir: str, count: int = 12, seed: int = 0) -> List[str]:
    """Create ``count`` varied synthetic samples (cycling through background kinds)."""
    rng = np.random.default_rng(seed)
    names = []
    for i in range(count):
        kind = BACKGROUNDS[i % len(BACKGROUNDS)]
        sample = make_photo(background=kind, rotation_deg=float(rng.uniform(-20, 20)),
                            perspective=float(rng.uniform(0.2, 1.0)), shadow=float(rng.choice([0, 0.3, 0.5])),
                            blur=float(rng.choice([0, 0, 1.2])), noise=float(rng.choice([0, 4, 8])), seed=int(seed * 1000 + i))
        name = f"synthetic_{i:02d}_{kind}"
        save_sample(sample, Path(out_dir), name)
        names.append(name)
    return names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic scanner test photos with ground truth.")
    parser.add_argument("--out", default="sample_data/synthetic")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    made = build_dataset(args.out, args.count, args.seed)
    print(f"Wrote {len(made)} samples to {args.out}")
