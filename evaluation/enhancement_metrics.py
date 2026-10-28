"""Enhancement comparison  (Member 3).

For every image: run detection + warp, then report simple readability numbers
for the raw flattened page and for each enhanced variant:
  contrast     paper-minus-ink separation (Otsu split)         higher is better
  sharpness    variance of Laplacian                           higher = crisper
  bg_std       std-dev of the estimated paper background       lower = more even light
and save a side-by-side montage (raw | color | grayscale | bw) for visual review.
These numbers describe the image; they do not prove OCR improved - use
``ocr_metrics`` for that.

    python -m evaluation.enhancement_metrics sample_data/synthetic --out enhancement_report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from backend.enhancement import (EnhancementOptions, background_uniformity, sharpness_score,
                                 text_paper_separation, to_grayscale)
from backend.pipeline import PipelineOptions, run_pipeline
from evaluation.dataset import iter_dataset, write_csv


def measure(gray: np.ndarray) -> dict:
    return {"contrast": round(text_paper_separation(gray), 1), "sharpness": round(sharpness_score(gray), 1),
            "bg_std": round(background_uniformity(gray), 2)}


def montage(images, height: int = 500) -> np.ndarray:
    tiles = []
    for img in images:
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        f = height / bgr.shape[0]
        tiles.append(cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    return np.hstack(tiles)


def run(folder: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, _ in iter_dataset(folder):
        res = run_pipeline(path.read_bytes(), path.name, PipelineOptions(run_ocr=False))
        if not res.ok:
            rows.append({"image": path.name, "error": f"{res.error.stage}/{res.error.code}"})
            continue
        raw_gray = to_grayscale(res.warp.image)
        sets = {"raw": raw_gray, "color": to_grayscale(res.enhancement.variants["color"]),
                "grayscale": res.enhancement.variants["grayscale"], "bw": res.enhancement.variants["bw"]}
        row = {"image": path.name}
        for name, g in sets.items():
            for k, v in measure(g).items():
                row[f"{name}_{k}"] = v
        rows.append(row)
        cv2.imwrite(str(out / f"{path.stem}_compare.png"),
                    montage([res.warp.image, res.enhancement.variants["color"],
                             res.enhancement.variants["grayscale"], res.enhancement.variants["bw"]]))
    write_csv(str(out / "enhancement_metrics.csv"), rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="enhancement_report")
    a = ap.parse_args()
    for r in run(a.folder, a.out):
        print(r)
    print(f"Montages and CSV written to {a.out}/")
