"""Document-detection evaluation  (Member 1).

Measures, on a folder of images WITH corner annotations (see dataset.py):
  * success rate     fraction of images where IoU(pred, truth) >= --iou-threshold
  * corner error     mean distance between matching corners / image diagonal
  * IoU              overlap of predicted and true quadrilaterals
  * failures         images where detection reported failure

The IoU threshold (default 0.90) is a team decision; state it in your report.
No numbers are built in: run it on your own data.

    python -m evaluation.detection_metrics sample_data/synthetic --csv detection.csv
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from backend.detection import detect_document, order_corners, quad_iou
from backend.input_handler import load_image, make_working_copy
from evaluation.dataset import iter_dataset, mean, write_csv


def corner_error(pred: np.ndarray, truth: np.ndarray, image_shape) -> float:
    """Mean Euclidean corner distance divided by the image diagonal (0 = perfect)."""
    h, w = image_shape[:2]
    diag = float(np.hypot(w, h))
    p, t = order_corners(pred), order_corners(truth)
    return float(np.mean(np.linalg.norm(p - t, axis=1)) / diag)


def evaluate_image(image_bgr: np.ndarray, truth: np.ndarray, max_side: int = 800) -> Dict:
    h, w = image_bgr.shape[:2]
    wc = make_working_copy(image_bgr, max_side)
    t0 = time.perf_counter()
    det = detect_document(wc.image, (wc.scale_x, wc.scale_y), (w, h))
    elapsed = time.perf_counter() - t0
    row = {"detected": det.success, "method": det.method, "quality": round(det.quality, 3),
           "failure_code": det.failure_code, "seconds": round(elapsed, 3)}
    if det.success:
        row["iou"] = round(quad_iou(det.corners, truth), 4)
        row["corner_error"] = round(corner_error(det.corners, truth, image_bgr.shape), 4)
    return row


def evaluate_folder(folder: str, iou_threshold: float = 0.90, max_side: int = 800):
    rows: List[Dict] = []
    for path, ann in iter_dataset(folder):
        if "corners" not in ann:
            continue
        try:
            loaded = load_image(path.read_bytes(), path.name)
        except Exception as exc:  # unreadable dataset image: record, continue
            rows.append({"image": path.name, "detected": False, "failure_code": f"load:{exc}"})
            continue
        # annotations are in coordinates of the stored image; rescale if load_image down-scaled it
        sx = loaded.image.shape[1] / loaded.original_size[0]
        sy = loaded.image.shape[0] / loaded.original_size[1]
        truth = np.asarray(ann["corners"], np.float32) * np.array([sx, sy], np.float32)
        row = {"image": path.name}
        row.update(evaluate_image(loaded.image, truth, max_side))
        row["success"] = bool(row.get("iou", 0.0) >= iou_threshold)
        rows.append(row)
    ious = [r["iou"] for r in rows if "iou" in r]
    errs = [r["corner_error"] for r in rows if "corner_error" in r]
    summary = {
        "images": len(rows),
        "detected": sum(1 for r in rows if r.get("detected")),
        "success_rate": (sum(1 for r in rows if r.get("success")) / len(rows)) if rows else None,
        "iou_threshold": iou_threshold, "mean_iou": mean(ious), "mean_corner_error": mean(errs),
    }
    return rows, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--iou-threshold", type=float, default=0.90)
    ap.add_argument("--csv", help="write per-image results to this CSV file")
    args = ap.parse_args()
    rows, summary = evaluate_folder(args.folder, args.iou_threshold)
    for r in rows:
        print(r)
    print("SUMMARY", summary)
    if args.csv:
        write_csv(args.csv, rows)
