"""Click-to-annotate tool for real photos  (Member 1).

Opens each image without a .json file; click the four page corners (any order),
press ENTER to save, R to redo, S to skip, ESC to quit.  Saves ``<image>.json``
with corners in ORIGINAL pixel coordinates, ordered TL,TR,BR,BL.
Requires the normal ``opencv-python`` package (a GUI window), not the headless build.

    python -m evaluation.annotate_corners path/to/photos
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from backend.detection import order_corners
from evaluation.dataset import IMAGE_SUFFIXES, load_annotation


def annotate(folder: str) -> None:
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or "corners" in load_annotation(path):
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        f = min(1.0, 1000 / max(img.shape[:2]))
        view = cv2.resize(img, None, fx=f, fy=f) if f < 1 else img.copy()
        points = []

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))

        cv2.namedWindow("annotate")
        cv2.setMouseCallback("annotate", on_mouse)
        while True:
            canvas = view.copy()
            for p in points:
                cv2.circle(canvas, p, 5, (0, 0, 255), -1)
            cv2.imshow("annotate", canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                return
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (ord("s"), ord("S")):
                break
            if key == 13 and len(points) == 4:
                corners = order_corners(np.array(points, np.float32) / f)
                ann = load_annotation(path)
                ann["corners"] = corners.round(1).tolist()
                path.with_suffix(".json").write_text(json.dumps(ann, indent=2), encoding="utf-8")
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    annotate(sys.argv[1] if len(sys.argv) > 1 else "sample_data")
