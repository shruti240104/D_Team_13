from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .deskew import _text_mask, rotate_keep_all

SEARCH_SIDE = 1000          # candidates are read on a copy of at most this size (speed)
PROFILE_SIDE = 800          # the projection profile runs on an even smaller copy
MIN_WORD_CONF = 60.0        # words below this do not count as "read"
MIN_WORD_LETTERS = 3


def read_quality(result) -> int:
    """Letters inside confidently read words (0 when OCR failed or found nothing)."""
    if result is None or getattr(result, "status", None) != "ok":
        return 0
    total = 0
    for w in result.words:
        letters = sum(ch.isalpha() for ch in w.text)
        if w.conf >= MIN_WORD_CONF and letters >= MIN_WORD_LETTERS and letters >= 0.6 * len(w.text):
            total += letters
    return total


def _shrink(image: np.ndarray, side: int) -> np.ndarray:
    s = min(1.0, side / float(max(image.shape[:2])))
    return cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else image


def candidate_angles(image: np.ndarray, k: int = 3, step: float = 2.0, min_sep: float = 12.0) -> List[float]:
    """Angle 0 plus the ``k`` strongest, well-separated peaks of the line-projection profile."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = _shrink(gray, PROFILE_SIDE)
    mask = _text_mask(gray)
    pad = int(max(mask.shape) * 0.3)
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cx, cy = mask.shape[1] / 2.0, mask.shape[0] / 2.0
    angles = np.arange(-90.0, 90.0, step)
    scores = []
    for a in angles:
        m = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        rot = cv2.warpAffine(mask, m, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
        scores.append(float(np.sum(np.diff(rot.sum(axis=1).astype(np.float64)) ** 2)))
    chosen: List[float] = [0.0]
    for i in np.argsort(scores)[::-1]:
        a = float(angles[i])
        # distance on a 180-degree circle (-90 and +90 are the same direction)
        if all(min(abs(a - c), 180.0 - abs(a - c)) >= min_sep for c in chosen):
            chosen.append(a)
        if len(chosen) >= k + 1:
            break
    return chosen


def search_orientation(image: np.ndarray, read: Callable[[np.ndarray], object], k: int = 3
                       ) -> Tuple[float, int, int]:
    small = _shrink(image, SEARCH_SIDE)
    best_angle, best_q, base_q = 0.0, -1, 0
    for theta in candidate_angles(image, k=k):
        for phi in (theta, theta + 180.0):
            q = read_quality(read(rotate_keep_all(small, phi)))
            if phi == 0.0:
                base_q = q
            if q > best_q:
                best_angle, best_q = phi, q
    return best_angle, best_q, base_q
