from __future__ import annotations

import cv2
import numpy as np

MIN_DESKEW_ANGLE = 0.4       # smaller tilts are ignored (not worth the resampling)
WORK_SIDE = 800              # the angle search runs on a copy of at most this size
MIN_GAIN = 1.05              # best angle must beat angle 0 by 5 % to be trusted


def _text_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (255 = ink) that is robust to uneven lighting."""
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)


def estimate_text_skew(image: np.ndarray, max_angle: float = 30.0) -> float:
    """Angle in degrees (counter-clockwise positive, OpenCV convention) by which the
    image must be rotated so that its text lines become horizontal.  0.0 if unsure."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    s = min(1.0, WORK_SIDE / float(max(gray.shape)))
    if s < 1.0:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = _text_mask(gray)
    if int(np.count_nonzero(mask)) < 0.002 * mask.size:      # (almost) no ink: nothing to align
        return 0.0
    pad = int(max(mask.shape) * 0.3)                           # room so rotated text is not cut off
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cx, cy = mask.shape[1] / 2.0, mask.shape[0] / 2.0

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(mask, m, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
        profile = rot.sum(axis=1).astype(np.float64)
        return float(np.sum(np.diff(profile) ** 2))

    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    best = float(max(coarse, key=score))
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    best = float(max(fine, key=score))
    base = score(0.0)
    if base > 0 and score(best) < MIN_GAIN * base:             # no clear winner: leave the page alone
        return 0.0
    return round(best, 2)


def paper_colour(image: np.ndarray):
    """Typical paper colour (median of the brightest 30 % of pixels), used to fill the new corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = gray >= np.percentile(gray, 70)
    if image.ndim == 3:
        return tuple(int(np.median(image[:, :, c][mask])) for c in range(3))
    return int(np.median(gray[mask]))


def rotate_keep_all(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by ``angle`` degrees (counter-clockwise positive) and enlarge the canvas so nothing is cut."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += new_w / 2.0 - w / 2.0
    m[1, 2] += new_h / 2.0 - h / 2.0
    return cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=paper_colour(image))


def deskew(image: np.ndarray, max_angle: float = 30.0):
    """Return ``(straightened_image, angle_applied)``; the image is returned unchanged if the tilt is tiny."""
    angle = estimate_text_skew(image, max_angle)
    if abs(angle) < MIN_DESKEW_ANGLE:
        return image, 0.0
    return rotate_keep_all(image, angle), angle