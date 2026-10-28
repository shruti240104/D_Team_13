from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .contracts import STAGE_GEOMETRY, ScannerError, WarpResult

MIN_OUTPUT_SIDE = 50
MAX_OUTPUT_SIDE = 5000
MIN_SIDE_PX = 20            # shortest allowed quad side in source pixels
MIN_AREA_FRACTION = 0.005   # quad must cover >= 0.5 % of the image
OUTSIDE_MARGIN_FRACTION = 0.02


def _fail(code: str, message: str, hint: str = "") -> ScannerError:
    return ScannerError(STAGE_GEOMETRY, code, message, hint)


def _signed_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def validate_quad(corners, image_shape: Tuple[int, ...]) -> np.ndarray:
    try:
        quad = np.asarray(corners, dtype=np.float32)
    except (TypeError, ValueError):
        raise _fail("bad_corners", "Corners must be numeric (x, y) pairs.") from None
    if quad.shape != (4, 2):
        raise _fail("bad_corners", f"Expected 4 corners of shape (4, 2), got {quad.shape}.")
    if not np.all(np.isfinite(quad)):
        raise _fail("bad_corners", "Corner coordinates contain NaN or infinite values.")

    h, w = image_shape[:2]
    sides = [float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
    if min(sides) < MIN_SIDE_PX:
        raise _fail("degenerate", "Two corners are (almost) identical or the page is far too small.",
                    "Place the four corners on the four different page corners.")
    area = abs(_signed_area(quad))
    if area < MIN_AREA_FRACTION * w * h:
        raise _fail("degenerate", "The selected region is too small to be a page.")

    # Convexity + orientation: the cross product of consecutive edges must have
    # the same sign at all four corners.  Positive (image y-axis points down)
    # means clockwise on screen = TL,TR,BR,BL.
    crosses = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        v1, v2 = b - a, c - b
        crosses.append(float(v1[0] * v2[1] - v1[1] * v2[0]))
    if all(c < 0 for c in crosses):
        raise _fail("corner_order", "Corners are in the wrong order (counter-clockwise or mirrored).",
                    "Order must be top-left, top-right, bottom-right, bottom-left.")
    if not all(c > 0 for c in crosses):
        raise _fail("not_convex", "The corners form a twisted or concave shape, not a page outline.",
                    "Check that corners are not swapped.")
    return quad


def compute_output_size(quad: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    longest = max(width, height)
    if longest > MAX_OUTPUT_SIDE:
        f = MAX_OUTPUT_SIDE / longest
        width, height = width * f, height * f
    return max(MIN_OUTPUT_SIDE, int(round(width))), max(MIN_OUTPUT_SIDE, int(round(height)))


def warp_perspective(image: np.ndarray, corners) -> WarpResult:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise _fail("bad_image", "The warp needs a BGR colour image.")
    quad = validate_quad(corners, image.shape)
    h, w = image.shape[:2]
    warnings: List[str] = []

    margin_x, margin_y = OUTSIDE_MARGIN_FRACTION * w, OUTSIDE_MARGIN_FRACTION * h
    outside = ((quad[:, 0] < -margin_x) | (quad[:, 0] > w - 1 + margin_x)
               | (quad[:, 1] < -margin_y) | (quad[:, 1] > h - 1 + margin_y))
    if outside.any():
        warnings.append("Some corners lie outside the photo; the page is cut off and edges were extended.")

    out_w, out_h = compute_output_size(quad)
    destination = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
        raise _fail("invalid_transform", "The perspective transform could not be computed from these corners.")

    warped = cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    if float(warped.std()) < 2.0:
        raise _fail("implausible_result", "The corrected image is blank; the corners probably do not cover a page.",
                    "Adjust the corners or retake the photo.")
    aspect = out_w / float(out_h)
    if aspect > 3.0 or aspect < 1.0 / 3.0:
        warnings.append(f"Unusual page proportions ({out_w}x{out_h}); check that the corners are correct.")
    return WarpResult(image=warped, matrix=matrix, output_size=(out_w, out_h),
                      source_corners=quad, warnings=warnings)


def rotate_quarter_turns(image: np.ndarray, degrees_clockwise: int) -> np.ndarray:
    """Rotate by 0, 90, 180 or 270 degrees clockwise (lossless, no interpolation)."""
    mapping = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    d = int(degrees_clockwise) % 360
    if d not in mapping:
        raise _fail("bad_rotation", f"Only multiples of 90 degrees are supported, got {degrees_clockwise}.")
    return image.copy() if mapping[d] is None else cv2.rotate(image, mapping[d])
