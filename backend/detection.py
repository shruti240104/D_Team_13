
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .contracts import STAGE_DETECTION, DetectionResult, ScannerError

# ---- tunable constants (document any change in the PR description) ------------
MIN_AREA_RATIO = 0.10     # page must cover >= 10 % of the working image
MAX_AREA_RATIO = 0.97     # ... and < 97 % (otherwise it is the picture frame)
ANGLE_LIMITS_DEG = (35.0, 145.0)   # reject quads with very skewed interior angles
MIN_SIDE_FRACTION = 0.15  # shortest side must be >= 15 % of the image diagonal
MIN_FILL = 0.85           # hull vs. 4-gon agreement
MAX_CANDIDATES_PER_MASK = 12
ACCEPT_QUALITY = 0.60     # at/above: confident enough
MIN_QUALITY = 0.45        # below: report failure instead of guessing
AMBIGUITY_MARGIN = 0.05   # another distinct candidate within this quality gap = ambiguous
EDGE_TOLERANCE_PX = 7     # dilation size used when checking edge support
BORDER_MARGIN_FRACTION = 0.01   # a corner this close to the image edge "touches the border"
MAX_BORDER_CORNERS = 2    # >= 3 corners on the border = it is the picture frame, not a page



# Geometry helpers
def order_corners(points: Sequence[Sequence[float]]) -> np.ndarray:
   
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise ScannerError(STAGE_DETECTION, "bad_corners", "Exactly four corner points are required.")
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    ordered = pts[np.argsort(angles)]            # left -> top -> right -> bottom (clockwise)
    start = int(np.argmin(ordered.sum(axis=1)))  # top-left = smallest x + y
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def polygon_area(quad: np.ndarray) -> float:
    """Absolute area by the shoelace formula."""
    x, y = quad[:, 0], quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two convex quadrilaterals (exact, via OpenCV)."""
    a = np.asarray(a, np.float32).reshape(4, 2)
    b = np.asarray(b, np.float32).reshape(4, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = polygon_area(a) + polygon_area(b) - float(inter)
    return float(inter) / union if union > 1e-9 else 0.0


def interior_angles(quad: np.ndarray) -> List[float]:
    """Interior angle (degrees) at each vertex of a quadrilateral."""
    out = []
    for i in range(4):
        prev_pt, pt, next_pt = quad[i - 1], quad[i], quad[(i + 1) % 4]
        v1, v2 = prev_pt - pt, next_pt - pt
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom < 1e-9:
            out.append(0.0)
            continue
        cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        out.append(float(np.degrees(np.arccos(cos))))
    return out


def draw_corners(image: np.ndarray, corners: Optional[np.ndarray], color=(0, 200, 0)) -> np.ndarray:
    """Return a copy of ``image`` (BGR) with the quadrilateral and numbered corners drawn."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if corners is None:
        return out
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    thick = max(2, int(max(out.shape[:2]) / 300))
    cv2.polylines(out, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], True, color, thick, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(round(x)), int(round(y))), thick * 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (int(x) + thick * 4, int(y) - thick * 4),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, thick / 4.0), (255, 0, 0), max(1, thick - 1), cv2.LINE_AA)
    return out



# Candidate scoring
@dataclass
class _Candidate:
    quad: np.ndarray          # ordered TL,TR,BR,BL in WORKING-image coordinates
    quality: float
    area_ratio: float
    fill: float
    angle_score: float
    edge_support: float
    method: str
    via_minrect: bool = False

def refine_document_corners(
    quad: np.ndarray,
    edge_map: np.ndarray,
    search_radius: int = 6
) -> np.ndarray:
    """
    Refine document corners by fitting straight lines
    to nearby edge pixels.

    Works with general quadrilateral documents.
    """

    quad = order_corners(quad)
    h, w = edge_map.shape[:2]

    refined_lines = []

    for i in range(4):

        p1 = quad[i]
        p2 = quad[(i + 1) % 4]

        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length < 10:
            return quad

        normal = np.array(
            [-direction[1], direction[0]],
            dtype=np.float32
        ) / length

        points = []

        # Sample points along the detected side.
        for t in np.linspace(0.08, 0.92, 80):

            center = p1 + t * direction

            best_point = None
            best_distance = float("inf")

            # Search perpendicular to the side.
            for offset in range(
                -search_radius,
                search_radius + 1
            ):

                sample = center + offset * normal

                x = int(round(sample[0]))
                y = int(round(sample[1]))

                if 0 <= x < w and 0 <= y < h:

                    if edge_map[y, x] > 0:

                        distance = abs(offset)

                        if distance < best_distance:
                            best_distance = distance
                            best_point = [x, y]

            if best_point is not None:
                points.append(best_point)

        if len(points) < 15:
            return quad

        points = np.asarray(points, dtype=np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            points,
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01
        ).flatten()

        refined_lines.append(
            (
                np.array([x0, y0]),
                np.array([vx, vy])
            )
        )

    def intersection(line1, line2):

        p1, d1 = line1
        p2, d2 = line2

        matrix = np.column_stack((d1, -d2))

        if abs(np.linalg.det(matrix)) < 1e-6:
            return None

        parameters = np.linalg.solve(
            matrix,
            p2 - p1
        )

        return p1 + parameters[0] * d1

    refined = []

    for i in range(4):

        previous_line = refined_lines[(i - 1) % 4]
        current_line = refined_lines[i]

        point = intersection(
            previous_line,
            current_line
        )

        if point is None:
            return quad

        refined.append(point)

    refined = np.asarray(
        refined,
        dtype=np.float32
    )

    # Avoid large unexpected corner movements.
    displacement = np.linalg.norm(
        refined - quad,
        axis=1
    )

    if np.max(displacement) > search_radius * 3:
        return quad

    if (
        np.any(refined[:, 0] < 0)
        or np.any(refined[:, 0] >= w)
        or np.any(refined[:, 1] < 0)
        or np.any(refined[:, 1] >= h)
    ):
        return quad

    if not cv2.isContourConvex(
        refined.astype(np.float32).reshape(-1, 1, 2)
    ):
        return quad

    return refined

def _edge_support(quad: np.ndarray, edge_map: np.ndarray) -> float:
    """Fraction of points sampled along the quad's sides that sit on an edge pixel."""
    h, w = edge_map.shape[:2]
    hits = total = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        for t in np.linspace(0.05, 0.95, 30):
            x, y = a + (b - a) * t
            xi, yi = int(round(x)), int(round(y))
            total += 1
            if 0 <= xi < w and 0 <= yi < h and edge_map[yi, xi] > 0:
                hits += 1
    return hits / total if total else 0.0


def _evaluate_contour(contour: np.ndarray, image_area: float, edge_map: np.ndarray,
                      method: str) -> Optional[_Candidate]:
   
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    ratio = hull_area / image_area
    if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
        return None

    perimeter = cv2.arcLength(hull, True)
    quad = None
    via_minrect = False
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
        if len(approx) < 4:
            break
    if quad is None:
        box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        box_area = polygon_area(box)
        if box_area > 0 and hull_area / box_area >= 0.88:
            quad, via_minrect = box, True
        else:
            return None

    quad = order_corners(quad)
    quad_area = polygon_area(quad)
    if quad_area <= 0:
        return None

    # --- plausibility checks -------------------------------------------------
    angles = interior_angles(quad)
    if min(angles) < ANGLE_LIMITS_DEG[0] or max(angles) > ANGLE_LIMITS_DEG[1]:
        return None
    diag = float(np.hypot(*(quad.max(axis=0) - quad.min(axis=0))))
    sides = [float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
    if min(sides) < MIN_SIDE_FRACTION * diag:
        return None
    fill = min(hull_area, quad_area) / max(hull_area, quad_area)
    if fill < MIN_FILL:
        return None

    
    img_h, img_w = edge_map.shape[:2]
    mx, my = BORDER_MARGIN_FRACTION * img_w, BORDER_MARGIN_FRACTION * img_h
    on_border = int(np.sum((quad[:, 0] <= mx) | (quad[:, 0] >= img_w - 1 - mx)
                           | (quad[:, 1] <= my) | (quad[:, 1] >= img_h - 1 - my)))
    if on_border > MAX_BORDER_CORNERS:
        return None

    # --- score -----------------------------------------------------------------
    area_ratio = quad_area / image_area
    angle_dev = max(abs(a - 90.0) for a in angles)
    angle_score = float(
        np.clip(
            1.0 - angle_dev / 60.0,
            0.0,
            1.0
        )
    )
    size_score = float(min(1.0, area_ratio / 0.4))
    support = _edge_support(quad, edge_map)
    quality = 0.30 * fill + 0.20 * angle_score + 0.20 * size_score + 0.30 * support
    if via_minrect:
        quality *= 0.9  # approximation: small penalty
    return _Candidate(quad, float(quality), float(area_ratio), float(fill), angle_score, support,
                      method, via_minrect)


def _candidates_from_mask(mask: np.ndarray, retrieval: int, image_area: float,
                          edge_map: np.ndarray, method: str) -> List[_Candidate]:
    contours, _ = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CANDIDATES_PER_MASK]
    out = []
    for c in contours:
        cand = _evaluate_contour(c, image_area, edge_map, method)
        if cand is not None:
            out.append(cand)
    return out



# Strategies: each returns (mask_or_edges, contour_retrieval_mode)
def _close(mask: np.ndarray, size: int, iterations: int = 1) -> np.ndarray:
    """Morphological closing = dilate then erode: fills small gaps and holes.

    For edge maps it reconnects broken page borders (so the contour becomes
    closed); for threshold masks it fills the holes left by dark text.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def _canny_edges(gray_blur: np.ndarray, sensitive: bool) -> np.ndarray:
    """Canny edges.  Normal mode derives thresholds from Otsu's split of the
    histogram (high = Otsu value, low = half of it), which adapts to dark and
    bright photos better than fixed numbers."""
    if sensitive:
        equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_blur)
        return cv2.Canny(equalised, 20, 60)
    otsu, _ = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(float(otsu), 40.0)
    return cv2.Canny(gray_blur, 0.5 * high, high)


def _mask_canny(gray_blur, bgr, sensitive=False):
    return _close(_canny_edges(gray_blur, sensitive), 5, iterations=2), cv2.RETR_LIST


def _mask_otsu(gray_blur, bgr):
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = _close(th, 15)                       # fill holes left by dark text
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop specks
    return th, cv2.RETR_EXTERNAL


def _mask_hsv(gray_blur, bgr):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    value_thr, _ = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > value_thr)).astype(np.uint8) * 255
    mask = _close(mask, 15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask, cv2.RETR_EXTERNAL


STRATEGIES: List[Tuple[str, Callable]] = [
    ("canny", lambda g, b: _mask_canny(g, b, sensitive=False)),
    ("canny_sensitive", lambda g, b: _mask_canny(g, b, sensitive=True)),
    ("otsu_threshold", _mask_otsu),
    ("hsv_paper", _mask_hsv),
]

_COLORS = [(255, 128, 0), (0, 200, 255), (255, 0, 200), (200, 200, 0)]



# Public API
def detect_document(
    working_bgr: np.ndarray,
    scale: Tuple[float, float] = (1.0, 1.0),
    original_size: Optional[Tuple[int, int]] = None,
) -> DetectionResult:
   
    if (not isinstance(working_bgr, np.ndarray) or working_bgr.ndim != 3
            or working_bgr.shape[2] != 3 or working_bgr.dtype != np.uint8):
        raise ScannerError(STAGE_DETECTION, "bad_input", "Detection needs a BGR uint8 image (H, W, 3).")

    h, w = working_bgr.shape[:2]
    image_area = float(h * w)
    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    # Gaussian blur suppresses paper grain / JPEG noise so Canny does not fire on it.
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge map used only to judge candidates (union of normal + sensitive edges).
    edge_union = cv2.bitwise_or(_canny_edges(gray_blur, False), _canny_edges(gray_blur, True))
    edge_tolerant = cv2.dilate(edge_union, np.ones((EDGE_TOLERANCE_PX, EDGE_TOLERANCE_PX), np.uint8))

    debug: Dict[str, np.ndarray] = {"gray": gray, "edges": edge_union}
    best_per_strategy: List[_Candidate] = []
    considered = 0
    for name, strategy in STRATEGIES:
        mask, retrieval = strategy(gray_blur, working_bgr)
        debug[f"mask_{name}"] = mask
        cands = _candidates_from_mask(mask, retrieval, image_area, edge_tolerant, name)
        considered += len(cands)
        if cands:
            best_per_strategy.append(max(cands, key=lambda c: c.quality))

    overlay = working_bgr.copy()
    for idx, cand in enumerate(best_per_strategy):
        cv2.polylines(overlay, [np.round(cand.quad).astype(np.int32).reshape(-1, 1, 2)], True,
                      _COLORS[idx % len(_COLORS)], 1, cv2.LINE_AA)

    if not best_per_strategy:
        debug["candidates"] = overlay
        return DetectionResult(
            success=False, method="", failure_code="no_candidates", candidates_considered=0,
            failure_reason=("No page-like four-sided region was found. The page may blend into the "
                            "background, be cut off by the image border, or not be in the photo."),
            debug=debug,
        )

    # Highest quality wins; max() keeps the earlier strategy on ties.
    best = max(best_per_strategy, key=lambda c: c.quality)
    agreeing = sum(1 for c in best_per_strategy if c is not best and quad_iou(c.quad, best.quad) >= 0.9)
    ambiguous = any(
        c is not best and quad_iou(c.quad, best.quad) < 0.5 and c.quality >= best.quality - AMBIGUITY_MARGIN
        for c in best_per_strategy
    )
    cv2.polylines(overlay, [np.round(best.quad).astype(np.int32).reshape(-1, 1, 2)], True,
                  (0, 255, 0), 3, cv2.LINE_AA)
    debug["candidates"] = overlay

    if best.quality < MIN_QUALITY:
        return DetectionResult(
            success=False, quality=best.quality, method=best.method, failure_code="low_quality",
            failure_reason=(f"Only weak candidates were found (best score {best.quality:.2f}, "
                            f"minimum {MIN_QUALITY:.2f}). The page outline is not reliable."),
            agreeing_methods=agreeing, ambiguous=ambiguous, candidates_considered=considered, debug=debug,
        )

    # Refine corners using the original, non-dilated edge map.
    refined_quad = refine_document_corners(
        best.quad,
        edge_union,
        search_radius=6
    )

    # Check whether refinement improves edge alignment.
    original_support = _edge_support(best.quad, edge_tolerant)
    refined_support = _edge_support(refined_quad, edge_tolerant)

    if refined_support >= original_support:
        final_quad = refined_quad
    else:
        final_quad = best.quad

    # Convert working-image coordinates to original-image coordinates.
    corners = final_quad * np.array(
        [scale[0], scale[1]],
        dtype=np.float32
    )
    if original_size is not None:
        corners[:, 0] = np.clip(corners[:, 0], 0, original_size[0] - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, original_size[1] - 1)

    warnings: List[str] = []
    if best.quality < ACCEPT_QUALITY:
        warnings.append(f"Detection is uncertain (score {best.quality:.2f}); check the outline or adjust the corners.")
    if ambiguous:
        warnings.append("Another, different region looks almost as page-like; verify the outline.")
    if best.via_minrect:
        warnings.append("The page is not exactly four-sided (curled/rounded corners?); a bounding rectangle was used.")

    return DetectionResult(
        success=True, corners=corners.astype(np.float32), quality=best.quality,
        method=best.method + ("+minrect" if best.via_minrect else ""),
        agreeing_methods=agreeing, ambiguous=ambiguous, candidates_considered=considered,
        warnings=warnings, debug=debug,
    )
