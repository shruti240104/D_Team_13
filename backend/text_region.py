"""Text-aware crop protection.

Problem this solves
-------------------
The page detector looks for a bright four-sided region.  On some photos it locks
onto something *smaller* than the page (a table, a box, a folded part, a shadow
edge) and the crop then cuts text off.  When detection fails altogether the
pipeline used to fall back to the whole photo, which is also not ideal.

The rule implemented here is simple: **never crop away text.**

  1. ``find_text_blocks``      finds where the text is (no OCR, pure image
                               processing: dark marks on a locally lighter
                               background, grouped into lines/blocks).
  2. ``expand_quad_to_text``   takes the detected page outline and pushes its
                               sides outwards, only as far as needed, until
                               every nearby text block is inside.  The outline
                               keeps its own tilt/perspective.
  3. ``text_region_quad``      when no page outline was found, returns a box
                               around all the text.  If the text is spread over
                               (almost) the whole photo that box is the full
                               photo - "text all over the page = full page".

All work happens on the down-scaled working copy; the returned corners are
multiplied by ``scale`` so they refer to the ORIGINAL image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---- tunable constants --------------------------------------------------------
INK_CONTRAST = 28            # a pixel is "ink" if it is this much darker than its surroundings
MIN_INK_AREA = 3             # px (working scale): smaller specks are noise
MAX_GLYPH_HEIGHT_FRACTION = 0.06  # a letter/word blob taller than this (of the longest side) is not text
MAX_GLYPH_WIDTH_FRACTION = 0.25   # ... or wider than this (touching letters join into whole words)
PAPER_MAX_SATURATION = 120    # text sits on (nearly) colourless paper; coloured desks / wood are not text
MAX_STROKE_ASPECT = 8.0      # long thin streaks (wood grain, ruled lines, edges) are not letters
MIN_STREAK_FILL = 0.12       # a big blob that fills < 12 % of its box is a thin diagonal streak, not a word
MIN_MEDIAN_GLYPH_HEIGHT = 0.003   # (of longest side) a block of tiny specks (noisy streak edges, dust) is not text
MIN_COMPONENTS_PER_BLOCK = 2     # a text block needs at least this many ink blobs (rejects grain/dirt)
BLOCK_GAP_FRACTION = 0.018   # blobs closer than this (of the longest side) are merged into one block
PROXIMITY_FRACTION = 0.25    # text farther than this outside the outline is treated as "not on the page"
PAD_FRACTION = 0.03         # safety margin kept around text
FULL_PAGE_FRACTION = 0.70    # text box covering >= this share of the photo -> use the whole photo


@dataclass
class TextBlock:
    points: np.ndarray       # (N, 2) float32: outline points of every ink blob
    bbox: Tuple[int, int, int, int]   # x, y, w, h of the block
    n_components: int


# =============================================================================
# 1. Where is the text?
# =============================================================================
def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Dark-on-lighter marks.  The local background is estimated with a large
    median filter (strokes are thin, so the median is the paper colour)."""
    side = max(gray.shape[:2])
    k = int(side * 0.04) | 1
    k = max(k, 15)
    background = cv2.medianBlur(gray, k)
    diff = cv2.subtract(background, gray)          # positive where the pixel is darker than paper
    return (diff >= INK_CONTRAST).astype(np.uint8) * 255


def find_text_blocks(working_bgr: np.ndarray) -> List[TextBlock]:
    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    side = max(h, w)

    ink = _ink_mask(gray)
    sat_bg = cv2.medianBlur(cv2.cvtColor(working_bgr, cv2.COLOR_BGR2HSV)[:, :, 1], max(15, int(side * 0.04) | 1))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    max_h, max_w = MAX_GLYPH_HEIGHT_FRACTION * side, MAX_GLYPH_WIDTH_FRACTION * side
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_INK_AREA or bw > max_w or bh > max_h:
            continue
        if max(bw, bh) / float(max(1, min(bw, bh))) > MAX_STROKE_ASPECT and max(bw, bh) > 0.015 * side:
            continue
        if max(bw, bh) > 0.03 * side and area < MIN_STREAK_FILL * bw * bh:
            continue
        cy, cx = y + bh // 2, x + bw // 2
        if sat_bg[min(cy, h - 1), min(cx, w - 1)] > PAPER_MAX_SATURATION:
            continue
        keep[i] = True
    if not keep.any():
        return []

    glyphs = np.where(keep[labels], 255, 0).astype(np.uint8)
    gap = max(3, int(BLOCK_GAP_FRACTION * side))
    merged = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap * 2 + 1, gap * 2 + 1)))
    m, mlabels, mstats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    # which kept glyph belongs to which merged block
    glyph_ids = np.where(keep)[0]
    block_of = {}
    for gid in glyph_ids:
        x, y, bw, bh, _a = stats[gid]
        block_of[gid] = int(mlabels[y + bh // 2, x + bw // 2]) or int(mlabels[y, x])

    members: dict = {}
    for gid, b in block_of.items():
        if b > 0:
            members.setdefault(b, []).append(gid)

    # Real outline points of every kept glyph blob (not bounding-box corners: on tilted text
    # a box corner sticks out beyond the page edge and would wrongly enlarge the crop).
    contours, _ = cv2.findContours(glyphs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_points: dict = {}
    for c in contours:
        x, y = int(c[0][0][0]), int(c[0][0][1])
        b = int(mlabels[y, x])
        if b > 0:
            hull_points.setdefault(b, []).append(cv2.convexHull(c).reshape(-1, 2))

    blocks: List[TextBlock] = []
    for b, ids in members.items():
        if len(ids) < MIN_COMPONENTS_PER_BLOCK or b not in hull_points:
            continue
        if np.median([stats[g][3] for g in ids]) < max(4.0, MIN_MEDIAN_GLYPH_HEIGHT * side):
            continue
        pts = np.vstack(hull_points[b]).astype(np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        blocks.append(TextBlock(pts, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), len(ids)))
    return blocks


# =============================================================================
# 2. Grow the detected outline until it holds the text
# =============================================================================
def _outward_normals(quad: np.ndarray) -> np.ndarray:
    """Unit outward normal of each side of a clockwise (on screen) TL,TR,BR,BL quad."""
    normals = []
    for i in range(4):
        d = quad[(i + 1) % 4] - quad[i]
        n = np.array([d[1], -d[0]], np.float32)
        normals.append(n / (np.linalg.norm(n) + 1e-9))
    return np.asarray(normals, np.float32)


def _signed_distances(points: np.ndarray, quad: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """(N, 4) distance of each point beyond each side (positive = outside that side)."""
    return np.stack([(points - quad[i]) @ normals[i] for i in range(4)], axis=1)


def _intersect(p1, d1, p2, d2) -> Optional[np.ndarray]:
    m = np.column_stack((d1, -d2))
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    t = np.linalg.solve(m, p2 - p1)
    return p1 + t[0] * d1


def _push_sides(quad: np.ndarray, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Move side i outwards by offsets[i] (>= 0) and re-intersect neighbouring sides."""
    lines = []
    for i in range(4):
        p = quad[i] + normals[i] * offsets[i]
        d = quad[(i + 1) % 4] - quad[i]
        lines.append((p, d / (np.linalg.norm(d) + 1e-9)))
    out = []
    for i in range(4):
        pt = _intersect(*lines[(i - 1) % 4], *lines[i])
        if pt is None:
            return quad
        out.append(pt)
    return np.asarray(out, np.float32)


def expand_quad_to_text(quad_working: np.ndarray, blocks: List[TextBlock],
                        image_shape: Tuple[int, ...]) -> Tuple[np.ndarray, int]:
    """Return ``(new_quad, n_blocks_added)`` in working coordinates.

    Text blocks that already sit inside the outline are left alone.  A block
    that sticks out of it (or lies just outside it) pulls the outline's sides
    outwards - only the sides that need it, only as far as needed.
    """
    h, w = image_shape[:2]
    side = max(h, w)
    prox = PROXIMITY_FRACTION * side
    pad = PAD_FRACTION * side
    quad = np.asarray(quad_working, np.float32).reshape(4, 2).copy()
    normals = _outward_normals(quad)

    included: set = set()
    for _ in range(6):                                   # growing can bring new blocks into reach
        changed = False
        offsets = np.zeros(4, np.float32)
        for idx, blk in enumerate(blocks):
            dist = _signed_distances(blk.points, quad, normals)       # (N, 4)
            outside_by = dist.max(axis=1)                              # >0: point is outside the quad
            if outside_by.max() <= -pad:                               # fully inside, with margin
                continue
            if idx not in included and outside_by.min() > prox:        # far away: not part of this page
                continue
            included.add(idx)
            offsets = np.maximum(offsets, dist.max(axis=0) + pad)
        offsets = np.maximum(offsets, 0.0)
        if offsets.max() > 0.5:
            quad = _push_sides(quad, normals, offsets)
            changed = True
        if not changed:
            break

    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad, len(included)


# =============================================================================
# 3. No outline found: box around the text
# =============================================================================
def text_region_quad(blocks: List[TextBlock], image_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Box around all text blocks (working coordinates), or ``None`` if no text.

    The box is returned as TL,TR,BR,BL.  When it would cover most of the photo
    the whole photo is returned (text spread over the page = full page).
    """
    if not blocks:
        return None
    h, w = image_shape[:2]
    side = max(h, w)
    pad = PAD_FRACTION * side * 2
    pts = np.vstack([b.points for b in blocks])

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)
    # Use the tilted box only when the text is clearly tilted, otherwise an upright box.
    tilt = abs(((angle + 45) % 90) - 45)
    if tilt > 4.0:
        box = cv2.boxPoints(((cx, cy), (rw + 2 * pad, rh + 2 * pad), angle)).astype(np.float32)
    else:
        x0, y0 = pts.min(axis=0) - pad
        x1, y1 = pts.max(axis=0) + pad
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)

    box[:, 0] = np.clip(box[:, 0], 0, w - 1)
    box[:, 1] = np.clip(box[:, 1], 0, h - 1)
    area = abs(cv2.contourArea(box.reshape(-1, 1, 2)))
    if area >= FULL_PAGE_FRACTION * w * h:
        m = 0.01
        return np.array([[m * w, m * h], [(1 - m) * w - 1, m * h],
                         [(1 - m) * w - 1, (1 - m) * h - 1], [m * w, (1 - m) * h - 1]], np.float32)
    return box
