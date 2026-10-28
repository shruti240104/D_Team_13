from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .contracts import (
    STAGE_DETECTION, STAGE_ENHANCEMENT, STAGE_INPUT, STAGE_PIPELINE, DetectionResult,
    EnhancementResult, LoadedImage, OcrResult, ScannerError, StageError, WarpResult,
)
from .deskew import deskew
from .detection import detect_document, draw_corners, order_corners
from .enhancement import EnhancementOptions, enhance
from .geometry import rotate_quarter_turns, warp_perspective
from .input_handler import MAX_FILE_BYTES, load_image, make_working_copy
from .ocr import detect_orientation, run_ocr
from .text_region import expand_quad_to_text, find_text_blocks, text_region_quad
from .orientation_search import read_quality, search_orientation
from .deskew import rotate_keep_all

logger = logging.getLogger(__name__)
OCR_SOURCES = ("grayscale", "bw", "color")


@dataclass
class PipelineOptions:
    enhancement: EnhancementOptions = field(default_factory=EnhancementOptions)
    detection_max_side: int = 800
    run_ocr: bool = True
    ocr_lang: str = "eng"
    ocr_psm: int = 3
    ocr_source: str = "grayscale"     # which enhanced variant is given to Tesseract
    auto_orient: bool = False         # fix 90/180/270 degree rotation using Tesseract OSD
    auto_deskew: bool = True          # straighten tilted text lines (+-30 degrees)
    deskew_max_angle: float = 45.0
    search_orientation: bool = True   # last resort: try other text directions when the first read is poor
    search_below_conf: float = 70.0   # ... i.e. when mean word confidence is below this
    max_file_bytes: int = MAX_FILE_BYTES


@dataclass
class PipelineResult:
    error: Optional[StageError] = None
    loaded: Optional[LoadedImage] = None
    original: Optional[np.ndarray] = None            # BGR, full working resolution
    detection: Optional[DetectionResult] = None
    corners: Optional[np.ndarray] = None             # corners actually used (original pixels)
    corner_source: str = "auto"                      # "auto" or "manual"
    outline_preview: Optional[np.ndarray] = None     # BGR, original with outline (<=1600 px)
    warp: Optional[WarpResult] = None
    enhancement: Optional[EnhancementResult] = None
    ocr: Optional[OcrResult] = None
    ocr_source: Optional[str] = None
    rotation_applied: int = 0
    deskew_angle: float = 0.0
    search_angle: float = 0.0                        # direction found by the last-resort search (0 = not used)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def warped(self) -> Optional[np.ndarray]:
        return self.warp.image if self.warp is not None else None


@contextmanager
def _timed(result: PipelineResult, name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        result.timings[name] = round(time.perf_counter() - start, 4)

def create_virtual_border(image: np.ndarray, margin_ratio: float = 0.01):
    """
    Creates virtual document corners when automatic
    document boundary detection fails.

    Returns corners in:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left order.
    """

    h, w = image.shape[:2]

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    corners = np.array([
        [margin_x, margin_y],
        [w - 1 - margin_x, margin_y],
        [w - 1 - margin_x, h - 1 - margin_y],
        [margin_x, h - 1 - margin_y]
    ], dtype=np.float32)

    return corners

def make_outline_preview(original: np.ndarray, corners: Optional[np.ndarray], max_side: int = 1600) -> np.ndarray:
    """Original image (down-scaled for display) with the outline drawn on it."""
    h, w = original.shape[:2]
    f = min(1.0, max_side / float(max(h, w)))
    small = cv2.resize(original, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1.0 else original.copy()
    if corners is None:
        return small
    return draw_corners(small, np.asarray(corners, np.float32) * f)

SURE_CONF = 60          # a word at or above this counts as "really read"
COMPLETE_FRACTION = 0.85  # a candidate must have read >= 85 % as much text as the most complete one


def _sure_chars(res) -> int:
    """Characters in words Tesseract is fairly sure about.  Junk tokens have low confidence, so they
    do not inflate this number (unlike a plain word count or a sum of confidences)."""
    return sum(len(w.text) for w in res.words if w.conf >= SURE_CONF)


def _pick_best(candidates):
    """candidates: list of (OcrResult, source).  Two-step choice:
    1. keep only candidates that read (almost) as much text as the most complete one - this rejects
       fragments (e.g. sparse-text mode returning a few words at 95 % confidence);
    2. among those, take the highest mean confidence - this rejects results padded with junk words."""
    top = max(_sure_chars(r) for r, _ in candidates)
    complete = [c for c in candidates if _sure_chars(c[0]) >= COMPLETE_FRACTION * top]
    return max(complete, key=lambda c: c[0].mean_confidence or 0.0)


def run_best_ocr(enhancement, lang="eng", source="grayscale", psm=3, backend=None):
    """Run the user's chosen source/PSM plus a few fallbacks and keep the best read (see _pick_best)."""
    attempts = [(source, psm)]
    for extra in (("grayscale", 6), ("grayscale", 4), ("color", psm)):
        if extra not in attempts:
            attempts.append(extra)

    good, first_failure = [], None             # keep the real error message (e.g. Tesseract missing)
    for src, p in attempts:
        image = enhancement.variants.get(src)
        if image is None:
            continue
        try:
            res = run_ocr(image, lang=lang, psm=p, backend=backend)
        except ScannerError as exc:            # e.g. invalid PSM: report it, do not hide it
            return OcrResult(status="error", message=exc.message, language=lang, psm=p), src
        except Exception:
            logger.exception("OCR attempt failed: source=%s psm=%s", src, p)
            continue
        if res.status == "ok":
            good.append((res, src))
        else:
            first_failure = first_failure or res
            if res.status in ("unavailable", "error"):
                break                          # retrying will not help
    if good:
        return _pick_best(good)
    return first_failure, source


def _scan_from_corners(result: PipelineResult, options: PipelineOptions, ocr_backend: Any) -> None:
    """Stages after detection: warp -> orientation -> enhance -> OCR (mutates ``result``)."""
    result.outline_preview = make_outline_preview(result.original, result.corners)

    with _timed(result, "warp"):
        result.warp = warp_perspective(result.original, result.corners)
    result.warnings.extend(result.warp.warnings)

    if options.auto_deskew:
        with _timed(result, "deskew"):
            straight, angle = deskew(result.warp.image, options.deskew_max_angle)
        if angle != 0.0:
            result.warp.image = straight
            result.warp.output_size = (straight.shape[1], straight.shape[0])
            result.deskew_angle = angle
            result.warnings.append(f"Tilted text was straightened by {abs(angle):.1f} degrees.")

    if options.auto_orient:
        with _timed(result, "orientation"):
            found = detect_orientation(result.warp.image, backend=ocr_backend)
        if found is None:
            result.warnings.append("Page orientation could not be determined automatically.")
        elif found[0] != 0:
            result.warp.image = rotate_quarter_turns(result.warp.image, found[0])
            result.warp.output_size = (result.warp.image.shape[1], result.warp.image.shape[0])
            result.rotation_applied = found[0]
            result.warnings.append(f"Page was rotated {found[0]} degrees clockwise (orientation detection).")

    with _timed(result, "enhance"):
        result.enhancement = enhance(result.warp.image, options.enhancement)
    result.warnings.extend(result.enhancement.warnings)

    if options.run_ocr:

        with _timed(result, "ocr"):

            best_ocr, best_source = run_best_ocr(
                result.enhancement, lang=options.ocr_lang,
                source=options.ocr_source, psm=options.ocr_psm, backend=ocr_backend)
            if best_ocr is not None:
                result.ocr, result.ocr_source = best_ocr, best_source
                if best_ocr.status in ("unavailable", "error"):
                    result.warnings.append(f"OCR could not run: {best_ocr.message}")
                elif best_ocr.status != "ok":
                    result.warnings.append(best_ocr.message or "OCR could not extract readable text.")
            else:
                result.ocr = OcrResult(status="no_text", message="No readable text was detected.")

        if options.search_orientation and options.auto_deskew:
            ocr = result.ocr
            if ocr is None or (ocr.status == "ok" and (ocr.mean_confidence or 0.0) < options.search_below_conf) \
                    or ocr.status == "no_text":
                with _timed(result, "orientation_search"):
                    _orientation_search(result, options, ocr_backend)

def _orientation_search(result: PipelineResult, options: PipelineOptions, ocr_backend: Any) -> None:
    """Last resort for a poor read: try the plausible text directions and keep the best one.

    The change is adopted only if the new read is clearly better; otherwise nothing changes.
    """
    base_q = read_quality(result.ocr)

    def read(img):
        return run_ocr(img, lang=options.ocr_lang, psm=6, backend=ocr_backend)

    best = None                                    # (quality, angle, source name, source image)
    for name, source in (("page", result.warp.image), ("photo", result.original)):
        angle, q, _ = search_orientation(source, read)
        if best is None or q > best[0]:
            best = (q, angle, name, source)
        if best[0] >= max(60, 2 * base_q):          # clearly good already: skip the second source
            break
    q, angle, name, source = best
    if q < max(1.5 * base_q, base_q + 15) or (angle == 0.0 and name == "page"):
        return                                     # no clear improvement: keep the normal result

    candidate = rotate_keep_all(source, angle) if angle != 0.0 else source
    candidate, _ = deskew(candidate, 10.0)         # remove the small residual tilt of the coarse angle
    old = (result.warp.image, result.warp.output_size, result.enhancement, result.ocr, result.ocr_source)
    result.warp.image = candidate
    result.warp.output_size = (candidate.shape[1], candidate.shape[0])
    result.enhancement = enhance(candidate, options.enhancement)
    new_ocr, new_src = run_best_ocr(result.enhancement, lang=options.ocr_lang, source=options.ocr_source,
                                    psm=options.ocr_psm, backend=ocr_backend)
    if new_ocr is None or read_quality(new_ocr) <= base_q:    # the full read did not confirm it: undo
        result.warp.image, result.warp.output_size, result.enhancement, result.ocr, result.ocr_source = old
        return
    result.ocr, result.ocr_source = new_ocr, new_src
    result.search_angle = angle
    if angle == 0.0:                               # same direction, but the whole photo reads better than the page
        result.warnings.append("First read was poor and the detected page outline looked wrong, "
                               "so the whole photo was read instead.")
    else:
        where = "page" if name == "page" else "whole photo"
        result.warnings.append(f"First read was poor, so other text directions were tried: "
                               f"the {where} was rotated by {angle:.0f} degrees.")


def _finish_with_error(result: PipelineResult, exc: Exception) -> PipelineResult:
    if isinstance(exc, ScannerError):
        result.error = exc.to_stage_error()
    else:
        logger.exception("Unexpected pipeline failure")
        result.error = StageError(STAGE_PIPELINE, "unexpected", f"Unexpected error: {exc}",
                                  "Try another image; if it keeps happening, report the problem to the team.")
    return result


def run_pipeline(data: bytes, filename: Optional[str] = None, options: Optional[PipelineOptions] = None,
                 ocr_backend: Any = None) -> PipelineResult:
    """Full automatic scan of an uploaded file.  Never raises; check ``result.ok``."""
    options = options or PipelineOptions()
    result = PipelineResult()
    try:
        with _timed(result, "load"):
            result.loaded = load_image(data, filename, max_bytes=options.max_file_bytes)
        result.original = result.loaded.image
        result.warnings.extend(result.loaded.warnings)

        with _timed(result, "detect"):
            working = make_working_copy(result.original, options.detection_max_side)
            h, w = result.original.shape[:2]
            result.detection = detect_document(working.image, (working.scale_x, working.scale_y), (w, h))
        result.warnings.extend(result.detection.warnings)

        scale = np.array([working.scale_x, working.scale_y], dtype=np.float32)
        blocks = find_text_blocks(working.image)

        if not result.detection.success:
            # No page outline found.  Crop around the text instead (text spread over the
            # whole photo gives the whole photo); with no text at all use the whole photo.
            box = text_region_quad(blocks, working.image.shape)
            if box is not None:
                result.corners = order_corners(box * scale)
                result.corner_source = "text"
                result.warnings.append(
                    "Page outline not detected, so the crop was fitted around the text. "
                    "Adjust the corners manually if you want something different.")
            else:
                result.corners = create_virtual_border(result.original, margin_ratio=0.01)
                result.corner_source = "virtual"
                result.warnings.append(
                    "Document boundary not detected. "
                    "Virtual border has been created using image boundaries.")
        else:
            # Page outline found: make sure it does not cut any text off.
            quad_w = result.detection.corners / scale
            grown, added = expand_quad_to_text(quad_w, blocks, working.image.shape)
                        # Safety: never crop tighter than the text itself.
            box = text_region_quad(blocks, working.image.shape)
            if box is not None:
                union = np.vstack([grown, box])
                x0, y0 = union.min(axis=0)
                x1, y1 = union.max(axis=0)
                hh, ww = working.image.shape[:2]
                rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
                rect[:, 0] = np.clip(rect[:, 0], 0, ww - 1)
                rect[:, 1] = np.clip(rect[:, 1], 0, hh - 1)
                # only use it if text box actually sticks out of the quad
                if cv2.contourArea(rect.reshape(-1, 1, 2)) > cv2.contourArea(grown.reshape(-1, 1, 2)) * 1.02:
                    grown = rect
            result.corners = (grown * scale).astype(np.float32)
            result.corner_source = "auto"
            if added and not np.allclose(grown, quad_w, atol=1.0):
                result.warnings.append(
                    "The detected page outline cut through some text, so the crop was enlarged to keep all text.")

        # Continue processing regardless of detection failure.
        _scan_from_corners(result, options, ocr_backend)
    except Exception as exc:  # noqa: BLE001 - converted to a structured error
        return _finish_with_error(result, exc)
    return result


def rescan_with_corners(previous: PipelineResult, corners, options: Optional[PipelineOptions] = None,
                        ocr_backend: Any = None) -> PipelineResult:
    """Re-run warp -> enhance -> OCR with user-chosen corners (any order is accepted).

    Starts from ``previous.original`` so the file is not decoded again.
    """
    options = options or PipelineOptions()
    result = PipelineResult(loaded=previous.loaded, original=previous.original,
                            detection=previous.detection, corner_source="manual")
    if previous.original is None:
        result.error = StageError(STAGE_INPUT, "no_image", "There is no loaded image to re-scan.")
        return result
    try:
        # order_corners makes the UI's four free points a clockwise TL,TR,BR,BL quad
        result.corners = order_corners(corners)
        result.warnings.append("Corners were set manually.")
        _scan_from_corners(result, options, ocr_backend)
    except Exception as exc:  # noqa: BLE001
        if result.outline_preview is None:
            result.outline_preview = make_outline_preview(result.original, None)
        return _finish_with_error(result, exc)
    return result