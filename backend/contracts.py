"""Shared data contracts (result types and the error type) for all modules.

This file is the "agreement" between the four team members.  Changing a field
here affects several modules, so changes need an issue and a review from every
owner (see docs/GIT_WORKFLOW.md).

Design rule: modules signal *expected* problems (bad file, no document found,
Tesseract missing) with a ``ScannerError`` carrying a stage, a machine-readable
code, a human message and a hint.  The pipeline converts it into a
``StageError`` so the UI never has to handle raw exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Stage names used in errors, timings and the UI.
STAGE_INPUT = "input"
STAGE_DETECTION = "detection"
STAGE_GEOMETRY = "geometry"
STAGE_ENHANCEMENT = "enhancement"
STAGE_OCR = "ocr"
STAGE_EXPORT = "export"
STAGE_PIPELINE = "pipeline"


@dataclass(frozen=True)
class StageError:
    """Structured, UI-friendly description of a failure."""

    stage: str
    code: str
    message: str
    hint: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.stage}/{self.code}] {self.message}"


class ScannerError(Exception):
    """Expected, explainable failure raised by a backend module."""

    def __init__(self, stage: str, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message
        self.hint = hint

    def to_stage_error(self) -> StageError:
        return StageError(self.stage, self.code, self.message, self.hint)


# --------------------------------------------------------------------------
# Member 2: input / geometry contracts
# --------------------------------------------------------------------------
@dataclass
class LoadedImage:
    """Output of ``input_handler.load_image``."""

    image: np.ndarray            # BGR uint8, EXIF-corrected, possibly down-scaled
    format: str                  # "JPEG" or "PNG"
    file_size: int               # bytes of the uploaded file
    original_size: Tuple[int, int]  # (width, height) as stored in the file (before EXIF/down-scale)
    exif_rotated: bool = False
    downscaled: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class WorkingCopy:
    """Down-scaled copy used for detection plus the scale needed to map back."""

    image: np.ndarray            # BGR uint8
    scale_x: float               # original_width  / working_width
    scale_y: float               # original_height / working_height


@dataclass
class WarpResult:
    """Output of ``geometry.warp_perspective``."""

    image: np.ndarray            # BGR uint8, the flattened page
    matrix: np.ndarray           # 3x3 float64 homography (source -> output)
    output_size: Tuple[int, int]  # (width, height)
    source_corners: np.ndarray   # (4, 2) float32, ordered TL,TR,BR,BL
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Member 1: detection contract
# --------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """Output of ``detection.detect_document``.

    ``quality`` is a heuristic ranking score in [0, 1] (see detection.py for the
    exact formula).  It is NOT a calibrated probability of being correct.
    """

    success: bool
    corners: Optional[np.ndarray] = None   # (4,2) float32, ORIGINAL pixel coords, TL,TR,BR,BL
    quality: float = 0.0
    method: str = ""
    failure_code: str = ""
    failure_reason: str = ""
    agreeing_methods: int = 0              # other strategies that found (almost) the same quad
    ambiguous: bool = False                # another, different, similarly good candidate exists
    candidates_considered: int = 0
    warnings: List[str] = field(default_factory=list)
    debug: Dict[str, np.ndarray] = field(default_factory=dict)  # working-scale debug images


# --------------------------------------------------------------------------
# Member 3: enhancement contract
# --------------------------------------------------------------------------
@dataclass
class QualityReport:
    """Simple, explainable image-quality indicators for the flattened page."""

    sharpness: float             # variance of Laplacian (resolution-normalised)
    ink_contrast: float          # mean(paper) - mean(ink) from an Otsu split, 0..255
    resolution: Tuple[int, int]  # (width, height) of the flattened page
    blur_suspected: bool = False
    low_contrast: bool = False
    low_resolution: bool = False


@dataclass
class EnhancementResult:
    """Named image variants.

    variants["color"]     BGR uint8   (illumination-normalised, white-balanced)
    variants["grayscale"] uint8 (H,W) (normalised, denoised, contrast-adjusted)
    variants["bw"]        uint8 (H,W) only 0 / 255 (adaptive threshold)
    """

    variants: Dict[str, np.ndarray]
    quality: QualityReport
    applied: List[str] = field(default_factory=list)   # which filters really ran
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Member 4: OCR contract
# --------------------------------------------------------------------------
@dataclass
class OcrWord:
    """One recognised word; box is in the coordinates of the image given to OCR."""

    text: str
    conf: float                  # Tesseract confidence 0..100 (NOT a guarantee)
    left: int
    top: int
    width: int
    height: int


@dataclass
class OcrResult:
    """status: 'ok' | 'no_text' | 'unavailable' | 'error'."""

    status: str
    text: str = ""
    words: List[OcrWord] = field(default_factory=list)
    mean_confidence: Optional[float] = None
    message: str = ""
    language: str = "eng"
    psm: int = 3

    def low_confidence_words(self, threshold: float = 60.0) -> List[OcrWord]:
        return [w for w in self.words if w.conf < threshold]
