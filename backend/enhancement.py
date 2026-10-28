"""Image enhancement  (Member 3 - Image Enhancement Engineer).

Input : the flattened page (BGR uint8) from ``geometry.warp_perspective``.
Output: ``EnhancementResult`` with three named variants
          "color"     BGR uint8, paper whitened, colours kept
          "grayscale" uint8 (H, W), the best general-purpose OCR input
          "bw"        uint8 (H, W), only 0 / 255 (adaptive threshold)
        plus a ``QualityReport`` (blur / low contrast / low resolution).

Filters are applied deliberately, not blindly:

1. Illumination normalisation (shadow removal).  A phone photo has a smooth
   brightness gradient (lamp, shadow of the phone).  We estimate the *paper
   background* with a large morphological closing (which erases thin dark
   text but keeps the smooth paper brightness), then divide the image by it.
   Paper becomes ~255 everywhere and ink keeps its relative darkness.
   Limits: very thick dark regions (big headings, photographs) can be treated
   as background and fade; a hard-edged shadow leaves a faint edge.
2. Denoising.  Removes sensor noise that would become speckles after
   thresholding.  Non-local means is good but slow, so large pages use a
   cheaper bilateral filter.
3. CLAHE (Contrast Limited Adaptive Histogram Equalisation).  Equalises
   contrast in small tiles instead of globally, so faint text in a dim region
   is boosted.  RISK: in flat regions it amplifies noise, so in "auto" mode it
   only runs when text/paper separation is weak, and always after denoising.
4. Adaptive thresholding for the "bw" variant.  A global threshold (one value
   for the whole image) fails under uneven light: shadowed paper turns black.
   An adaptive threshold decides per pixel by comparing it with the Gaussian
   weighted mean of its neighbourhood (``block_size`` x ``block_size``) minus a
   constant C.  RISK: block too small -> hollow thick strokes; large solid
   areas become noisy; faint text can vanish.  Thresholding discards
   information, which is why the colour and grey variants are always kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .contracts import STAGE_ENHANCEMENT, EnhancementResult, QualityReport, ScannerError

BLUR_SHARPNESS_THRESHOLD = 80.0     # heuristic: Laplacian variance below this = suspected blur
LOW_CONTRAST_THRESHOLD = 90.0       # heuristic: paper-ink separation below this
LOW_RESOLUTION_LONG_SIDE = 1000     # px of the flattened page
CLAHE_AUTO_SEPARATION = 120.0       # "auto" CLAHE runs only below this separation
NLM_MAX_PIXELS = 2_500_000          # non-local means only for pages up to ~2.5 MP


@dataclass
class EnhancementOptions:
    """User-adjustable enhancement settings.

    shadow_removal  illumination normalisation on/off
    denoise         denoise on/off
    clahe           "auto" (only for weak contrast), "on" or "off"
    block_size      adaptive-threshold neighbourhood (odd); None = automatic
    threshold_c     constant subtracted from the local mean (larger = cleaner background)
    """

    shadow_removal: bool = True
    denoise: bool = True
    clahe: str = "auto"
    block_size: Optional[int] = None
    threshold_c: int = 10
    clahe_clip: float = 2.0


def _fail(code: str, message: str) -> ScannerError:
    return ScannerError(STAGE_ENHANCEMENT, code, message)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """BGR -> grey using OpenCV's luminance weights."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# Building blocks (public so they can be unit-tested and compared)
# --------------------------------------------------------------------------
def estimate_background(gray: np.ndarray, work_side: int = 600) -> np.ndarray:
    """Estimate the smooth paper-brightness surface of a grey page.

    Works on a copy shrunk to ``work_side`` px (the background is smooth, so
    nothing is lost, and a large kernel on a 12 MP image would take seconds).
    """
    h, w = gray.shape
    s = min(1.0, work_side / float(max(h, w)))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else gray
    k = _odd(max(9, round(0.04 * max(small.shape))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # closing = dilate then erode: removes dark features thinner than k (the text)
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), k / 2.0)
    return cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by the estimated background so the paper becomes uniformly white."""
    background = estimate_background(gray)
    return cv2.divide(gray, background, scale=255)


def denoise_gray(gray: np.ndarray) -> np.ndarray:
    """Non-local means for normal pages, bilateral filter for very large ones."""
    if gray.size <= NLM_MAX_PIXELS:
        return cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)
    return cv2.bilateralFilter(gray, 5, 35, 5)


def apply_clahe(gray: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Tile-wise contrast equalisation with a clip limit that caps noise amplification."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def text_paper_separation(gray: np.ndarray) -> float:
    """Mean(paper) - mean(ink) from an Otsu split (0 if there is no ink/paper split).

    A clean printed page gives ~150-200; a washed-out one gives much less.  It is
    used for the low-contrast warning and the "auto" CLAHE decision.
    """
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink, paper = gray[gray <= t], gray[gray > t]
    if ink.size == 0 or paper.size == 0:
        return 0.0
    return float(paper.mean() - ink.mean())


def sharpness_score(gray: np.ndarray, norm_side: int = 1200) -> float:
    """Variance of the Laplacian after down-scaling to at most ``norm_side`` px.

    The Laplacian responds to rapid intensity change; blurry text has weak
    edges, so low variance suggests blur.  Down-scaling first makes the value
    comparable between a 1200 px and a 4000 px page.  It also depends on how
    much text there is, so use it as a hint, not a proof.
    """
    longest = max(gray.shape)
    if longest > norm_side:
        f = norm_side / float(longest)
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def background_uniformity(gray: np.ndarray) -> float:
    """Standard deviation of the estimated paper background (lower = more even light)."""
    return float(estimate_background(gray).std())


def adaptive_threshold_bw(gray: np.ndarray, block_size: Optional[int] = None, c: int = 10) -> np.ndarray:
    """Black-and-white image where each pixel is compared with its local mean."""
    if block_size is None:
        block_size = _odd(max(15, min(gray.shape) // 25))
    block_size = _odd(max(3, block_size))
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)


def assess_quality(gray: np.ndarray) -> QualityReport:
    """Blur / contrast / resolution indicators for the flattened page."""
    h, w = gray.shape
    sharp = sharpness_score(gray)
    sep = text_paper_separation(gray)
    return QualityReport(
        sharpness=sharp, ink_contrast=sep, resolution=(w, h),
        blur_suspected=sharp < BLUR_SHARPNESS_THRESHOLD,
        low_contrast=sep < LOW_CONTRAST_THRESHOLD,
        low_resolution=max(w, h) < LOW_RESOLUTION_LONG_SIDE,
    )


def _white_balance_lab(lab: np.ndarray) -> np.ndarray:
    """Shift the a/b channels so the paper (brightest 40 % of pixels) is neutral grey."""
    l_ch, a_ch, b_ch = cv2.split(lab)
    mask = l_ch >= np.percentile(l_ch, 60)
    if mask.sum() < 100:
        return lab
    shift_a = float(np.median(a_ch[mask])) - 128.0
    shift_b = float(np.median(b_ch[mask])) - 128.0
    a2 = np.clip(a_ch.astype(np.float32) - shift_a, 0, 255).astype(np.uint8)
    b2 = np.clip(b_ch.astype(np.float32) - shift_b, 0, 255).astype(np.uint8)
    return cv2.merge([l_ch, a2, b2])


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def enhance(page_bgr: np.ndarray, options: Optional[EnhancementOptions] = None) -> EnhancementResult:
    """Create the colour, grayscale and black-and-white variants of a flattened page."""
    if not isinstance(page_bgr, np.ndarray) or page_bgr.ndim != 3 or page_bgr.shape[2] != 3 \
            or page_bgr.dtype != np.uint8 or page_bgr.size == 0:
        raise _fail("bad_image", "Enhancement needs a non-empty BGR uint8 image.")
    opts = options or EnhancementOptions()
    if opts.clahe not in ("auto", "on", "off"):
        raise _fail("bad_option", "clahe must be 'auto', 'on' or 'off'.")
    applied: List[str] = []
    warnings: List[str] = []

    gray_raw = to_grayscale(page_bgr)
    quality = assess_quality(gray_raw)
    if quality.blur_suspected:
        warnings.append(f"The page looks blurry (sharpness {quality.sharpness:.0f}); OCR may be inaccurate. "
                        "Retake the photo with steadier hands or better focus.")
    if quality.low_contrast:
        warnings.append("Text/paper contrast is low; faint text may be lost in black-and-white mode.")
    if quality.low_resolution:
        warnings.append(f"Low resolution ({quality.resolution[0]}x{quality.resolution[1]}); small text may be unreadable.")

    # ---- grayscale pipeline --------------------------------------------------
    gray = gray_raw
    if opts.shadow_removal:
        gray = normalize_illumination(gray)
        applied.append("illumination_normalization")
    if opts.denoise:
        gray = denoise_gray(gray)
        applied.append("denoise")
    use_clahe = opts.clahe == "on" or (opts.clahe == "auto" and text_paper_separation(gray) < CLAHE_AUTO_SEPARATION)
    if use_clahe:
        gray = apply_clahe(gray, opts.clahe_clip)
        applied.append("clahe")

    # ---- black and white -----------------------------------------------------
    bw = adaptive_threshold_bw(gray, opts.block_size, opts.threshold_c)
    applied.append("adaptive_threshold")

    # ---- colour (work on lightness only so colours are not distorted) ---------
    lab = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    if opts.shadow_removal:
        l_ch = normalize_illumination(l_ch)
    if opts.denoise:
        l_ch = denoise_gray(l_ch)
    if use_clahe:
        l_ch = apply_clahe(l_ch, opts.clahe_clip)
    lab[:, :, 0] = l_ch
    color = cv2.cvtColor(_white_balance_lab(lab), cv2.COLOR_LAB2BGR)

    return EnhancementResult(variants={"color": color, "grayscale": gray, "bw": bw},
                             quality=quality, applied=applied, warnings=warnings)
