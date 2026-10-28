"""OCR integration  (Member 4 - OCR & Application Engineer).

What OCR does
-------------
Optical Character Recognition turns the *picture* of text into *characters*.
We use the existing, pretrained **Tesseract** engine through ``pytesseract``
(a thin wrapper that runs the ``tesseract`` program).  Nothing is trained here.

Important facts for the viva
----------------------------
* ``pip install pytesseract`` does NOT install Tesseract itself.  The
  ``tesseract`` executable and the language data (``eng.traineddata``) must be
  installed separately (see README).
* Tesseract first finds text blocks/lines/words (layout analysis, controlled
  by the *page segmentation mode*, PSM) and then recognises characters.
  Image quality decides the result: skew, shadows, blur, and small text height
  (it works best when capital letters are roughly 20-30 px tall) all hurt.
  That is why this project corrects perspective and illumination first.
* The per-word *confidence* (0-100) is Tesseract's own estimate.  It is useful
  for spotting doubtful words, but a high value does not guarantee a correct
  word and a low value does not guarantee a wrong one.  Measure real accuracy
  with CER/WER (``evaluation/ocr_metrics.py``).

Testability
-----------
``run_ocr`` accepts a ``backend`` object with ``image_to_data`` (default:
``pytesseract``), so tests can inject a fake engine and run without Tesseract.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .contracts import STAGE_OCR, OcrResult, OcrWord, ScannerError

try:  # pytesseract may be missing in a broken install; report it instead of crashing
    import pytesseract
except ImportError:  # pragma: no cover - exercised only on broken installs
    pytesseract = None

WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)
VALID_PSM = tuple(range(0, 14))
MIN_LONG_SIDE_FOR_OCR = 1600   # upscale smaller pages so text is tall enough
MAX_UPSCALE = 3.0
OCR_PAD = 10                   # white border (Tesseract is more reliable with margins)
TARGET_TEXT_HEIGHT = 24.0
OSD_LONG_SIDE = 3200           # orientation detector needs many pixels per character


# --------------------------------------------------------------------------
# Locating / checking Tesseract
# --------------------------------------------------------------------------
def configure_tesseract(cmd: Optional[str] = None) -> Optional[str]:
    """Point pytesseract at the Tesseract executable and return the path used.

    Search order: explicit ``cmd`` argument -> ``TESSERACT_CMD`` environment
    variable -> ``PATH`` -> the usual Windows install folders.  Returns None if
    nothing was found (``run_ocr`` will then report "unavailable").
    """
    if pytesseract is None:
        return None
    candidates = [cmd, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"), *WINDOWS_TESSERACT_PATHS]
    for path in candidates:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


def check_tesseract(lang: str = "eng", backend: Any = None) -> Tuple[bool, str]:
    """Return (available, message) describing whether OCR can run with ``lang``."""
    backend = backend if backend is not None else pytesseract
    if backend is None:
        return False, "The 'pytesseract' Python package is not installed (pip install pytesseract)."
    try:
        version = backend.get_tesseract_version()
    except Exception:
        return False, ("Tesseract executable not found. Install Tesseract OCR and add it to PATH "
                       "or set the TESSERACT_CMD environment variable.")
    try:
        available = set(backend.get_languages(config=""))
    except Exception:
        return True, f"Tesseract {version} found (could not list languages)."
    missing = [code for code in lang.split("+") if code not in available]
    if missing:
        return False, (f"Tesseract {version} found, but language data missing: {', '.join(missing)}. "
                       "Install the .traineddata file(s) into Tesseract's tessdata folder.")
    return True, f"Tesseract {version} ready (languages: {lang})."


# --------------------------------------------------------------------------
# Preparing the image for Tesseract
# --------------------------------------------------------------------------
def estimate_text_height(image):
    """Median height (px) of the letter-sized blobs in ``image``; None if it cannot be measured.

    Never raises: any problem just means "unknown", and the caller keeps the original size.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        g = cv2.GaussianBlur(gray, (0, 0), 1.0)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        H, W = gray.shape[:2]
        hs = [st[i, 3] for i in range(1, n)
              if st[i, 4] >= 0.00002 * H * W and 0.01 * H < st[i, 3] < 0.9 * H and st[i, 2] < 0.5 * W]
        return float(np.median(hs)) if len(hs) >= 5 else None
    except Exception:
        return None


def prepare_for_ocr(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert to grey, rescale so letters are ~TARGET_TEXT_HEIGHT px tall, add a white margin.

    Returns (prepared_image, scale).  Pixel (x, y) of the prepared image maps
    back to ((x - OCR_PAD) / scale, (y - OCR_PAD) / scale) in the input.
    Large/blurry text is shrunk (INTER_AREA), tiny text is enlarged (INTER_CUBIC).
    """
    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ScannerError(STAGE_OCR, "bad_image", "OCR needs a non-empty uint8 image.")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ScannerError(STAGE_OCR, "bad_image", "OCR image must be 2-D (grey) or 3-channel (BGR).")
    text_h = estimate_text_height(image)
    scale = float(np.clip(TARGET_TEXT_HEIGHT / text_h, 0.2, 3.0)) if text_h else 1.0
    if abs(scale - 1.0) > 0.05:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)
    else:
        scale = 1.0
    image = cv2.copyMakeBorder(image, OCR_PAD, OCR_PAD, OCR_PAD, OCR_PAD, cv2.BORDER_CONSTANT, value=255)
    return image, scale


# --------------------------------------------------------------------------
# Running OCR
# --------------------------------------------------------------------------
def _classify_exception(exc: Exception) -> OcrResult:
    """Map any backend exception to an OcrResult with a helpful message."""
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()
    not_found = (pytesseract is not None and isinstance(exc, pytesseract.TesseractNotFoundError)) \
        or isinstance(exc, (FileNotFoundError, PermissionError)) or name == "TesseractNotFoundError"
    if not_found:
        return OcrResult(status="unavailable", message=(
            "Tesseract executable not found. Install Tesseract OCR and add it to PATH, "
            "or set the TESSERACT_CMD environment variable. Your scanned image is still available."))
    if any(s in lowered for s in ("failed loading language", "error opening data file", "traineddata", "tessdata_prefix")):
        return OcrResult(status="unavailable", message=(
            "Tesseract language data is missing (for example eng.traineddata). Install the language "
            "data or choose another language. Your scanned image is still available."))
    return OcrResult(status="error", message=f"Tesseract failed: {text[:300]}")


def run_ocr(image: np.ndarray, lang: str = "eng", psm: int = 3, backend: Any = None) -> OcrResult:
    """Recognise text in ``image`` (grey/B&W uint8 2-D, or BGR) and return an ``OcrResult``.

    psm 3  fully automatic page segmentation (default, multi-column pages)
    psm 4  single column of variable-size text      psm 6  one uniform block of text
    psm 11 sparse text (receipts, scattered labels)

    Status values
      ok           text found
      no_text      OCR ran but found nothing (blank / unreadable page)
      unavailable  Tesseract or the language data is not installed
      error        any other Tesseract failure
    Never raises for engine problems (the scan must still be delivered);
    raises ``ScannerError`` only for invalid arguments.
    """
    if psm not in VALID_PSM:
        raise ScannerError(STAGE_OCR, "bad_psm", f"Page segmentation mode must be 0-13, got {psm}.")
    prepared, scale = prepare_for_ocr(image)
    engine = backend if backend is not None else pytesseract
    if engine is None:
        return OcrResult(status="unavailable", language=lang, psm=psm,
                         message="The 'pytesseract' package is not installed (pip install pytesseract).")
    try:
        data = engine.image_to_data(prepared, lang=lang, config=f"--oem 1 --psm {psm} --dpi 300 -c preserve_interword_spaces=1", output_type="dict")
    except Exception as exc:  # any engine failure must degrade gracefully
        result = _classify_exception(exc)
        result.language, result.psm = lang, psm
        return result

    words: List[OcrWord] = []
    lines: dict = {}
    order: List[tuple] = []
    count = len(data.get("text", []))
    for i in range(count):
        text = str(data["text"][i]).strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 0:       # conf -1 rows describe page/block/line structure, not words
            continue
        left = max(0, int(round((int(data["left"][i]) - OCR_PAD) / scale)))
        top = max(0, int(round((int(data["top"][i]) - OCR_PAD) / scale)))
        width = int(round(int(data["width"][i]) / scale))
        height = int(round(int(data["height"][i]) / scale))
        words.append(OcrWord(text, conf, left, top, width, height))
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(text)

    if not words:
        return OcrResult(status="no_text", language=lang, psm=psm,
                         message="No text was recognised. The page may be blank, too blurry, or too small.")

    # Rebuild text: words -> lines; a blank line separates paragraphs/blocks.
    out_lines: List[str] = []
    previous = None
    for key in order:
        if previous is not None and key[:2] != previous[:2]:
            out_lines.append("")
        out_lines.append(" ".join(lines[key]))
        previous = key
    mean_conf = float(np.mean([w.conf for w in words]))
    return OcrResult(status="ok", text="\n".join(out_lines), words=words, mean_confidence=mean_conf,
                     language=lang, psm=psm)


def prepare_for_osd(image: np.ndarray) -> np.ndarray:
    """Preparation for Tesseract's orientation detector.

    Text-height scaling cannot be used here: the height estimate is only meaningful when the
    text is upright, and OSD is exactly the step that finds out whether it is.  OSD needs
    enough pixels per character, so small pages are simply enlarged (long side up to 3200, max 4x).
    A deskewed page has a larger, padded canvas, so a smaller target leaves the letters too small.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    longest = max(image.shape[:2])
    if longest < OSD_LONG_SIDE:
        f = min(4.0, OSD_LONG_SIDE / float(longest))
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(image, OCR_PAD, OCR_PAD, OCR_PAD, OCR_PAD, cv2.BORDER_CONSTANT, value=255)


def detect_orientation(image: np.ndarray, backend: Any = None, min_confidence: float = 2.0) -> Optional[Tuple[int, float]]:
    """Ask Tesseract's orientation detector how far to rotate the page clockwise.

    Returns (degrees_clockwise in {0, 90, 180, 270}, confidence) or None when
    it cannot decide (too little text, missing ``osd`` data, Tesseract absent).
    """
    engine = backend if backend is not None else pytesseract
    if engine is None:
        return None
    try:
        prepared = prepare_for_osd(image)
        info = engine.image_to_osd(prepared, config="--psm 0", output_type="dict")
        rotate = int(info.get("rotate", 0))
        conf = float(info.get("orientation_conf", 0.0))
    except Exception:
        return None
    if conf < min_confidence or rotate not in (0, 90, 180, 270):
        return None
    return rotate, conf


def draw_word_boxes(image: np.ndarray, words: List[OcrWord], low_threshold: float = 60.0) -> np.ndarray:
    """BGR copy of ``image`` with green boxes (confident) and red boxes (doubtful words).

    ``image`` must be the image that was given to ``run_ocr`` (same coordinates).
    """
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for w in words:
        color = (0, 160, 0) if w.conf >= low_threshold else (0, 0, 255)
        cv2.rectangle(out, (w.left, w.top), (w.left + w.width, w.top + w.height), color, 2)
    return out