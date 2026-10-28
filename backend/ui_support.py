"""Pure helper functions for the Streamlit UI  (Member 4).

Everything that can be tested without a browser lives here, so ``app.py`` stays
a thin layer that only draws widgets.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .contracts import OcrResult
from .detection import order_corners
from .export import encode_png, images_to_pdf, text_to_utf8_bytes
from .pipeline import PipelineResult

VARIANT_LABELS: Dict[str, str] = {
    "color": "Color (enhanced)",
    "grayscale": "Grayscale",
    "bw": "Black & white (adaptive threshold)",
}
CORNER_NAMES = ("Top-left", "Top-right", "Bottom-right", "Bottom-left")


def corners_to_percent(corners: np.ndarray, image_shape: Sequence[int]) -> List[float]:
    """[x1, y1, ..., x4, y4] as percentages (0-100) of image width/height."""
    h, w = image_shape[:2]
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return [float(v) for p in pts for v in (100.0 * p[0] / max(w - 1, 1), 100.0 * p[1] / max(h - 1, 1))]


def percent_to_corners(values: Sequence[float], image_shape: Sequence[int]) -> np.ndarray:
    """Inverse of ``corners_to_percent``; the result is ordered TL,TR,BR,BL."""
    h, w = image_shape[:2]
    pts = np.array(values, dtype=np.float64).reshape(4, 2)
    pts[:, 0] *= (w - 1) / 100.0
    pts[:, 1] *= (h - 1) / 100.0
    return order_corners(pts)


def default_manual_corners(image_shape: Sequence[int], inset: float = 0.10) -> np.ndarray:
    """A centred rectangle used as the starting point for manual placement."""
    h, w = image_shape[:2]
    x0, x1 = inset * (w - 1), (1 - inset) * (w - 1)
    y0, y1 = inset * (h - 1), (1 - inset) * (h - 1)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def get_variant(result: PipelineResult, key: str) -> Optional[np.ndarray]:
    if result.enhancement is None:
        return None
    return result.enhancement.variants.get(key)


def variant_png_bytes(result: PipelineResult, key: str) -> bytes:
    img = get_variant(result, key)
    if img is None:
        raise KeyError(f"No enhanced variant named '{key}'.")
    return encode_png(img)


def variant_pdf_bytes(result: PipelineResult, key: str) -> bytes:
    img = get_variant(result, key)
    if img is None:
        raise KeyError(f"No enhanced variant named '{key}'.")
    return images_to_pdf([img])


def ocr_text_bytes(ocr: Optional[OcrResult]) -> bytes:
    return text_to_utf8_bytes(ocr.text if ocr is not None else "")


def describe_ocr(ocr: Optional[OcrResult]) -> str:
    """One-sentence, user-facing summary of the OCR outcome."""
    if ocr is None:
        return "OCR was turned off."
    if ocr.status == "ok":
        n = len(ocr.words)
        return f"Recognised {n} words (mean Tesseract confidence {ocr.mean_confidence:.0f}%)."
    return ocr.message or f"OCR status: {ocr.status}."
