"""Exporting results  (Member 4 - OCR & Application Engineer).

All functions return ``bytes`` so the Streamlit download buttons, the tests and
the command-line tools can use the same code.

* PNG  : lossless, so thin text strokes are not damaged (JPEG would add ringing).
* TXT  : UTF-8 (so accented and non-Latin characters survive).
* PDF  : built with Pillow.  One page per image; B&W pages use 1-bit mode which
         is compact and keeps text edges crisp.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from PIL import Image

from .contracts import STAGE_EXPORT, ScannerError


def _fail(code: str, message: str) -> ScannerError:
    return ScannerError(STAGE_EXPORT, code, message)


def _check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise _fail("bad_image", "Nothing to export: expected a non-empty uint8 image.")


def encode_png(image: np.ndarray) -> bytes:
    """PNG bytes for a grey (H,W) or BGR (H,W,3) image."""
    _check_image(image)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise _fail("encode_failed", "The image could not be encoded as PNG.")
    return buf.tobytes()


def text_to_utf8_bytes(text: str) -> bytes:
    """Text file content as UTF-8 bytes (no BOM; Notepad and VS Code read it correctly)."""
    return (text or "").encode("utf-8")


def _to_pil(image: np.ndarray) -> Image.Image:
    _check_image(image)
    if image.ndim == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 2-D: if it contains only 0/255 it is a B&W page -> 1-bit PDF page
    if np.isin(np.unique(image), (0, 255)).all():
        return Image.fromarray(image).convert("1")
    return Image.fromarray(image)


def images_to_pdf(images: Sequence[np.ndarray], dpi: int = 150) -> bytes:
    """PDF with one page per image (all pages are the same ``dpi``)."""
    if not images:
        raise _fail("no_images", "At least one image is required to build a PDF.")
    pages: List[Image.Image] = [_to_pil(img) for img in images]
    buf = io.BytesIO()
    try:
        pages[0].save(buf, format="PDF", resolution=float(dpi), save_all=True, append_images=pages[1:])
    except Exception as exc:
        raise _fail("pdf_failed", f"PDF creation failed: {exc}") from None
    return buf.getvalue()


def save_debug_bundle(result, folder) -> List[Path]:
    """Write every intermediate image of a ``PipelineResult`` into ``folder``.

    Use it to find out which stage caused a bad result: look at
    00_original, 01_outline, 02_warped, 03_* variants in order.
    """
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def _write(name: str, img) -> None:
        if img is not None:
            path = out / name
            cv2.imwrite(str(path), img)
            written.append(path)

    _write("00_original.png", result.original)
    _write("01_outline.png", result.outline_preview)
    if result.detection is not None:
        for key, img in result.detection.debug.items():
            _write(f"detect_{key}.png", img)
    if result.warp is not None:
        _write("02_warped.png", result.warp.image)
    if result.enhancement is not None:
        for key, img in result.enhancement.variants.items():
            _write(f"03_{key}.png", img)
    return written
