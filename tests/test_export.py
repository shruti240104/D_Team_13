"""Member 4: exported file validity."""
import cv2
import numpy as np
import pytest

from backend.contracts import ScannerError
from backend.export import encode_png, images_to_pdf, text_to_utf8_bytes


def _decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def test_png_roundtrip_colour_and_gray_is_lossless():
    color = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    gray = np.random.default_rng(1).integers(0, 255, (40, 60), dtype=np.uint8)
    for img in (color, gray):
        data = encode_png(img)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert np.array_equal(_decode(data), img)


def test_utf8_text_with_non_ascii_survives():
    text = "Café déjà vu — नमस्ते — 你好"
    assert text_to_utf8_bytes(text).decode("utf-8") == text
    assert text_to_utf8_bytes(None) == b""


def test_pdf_is_valid_and_has_one_page_per_image():
    pages = [np.full((300, 200, 3), 255, np.uint8), np.full((300, 200), 128, np.uint8),
             np.where(np.arange(200)[None, :] % 2 == 0, 0, 255).astype(np.uint8).repeat(300, axis=0)]
    data = images_to_pdf(pages)
    assert data.startswith(b"%PDF-") and b"%%EOF" in data[-1024:]
    import re
    assert len(re.findall(rb"/Type\s*/Page\b", data)) == 3


def test_export_rejects_bad_images():
    for bad in (None, np.zeros((0, 0), np.uint8), np.zeros((5, 5), np.float32)):
        with pytest.raises(ScannerError):
            encode_png(bad)
    with pytest.raises(ScannerError):
        images_to_pdf([])
