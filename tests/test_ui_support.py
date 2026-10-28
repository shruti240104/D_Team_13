"""Member 4: pure helper functions used by the Streamlit app."""
import numpy as np
import pytest

from backend.export import encode_png
from backend.geometry import validate_quad
from backend.pipeline import PipelineOptions, run_pipeline
from backend.ui_support import (VARIANT_LABELS, corners_to_percent, default_manual_corners, describe_ocr,
                                ocr_text_bytes, percent_to_corners, variant_pdf_bytes, variant_png_bytes)
from backend.contracts import OcrResult
from tests.helpers import jpeg_bytes, photo


def test_percent_roundtrip():
    shape = (900, 1200, 3)
    corners = np.array([[100, 80], [1000, 120], [1050, 800], [60, 760]], np.float32)
    back = percent_to_corners(corners_to_percent(corners, shape), shape)
    assert np.allclose(back, corners, atol=1e-3)


def test_percent_to_corners_orders_arbitrary_points():
    values = [90, 90, 10, 10, 90, 10, 10, 90]   # scrambled order
    ordered = percent_to_corners(values, (1000, 1000, 3))
    validate_quad(ordered, (1000, 1000, 3))     # raises if not clockwise TL,TR,BR,BL


def test_default_manual_corners_are_valid():
    validate_quad(default_manual_corners((900, 1200, 3)), (900, 1200, 3))


def test_downloads_come_from_the_real_pipeline_result():
    res = run_pipeline(jpeg_bytes(photo("dark", 5, 0.3).image), "x.jpg", PipelineOptions(run_ocr=False))
    for key in VARIANT_LABELS:
        assert variant_png_bytes(res, key)[:4] == b"\x89PNG"
        assert variant_pdf_bytes(res, key).startswith(b"%PDF")
    with pytest.raises(KeyError):
        variant_png_bytes(res, "nonexistent")


def test_ocr_text_bytes_and_description():
    assert ocr_text_bytes(None) == b""
    ok = OcrResult(status="ok", text="é", mean_confidence=91.0, words=[object()])
    assert ocr_text_bytes(ok) == "é".encode("utf-8") and "91" in describe_ocr(ok)
    assert "not found" in describe_ocr(OcrResult(status="unavailable", message="Tesseract not found"))
    assert "off" in describe_ocr(None)
