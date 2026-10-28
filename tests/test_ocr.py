"""Member 4: OCR integration.  Mock-based tests need no Tesseract; real ones are marked."""
import numpy as np
import pytest

from backend.contracts import ScannerError
from backend.ocr import (check_tesseract, detect_orientation, draw_word_boxes, prepare_for_ocr, run_ocr, OCR_PAD)
from evaluation.ocr_metrics import cer
from evaluation.synthetic import DEFAULT_LINES, render_text_page
from tests.helpers import FakeOcrBackend, requires_tesseract

# (text, conf, left, top, width, height, block, par, line)
TWO_LINES = [
    ("Hello", 96, 110, 120, 80, 30, 1, 1, 1), ("World", 92, 220, 120, 90, 30, 1, 1, 1),
    ("Second", 40, 110, 200, 100, 30, 1, 1, 2),
    ("Para", 88, 110, 400, 60, 30, 2, 1, 1),
]


def test_text_is_rebuilt_in_lines_and_paragraphs():
    res = run_ocr(np.full((1700, 1200), 255, np.uint8), backend=FakeOcrBackend(TWO_LINES))
    assert res.status == "ok"
    assert res.text == "Hello World\nSecond\n\nPara"


def test_confidences_and_structure_rows():
    res = run_ocr(np.full((1700, 1200), 255, np.uint8), backend=FakeOcrBackend(TWO_LINES))
    assert len(res.words) == 4                       # the conf=-1 structure row is ignored
    assert res.mean_confidence == pytest.approx((96 + 92 + 40 + 88) / 4)
    assert [w.text for w in res.low_confidence_words(60)] == ["Second"]


def test_word_boxes_are_mapped_back_for_upscaled_and_padded_input():
    small = np.full((800, 600), 255, np.uint8)       # long side 800 < 1600 -> upscaled x2
    fake = FakeOcrBackend([("A", 90, 2 * 50 + OCR_PAD, 2 * 60 + OCR_PAD, 2 * 20, 2 * 10, 1, 1, 1)])
    res = run_ocr(small, backend=fake)
    w = res.words[0]
    assert (w.left, w.top, w.width, w.height) == (50, 60, 20, 10)
    assert fake.calls[0][0] == (1600 + 2 * OCR_PAD, 1200 + 2 * OCR_PAD)


def test_no_text_status():
    res = run_ocr(np.full((900, 700), 255, np.uint8), backend=FakeOcrBackend([]))
    assert res.status == "no_text" and res.message and res.text == ""


def test_missing_tesseract_is_reported_not_raised():
    import pytesseract
    res = run_ocr(np.full((900, 700), 255, np.uint8), backend=FakeOcrBackend(error=pytesseract.TesseractNotFoundError()))
    assert res.status == "unavailable" and "Tesseract" in res.message


def test_missing_language_data_is_reported():
    import pytesseract
    err = pytesseract.TesseractError(1, "Failed loading language 'xyz'")
    res = run_ocr(np.full((900, 700), 255, np.uint8), lang="xyz", backend=FakeOcrBackend(error=err))
    assert res.status == "unavailable" and "language" in res.message.lower()


def test_other_engine_failure_is_status_error():
    res = run_ocr(np.full((900, 700), 255, np.uint8), backend=FakeOcrBackend(error=RuntimeError("boom")))
    assert res.status == "error" and "boom" in res.message


def test_invalid_arguments_raise():
    with pytest.raises(ScannerError):
        run_ocr(np.full((900, 700), 255, np.uint8), psm=99, backend=FakeOcrBackend())
    with pytest.raises(ScannerError):
        run_ocr(np.zeros((0, 0), np.uint8), backend=FakeOcrBackend())


def test_prepare_converts_color_and_pads():
    img, scale = prepare_for_ocr(np.zeros((2000, 1500, 3), np.uint8))
    assert img.ndim == 2 and scale == 1.0 and img.shape == (2000 + 2 * OCR_PAD, 1500 + 2 * OCR_PAD)


def test_orientation_with_fake_backend():
    assert detect_orientation(np.full((900, 700), 255, np.uint8), backend=FakeOcrBackend()) == (90, 5.0)
    assert detect_orientation(np.full((900, 700), 255, np.uint8), backend=FakeOcrBackend(error=RuntimeError())) is None


def test_check_tesseract_reports_missing_executable():
    class Broken:
        def get_tesseract_version(self):
            raise EnvironmentError("missing")
    ok, msg = check_tesseract("eng", backend=Broken())
    assert not ok and "not found" in msg


def test_draw_word_boxes_colours_by_confidence():
    # fake coordinates include the OCR_PAD margin, so 'Hello' maps back to left=100, top=110
    res = run_ocr(np.full((1700, 1200), 255, np.uint8), backend=FakeOcrBackend(TWO_LINES))
    img = draw_word_boxes(np.full((1700, 1200), 255, np.uint8), res.words)
    assert img.shape == (1700, 1200, 3) and (img[110, 120] == (0, 160, 0)).all() and (img[190, 120] == (0, 0, 255)).all()


# ---- tests that need the real engine -------------------------------------------
@requires_tesseract
def test_real_tesseract_reads_clean_rendered_page():
    page = render_text_page()
    res = run_ocr(page)
    assert res.status == "ok" and res.mean_confidence > 60
    reference = "\n".join(l for l in DEFAULT_LINES if l)
    assert cer(reference, res.text) < 0.1


@requires_tesseract
def test_real_tesseract_blank_page_gives_no_text():
    assert run_ocr(np.full((1000, 800), 255, np.uint8)).status == "no_text"


@requires_tesseract
def test_real_tesseract_unknown_language_is_unavailable():
    res = run_ocr(render_text_page(), lang="zzz_not_installed")
    assert res.status in ("unavailable", "error") and res.message
