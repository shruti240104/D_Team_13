"""Member 2 (+ all): pipeline orchestration, structured errors, manual corners."""
import numpy as np

from backend.detection import quad_iou
from backend.enhancement import EnhancementOptions
from backend.pipeline import PipelineOptions, rescan_with_corners, run_pipeline
from evaluation.ocr_metrics import cer
from tests.helpers import FakeOcrBackend, jpeg_bytes, photo, requires_tesseract

WORDS = [("Hello", 95, 10, 10, 50, 20, 1, 1, 1)]


def _run(sample, **kw):
    return run_pipeline(jpeg_bytes(sample.image), "photo.jpg", PipelineOptions(**kw), ocr_backend=FakeOcrBackend(WORDS))


def test_happy_path_produces_all_outputs_with_mock_ocr():
    s = photo("wood", 10, 0.6)
    res = _run(s)
    assert res.ok and res.error is None
    assert quad_iou(res.corners, s.corners) > 0.95 and res.corner_source == "auto"
    assert res.original is not None and res.outline_preview is not None and res.warped is not None
    assert set(res.enhancement.variants) == {"color", "grayscale", "bw"}
    assert res.ocr.status == "ok" and res.ocr.text == "Hello" and res.ocr_source == "grayscale"
    assert {"load", "detect", "warp", "enhance", "ocr"} <= set(res.timings)


def test_ocr_source_option_is_used():
    fake = FakeOcrBackend(WORDS)
    res = run_pipeline(jpeg_bytes(photo().image), "x.jpg", PipelineOptions(ocr_source="bw"), ocr_backend=fake)
    assert res.ocr_source == "bw"


def test_corrupted_file_gives_structured_input_error():
    res = run_pipeline(b"garbage" * 50, "x.jpg")
    assert not res.ok and (res.error.stage, res.error.code) == ("input", "not_an_image")
    assert res.error.hint is not None and res.original is None


def test_unsupported_extension_error():
    res = run_pipeline(jpeg_bytes(photo().image), "notes.pdf")
    assert res.error.code == "unsupported_extension"


def test_no_document_falls_back_to_whole_photo_and_keeps_debug_for_the_ui():
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(120, 8, (700, 900, 3)), 0, 255).astype(np.uint8)
    res = run_pipeline(jpeg_bytes(img), "empty_desk.jpg", ocr_backend=FakeOcrBackend())
    assert res.ok and res.corner_source == "virtual"          # no outline, no text -> whole photo
    assert res.original is not None and res.outline_preview is not None and res.detection.debug


def test_no_outline_crops_around_the_text_instead_of_the_whole_photo():
    import cv2
    img = np.full((900, 1200, 3), 238, np.uint8)               # page blends into the background
    for i, y in enumerate(range(300, 480, 36)):
        cv2.putText(img, "Invoice total amount due today", (400, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (20, 20, 20), 2, cv2.LINE_AA)
    res = run_pipeline(jpeg_bytes(img), "blank_page.jpg", ocr_backend=FakeOcrBackend())
    assert res.ok and res.corner_source == "text"
    xs, ys = res.corners[:, 0], res.corners[:, 1]
    assert xs.min() < 400 and xs.max() > 800 and ys.min() < 280 and ys.max() > 470   # all text kept
    assert (xs.max() - xs.min()) < 1000 and (ys.max() - ys.min()) < 600               # but not the whole photo


def test_ocr_unavailable_does_not_lose_the_scan():
    import pytesseract
    res = run_pipeline(jpeg_bytes(photo().image), "x.jpg",
                       ocr_backend=FakeOcrBackend(error=pytesseract.TesseractNotFoundError()))
    assert res.ok and res.ocr.status == "unavailable" and res.enhancement is not None
    assert any("OCR could not run" in w for w in res.warnings)


def test_invalid_psm_is_contained():
    res = run_pipeline(jpeg_bytes(photo().image), "x.jpg", PipelineOptions(ocr_psm=77), ocr_backend=FakeOcrBackend())
    assert res.ok and res.ocr.status == "error"


def test_ocr_can_be_disabled():
    res = run_pipeline(jpeg_bytes(photo().image), "x.jpg", PipelineOptions(run_ocr=False))
    assert res.ok and res.ocr is None


def test_manual_corners_really_change_the_result():
    s = photo("dark", 0, 0)
    first = _run(s)
    assert first.ok
    # user shrinks the selection to the upper half of the detected page
    c = first.corners.copy()
    c[2] = (c[1] + c[2]) / 2
    c[3] = (c[0] + c[3]) / 2
    second = rescan_with_corners(first, c, PipelineOptions(), FakeOcrBackend(WORDS))
    assert second.ok and second.corner_source == "manual"
    assert second.warp.output_size[1] < first.warp.output_size[1] * 0.6
    assert np.allclose(second.corners, c, atol=1e-3)
    assert second.original is first.original            # no re-decoding


def test_manual_corners_accept_unordered_points_and_reject_degenerate():
    first = _run(photo("dark"))
    scrambled = first.corners[[2, 0, 3, 1]]
    assert rescan_with_corners(first, scrambled, ocr_backend=FakeOcrBackend()).ok
    bad = rescan_with_corners(first, np.array([[5, 5], [5, 5], [5, 5], [5, 5]]), ocr_backend=FakeOcrBackend())
    assert not bad.ok and bad.error.stage == "geometry"
    assert bad.original is not None


def test_rescan_without_image_returns_error():
    from backend.pipeline import PipelineResult
    assert rescan_with_corners(PipelineResult(), np.zeros((4, 2))).error.code == "no_image"


def test_enhancement_options_flow_through():
    res = run_pipeline(jpeg_bytes(photo().image), "x.jpg",
                       PipelineOptions(enhancement=EnhancementOptions(shadow_removal=False, denoise=False, clahe="off"), run_ocr=False))
    assert res.enhancement.applied == ["adaptive_threshold"]


def test_auto_orient_rotates_with_mock_osd():
    res = run_pipeline(jpeg_bytes(photo().image), "x.jpg", PipelineOptions(auto_orient=True), ocr_backend=FakeOcrBackend(WORDS))
    assert res.rotation_applied == 90 and res.warp.output_size == (res.warp.image.shape[1], res.warp.image.shape[0])


@requires_tesseract
def test_real_end_to_end_ocr_on_synthetic_photo():
    s = photo("wood", 10, 0.6, seed=7)
    res = run_pipeline(jpeg_bytes(s.image), "photo.jpg")
    assert res.ok and res.ocr.status == "ok"
    assert cer(s.text, res.ocr.text) < 0.15


@requires_tesseract
def test_real_orientation_correction_of_sideways_page():
    import cv2
    s = photo("dark", 0, 0, seed=2)
    sideways = cv2.rotate(s.image, cv2.ROTATE_90_CLOCKWISE)
    res = run_pipeline(jpeg_bytes(sideways), "x.jpg", PipelineOptions(auto_orient=True))
    assert res.ok
    # either the page was rotated back and OCR succeeded, or the pipeline honestly reported it could not tell
    assert res.rotation_applied != 0 or any("orientation" in w.lower() for w in res.warnings)
