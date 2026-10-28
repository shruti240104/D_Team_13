"""Member 1: corner ordering, detection success/failure, coordinate scaling."""
import numpy as np
import pytest

from backend.contracts import ScannerError
from backend.detection import detect_document, draw_corners, order_corners, polygon_area, quad_iou
from backend.input_handler import make_working_copy
from tests.helpers import photo

SQUARE = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], np.float32)


def _detect(sample):
    h, w = sample.image.shape[:2]
    wc = make_working_copy(sample.image, 800)
    return detect_document(wc.image, (wc.scale_x, wc.scale_y), (w, h))


# ---- corner ordering -------------------------------------------------------
def test_order_corners_is_permutation_invariant():
    rng = np.random.default_rng(0)
    for _ in range(20):
        shuffled = SQUARE[rng.permutation(4)]
        assert np.allclose(order_corners(shuffled), SQUARE)


def test_order_corners_on_rotated_quad_is_clockwise():
    quad = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], np.float32)  # diamond (45 deg)
    ordered = order_corners(quad[::-1])
    area = np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1)) - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1))
    assert area > 0  # clockwise on screen


def test_order_corners_requires_four_points():
    with pytest.raises(ScannerError):
        order_corners([[0, 0], [1, 1], [2, 2]])


def test_polygon_area_and_iou():
    assert polygon_area(SQUARE) == 10000
    assert quad_iou(SQUARE, SQUARE) == pytest.approx(1.0)
    assert quad_iou(SQUARE, SQUARE + 1000) == pytest.approx(0.0)
    assert quad_iou(SQUARE, SQUARE + np.array([50, 0], np.float32)) == pytest.approx(1 / 3, abs=1e-3)


# ---- successful detection on synthetic photos -------------------------------
def test_detects_clean_page_with_high_overlap():
    s = photo("dark", 0, 0)
    det = _detect(s)
    assert det.success and det.corners.shape == (4, 2) and det.corners.dtype == np.float32
    assert quad_iou(det.corners, s.corners) > 0.95


def test_detects_rotated_page_with_perspective():
    s = photo("wood", 15, 0.8)
    det = _detect(s)
    assert det.success and quad_iou(det.corners, s.corners) > 0.95


def test_detects_page_on_cluttered_background():
    s = photo("cluttered", -12, 0.5)
    det = _detect(s)
    assert det.success and quad_iou(det.corners, s.corners) > 0.9


def test_corners_are_in_original_coordinates_after_scaling():
    s = photo("dark", 5, 0.3)   # 1200x900 photo, working copy is 800 wide -> scale 1.5
    det = _detect(s)
    assert det.corners[:, 0].max() > 800   # impossible if still in working coordinates
    assert np.linalg.norm(det.corners - s.corners, axis=1).max() < 25


def test_result_carries_debug_images_and_method():
    det = _detect(photo("dark"))
    assert det.method and "edges" in det.debug and "candidates" in det.debug
    assert 0.0 <= det.quality <= 1.0


# ---- failures must be graceful ------------------------------------------------
def test_no_document_returns_clear_failure():
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(120, 10, (600, 800, 3)), 0, 255).astype(np.uint8)  # plain noisy surface
    det = detect_document(img)
    assert not det.success and det.failure_code in ("no_candidates", "low_quality") and det.failure_reason


def test_blank_white_image_fails_without_crashing():
    det = detect_document(np.full((600, 800, 3), 255, np.uint8))
    assert not det.success and det.corners is None


def test_wrong_input_type_raises_scanner_error():
    with pytest.raises(ScannerError):
        detect_document(np.zeros((100, 100), np.uint8))


def test_white_paper_on_almost_white_table_never_returns_wrong_success_silently():
    """Documented limitation: the result may be success OR failure, but if it is a
    success the corners must at least be a valid quadrilateral inside the image."""
    s = photo("lowcontrast", 0, 0)
    det = _detect(s)
    if det.success:
        assert det.corners.shape == (4, 2)
        assert det.corners.min() >= 0
    else:
        assert det.failure_reason


def test_draw_corners_keeps_shape():
    img = np.zeros((200, 300, 3), np.uint8)
    out = draw_corners(img, SQUARE)
    assert out.shape == img.shape and out.sum() > 0 and img.sum() == 0
