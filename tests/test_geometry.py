"""Member 2: homography, output size, degenerate geometry, synthetic ground truth."""
import numpy as np
import pytest

from backend.contracts import ScannerError
from backend.geometry import compute_output_size, rotate_quarter_turns, validate_quad, warp_perspective
from tests.helpers import photo


def _code(fn, *a):
    with pytest.raises(ScannerError) as info:
        fn(*a)
    return info.value.code


def _scene():
    """Photo with a red rectangle at known coordinates, drawn on a grey background."""
    img = np.full((600, 800, 3), 90, np.uint8)
    img[100:400, 200:600] = (0, 0, 255)
    quad = np.array([[200, 100], [599, 100], [599, 399], [200, 399]], np.float32)
    return img, quad


def test_warp_of_axis_aligned_rectangle_recovers_it():
    img, quad = _scene()
    res = warp_perspective(img, quad)
    assert res.output_size == (399, 299)
    assert res.image.shape == (299, 399, 3)
    assert (res.image.reshape(-1, 3) == (0, 0, 255)).all(axis=1).mean() > 0.99


def test_homography_maps_source_corners_to_output_corners():
    img, quad = _scene()
    res = warp_perspective(img, quad)
    pts = np.hstack([quad, np.ones((4, 1))]) @ res.matrix.T
    mapped = pts[:, :2] / pts[:, 2:3]
    w, h = res.output_size
    assert np.allclose(mapped, [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], atol=1e-3)


def test_output_size_uses_longer_opposite_sides():
    quad = np.array([[0, 0], [100, 0], [90, 200], [10, 200]], np.float32)  # keystone: top 100, bottom 80
    assert compute_output_size(quad) == (100, 200)


def test_warp_of_rotated_page_matches_the_original_page():
    """Rotation only (no keystone) keeps the aspect ratio, so the result can be compared pixel-wise."""
    import cv2
    s = photo("wood", 12, 0.0, seed=4)
    res = warp_perspective(s.image, s.corners)
    # Compare at a resolution both images can support: the photo shows the page at ~500 px,
    # so comparing against the 1240 px original would only measure resampling blur.
    size = (310, 438)
    page = cv2.resize(s.page, size, interpolation=cv2.INTER_AREA).astype(np.float32).mean(axis=2)
    out = cv2.resize(res.image, size, interpolation=cv2.INTER_AREA).astype(np.float32).mean(axis=2)
    assert np.corrcoef(page.ravel(), out.ravel())[0, 1] > 0.9


def test_warp_of_keystone_page_is_upright_and_close_in_proportions():
    """With perspective, 4 points cannot recover the exact aspect ratio (documented limitation)
    but the page must be flattened: output aspect within 15 % of the true page aspect."""
    s = photo("wood", 10, 0.8, seed=4)
    w, h = warp_perspective(s.image, s.corners).output_size
    true_aspect = s.page.shape[1] / s.page.shape[0]
    assert abs((w / h) / true_aspect - 1) < 0.15


def test_wrong_order_counter_clockwise_is_rejected():
    img, quad = _scene()
    assert _code(warp_perspective, img, quad[::-1]) == "corner_order"


def test_bow_tie_is_rejected():
    img, quad = _scene()
    # a bow-tie has (almost) zero signed area, so it is caught as degenerate or as not convex
    assert _code(warp_perspective, img, quad[[0, 2, 1, 3]]) in ("not_convex", "degenerate")
    # a non-symmetric twisted quad has non-zero area and must be caught by the convexity check
    twisted = np.array([[200, 100], [599, 399], [599, 100], [250, 450]], np.float32)
    assert _code(warp_perspective, img, twisted) in ("not_convex", "corner_order")


def test_duplicate_corners_are_degenerate():
    img, quad = _scene()
    quad[1] = quad[0]
    assert _code(warp_perspective, img, quad) == "degenerate"


def test_collinear_points_are_degenerate():
    img = np.full((600, 800, 3), 90, np.uint8)
    line = np.array([[100, 100], [200, 100], [300, 100], [400, 100]], np.float32)
    assert _code(warp_perspective, img, line) == "degenerate"


def test_bad_shapes_and_nan():
    img, quad = _scene()
    assert _code(warp_perspective, img, quad[:3]) == "bad_corners"
    bad = quad.copy()
    bad[0, 0] = np.nan
    assert _code(warp_perspective, img, bad) == "bad_corners"
    assert _code(warp_perspective, np.zeros((10, 10), np.uint8), quad) == "bad_image"


def test_blank_region_is_reported_as_implausible():
    img = np.full((600, 800, 3), 128, np.uint8)
    quad = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], np.float32)
    assert _code(warp_perspective, img, quad) == "implausible_result"


def test_corner_outside_image_gives_warning_not_crash():
    img, quad = _scene()
    quad[1] = [900, 100]
    res = warp_perspective(img, quad)
    assert any("outside" in w for w in res.warnings)


def test_validate_quad_returns_float32_copy():
    img, quad = _scene()
    out = validate_quad(quad.astype(np.float64), img.shape)
    assert out.dtype == np.float32 and out.shape == (4, 2)


def test_rotate_quarter_turns():
    img = np.zeros((100, 200, 3), np.uint8)
    img[0, 0] = 255
    assert rotate_quarter_turns(img, 90).shape == (200, 100, 3)
    assert rotate_quarter_turns(img, 90)[0, 99].tolist() == [255, 255, 255]  # top-left goes to top-right
    assert rotate_quarter_turns(img, 180).shape == img.shape
    assert rotate_quarter_turns(img, 0) is not img
    assert _code(rotate_quarter_turns, img, 45) == "bad_rotation"
