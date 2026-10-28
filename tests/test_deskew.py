"""Tests for backend.deskew (text-line straightening)."""
import cv2
import numpy as np

from backend.deskew import MIN_DESKEW_ANGLE, deskew, estimate_text_skew, rotate_keep_all
from evaluation.synthetic import render_text_page


def _page():
    return cv2.resize(render_text_page(), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def test_straight_page_is_left_alone():
    out, angle = deskew(_page())
    assert angle == 0.0 and out.shape == _page().shape


def test_tilt_is_estimated_within_one_degree():
    for tilt in (-20, -9, 4, 12, 25):
        tilted = rotate_keep_all(_page(), tilt)       # tilt the text by `tilt`
        est = estimate_text_skew(tilted)
        assert abs(est + tilt) <= 1.0, (tilt, est)    # the correction is the opposite rotation


def test_deskew_makes_lines_horizontal_again():
    tilted = rotate_keep_all(_page(), 15)
    fixed, angle = deskew(tilted)
    assert abs(angle) >= MIN_DESKEW_ANGLE
    assert abs(estimate_text_skew(fixed)) <= 0.6


def test_blank_page_gives_no_rotation():
    blank = np.full((400, 300, 3), 240, np.uint8)
    assert estimate_text_skew(blank) == 0.0