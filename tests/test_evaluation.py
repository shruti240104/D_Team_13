"""Metric tools (Members 1, 2, 4): known-answer tests."""
import numpy as np
import pytest

from evaluation.detection_metrics import corner_error
from evaluation.ocr_metrics import cer, edit_distance, wer
from evaluation.synthetic import make_photo

QUAD = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], np.float32)


def test_edit_distance_known_values():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("abc", "abc") == 0


def test_cer_and_wer():
    assert cer("hello world", "hello world") == 0.0
    assert cer("abcd", "abxd") == pytest.approx(0.25)
    assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert cer("a  b\n c", "a b c") == 0.0            # whitespace is normalised
    assert cer("", "") == 0.0 and cer("", "x") == 1.0


def test_corner_error_is_normalised_by_diagonal():
    assert corner_error(QUAD, QUAD, (1000, 1000)) == 0.0
    shifted = QUAD + np.array([10, 0], np.float32)
    assert corner_error(shifted, QUAD, (600, 800)) == pytest.approx(10 / 1000.0)


def test_synthetic_generator_is_deterministic_and_has_valid_truth():
    a, b = make_photo(seed=3, perspective=0.5), make_photo(seed=3, perspective=0.5)
    assert np.array_equal(a.image, b.image) and np.allclose(a.corners, b.corners)
    assert a.corners.shape == (4, 2) and a.text.startswith("DOCUMENT")
