"""Member 3: output formats, shadow handling, quality warnings, option validation."""
import cv2
import numpy as np
import pytest

from backend.contracts import ScannerError
from backend.enhancement import (EnhancementOptions, adaptive_threshold_bw, background_uniformity, enhance,
                                 normalize_illumination, sharpness_score, text_paper_separation, to_grayscale)
from evaluation.synthetic import render_text_page
from tests.helpers import photo


def _page_with_shadow(strength=0.55):
    page = render_text_page()
    ramp = np.linspace(1.0, 1.0 - strength, page.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(page.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def test_variants_have_documented_formats():
    res = enhance(render_text_page())
    color, gray, bw = res.variants["color"], res.variants["grayscale"], res.variants["bw"]
    assert color.ndim == 3 and color.shape[2] == 3 and color.dtype == np.uint8
    assert gray.ndim == 2 and gray.dtype == np.uint8
    assert bw.ndim == 2 and set(np.unique(bw)).issubset({0, 255})
    assert color.shape[:2] == gray.shape == bw.shape


def test_bw_keeps_text_dark_and_paper_white():
    bw = enhance(render_text_page()).variants["bw"]
    black_fraction = (bw == 0).mean()
    assert 0.005 < black_fraction < 0.25   # some ink, mostly paper


def test_adaptive_beats_global_threshold_under_shadow():
    shadowed = _page_with_shadow()
    gray = to_grayscale(shadowed)
    _, global_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_side = (slice(None), slice(int(gray.shape[1] * 0.8), None))     # deepest shadow, margin area
    adaptive_bw = enhance(shadowed).variants["bw"]
    assert (global_bw[dark_side] == 0).mean() > (adaptive_bw[dark_side] == 0).mean()
    assert (adaptive_bw[dark_side] == 0).mean() < 0.10


def test_illumination_normalisation_evens_out_background():
    gray = to_grayscale(_page_with_shadow())
    assert background_uniformity(normalize_illumination(gray)) < background_uniformity(gray) / 3


def test_shadow_removal_can_be_switched_off():
    res = enhance(render_text_page(), EnhancementOptions(shadow_removal=False, denoise=False, clahe="off"))
    assert res.applied == ["adaptive_threshold"]


def test_clahe_auto_skips_clean_pages_and_runs_on_weak_contrast():
    clean = enhance(render_text_page())
    assert "clahe" not in clean.applied
    faint = (255 - (255 - render_text_page().astype(np.float32)) * 0.25).astype(np.uint8)  # washed-out ink
    assert "clahe" in enhance(faint).applied


def test_blur_is_detected_and_sharp_page_is_not_flagged():
    sharp = enhance(render_text_page())
    blurred = enhance(cv2.GaussianBlur(render_text_page(), (0, 0), 6))
    assert not sharp.quality.blur_suspected
    assert blurred.quality.blur_suspected and any("blurry" in w for w in blurred.warnings)
    assert sharpness_score(to_grayscale(render_text_page())) > sharpness_score(to_grayscale(cv2.GaussianBlur(render_text_page(), (0, 0), 6)))


def test_low_contrast_and_low_resolution_warnings():
    faint = (255 - (255 - render_text_page().astype(np.float32)) * 0.15).astype(np.uint8)
    assert enhance(faint).quality.low_contrast
    small = cv2.resize(render_text_page(), (400, 566))
    assert enhance(small).quality.low_resolution
    assert text_paper_separation(np.full((50, 50), 200, np.uint8)) == 0.0


def test_color_variant_keeps_colour_information():
    page = render_text_page()
    page[:, :, 2] = np.clip(page[:, :, 2].astype(int) + 0, 0, 255)
    page[300:420, 100:700] = (200, 60, 60)  # a blue box
    color = enhance(page).variants["color"]
    assert color[360, 400, 0] > color[360, 400, 2] + 30   # still blue-ish (BGR: B > R)


def test_invalid_inputs_raise_scanner_error():
    for bad in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((50, 50), np.uint8), np.zeros((50, 50, 3), np.float32)):
        with pytest.raises(ScannerError):
            enhance(bad)
    with pytest.raises(ScannerError):
        enhance(render_text_page(), EnhancementOptions(clahe="maybe"))


def test_adaptive_threshold_accepts_even_block_size():
    gray = to_grayscale(render_text_page())
    assert adaptive_threshold_bw(gray, block_size=40).shape == gray.shape  # corrected to odd internally


def test_enhancement_on_warped_synthetic_photo():
    from backend.geometry import warp_perspective
    s = photo("wood", 8, 0.5, shadow=0.5)
    res = enhance(warp_perspective(s.image, s.corners).image)
    assert set(res.variants) == {"color", "grayscale", "bw"}
