"""Tests for the text-aware crop protection (backend/text_region.py)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.detection import order_corners
from backend.text_region import expand_quad_to_text, find_text_blocks, text_region_quad


def _page(h=600, w=800, text_box=(150, 100, 650, 500), bg=(120, 150, 190)):
    """Light page photo on a coloured desk, with text lines inside ``text_box``."""
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.rectangle(img, (100, 60), (700, 540), (245, 245, 245), -1)       # the paper
    x0, y0, x1, y1 = text_box
    for y in range(y0, y1, 34):
        cv2.putText(img, "The quick brown fox jumps over", (x0, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def _inside(points, quad):
    return all(cv2.pointPolygonTest(quad.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), False) >= 0
               for x, y in points)


def test_blocks_found_on_paper_text():
    blocks = find_text_blocks(_page())
    assert blocks, "text on a page must be found"


def test_coloured_desk_texture_is_not_text():
    rng = np.random.default_rng(0)
    img = np.full((600, 800, 3), (60, 110, 170), np.uint8)
    img = np.clip(img + rng.integers(-25, 25, img.shape), 0, 255).astype(np.uint8)
    assert find_text_blocks(img) == []


def test_small_detected_quad_is_grown_to_hold_all_text():
    img = _page()
    blocks = find_text_blocks(img)
    small = np.array([[300, 200], [500, 200], [500, 350], [300, 350]], np.float32)   # cuts the text
    grown, added = expand_quad_to_text(small, blocks, img.shape)
    assert added >= 1
    pts = np.vstack([b.points for b in blocks])
    assert _inside(pts, grown)


def test_quad_already_holding_text_is_unchanged():
    img = _page()
    blocks = find_text_blocks(img)
    page = np.array([[100, 60], [700, 60], [700, 540], [100, 540]], np.float32)
    grown, _ = expand_quad_to_text(page, blocks, img.shape)
    assert np.allclose(grown, page, atol=1.0)


def test_tilted_quad_keeps_its_tilt_when_grown():
    img = _page()
    blocks = find_text_blocks(img)
    small = np.array([[320, 190], [520, 210], [510, 360], [310, 340]], np.float32)
    grown, _ = expand_quad_to_text(small, blocks, img.shape)
    top_before = small[1] - small[0]
    top_after = grown[1] - grown[0]
    cos = float(np.dot(top_before, top_after) / (np.linalg.norm(top_before) * np.linalg.norm(top_after)))
    assert cos > 0.999


def test_far_away_text_is_not_pulled_in():
    img = _page(text_box=(150, 100, 650, 140))
    cv2.putText(img, "unrelated sticker text here", (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    blocks = find_text_blocks(img)
    page = np.array([[100, 60], [700, 60], [700, 300], [100, 300]], np.float32)
    grown, _ = expand_quad_to_text(page, blocks, img.shape)
    assert grown[:, 1].max() < 450


def test_text_box_when_no_outline_is_around_the_text_only():
    img = _page(text_box=(150, 100, 400, 200))
    blocks = find_text_blocks(img)
    box = text_region_quad(blocks, img.shape)
    assert box is not None
    area = cv2.contourArea(box.reshape(-1, 1, 2))
    assert area < 0.5 * img.shape[0] * img.shape[1]
    pts = np.vstack([b.points for b in blocks])
    assert _inside(pts, order_corners(box))


def test_text_all_over_gives_full_photo():
    img = np.full((600, 800, 3), 245, np.uint8)              # the page fills the photo
    for y in range(40, 580, 34):
        for x in (20, 400):
            cv2.putText(img, "The quick brown fox", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    box = text_region_quad(find_text_blocks(img), img.shape)
    h, w = img.shape[:2]
    assert cv2.contourArea(box.reshape(-1, 1, 2)) > 0.9 * w * h


def test_no_text_returns_none():
    assert text_region_quad([], (600, 800, 3)) is None
