"""Member 2: input validation, EXIF handling, working copy, coordinate scaling."""
import io

import numpy as np
import pytest
from PIL import Image

from backend.contracts import ScannerError
from backend.input_handler import load_image, make_working_copy, scale_corners
from tests.helpers import jpeg_bytes, photo


def _png_bytes(arr_or_img, mode=None):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    if mode:
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _code(fn, *a, **k):
    with pytest.raises(ScannerError) as info:
        fn(*a, **k)
    return info.value.code


def test_valid_jpeg_returns_bgr_uint8():
    loaded = load_image(jpeg_bytes(photo().image), "page.jpg")
    assert loaded.image.dtype == np.uint8 and loaded.image.ndim == 3 and loaded.image.shape[2] == 3
    assert loaded.format == "JPEG"


def test_valid_png_is_accepted():
    arr = np.full((400, 500, 3), 200, np.uint8)
    assert load_image(_png_bytes(arr), "scan.PNG").format == "PNG"


def test_bgr_order_is_correct():
    rgb = np.zeros((400, 400, 3), np.uint8)
    rgb[:, :, 0] = 255  # pure RED in RGB
    loaded = load_image(_png_bytes(rgb))
    assert loaded.image[10, 10].tolist() == [0, 0, 255]  # red in BGR order


def test_empty_file():
    assert _code(load_image, b"") == "empty_file"


def test_wrong_type():
    assert _code(load_image, "not bytes") == "bad_type"


def test_random_bytes_are_not_an_image():
    assert _code(load_image, b"this is definitely not an image" * 20, "x.jpg") == "not_an_image"


def test_truncated_jpeg_is_reported_as_corrupted():
    data = jpeg_bytes(photo().image)
    assert _code(load_image, data[: len(data) // 3], "x.jpg") == "corrupted_image"


def test_unsupported_extension():
    assert _code(load_image, jpeg_bytes(photo().image), "page.pdf") == "unsupported_extension"


def test_unsupported_format_with_valid_extension():
    buf = io.BytesIO()
    Image.new("RGB", (400, 400), "white").save(buf, "GIF")
    assert _code(load_image, buf.getvalue(), "fake.png") == "unsupported_format"


def test_file_too_large_uses_limit_parameter():
    assert _code(load_image, jpeg_bytes(photo().image), "x.jpg", max_bytes=1000) == "file_too_large"


def test_image_too_small():
    assert _code(load_image, _png_bytes(np.zeros((100, 100, 3), np.uint8)), "x.png") == "image_too_small"


def test_rgba_transparency_becomes_white_paper():
    rgba = np.zeros((400, 400, 4), np.uint8)  # fully transparent
    loaded = load_image(_png_bytes(rgba))
    assert loaded.image.min() == 255


def test_grayscale_and_palette_png():
    gray = np.full((400, 400), 128, np.uint8)
    assert load_image(_png_bytes(gray)).image.shape == (400, 400, 3)
    assert load_image(_png_bytes(Image.new("RGB", (400, 400), "white"), "P")).image.shape == (400, 400, 3)


def test_sixteen_bit_png_does_not_crash():
    arr = (np.arange(400 * 400, dtype=np.uint32).reshape(400, 400) % 65535).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    assert load_image(buf.getvalue()).image.shape == (400, 400, 3)


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (600, 400), "white")  # stored landscape
    exif = Image.Exif()
    exif[0x0112] = 6                             # "rotate 90 degrees to view"
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    loaded = load_image(buf.getvalue(), "phone.jpg")
    assert loaded.exif_rotated and loaded.image.shape[:2] == (600, 400)


def test_large_image_is_downscaled_with_warning():
    arr = np.full((1200, 1600, 3), 180, np.uint8)
    loaded = load_image(_png_bytes(arr), max_side=800)
    assert max(loaded.image.shape[:2]) == 800 and loaded.downscaled and loaded.warnings


def test_working_copy_scale_factors_map_back():
    img = np.zeros((1500, 2000, 3), np.uint8)
    wc = make_working_copy(img, 800)
    assert max(wc.image.shape[:2]) == 800
    h, w = wc.image.shape[:2]
    assert abs(wc.scale_x * w - 2000) < 1e-6 and abs(wc.scale_y * h - 1500) < 1e-6


def test_working_copy_of_small_image_is_unscaled():
    wc = make_working_copy(np.zeros((300, 400, 3), np.uint8), 800)
    assert (wc.scale_x, wc.scale_y) == (1.0, 1.0)


def test_scale_corners():
    out = scale_corners(np.array([[10, 20], [30, 40], [50, 60], [70, 80]]), (2.0, 3.0))
    assert out[1].tolist() == [60.0, 120.0]
