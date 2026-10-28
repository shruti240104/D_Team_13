from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .contracts import STAGE_INPUT, LoadedImage, ScannerError, WorkingCopy

SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MAX_FILE_BYTES = 20 * 1024 * 1024
MIN_SIDE_PX = 300
MAX_PIXELS = 60_000_000
MAX_SIDE_PX = 4000
EXIF_ORIENTATION_TAG = 0x0112


def _fail(code: str, message: str, hint: str = "") -> ScannerError:
    return ScannerError(STAGE_INPUT, code, message, hint)


def load_image(
    data: bytes,
    filename: Optional[str] = None,
    max_bytes: int = MAX_FILE_BYTES,
    min_side: int = MIN_SIDE_PX,
    max_side: int = MAX_SIDE_PX,
) -> LoadedImage:
    if not isinstance(data, (bytes, bytearray)):
        raise _fail("bad_type", "The uploaded data is not a byte string.")
    if len(data) == 0:
        raise _fail("empty_file", "The uploaded file is empty.", "Choose a different file.")
    if len(data) > max_bytes:
        raise _fail(
            "file_too_large",
            f"The file is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB.",
            "Reduce the photo resolution or compress it and try again.",
        )
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise _fail(
                "unsupported_extension",
                f"Unsupported file type '{ext or '(none)'}'.",
                "Upload a .jpg, .jpeg, .png or .webp image.",
            )

    try:
        pil = Image.open(io.BytesIO(bytes(data)))
    except Image.DecompressionBombError:
        raise _fail("too_many_pixels", "The image has an extreme number of pixels.") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise _fail(
            "not_an_image",
            "The file could not be read as an image.",
            "It may be corrupted or not really a JPEG/PNG file.",
        ) from None

    fmt = pil.format
    if fmt not in SUPPORTED_FORMATS:
        raise _fail(
            "unsupported_format",
            f"Image format '{fmt}' is not supported.",
            "Convert the image to JPEG or PNG.",
        )

    width, height = pil.size
    if width * height > MAX_PIXELS:
        raise _fail(
            "too_many_pixels",
            f"The image is {width}x{height} pixels, which is too large to process.",
            "Resize the image first.",
        )

    try:
        pil.load()
    except (OSError, SyntaxError, ValueError) as exc:
        raise _fail(
            "corrupted_image",
            "The image data is damaged or incomplete.",
            f"Decoder message: {exc}",
        ) from None

    try:
        orientation = pil.getexif().get(EXIF_ORIENTATION_TAG, 1)
    except Exception:
        orientation = 1

    exif_rotated = orientation not in (0, 1)
    if exif_rotated:
        pil = ImageOps.exif_transpose(pil)

    try:
        if pil.mode in ("RGBA", "LA") or (
            pil.mode == "P" and "transparency" in pil.info
        ):
            rgba = pil.convert("RGBA")
            backdrop = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            backdrop.alpha_composite(rgba)
            rgb = backdrop.convert("RGB")
        else:
            rgb = pil.convert("RGB")
    except ValueError:
        arr16 = np.asarray(pil).astype(np.float32)
        if arr16.max() > 255:
            arr16 = arr16 / 256.0
        rgb = Image.fromarray(np.clip(arr16, 0, 255).astype(np.uint8)).convert("RGB")

    bgr = cv2.cvtColor(
        np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)),
        cv2.COLOR_RGB2BGR,
    )

    h, w = bgr.shape[:2]
    if min(h, w) < min_side:
        raise _fail(
            "image_too_small",
            f"The image is only {w}x{h} pixels; the shortest side must be at least {min_side}.",
            "Use a higher-resolution photo so the text stays readable.",
        )

    warnings = []
    downscaled = False

    if max(h, w) > max_side:
        f = max_side / float(max(h, w))
        bgr = cv2.resize(
            bgr,
            (int(round(w * f)), int(round(h * f))),
            interpolation=cv2.INTER_AREA,
        )
        downscaled = True
        warnings.append(
            f"Large image ({w}x{h}) was scaled down to "
            f"{bgr.shape[1]}x{bgr.shape[0]} for speed."
        )

    if exif_rotated:
        warnings.append("EXIF orientation was applied (the photo was stored rotated).")

    return LoadedImage(
        image=bgr,
        format=fmt,
        file_size=len(data),
        original_size=(width, height),
        exif_rotated=exif_rotated,
        downscaled=downscaled,
        warnings=warnings,
    )


def make_working_copy(image: np.ndarray, max_side: int = 800) -> WorkingCopy:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise _fail("bad_image", "Expected a 3-channel BGR image.")

    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return WorkingCopy(image=image.copy(), scale_x=1.0, scale_y=1.0)

    f = max_side / float(longest)
    new_w = max(1, int(round(w * f)))
    new_h = max(1, int(round(h * f)))

    small = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    return WorkingCopy(
        image=small,
        scale_x=w / float(new_w),
        scale_y=h / float(new_h),
    )


def scale_corners(corners: np.ndarray, scale: Tuple[float, float]) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return pts * np.array([scale[0], scale[1]], dtype=np.float32)