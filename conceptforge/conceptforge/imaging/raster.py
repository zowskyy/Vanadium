"""Image IO and resampling built on Pillow, normalised to float RGBA arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 512_000_000


def load_rgba(path: str | Path) -> np.ndarray:
    """Load an image as a float32 ``(H, W, 4)`` array in 0..1.

    EXIF orientation is applied so phone photos of a sketchbook come in upright.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"concept art not found: {path}")
    with Image.open(path) as img:
        img = _apply_exif_orientation(img)
        if img.mode not in ("RGBA", "RGB", "L", "LA"):
            img = img.convert("RGBA")
        rgba = img.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr)


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(img)
    except Exception:  # pragma: no cover - malformed EXIF
        return img


def save_image(array: np.ndarray, path: str | Path) -> Path:
    """Write a float 0..1 or uint8 array (H,W), (H,W,3) or (H,W,4) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(array)).save(path)
    return path


def to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == bool:
        return (arr.astype(np.uint8) * 255)
    return np.clip(np.rint(arr.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)


def resize_rgba(rgba: np.ndarray, size: tuple[int, int], resample: int | None = None) -> np.ndarray:
    """Resize an RGBA float array to ``(width, height)``."""
    width, height = int(size[0]), int(size[1])
    if rgba.shape[1] == width and rgba.shape[0] == height:
        return rgba
    mode = Image.LANCZOS if resample is None else resample
    img = Image.fromarray(to_uint8(rgba), mode="RGBA").resize((width, height), mode)
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask with area averaging, then re-threshold at 0.5."""
    width, height = int(size[0]), int(size[1])
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask.astype(bool)
    img = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L")
    resized = np.asarray(img.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0
    return resized >= 0.5


def fit_within(shape: tuple[int, int], longest_edge: int) -> tuple[int, int]:
    """Return ``(width, height)`` scaled so the longest edge matches, no upscale."""
    height, width = int(shape[0]), int(shape[1])
    longest = max(height, width)
    if longest <= longest_edge or longest == 0:
        return width, height
    scale = longest_edge / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def premultiplied_rgb(rgba: np.ndarray, background: float = 1.0) -> np.ndarray:
    """Composite RGBA over a flat background, for previews and debug dumps."""
    rgba = np.asarray(rgba, dtype=np.float64)
    alpha = rgba[..., 3:4]
    return rgba[..., :3] * alpha + background * (1.0 - alpha)


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight ``(x0, y0, x1, y1)`` bounds (exclusive max) of a boolean mask."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return 0, 0, mask.shape[1], mask.shape[0]
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
