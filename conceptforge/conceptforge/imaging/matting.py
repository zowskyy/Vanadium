"""Turn concept art into a clean character silhouette.

Three sources of truth are tried in order:

1. A real alpha channel, if the artwork ships one.
2. Border-seeded flood fill in colour space, which handles the overwhelmingly
   common case of art on a flat (white, grey, or coloured) backdrop.
3. A global colour-distance threshold as a last resort.

Whatever produces the matte, the result goes through the same cleanup: keep
significant blobs, fill interior holes (so eye whites and highlights do not
punch through the body), and close pixel-level noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conceptforge import ndops
from conceptforge.imaging.raster import luminance


@dataclass
class MatteResult:
    mask: np.ndarray
    """Boolean (H, W) silhouette."""

    coverage: float
    """Fraction of the frame the silhouette occupies."""

    source: str
    """Which strategy produced the matte: ``alpha``, ``floodfill`` or ``threshold``."""

    background_color: np.ndarray
    """Estimated backdrop colour, reused by the texture baker to reject fringe."""


def extract_matte(
    rgba: np.ndarray,
    alpha_threshold: float = 0.5,
    background_tolerance: float = 0.14,
    min_blob_ratio: float = 0.02,
    morph_radius: int = 2,
) -> MatteResult:
    rgba = np.asarray(rgba, dtype=np.float64)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("extract_matte expects an (H, W, 4) RGBA array")
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    background = estimate_background_color(rgb)

    if _alpha_is_informative(alpha):
        mask = alpha >= alpha_threshold
        source = "alpha"
    else:
        mask = _flood_fill_matte(rgb, background, background_tolerance)
        source = "floodfill"
        if not _plausible(mask):
            mask = _threshold_matte(rgb, background, background_tolerance)
            source = "threshold"

    mask = clean_mask(mask, min_blob_ratio=min_blob_ratio, morph_radius=morph_radius)
    coverage = float(mask.mean())
    return MatteResult(mask=mask, coverage=coverage, source=source, background_color=background)


def _alpha_is_informative(alpha: np.ndarray) -> bool:
    """True when the alpha channel actually carries a matte."""
    transparent = float((alpha < 0.5).mean())
    return 0.02 < transparent < 0.995


def estimate_background_color(rgb: np.ndarray) -> np.ndarray:
    """Median colour of a thin border band, which is backdrop in practice."""
    h, w = rgb.shape[:2]
    band = max(1, int(round(0.02 * max(h, w))))
    samples = np.concatenate(
        [
            rgb[:band].reshape(-1, 3),
            rgb[-band:].reshape(-1, 3),
            rgb[:, :band].reshape(-1, 3),
            rgb[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0)


def _color_distance(rgb: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Perceptually weighted RGB distance, 0..~1."""
    diff = rgb - np.asarray(color, dtype=np.float64).reshape(1, 1, 3)
    weights = np.array([0.30, 0.59, 0.11])
    chroma = np.sqrt((diff * diff * weights).sum(axis=2))
    # Luminance difference alone separates line art on white paper.
    lum = np.abs(luminance(rgb) - float(luminance(np.asarray(color).reshape(1, 1, 3))[0, 0]))
    return np.maximum(chroma, lum * 0.75)


def _flood_fill_matte(rgb: np.ndarray, background: np.ndarray, tolerance: float) -> np.ndarray:
    """Background = border-connected pixels within ``tolerance`` of backdrop."""
    similar = _color_distance(rgb, background) <= max(tolerance, 1e-4)
    labels, count = ndops.connected_components(similar, connectivity=2)
    if count == 0:
        return np.ones(rgb.shape[:2], dtype=bool)
    border = np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]]
    )
    border_labels = np.unique(border[border >= 0])
    if border_labels.size == 0:
        return np.ones(rgb.shape[:2], dtype=bool)
    return ~np.isin(labels, border_labels)


def _threshold_matte(rgb: np.ndarray, background: np.ndarray, tolerance: float) -> np.ndarray:
    distance = _color_distance(rgb, background)
    level = max(tolerance, _otsu_threshold(distance))
    return distance > level


def _otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    """Classic Otsu split, used when flood fill cannot find a backdrop."""
    v = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return hi
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return hi
    p = hist / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_total * omega - mu) ** 2 / denom, 0.0)
    return float(centers[int(np.argmax(sigma_b))])


def _plausible(mask: np.ndarray) -> bool:
    coverage = float(mask.mean())
    return 0.01 < coverage < 0.92


def clean_mask(mask: np.ndarray, min_blob_ratio: float = 0.02, morph_radius: int = 2) -> np.ndarray:
    """Remove speckle, fill interior holes, and soften the boundary.

    Interior hole filling matters more than it sounds: unfilled highlights or
    logo shapes become tunnels through the reconstructed body.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    if morph_radius > 0:
        mask = ndops.binary_close(mask, morph_radius)
        mask = ndops.binary_open(mask, max(1, morph_radius - 1))
        if not mask.any():
            mask = np.asarray(mask).astype(bool)
    labels, sizes = ndops.component_sizes(mask, connectivity=2)
    if sizes.size:
        biggest = int(sizes.max())
        keep = np.flatnonzero(sizes >= max(4, int(min_blob_ratio * biggest)))
        mask = np.isin(labels, keep)
    return ndops.binary_fill_holes(mask)


def soft_alpha(mask: np.ndarray, feather: float = 1.2) -> np.ndarray:
    """Anti-aliased alpha from a binary mask, for debug composites."""
    sdf = ndops.signed_distance(mask)
    return np.clip(0.5 + sdf / max(2.0 * feather, 1e-6), 0.0, 1.0)
