"""Split a character turnaround sheet into individual orthographic views.

Production concept art usually arrives as one image containing a front view, a
side view and often a back view side by side. Detecting and labelling those
panels is what lets the reconstruction stage carve a real visual hull instead of
inflating a single silhouette.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conceptforge import ndops
from conceptforge.imaging.raster import mask_bounds


@dataclass
class Panel:
    """One detected view inside a concept sheet."""

    rgba: np.ndarray
    mask: np.ndarray
    x0: int
    y0: int
    x1: int
    y1: int
    symmetry: float
    """0..1 left/right mirror agreement of the silhouette (1 = perfect)."""

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def aspect(self) -> float:
        return self.silhouette_width / max(self.silhouette_height, 1)

    @property
    def silhouette_width(self) -> int:
        x0, _, x1, _ = mask_bounds(self.mask)
        return x1 - x0

    @property
    def silhouette_height(self) -> int:
        _, y0, _, y1 = mask_bounds(self.mask)
        return y1 - y0


def symmetry_score(mask: np.ndarray) -> float:
    """Intersection-over-union of the silhouette with its mirror image.

    The mirror is taken about the silhouette's area centroid rather than the
    bounding-box centre, which is far more stable for characters holding a prop
    on one side.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return 0.0
    x0, y0, x1, y1 = mask_bounds(mask)
    crop = mask[y0:y1, x0:x1]
    cols = np.flatnonzero(crop.any(axis=0))
    centroid = float(np.average(np.arange(crop.shape[1]), weights=crop.sum(axis=0)))
    shift = int(round(2.0 * centroid - (crop.shape[1] - 1)))
    mirrored = crop[:, ::-1]
    if shift:
        mirrored = np.roll(mirrored, shift, axis=1)
        if shift > 0:
            mirrored[:, :shift] = False
        else:
            mirrored[:, shift:] = False
    del cols
    union = float((crop | mirrored).sum())
    if union <= 0:
        return 0.0
    return float((crop & mirrored).sum()) / union


def split_panels(
    rgba: np.ndarray,
    mask: np.ndarray,
    min_gap_ratio: float = 0.012,
    min_height_ratio: float = 0.45,
    max_panels: int = 6,
) -> list[Panel]:
    """Segment a sheet on vertical whitespace gutters.

    Returns panels left to right. A single-figure image yields one panel, which
    is the normal single-view path.
    """
    mask = np.asarray(mask).astype(bool)
    rgba = np.asarray(rgba, dtype=np.float64)
    if not mask.any():
        raise ValueError("cannot split an empty silhouette")

    occupied = mask.any(axis=0)
    min_gap = max(2, int(round(min_gap_ratio * mask.shape[1])))
    spans = _runs(occupied, min_gap=min_gap)
    if not spans:
        spans = [(0, mask.shape[1])]

    heights = []
    for x0, x1 in spans:
        sub = mask[:, x0:x1]
        _, y0, _, y1 = mask_bounds(sub)
        heights.append(y1 - y0)
    tallest = max(heights) if heights else 1

    panels: list[Panel] = []
    for (x0, x1), height in zip(spans, heights):
        if height < min_height_ratio * tallest:
            continue  # callouts, prop studies, text blocks
        sub_mask = mask[:, x0:x1]
        sub_mask = ndops.largest_component(sub_mask)
        _, y0, _, y1 = mask_bounds(sub_mask)
        panels.append(
            Panel(
                rgba=np.ascontiguousarray(rgba[:, x0:x1]),
                mask=np.ascontiguousarray(sub_mask),
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                symmetry=symmetry_score(sub_mask),
            )
        )
    if len(panels) > max_panels:
        panels.sort(key=lambda p: p.silhouette_height * p.silhouette_width, reverse=True)
        panels = sorted(panels[:max_panels], key=lambda p: p.x0)
    return panels


def _runs(occupied: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Contiguous occupied spans, merging gaps narrower than ``min_gap``."""
    occupied = np.asarray(occupied).astype(bool)
    if not occupied.any():
        return []
    padded = np.concatenate([[False], occupied, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    spans: list[tuple[int, int]] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if spans and s - spans[-1][1] < min_gap:
            spans[-1] = (spans[-1][0], e)
        else:
            spans.append((s, e))
    return spans


def classify_panels(panels: list[Panel], panel_order: list[str] | None = None) -> dict[str, Panel]:
    """Label panels as ``front``, ``side`` and ``back``.

    Heuristics, in order of reliability:

    * A side view is markedly less left/right symmetric and narrower than a
      front or back view of the same character.
    * Turnarounds are conventionally ordered front, (three-quarter), side, back
      from left to right, so among the symmetric panels the leftmost is front.

    ``panel_order`` overrides all of it when the artist's layout is unusual.
    """
    if not panels:
        return {}
    if panel_order:
        names = [str(n).lower() for n in panel_order]
        return {
            name: panel
            for name, panel in zip(names, panels)
            if name in ("front", "side", "back")
        }
    if len(panels) == 1:
        return {"front": panels[0]}

    scores = np.array([p.symmetry for p in panels])
    aspects = np.array([p.aspect for p in panels])
    # Normalised "side-ness": asymmetric and narrow.
    narrowness = 1.0 - _rank01(aspects)
    sideness = 0.65 * (1.0 - _rank01(scores)) + 0.35 * narrowness

    side_index = int(np.argmax(sideness))
    remaining = [i for i in range(len(panels)) if i != side_index]
    views: dict[str, Panel] = {"side": panels[side_index]}
    if remaining:
        views["front"] = panels[remaining[0]]
    if len(remaining) > 1:
        # The back view is the symmetric panel furthest from the front panel.
        views["back"] = panels[remaining[-1]]
    return views


def profile_faces_right(mask: np.ndarray) -> bool:
    """Guess which way a profile view is facing.

    ConceptForge builds characters facing ``+Z``, and the side view supplies the
    depth axis, so getting this backwards would reconstruct the character
    inside-out. Two silhouette cues vote:

    * the feet, because toes point forwards and heels do not;
    * the chest relative to the pelvis, because a chest is carried in front of
      the hips in essentially every character design.

    The foot cue is weighted higher: it is a larger, more consistent offset than
    the torso lean, and it survives capes and backpacks.
    """
    mask = np.asarray(mask).astype(bool)
    _, y0, _, y1 = mask_bounds(mask)
    if not mask.any() or y1 - y0 < 8:
        return True
    height = float(y1 - y0)
    columns = np.arange(mask.shape[1], dtype=np.float64)

    def band_centroid(lo_fraction: float, hi_fraction: float) -> float | None:
        lo = int(y0 + lo_fraction * height)
        hi = int(y0 + hi_fraction * height)
        weights = mask[max(0, lo) : max(lo + 1, hi)].sum(axis=0).astype(np.float64)
        if weights.sum() <= 0:
            return None
        return float(np.average(columns, weights=weights))

    feet = band_centroid(0.94, 1.0)
    lower_leg = band_centroid(0.80, 0.92)
    chest = band_centroid(0.20, 0.38)
    pelvis = band_centroid(0.45, 0.60)

    score = 0.0
    if feet is not None and lower_leg is not None:
        score += 2.0 * (feet - lower_leg) / height
    if chest is not None and pelvis is not None:
        score += 1.0 * (chest - pelvis) / height
    return score >= 0.0


def _rank01(values: np.ndarray) -> np.ndarray:
    """Map values to 0..1 by rank, robust to outliers and equal values."""
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values)
    order = np.argsort(np.argsort(values))
    return order / float(values.size - 1)
