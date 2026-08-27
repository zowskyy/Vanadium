"""Assemble concept art into a set of aligned orthographic views.

The reconstruction stage needs every view to agree on scale and on where the
ground and the centre axis are; a front view 900 px tall and a side view 870 px
tall would otherwise carve a lopsided hull. This module does that normalisation
and hands back a :class:`ConceptSheet`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from conceptforge.config import ImagingConfig
from conceptforge.imaging import landmarks as lm
from conceptforge.imaging.matting import MatteResult, clean_mask, extract_matte
from conceptforge.imaging.raster import fit_within, load_rgba, mask_bounds, resize_rgba, to_uint8
from conceptforge.imaging.sheet import Panel, classify_panels, split_panels
from conceptforge.reporting import NULL_REPORTER, Reporter

VIEW_ORDER = ("front", "side", "back")


@dataclass
class ConceptView:
    """One orthographic view, resampled into the shared alignment frame."""

    name: str
    rgba: np.ndarray
    mask: np.ndarray
    #: Column of the character's centre axis inside this view's canvas.
    pivot_x: float
    #: Rows of the top of the head and the ground plane inside the canvas.
    top_y: float
    bottom_y: float

    @property
    def size(self) -> tuple[int, int]:
        return self.mask.shape[1], self.mask.shape[0]

    def silhouette_bounds(self) -> tuple[int, int, int, int]:
        return mask_bounds(self.mask)


@dataclass
class ConceptSheet:
    """Everything the geometry stage needs to know about the artwork."""

    views: dict[str, ConceptView]
    landmarks: lm.CharacterLandmarks
    canvas_size: tuple[int, int]
    matte_source: str
    background_color: np.ndarray
    panel_count: int
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def front(self) -> ConceptView:
        return self.views["front"]

    @property
    def side(self) -> ConceptView | None:
        return self.views.get("side")

    @property
    def back(self) -> ConceptView | None:
        return self.views.get("back")

    @property
    def view_names(self) -> list[str]:
        return [name for name in VIEW_ORDER if name in self.views]

    def debug_images(self) -> dict[str, np.ndarray]:
        """Composites written out when ``export.write_debug`` is on."""
        out: dict[str, np.ndarray] = {}
        for name, view in self.views.items():
            rgb = view.rgba[..., :3] * view.mask[..., None]
            checker = _checkerboard(view.mask.shape) * (1.0 - view.mask[..., None])
            out[f"view_{name}"] = np.clip(rgb + checker, 0.0, 1.0)
            out[f"mask_{name}"] = view.mask.astype(np.float64)
        out["landmarks_front"] = draw_landmarks(self.front, self.landmarks)
        return out


def analyze_artwork(
    source: str | Path | np.ndarray,
    config: ImagingConfig | None = None,
    reporter: Reporter | None = None,
) -> ConceptSheet:
    """Load concept art and turn it into aligned, labelled views."""
    config = config or ImagingConfig()
    reporter = reporter or NULL_REPORTER

    rgba = load_rgba(source) if not isinstance(source, np.ndarray) else np.asarray(source, dtype=np.float32)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("concept art must be an RGBA image")
    target = fit_within(rgba.shape[:2], config.working_resolution)
    rgba = resize_rgba(rgba, target)

    matte: MatteResult = extract_matte(
        rgba,
        alpha_threshold=config.alpha_threshold,
        background_tolerance=config.background_tolerance,
        min_blob_ratio=config.min_blob_ratio,
        morph_radius=config.morph_radius,
    )
    if not matte.mask.any():
        raise ValueError(
            "no character found in the artwork: the silhouette came out empty. "
            "Try artwork on a plainer background, or supply a PNG with alpha."
        )

    panels = (
        split_panels(rgba, matte.mask)
        if config.auto_split_sheet
        else [_whole_image_panel(rgba, matte.mask)]
    )
    if not panels:
        panels = [_whole_image_panel(rgba, matte.mask)]
    labelled = classify_panels(panels, list(config.panel_order) if config.panel_order else None)
    if "front" not in labelled:
        labelled["front"] = panels[0]

    views, canvas_size = _align_views(labelled, config)
    front_landmarks = lm.detect_landmarks(views["front"].mask)

    stats = {
        "matte": matte.source,
        "coverage": round(float(matte.coverage), 4),
        "panels": len(panels),
        "views": ",".join(name for name in VIEW_ORDER if name in views),
        "landmark_confidence": round(front_landmarks.overall_confidence, 3),
    }
    for note in front_landmarks.notes:
        reporter.warn(note)

    return ConceptSheet(
        views=views,
        landmarks=front_landmarks,
        canvas_size=canvas_size,
        matte_source=matte.source,
        background_color=matte.background_color,
        panel_count=len(panels),
        stats=stats,
    )


def _whole_image_panel(rgba: np.ndarray, mask: np.ndarray) -> Panel:
    from conceptforge.imaging.sheet import symmetry_score

    _, y0, _, y1 = mask_bounds(mask)
    return Panel(
        rgba=rgba,
        mask=mask,
        x0=0,
        y0=int(y0),
        x1=int(rgba.shape[1]),
        y1=int(y1),
        symmetry=symmetry_score(mask),
    )


def _align_views(
    labelled: Mapping[str, Panel], config: ImagingConfig
) -> tuple[dict[str, ConceptView], tuple[int, int]]:
    """Resample every panel so the figure has one shared height and centre.

    The canvas is padded by 6% so that later mesh smoothing and texture
    dilation never run off the edge of the data.
    """
    front = labelled["front"]
    reference_height = max(1, front.silhouette_height)
    scaled: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name in VIEW_ORDER:
        panel = labelled.get(name)
        if panel is None:
            continue
        crop_rgba, crop_mask = _crop_to_silhouette(panel)
        if crop_mask.shape[0] < 2 or crop_mask.shape[1] < 2:
            continue
        scale = reference_height / float(crop_mask.shape[0])
        new_size = (max(2, int(round(crop_mask.shape[1] * scale))), reference_height)
        rgba_s = resize_rgba(crop_rgba, new_size)
        mask_s = _resize_mask_soft(crop_mask, new_size)
        mask_s = clean_mask(mask_s, min_blob_ratio=config.min_blob_ratio, morph_radius=1)
        if not mask_s.any():
            continue
        axis = lm.symmetry_axis(mask_s) if name in ("front", "back") else _bbox_center_x(mask_s)
        scaled[name] = (rgba_s, mask_s, axis)

    if "front" not in scaled:
        raise ValueError("front view could not be prepared from the artwork")

    pad_y = max(4, int(round(0.03 * reference_height)))
    canvas_h = reference_height + 2 * pad_y
    half_width = 0
    for _, mask_s, axis in scaled.values():
        half_width = max(half_width, int(np.ceil(max(axis, mask_s.shape[1] - axis))))
    canvas_w = 2 * (half_width + max(4, int(round(0.03 * reference_height)))) + 1

    views: dict[str, ConceptView] = {}
    for name, (rgba_s, mask_s, axis) in scaled.items():
        offset_x = int(round(canvas_w * 0.5 - axis))
        canvas_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
        canvas_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        _paste(canvas_rgba, rgba_s, offset_x, pad_y)
        _paste(canvas_mask, mask_s, offset_x, pad_y)
        pivot = canvas_w * 0.5
        if name in ("front", "back") and config.symmetrize > 0.0:
            canvas_mask = lm.symmetrize(canvas_mask, pivot, config.symmetrize)
            canvas_mask = clean_mask(canvas_mask, min_blob_ratio=config.min_blob_ratio, morph_radius=1)
        views[name] = ConceptView(
            name=name,
            rgba=canvas_rgba,
            mask=canvas_mask,
            pivot_x=float(pivot),
            top_y=float(pad_y),
            bottom_y=float(pad_y + reference_height),
        )
    return views, (canvas_w, canvas_h)


def _crop_to_silhouette(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = mask_bounds(panel.mask)
    return (
        np.ascontiguousarray(panel.rgba[y0:y1, x0:x1]),
        np.ascontiguousarray(panel.mask[y0:y1, x0:x1]),
    )


def _resize_mask_soft(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Area-averaged mask resize; keeps thin fingers alive better than nearest."""
    img = Image.fromarray(to_uint8(mask.astype(np.float64)), mode="L")
    resized = np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    return resized >= 0.45


def _bbox_center_x(mask: np.ndarray) -> float:
    x0, _, x1, _ = mask_bounds(mask)
    return 0.5 * (x0 + x1)


def _paste(canvas: np.ndarray, patch: np.ndarray, offset_x: int, offset_y: int) -> None:
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    sx0 = max(0, -offset_x)
    sy0 = max(0, -offset_y)
    dx0 = max(0, offset_x)
    dy0 = max(0, offset_y)
    w = min(pw - sx0, cw - dx0)
    h = min(ph - sy0, ch - dy0)
    if w <= 0 or h <= 0:
        return
    canvas[dy0 : dy0 + h, dx0 : dx0 + w] = patch[sy0 : sy0 + h, sx0 : sx0 + w]


def _checkerboard(shape: tuple[int, int], cell: int = 16) -> np.ndarray:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    pattern = (((xs // cell) + (ys // cell)) % 2).astype(np.float64)
    return (0.22 + 0.10 * pattern)[..., None] * np.ones(3)


def draw_landmarks(view: ConceptView, marks: lm.CharacterLandmarks) -> np.ndarray:
    """Debug overlay: silhouette plus the detected skeleton landmarks."""
    base = view.mask.astype(np.float64)[..., None] * np.array([0.16, 0.18, 0.22])
    base += (1.0 - view.mask.astype(np.float64))[..., None] * np.array([0.05, 0.05, 0.06])
    chains = [
        ([marks.head_center, np.array([marks.center_x, marks.chin_y]),
          np.array([marks.center_x, marks.shoulder_y]),
          np.array([marks.center_x, marks.waist_y]),
          np.array([marks.center_x, marks.hip_y])], np.array([1.0, 0.85, 0.2])),
        ([marks.shoulder_l, marks.elbow_l, marks.hand_l], np.array([0.2, 0.9, 1.0])),
        ([marks.shoulder_r, marks.elbow_r, marks.hand_r], np.array([1.0, 0.35, 0.5])),
        ([marks.hip_l, marks.knee_l, marks.ankle_l, marks.toe_l], np.array([0.35, 1.0, 0.45])),
        ([marks.hip_r, marks.knee_r, marks.ankle_r, marks.toe_r], np.array([1.0, 0.6, 0.15])),
    ]
    for points, color in chains:
        for a, b in zip(points[:-1], points[1:]):
            _draw_line(base, a, b, color)
        for p in points:
            _draw_dot(base, p, color)
    return np.clip(base, 0.0, 1.0)


def _draw_line(image: np.ndarray, a: np.ndarray, b: np.ndarray, color: np.ndarray) -> None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    steps = max(2, int(np.linalg.norm(b - a)))
    ts = np.linspace(0.0, 1.0, steps)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, image.shape[1] - 1)
    ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, image.shape[0] - 1)
    image[ys, xs] = color


def _draw_dot(image: np.ndarray, p: np.ndarray, color: np.ndarray, radius: int = 3) -> None:
    x, y = int(round(float(p[0]))), int(round(float(p[1])))
    y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
    image[y0:y1, x0:x1] = np.minimum(1.0, color * 1.0)
