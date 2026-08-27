"""Build a 3D implicit solid from 2D orthographic views.

This is the step that actually invents the third dimension. Two mechanisms are
combined:

**Visual hull.** Each orthographic view is a constraint: a point is inside the
character only if it projects inside every silhouette. Extruding each view's
signed distance field along its unseen axis and intersecting them gives the
visual hull. Front and back views constrain X/Y; a side view constrains Z/Y.

**Silhouette inflation.** A hull built from a handful of axis-aligned views has
a characteristic failure: at a given height, every part of the body is granted
the full depth of the deepest part, so arms come out as wide slabs. Inflation
fixes it by bounding the depth at each point by how far that point is from the
front silhouette's outline - thin features get thin depth, the torso stays full.
Inflation is also the *only* mechanism available when the artwork has a single
view, where it reproduces the classic "Teddy" inflation result.

The two are intersected with a polynomial smooth minimum rather than a hard
``min`` so shoulders, hips and armpits come out rounded instead of creased, and
the result is filtered before surfacing so the source image's pixel grid does
not print staircases onto the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from conceptforge import ndops
from conceptforge.config import GeometryConfig
from conceptforge.imaging.views import ConceptSheet, ConceptView
from conceptforge.ndops import bilinear_sample
from conceptforge.reporting import NULL_REPORTER, Reporter


@dataclass
class VoxelField:
    """A scalar field on a regular grid, positive inside the character.

    Coordinates are *height units*: the character is 1.0 tall, standing with
    its feet at ``y = 0``, centred on ``x = 0``, facing ``+z``.
    """

    values: np.ndarray
    origin: np.ndarray
    spacing: float
    stats: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape  # type: ignore[return-value]

    def occupancy(self, level: float = 0.0) -> np.ndarray:
        return self.values > level

    def world_of_index(self, index: np.ndarray) -> np.ndarray:
        return self.origin[None, :] + np.asarray(index, dtype=np.float64) * self.spacing

    def index_of_world(self, world: np.ndarray) -> np.ndarray:
        return (np.asarray(world, dtype=np.float64) - self.origin[None, :]) / self.spacing

    def sample(self, world: np.ndarray) -> np.ndarray:
        return ndops.trilinear_sample(self.values, self.index_of_world(world))

    def slice_previews(self) -> dict[str, np.ndarray]:
        """Mid-plane slices, written out as debug images."""
        nx, ny, nz = self.shape
        return {
            "field_xy": ndops.normalized(self.values[:, :, nz // 2].T[::-1]),
            "field_zy": ndops.normalized(self.values[nx // 2, :, :].T[::-1]),
            "field_xz": ndops.normalized(self.values[:, ny // 2, :].T),
        }


def build_field(
    sheet: ConceptSheet,
    config: GeometryConfig | None = None,
    reporter: Reporter | None = None,
) -> VoxelField:
    """Reconstruct a volumetric solid from the aligned concept views."""
    config = config or GeometryConfig()
    reporter = reporter or NULL_REPORTER

    front = sheet.front
    height_px = float(front.bottom_y - front.top_y)
    if height_px < 8:
        raise ValueError("the character is too small in the artwork to reconstruct")

    spacing = 1.0 / float(max(16, config.voxel_height))
    margin = 4.0 * spacing

    x_half = _lateral_extent(front) + margin
    if sheet.back is not None:
        x_half = max(x_half, _lateral_extent(sheet.back) + margin)
    if sheet.side is not None:
        z_half = _lateral_extent(sheet.side) + margin
    else:
        z_half = 0.5 * config.depth_ratio + margin

    nx = _odd_count(x_half, spacing)
    nz = _odd_count(z_half, spacing)
    ny = int(np.ceil((1.0 + 2.0 * margin) / spacing)) + 1
    origin = np.array([-0.5 * (nx - 1) * spacing, -margin, -0.5 * (nz - 1) * spacing])

    xs = origin[0] + np.arange(nx) * spacing
    ys = origin[1] + np.arange(ny) * spacing
    zs = origin[2] + np.arange(nz) * spacing

    # -- view constraints -------------------------------------------------- #
    front_sdf = _view_distance_field(front, xs, ys, height_px)          # (nx, ny)
    field = front_sdf[:, :, None]

    if sheet.back is not None:
        # A back view constrains the same axes as the front, mirrored in x.
        back_sdf = _view_distance_field(sheet.back, -xs, ys, height_px)
        field = np.minimum(field, back_sdf[:, :, None])
        front_sdf = np.minimum(front_sdf, back_sdf)

    blend = config.blend_radius * spacing
    used_side = False
    depth_budget = None
    if sheet.side is not None:
        side_sdf = _view_distance_field(sheet.side, zs, ys, height_px)    # (nz, ny)
        field = ndops.smooth_min(field, side_sdf.T[None, :, :], blend)
        depth_budget = _row_depth_budget(side_sdf, zs, spacing)
        used_side = True

    inflation = _inflation_field(front_sdf, zs, depth_budget, config, spacing)
    field = ndops.smooth_min(field, inflation, blend)

    # -- conditioning ------------------------------------------------------ #
    if config.field_smoothing > 0:
        field = ndops.gaussian_blur(field, config.field_smoothing)

    # Guarantee the surface closes: the outermost shell must read as outside.
    field = _seal_border(field)

    occupied = field > 0.0
    if not occupied.any():
        raise ValueError("volumetric reconstruction produced an empty solid")
    main = ndops.largest_component(occupied, connectivity=1)
    dropped = int(occupied.sum() - main.sum())
    if dropped > 0:
        # Suppress detached blobs (props drawn beside the figure, matte specks)
        # rather than letting them surface as floating geometry.
        field = np.where(occupied & ~main, -np.abs(field), field)

    stats = {
        "grid": f"{nx}x{ny}x{nz}",
        "voxels": int(nx * ny * nz),
        "views_used": ("front" if not used_side else "front+side")
        + ("+back" if sheet.back is not None else ""),
        "fill_ratio": round(float(main.mean()), 4),
        "detached_voxels_dropped": dropped,
        "depth_source": "side view" if used_side else f"inflation({config.inflation_power})",
    }
    reporter.info(f"reconstructed {stats['grid']} field from {stats['views_used']}")
    return VoxelField(values=field, origin=origin, spacing=spacing, stats=stats)


def _odd_count(half_extent: float, spacing: float) -> int:
    """Cell count spanning ``+/-half_extent``, forced odd so 0 is a sample."""
    n = int(np.ceil(half_extent / spacing))
    return 2 * max(2, n) + 1


def _lateral_extent(view: ConceptView) -> float:
    """Largest horizontal distance from the pivot to the silhouette, in heights."""
    mask = view.mask
    if not mask.any():
        return 0.1
    columns = np.flatnonzero(mask.any(axis=0))
    height_px = max(float(view.bottom_y - view.top_y), 1.0)
    left = view.pivot_x - float(columns[0])
    right = float(columns[-1]) - view.pivot_x
    return float(max(left, right, 1.0) / height_px)


def _view_distance_field(
    view: ConceptView, lateral: np.ndarray, vertical: np.ndarray, height_px: float
) -> np.ndarray:
    """Sample a view's silhouette signed distance on a world-space grid.

    ``lateral`` are world coordinates along the view's horizontal axis and
    ``vertical`` along Y. Returns a ``(len(lateral), len(vertical))`` array of
    signed distances in height units, positive inside the silhouette.
    """
    sdf_px = ndops.signed_distance(view.mask)
    columns = view.pivot_x + np.asarray(lateral, dtype=np.float64) * height_px
    rows = view.bottom_y - np.asarray(vertical, dtype=np.float64) * height_px
    grid_c, grid_r = np.meshgrid(columns, rows, indexing="ij")
    samples = bilinear_sample(sdf_px, np.stack([grid_c.ravel(), grid_r.ravel()], axis=1))
    field = samples.reshape(grid_c.shape) / height_px

    # bilinear_sample clamps to the image edge, which would smear the silhouette
    # outwards forever. Anything off-canvas is definitively outside.
    outside = (
        (grid_c < 0)
        | (grid_c > view.mask.shape[1] - 1)
        | (grid_r < 0)
        | (grid_r > view.mask.shape[0] - 1)
    )
    return np.where(outside, -np.abs(field) - 1e-3, field)


def _row_depth_budget(side_sdf: np.ndarray, zs: np.ndarray, spacing: float) -> np.ndarray:
    """How deep the character is at each height, read from the side view.

    Returns half-depths per Y row, in height units.
    """
    inside = side_sdf > 0.0
    magnitudes = np.abs(zs)[:, None]
    budget = np.where(inside.any(axis=0), (inside * magnitudes).max(axis=0), 0.0)
    # A voxel of slack so the inflation bound never clips the hull it refines.
    return _smooth_rows(budget, 1.5) + spacing


def _inflation_field(
    front_sdf: np.ndarray,
    zs: np.ndarray,
    depth_budget: np.ndarray | None,
    config: GeometryConfig,
    spacing: float,
) -> np.ndarray:
    """Depth bound derived from the front silhouette's interior distance.

    ``half_depth(x, y) = D(y) * (d(x, y) / d_max(y)) ** power`` where ``d`` is
    the distance from ``(x, y)`` to the silhouette outline and ``D(y)`` is the
    depth available at that height.

    Both the distance normalisation and the depth budget are taken **per row**,
    which is the detail that makes this useful alongside a visual hull. A single
    global normalisation would compare the head's interior distance against the
    torso's and conclude the head must be shallow; per row, the head is compared
    only against itself, so it keeps the depth the side view grants it while the
    arms are still thinned relative to the torso beside them.

    The exponent shapes the cross-section: below 1 gives fuller, rounder bodies,
    above 1 gives flatter slabs with sharper silhouette edges.
    """
    interior = np.maximum(front_sdf, 0.0)
    row_max = _smooth_rows(interior.max(axis=0), 1.5)
    global_max = float(row_max.max())
    if global_max <= 1e-9:
        return np.full((front_sdf.shape[0], front_sdf.shape[1], zs.size), -1.0)

    if depth_budget is None:
        # Without a side view there is nothing to measure depth against, so it
        # is predicted from how wide the character is at each height.
        budget = 0.5 * config.depth_ratio * np.power(row_max / global_max, 0.65) + spacing
    else:
        budget = depth_budget

    normalised = interior / np.maximum(row_max[None, :], 1e-9)
    half_depth = budget[None, :] * np.power(
        np.clip(normalised, 0.0, 1.0), max(config.inflation_power, 1e-3)
    )
    return half_depth[:, :, None] - np.abs(zs)[None, None, :]


def _smooth_rows(values: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a per-row profile so depth does not band between voxel rows."""
    if sigma <= 0.3 or values.size < 3:
        return np.asarray(values, dtype=np.float64)
    kernel = ndops.gaussian_kernel(sigma)
    padded = np.pad(np.asarray(values, dtype=np.float64), (kernel.size // 2,) * 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _seal_border(field: np.ndarray) -> np.ndarray:
    """Force the outer shell of the grid negative so the isosurface closes.

    Without this, a character whose silhouette touches the canvas edge (a
    tightly cropped drawing) would produce an open, non-manifold mesh.
    """
    out = field.copy()
    limit = -1e-3
    for axis in range(3):
        for index in (0, out.shape[axis] - 1):
            sl: list[slice | int] = [slice(None)] * 3
            sl[axis] = index
            out[tuple(sl)] = np.minimum(out[tuple(sl)], limit)
    return out
