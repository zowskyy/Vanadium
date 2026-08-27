"""Drive the geometry stage: views in, production-ready mesh out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conceptforge.config import GeometryConfig
from conceptforge.geometry.decimate import build_lod_chain, decimate
from conceptforge.geometry.marching import marching_tetrahedra
from conceptforge.geometry.mesh import Mesh
from conceptforge.geometry.smoothing import taubin_smooth
from conceptforge.geometry.uv import planar_front_back, uv_area_distortion
from conceptforge.geometry.volume import VoxelField, build_field
from conceptforge.imaging.views import ConceptSheet
from conceptforge.reporting import NULL_REPORTER, Reporter


@dataclass
class GeometryResult:
    """The reconstructed character surface and everything derived alongside it."""

    mesh: Mesh
    """Base mesh, UV-mapped, in metres with feet on Y=0 and facing +Z."""

    field: VoxelField
    """The implicit solid, retained for skinning and joint depth lookup."""

    lods: list[Mesh] = field(default_factory=list)
    scale: float = 1.0
    """Height units -> world units uniform factor applied to the mesh."""

    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    """4x4 matrix taking voxel-field (height unit) space into mesh world space.

    Anything that needs to land in the same space as the mesh - the skeleton,
    joint depth probes, texture projection - goes through this.
    """

    stats: dict[str, Any] = field(default_factory=dict)


def build_geometry(
    sheet: ConceptSheet,
    config: GeometryConfig | None = None,
    reporter: Reporter | None = None,
) -> GeometryResult:
    """Reconstruct, condition and parameterise the character surface."""
    config = config or GeometryConfig()
    reporter = reporter or NULL_REPORTER

    voxels = build_field(sheet, config, reporter)
    mesh = marching_tetrahedra(
        voxels.values, level=0.0, origin=voxels.origin, spacing=voxels.spacing
    )
    if mesh.face_count == 0:
        raise ValueError("surface extraction produced no geometry")
    raw_faces = mesh.face_count

    mesh.keep_largest_component()
    if config.smooth_iterations > 0:
        taubin_smooth(mesh, config.smooth_iterations, config.smooth_lambda, config.smooth_mu)

    smoothed_faces = mesh.face_count
    watertight = mesh.is_watertight()
    if config.target_triangles and mesh.face_count > config.target_triangles:
        mesh = decimate(mesh, config.target_triangles)
        # A light second fairing pass removes the faceting that collapsing
        # long, thin triangles can leave behind.
        taubin_smooth(mesh, max(2, config.smooth_iterations // 4), config.smooth_lambda, config.smooth_mu)

    mesh.compute_normals()
    transform = _normalise_scale(mesh, config.character_height)
    scale = float(transform[0, 0])
    mesh.compute_normals()
    # UV assignment splits vertices along the front/back seam, so watertightness
    # is measured on the solid before that point.
    watertight_after_decimation = mesh.is_watertight()
    mesh = planar_front_back(mesh)

    lods = build_lod_chain(mesh, list(config.lod_ratios)) if config.lod_ratios else []

    stats: dict[str, Any] = {
        "raw_triangles": raw_faces,
        "smoothed_triangles": smoothed_faces,
        "triangles": mesh.face_count,
        "vertices": mesh.vertex_count,
        "watertight_extracted": watertight,
        "watertight_decimated": watertight_after_decimation,
        "lods": [m.face_count for m in lods],
        "height_m": round(float(mesh.size()[1]), 4),
        "depth_m": round(float(mesh.size()[2]), 4),
        "width_m": round(float(mesh.size()[0]), 4),
    }
    stats.update(voxels.stats)
    stats.update(uv_area_distortion(mesh))
    return GeometryResult(
        mesh=mesh, field=voxels, lods=lods, scale=scale, transform=transform, stats=stats
    )


def _normalise_scale(mesh: Mesh, target_height: float) -> np.ndarray:
    """Scale the mesh to real-world height, feet on the ground, centred in X/Z.

    Engines and DCCs assume a character stands on the origin at true scale; a
    model that arrives 1.0 units tall and centred on its own middle costs an
    artist a manual fix on every import. Returns the transform applied.
    """
    lo, hi = mesh.bounds()
    current = float(hi[1] - lo[1])
    if current < 1e-9:
        return np.eye(4)
    factor = float(target_height) / current
    origin = np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])
    mesh.vertices = (mesh.vertices - origin) * factor
    transform = np.eye(4)
    transform[0, 0] = transform[1, 1] = transform[2, 2] = factor
    transform[:3, 3] = -origin * factor
    return transform
