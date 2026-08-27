"""UV parameterisation.

The default layout is a front/back planar projection: the atlas' left half holds
everything facing the camera in the front view, the right half holds the back.
It is chosen deliberately over a conformal unwrap because it is *exactly* the
projection the concept art was drawn in, so texture baking becomes a
pixel-for-pixel transfer of the artist's work rather than a resampling with
invented seams.

The known cost is stretching on surfaces that face sideways, where a planar
projection has little area to work with. :func:`cylindrical` is provided for
cases that prefer even area distribution over fidelity to the source views.
"""

from __future__ import annotations

import numpy as np

from conceptforge.geometry.mesh import Mesh

#: Which half of the atlas a surface belongs to.
FRONT, BACK = 0, 1


def planar_front_back(mesh: Mesh, padding: float = 0.008) -> Mesh:
    """Assign UVs by projecting front-facing and back-facing surfaces.

    Vertices on the silhouette boundary are duplicated so the two halves can
    carry independent UVs. Returns a new mesh; positions and topology are
    otherwise unchanged.
    """
    if mesh.face_count == 0:
        result = mesh.copy()
        result.uvs = np.zeros((mesh.vertex_count, 2))
        return result

    normals = mesh.face_normals()
    face_side = np.where(normals[:, 2] >= 0.0, FRONT, BACK)

    # Split vertices per (vertex, side) so the seam can be discontinuous.
    corner_sides = np.repeat(face_side[:, None], 3, axis=1)
    keys = mesh.faces.astype(np.int64) * 2 + corner_sides
    unique_keys, inverse = np.unique(keys.ravel(), return_inverse=True)
    faces = inverse.astype(np.int64).reshape(-1, 3)
    source_vertex = unique_keys // 2
    side = (unique_keys % 2).astype(np.int64)

    vertices = mesh.vertices[source_vertex]
    vertex_normals = None
    if mesh.normals is not None:
        vertex_normals = mesh.normals[source_vertex]
    colors = None if mesh.colors is None else mesh.colors[source_vertex]

    lo, hi = mesh.bounds()
    span_x = max(float(hi[0] - lo[0]), 1e-9)
    span_y = max(float(hi[1] - lo[1]), 1e-9)
    u_norm = (vertices[:, 0] - lo[0]) / span_x
    v_norm = (vertices[:, 1] - lo[1]) / span_y

    pad = float(np.clip(padding, 0.0, 0.2))
    half_width = 0.5 - 2.0 * pad
    u = np.where(side == FRONT, pad + u_norm * half_width, 1.0 - pad - u_norm * half_width)
    v = pad + (1.0 - v_norm) * (1.0 - 2.0 * pad)

    result = Mesh(
        vertices=vertices,
        faces=faces,
        normals=vertex_normals,
        uvs=np.stack([u, v], axis=1),
        colors=colors,
        metadata=dict(mesh.metadata),
    )
    result.metadata["uv_layout"] = "planar_front_back"
    result.metadata["uv_side"] = side
    if vertex_normals is None:
        result.compute_normals()
    return result


def cylindrical(mesh: Mesh, padding: float = 0.004, seam_angle: float = np.pi) -> Mesh:
    """Wrap UVs around the Y axis.

    Distributes area more evenly than a planar projection but introduces a
    vertical seam and pinches at the poles.
    """
    if mesh.face_count == 0:
        result = mesh.copy()
        result.uvs = np.zeros((mesh.vertex_count, 2))
        return result

    centre = mesh.centroid()
    offsets = mesh.vertices - centre
    angle = np.arctan2(offsets[:, 0], offsets[:, 2])
    wrapped = np.mod(angle - seam_angle, 2.0 * np.pi) / (2.0 * np.pi)

    # Duplicate vertices on faces that straddle the seam, otherwise a triangle
    # would smear the whole texture across itself.
    faces = mesh.faces.copy()
    uvs_u = wrapped.copy()
    vertices = mesh.vertices.copy()
    extra_positions: list[np.ndarray] = []
    extra_u: list[float] = []
    for f in range(faces.shape[0]):
        corners = faces[f]
        us = uvs_u[corners]
        if us.max() - us.min() <= 0.5:
            continue
        for slot, corner in enumerate(corners):
            if uvs_u[corner] < 0.5:
                extra_positions.append(vertices[corner])
                extra_u.append(float(uvs_u[corner]) + 1.0)
                faces[f, slot] = vertices.shape[0] + len(extra_positions) - 1
    if extra_positions:
        vertices = np.concatenate([vertices, np.stack(extra_positions)], axis=0)
        uvs_u = np.concatenate([uvs_u, np.asarray(extra_u)], axis=0)

    lo, hi = mesh.bounds()
    span_y = max(float(hi[1] - lo[1]), 1e-9)
    v_norm = (vertices[:, 1] - lo[1]) / span_y
    pad = float(np.clip(padding, 0.0, 0.2))
    u = pad + np.mod(uvs_u, 1.0 + 1e-9) * (1.0 - 2.0 * pad)
    v = pad + (1.0 - v_norm) * (1.0 - 2.0 * pad)

    result = Mesh(vertices=vertices, faces=faces, uvs=np.stack([u, v], axis=1))
    result.compute_normals()
    result.metadata["uv_layout"] = "cylindrical"
    return result


def uv_area_distortion(mesh: Mesh) -> dict[str, float]:
    """Ratio of UV area to 3D area per triangle, summarised.

    A quick, honest quality readout for the layout: 1.0 everywhere means a
    perfectly area-preserving map, and the low percentile shows how badly the
    worst-stretched faces are packed.
    """
    if mesh.uvs is None or mesh.face_count == 0:
        return {}
    uv = mesh.uvs[mesh.faces]
    uv_area = 0.5 * np.abs(
        (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
        - (uv[:, 2, 0] - uv[:, 0, 0]) * (uv[:, 1, 1] - uv[:, 0, 1])
    )
    mesh_area = mesh.face_areas()
    total_uv = float(uv_area.sum())
    total_mesh = float(mesh_area.sum())
    if total_uv <= 0 or total_mesh <= 0:
        return {}
    relative = (uv_area / max(total_uv, 1e-18)) / np.maximum(mesh_area / total_mesh, 1e-18)
    return {
        "uv_coverage": round(total_uv, 4),
        "distortion_median": round(float(np.median(relative)), 4),
        "distortion_p05": round(float(np.percentile(relative, 5)), 4),
        "degenerate_faces": int((uv_area <= 1e-12).sum()),
    }
