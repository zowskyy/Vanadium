"""Isosurface extraction by marching tetrahedra.

Marching *tetrahedra* rather than marching cubes: splitting each cell into six
tetrahedra removes the ambiguous-face cases that make marching cubes able to
emit cracks, so the output is watertight and manifold by construction. It also
needs no 256-entry lookup table - the 16 tetrahedron cases are derived at import
time from first principles.

The extra triangles a tetrahedral decomposition produces are removed later by
:mod:`conceptforge.geometry.decimate`.
"""

from __future__ import annotations

import numpy as np

from conceptforge.geometry.mesh import Mesh
from conceptforge.ndops import trilinear_sample

#: Cell corner offsets, indexed by the bit pattern (dx, dy, dz).
CUBE_CORNERS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)

#: Six tetrahedra tiling the cube, all sharing the 0-6 main diagonal. This is
#: the standard decomposition; sharing one diagonal is what keeps neighbouring
#: cells consistent and therefore the surface crack-free.
CUBE_TETRAHEDRA: tuple[tuple[int, int, int, int], ...] = (
    (0, 5, 1, 6),
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
)

Triangle = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def _build_case_table() -> list[list[Triangle]]:
    """Triangles per inside/outside pattern of a tetrahedron's four corners.

    Each triangle corner is the tetrahedron edge it lies on, given as a pair of
    local corner indices. Winding is fixed up later from the field gradient, so
    only the edge sets matter here.
    """
    table: list[list[Triangle]] = []
    for case in range(16):
        inside = [i for i in range(4) if case & (1 << i)]
        outside = [i for i in range(4) if not case & (1 << i)]
        triangles: list[Triangle] = []
        if len(inside) == 1:
            a = inside[0]
            triangles.append(tuple((a, o) for o in outside))  # type: ignore[arg-type]
        elif len(inside) == 3:
            a = outside[0]
            triangles.append(tuple((a, i) for i in inside))  # type: ignore[arg-type]
        elif len(inside) == 2:
            a, b = inside
            c, d = outside
            # Walk the four cut edges around the quad, then split it.
            quad = [(a, c), (a, d), (b, d), (b, c)]
            triangles.append((quad[0], quad[1], quad[2]))
            triangles.append((quad[0], quad[2], quad[3]))
        table.append(triangles)
    return table


CASE_TABLE = _build_case_table()


def marching_tetrahedra(
    field: np.ndarray,
    level: float = 0.0,
    origin: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: float | tuple[float, float, float] = 1.0,
) -> Mesh:
    """Extract the ``field == level`` isosurface.

    Parameters
    ----------
    field:
        3D scalar volume; values above ``level`` are inside the solid.
    level:
        Isovalue to surface.
    origin, spacing:
        Map voxel indices to world coordinates as ``origin + index * spacing``.

    Returns
    -------
    Mesh
        Welded, outward-oriented triangle mesh. Empty if the field never
        crosses ``level``.
    """
    field = np.ascontiguousarray(np.asarray(field, dtype=np.float64))
    if field.ndim != 3:
        raise ValueError("marching_tetrahedra expects a 3D field")
    nx, ny, nz = field.shape
    if min(nx, ny, nz) < 2:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    inside = field > level
    if not inside.any() or inside.all():
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    cell_shape = (nx - 1, ny - 1, nz - 1)
    stride_y, stride_x = nz, ny * nz

    def corner_view(array: np.ndarray, corner: int) -> np.ndarray:
        dx, dy, dz = CUBE_CORNERS[corner]
        return array[dx : dx + nx - 1, dy : dy + ny - 1, dz : dz + nz - 1]

    inside_corners = [corner_view(inside, c) for c in range(8)]

    # Per triangle corner (0..2), the pair of grid corners whose edge it lies on.
    tri_p: list[list[np.ndarray]] = [[], [], []]
    tri_q: list[list[np.ndarray]] = [[], [], []]

    for tet in CUBE_TETRAHEDRA:
        case = np.zeros(cell_shape, dtype=np.uint8)
        for bit, corner in enumerate(tet):
            case |= inside_corners[corner].astype(np.uint8) << bit
        for case_value, triangles in enumerate(CASE_TABLE):
            if not triangles:
                continue
            selected = np.flatnonzero((case == case_value).ravel())
            if selected.size == 0:
                continue
            ci, cj, ck = np.unravel_index(selected, cell_shape)
            flats = {}
            for corner in set(idx for tri in triangles for edge in tri for idx in edge):
                dx, dy, dz = CUBE_CORNERS[tet[corner]]
                flats[corner] = (ci + dx) * stride_x + (cj + dy) * stride_y + (ck + dz)
            for triangle in triangles:
                for slot, (a, b) in enumerate(triangle):
                    tri_p[slot].append(flats[a])
                    tri_q[slot].append(flats[b])

    if not tri_p[0]:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    p = np.stack([np.concatenate(tri_p[slot]) for slot in range(3)], axis=1)
    q = np.stack([np.concatenate(tri_q[slot]) for slot in range(3)], axis=1)

    # Weld by edge identity: two cells that cut the same grid edge must produce
    # the exact same vertex, which is what makes the result watertight.
    lo = np.minimum(p, q)
    hi = np.maximum(p, q)
    total = nx * ny * nz
    keys = lo.astype(np.int64) * total + hi.astype(np.int64)
    unique_keys, inverse = np.unique(keys.ravel(), return_inverse=True)
    faces = inverse.astype(np.int64).reshape(-1, 3)

    lo_flat = unique_keys // total
    hi_flat = unique_keys % total
    flat_field = field.ravel()
    value_a = flat_field[lo_flat]
    value_b = flat_field[hi_flat]
    denom = value_b - value_a
    t = np.where(np.abs(denom) > 1e-12, (level - value_a) / np.where(np.abs(denom) > 1e-12, denom, 1.0), 0.5)
    t = np.clip(t, 0.0, 1.0)

    index_a = np.stack(np.unravel_index(lo_flat, field.shape), axis=1).astype(np.float64)
    index_b = np.stack(np.unravel_index(hi_flat, field.shape), axis=1).astype(np.float64)
    index_positions = index_a + t[:, None] * (index_b - index_a)

    spacing_vec = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (3,))
    vertices = np.asarray(origin, dtype=np.float64) + index_positions * spacing_vec

    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.remove_degenerate_faces()
    _orient_outward(mesh, field, index_positions, spacing_vec)
    mesh.remove_unused_vertices()
    return mesh


def _orient_outward(
    mesh: Mesh, field: np.ndarray, index_positions: np.ndarray, spacing: np.ndarray
) -> None:
    """Flip faces so normals point away from the solid.

    The field gradient points towards increasing values, i.e. into the solid, so
    the outward normal is the negated gradient. Comparing each face's winding
    normal against it is cheaper and more robust than tracking orientation
    through the case table.
    """
    if mesh.face_count == 0:
        return
    gradients = np.stack(
        [
            trilinear_sample(component, index_positions)
            for component in np.gradient(field, edge_order=1)
        ],
        axis=1,
    )
    # np.gradient works in index space; convert to world space.
    gradients = gradients / spacing[None, :]
    outward = -gradients
    face_outward = outward[mesh.faces].sum(axis=1)
    face_normals = mesh.face_normals(normalized=False)
    flip = np.einsum("ij,ij->i", face_normals, face_outward) < 0.0
    if flip.any():
        mesh.faces[flip] = mesh.faces[flip][:, ::-1]
