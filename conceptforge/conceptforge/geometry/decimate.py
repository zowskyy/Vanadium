"""Mesh simplification by quadric error edge collapse.

Marching tetrahedra emits a very dense, very uniform mesh; a production asset
wants a controllable triangle budget. This is Garland & Heckbert quadric error
metric simplification, with one change that matters in Python: instead of
collapsing one edge at a time through a priority queue (hundreds of thousands of
interpreted iterations), each round selects a *maximal independent set* of cheap
edges - no two sharing a vertex - and collapses all of them with vectorised
array operations. Roughly a quarter of the vertices go per round, so a target
is reached in a handful of rounds while the error metric stays exact.

Collapses are also gated on the link condition, which is what keeps the mesh
manifold and watertight through simplification.
"""

from __future__ import annotations

import numpy as np

from conceptforge.geometry.mesh import Mesh


def decimate(mesh: Mesh, target_faces: int, max_rounds: int = 24) -> Mesh:
    """Simplify ``mesh`` towards ``target_faces`` triangles.

    Returns a new mesh; the input is left untouched. Vertex attributes are
    dropped because collapsing moves vertices to new positions.
    """
    target_faces = int(target_faces)
    if target_faces <= 0 or mesh.face_count <= target_faces:
        return mesh.copy()

    working = Mesh(mesh.vertices.copy(), mesh.faces.copy())
    for _ in range(max_rounds):
        if working.face_count <= target_faces:
            break
        removable = working.face_count - target_faces
        # Each collapse of a manifold interior edge removes two triangles.
        budget = max(1, int(np.ceil(removable / 2.0)))
        collapsed = _collapse_round(working, budget)
        if collapsed == 0:
            break
    working.remove_degenerate_faces().remove_unused_vertices()
    return working


def face_quadrics(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex 4x4 quadric matrices, area weighted.

    A face's plane ``n . x + d = 0`` contributes the outer product of
    ``(nx, ny, nz, d)``; summing over incident faces gives a matrix whose
    quadratic form is the sum of squared distances to those planes.
    """
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-18)
    d = -np.einsum("ij,ij->i", normals, a)
    planes = np.concatenate([normals, d[:, None]], axis=1)
    weighted = areas[:, None, None] * np.einsum("ij,ik->ijk", planes, planes)

    quadrics = np.zeros((vertices.shape[0], 4, 4), dtype=np.float64)
    for corner in range(3):
        np.add.at(quadrics, faces[:, corner], weighted)
    return quadrics


def _optimal_positions(
    quadrics: np.ndarray, edges: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Best collapse target and its error, for every candidate edge."""
    combined = quadrics[edges[:, 0]] + quadrics[edges[:, 1]]
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])

    a = combined[:, :3, :3]
    rhs = -combined[:, :3, 3]
    determinants = np.linalg.det(a)
    scale = np.maximum(np.abs(a).max(axis=(1, 2)), 1e-12)
    solvable = np.abs(determinants) > 1e-10 * scale**3

    targets = midpoints.copy()
    if solvable.any():
        # NumPy 2 requires an explicit column vector for batched solves.
        solved = np.linalg.solve(a[solvable], rhs[solvable][:, :, None])[:, :, 0]
        # An ill-conditioned quadric can place the optimum far outside the local
        # neighbourhood; fall back to the midpoint when it does.
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)[solvable]
        drift = np.linalg.norm(solved - midpoints[solvable], axis=1)
        accept = drift <= 2.0 * np.maximum(lengths, 1e-12)
        slots = np.flatnonzero(solvable)[accept]
        targets[slots] = solved[accept]

    homogeneous = np.concatenate([targets, np.ones((targets.shape[0], 1))], axis=1)
    errors = np.einsum("ij,ijk,ik->i", homogeneous, combined, homogeneous)
    return targets, np.maximum(errors, 0.0)


def _collapse_round(mesh: Mesh, budget: int) -> int:
    """Collapse up to ``budget`` independent edges. Returns how many happened."""
    if mesh.face_count == 0:
        return 0
    edges = mesh.edges()
    if edges.shape[0] == 0:
        return 0
    quadrics = face_quadrics(mesh.vertices, mesh.faces)
    targets, errors = _optimal_positions(quadrics, edges, mesh.vertices)
    order = np.argsort(errors, kind="stable")

    offsets, neighbours = mesh.adjacency()
    neighbour_sets = [
        set(neighbours[offsets[v] : offsets[v + 1]].tolist()) for v in range(mesh.vertex_count)
    ]

    locked = np.zeros(mesh.vertex_count, dtype=bool)
    remap = np.arange(mesh.vertex_count, dtype=np.int64)
    new_positions = mesh.vertices.copy()
    performed = 0

    for edge_index in order:
        if performed >= budget:
            break
        u, v = int(edges[edge_index, 0]), int(edges[edge_index, 1])
        if locked[u] or locked[v]:
            continue
        # Link condition: in a manifold mesh an interior edge's endpoints share
        # exactly the two vertices opposite it. Anything else would fold the
        # surface or create a non-manifold edge.
        if len(neighbour_sets[u] & neighbour_sets[v]) != 2:
            continue
        locked[u] = True
        locked[v] = True
        for shared in neighbour_sets[u] & neighbour_sets[v]:
            locked[shared] = True
        remap[v] = u
        new_positions[u] = targets[edge_index]
        performed += 1

    if performed == 0:
        return 0

    mesh.vertices = new_positions
    mesh.faces = remap[mesh.faces]
    mesh.remove_degenerate_faces()
    mesh.remove_unused_vertices()
    mesh.normals = None
    return performed


def build_lod_chain(mesh: Mesh, ratios: list[float] | tuple[float, ...]) -> list[Mesh]:
    """Progressively simplified copies, each a fraction of the base face count.

    Ratios are relative to the *input* mesh, and each level is derived from the
    previous one so the chain stays consistent.
    """
    chain: list[Mesh] = []
    base = mesh.face_count
    current = mesh
    for ratio in ratios:
        target = int(max(64, round(base * float(ratio))))
        if target >= current.face_count:
            continue
        current = decimate(current, target)
        current.compute_normals()
        chain.append(current)
    return chain
