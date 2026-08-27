"""Mesh fairing.

Plain Laplacian smoothing shrinks a closed mesh a little on every pass, which
over a dozen iterations visibly deflates a character. Taubin smoothing avoids it
by alternating a positive (smoothing) pass with a slightly larger negative
(unshrinking) pass, giving a low-pass filter with near-unity gain in the
pass band.
"""

from __future__ import annotations

import numpy as np

from conceptforge.geometry.mesh import Mesh


def umbrella(vertices: np.ndarray, offsets: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Uniform-weight Laplacian: mean of neighbours minus the vertex itself."""
    counts = np.diff(offsets).astype(np.float64)
    sums = np.zeros_like(vertices)
    targets = np.repeat(np.arange(vertices.shape[0]), np.diff(offsets))
    for axis in range(vertices.shape[1]):
        np.add.at(sums[:, axis], targets, vertices[neighbours, axis])
    safe = np.maximum(counts, 1.0)[:, None]
    return sums / safe - vertices


def taubin_smooth(
    mesh: Mesh,
    iterations: int = 12,
    lam: float = 0.55,
    mu: float = -0.58,
    pin: np.ndarray | None = None,
) -> Mesh:
    """Volume-preserving fairing, in place.

    Parameters
    ----------
    iterations:
        Number of lambda/mu pairs. Cost is linear.
    lam, mu:
        Filter coefficients. ``mu`` must be more negative than ``lam`` is
        positive; the classic choice satisfies ``1/lam + 1/mu ~= 5``.
    pin:
        Optional boolean mask of vertices to hold in place.
    """
    if mesh.face_count == 0 or iterations <= 0:
        return mesh
    if not (mu < 0.0 < lam):
        raise ValueError("taubin_smooth needs lam > 0 > mu")
    offsets, neighbours = mesh.adjacency()
    vertices = mesh.vertices
    free = None if pin is None else ~np.asarray(pin, dtype=bool)
    for _ in range(int(iterations)):
        for weight in (lam, mu):
            delta = umbrella(vertices, offsets, neighbours) * weight
            if free is not None:
                delta[~free] = 0.0
            vertices = vertices + delta
    mesh.vertices = np.ascontiguousarray(vertices)
    mesh.normals = None
    return mesh


def laplacian_smooth_scalar(
    values: np.ndarray, offsets: np.ndarray, neighbours: np.ndarray, iterations: int = 2, rate: float = 0.5
) -> np.ndarray:
    """Diffuse per-vertex scalars (or vectors) over the mesh graph.

    Used to soften skin weights across joints after the geodesic solve.
    """
    values = np.array(values, dtype=np.float64, copy=True)
    if iterations <= 0:
        return values
    flat = values.reshape(values.shape[0], -1)
    counts = np.maximum(np.diff(offsets).astype(np.float64), 1.0)[:, None]
    targets = np.repeat(np.arange(flat.shape[0]), np.diff(offsets))
    for _ in range(int(iterations)):
        sums = np.zeros_like(flat)
        for axis in range(flat.shape[1]):
            np.add.at(sums[:, axis], targets, flat[neighbours, axis])
        flat = flat + rate * (sums / counts - flat)
    return flat.reshape(values.shape)
