"""Skin weight solving.

Weights are computed from **geodesic** distance across the mesh surface, not
straight-line distance. The difference is the single biggest factor in whether
an auto-rig deforms acceptably: an arm resting against the ribs is millimetres
from the torso in space but a long way around the surface, so a Euclidean solver
binds ribcage vertices to the upper arm and the chest tears open when the arm
lifts. Surface distance cannot make that mistake.

Each bone seeds a Dijkstra search from the vertices near its segment and
propagates outward with a cutoff proportional to bone length, which keeps the
solve linear in mesh size rather than quadratic in bones times vertices.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conceptforge.config import RiggingConfig
from conceptforge.geometry.mesh import Mesh
from conceptforge.geometry.smoothing import laplacian_smooth_scalar
from conceptforge.mathutil import point_segment_distance
from conceptforge.rigging.skeleton import Skeleton


@dataclass
class SkinBinding:
    """Per-vertex bone influences, already trimmed and normalised."""

    joints: np.ndarray
    """(V, K) joint indices."""

    weights: np.ndarray
    """(V, K) weights summing to 1 per row."""

    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def influences(self) -> int:
        return int(self.joints.shape[1])

    def deform(self, vertices: np.ndarray, skinning_matrices: np.ndarray) -> np.ndarray:
        """Linear blend skinning of ``vertices`` (V,3)."""
        vertices = np.asarray(vertices, dtype=np.float64)
        out = np.zeros_like(vertices)
        homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
        for slot in range(self.influences):
            matrices = skinning_matrices[self.joints[:, slot]]
            contribution = np.einsum("vij,vj->vi", matrices[:, :3, :], homogeneous)
            out += self.weights[:, slot : slot + 1] * contribution
        return out

    def deform_normals(self, normals: np.ndarray, skinning_matrices: np.ndarray) -> np.ndarray:
        normals = np.asarray(normals, dtype=np.float64)
        out = np.zeros_like(normals)
        for slot in range(self.influences):
            matrices = skinning_matrices[self.joints[:, slot]]
            out += self.weights[:, slot : slot + 1] * np.einsum(
                "vij,vj->vi", matrices[:, :3, :3], normals
            )
        lengths = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(lengths, 1e-12)


def bind_skin(
    mesh: Mesh,
    skeleton: Skeleton,
    config: RiggingConfig | None = None,
) -> SkinBinding:
    """Solve skin weights binding ``mesh`` to ``skeleton``."""
    config = config or RiggingConfig()
    if mesh.vertex_count == 0:
        raise ValueError("cannot bind an empty mesh")

    segments = skeleton.bone_segments()
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    joint_count = len(skeleton)
    vertices = mesh.vertices

    offsets, neighbours = mesh.adjacency()
    edge_weights = np.linalg.norm(vertices[neighbours] - vertices[np.repeat(np.arange(mesh.vertex_count), np.diff(offsets))], axis=1)
    scale = float(np.median(edge_weights[edge_weights > 0])) if (edge_weights > 0).any() else 1e-3

    raw = np.zeros((mesh.vertex_count, joint_count), dtype=np.float64)
    euclidean_only = 0
    for bone in range(joint_count):
        start, end = segments[bone]
        distance = point_segment_distance(vertices, start, end)
        radius = max(float(lengths[bone]) * 0.30, 3.0 * scale)
        seeds = np.flatnonzero(distance <= radius)
        if seeds.size == 0:
            # A very short bone (a clavicle on a stocky character) may not
            # capture any vertex; take the closest handful instead.
            seeds = np.argsort(distance)[: max(3, mesh.vertex_count // 400)]

        if config.geodesic:
            cutoff = max(float(lengths[bone]) * 2.5, 12.0 * scale)
            field = _dijkstra(
                offsets, neighbours, edge_weights, seeds, distance[seeds], cutoff, mesh.vertex_count
            )
        else:
            field = distance
            euclidean_only += 1

        floor = max(0.25 * radius, 0.5 * scale)
        with np.errstate(divide="ignore", over="ignore"):
            weight = np.where(
                np.isfinite(field), 1.0 / np.power(np.maximum(field, floor), config.falloff), 0.0
            )
        raw[:, bone] = weight

    # Vertices out of every bone's reach (isolated geometry) fall back to the
    # nearest bone so no vertex is left unskinned.
    unreached = raw.sum(axis=1) <= 0.0
    if unreached.any():
        for vertex in np.flatnonzero(unreached):
            distances = np.array(
                [point_segment_distance(vertices[vertex : vertex + 1], *segments[b])[0] for b in range(joint_count)]
            )
            raw[vertex, int(np.argmin(distances))] = 1.0

    normalised = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-18)
    if config.weight_smoothing > 0:
        normalised = laplacian_smooth_scalar(
            normalised, offsets, neighbours, iterations=int(config.weight_smoothing), rate=0.45
        )
        normalised = np.maximum(normalised, 0.0)
        normalised /= np.maximum(normalised.sum(axis=1, keepdims=True), 1e-18)

    joints, weights = _trim_influences(normalised, max(1, int(config.max_influences)))

    stats = {
        "influences": int(joints.shape[1]),
        "mode": "geodesic" if config.geodesic else "euclidean",
        "mean_active_bones": round(float((weights > 1e-4).sum(axis=1).mean()), 2),
        "max_weight_error": round(float(np.abs(weights.sum(axis=1) - 1.0).max()), 8),
        "bones_without_seeds": euclidean_only,
        "unreached_vertices": int(unreached.sum()),
    }
    return SkinBinding(joints=joints, weights=weights, stats=stats)


def _dijkstra(
    offsets: np.ndarray,
    neighbours: np.ndarray,
    edge_weights: np.ndarray,
    seeds: np.ndarray,
    seed_distances: np.ndarray,
    cutoff: float,
    vertex_count: int,
) -> np.ndarray:
    """Multi-source shortest path over the mesh graph, stopped at ``cutoff``."""
    distances = np.full(vertex_count, np.inf)
    heap: list[tuple[float, int]] = []
    for vertex, seed_distance in zip(seeds.tolist(), np.asarray(seed_distances, dtype=np.float64).tolist()):
        if seed_distance < distances[vertex]:
            distances[vertex] = seed_distance
            heap.append((seed_distance, int(vertex)))
    heapq.heapify(heap)

    while heap:
        current, vertex = heapq.heappop(heap)
        if current > distances[vertex]:
            continue
        if current > cutoff:
            break
        begin, end = int(offsets[vertex]), int(offsets[vertex + 1])
        for slot in range(begin, end):
            other = int(neighbours[slot])
            candidate = current + float(edge_weights[slot])
            if candidate < distances[other] and candidate <= cutoff:
                distances[other] = candidate
                heapq.heappush(heap, (candidate, other))
    return distances


def _trim_influences(weights: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ``limit`` strongest influences per vertex and renormalise.

    Four is the hardware norm for GPU skinning; trimming here rather than at
    export time means the smoothing pass above operates on the full field and
    the trimmed result is still exactly normalised.
    """
    count = min(limit, weights.shape[1])
    order = np.argsort(-weights, axis=1, kind="stable")[:, :count]
    picked = np.take_along_axis(weights, order, axis=1)
    picked = np.maximum(picked, 0.0)
    totals = picked.sum(axis=1, keepdims=True)
    # A row with no influence at all would divide by zero; pin it to its first
    # (best) joint.
    empty = totals[:, 0] <= 1e-18
    if empty.any():
        picked[empty] = 0.0
        picked[empty, 0] = 1.0
        totals = picked.sum(axis=1, keepdims=True)
    return order.astype(np.int64), picked / totals


def weight_heatmap_colors(binding: SkinBinding, joint_count: int) -> np.ndarray:
    """Per-vertex colours showing the dominant bone, for debug renders."""
    dominant = binding.joints[np.arange(binding.joints.shape[0]), np.argmax(binding.weights, axis=1)]
    golden = 0.61803398875
    hue = np.mod(dominant * golden, 1.0)
    del joint_count
    return _hsv_to_rgb(hue, np.full_like(hue, 0.62), np.full_like(hue, 0.95))


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = np.mod(i, 6)
    options = np.stack(
        [
            np.stack([v, t, p], axis=1),
            np.stack([q, v, p], axis=1),
            np.stack([p, v, t], axis=1),
            np.stack([p, q, v], axis=1),
            np.stack([t, p, v], axis=1),
            np.stack([v, p, q], axis=1),
        ],
        axis=0,
    )
    return options[i, np.arange(h.size)]
