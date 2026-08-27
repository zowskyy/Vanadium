"""Triangle mesh container and topology utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from conceptforge.mathutil import normalize, unique_rows_with_inverse


@dataclass
class Mesh:
    """An indexed triangle mesh with optional normals, UVs and vertex colours.

    Arrays are plain NumPy so they can go straight into a glTF buffer:
    ``vertices`` (V,3) float64, ``faces`` (F,3) int64, ``normals`` (V,3),
    ``uvs`` (V,2), ``colors`` (V,3).
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None
    colors: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3))
        self.faces = np.ascontiguousarray(np.asarray(self.faces, dtype=np.int64).reshape(-1, 3))
        for name in ("normals", "uvs", "colors"):
            value = getattr(self, name)
            if value is not None:
                width = 2 if name == "uvs" else 3
                setattr(self, name, np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1, width)))

    # -- basic stats ------------------------------------------------------ #
    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.vertex_count == 0:
            zero = np.zeros(3)
            return zero, zero.copy()
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def size(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0) if self.vertex_count else np.zeros(3)

    def face_normals(self, normalized: bool = True) -> np.ndarray:
        a = self.vertices[self.faces[:, 0]]
        b = self.vertices[self.faces[:, 1]]
        c = self.vertices[self.faces[:, 2]]
        n = np.cross(b - a, c - a)
        return normalize(n) if normalized else n

    def face_areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.face_normals(normalized=False), axis=1)

    def surface_area(self) -> float:
        return float(self.face_areas().sum())

    def volume(self) -> float:
        """Signed volume via the divergence theorem (needs a closed mesh)."""
        a = self.vertices[self.faces[:, 0]]
        b = self.vertices[self.faces[:, 1]]
        c = self.vertices[self.faces[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    # -- topology --------------------------------------------------------- #
    def edges(self, unique: bool = True) -> np.ndarray:
        """All edges as (E,2) with the lower index first."""
        f = self.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        e = np.sort(e, axis=1)
        if not unique:
            return e
        return np.unique(e, axis=0)

    def edge_lengths(self) -> np.ndarray:
        e = self.edges()
        return np.linalg.norm(self.vertices[e[:, 0]] - self.vertices[e[:, 1]], axis=1)

    def is_watertight(self) -> bool:
        """True when every edge is shared by exactly two faces."""
        if self.face_count == 0:
            return False
        e = self.edges(unique=False)
        _, counts = np.unique(e, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    def euler_characteristic(self) -> int:
        return self.vertex_count - self.edges().shape[0] + self.face_count

    def adjacency(self) -> tuple[np.ndarray, np.ndarray]:
        """Neighbour lists in CSR form: ``(offsets, neighbours)``.

        ``neighbours[offsets[v]:offsets[v + 1]]`` are the vertices adjacent to
        ``v``. Used by smoothing, weight diffusion and geodesic search.
        """
        e = self.edges()
        both = np.concatenate([e, e[:, ::-1]], axis=0)
        order = np.argsort(both[:, 0], kind="stable")
        sorted_edges = both[order]
        counts = np.bincount(sorted_edges[:, 0], minlength=self.vertex_count)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return offsets, np.ascontiguousarray(sorted_edges[:, 1])

    # -- cleanup ---------------------------------------------------------- #
    def compute_normals(self) -> "Mesh":
        """Area-weighted vertex normals (in place)."""
        n = np.zeros_like(self.vertices)
        fn = self.face_normals(normalized=False)
        for k in range(3):
            np.add.at(n, self.faces[:, k], fn)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        degenerate = lengths[:, 0] < 1e-12
        n = n / np.maximum(lengths, 1e-12)
        if degenerate.any():
            n[degenerate] = np.array([0.0, 1.0, 0.0])
        self.normals = n
        return self

    def remove_degenerate_faces(self) -> "Mesh":
        f = self.faces
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        if keep.all():
            return self
        self.faces = np.ascontiguousarray(f[keep])
        return self

    def remove_unused_vertices(self) -> "Mesh":
        used = np.zeros(self.vertex_count, dtype=bool)
        used[self.faces.ravel()] = True
        if used.all():
            return self
        remap = np.full(self.vertex_count, -1, dtype=np.int64)
        remap[used] = np.arange(int(used.sum()))
        self.vertices = np.ascontiguousarray(self.vertices[used])
        for name in ("normals", "uvs", "colors"):
            value = getattr(self, name)
            if value is not None and value.shape[0] == used.size:
                setattr(self, name, np.ascontiguousarray(value[used]))
        self.faces = remap[self.faces]
        return self

    def weld(self, tolerance: float = 1e-6) -> "Mesh":
        """Merge coincident vertices and drop the faces that collapse."""
        quantised = np.round(self.vertices / max(tolerance, 1e-12)) * tolerance
        unique, inverse = unique_rows_with_inverse(quantised)
        if unique.shape[0] == self.vertex_count:
            return self
        averaged = np.zeros_like(unique)
        counts = np.bincount(inverse, minlength=unique.shape[0]).astype(np.float64)
        for k in range(3):
            np.add.at(averaged[:, k], inverse, self.vertices[:, k])
        self.vertices = averaged / counts[:, None]
        self.faces = inverse[self.faces]
        for name in ("normals", "uvs", "colors"):
            setattr(self, name, None)
        return self.remove_degenerate_faces().remove_unused_vertices()

    def components(self) -> list[np.ndarray]:
        """Vertex index arrays, one per connected component."""
        offsets, neighbours = self.adjacency()
        labels = np.full(self.vertex_count, -1, dtype=np.int64)
        groups: list[np.ndarray] = []
        for seed in range(self.vertex_count):
            if labels[seed] >= 0:
                continue
            label = len(groups)
            stack = [seed]
            labels[seed] = label
            members = [seed]
            while stack:
                v = stack.pop()
                for n in neighbours[offsets[v] : offsets[v + 1]]:
                    if labels[n] < 0:
                        labels[n] = label
                        stack.append(int(n))
                        members.append(int(n))
            groups.append(np.array(members, dtype=np.int64))
        return groups

    def keep_largest_component(self) -> "Mesh":
        """Drop floating debris: stray blobs from noisy artwork."""
        groups = self.components()
        if len(groups) <= 1:
            return self
        biggest = max(groups, key=lambda g: g.size)
        keep = np.zeros(self.vertex_count, dtype=bool)
        keep[biggest] = True
        face_keep = keep[self.faces].all(axis=1)
        self.faces = np.ascontiguousarray(self.faces[face_keep])
        return self.remove_unused_vertices()

    # -- transforms ------------------------------------------------------- #
    def transformed(self, matrix: np.ndarray) -> "Mesh":
        matrix = np.asarray(matrix, dtype=np.float64)
        vertices = self.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        normals = None
        if self.normals is not None:
            inverse_transpose = np.linalg.inv(matrix[:3, :3]).T
            normals = normalize(self.normals @ inverse_transpose.T)
        return Mesh(
            vertices=vertices,
            faces=self.faces.copy(),
            normals=normals,
            uvs=None if self.uvs is None else self.uvs.copy(),
            colors=None if self.colors is None else self.colors.copy(),
            metadata=dict(self.metadata),
        )

    def scaled_to_height(self, height: float, ground_at_zero: bool = True) -> "Mesh":
        """Uniformly scale so the Y extent is ``height`` and feet sit on Y=0."""
        lo, hi = self.bounds()
        current = float(hi[1] - lo[1])
        if current < 1e-9:
            return self
        factor = float(height) / current
        origin = np.array(
            [
                0.5 * (lo[0] + hi[0]),
                lo[1] if ground_at_zero else 0.5 * (lo[1] + hi[1]),
                0.5 * (lo[2] + hi[2]),
            ]
        )
        self.vertices = (self.vertices - origin) * factor
        return self

    def copy(self) -> "Mesh":
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=None if self.normals is None else self.normals.copy(),
            uvs=None if self.uvs is None else self.uvs.copy(),
            colors=None if self.colors is None else self.colors.copy(),
            metadata=dict(self.metadata),
        )

    def stats(self) -> dict[str, object]:
        lengths = self.edge_lengths() if self.face_count else np.zeros(1)
        return {
            "vertices": self.vertex_count,
            "triangles": self.face_count,
            "watertight": self.is_watertight(),
            "surface_area": round(self.surface_area(), 5),
            "volume": round(self.volume(), 6),
            "mean_edge": round(float(lengths.mean()), 5),
            "size": [round(float(v), 4) for v in self.size()],
        }


def concatenate(meshes: Iterable[Mesh]) -> Mesh:
    """Merge meshes into one, offsetting face indices."""
    meshes = [m for m in meshes if m.face_count > 0]
    if not meshes:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    vertices, faces = [], []
    offset = 0
    for m in meshes:
        vertices.append(m.vertices)
        faces.append(m.faces + offset)
        offset += m.vertex_count
    return Mesh(np.concatenate(vertices), np.concatenate(faces))
