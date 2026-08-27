"""Small vector / quaternion / matrix helpers shared across the pipeline.

Conventions used everywhere in ConceptForge:

* Right-handed, Y-up, -Z forward (glTF convention). X grows to the character's
  left as seen by the viewer of the front concept view.
* Quaternions are stored ``(x, y, z, w)`` to match glTF accessors directly.
* Matrices are 4x4 row-major NumPy arrays and multiply column vectors, i.e.
  ``p' = M @ p``. The glTF exporter transposes on write because glTF stores
  column-major.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# vectors
# --------------------------------------------------------------------------- #
def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return ``v`` scaled to unit length along ``axis`` (zero-safe)."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def lerp(a, b, t):
    return np.asarray(a) * (1.0 - t) + np.asarray(b) * t


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def smoothstep(edge0: float, edge1: float, x):
    """Hermite smoothstep, used for animation easing and falloffs."""
    t = clamp((np.asarray(x, dtype=np.float64) - edge0) / max(edge1 - edge0, EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point in ``points`` (N,3) to segment ``a``-``b``.

    Returns an (N,) array. Also the workhorse of the skinning seed search.
    """
    points = np.asarray(points, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom < EPS:
        return np.linalg.norm(points - a, axis=1)
    t = clamp(((points - a) @ ab) / denom, 0.0, 1.0)[:, None]
    proj = a + t * ab
    return np.linalg.norm(points - proj, axis=1)


# --------------------------------------------------------------------------- #
# quaternions (x, y, z, w)
# --------------------------------------------------------------------------- #
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=np.float64))
    h = 0.5 * float(angle)
    s = np.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(h)])


def quat_from_euler(rx: float, ry: float, rz: float, order: str = "XYZ") -> np.ndarray:
    """Compose a quaternion from Euler angles in radians.

    ``order`` lists the axes in application order, so ``"XYZ"`` means
    ``q = qx * qy * qz`` and therefore Z is applied to the vector first.
    """
    parts = {
        "X": quat_from_axis_angle((1.0, 0.0, 0.0), rx),
        "Y": quat_from_axis_angle((0.0, 1.0, 0.0), ry),
        "Z": quat_from_axis_angle((0.0, 0.0, 1.0), rz),
    }
    q = IDENTITY_QUAT.copy()
    for axis in order:
        q = quat_mul(q, parts[axis])
    return q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` (3,) or (N,3) by quaternion ``q``."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = q[:3]
    w = q[3]
    single = v.ndim == 1
    vv = v.reshape(1, 3) if single else v
    t = 2.0 * np.cross(np.broadcast_to(u, vv.shape), vv)
    out = vv + w * t + np.cross(np.broadcast_to(u, vv.shape), t)
    return out[0] if single else out


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize(np.asarray(q0, dtype=np.float64))
    q1 = normalize(np.asarray(q1, dtype=np.float64))
    d = float(q0 @ q1)
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        return normalize(q0 + t * (q1 - q0))
    theta = np.arccos(clamp(d, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / s


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize(np.asarray(q, dtype=np.float64))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.eye(4)
    m[:3, :3] = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    return m


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        )
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(r[i, i] - r[j, j] - r[k, k] + 1.0, EPS)) * 2.0
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    q[3] = (r[k, j] - r[j, k]) / s
    return normalize(q)


def quat_look_rotation(direction: Sequence[float], reference: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Shortest-arc rotation taking ``reference`` onto ``direction``.

    Used to orient generated bones along the limb axis derived from the art.
    """
    a = normalize(np.asarray(reference, dtype=np.float64))
    b = normalize(np.asarray(direction, dtype=np.float64))
    d = float(a @ b)
    if d > 1.0 - 1e-9:
        return IDENTITY_QUAT.copy()
    if d < -1.0 + 1e-9:
        # 180 degrees: any perpendicular axis works.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        return quat_from_axis_angle(axis, np.pi)
    axis = np.cross(a, b)
    return normalize(np.concatenate([axis, [1.0 + d]]))


# --------------------------------------------------------------------------- #
# matrices
# --------------------------------------------------------------------------- #
def translation_matrix(t: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def scale_matrix(s: Sequence[float] | float) -> np.ndarray:
    m = np.eye(4)
    s = np.asarray(s, dtype=np.float64)
    m[0, 0], m[1, 1], m[2, 2] = np.broadcast_to(s, (3,))
    return m


def compose_trs(translation: Sequence[float], rotation: Sequence[float], scale: Sequence[float] | float = 1.0) -> np.ndarray:
    return translation_matrix(translation) @ quat_to_matrix(np.asarray(rotation)) @ scale_matrix(scale)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix to (N,3) points."""
    points = np.asarray(points, dtype=np.float64)
    return points @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """World-to-view matrix for the software renderer."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = normalize(target - eye)
    up_v = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up_v)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right = normalize(right)
    true_up = np.cross(right, forward)
    m = np.eye(4)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4))
    m[0, 0] = f / max(aspect, EPS)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def unique_rows_with_inverse(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``np.unique(axis=0)`` with an inverse map, but fast for float vertices.

    Values are quantised to a tight tolerance first so that geometrically
    identical vertices produced by independent marching-tetrahedra cells weld
    into one.
    """
    arr = np.asarray(arr, dtype=np.float64)
    keys = np.round(arr, 6)
    view = np.ascontiguousarray(keys).view([("", keys.dtype)] * keys.shape[1]).ravel()
    _, index, inverse = np.unique(view, return_index=True, return_inverse=True)
    return arr[index], inverse.astype(np.int64).ravel()


def weighted_average_quaternions(quats: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    """Cheap quaternion blend: sign-aligned linear average then renormalise."""
    quats = [np.asarray(q, dtype=np.float64) for q in quats]
    weights = list(weights)
    if not quats:
        return IDENTITY_QUAT.copy()
    ref = quats[0]
    acc = np.zeros(4)
    for q, w in zip(quats, weights):
        acc += (q if float(q @ ref) >= 0.0 else -q) * w
    if np.linalg.norm(acc) < EPS:
        return IDENTITY_QUAT.copy()
    return normalize(acc)
