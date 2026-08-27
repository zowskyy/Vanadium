"""N-dimensional array operations the pipeline is built on.

ConceptForge deliberately depends only on NumPy and Pillow, so the handful of
image/volume primitives normally pulled in from SciPy or scikit-image live
here. They are all exact (no chamfer approximations) and vectorised across
scanlines so they run at interactive speed on both 2D masks and 3D fields.
"""

from __future__ import annotations

import numpy as np

INF = np.inf


# --------------------------------------------------------------------------- #
# separable filtering
# --------------------------------------------------------------------------- #
def gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    sigma = float(max(sigma, 1e-6))
    radius = max(1, int(np.ceil(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def convolve1d(arr: np.ndarray, kernel: np.ndarray, axis: int, mode: str = "reflect") -> np.ndarray:
    """Convolve along a single axis by accumulating shifted slices.

    Faster than ``np.apply_along_axis`` and works for any dimensionality, which
    matters because the same code path filters 2D mattes and 3D density fields.
    """
    arr = np.asarray(arr, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    radius = kernel.size // 2
    if radius == 0:
        return arr * kernel[0]
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(arr, pad, mode=mode)
    out = np.zeros_like(arr)
    n = arr.shape[axis]
    for i, weight in enumerate(kernel):
        if weight == 0.0:
            continue
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + n)
        out += weight * padded[tuple(sl)]
    return out


def gaussian_blur(arr: np.ndarray, sigma: float | tuple[float, ...], mode: str = "reflect") -> np.ndarray:
    """True separable Gaussian blur over every axis."""
    arr = np.asarray(arr, dtype=np.float64)
    sigmas = (sigma,) * arr.ndim if np.isscalar(sigma) else tuple(sigma)
    if len(sigmas) != arr.ndim:
        raise ValueError("sigma must be scalar or one value per axis")
    out = arr
    for axis, s in enumerate(sigmas):
        if s and s > 1e-6:
            out = convolve1d(out, gaussian_kernel(float(s)), axis, mode=mode)
    return out


def sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel dI/dx, dI/dy for a 2D array (used for normal-map synthesis)."""
    image = np.asarray(image, dtype=np.float64)
    smooth = np.array([1.0, 2.0, 1.0]) / 4.0
    deriv = np.array([-1.0, 0.0, 1.0]) / 2.0
    gx = convolve1d(convolve1d(image, deriv, axis=1), smooth, axis=0)
    gy = convolve1d(convolve1d(image, deriv, axis=0), smooth, axis=1)
    return gx, gy


# --------------------------------------------------------------------------- #
# exact euclidean distance transform
# --------------------------------------------------------------------------- #
def _edt_axis(f: np.ndarray, axis: int) -> np.ndarray:
    """One exact squared-EDT pass (Felzenszwalb & Huttenlocher lower envelope).

    The published algorithm walks a single scanline with a parabola stack; this
    implementation keeps one stack per scanline and steps them forward together
    so the whole volume is processed with NumPy-width operations.
    """
    f = np.moveaxis(np.asarray(f, dtype=np.float64), axis, 0)
    shape = f.shape
    n = shape[0]
    flat = f.reshape(n, -1)
    lines = flat.shape[1]
    if n == 1:
        return np.moveaxis(flat.reshape(shape), 0, axis)

    cols = np.arange(lines)
    v = np.zeros((n, lines), dtype=np.int64)          # parabola apex locations
    z = np.empty((n + 1, lines), dtype=np.float64)    # envelope breakpoints
    k = np.zeros(lines, dtype=np.int64)               # top of each stack
    z[0] = -INF
    z[1] = INF

    for q in range(1, n):
        fq = flat[q]
        while True:
            vk = v[k, cols]
            denom = 2.0 * (q - vk)
            s = ((fq + q * q) - (flat[vk, cols] + vk * vk)) / denom
            pop = s <= z[k, cols]
            if not pop.any():
                break
            k = np.where(pop, k - 1, k)
        k = k + 1
        v[k, cols] = q
        z[k, cols] = s
        z[k + 1, cols] = INF

    out = np.empty_like(flat)
    k = np.zeros(lines, dtype=np.int64)
    for q in range(n):
        while True:
            advance = z[k + 1, cols] < q
            if not advance.any():
                break
            k = np.where(advance, k + 1, k)
        vk = v[k, cols]
        out[q] = (q - vk) ** 2 + flat[vk, cols]
    return np.moveaxis(out.reshape(shape), 0, axis)


def edt_squared(mask: np.ndarray) -> np.ndarray:
    """Exact squared Euclidean distance from every cell to the nearest zero.

    ``mask`` is treated as boolean: True cells get their distance to the
    closest False cell, False cells get 0.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float64)
    # A finite sentinel rather than +inf: the lower-envelope pass subtracts
    # parabola offsets and inf - inf would poison whole scanlines with NaN.
    unreachable = 4.0 * float(sum(s * s for s in mask.shape)) + 1.0
    if mask.all():
        return np.full(mask.shape, unreachable)
    f = np.where(mask, unreachable, 0.0)
    for axis in range(mask.ndim):
        f = _edt_axis(f, axis)
    return np.minimum(f, unreachable)


def edt(mask: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance transform (see :func:`edt_squared`)."""
    return np.sqrt(edt_squared(mask))


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance field: positive inside ``mask``, negative outside."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any() or mask.all():
        return np.where(mask, edt(mask), -edt(~mask))
    inside = edt(mask)
    outside = edt(~mask)
    return np.where(mask, inside, -outside)


# --------------------------------------------------------------------------- #
# binary morphology (exact disk/ball structuring elements via the EDT)
# --------------------------------------------------------------------------- #
def binary_dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask).astype(bool)
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    return edt_squared(~mask) <= radius * radius


def binary_erode(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask).astype(bool)
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    return edt_squared(mask) > radius * radius


def binary_close(mask: np.ndarray, radius: float) -> np.ndarray:
    return binary_erode(binary_dilate(mask, radius), radius)


def binary_open(mask: np.ndarray, radius: float) -> np.ndarray:
    return binary_dilate(binary_erode(mask, radius), radius)


def binary_fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill cavities not connected to the array border."""
    mask = np.asarray(mask).astype(bool)
    background = ~mask
    labels, count = connected_components(background)
    if count == 0:
        return mask
    border_labels = set()
    for axis in range(mask.ndim):
        for index in (0, mask.shape[axis] - 1):
            sl = [slice(None)] * mask.ndim
            sl[axis] = index
            border_labels.update(np.unique(labels[tuple(sl)]).tolist())
    border_labels.discard(-1)
    keep_background = np.isin(labels, list(border_labels)) if border_labels else np.zeros_like(mask)
    return mask | (background & ~keep_background)


# --------------------------------------------------------------------------- #
# connected components
# --------------------------------------------------------------------------- #
def connected_components(mask: np.ndarray, connectivity: int = 1) -> tuple[np.ndarray, int]:
    """Label connected cells of ``mask``.

    Returns ``(labels, count)`` where background cells are ``-1`` and labels are
    contiguous from 0. Implemented as a parallel union-find (hook + pointer
    jumping) so it stays vectorised for large 2D mattes and 3D volumes.

    ``connectivity`` 1 means face neighbours only; 2 also links diagonal
    neighbours (8-connectivity in 2D).
    """
    mask = np.asarray(mask).astype(bool)
    index = np.full(mask.shape, -1, dtype=np.int64)
    flat_fg = np.flatnonzero(mask.ravel())
    if flat_fg.size == 0:
        return index, 0
    index.reshape(-1)[flat_fg] = np.arange(flat_fg.size)

    edges: list[np.ndarray] = []
    offsets = _neighbour_offsets(mask.ndim, connectivity)
    for offset in offsets:
        a_sl, b_sl = [], []
        for d in offset:
            if d == 0:
                a_sl.append(slice(None))
                b_sl.append(slice(None))
            elif d > 0:
                a_sl.append(slice(0, -d))
                b_sl.append(slice(d, None))
            else:
                a_sl.append(slice(-d, None))
                b_sl.append(slice(0, d))
        a = index[tuple(a_sl)]
        b = index[tuple(b_sl)]
        both = (a >= 0) & (b >= 0)
        if both.any():
            edges.append(np.stack([a[both].ravel(), b[both].ravel()], axis=1))

    parent = np.arange(flat_fg.size, dtype=np.int64)
    if edges:
        edge_array = np.concatenate(edges, axis=0)
        u = edge_array[:, 0]
        w = edge_array[:, 1]
        for _ in range(64):
            ru, rw = parent[u], parent[w]
            lo = np.minimum(ru, rw)
            hi = np.maximum(ru, rw)
            nxt = parent.copy()
            np.minimum.at(nxt, hi, lo)
            np.minimum.at(nxt, lo, lo)
            for _ in range(32):  # pointer jumping to full compression
                jumped = nxt[nxt]
                if np.array_equal(jumped, nxt):
                    break
                nxt = jumped
            if np.array_equal(nxt, parent):
                break
            parent = nxt

    roots, labels = np.unique(parent, return_inverse=True)
    index.reshape(-1)[flat_fg] = labels.astype(np.int64).ravel()
    return index, int(roots.size)


def _neighbour_offsets(ndim: int, connectivity: int) -> list[tuple[int, ...]]:
    """Forward-only neighbour offsets, so each edge is generated once."""
    offsets: list[tuple[int, ...]] = []
    for combo in np.ndindex(*([3] * ndim)):
        delta = tuple(c - 1 for c in combo)
        if all(d == 0 for d in delta):
            continue
        if sum(abs(d) for d in delta) > connectivity:
            continue
        # Keep only lexicographically positive directions.
        for d in delta:
            if d > 0:
                offsets.append(delta)
                break
            if d < 0:
                break
    return offsets


def largest_component(mask: np.ndarray, connectivity: int = 2) -> np.ndarray:
    labels, count = connected_components(mask, connectivity=connectivity)
    if count <= 1:
        return np.asarray(mask).astype(bool)
    sizes = np.bincount(labels[labels >= 0].ravel(), minlength=count)
    return labels == int(np.argmax(sizes))


def component_sizes(mask: np.ndarray, connectivity: int = 2) -> tuple[np.ndarray, np.ndarray]:
    labels, count = connected_components(mask, connectivity=connectivity)
    if count == 0:
        return labels, np.zeros(0, dtype=np.int64)
    return labels, np.bincount(labels[labels >= 0].ravel(), minlength=count)


def remove_small_components(mask: np.ndarray, min_size: int, connectivity: int = 2) -> np.ndarray:
    labels, sizes = component_sizes(mask, connectivity=connectivity)
    if sizes.size == 0:
        return np.asarray(mask).astype(bool)
    keep = np.flatnonzero(sizes >= max(1, int(min_size)))
    if keep.size == 0:
        return largest_component(mask, connectivity=connectivity)
    return np.isin(labels, keep)


# --------------------------------------------------------------------------- #
# smooth operators used by the implicit surface builder
# --------------------------------------------------------------------------- #
def smooth_min(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """Polynomial smooth minimum.

    Behaves like ``min(a, b)`` far from the crease and rounds the transition
    over ``radius``. Intersecting silhouette constraints with this instead of a
    hard ``min`` is what keeps generated shoulders and hips organic rather than
    faceted.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if radius <= 1e-9:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / radius, 0.0, 1.0)
    return b * (1.0 - h) + a * h - radius * h * (1.0 - h)


def smooth_max(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    return -smooth_min(-np.asarray(a), -np.asarray(b), radius)


def normalized(arr: np.ndarray) -> np.ndarray:
    """Rescale to 0..1, tolerating constant input."""
    arr = np.asarray(arr, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def trilinear_sample(volume: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample a 3D volume at fractional (x, y, z) index coordinates."""
    volume = np.asarray(volume, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    shape = np.array(volume.shape, dtype=np.float64) - 1.0
    p = np.clip(p, 0.0, shape)
    i0 = np.floor(p).astype(np.int64)
    i1 = np.minimum(i0 + 1, np.array(volume.shape) - 1)
    t = p - i0
    out = np.zeros(p.shape[0])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wx = t[:, 0] if dx else 1.0 - t[:, 0]
                wy = t[:, 1] if dy else 1.0 - t[:, 1]
                wz = t[:, 2] if dz else 1.0 - t[:, 2]
                xs = i1[:, 0] if dx else i0[:, 0]
                ys = i1[:, 1] if dy else i0[:, 1]
                zs = i1[:, 2] if dz else i0[:, 2]
                out += wx * wy * wz * volume[xs, ys, zs]
    return out


def bilinear_sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample a 2D (H, W) or (H, W, C) image at fractional (x, y) pixels."""
    image = np.asarray(image, dtype=np.float64)
    single_channel = image.ndim == 2
    if single_channel:
        image = image[:, :, None]
    h, w = image.shape[:2]
    x = np.clip(np.asarray(uv, dtype=np.float64)[:, 0], 0.0, w - 1.0)
    y = np.clip(np.asarray(uv, dtype=np.float64)[:, 1], 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = (x - x0)[:, None]
    ty = (y - y0)[:, None]
    top = image[y0, x0] * (1 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1 - tx) + image[y1, x1] * tx
    out = top * (1 - ty) + bottom * ty
    return out[:, 0] if single_channel else out
