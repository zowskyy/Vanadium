"""A small software rasteriser.

ConceptForge needs to be able to *show* its output without a GPU, a display, or
a browser: continuous integration has none of them, and neither does a render
farm node running a batch conversion. This module rasterises the generated
character to PNG contact sheets and animated GIFs so every run can prove what it
produced.

It is a conventional z-buffered triangle rasteriser: perspective projection,
per-vertex normals interpolated with perspective-correct barycentrics, a
three-light studio rig, optional albedo texture, and a soft ground shadow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from conceptforge.geometry.mesh import Mesh
from conceptforge.mathutil import look_at, normalize, perspective
from conceptforge.ndops import bilinear_sample


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_y: float = 32.0
    near: float = 0.02
    far: float = 100.0

    def matrices(self, aspect: float) -> tuple[np.ndarray, np.ndarray]:
        return look_at(self.eye, self.target, self.up), perspective(self.fov_y, aspect, self.near, self.far)


@dataclass
class Light:
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float = 1.0


@dataclass
class RenderSettings:
    width: int = 512
    height: int = 512
    background_top: tuple[float, float, float] = (0.13, 0.15, 0.19)
    background_bottom: tuple[float, float, float] = (0.04, 0.05, 0.07)
    ambient: tuple[float, float, float] = (0.24, 0.26, 0.30)
    lights: Sequence[Light] = field(
        default_factory=lambda: (
            Light((-0.45, -0.75, -0.50), (1.0, 0.97, 0.92), 1.05),
            Light((0.80, -0.25, 0.35), (0.55, 0.62, 0.80), 0.55),
            Light((0.10, 0.60, -0.85), (0.85, 0.80, 0.75), 0.35),
        )
    )
    specular: float = 0.22
    shininess: float = 24.0
    ground_shadow: bool = True
    shadow_strength: float = 0.5
    #: Flat colour used when the mesh has no texture and no vertex colours.
    base_color: tuple[float, float, float] = (0.72, 0.71, 0.70)
    #: Draw a faint contact grid so scale and grounding are legible.
    grid: bool = True


def render_mesh(
    mesh: Mesh,
    camera: Camera,
    settings: RenderSettings | None = None,
    texture: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterise ``mesh`` and return an ``(H, W, 3)`` float image in 0..1."""
    settings = settings or RenderSettings()
    width, height = int(settings.width), int(settings.height)
    image = _background(settings)
    depth = np.full((height, width), np.inf)

    if mesh.face_count == 0:
        return image

    if settings.grid or settings.ground_shadow:
        _draw_ground(image, depth, mesh, camera, settings)

    view, projection = camera.matrices(width / max(height, 1))
    clip = _project(mesh.vertices, view, projection)
    screen, w_clip, visible = _to_screen(clip, width, height)
    if not visible.any():
        return image

    normals = mesh.normals if mesh.normals is not None else mesh.copy().compute_normals().normals
    colors = mesh.colors
    uvs = mesh.uvs

    faces = mesh.faces
    # Cull back faces in screen space (signed area) and anything clipped.
    face_visible = visible[faces].all(axis=1)
    p0, p1, p2 = (screen[faces[:, k]] for k in range(3))
    area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    face_visible &= area < -1e-9  # negative because screen Y points down
    order = np.flatnonzero(face_visible)

    eye = np.asarray(camera.eye, dtype=np.float64)
    for f in order:
        tri = faces[f]
        _shade_triangle(
            image,
            depth,
            screen[tri],
            w_clip[tri],
            mesh.vertices[tri],
            normals[tri],
            None if uvs is None else uvs[tri],
            None if colors is None else colors[tri],
            texture,
            settings,
            eye,
        )
    return np.clip(image, 0.0, 1.0)


def _background(settings: RenderSettings) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, settings.height)[:, None, None]
    top = np.asarray(settings.background_top, dtype=np.float64).reshape(1, 1, 3)
    bottom = np.asarray(settings.background_bottom, dtype=np.float64).reshape(1, 1, 3)
    return np.broadcast_to(top * (1.0 - ramp) + bottom * ramp, (settings.height, settings.width, 3)).copy()


def _project(points: np.ndarray, view: np.ndarray, projection: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return homogeneous @ (projection @ view).T


def _to_screen(clip: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = clip[:, 3]
    valid = w > 1e-6
    safe_w = np.where(valid, w, 1.0)
    ndc = clip[:, :3] / safe_w[:, None]
    screen = np.empty((clip.shape[0], 3))
    screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * (height - 1)
    screen[:, 2] = ndc[:, 2]
    inside = valid & (ndc[:, 2] > -1.2) & (ndc[:, 2] < 1.2)
    return screen, safe_w, inside


def _shade_triangle(
    image: np.ndarray,
    depth: np.ndarray,
    screen: np.ndarray,
    w_clip: np.ndarray,
    world: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray | None,
    colors: np.ndarray | None,
    texture: np.ndarray | None,
    settings: RenderSettings,
    eye: np.ndarray,
) -> None:
    height, width = depth.shape
    x_min = max(0, int(np.floor(screen[:, 0].min())))
    x_max = min(width - 1, int(np.ceil(screen[:, 0].max())))
    y_min = max(0, int(np.floor(screen[:, 1].min())))
    y_max = min(height - 1, int(np.ceil(screen[:, 1].max())))
    if x_max < x_min or y_max < y_min:
        return

    ys, xs = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    px = xs + 0.5
    py = ys + 0.5

    ax, ay = screen[0, 0], screen[0, 1]
    bx, by = screen[1, 0], screen[1, 1]
    cx, cy = screen[2, 0], screen[2, 1]
    denom = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(denom) < 1e-12:
        return
    l1 = ((px - ax) * (cy - ay) - (cx - ax) * (py - ay)) / denom
    l2 = ((bx - ax) * (py - ay) - (px - ax) * (by - ay)) / denom
    l0 = 1.0 - l1 - l2
    inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
    if not inside.any():
        return

    bary = np.stack([l0, l1, l2], axis=-1)[inside]
    z = bary @ screen[:, 2]
    region = (slice(y_min, y_max + 1), slice(x_min, x_max + 1))
    current = depth[region][inside]
    closer = z < current
    if not closer.any():
        return

    bary = bary[closer]
    z = z[closer]
    # Perspective-correct interpolation of everything but position.
    inv_w = bary / w_clip[None, :]
    total = inv_w.sum(axis=1, keepdims=True)
    persp = inv_w / np.maximum(total, 1e-18)

    positions = persp @ world
    shading_normals = normalize(persp @ normals)

    albedo = np.broadcast_to(np.asarray(settings.base_color, dtype=np.float64), (bary.shape[0], 3)).copy()
    if colors is not None:
        albedo = persp @ colors
    if texture is not None and uvs is not None:
        uv = persp @ uvs
        tex_h, tex_w = texture.shape[:2]
        sample_xy = np.stack([uv[:, 0] * (tex_w - 1), uv[:, 1] * (tex_h - 1)], axis=1)
        sampled = bilinear_sample(texture[..., :3], sample_xy)
        albedo = np.clip(sampled, 0.0, 1.0)

    view_dir = normalize(eye[None, :] - positions)
    lit = np.asarray(settings.ambient, dtype=np.float64)[None, :] * albedo
    for light in settings.lights:
        direction = normalize(-np.asarray(light.direction, dtype=np.float64))
        ndotl = np.maximum(shading_normals @ direction, 0.0)[:, None]
        color = np.asarray(light.color, dtype=np.float64)[None, :] * light.intensity
        lit += albedo * color * ndotl
        if settings.specular > 0.0:
            half = normalize(direction[None, :] + view_dir)
            spec = np.maximum(np.einsum("ij,ij->i", shading_normals, half), 0.0) ** settings.shininess
            lit += color * (settings.specular * spec)[:, None]

    # Write back through the two masks.
    flat_index = np.flatnonzero(inside.ravel())[closer]
    target_rgb = image[region].reshape(-1, 3)
    target_depth = depth[region].reshape(-1)
    target_rgb[flat_index] = np.clip(lit, 0.0, 1.0)
    target_depth[flat_index] = z
    image[region] = target_rgb.reshape(y_max - y_min + 1, x_max - x_min + 1, 3)
    depth[region] = target_depth.reshape(y_max - y_min + 1, x_max - x_min + 1)


def _draw_ground(
    image: np.ndarray,
    depth: np.ndarray,
    mesh: Mesh,
    camera: Camera,
    settings: RenderSettings,
) -> None:
    """Project a ground quad with a radial contact shadow and a faint grid.

    Grounding cues matter more than they look: without them a floating render
    gives no sense of whether the feet actually reach Y=0.
    """
    lo, hi = mesh.bounds()
    radius = 1.6 * float(max(hi[0] - lo[0], hi[2] - lo[2], 0.4))
    centre = np.array([0.5 * (lo[0] + hi[0]), 0.0, 0.5 * (lo[2] + hi[2])])
    resolution = 96
    axis = np.linspace(-radius, radius, resolution)
    gx, gz = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack(
        [gx.ravel() + centre[0], np.zeros(gx.size), gz.ravel() + centre[2]], axis=1
    )

    view, projection = camera.matrices(settings.width / max(settings.height, 1))
    clip = _project(points, view, projection)
    screen, _, visible = _to_screen(clip, settings.width, settings.height)
    if not visible.any():
        return

    distance = np.linalg.norm(points[:, [0, 2]] - centre[[0, 2]][None, :], axis=1) / max(radius, 1e-9)
    fade = np.clip(1.0 - distance, 0.0, 1.0) ** 1.5
    shadow = np.clip(1.0 - distance * 2.6, 0.0, 1.0) ** 2 if settings.ground_shadow else np.zeros_like(fade)
    grid_line = np.zeros_like(fade)
    if settings.grid:
        cell = radius / 4.0
        for axis_values in (points[:, 0] - centre[0], points[:, 2] - centre[2]):
            phase = np.abs(np.mod(axis_values + 0.5 * cell, cell) - 0.5 * cell)
            grid_line = np.maximum(grid_line, np.clip(1.0 - phase / (0.04 * cell), 0.0, 1.0))

    base = np.array([0.10, 0.11, 0.13])
    color = base[None, :] * (0.5 + 0.5 * fade)[:, None]
    color += grid_line[:, None] * np.array([0.05, 0.06, 0.08])[None, :]
    color *= (1.0 - settings.shadow_strength * shadow)[:, None]
    coverage = fade * 0.95

    xs = np.rint(screen[:, 0]).astype(int)
    ys = np.rint(screen[:, 1]).astype(int)
    ok = visible & (xs >= 0) & (xs < settings.width) & (ys >= 0) & (ys < settings.height)
    # Splat with a small kernel: the ground grid is coarse relative to pixels.
    for dy in (-1, 0, 1, 2):
        for dx in (-1, 0, 1, 2):
            yy = np.clip(ys[ok] + dy, 0, settings.height - 1)
            xx = np.clip(xs[ok] + dx, 0, settings.width - 1)
            keep = screen[ok, 2] < depth[yy, xx]
            alpha = (coverage[ok] * keep)[:, None]
            image[yy, xx] = image[yy, xx] * (1.0 - alpha) + color[ok] * alpha
            depth[yy[keep], xx[keep]] = screen[ok, 2][keep]


# --------------------------------------------------------------------------- #
# framing helpers
# --------------------------------------------------------------------------- #
def frame_camera(
    mesh: Mesh,
    azimuth_deg: float = 20.0,
    elevation_deg: float = 8.0,
    margin: float = 1.22,
    fov_y: float = 32.0,
) -> Camera:
    """Place a camera that frames the whole mesh from a given angle."""
    lo, hi = mesh.bounds()
    centre = 0.5 * (lo + hi)
    extent = float(max(hi[1] - lo[1], hi[0] - lo[0], 0.2))
    distance = margin * 0.5 * extent / np.tan(np.radians(fov_y) * 0.5)
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    offset = np.array(
        [np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)], dtype=np.float64
    )
    return Camera(eye=centre + offset * distance, target=centre, fov_y=fov_y)


def turntable(
    mesh: Mesh,
    frames: int = 6,
    settings: RenderSettings | None = None,
    texture: np.ndarray | None = None,
    elevation_deg: float = 8.0,
) -> list[np.ndarray]:
    """Render ``frames`` views evenly spaced around the character."""
    images = []
    for i in range(int(frames)):
        azimuth = 360.0 * i / max(1, int(frames))
        camera = frame_camera(mesh, azimuth_deg=azimuth, elevation_deg=elevation_deg)
        images.append(render_mesh(mesh, camera, settings, texture))
    return images


def contact_sheet(images: Sequence[np.ndarray], columns: int | None = None, gap: int = 6) -> np.ndarray:
    """Tile images into a single sheet."""
    images = [np.asarray(im, dtype=np.float64) for im in images]
    if not images:
        return np.zeros((1, 1, 3))
    columns = columns or min(len(images), 4)
    rows = int(np.ceil(len(images) / columns))
    h, w = images[0].shape[:2]
    sheet = np.zeros((rows * h + (rows - 1) * gap, columns * w + (columns - 1) * gap, 3))
    sheet[:] = 0.02
    for index, image in enumerate(images):
        r, c = divmod(index, columns)
        y = r * (h + gap)
        x = c * (w + gap)
        sheet[y : y + h, x : x + w] = image[..., :3]
    return sheet


def write_gif(
    images: Iterable[np.ndarray], path: str | Path, fps: int = 12, loop: int = 0
) -> Path:
    """Write an animated GIF (used for the animation preview)."""
    from PIL import Image

    frames = [Image.fromarray(np.clip(np.asarray(im) * 255, 0, 255).astype(np.uint8)) for im in images]
    if not frames:
        raise ValueError("no frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantised = [f.convert("P", palette=Image.ADAPTIVE, colors=200) for f in frames]
    quantised[0].save(
        path,
        save_all=True,
        append_images=quantised[1:],
        duration=max(20, int(round(1000.0 / max(fps, 1)))),
        loop=loop,
        optimize=True,
        disposal=2,
    )
    return path
