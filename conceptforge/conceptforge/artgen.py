"""Procedural concept-art generator.

Shipping a working 2D-to-3D pipeline requires 2D input, and vendoring artwork
into a repository is a licensing problem. This module synthesises character
turnaround sheets (front / side / back, A-pose, flat colour regions) from a
small parameter set, which gives the test suite and the demos deterministic
input that exercises every stage: sheet splitting, view classification,
landmark detection, multi-view carving and texture projection.

It is a drawing tool, not a stage of the pipeline; nothing in
:mod:`conceptforge.pipeline` imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from conceptforge.imaging.raster import save_image

Color = tuple[float, float, float]


# --------------------------------------------------------------------------- #
# 2D signed distance primitives
# --------------------------------------------------------------------------- #
@dataclass
class Shape:
    """A drawable region: a signed distance function plus a bounding box."""

    sdf: Callable[[np.ndarray, np.ndarray], np.ndarray]
    bbox: tuple[float, float, float, float]
    color: Color
    softness: float = 1.0


def _bbox_union(points: Sequence[Sequence[float]], pad: float) -> tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=np.float64)
    return (
        float(pts[:, 0].min() - pad),
        float(pts[:, 1].min() - pad),
        float(pts[:, 0].max() + pad),
        float(pts[:, 1].max() + pad),
    )


def ellipse(center: Sequence[float], radii: Sequence[float], color: Color, rotation: float = 0.0) -> Shape:
    cx, cy = float(center[0]), float(center[1])
    rx, ry = float(radii[0]), float(radii[1])
    cos_t, sin_t = np.cos(-rotation), np.sin(-rotation)

    def sdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx, dy = x - cx, y - cy
        u = dx * cos_t - dy * sin_t
        v = dx * sin_t + dy * cos_t
        # Normalised implicit, rescaled so the gradient is roughly unit length
        # near the boundary (good enough for antialiased coverage).
        k = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        return (k - 1.0) * min(rx, ry)

    reach = max(rx, ry)
    return Shape(sdf, (cx - reach, cy - reach, cx + reach, cy + reach), color)


def capsule(a: Sequence[float], b: Sequence[float], radius: float, color: Color) -> Shape:
    """Exact distance to a segment with round caps."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab) or 1e-9

    def sdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        wx, wy = x - a[0], y - a[1]
        t = np.clip((wx * ab[0] + wy * ab[1]) / denom, 0.0, 1.0)
        return np.hypot(wx - t * ab[0], wy - t * ab[1]) - radius

    return Shape(sdf, _bbox_union([a, b], radius + 2.0), color)


def tapered_capsule(
    a: Sequence[float], b: Sequence[float], r1: float, r2: float, color: Color, steps: int | None = None
) -> Shape:
    """Union of discs swept along a segment with linearly varying radius.

    Sampling the sweep rather than solving the round-cone analytically keeps the
    code short; the step count is chosen so consecutive discs always overlap,
    which is what a naive fixed step count gets wrong for long thin limbs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if abs(r1 - r2) < 1e-9:
        return capsule(a, b, r1, color)
    length = float(np.linalg.norm(b - a))
    if steps is None:
        smallest = max(min(r1, r2), 0.5)
        steps = int(np.clip(np.ceil(4.0 * length / smallest), 8, 512))
    ts = np.linspace(0.0, 1.0, steps)
    centers = a[None, :] + ts[:, None] * (b - a)[None, :]
    radii = r1 + ts * (r2 - r1)

    def sdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        best = np.full(x.shape, np.inf)
        for (cx, cy), r in zip(centers, radii):
            np.minimum(best, np.hypot(x - cx, y - cy) - r, out=best)
        return best

    return Shape(sdf, _bbox_union([a, b], max(r1, r2) + 2.0), color)


def rounded_box(
    center: Sequence[float], half: Sequence[float], radius: float, color: Color
) -> Shape:
    cx, cy = float(center[0]), float(center[1])
    hx, hy = max(float(half[0]) - radius, 0.0), max(float(half[1]) - radius, 0.0)

    def sdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        qx = np.abs(x - cx) - hx
        qy = np.abs(y - cy) - hy
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside - radius

    reach = (float(half[0]) + 2.0, float(half[1]) + 2.0)
    return Shape(sdf, (cx - reach[0], cy - reach[1], cx + reach[0], cy + reach[1]), color)


def polygon(points: Sequence[Sequence[float]], color: Color, radius: float = 0.0) -> Shape:
    """Signed distance to a closed polygon, optionally corner-rounded.

    ``radius`` offsets the boundary outward, which rounds convex corners. This
    is how torsos are drawn: a tapered capsule would balloon its round cap above
    the shoulder line and swallow the neck.
    """
    pts = np.asarray(points, dtype=np.float64)

    def sdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = np.full(x.shape, np.inf)
        inside = np.zeros(x.shape, dtype=bool)
        n = pts.shape[0]
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            e = p1 - p0
            wx, wy = x - p0[0], y - p0[1]
            denom = float(e @ e) or 1e-9
            t = np.clip((wx * e[0] + wy * e[1]) / denom, 0.0, 1.0)
            np.minimum(d, np.hypot(wx - t * e[0], wy - t * e[1]), out=d)
            # Even-odd test: does a ray cast in +x cross this edge?
            straddles = (p0[1] > y) != (p1[1] > y)
            if abs(e[1]) > 1e-12:
                x_cross = p0[0] + wy * (e[0] / e[1])
                inside ^= straddles & (x < x_cross)
        return np.where(inside, -d, d) - radius

    return Shape(sdf, _bbox_union(pts, radius + 2.0), color)


# --------------------------------------------------------------------------- #
# character parameters
# --------------------------------------------------------------------------- #
@dataclass
class Palette:
    skin: Color = (0.86, 0.68, 0.55)
    hair: Color = (0.24, 0.16, 0.12)
    torso: Color = (0.24, 0.42, 0.62)
    torso_shadow: Color = (0.18, 0.32, 0.50)
    legs: Color = (0.30, 0.30, 0.36)
    boots: Color = (0.20, 0.15, 0.13)
    gloves: Color = (0.42, 0.28, 0.18)
    belt: Color = (0.32, 0.22, 0.14)
    accent: Color = (0.82, 0.66, 0.24)
    cape: Color = (0.52, 0.16, 0.20)


@dataclass
class CharacterSpec:
    """Proportions expressed as fractions of total figure height."""

    name: str = "hero"
    head_height: float = 0.125
    head_width: float = 0.093
    neck_width: float = 0.040
    shoulder_width: float = 0.230
    chest_width: float = 0.190
    waist_width: float = 0.145
    hip_width: float = 0.175
    arm_spread: float = 0.62
    """A-pose openness: 0 = arms straight down, 1 = T-pose."""
    upper_arm: float = 0.175
    lower_arm: float = 0.165
    arm_thickness: float = 0.036
    leg_gap: float = 0.052
    thigh_thickness: float = 0.056
    calf_thickness: float = 0.042
    foot_length: float = 0.070
    depth: float = 0.155
    """Body depth (front-to-back) as a fraction of height, for the side view."""
    cape: bool = False
    palette: Palette = field(default_factory=Palette)


PRESETS: dict[str, CharacterSpec] = {
    "hero": CharacterSpec(name="hero"),
    "scout": CharacterSpec(
        name="scout",
        head_height=0.118,
        shoulder_width=0.200,
        chest_width=0.165,
        waist_width=0.125,
        hip_width=0.155,
        arm_spread=0.52,
        arm_thickness=0.030,
        thigh_thickness=0.048,
        calf_thickness=0.036,
        depth=0.130,
        palette=Palette(
            torso=(0.28, 0.46, 0.32),
            torso_shadow=(0.20, 0.36, 0.25),
            legs=(0.36, 0.32, 0.24),
            hair=(0.52, 0.34, 0.14),
            accent=(0.78, 0.72, 0.40),
            gloves=(0.34, 0.26, 0.16),
        ),
    ),
    "brute": CharacterSpec(
        name="brute",
        head_height=0.115,
        head_width=0.100,
        neck_width=0.058,
        shoulder_width=0.300,
        chest_width=0.250,
        waist_width=0.205,
        hip_width=0.215,
        arm_spread=0.72,
        upper_arm=0.180,
        lower_arm=0.160,
        arm_thickness=0.052,
        thigh_thickness=0.072,
        calf_thickness=0.056,
        depth=0.210,
        palette=Palette(
            skin=(0.62, 0.72, 0.52),
            torso=(0.42, 0.26, 0.20),
            torso_shadow=(0.32, 0.19, 0.15),
            legs=(0.26, 0.24, 0.26),
            hair=(0.14, 0.12, 0.12),
            accent=(0.70, 0.58, 0.22),
        ),
    ),
    "mage": CharacterSpec(
        name="mage",
        head_height=0.122,
        shoulder_width=0.205,
        chest_width=0.180,
        waist_width=0.150,
        hip_width=0.190,
        arm_spread=0.58,
        arm_thickness=0.033,
        thigh_thickness=0.058,
        calf_thickness=0.044,
        depth=0.165,
        cape=True,
        palette=Palette(
            torso=(0.32, 0.24, 0.52),
            torso_shadow=(0.24, 0.18, 0.42),
            legs=(0.28, 0.22, 0.42),
            hair=(0.80, 0.78, 0.74),
            accent=(0.86, 0.74, 0.32),
            cape=(0.20, 0.16, 0.38),
        ),
    ),
}


# --------------------------------------------------------------------------- #
# view construction
# --------------------------------------------------------------------------- #
def _landmark_ys(spec: CharacterSpec, height: float, top: float) -> dict[str, float]:
    h = height
    chin = top + spec.head_height * h
    shoulder = chin + 0.062 * h
    return {
        "crown": top,
        "chin": chin,
        "shoulder": shoulder,
        "chest": shoulder + 0.090 * h,
        "waist": shoulder + 0.215 * h,
        "hip": shoulder + 0.320 * h,
        "knee": top + 0.735 * h,
        "ankle": top + 0.945 * h,
        "ground": top + h,
    }


def _front_shapes(spec: CharacterSpec, height: float, top: float, cx: float, mirror: bool) -> list[Shape]:
    h = height
    p = spec.palette
    y = _landmark_ys(spec, height, top)
    shoulder_half = 0.5 * spec.shoulder_width * h
    hip_half = 0.5 * spec.hip_width * h
    shapes: list[Shape] = []

    if spec.cape:
        shapes.append(
            polygon(
                [
                    (cx - shoulder_half * 1.05, y["shoulder"]),
                    (cx + shoulder_half * 1.05, y["shoulder"]),
                    (cx + shoulder_half * 1.55, y["knee"]),
                    (cx - shoulder_half * 1.55, y["knee"]),
                ],
                p.cape,
            )
        )

    # legs
    for side in (-1, 1):
        gap = side * spec.leg_gap * h
        hip_x = cx + side * hip_half * 0.52
        knee_x = cx + gap
        ankle_x = cx + gap
        shapes.append(
            tapered_capsule(
                (hip_x, y["hip"] - 0.01 * h),
                (knee_x, y["knee"]),
                spec.thigh_thickness * h,
                spec.calf_thickness * h * 1.05,
                p.legs,
            )
        )
        shapes.append(
            tapered_capsule(
                (knee_x, y["knee"]),
                (ankle_x, y["ankle"]),
                spec.calf_thickness * h * 1.05,
                spec.calf_thickness * h * 0.80,
                p.legs,
            )
        )
        shapes.append(
            rounded_box(
                (ankle_x + side * 0.004 * h, y["ground"] - 0.022 * h),
                (spec.calf_thickness * h * 0.95, 0.026 * h),
                0.014 * h,
                p.boots,
            )
        )

    # torso: a rounded trapezoid through shoulder / chest / waist / hip widths
    corner = 0.018 * h
    chest_half = 0.5 * spec.chest_width * h
    waist_half_w = 0.5 * spec.waist_width * h
    torso_pts = [
        (cx - chest_half + corner, y["shoulder"] + corner),
        (cx + chest_half - corner, y["shoulder"] + corner),
        (cx + chest_half * 0.99 - corner, y["chest"]),
        (cx + waist_half_w - corner, y["waist"]),
        (cx + hip_half * 0.92 - corner, y["hip"]),
        (cx - hip_half * 0.92 + corner, y["hip"]),
        (cx - waist_half_w + corner, y["waist"]),
        (cx - chest_half * 0.99 + corner, y["chest"]),
    ]
    shapes.append(polygon(torso_pts, p.torso, radius=corner))
    shapes.append(
        polygon(
            [
                (cx - waist_half_w + corner, y["waist"]),
                (cx + waist_half_w - corner, y["waist"]),
                (cx + hip_half * 0.92 - corner, y["hip"] + 0.012 * h),
                (cx - hip_half * 0.92 + corner, y["hip"] + 0.012 * h),
            ],
            p.torso_shadow if not mirror else p.torso,
            radius=corner,
        )
    )
    shapes.append(
        rounded_box((cx, y["waist"] + 0.012 * h), (0.5 * spec.waist_width * h * 1.02, 0.020 * h), 0.008 * h, p.belt)
    )
    if not mirror:
        shapes.append(
            rounded_box(
                (cx, y["chest"] + 0.01 * h),
                (0.5 * spec.chest_width * h * 0.42, 0.055 * h),
                0.012 * h,
                p.accent,
            )
        )

    # arms, opened by arm_spread
    angle = np.radians(12.0 + 60.0 * spec.arm_spread)
    for side in (-1, 1):
        shoulder_pt = np.array([cx + side * shoulder_half * 0.92, y["shoulder"] + 0.012 * h])
        direction = np.array([side * np.sin(angle), np.cos(angle)])
        elbow = shoulder_pt + direction * spec.upper_arm * h
        fore_dir = np.array([side * np.sin(angle * 0.72), np.cos(angle * 0.72)])
        wrist = elbow + fore_dir * spec.lower_arm * h
        shapes.append(
            tapered_capsule(
                shoulder_pt, elbow, spec.arm_thickness * h * 1.12, spec.arm_thickness * h * 0.92, p.torso
            )
        )
        shapes.append(
            tapered_capsule(
                elbow, wrist, spec.arm_thickness * h * 0.92, spec.arm_thickness * h * 0.74, p.skin
            )
        )
        shapes.append(ellipse(wrist + fore_dir * 0.022 * h, (spec.arm_thickness * h * 0.86,) * 2, p.gloves))
        shapes.append(ellipse(shoulder_pt, (spec.arm_thickness * h * 1.22,) * 2, p.torso))

    # neck and head
    shapes.append(
        capsule((cx, y["chin"] - 0.012 * h), (cx, y["shoulder"] + 0.012 * h), 0.5 * spec.neck_width * h, p.skin)
    )
    head_c = (cx, top + 0.5 * spec.head_height * h + 0.004 * h)
    head_r = (0.5 * spec.head_width * h, 0.5 * spec.head_height * h)
    shapes.append(ellipse(head_c, head_r, p.skin))
    shapes.append(
        ellipse(
            (head_c[0], head_c[1] - 0.018 * h),
            (head_r[0] * 1.06, head_r[1] * 0.72),
            p.hair,
        )
    )
    if not mirror:
        eye_dx = head_r[0] * 0.42
        eye_y = head_c[1] + head_r[1] * 0.08
        for side in (-1, 1):
            shapes.append(
                ellipse((head_c[0] + side * eye_dx, eye_y), (head_r[0] * 0.16, head_r[1] * 0.10), (0.10, 0.10, 0.12))
            )
    return shapes


def _side_shapes(spec: CharacterSpec, height: float, top: float, cx: float) -> list[Shape]:
    """Profile view: depth silhouette with a nose, hair mass and a boot toe."""
    h = height
    p = spec.palette
    y = _landmark_ys(spec, height, top)
    depth = spec.depth * h
    shapes: list[Shape] = []

    if spec.cape:
        shapes.append(
            polygon(
                [
                    (cx - depth * 0.30, y["shoulder"]),
                    (cx - depth * 0.62, y["chest"]),
                    (cx - depth * 0.75, y["knee"]),
                    (cx - depth * 0.10, y["knee"]),
                ],
                p.cape,
            )
        )

    # single leg silhouette (legs overlap in profile)
    shapes.append(
        tapered_capsule(
            (cx + depth * 0.02, y["hip"] - 0.01 * h),
            (cx - depth * 0.04, y["knee"]),
            spec.thigh_thickness * h * 1.05,
            spec.calf_thickness * h * 1.05,
            p.legs,
        )
    )
    shapes.append(
        tapered_capsule(
            (cx - depth * 0.04, y["knee"]),
            (cx + depth * 0.02, y["ankle"]),
            spec.calf_thickness * h * 1.05,
            spec.calf_thickness * h * 0.82,
            p.legs,
        )
    )
    shapes.append(
        rounded_box(
            (cx + depth * 0.16, y["ground"] - 0.022 * h),
            (spec.foot_length * h * 0.62, 0.026 * h),
            0.014 * h,
            p.boots,
        )
    )

    # torso profile: chest forward, small lumbar curve
    corner = 0.016 * h
    shapes.append(
        polygon(
            [
                (cx - depth * 0.42 + corner, y["shoulder"] + corner),
                (cx + depth * 0.54 - corner, y["shoulder"] + corner),
                (cx + depth * 0.58 - corner, y["chest"]),
                (cx + depth * 0.40 - corner, y["waist"]),
                (cx + depth * 0.46 - corner, y["hip"]),
                (cx - depth * 0.52 + corner, y["hip"]),
                (cx - depth * 0.46 + corner, y["waist"]),
                (cx - depth * 0.50 + corner, y["chest"]),
            ],
            p.torso,
            radius=corner,
        )
    )
    shapes.append(
        polygon(
            [
                (cx - depth * 0.46 + corner, y["waist"]),
                (cx + depth * 0.40 - corner, y["waist"]),
                (cx + depth * 0.46 - corner, y["hip"] + 0.012 * h),
                (cx - depth * 0.52 + corner, y["hip"] + 0.012 * h),
            ],
            p.torso_shadow,
            radius=corner,
        )
    )
    shapes.append(
        rounded_box((cx, y["waist"] + 0.012 * h), (depth * 0.47, 0.020 * h), 0.008 * h, p.belt)
    )

    # arm hangs beside the body, so in profile it reads as one limb
    angle = np.radians(6.0 + 12.0 * spec.arm_spread)
    shoulder_pt = np.array([cx + depth * 0.04, y["shoulder"] + 0.012 * h])
    elbow = shoulder_pt + np.array([-np.sin(angle), np.cos(angle)]) * spec.upper_arm * h
    wrist = elbow + np.array([np.sin(angle * 1.4), np.cos(angle * 1.4)]) * spec.lower_arm * h
    shapes.append(tapered_capsule(shoulder_pt, elbow, spec.arm_thickness * h * 1.10, spec.arm_thickness * h * 0.92, p.torso))
    shapes.append(tapered_capsule(elbow, wrist, spec.arm_thickness * h * 0.92, spec.arm_thickness * h * 0.76, p.skin))
    shapes.append(ellipse(wrist, (spec.arm_thickness * h * 0.84,) * 2, p.gloves))

    # neck, head, face profile
    shapes.append(
        capsule(
            (cx - depth * 0.02, y["chin"] - 0.012 * h),
            (cx, y["shoulder"] + 0.012 * h),
            0.5 * spec.neck_width * h * 1.05,
            p.skin,
        )
    )
    head_c = (cx - depth * 0.02, top + 0.5 * spec.head_height * h + 0.004 * h)
    head_r = (0.5 * spec.head_height * h * 0.86, 0.5 * spec.head_height * h)
    shapes.append(ellipse(head_c, head_r, p.skin))
    shapes.append(
        polygon(
            [
                (head_c[0] + head_r[0] * 0.60, head_c[1] - head_r[1] * 0.05),
                (head_c[0] + head_r[0] * 1.30, head_c[1] + head_r[1] * 0.16),
                (head_c[0] + head_r[0] * 0.60, head_c[1] + head_r[1] * 0.34),
            ],
            p.skin,
        )
    )
    shapes.append(
        ellipse(
            (head_c[0] - head_r[0] * 0.30, head_c[1] - head_r[1] * 0.16),
            (head_r[0] * 1.05, head_r[1] * 0.78),
            p.hair,
        )
    )
    return shapes


def render_shapes(
    shapes: Sequence[Shape],
    size: tuple[int, int],
    background: Color | None = None,
) -> np.ndarray:
    """Painter's-algorithm rasteriser with antialiased coverage.

    Returns an ``(H, W, 4)`` float array. With ``background=None`` the result is
    transparent outside the character.
    """
    width, height = int(size[0]), int(size[1])
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    alpha = np.zeros((height, width), dtype=np.float64)
    if background is not None:
        rgb[:] = np.asarray(background, dtype=np.float64)
        alpha[:] = 1.0

    for shape in shapes:
        x0 = max(0, int(np.floor(shape.bbox[0])))
        y0 = max(0, int(np.floor(shape.bbox[1])))
        x1 = min(width, int(np.ceil(shape.bbox[2])) + 1)
        y1 = min(height, int(np.ceil(shape.bbox[3])) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        d = shape.sdf(xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5)
        coverage = np.clip(0.5 - d / max(shape.softness, 1e-6), 0.0, 1.0)
        if not coverage.any():
            continue
        cov = coverage[..., None]
        color = np.asarray(shape.color, dtype=np.float64).reshape(1, 1, 3)
        region_rgb = rgb[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1] = region_rgb * (1.0 - cov) + color * cov
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], coverage)

    return np.concatenate([rgb, alpha[..., None]], axis=2)


def _shade(rgba: np.ndarray, strength: float = 0.22) -> np.ndarray:
    """Add a soft directional gradient so the art is not perfectly flat."""
    h, w = rgba.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    ramp = 0.5 + 0.5 * (xs / max(w - 1, 1) - 0.5) * 2.0
    vertical = 1.0 - 0.35 * (ys / max(h - 1, 1))
    factor = (1.0 - strength) + strength * (0.35 + 0.65 * (1.0 - ramp)) * vertical
    out = rgba.copy()
    out[..., :3] = np.clip(out[..., :3] * factor[..., None], 0.0, 1.0)
    return out


def generate_sheet(
    spec: CharacterSpec | str = "hero",
    figure_height: int = 900,
    views: Sequence[str] = ("front", "side", "back"),
    background: Color | None = (0.93, 0.93, 0.90),
    shading: bool = True,
) -> np.ndarray:
    """Render a multi-view turnaround sheet as an RGBA array."""
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"unknown character preset {spec!r}; have {sorted(PRESETS)}")
        spec = PRESETS[spec]

    figure_height = int(figure_height)
    pad_y = int(0.07 * figure_height)
    pad_x = max(8, int(0.045 * figure_height))
    panel_h = figure_height + 2 * pad_y
    gutter = max(12, int(0.05 * figure_height))
    order = [v for v in views if v in ("front", "side", "back")]

    # Lay each panel out from the true extent of its shapes, so a wide A-pose or
    # a trailing cape is never clipped by the panel edge. Clipped limbs would
    # both look wrong and defeat the landmark detector downstream.
    panels: list[np.ndarray] = []
    for view in order:
        builder = (
            (lambda cx: _side_shapes(spec, figure_height, pad_y, cx))
            if view == "side"
            else (lambda cx, v=view: _front_shapes(spec, figure_height, pad_y, cx, mirror=(v == "back")))
        )
        probe = builder(0.0)
        x_min = min(s.bbox[0] for s in probe)
        x_max = max(s.bbox[2] for s in probe)
        panel_w = int(np.ceil(x_max - x_min)) + 2 * pad_x
        panel = render_shapes(builder(pad_x - x_min), (panel_w, panel_h), background=background)
        panels.append(_shade(panel) if shading else panel)

    total_w = sum(p.shape[1] for p in panels) + gutter * (len(panels) - 1)
    sheet = np.zeros((panel_h, total_w, 4), dtype=np.float64)
    if background is not None:
        sheet[..., :3] = np.asarray(background, dtype=np.float64)
        sheet[..., 3] = 1.0
    x_cursor = 0
    for panel in panels:
        sheet[:, x_cursor : x_cursor + panel.shape[1]] = panel
        x_cursor += panel.shape[1] + gutter
    return np.clip(sheet, 0.0, 1.0)


def write_sheet(
    path: str | Path,
    spec: CharacterSpec | str = "hero",
    figure_height: int = 900,
    views: Sequence[str] = ("front", "side", "back"),
    background: Color | None = (0.93, 0.93, 0.90),
) -> Path:
    sheet = generate_sheet(spec, figure_height=figure_height, views=views, background=background)
    return save_image(sheet, path)
