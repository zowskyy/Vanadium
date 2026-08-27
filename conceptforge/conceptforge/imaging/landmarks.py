"""Read anatomy out of a character silhouette.

Everything the auto-rigger needs is derived here: where the neck, shoulders,
hips and joints are, in the pixel space of the front view. The analysis is
silhouette-only (no learned model), which makes it deterministic and auditable
- important when a rig has to be signed off for production.

Coordinate note
---------------
Image ``x`` increases to the right of the front view. ConceptForge exports with
the character facing ``+Z``, so screen right is ``+X`` in world space, which is
the character's **left** side. Landmarks suffixed ``_l`` therefore live at
larger image ``x`` than those suffixed ``_r``.

The detector expects the relaxed A-pose or T-pose that character turnarounds are
drawn in. When a feature cannot be found (a robe hiding the legs, arms fused to
the torso) it falls back to canonical figure-drawing proportions and records
lowered confidence rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from conceptforge import ndops
from conceptforge.imaging.raster import mask_bounds

#: Canonical fractions of total figure height, measured from the top of the
#: head. Standard eight-head heroic proportions, used as priors and fallbacks.
CANONICAL = {
    "chin": 0.128,
    "shoulder": 0.185,
    "chest": 0.275,
    "waist": 0.395,
    "hip": 0.505,
    "knee": 0.735,
    "ankle": 0.955,
    "shoulder_half_width": 0.105,
    "hip_half_width": 0.070,
    "elbow": 0.365,
    "wrist": 0.475,
}


@dataclass
class CharacterLandmarks:
    """Anatomical landmarks in front-view pixel coordinates."""

    height_px: float
    width_px: float
    center_x: float
    top_y: float
    bottom_y: float

    chin_y: float
    shoulder_y: float
    chest_y: float
    waist_y: float
    hip_y: float

    head_center: np.ndarray
    head_radius: float

    shoulder_l: np.ndarray
    shoulder_r: np.ndarray
    elbow_l: np.ndarray
    elbow_r: np.ndarray
    hand_l: np.ndarray
    hand_r: np.ndarray

    hip_l: np.ndarray
    hip_r: np.ndarray
    knee_l: np.ndarray
    knee_r: np.ndarray
    ankle_l: np.ndarray
    ankle_r: np.ndarray
    toe_l: np.ndarray
    toe_r: np.ndarray

    confidence: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        if not self.confidence:
            return 0.0
        return float(np.mean(list(self.confidence.values())))

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                out[key] = [round(float(v), 2) for v in value.tolist()]
            elif isinstance(value, float):
                out[key] = round(value, 2)
            else:
                out[key] = value
        return out

    def scaled(self, factor: float) -> "CharacterLandmarks":
        """Uniformly scale every landmark (used when views are resampled)."""
        kwargs = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                kwargs[key] = value * factor
            elif isinstance(value, float) and key not in ("",):
                kwargs[key] = value * factor
            else:
                kwargs[key] = value
        return CharacterLandmarks(**kwargs)


# --------------------------------------------------------------------------- #
# silhouette primitives
# --------------------------------------------------------------------------- #
def row_runs(mask: np.ndarray, y: int) -> list[tuple[int, int]]:
    """Horizontal runs of foreground in row ``y`` as ``(start, end)`` pairs."""
    row = np.asarray(mask[int(y)]).astype(bool)
    if not row.any():
        return []
    padded = np.concatenate([[False], row, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def run_counts(mask: np.ndarray) -> np.ndarray:
    """Number of foreground runs in each row (vectorised over all rows)."""
    mask = np.asarray(mask).astype(bool)
    padded = np.pad(mask, ((0, 0), (1, 0)), constant_values=False)
    return (mask & ~padded[:, :-1]).sum(axis=1)


def row_widths(mask: np.ndarray) -> np.ndarray:
    """Foreground pixel count per row."""
    return np.asarray(mask).astype(bool).sum(axis=1).astype(np.float64)


def center_run_widths(mask: np.ndarray, center_x: float) -> np.ndarray:
    """Width of the foreground run straddling ``center_x``, for every row.

    This is the profile the detectors reason about. A plain per-row foreground
    count is useless on an A-pose because the arms add pixels beside the torso
    and turn the waist into a local *maximum*; the run through the centre axis
    tracks the torso alone.
    """
    mask = np.asarray(mask).astype(bool)
    cx = int(np.clip(round(center_x), 0, mask.shape[1] - 1))
    inside = mask[:, cx]

    # Distance to the nearest background pixel left of / right of the axis.
    left_gap = ~mask[:, cx::-1]
    right_gap = ~mask[:, cx:]
    left_hit = left_gap.any(axis=1)
    right_hit = right_gap.any(axis=1)
    left = np.where(left_hit, cx - np.argmax(left_gap, axis=1), -1)
    right = np.where(right_hit, cx + np.argmax(right_gap, axis=1), mask.shape[1])
    return np.where(inside, (right - left - 1).astype(np.float64), 0.0)


def row_extents(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leftmost and rightmost foreground column per row (-1 where empty)."""
    mask = np.asarray(mask).astype(bool)
    any_row = mask.any(axis=1)
    first = np.argmax(mask, axis=1)
    last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    return np.where(any_row, first, -1), np.where(any_row, last, -1)


def run_containing(runs: Sequence[tuple[int, int]], x: float) -> tuple[int, int] | None:
    for start, end in runs:
        if start <= x < end:
            return start, end
    return None


def symmetry_axis(mask: np.ndarray) -> float:
    """Best vertical mirror axis, found by maximising mirror overlap.

    Starts from the area centroid and searches a small window, which handles
    characters whose silhouette is unbalanced by a cape or weapon.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask.shape[1] * 0.5
    col_weight = mask.sum(axis=0).astype(np.float64)
    centroid = float(np.average(np.arange(mask.shape[1]), weights=col_weight))
    x0, _, x1, _ = mask_bounds(mask)
    search = max(2, int(round(0.06 * (x1 - x0))))
    best_axis, best_score = centroid, -1.0
    for offset in range(-search, search + 1):
        axis = centroid + offset
        shift = int(round(2.0 * axis))
        mirrored = _mirror_about(mask, shift)
        union = float((mask | mirrored).sum())
        score = float((mask & mirrored).sum()) / union if union else 0.0
        if score > best_score:
            best_axis, best_score = axis, score
    return best_axis


def _mirror_about(mask: np.ndarray, shift: int) -> np.ndarray:
    """Mirror columns about ``shift / 2`` (integer pixel accuracy)."""
    width = mask.shape[1]
    src = shift - np.arange(width)
    valid = (src >= 0) & (src < width)
    out = np.zeros_like(mask)
    out[:, valid] = mask[:, src[valid]]
    return out


def symmetrize(mask: np.ndarray, axis: float, strength: float = 1.0) -> np.ndarray:
    """Blend a mask with its mirror image about ``axis``.

    ``strength`` 1 takes the union (fully symmetric), 0 leaves the mask alone.
    Intermediate values keep asymmetric detail only where it is substantial,
    which preserves an over-the-shoulder cape while fixing a hand drawn slightly
    higher than the other.
    """
    mask = np.asarray(mask).astype(bool)
    if strength <= 0.0:
        return mask
    mirrored = _mirror_about(mask, int(round(2.0 * axis)))
    if strength >= 1.0:
        return mask | mirrored
    blended = mask.astype(np.float64) * (1.0 - strength * 0.5) + mirrored.astype(np.float64) * (
        strength * 0.5
    )
    return blended >= 0.5 - 1e-9


# --------------------------------------------------------------------------- #
# limb tracing
# --------------------------------------------------------------------------- #
def trace_limb_centerline(
    mask: np.ndarray, start: np.ndarray, end: np.ndarray, samples: int = 24
) -> np.ndarray:
    """Medial curve of a limb between two endpoints.

    Walks the straight line from ``start`` to ``end`` and, at each step, centres
    the sample inside the silhouette along the perpendicular. The result follows
    a bent arm or leg instead of cutting the corner, so the derived elbow and
    knee sit inside the mesh.
    """
    mask = np.asarray(mask).astype(bool)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return np.repeat(start[None, :], samples, axis=0)
    normal = np.array([-direction[1], direction[0]]) / length
    search = max(3, int(round(0.35 * length)))
    height, width = mask.shape
    points = []
    for t in np.linspace(0.0, 1.0, samples):
        base = start + t * direction
        offsets = np.arange(-search, search + 1, dtype=np.float64)
        candidates = base[None, :] + offsets[:, None] * normal[None, :]
        xi = np.clip(np.rint(candidates[:, 0]).astype(int), 0, width - 1)
        yi = np.clip(np.rint(candidates[:, 1]).astype(int), 0, height - 1)
        inside = mask[yi, xi]
        centre_index = search
        if inside[centre_index]:
            lo = centre_index
            while lo > 0 and inside[lo - 1]:
                lo -= 1
            hi = centre_index
            while hi < inside.size - 1 and inside[hi + 1]:
                hi += 1
            points.append(base + 0.5 * (offsets[lo] + offsets[hi]) * normal)
        elif inside.any():
            best = int(np.flatnonzero(inside)[np.argmin(np.abs(np.flatnonzero(inside) - centre_index))])
            points.append(candidates[best])
        else:
            points.append(base)
    curve = np.asarray(points, dtype=np.float64)
    # Light smoothing removes the jitter from pixel-quantised centring.
    if curve.shape[0] >= 5:
        kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
        for axis in (0, 1):
            padded = np.pad(curve[:, axis], (2, 2), mode="edge")
            curve[:, axis] = np.convolve(padded, kernel, mode="valid")
    curve[0] = start
    curve[-1] = end
    return curve


def curve_point_at(curve: np.ndarray, fraction: float) -> np.ndarray:
    """Point at ``fraction`` of the arc length along a polyline."""
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape[0] == 1:
        return curve[0].copy()
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-9:
        return curve[0].copy()
    target = np.clip(fraction, 0.0, 1.0) * total
    acc = np.concatenate([[0.0], np.cumsum(seg)])
    i = int(np.searchsorted(acc, target, side="right")) - 1
    i = int(np.clip(i, 0, curve.shape[0] - 2))
    local = (target - acc[i]) / max(seg[i], 1e-9)
    return curve[i] + local * (curve[i + 1] - curve[i])


def perpendicular_widths(
    mask: np.ndarray, start: np.ndarray, end: np.ndarray, samples: int = 48, reach: float | None = None
) -> np.ndarray:
    """Width of the silhouette across the ``start``-``end`` line, per sample.

    Marching along a limb and watching this profile is how the shoulder and hip
    joints are found: limb width is near constant, then jumps where the limb
    meets the body mass.
    """
    mask = np.asarray(mask).astype(bool)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return np.zeros(samples)
    normal = np.array([-direction[1], direction[0]]) / length
    span = int(round(reach if reach is not None else 0.75 * length))
    span = max(4, span)
    offsets = np.arange(-span, span + 1, dtype=np.float64)
    height, width = mask.shape
    out = np.zeros(samples)
    for i, t in enumerate(np.linspace(0.0, 1.0, samples)):
        base = start + t * direction
        candidates = base[None, :] + offsets[:, None] * normal[None, :]
        xi = np.clip(np.rint(candidates[:, 0]).astype(int), 0, width - 1)
        yi = np.clip(np.rint(candidates[:, 1]).astype(int), 0, height - 1)
        inside = mask[yi, xi]
        if not inside[span]:
            out[i] = 0.0
            continue
        lo = span
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        hi = span
        while hi < inside.size - 1 and inside[hi + 1]:
            hi += 1
        out[i] = float(offsets[hi] - offsets[lo])
    return out


def shoulder_from_arm(
    mask: np.ndarray, hand: np.ndarray, neck_base: np.ndarray, height: float
) -> np.ndarray | None:
    """Find the shoulder joint by walking inward from the hand.

    Silhouette width alone cannot separate a deltoid from a torso, and it cannot
    see a shoulder at all under a robe. Walking the arm inward can: the arm has a
    roughly constant width, and the row where that width suddenly grows is where
    the arm enters the body. That transition is the shoulder.
    """
    hand = np.asarray(hand, dtype=np.float64)
    neck_base = np.asarray(neck_base, dtype=np.float64)
    widths = perpendicular_widths(mask, hand, neck_base, samples=56)
    valid = widths > 0
    if valid.sum() < 12:
        return None
    # Use the middle of the arm as the reference: the hand end tapers and the
    # inner end is already contaminated by the body.
    forearm = widths[6:22]
    forearm = forearm[forearm > 0]
    if forearm.size < 4:
        return None
    arm_width = float(np.median(forearm))
    threshold = max(arm_width * 1.7, arm_width + 0.018 * height)
    crossing = np.flatnonzero(widths > threshold)
    if crossing.size == 0:
        return None
    index = int(crossing[0])
    if index < 8:  # the "arm" was never distinguishable from the body
        return None
    t = index / float(widths.size - 1)
    return hand + t * (neck_base - hand)


def farthest_lateral_point(
    mask: np.ndarray, center_x: float, side: int, y_range: tuple[int, int]
) -> np.ndarray | None:
    """Silhouette point furthest from the centre axis on one side.

    ``side`` is +1 for larger x (the character's left). This robustly finds hands
    in both A-pose and T-pose, where the hands are always the laterally extreme
    features above the hips.
    """
    mask = np.asarray(mask).astype(bool)
    y0, y1 = int(max(0, y_range[0])), int(min(mask.shape[0], y_range[1]))
    if y1 <= y0:
        return None
    band = mask[y0:y1]
    if not band.any():
        return None
    xs = np.arange(mask.shape[1], dtype=np.float64)
    signed = (xs - center_x) * side
    scored = np.where(band, signed[None, :], -np.inf)
    flat = int(np.argmax(scored))
    y, x = np.unravel_index(flat, scored.shape)
    if not np.isfinite(scored[y, x]):
        return None
    return np.array([float(x), float(y + y0)])


# --------------------------------------------------------------------------- #
# main detector
# --------------------------------------------------------------------------- #
def detect_landmarks(mask: np.ndarray) -> CharacterLandmarks:
    """Locate anatomical landmarks in a front-view silhouette."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        raise ValueError("cannot detect landmarks in an empty silhouette")

    x0, y0, x1, y1 = mask_bounds(mask)
    height = float(y1 - y0)
    width = float(x1 - x0)
    center_x = symmetry_axis(mask)
    confidence: dict[str, float] = {}
    notes: list[str] = []

    torso_profile = center_run_widths(mask, center_x)
    counts = run_counts(mask)

    def canonical(key: str) -> float:
        return y0 + CANONICAL[key] * height

    split_y, leg_runs = _find_leg_split(counts, y0, y1)

    # -- head and neck ---------------------------------------------------- #
    chin_y, neck_base_y, chin_conf = _find_neck(torso_profile, y0, height)
    confidence["neck"] = chin_conf
    if chin_conf < 0.35:
        notes.append("neck not clearly visible; using canonical head proportion")
        chin_y = 0.5 * (chin_y + canonical("chin"))
        neck_base_y = max(neck_base_y, chin_y + 0.02 * height)

    # -- shoulders, waist, hips ------------------------------------------- #
    shoulder_y, shoulder_conf = _find_shoulders(torso_profile, neck_base_y, y0, height)
    confidence["shoulders"] = shoulder_conf

    # Long hair or a high collar can pull the width pinch above the real jaw.
    # The neck occupies a limited span above the shoulders whatever the design.
    clamped_chin = float(np.clip(chin_y, shoulder_y - 0.10 * height, shoulder_y - 0.025 * height))
    if abs(clamped_chin - chin_y) > 0.01 * height:
        notes.append("chin line adjusted to sit a plausible neck length above the shoulders")
        confidence["neck"] = min(confidence["neck"], 0.6)
    chin_y = clamped_chin
    neck_base_y = float(np.clip(neck_base_y, chin_y + 0.01 * height, shoulder_y))

    head_band = mask[y0 : int(chin_y) + 1]
    if head_band.any():
        head_cols = np.flatnonzero(head_band.any(axis=0))
        head_center = np.array(
            [
                float(0.5 * (head_cols[0] + head_cols[-1])),
                float(y0 + 0.5 * (int(chin_y) - y0)),
            ]
        )
        head_radius = max(2.0, 0.5 * float(head_cols[-1] - head_cols[0]))
    else:  # pragma: no cover - guarded by mask.any()
        head_center = np.array([center_x, canonical("chin") * 0.5])
        head_radius = 0.06 * height

    waist_y, waist_conf = _find_waist(torso_profile, shoulder_y, split_y, y0, height)
    confidence["waist"] = waist_conf
    hip_y, hip_conf = _estimate_hip_line(split_y, leg_runs, waist_y, y0, height)
    confidence["hips"] = hip_conf
    if not leg_runs:
        notes.append("legs never separate in the artwork; hips placed by proportion")
    chest_y = shoulder_y + 0.35 * (waist_y - shoulder_y)

    # -- torso widths ------------------------------------------------------ #
    hip_half = _profile_half_width(torso_profile, hip_y, height)
    waist_half = _profile_half_width(torso_profile, waist_y, height)
    shoulder_half = _shoulder_half_width(torso_profile, shoulder_y, waist_half, height)

    # -- arms ------------------------------------------------------------- #
    arm_band = (int(shoulder_y - 0.02 * height), int(hip_y + 0.12 * height))
    hand_l = farthest_lateral_point(mask, center_x, +1, arm_band)
    hand_r = farthest_lateral_point(mask, center_x, -1, arm_band)
    shoulder_l = np.array([center_x + shoulder_half, shoulder_y])
    shoulder_r = np.array([center_x - shoulder_half, shoulder_y])

    arm_conf = 1.0
    if hand_l is None or hand_r is None:
        arm_conf = 0.2
        notes.append("arms not detected; using canonical arm placement")
        hand_l = np.array([center_x + 1.35 * shoulder_half, y0 + CANONICAL["wrist"] * height])
        hand_r = np.array([center_x - 1.35 * shoulder_half, y0 + CANONICAL["wrist"] * height])
    else:
        reach_l = abs(hand_l[0] - center_x)
        reach_r = abs(hand_r[0] - center_x)
        if max(reach_l, reach_r) < 1.12 * shoulder_half:
            arm_conf = 0.35
            notes.append("arms held close to the body; arm chain estimated")
    confidence["arms"] = arm_conf

    # Refine the shoulders by walking in from the hands, which sees through a
    # robe or a fused deltoid that the width profile cannot.
    neck_base_point = np.array([center_x, neck_base_y])
    marched = [
        shoulder_from_arm(mask, hand_l, neck_base_point, height),
        shoulder_from_arm(mask, hand_r, neck_base_point, height),
    ]
    offsets = [abs(float(p[0]) - center_x) for p in marched if p is not None]
    if offsets:
        # Both shoulders share one offset: the front view is symmetrised, and a
        # rig with mismatched shoulders reads as a deformity. Only the lateral
        # position is taken from the march - the shoulder *row* is already well
        # determined by the width ramp, whereas the march crosses into the body
        # some way down the deltoid.
        reach = max(abs(float(hand_l[0]) - center_x), abs(float(hand_r[0]) - center_x))
        shoulder_half = float(np.clip(np.mean(offsets), 0.035 * height, 0.55 * reach))
        confidence["shoulders"] = min(1.0, confidence["shoulders"] + 0.1 * len(offsets))
        shoulder_l = np.array([center_x + shoulder_half, shoulder_y])
        shoulder_r = np.array([center_x - shoulder_half, shoulder_y])

    arm_curve_l = trace_limb_centerline(mask, shoulder_l, hand_l)
    arm_curve_r = trace_limb_centerline(mask, shoulder_r, hand_r)
    elbow_l = curve_point_at(arm_curve_l, 0.48)
    elbow_r = curve_point_at(arm_curve_r, 0.48)

    # -- legs ------------------------------------------------------------- #
    hip_l = np.array([center_x + hip_half * 0.62, hip_y])
    hip_r = np.array([center_x - hip_half * 0.62, hip_y])
    foot_l, foot_r = _find_feet(mask, center_x, hip_y, y1, leg_runs)
    leg_curve_l = trace_limb_centerline(mask, hip_l, foot_l)
    leg_curve_r = trace_limb_centerline(mask, hip_r, foot_r)
    knee_l = curve_point_at(leg_curve_l, 0.47)
    knee_r = curve_point_at(leg_curve_r, 0.47)
    ankle_l = curve_point_at(leg_curve_l, 0.93)
    ankle_r = curve_point_at(leg_curve_r, 0.93)
    toe_l = np.array([foot_l[0], min(y1 - 1.0, foot_l[1] + 0.012 * height)])
    toe_r = np.array([foot_r[0], min(y1 - 1.0, foot_r[1] + 0.012 * height)])
    confidence["legs"] = 0.9 if leg_runs else 0.4

    return CharacterLandmarks(
        height_px=height,
        width_px=width,
        center_x=float(center_x),
        top_y=float(y0),
        bottom_y=float(y1),
        chin_y=float(chin_y),
        shoulder_y=float(shoulder_y),
        chest_y=float(chest_y),
        waist_y=float(waist_y),
        hip_y=float(hip_y),
        head_center=head_center,
        head_radius=float(head_radius),
        shoulder_l=shoulder_l,
        shoulder_r=shoulder_r,
        elbow_l=elbow_l,
        elbow_r=elbow_r,
        hand_l=np.asarray(hand_l, dtype=np.float64),
        hand_r=np.asarray(hand_r, dtype=np.float64),
        hip_l=hip_l,
        hip_r=hip_r,
        knee_l=knee_l,
        knee_r=knee_r,
        ankle_l=ankle_l,
        ankle_r=ankle_r,
        toe_l=toe_l,
        toe_r=toe_r,
        confidence=confidence,
        notes=notes,
    )


def _find_neck(widths: np.ndarray, y0: int, height: float) -> tuple[float, float, float]:
    """Locate the neck as the narrow plateau between head and shoulders.

    Returns ``(chin_y, neck_base_y, confidence)``. The neck is a *band*, not a
    single row, and both ends are useful: the chin anchors the head bone and the
    base anchors the shoulder search.
    """
    lo = int(y0 + 0.04 * height)
    hi = int(y0 + 0.32 * height)
    fallback = (
        y0 + CANONICAL["chin"] * height,
        y0 + CANONICAL["shoulder"] * height,
        0.0,
    )
    if hi - lo < 4:
        return fallback
    smoothed = _smooth1d(widths[lo:hi], max(1.0, 0.004 * height))
    index = int(np.argmin(smoothed))
    neck_width = float(smoothed[index])
    head_width = float(smoothed[: max(1, index)].max()) if index > 0 else neck_width
    below = float(smoothed[index + 1 :].max()) if index + 1 < smoothed.size else neck_width
    # A real neck is a pinch: narrower than both the head above and the
    # shoulders below.
    pinch = 1.0 - neck_width / max(1e-6, 0.5 * (head_width + below))
    confidence = float(np.clip(pinch * 2.2, 0.0, 1.0))

    # The neck spans the rows that are much closer to the pinch width than to
    # the head above / shoulders below, measured relative to each contrast so
    # the result does not depend on absolute pixel widths.
    chin_limit = neck_width + 0.35 * (head_width - neck_width)
    base_limit = neck_width + 0.35 * (below - neck_width)
    first = index
    while first > 0 and smoothed[first - 1] <= chin_limit:
        first -= 1
    last = index
    while last < smoothed.size - 1 and smoothed[last + 1] <= base_limit:
        last += 1
    chin = float(lo + first)
    neck_base = float(lo + last)
    if neck_base - chin < 0.008 * height:
        neck_base = chin + 0.02 * height
    return chin, neck_base, confidence


def _find_leg_split(counts: np.ndarray, y0: int, y1: int) -> tuple[float | None, bool]:
    """Highest row from which the silhouette stays split into two limbs.

    This is *not* the hip line: characters drawn with the thighs touching only
    separate below the knee. It is a lower bound, interpreted by
    :func:`_estimate_hip_line`.
    """
    height = float(y1 - y0)
    lo = int(y0 + 0.35 * height)
    hi = int(y1 - 0.04 * height)
    if hi - lo < 4:
        return None, False
    split = counts[lo:hi] >= 2
    if not split.any():
        return None, False
    index = split.size - 1
    while index >= 0 and not split[index]:
        index -= 1
    # Walk up while rows keep showing two limbs, tolerating a few merged rows
    # (crossed ankles, a shadow bridging the feet).
    tolerance = max(2, int(0.02 * height))
    misses = 0
    top = index
    while index >= 0:
        if split[index]:
            top = index
            misses = 0
        else:
            misses += 1
            if misses > tolerance:
                break
        index -= 1
    return float(lo + top), True


def _estimate_hip_line(
    split_y: float | None, legs_found: bool, waist_y: float, y0: int, height: float
) -> tuple[float, float]:
    """Fuse three hip estimates: waist offset, leg split, canonical proportion.

    The waist is the most reliable observable (a width minimum, never
    contaminated by arms), and the pelvis sits a fixed proportion below it. The
    leg split is a weaker cue because heavy thighs stay in contact well past the
    crotch, so its influence decays with how far it disagrees.
    """
    canonical_hip = y0 + CANONICAL["hip"] * height
    waist_based = waist_y + (CANONICAL["hip"] - CANONICAL["waist"]) * height

    estimates = [(canonical_hip, 0.8), (waist_based, 1.6)]
    confidence = 0.55
    if legs_found and split_y is not None:
        observed = split_y - 0.045 * height
        disagreement = abs(observed - waist_based) / (0.12 * height)
        weight = 1.4 * float(np.exp(-0.5 * disagreement * disagreement))
        estimates.append((observed, weight))
        confidence = float(np.clip(0.5 + 0.5 * weight / 1.4, 0.35, 1.0))

    total = sum(w for _, w in estimates)
    hip = sum(value * w for value, w in estimates) / total
    hip = float(np.clip(hip, waist_y + 0.06 * height, waist_y + 0.20 * height))
    return float(np.clip(hip, y0 + 0.44 * height, y0 + 0.60 * height)), confidence


def _find_shoulders(
    widths: np.ndarray, neck_base_y: float, y0: int, height: float
) -> tuple[float, float]:
    """Shoulders = where the width ramps up from the neck to the torso."""
    lo = int(neck_base_y)
    hi = int(min(neck_base_y + 0.18 * height, y0 + 0.42 * height))
    default = float(min(neck_base_y + 0.02 * height, y0 + CANONICAL["shoulder"] * height))
    if hi - lo < 3:
        return default, 0.3
    band = _smooth1d(widths[lo:hi], max(1.0, 0.003 * height))
    peak = float(band.max())
    base = float(band.min())
    if peak - base < 1e-6:
        return default, 0.3
    reached = np.flatnonzero(band >= base + 0.70 * (peak - base))
    if reached.size == 0:
        return default, 0.4
    ramp = (peak - base) / max(peak, 1e-6)
    return float(lo + reached[0]), float(np.clip(ramp * 1.5, 0.2, 1.0))


def _find_waist(
    profile: np.ndarray, shoulder_y: float, split_y: float | None, y0: int, height: float
) -> tuple[float, float]:
    """Waist = narrowest torso row between chest and pelvis.

    The search stops above where the legs part, because the centre-axis run
    collapses to zero in the gap between the legs and would otherwise win.
    """
    lo = int(max(shoulder_y + 0.08 * height, y0 + 0.24 * height))
    hi = int(min(shoulder_y + 0.34 * height, y0 + 0.52 * height))
    if split_y is not None:
        hi = int(min(hi, split_y - 0.03 * height))
    canonical_waist = float(y0 + CANONICAL["waist"] * height)
    if hi - lo < 5:
        return canonical_waist, 0.2
    band = _smooth1d(profile[lo:hi], max(1.5, 0.004 * height))
    index = int(np.argmin(band))
    # A believable waist is a pinch bracketed by wider chest and hips. A robe or
    # a barrel-chested design has no pinch, and the argmin then lands wherever
    # the band happens to end - detectable, and worth distrusting.
    above = float(band[: index + 1].max())
    below = float(band[index:].max())
    pinch = 1.0 - band[index] / max(0.5 * (above + below), 1e-6)
    at_edge = index <= 1 or index >= band.size - 2
    conf = float(np.clip(pinch * 3.0, 0.0, 1.0)) * (0.3 if at_edge else 1.0)
    observed = float(lo + index)
    return observed * conf + canonical_waist * (1.0 - conf), conf


def _profile_half_width(profile: np.ndarray, y: float, height: float) -> float:
    """Half width of the torso run at row ``y``, taking the local maximum."""
    row = int(np.clip(y, 0, profile.size - 1))
    lo = max(0, row - 1)
    hi = min(profile.size, row + 2)
    best = 0.5 * float(profile[lo:hi].max())
    return best if best > 1.0 else max(2.0, 0.06 * height)


#: Ratio of the shoulder-joint offset to the waist half width in canonical
#: eight-head figure proportions. See :data:`CANONICAL`.
_SHOULDER_TO_WAIST = CANONICAL["shoulder_half_width"] / (0.5 * 0.145)


def _shoulder_half_width(
    profile: np.ndarray, shoulder_y: float, waist_half: float, height: float
) -> float:
    """Where the shoulder joints sit, laterally.

    A silhouette cannot answer this directly: in both A-pose and T-pose the
    deltoid is fused to the torso, so the centre run at the shoulder line
    already includes part of the arm. The waist, by contrast, is measured
    cleanly, and the shoulder-to-waist ratio is one of the most stable
    relationships in figure drawing - so the estimate is anchored there and only
    bounded by what the silhouette allows.
    """
    estimate = _SHOULDER_TO_WAIST * waist_half
    silhouette_half = _profile_half_width(profile, shoulder_y + 0.02 * height, height)
    estimate = min(estimate, 0.95 * silhouette_half)
    return float(max(estimate, 1.05 * waist_half, 0.045 * height))


def _find_feet(
    mask: np.ndarray, center_x: float, hip_y: float, y1: int, legs_found: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest point of each leg."""
    bottom = int(y1) - 1
    for probe in range(bottom, max(int(hip_y), bottom - mask.shape[0]), -1):
        runs = row_runs(mask, probe)
        if len(runs) >= 2 and legs_found:
            runs = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2]
            runs.sort(key=lambda r: 0.5 * (r[0] + r[1]))
            right_run, left_run = runs[0], runs[1]
            return (
                np.array([0.5 * (left_run[0] + left_run[1]), float(probe)]),
                np.array([0.5 * (right_run[0] + right_run[1]), float(probe)]),
            )
        if runs and not legs_found:
            start, end = max(runs, key=lambda r: r[1] - r[0])
            quarter = 0.25 * (end - start)
            return (
                np.array([end - quarter, float(probe)]),
                np.array([start + quarter, float(probe)]),
            )
    return (
        np.array([center_x + 0.03 * mask.shape[1], float(bottom)]),
        np.array([center_x - 0.03 * mask.shape[1], float(bottom)]),
    )


def _smooth1d(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.3 or values.size < 3:
        return np.asarray(values, dtype=np.float64)
    kernel = ndops.gaussian_kernel(sigma)
    padded = np.pad(np.asarray(values, dtype=np.float64), (kernel.size // 2,) * 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")
