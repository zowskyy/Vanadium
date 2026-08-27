"""Fit a humanoid skeleton to the reconstructed character.

The 2D landmark detector supplies X and Y for every joint. The missing
coordinate, depth, comes from the reconstructed volume: for each joint, the
solid is sampled along Z at that (X, Y) column and the joint is placed at the
field-weighted centre, which is the medial axis of the body there. That puts
knees inside knees and the pelvis inside the pelvis instead of on the surface.

Left/right joints are then mirrored to a shared offset. Asymmetric limb
placement is one of the most visible rig defects - it makes a walk cycle limp -
and the front view has already been symmetrised, so there is no information to
lose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conceptforge.config import RiggingConfig
from conceptforge.geometry.volume import VoxelField
from conceptforge.imaging.landmarks import CharacterLandmarks
from conceptforge.imaging.views import ConceptView
from conceptforge.rigging.skeleton import Skeleton, skeleton_from_positions


@dataclass
class RigFit:
    skeleton: Skeleton
    stats: dict[str, Any] = field(default_factory=dict)


def fit_skeleton(
    landmarks: CharacterLandmarks,
    front: ConceptView,
    voxels: VoxelField,
    transform: np.ndarray,
    config: RiggingConfig | None = None,
) -> RigFit:
    """Place a humanoid skeleton inside the reconstructed character.

    Parameters
    ----------
    landmarks:
        Front-view landmarks, in that view's pixel coordinates.
    front:
        The front view, used to map pixels to height units.
    voxels:
        The implicit solid, sampled for joint depth. In height units.
    transform:
        4x4 matrix taking height units into final world units, i.e. the same
        scaling the mesh received.
    """
    config = config or RiggingConfig()
    height_px = max(float(front.bottom_y - front.top_y), 1.0)

    def to_height_units(point: np.ndarray) -> tuple[float, float]:
        x = (float(point[0]) - front.pivot_x) / height_px
        y = (front.bottom_y - float(point[1])) / height_px
        return x, y

    def place(point: np.ndarray, depth_bias: float = 0.0) -> np.ndarray:
        x, y = to_height_units(point)
        z = _medial_depth(voxels, x, y) + depth_bias
        return np.array([x, y, z])

    marks = landmarks
    hip_y = to_height_units(np.array([marks.center_x, marks.hip_y]))[1]
    waist_y = to_height_units(np.array([marks.center_x, marks.waist_y]))[1]
    chest_y = to_height_units(np.array([marks.center_x, marks.chest_y]))[1]
    shoulder_y = to_height_units(marks.shoulder_l)[1]
    chin_y = to_height_units(np.array([marks.center_x, marks.chin_y]))[1]
    crown_y = to_height_units(np.array([marks.center_x, marks.top_y]))[1]

    def axis(y: float) -> np.ndarray:
        """A point on the centre line at height ``y``, in view pixels."""
        return np.array([marks.center_x, front.bottom_y - y * height_px])

    positions: dict[str, np.ndarray] = {}
    # Spine chain: evenly distributed between pelvis and the base of the neck,
    # which is what animation clips assume when they distribute a spine twist.
    positions["Hips"] = place(axis(hip_y))
    positions["Spine"] = place(axis(hip_y + 0.34 * (waist_y - hip_y)))
    positions["Spine1"] = place(axis(waist_y + 0.45 * (chest_y - waist_y)))
    positions["Spine2"] = place(axis(chest_y + 0.55 * (shoulder_y - chest_y)))
    neck_y = shoulder_y + 0.35 * (chin_y - shoulder_y)
    positions["Neck"] = place(axis(neck_y))
    positions["Head"] = place(axis(chin_y + 0.12 * (crown_y - chin_y)))
    positions["HeadTop_End"] = place(axis(crown_y - 0.02 * (crown_y - chin_y)))

    arm_chain = {
        "LeftArm": marks.shoulder_l,
        "LeftForeArm": marks.elbow_l,
        "LeftHand": _toward(marks.elbow_l, marks.hand_l, 0.86),
        "RightArm": marks.shoulder_r,
        "RightForeArm": marks.elbow_r,
        "RightHand": _toward(marks.elbow_r, marks.hand_r, 0.86),
    }
    for name, point in arm_chain.items():
        positions[name] = place(point)
    # Clavicles start near the spine and travel out towards the shoulder joint.
    for side in ("Left", "Right"):
        positions[f"{side}Shoulder"] = positions["Spine2"] + 0.35 * (
            positions[f"{side}Arm"] - positions["Spine2"]
        )

    leg_chain = {
        "LeftUpLeg": marks.hip_l,
        "LeftLeg": marks.knee_l,
        "LeftFoot": marks.ankle_l,
        "LeftToeBase": marks.toe_l,
        "RightUpLeg": marks.hip_r,
        "RightLeg": marks.knee_r,
        "RightFoot": marks.ankle_r,
        "RightToeBase": marks.toe_r,
    }
    for name, point in leg_chain.items():
        positions[name] = place(point)

    _pull_inside(voxels, positions)
    _mirror_pairs(positions)
    _straighten_spine(positions)
    _clamp_to_ground(positions)

    tails: dict[str, np.ndarray] = {}
    for side in ("Left", "Right"):
        hand_tip = place(marks.hand_l if side == "Left" else marks.hand_r)
        tails[f"{side}Hand"] = positions[f"{side}Hand"] + 0.9 * (hand_tip - positions[f"{side}Hand"])
        toe = positions[f"{side}ToeBase"]
        tails[f"{side}ToeBase"] = toe + np.array([0.0, 0.0, 0.035])
    tails["HeadTop_End"] = positions["HeadTop_End"] + np.array([0.0, 0.012, 0.0])

    # Apply the same height-units -> world transform the mesh received.
    matrix = np.asarray(transform, dtype=np.float64)
    world = {name: _apply(matrix, p) for name, p in positions.items()}
    world_tails = {name: _apply(matrix, p) for name, p in tails.items()}

    skeleton = skeleton_from_positions(world, tails=world_tails)
    lengths = skeleton.bone_lengths()
    stats = {
        "joints": len(skeleton),
        "rig_height_m": round(skeleton.height(), 4),
        "shortest_bone_m": round(float(lengths[lengths > 1e-6].min()), 4),
        "longest_bone_m": round(float(lengths.max()), 4),
        "landmark_confidence": round(landmarks.overall_confidence, 3),
    }
    return RigFit(skeleton=skeleton, stats=stats)


def _row_of(front: ConceptView, height_px: float, y: float) -> float:
    """Inverse of the pixel-to-height-unit map, for the Y axis."""
    return front.bottom_y - y * height_px


def _toward(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) + t * (np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64))


def _apply(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ np.asarray(point, dtype=np.float64) + matrix[:3, 3]


def _medial_depth(voxels: VoxelField, x: float, y: float, search_radius: int = 2) -> float:
    """Field-weighted centre of the solid along Z at column ``(x, y)``.

    Falls back to a small neighbourhood search when the column happens to miss
    the body, which can occur for a landmark sitting a pixel outside the
    silhouette after smoothing.
    """
    index = voxels.index_of_world(np.array([[x, y, 0.0]]))[0]
    nx, ny, nz = voxels.shape
    i0 = int(np.clip(round(index[0]), 0, nx - 1))
    j0 = int(np.clip(round(index[1]), 0, ny - 1))
    zs = voxels.origin[2] + np.arange(nz) * voxels.spacing

    for radius in range(0, max(1, search_radius) + 1):
        best: tuple[float, float] | None = None
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if radius > 0 and abs(di) != radius and abs(dj) != radius:
                    continue
                i = int(np.clip(i0 + di, 0, nx - 1))
                j = int(np.clip(j0 + dj, 0, ny - 1))
                column = voxels.values[i, j, :]
                weights = np.maximum(column, 0.0)
                total = float(weights.sum())
                if total <= 1e-12:
                    continue
                centre = float((weights * zs).sum() / total)
                if best is None or total > best[1]:
                    best = (centre, total)
        if best is not None:
            return best[0]
    return 0.0


def _pull_inside(voxels: VoxelField, positions: dict[str, np.ndarray], reach: float = 0.07) -> None:
    """Nudge any joint that landed outside the solid back into it.

    Leaf joints suffer most: a fingertip or the crown of the head is derived
    from the silhouette's extreme point, which sits *on* the surface, and mesh
    smoothing then pulls the surface inside it. A joint outside its own geometry
    pivots visibly wrongly, so each one searches its neighbourhood for the
    closest interior point.
    """
    step = max(voxels.spacing, 1e-4)
    steps = max(1, int(round(reach / step)))
    offsets: list[np.ndarray] = []
    for radius in range(1, steps + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != radius:
                        continue
                    offsets.append(np.array([dx, dy, dz], dtype=np.float64) * step)
    if not offsets:
        return
    candidate_offsets = np.stack(offsets, axis=0)

    for name, point in positions.items():
        if float(voxels.sample(point[None, :])[0]) > 0.0:
            continue
        candidates = point[None, :] + candidate_offsets
        values = voxels.sample(candidates)
        interior = np.flatnonzero(values > 0.0)
        if interior.size == 0:
            continue
        # Offsets are generated in shells of increasing radius, so the first
        # interior hit is already the closest one.
        positions[name] = candidates[interior[0]]


def _clamp_to_ground(positions: dict[str, np.ndarray]) -> None:
    """Keep every joint at or above the ground plane."""
    for point in positions.values():
        point[1] = max(float(point[1]), 0.0)


def _mirror_pairs(positions: dict[str, np.ndarray]) -> None:
    """Force left/right joints onto mirrored positions."""
    for left in [name for name in positions if name.startswith("Left")]:
        right = "Right" + left[len("Left") :]
        if right not in positions:
            continue
        a, b = positions[left], positions[right]
        offset = 0.5 * (abs(float(a[0])) + abs(float(b[0])))
        mean_y = 0.5 * (float(a[1]) + float(b[1]))
        mean_z = 0.5 * (float(a[2]) + float(b[2]))
        positions[left] = np.array([offset, mean_y, mean_z])
        positions[right] = np.array([-offset, mean_y, mean_z])


def _straighten_spine(positions: dict[str, np.ndarray]) -> None:
    """Snap the spine, neck and head onto x = 0.

    They are already close, but an exactly centred spine means a mirrored
    animation clip is exactly symmetric, and it keeps the root's forward axis
    honest.
    """
    for name in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End"):
        if name in positions:
            positions[name][0] = 0.0
