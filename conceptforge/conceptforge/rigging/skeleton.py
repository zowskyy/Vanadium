"""Skeleton representation and pose evaluation.

Design decisions that matter downstream:

**Bone naming follows the Mixamo/Unity humanoid convention** (``Hips``,
``Spine``, ``LeftUpLeg``, ``LeftForeArm``, ...). This is not cosmetic: Unity's
humanoid avatar mapper, Mixamo's retargeter and most engine importers recognise
these names and will auto-configure the rig, so an exported character can accept
third-party animation without a manual bone-mapping pass.

**Rest local rotations are identity.** Every joint's rest pose is a pure
translation from its parent. That makes an animated local rotation *equal* to
the rotation relative to rest, which in turn means a clip authored against the
canonical humanoid applies to any character this software generates without a
per-character retargeting solve. It also makes inverse bind matrices exact
translations, avoiding a class of floating-point skinning error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from conceptforge.mathutil import IDENTITY_QUAT, quat_to_matrix

#: Canonical humanoid hierarchy: ``(name, parent name or None)``, root first.
HUMANOID_HIERARCHY: tuple[tuple[str, str | None], ...] = (
    ("Hips", None),
    ("Spine", "Hips"),
    ("Spine1", "Spine"),
    ("Spine2", "Spine1"),
    ("Neck", "Spine2"),
    ("Head", "Neck"),
    ("HeadTop_End", "Head"),
    ("LeftShoulder", "Spine2"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("RightShoulder", "Spine2"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("LeftUpLeg", "Hips"),
    ("LeftLeg", "LeftUpLeg"),
    ("LeftFoot", "LeftLeg"),
    ("LeftToeBase", "LeftFoot"),
    ("RightUpLeg", "Hips"),
    ("RightLeg", "RightUpLeg"),
    ("RightFoot", "RightLeg"),
    ("RightToeBase", "RightFoot"),
)

#: Mirror pairs, used to enforce symmetry and to mirror animation clips.
SYMMETRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("LeftShoulder", "RightShoulder"),
    ("LeftArm", "RightArm"),
    ("LeftForeArm", "RightForeArm"),
    ("LeftHand", "RightHand"),
    ("LeftUpLeg", "RightUpLeg"),
    ("LeftLeg", "RightLeg"),
    ("LeftFoot", "RightFoot"),
    ("LeftToeBase", "RightToeBase"),
)


@dataclass
class Joint:
    """One bone. ``rest_translation`` is the offset from the parent at rest."""

    name: str
    parent: int
    rest_translation: np.ndarray
    #: Where the bone's far end sits in rest world space. For joints with a
    #: single child this is the child's position; leaves get an explicit
    #: estimate. Only used for skinning and visualisation, never for transforms.
    tail: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.rest_translation = np.asarray(self.rest_translation, dtype=np.float64).reshape(3)
        if self.tail is not None:
            self.tail = np.asarray(self.tail, dtype=np.float64).reshape(3)


class Skeleton:
    """A joint hierarchy with rest pose and forward kinematics."""

    def __init__(self, joints: Sequence[Joint]) -> None:
        self.joints: list[Joint] = list(joints)
        self._index = {joint.name: i for i, joint in enumerate(self.joints)}
        for i, joint in enumerate(self.joints):
            if joint.parent >= i:
                raise ValueError(
                    f"joint {joint.name!r} must come after its parent; "
                    "the hierarchy has to be topologically ordered"
                )

    # -- structure -------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.joints)

    @property
    def names(self) -> list[str]:
        return [joint.name for joint in self.joints]

    @property
    def parents(self) -> np.ndarray:
        return np.array([joint.parent for joint in self.joints], dtype=np.int64)

    def index(self, name: str) -> int:
        if name not in self._index:
            raise KeyError(f"no joint named {name!r}; have {', '.join(self.names)}")
        return self._index[name]

    def has(self, name: str) -> bool:
        return name in self._index

    def children_of(self, index: int) -> list[int]:
        return [i for i, joint in enumerate(self.joints) if joint.parent == index]

    def descendants_of(self, index: int) -> list[int]:
        out: list[int] = []
        stack = [index]
        while stack:
            current = stack.pop()
            for child in self.children_of(current):
                out.append(child)
                stack.append(child)
        return out

    # -- rest pose -------------------------------------------------------- #
    @property
    def rest_local_translations(self) -> np.ndarray:
        return np.stack([joint.rest_translation for joint in self.joints], axis=0)

    def rest_positions(self) -> np.ndarray:
        """World-space joint positions at rest, shape (J, 3)."""
        positions = np.zeros((len(self), 3))
        for i, joint in enumerate(self.joints):
            if joint.parent < 0:
                positions[i] = joint.rest_translation
            else:
                positions[i] = positions[joint.parent] + joint.rest_translation
        return positions

    def rest_global_matrices(self) -> np.ndarray:
        """(J, 4, 4) rest transforms - pure translations by construction."""
        positions = self.rest_positions()
        matrices = np.broadcast_to(np.eye(4), (len(self), 4, 4)).copy()
        matrices[:, :3, 3] = positions
        return matrices

    def inverse_bind_matrices(self) -> np.ndarray:
        """(J, 4, 4) matrices taking mesh space into each joint's rest space."""
        matrices = np.broadcast_to(np.eye(4), (len(self), 4, 4)).copy()
        matrices[:, :3, 3] = -self.rest_positions()
        return matrices

    def bone_segments(self) -> np.ndarray:
        """(J, 2, 3) rest-space start/end points for every bone."""
        positions = self.rest_positions()
        segments = np.zeros((len(self), 2, 3))
        segments[:, 0] = positions
        for i, joint in enumerate(self.joints):
            if joint.tail is not None:
                segments[i, 1] = joint.tail
                continue
            children = self.children_of(i)
            if children:
                segments[i, 1] = positions[children].mean(axis=0)
            else:
                parent = joint.parent
                direction = (
                    positions[i] - positions[parent] if parent >= 0 else np.array([0.0, 0.05, 0.0])
                )
                length = float(np.linalg.norm(direction))
                unit = direction / length if length > 1e-9 else np.array([0.0, 1.0, 0.0])
                segments[i, 1] = positions[i] + unit * max(length * 0.4, 1e-3)
        return segments

    def bone_lengths(self) -> np.ndarray:
        segments = self.bone_segments()
        return np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)

    def height(self) -> float:
        positions = self.rest_positions()
        return float(positions[:, 1].max() - positions[:, 1].min())

    # -- posing ----------------------------------------------------------- #
    def pose_matrices(
        self,
        rotations: np.ndarray | None = None,
        translations: np.ndarray | None = None,
        root_motion: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward kinematics. Returns (J, 4, 4) global transforms.

        ``rotations`` are local quaternions (x, y, z, w) relative to rest;
        ``translations`` override rest local translations where given;
        ``root_motion`` is added to the root's translation.
        """
        count = len(self)
        if rotations is None:
            rotations = np.broadcast_to(IDENTITY_QUAT, (count, 4)).copy()
        rotations = np.asarray(rotations, dtype=np.float64).reshape(count, 4)
        local_translations = self.rest_local_translations.copy()
        if translations is not None:
            supplied = np.asarray(translations, dtype=np.float64).reshape(count, 3)
            local_translations = supplied
        if root_motion is not None:
            local_translations[0] = local_translations[0] + np.asarray(root_motion, dtype=np.float64)

        globals_ = np.empty((count, 4, 4))
        for i, joint in enumerate(self.joints):
            local = quat_to_matrix(rotations[i])
            local[:3, 3] = local_translations[i]
            globals_[i] = local if joint.parent < 0 else globals_[joint.parent] @ local
        return globals_

    def skinning_matrices(
        self,
        rotations: np.ndarray | None = None,
        translations: np.ndarray | None = None,
        root_motion: np.ndarray | None = None,
    ) -> np.ndarray:
        """(J, 4, 4) matrices to apply to bound mesh vertices."""
        return self.pose_matrices(rotations, translations, root_motion) @ self.inverse_bind_matrices()

    def posed_positions(
        self,
        rotations: np.ndarray | None = None,
        translations: np.ndarray | None = None,
        root_motion: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.pose_matrices(rotations, translations, root_motion)[:, :3, 3]

    # -- serialisation ---------------------------------------------------- #
    def to_dict(self) -> dict[str, object]:
        positions = self.rest_positions()
        return {
            "joints": [
                {
                    "name": joint.name,
                    "parent": None if joint.parent < 0 else self.joints[joint.parent].name,
                    "rest_translation": [round(float(v), 6) for v in joint.rest_translation],
                    "rest_position": [round(float(v), 6) for v in positions[i]],
                }
                for i, joint in enumerate(self.joints)
            ],
            "height": round(self.height(), 5),
        }

    def symmetric_index_pairs(self) -> list[tuple[int, int]]:
        pairs = []
        for left, right in SYMMETRIC_PAIRS:
            if self.has(left) and self.has(right):
                pairs.append((self.index(left), self.index(right)))
        return pairs


def skeleton_from_positions(
    positions: dict[str, np.ndarray],
    hierarchy: Iterable[tuple[str, str | None]] = HUMANOID_HIERARCHY,
    tails: dict[str, np.ndarray] | None = None,
) -> Skeleton:
    """Build a skeleton from world-space joint positions.

    Joints missing from ``positions`` are skipped, and their children are
    reparented to the nearest present ancestor, so a partial rig (say, a
    character with no separate clavicles) still produces a valid hierarchy.
    """
    tails = tails or {}
    order = [name for name, _ in hierarchy]
    parent_map = dict(hierarchy)
    present = [name for name in order if name in positions]
    if not present:
        raise ValueError("no joint positions supplied")

    def nearest_present_ancestor(name: str) -> str | None:
        current = parent_map.get(name)
        while current is not None and current not in positions:
            current = parent_map.get(current)
        return current

    index_of = {name: i for i, name in enumerate(present)}
    joints: list[Joint] = []
    for name in present:
        ancestor = nearest_present_ancestor(name)
        parent_index = index_of[ancestor] if ancestor is not None else -1
        offset = (
            np.asarray(positions[name], dtype=np.float64)
            if parent_index < 0
            else np.asarray(positions[name], dtype=np.float64) - np.asarray(positions[ancestor], dtype=np.float64)
        )
        joints.append(Joint(name=name, parent=parent_index, rest_translation=offset, tail=tails.get(name)))
    return Skeleton(joints)
