"""Drive the rigging stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conceptforge.config import RiggingConfig
from conceptforge.geometry.build import GeometryResult
from conceptforge.imaging.views import ConceptSheet
from conceptforge.reporting import NULL_REPORTER, Reporter
from conceptforge.rigging.autorig import fit_skeleton
from conceptforge.rigging.skeleton import Skeleton
from conceptforge.rigging.skinning import SkinBinding, bind_skin


@dataclass
class RigResult:
    skeleton: Skeleton
    binding: SkinBinding
    stats: dict[str, Any] = field(default_factory=dict)


def build_rig(
    sheet: ConceptSheet,
    geometry: GeometryResult,
    config: RiggingConfig | None = None,
    reporter: Reporter | None = None,
) -> RigResult:
    """Fit a skeleton to the reconstructed mesh and solve skin weights."""
    config = config or RiggingConfig()
    reporter = reporter or NULL_REPORTER

    fit = fit_skeleton(sheet.landmarks, sheet.front, geometry.field, geometry.transform, config)
    binding = bind_skin(geometry.mesh, fit.skeleton, config)

    stats: dict[str, Any] = {}
    stats.update(fit.stats)
    stats.update(binding.stats)
    stats["joints_inside_mesh"] = _joints_inside_ratio(fit.skeleton, geometry)
    reporter.info(
        f"rigged {len(fit.skeleton)} joints, "
        f"{binding.stats['mean_active_bones']} mean influences per vertex"
    )
    return RigResult(skeleton=fit.skeleton, binding=binding, stats=stats)


def _joints_inside_ratio(skeleton: Skeleton, geometry: GeometryResult) -> float:
    """Fraction of joints that land inside the solid - a rig sanity check.

    Anything much below 1.0 means joints are poking through the surface, which
    shows up immediately as a limb that pivots outside its own geometry.
    """
    height_units = _to_field_space(skeleton.rest_positions(), geometry.transform)
    samples = geometry.field.sample(height_units)
    return round(float((samples > 0).mean()), 3)


def _to_field_space(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(np.asarray(transform, dtype=np.float64))
    return np.asarray(points, dtype=np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
