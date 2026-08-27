"""Configuration objects for the ConceptForge pipeline.

Every stage takes its settings from a nested dataclass hanging off
:class:`ForgeConfig`. Configs round-trip through plain dicts (and therefore
JSON), which is how the CLI ``--config`` flag and the web studio pass overrides.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class ImagingConfig:
    """Concept-art interpretation."""

    #: Longest edge the artwork is resampled to before analysis. Larger keeps
    #: more silhouette detail at the cost of quadratic analysis time.
    working_resolution: int = 1024
    #: Alpha threshold for artwork that already ships a matte.
    alpha_threshold: float = 0.5
    #: Tolerance for flood-fill background removal on opaque artwork,
    #: as a fraction of the 0..1 colour range.
    background_tolerance: float = 0.14
    #: Drop mask blobs smaller than this fraction of the largest blob.
    min_blob_ratio: float = 0.02
    #: Close/open radius (px at working resolution) used to clean the matte.
    morph_radius: int = 2
    #: Enforce left/right symmetry on the front view by mirroring the more
    #: complete half. Set 0 to disable, 1 for full mirror, values between
    #: blend.
    symmetrize: float = 0.65
    #: Automatically split a multi-view turnaround sheet into front/side/back.
    auto_split_sheet: bool = True
    #: Explicit view assignment for sheet panels, left to right, when
    #: ``auto_split_sheet`` cannot be trusted. e.g. ["front", "side", "back"].
    panel_order: Sequence[str] | None = None


@dataclass
class GeometryConfig:
    """Volumetric reconstruction and mesh conditioning."""

    #: Voxel resolution along the character's height. 128 is a good default;
    #: 192-256 for hero assets.
    voxel_height: int = 144
    #: Depth of the character as a fraction of height, used when no side view
    #: is available and as the carving depth budget when one is.
    depth_ratio: float = 0.22
    #: Exponent applied to the silhouette distance field when inflating a
    #: single view. <1 gives fuller bodies, >1 gives flatter, sharper forms.
    inflation_power: float = 0.72
    #: Radius of the smooth-min used to blend the view constraints, in voxels.
    #: Rounds off the hard creases a boolean intersection would leave.
    blend_radius: float = 2.4
    #: Gaussian blur sigma (voxels) applied to the implicit field before
    #: surfacing. Removes staircase artefacts from the source pixels.
    field_smoothing: float = 1.15
    #: Taubin smoothing iterations on the raw surface.
    smooth_iterations: int = 14
    #: Taubin lambda/mu pair; mu must be more negative than lambda.
    smooth_lambda: float = 0.55
    smooth_mu: float = -0.58
    #: Target triangle count after decimation. 0 disables decimation.
    target_triangles: int = 24000
    #: Emit a second, lower-poly LOD chain (ratios of the base mesh).
    lod_ratios: Sequence[float] = (0.45, 0.18)
    #: Final character height in scene units (metres) after normalisation.
    character_height: float = 1.75


@dataclass
class RiggingConfig:
    """Skeleton fitting and skin weight solving."""

    enabled: bool = True
    #: Number of bone influences kept per vertex (4 is the hardware norm).
    max_influences: int = 4
    #: Falloff sharpness for geodesic weights. Higher = more rigid joints.
    falloff: float = 2.6
    #: Laplacian smoothing passes applied to the weight field.
    weight_smoothing: int = 3
    #: Use mesh-surface geodesic distances instead of straight-line distances.
    #: Prevents an arm bone from grabbing nearby torso vertices.
    geodesic: bool = True
    #: Add twist bones between shoulder/elbow and elbow/wrist.
    twist_bones: bool = False
    #: Insert IK target/pole helper nodes for hands and feet.
    ik_helpers: bool = True


@dataclass
class AnimationConfig:
    """Which clips to generate and how."""

    enabled: bool = True
    #: Clip names from :mod:`conceptforge.animation.library`.
    clips: Sequence[str] = ("idle", "walk", "run", "jump", "wave", "turn_left")
    #: Samples per second baked into the exported clips.
    fps: int = 30
    #: Global multiplier on procedural motion amplitude.
    intensity: float = 1.0
    #: Ground the feet with two-bone IK after the procedural pass.
    foot_ik: bool = True
    #: Bake an extra A-pose/T-pose reference clip for DCC round-tripping.
    include_rest_clip: bool = True


@dataclass
class TexturingConfig:
    """Texture projection and PBR map synthesis."""

    enabled: bool = True
    #: Square texture atlas resolution.
    resolution: int = 1024
    #: Iterations of push-pull inpainting used to fill unprojected regions.
    inpaint_iterations: int = 64
    #: Dilate the projected colour by this many pixels to avoid seam bleed.
    seam_dilation: int = 3
    #: Derive a tangent-space normal map from albedo luminance.
    normal_map: bool = True
    normal_strength: float = 0.85
    #: Derive roughness/metallic from albedo statistics.
    material_maps: bool = True
    base_roughness: float = 0.62


@dataclass
class ExportConfig:
    """Output formats."""

    #: Any of: glb, gltf, obj, ply, bvh, usda.
    formats: Sequence[str] = ("glb", "obj", "usda", "bvh")
    #: Embed textures in the GLB buffer (always true for .glb).
    embed_textures: bool = True
    #: Also write the intermediate masks/atlas/field previews.
    write_debug: bool = True
    #: Y-up (glTF/Unity) or Z-up (Blender/Unreal-ish) export basis.
    up_axis: str = "Y"


@dataclass
class PreviewConfig:
    """Headless software-rendered previews."""

    enabled: bool = True
    resolution: int = 512
    #: Number of turntable frames rendered for the still contact sheet.
    turntable_frames: int = 6
    #: Clip sampled for the animation strip, and how many frames.
    animation_clip: str = "walk"
    animation_frames: int = 8
    #: Write an animated GIF of ``animation_clip``.
    gif: bool = True
    gif_frames: int = 24


@dataclass
class ForgeConfig:
    """Top-level pipeline configuration."""

    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    rigging: RiggingConfig = field(default_factory=RiggingConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    texturing: TexturingConfig = field(default_factory=TexturingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    #: Free-form label written into export metadata.
    character_name: str = "Character"
    #: Deterministic seed for every stochastic step (there are few, but the
    #: pipeline must be reproducible for production sign-off).
    seed: int = 7

    # -- serialisation ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ForgeConfig":
        return _build(cls, data or {})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ForgeConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def merged(self, overrides: Mapping[str, Any] | None) -> "ForgeConfig":
        """Return a copy with ``overrides`` applied on top (deep merge)."""
        return ForgeConfig.from_dict(_deep_merge(self.to_dict(), overrides or {}))

    @classmethod
    def preset(cls, name: str) -> "ForgeConfig":
        if name not in QUALITY_PRESETS:
            raise ValueError(
                f"unknown preset {name!r}; choose from {', '.join(sorted(QUALITY_PRESETS))}"
            )
        return cls.from_dict(_deep_merge(cls().to_dict(), QUALITY_PRESETS[name]))


def _build(target_cls, data: Mapping[str, Any]):
    """Instantiate ``target_cls`` from a nested mapping, rejecting typos.

    ``from __future__ import annotations`` turns every annotation into a string,
    so nested dataclasses are resolved through the explicit ``_NESTED`` table
    rather than by introspecting ``field.type``.
    """
    known = {f.name for f in fields(target_cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {key!r} for {target_cls.__name__}")
        nested = _NESTED.get((target_cls.__name__, key))
        if nested is not None:
            if isinstance(value, nested):
                kwargs[key] = value
            else:
                kwargs[key] = _build(nested, value or {})
        else:
            kwargs[key] = value
    return target_cls(**kwargs)


_NESTED = {
    ("ForgeConfig", "imaging"): ImagingConfig,
    ("ForgeConfig", "geometry"): GeometryConfig,
    ("ForgeConfig", "rigging"): RiggingConfig,
    ("ForgeConfig", "animation"): AnimationConfig,
    ("ForgeConfig", "texturing"): TexturingConfig,
    ("ForgeConfig", "export"): ExportConfig,
    ("ForgeConfig", "preview"): PreviewConfig,
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


#: Named quality presets. ``draft`` is for iteration (seconds), ``hero`` for
#: final production assets.
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "draft": {
        "imaging": {"working_resolution": 512},
        "geometry": {
            "voxel_height": 96,
            "smooth_iterations": 8,
            "target_triangles": 8000,
            "lod_ratios": [],
        },
        "texturing": {"resolution": 512, "inpaint_iterations": 32},
        "animation": {"clips": ["idle", "walk"], "fps": 24},
        "export": {"formats": ["glb"]},
        "preview": {"turntable_frames": 4, "animation_frames": 6, "gif": False},
    },
    "standard": {},
    "hero": {
        "imaging": {"working_resolution": 1536},
        "geometry": {
            "voxel_height": 224,
            "smooth_iterations": 20,
            "target_triangles": 60000,
            "lod_ratios": [0.5, 0.22, 0.08],
        },
        "rigging": {"weight_smoothing": 4, "twist_bones": True},
        "texturing": {"resolution": 2048, "inpaint_iterations": 96},
        "animation": {
            "clips": ["idle", "breathe", "walk", "run", "jump", "wave", "turn_left", "turn_right"],
            "fps": 30,
        },
        "export": {"formats": ["glb", "gltf", "obj", "ply", "usda", "bvh"]},
        "preview": {"resolution": 768, "turntable_frames": 8, "animation_frames": 10},
    },
}
