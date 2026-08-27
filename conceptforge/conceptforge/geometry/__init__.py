"""Volumetric reconstruction and mesh processing."""

from conceptforge.geometry.build import GeometryResult, build_geometry
from conceptforge.geometry.decimate import build_lod_chain, decimate
from conceptforge.geometry.marching import marching_tetrahedra
from conceptforge.geometry.mesh import Mesh, concatenate
from conceptforge.geometry.smoothing import laplacian_smooth_scalar, taubin_smooth
from conceptforge.geometry.uv import cylindrical, planar_front_back, uv_area_distortion
from conceptforge.geometry.volume import VoxelField, build_field

__all__ = [
    "GeometryResult",
    "Mesh",
    "VoxelField",
    "build_field",
    "build_geometry",
    "build_lod_chain",
    "concatenate",
    "cylindrical",
    "decimate",
    "laplacian_smooth_scalar",
    "marching_tetrahedra",
    "planar_front_back",
    "taubin_smooth",
    "uv_area_distortion",
]
