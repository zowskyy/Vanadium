"""ConceptForge: 2D concept art to production-ready animated 3D characters.

The public surface is intentionally small::

    from conceptforge import ForgeConfig, forge_character

    result = forge_character("art/hero_sheet.png", "out/hero", ForgeConfig())
    print(result.exports["glb"])

Everything else lives in focused subpackages (:mod:`conceptforge.imaging`,
:mod:`conceptforge.geometry`, :mod:`conceptforge.rigging`,
:mod:`conceptforge.animation`, :mod:`conceptforge.texturing`,
:mod:`conceptforge.exporters`, :mod:`conceptforge.render`).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ForgeConfig",
    "ForgeResult",
    "forge_character",
    "QUALITY_PRESETS",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy-import shim
    # Lazy so that `import conceptforge` stays cheap and free of NumPy import
    # cost for callers that only want the version string.
    if name in ("ForgeConfig", "QUALITY_PRESETS"):
        from conceptforge import config as _config

        return getattr(_config, name)
    if name in ("ForgeResult", "forge_character"):
        from conceptforge import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
