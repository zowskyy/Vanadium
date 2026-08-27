"""Progress reporting shared by the CLI, the web studio and the test suite.

A :class:`Reporter` is threaded through the pipeline. It does three jobs:
log human-readable progress, accumulate timings, and collect structured stage
statistics that end up in ``forge_report.json`` next to the exported assets.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass
class StageRecord:
    name: str
    seconds: float
    stats: dict[str, Any] = field(default_factory=dict)


class Reporter:
    """Collects stage timings/stats and optionally streams progress lines.

    Parameters
    ----------
    verbose:
        Print progress to ``stream``.
    stream:
        Where progress goes; defaults to stderr so stdout stays parseable.
    on_progress:
        Optional callback ``(fraction, message)`` used by the web studio to
        drive its progress bar.
    """

    def __init__(
        self,
        verbose: bool = True,
        stream=None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> None:
        self.verbose = verbose
        self.stream = stream if stream is not None else sys.stderr
        self.on_progress = on_progress
        self.stages: list[StageRecord] = []
        self.warnings: list[str] = []
        self._t0 = time.perf_counter()
        self._planned: list[str] = []
        self._completed = 0

    # -- lifecycle -------------------------------------------------------- #
    def plan(self, stage_names: list[str]) -> None:
        """Declare the stage list so progress fractions are meaningful."""
        self._planned = list(stage_names)
        self._completed = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Time a stage; yield a dict the caller fills with statistics."""
        stats: dict[str, Any] = {}
        self._emit(self._fraction(), f"{name}...")
        start = time.perf_counter()
        try:
            yield stats
        finally:
            elapsed = time.perf_counter() - start
            self.stages.append(StageRecord(name, elapsed, dict(stats)))
            self._completed += 1
            detail = ", ".join(f"{k}={_fmt(v)}" for k, v in stats.items() if not k.startswith("_"))
            suffix = f" ({detail})" if detail else ""
            self._emit(self._fraction(), f"{name} done in {elapsed:.2f}s{suffix}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self._emit(self._fraction(), f"warning: {message}")

    def info(self, message: str) -> None:
        self._emit(self._fraction(), message)

    # -- output ----------------------------------------------------------- #
    @property
    def total_seconds(self) -> float:
        return time.perf_counter() - self._t0

    def _fraction(self) -> float:
        if not self._planned:
            return 0.0
        return min(1.0, self._completed / max(1, len(self._planned)))

    def _emit(self, fraction: float, message: str) -> None:
        if self.verbose:
            print(f"[{fraction * 100:5.1f}%] {message}", file=self.stream, flush=True)
        if self.on_progress is not None:
            self.on_progress(fraction, message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_seconds": round(self.total_seconds, 3),
            "warnings": list(self.warnings),
            "stages": [
                {"name": s.name, "seconds": round(s.seconds, 3), "stats": s.stats}
                for s in self.stages
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_jsonable), encoding="utf-8")
        return path


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, (list, tuple)):
        return f"[{len(value)}]"
    return str(value)


def _jsonable(value: Any):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:  # pragma: no cover - numpy always present in practice
        pass
    return str(value)


NULL_REPORTER = Reporter(verbose=False)
