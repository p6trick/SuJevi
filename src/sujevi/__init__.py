"""Local model runners used by SuJevi experiments.

Runners are imported lazily because each model has an isolated environment and
its own optional dependencies. Importing one runner must not require another
model's packages.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "JeffLocalRuntime",
    "JeffPrediction",
    "JeffQuestionRepresentation",
    "KevLocalRuntime",
    "KevPrediction",
    "LocalJevOmni",
    "LocalOneJev",
]


def __getattr__(name: str) -> Any:
    if name in {
        "JeffLocalRuntime",
        "JeffPrediction",
        "JeffQuestionRepresentation",
    }:
        from .jeff_local import (
            JeffLocalRuntime,
            JeffPrediction,
            JeffQuestionRepresentation,
        )

        exports = {
            "JeffLocalRuntime": JeffLocalRuntime,
            "JeffPrediction": JeffPrediction,
            "JeffQuestionRepresentation": JeffQuestionRepresentation,
        }
    elif name in {"KevLocalRuntime", "KevPrediction"}:
        from .kev_local import KevLocalRuntime, KevPrediction

        exports = {
            "KevLocalRuntime": KevLocalRuntime,
            "KevPrediction": KevPrediction,
        }
    elif name == "LocalJevOmni":
        from .jev_omni_local import LocalJevOmni

        exports = {"LocalJevOmni": LocalJevOmni}
    elif name == "LocalOneJev":
        from .onejev_local import LocalOneJev

        exports = {"LocalOneJev": LocalOneJev}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals().update(exports)
    return exports[name]
