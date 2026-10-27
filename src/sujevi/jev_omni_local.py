"""Run the official Jev-Omni classifier from a local snapshot."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Jev-Omni"


def _load_release_module(model_path: Path):
    source = model_path / "jev_omni.py"
    if not source.exists():
        raise FileNotFoundError(f"Jev-Omni loader is missing: {source}")
    spec = importlib.util.spec_from_file_location("sujevi_jev_omni_release", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import Jev-Omni loader from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalJevOmni:
    """Reusable in-process Jev-Omni model.

    The upstream release loader normally downloads its own snapshot. SuJevi
    redirects that step to the project-local snapshot so experiments are
    reproducible and do not duplicate the 24 GB checkpoint.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, *, device: str = "cuda") -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model directory does not exist: {self.model_path}; "
                "run python scripts/download_models.py --jev-omni"
            )
        release = _load_release_module(self.model_path)
        release.snapshot_download = lambda *_args, **_kwargs: str(self.model_path)
        self.classifier = release.load_jev_omni(model_id=str(self.model_path), device=device)

    def predict(
        self,
        *,
        state: Any,
        question: str,
        options: list[str],
        media: str | Path | None = None,
        modality: str = "text",
        video_frames: int = 16,
    ) -> dict[str, Any]:
        return self.classifier.predict(
            state=state,
            question=question,
            options=options,
            media=str(media) if media is not None else None,
            modality=modality,
            video_frames=video_frames,
        )
