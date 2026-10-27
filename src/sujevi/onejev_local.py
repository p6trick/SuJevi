"""Run OneJev in-process without an HTTP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qev.calibrate import Calibration
from qev.mm_engine import MMDecisionEngine
from qev.schema import SystemOneRequest, SystemOneResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "OneJev-4B"


class LocalOneJev:
    """Reusable in-process OneJev model for inference and evaluation."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        head_dtype: str = "float32",
        fork_mode: str = "auto",
        media_root: str | Path | None = None,
        cuda_graphs: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model directory does not exist: {self.model_path}; "
                "run python scripts/download_models.py --onejev"
            )

        calibration_path = self.model_path / "calibration.json"
        calibration = Calibration.load(calibration_path) if calibration_path.exists() else None
        self.engine = MMDecisionEngine(
            str(self.model_path),
            device=device,
            dtype=dtype,
            head_dtype=head_dtype,
            fork_mode=fork_mode,
            media_root=str(Path(media_root).expanduser().resolve()) if media_root else None,
            allow_local_paths=True,
            cuda_graphs=cuda_graphs,
            calibration=calibration,
        )

    def system_one(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        media: list[dict[str, Any]] | None = None,
        debias: int = 1,
        debug: bool = False,
    ) -> tuple[SystemOneResponse, dict[str, Any]]:
        question_dicts = {
            name: question.to_dict() if hasattr(question, "to_dict") else question
            for name, question in questions.items()
        }
        request = SystemOneRequest(
            state=state,
            model=self.model_path.name,
            questions=question_dicts,
            media=media,
        )
        response, meta = self.engine.decide(
            request,
            media=media,
            debias=debias,
            debug=debug,
        )
        response.model = self.model_path.name
        return response, meta
