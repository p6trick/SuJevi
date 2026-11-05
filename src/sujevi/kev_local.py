"""In-process Kev runtime built from the official trainable model components."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.device import sync
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, admit


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class KevPrediction:
    """Official native answer plus every question's complete distribution."""

    native_response: dict[str, Any]
    probability_distributions: dict[str, dict[str, float]]


class KevLocalRuntime:
    """Load and score Kev without an HTTP server.

    The official LoRA adapter is deliberately not merged into the backbone.
    ``model``, ``adapter_model``, ``backbone``, and ``decision_head`` remain
    accessible for inspection and future training work. Fine-tuning should use
    the official ``kev.train --init_from`` path rather than mutating an
    inference instance.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "bf16",
    ) -> None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not (checkpoint_path / "head.pt").is_file():
            raise FileNotFoundError(f"Kev head.pt not found under {checkpoint_path}")
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype_name = dtype
        self.checkpoint = Checkpoint(str(checkpoint_path))
        self.load_options = LoadOptions(
            dtype=DTYPES[dtype],
            merge=False,
            backend="torch",
            cuda_graphs=False,
            fused=False,
        )
        self.tokenizer, self.model = self.checkpoint.load(device, self.load_options)
        self.model.eval()

    @property
    def metadata(self):
        return self.checkpoint.meta

    @property
    def adapter_model(self):
        """The unmerged PEFT model wrapping the backbone."""
        return self.model.lm

    @property
    def backbone(self):
        """The underlying Qwen model beneath PEFT, when available."""
        getter = getattr(self.adapter_model, "get_base_model", None)
        return getter() if getter is not None else self.adapter_model

    @property
    def decision_head(self):
        """Kev's fp32 pointer head."""
        return self.model.head

    def predict(self, request: dict[str, Any] | SystemOneRequest) -> KevPrediction:
        req = (
            request
            if isinstance(request, SystemOneRequest)
            else SystemOneRequest.model_validate(request)
        )
        record, question_metadata = to_record(req)
        encoding = admit(self.model, self.tokenizer, record, truncate=False)

        sync(self.device)
        started = time.perf_counter()
        probabilities = self.model.probs(encoding)
        sync(self.device)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        probability_rows = [row.tolist() for row in probabilities]
        answers = to_answers(probability_rows, question_metadata)
        distributions = {
            metadata["id"]: {
                key: float(value)
                for key, value in zip(metadata["keys"], row)
            }
            for metadata, row in zip(question_metadata, probability_rows)
        }
        native_response = {
            "model": req.model,
            "answers": answers,
            "usage": {
                "input_tokens": len(encoding["ids"]),
                "output_tokens": output_tokens(self.tokenizer, answers),
            },
            "latency_ms": latency_ms,
        }
        return KevPrediction(native_response, distributions)


__all__ = ["KevLocalRuntime", "KevPrediction", "SERVE_MAX_STATE", "SERVE_MAX_BRANCH"]
