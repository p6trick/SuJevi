"""In-process Jeff runtime using the official :class:`jeff.JeffModel`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jeff import JeffModel


@dataclass(frozen=True)
class JeffQuestionRepresentation:
    """All routing-relevant outputs from one Jeff forward pass."""

    question_type: str
    option_ids: list[str]
    raw_logits: dict[str, float]
    raw_probability_distribution: dict[str, float]
    calibrated_probability_distribution: dict[str, float]
    option_id_probability_mapping: list[dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "option_ids": self.option_ids,
            "raw_logits": self.raw_logits,
            "raw_probability_distribution": self.raw_probability_distribution,
            "calibrated_probability_distribution": self.calibrated_probability_distribution,
            "option_id_probability_mapping": self.option_id_probability_mapping,
            "default_model_representation": self.calibrated_probability_distribution,
        }


@dataclass(frozen=True)
class JeffPrediction:
    """Official native answers and complete per-question representations."""

    native_response: dict[str, Any]
    representations: dict[str, JeffQuestionRepresentation]


class JeffLocalRuntime:
    """Load and score a local Jeff checkpoint without an HTTP server.

    ``commit=False`` keeps Noul as a calibrated probability instead of applying
    the optional JevBench abstention-band commitment. Reasoning is disabled so
    every question is represented by the released one-pass decision model.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
    ) -> None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        required = (
            "config.json",
            "head.safetensors",
            "head_config.json",
            "jeff_config.json",
            "serving_rule.json",
        )
        missing = [name for name in required if not (checkpoint_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Jeff checkpoint is incomplete under {checkpoint_path}; missing: {missing}"
            )

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = JeffModel(
            checkpoint_path,
            device=device,
            commit=False,
            reasoning=False,
        )

    @property
    def backbone(self):
        """The released Qwen causal language model."""
        return self.model.decider.model

    @property
    def base_model(self):
        """The Qwen text backbone used for the decision forward pass."""
        return self.model.decider.base

    @property
    def decision_head(self):
        """Jeff's fp32 residual decision head."""
        return self.model.decider.head

    @property
    def tokenizer(self):
        return self.model.decider.tokenizer

    @property
    def serving_rule(self) -> dict[str, Any]:
        return self.model.rule

    @staticmethod
    def _calibrated_distribution(
        question_type: str,
        option_ids: list[str],
        answer: dict[str, Any],
    ) -> dict[str, float]:
        if question_type == "noul":
            p_true = float(answer["noul"])
            values = {"true": p_true, "false": 1.0 - p_true}
            return {option_id: values[option_id] for option_id in option_ids}
        return {
            option_id: float(answer["probabilities"][option_id])
            for option_id in option_ids
        }

    def predict(self, request: dict[str, Any]) -> JeffPrediction:
        if "state" not in request or "questions" not in request:
            raise ValueError("request must contain state and questions")
        if not isinstance(request["questions"], dict) or not request["questions"]:
            raise ValueError("request.questions must be a nonempty object")

        self.model.decider.start_request()
        answers: dict[str, Any] = {}
        representations: dict[str, JeffQuestionRepresentation] = {}
        total_input_tokens = 0
        total_latency_seconds = 0.0

        for question_id, question in request["questions"].items():
            row = self.model.row(request["state"], question, key=question_id)
            option_ids = [option["id"] for option in row["options"]]
            if len(option_ids) > 16:
                raise ValueError(
                    "raw-logit representations require at most 16 options; "
                    f"question {question_id!r} has {len(option_ids)}"
                )
            probabilities, logits, input_tokens, latency_seconds = (
                self.model.decider.decide(row)
            )
            answer = self.model.answer(question["type"], option_ids, probabilities)
            calibrated = self._calibrated_distribution(
                question["type"], option_ids, answer
            )

            raw_logits = {
                option_id: float(value)
                for option_id, value in zip(option_ids, logits, strict=True)
            }
            raw_distribution = {
                option_id: float(value)
                for option_id, value in zip(option_ids, probabilities, strict=True)
            }
            option_mapping = [
                {"option_id": option_id, "probability": calibrated[option_id]}
                for option_id in option_ids
            ]

            answers[question_id] = answer
            representations[question_id] = JeffQuestionRepresentation(
                question_type=question["type"],
                option_ids=option_ids,
                raw_logits=raw_logits,
                raw_probability_distribution=raw_distribution,
                calibrated_probability_distribution=calibrated,
                option_id_probability_mapping=option_mapping,
            )
            total_input_tokens += int(input_tokens)
            total_latency_seconds += float(latency_seconds)

        native_response = {
            "model": request.get("model", self.model.name),
            "answers": answers,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": 0,
            },
            "latency_ms": round(total_latency_seconds * 1000, 1),
        }
        return JeffPrediction(native_response, representations)


__all__ = [
    "JeffLocalRuntime",
    "JeffPrediction",
    "JeffQuestionRepresentation",
]
