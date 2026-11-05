"""In-process Laya runtime using the official version-pinned Agent path."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from laya import load
from laya.common import (
    QTYPES,
    collapsed_options,
    collate_items,
    temp_bucket,
    unpermute_probs,
)


@dataclass(frozen=True)
class LayaQuestionRepresentation:
    """All routing-relevant outputs from one Laya decision row."""

    question_type: str
    option_ids: list[str]
    raw_logits: dict[str, float]
    raw_probability_distribution: dict[str, float]
    calibrated_probability_distribution: dict[str, float]
    option_id_probability_mapping: list[dict[str, float | str]]
    temperature_bucket: str
    applied_temperature: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "option_ids": self.option_ids,
            "raw_logits": self.raw_logits,
            "raw_probability_distribution": self.raw_probability_distribution,
            "calibrated_probability_distribution": self.calibrated_probability_distribution,
            "option_id_probability_mapping": self.option_id_probability_mapping,
            "temperature_bucket": self.temperature_bucket,
            "applied_temperature": self.applied_temperature,
            "default_model_representation": self.calibrated_probability_distribution,
        }


@dataclass(frozen=True)
class LayaPrediction:
    """Official native answers and complete per-question representations."""

    native_response: dict[str, Any]
    representations: dict[str, LayaQuestionRepresentation]


class LayaLocalRuntime:
    """Load a local Laya checkpoint and expose pre/post-temperature outputs.

    The runtime follows the official Agent's validation, input construction,
    collation, forward, option-order restoration, temperature selection, and
    answer decoder. No HTTP, ONNX, TileLang, compilation, or quantization path
    is used.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
    ) -> None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        required = (
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "encoder/config.json",
        )
        missing = [name for name in required if not (checkpoint_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Laya checkpoint is incomplete under {checkpoint_path}; missing: {missing}"
            )

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.agent = load(
            str(checkpoint_path),
            device=device,
            fast=False,
            compile=False,
            backend="eager",
        )

    @property
    def backbone(self):
        return self.agent.model.encoder

    @property
    def decision_head(self):
        return self.agent.model.head

    @property
    def option_scorer(self):
        return self.agent.model.scorer

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max()
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum()

    @staticmethod
    def _option_ids(question_type: str, internal_question: dict[str, Any]) -> list[str]:
        if question_type == "noul":
            return ["false", "true"]
        if question_type == "choice":
            return [str(key) for key in internal_question["crit"].keys()]
        return [str(index) for index in range(len(internal_question["crit"]))]

    @torch.no_grad()
    def predict(self, request: dict[str, Any]) -> LayaPrediction:
        if "state" not in request or "questions" not in request:
            raise ValueError("request must contain state and questions")
        questions = request["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("request.questions must be a nonempty object")

        question_ids = list(questions)
        for question_id in question_ids:
            self.agent._check_question(question_id, questions[question_id])
        internal = {
            question_id: self.agent._to_internal(questions[question_id])
            for question_id in question_ids
        }
        items = self.agent._encode_state(request["state"], question_ids, internal)
        batch = collate_items([items], self.agent.tok.pad_token_id)

        started = time.perf_counter()
        logits, act = self.agent._forward(batch)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        answers = self.agent._decode_answers(
            logits,
            act,
            items,
            question_ids,
            internal,
            0,
        )

        representations: dict[str, LayaQuestionRepresentation] = {}
        for row, (question_id, item) in enumerate(zip(question_ids, items, strict=True)):
            question = internal[question_id]
            question_type = question["t"]
            option_count = len(item["markers"])
            qtype = QTYPES[question_type]
            bucket = temp_bucket(qtype, option_count)
            temperature = float(
                self.agent.temperature_by_options.get(
                    bucket,
                    self.agent.temperature[qtype],
                )
            )

            slot_logits = np.asarray(logits[row, :option_count])
            canonical_logits = unpermute_probs(
                slot_logits.copy(),
                question.get("option_order"),
            )
            raw_probabilities = unpermute_probs(
                self._softmax(slot_logits),
                question.get("option_order"),
            )
            calibrated_probabilities = unpermute_probs(
                self._softmax(slot_logits / temperature),
                question.get("option_order"),
            )
            option_ids = self._option_ids(question_type, question)

            raw_logits = {
                option_id: float(value)
                for option_id, value in zip(option_ids, canonical_logits, strict=True)
            }
            raw_distribution = {
                option_id: float(value)
                for option_id, value in zip(option_ids, raw_probabilities, strict=True)
            }
            calibrated_distribution = {
                option_id: float(value)
                for option_id, value in zip(
                    option_ids,
                    calibrated_probabilities,
                    strict=True,
                )
            }
            representations[question_id] = LayaQuestionRepresentation(
                question_type=question_type,
                option_ids=option_ids,
                raw_logits=raw_logits,
                raw_probability_distribution=raw_distribution,
                calibrated_probability_distribution=calibrated_distribution,
                option_id_probability_mapping=[
                    {
                        "option_id": option_id,
                        "probability": calibrated_distribution[option_id],
                    }
                    for option_id in option_ids
                ],
                temperature_bucket=bucket,
                applied_temperature=temperature,
            )

        state_stats = [item["state_stats"] for item in items]
        state_tokens_dropped = max(stat["state_tokens_dropped"] for stat in state_stats)
        usage: dict[str, Any] = {
            "input_tokens": int(batch["attention_mask"].sum()),
            "output_tokens": 0,
            "state_tokens": state_stats[0]["state_tokens"],
            "state_tokens_dropped": state_tokens_dropped,
            "truncated": state_tokens_dropped > 0,
            "truncated_questions": [
                question_id
                for question_id, stat in zip(question_ids, state_stats, strict=True)
                if stat["truncated"]
            ],
        }
        collapsed = collapsed_options(question_ids, items)
        if collapsed:
            usage["options"] = collapsed

        native_response = {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": usage,
            "latency_ms": latency_ms,
        }
        return LayaPrediction(native_response, representations)


__all__ = [
    "LayaLocalRuntime",
    "LayaPrediction",
    "LayaQuestionRepresentation",
]
