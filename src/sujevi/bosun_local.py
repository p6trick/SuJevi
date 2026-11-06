"""In-process Bosun runtime using the checkpoint's official model class."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


BOSUN_REVISION = "1d8dc82a20e4a32ed60927a47272d6efff48eed2"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
QUESTION_TYPES = frozenset({"noul", "choice", "score"})


@dataclass(frozen=True)
class BosunQuestionRepresentation:
    """All routing-relevant outputs from one official Bosun prediction."""

    question_type: str
    option_ids: list[str]
    raw_logits: dict[str, float]
    raw_probability_distribution: dict[str, float]
    calibrated_probability_distribution: dict[str, float]
    option_id_probability_mapping: list[dict[str, float | str]]
    presented_slot_mapping: list[dict[str, float | int | str]]
    candidate_to_slot: dict[str, int]
    slot_logits: list[float]
    slot_probability_distribution: list[float]
    prompt_schema: str
    seed: int
    row_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "option_ids": self.option_ids,
            "raw_logits": self.raw_logits,
            "raw_probability_distribution": self.raw_probability_distribution,
            "calibrated_probability_distribution": self.calibrated_probability_distribution,
            "option_id_probability_mapping": self.option_id_probability_mapping,
            "presented_slot_mapping": self.presented_slot_mapping,
            "candidate_to_slot": self.candidate_to_slot,
            "slot_logits": self.slot_logits,
            "slot_probability_distribution": self.slot_probability_distribution,
            "prompt_schema": self.prompt_schema,
            "probability_rule": "masked_softmax_over_presented_slots",
            "calibration_method": "identity",
            "seed": self.seed,
            "row_id": self.row_id,
            "default_model_representation": self.calibrated_probability_distribution,
        }


@dataclass(frozen=True)
class BosunPrediction:
    """Typed answers and complete per-question Bosun representations."""

    native_response: dict[str, Any]
    representations: dict[str, BosunQuestionRepresentation]


class BosunLocalRuntime:
    """Load pinned local artifacts and call the official Bosun readout.

    The checkpoint's reviewed remote code reconstructs the pinned Qwen base,
    restores learned decision-token embedding/output rows, and applies the
    PEFT adapter. Network access is disabled by ``local_files_only=True``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        base_checkpoint_path: str | Path,
        *,
        cache_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.base_checkpoint_path = Path(base_checkpoint_path).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.device = device
        self.dtype = dtype

        self._validate_snapshots()
        self._validate_manifest()

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.checkpoint_path),
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=str(self.cache_dir),
            dtype=dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _validate_snapshots(self) -> None:
        if self.checkpoint_path.name != BOSUN_REVISION:
            raise ValueError(
                f"unexpected Bosun revision: {self.checkpoint_path.name}; "
                f"expected {BOSUN_REVISION}"
            )
        if self.base_checkpoint_path.name != BASE_REVISION:
            raise ValueError(
                f"unexpected Qwen base revision: {self.base_checkpoint_path.name}; "
                f"expected {BASE_REVISION}"
            )
        required = (
            "config.json",
            "configuration_bosun.py",
            "modeling_bosun.py",
            "decision_embeddings.safetensors",
            "serving.json",
            "manifest.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        )
        missing = [name for name in required if not (self.checkpoint_path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete Bosun snapshot; missing: {missing}")
        if not (self.base_checkpoint_path / "config.json").is_file():
            raise FileNotFoundError("Qwen base config.json is missing")
        if not list(self.base_checkpoint_path.glob("model-*.safetensors")):
            raise FileNotFoundError("Qwen base Safetensors shards are missing")

        config = json.loads((self.checkpoint_path / "config.json").read_text())
        serving = json.loads((self.checkpoint_path / "serving.json").read_text())
        if config["base_model_revision"] != BASE_REVISION:
            raise ValueError("config.json has an unexpected base revision")
        if serving["base_revision"] != BASE_REVISION:
            raise ValueError("serving.json has an unexpected base revision")
        if config["transformers_version"] != "4.57.0":
            raise ValueError("Bosun config does not declare Transformers 4.57.0")
        if config["decision_token_assignment"] != "presented_slot":
            raise ValueError("unsupported Bosun decision-token assignment")

    def _validate_manifest(self) -> None:
        manifest = json.loads((self.checkpoint_path / "manifest.json").read_text())
        if manifest["base_revision"] != BASE_REVISION:
            raise ValueError("manifest has an unexpected base revision")
        for relative_name, expected_digest in manifest["files"].items():
            path = self.checkpoint_path / relative_name
            if not path.is_file():
                raise FileNotFoundError(f"manifest file is missing: {relative_name}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise RuntimeError(f"manifest checksum mismatch: {relative_name}")

    @property
    def adapter_model(self):
        return self.model.model

    @property
    def base_model(self):
        return self.adapter_model.get_base_model()

    @staticmethod
    def _candidates(question_type: str, criteria: Any) -> list[dict[str, str]]:
        if question_type in {"noul", "choice"}:
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"{question_type} criteria must be an object with at least two options")
            if question_type == "noul" and set(criteria) != {"false", "true"}:
                raise ValueError("noul criteria must contain exactly false and true")
            return [
                {
                    "id": str(option_id),
                    "label": str(option_id),
                    "description": str(description),
                }
                for option_id, description in criteria.items()
            ]
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria must be a list with at least two levels")
        return [
            {
                "id": str(index),
                "label": str(index),
                "description": str(description),
            }
            for index, description in enumerate(criteria)
        ]

    @staticmethod
    def _typed_answer(
        question_type: str,
        option_ids: list[str],
        probabilities: dict[str, float],
        criteria: Any,
    ) -> dict[str, Any]:
        selected = max(option_ids, key=probabilities.__getitem__)
        if question_type == "noul":
            return {"type": "noul", "noul": probabilities["true"]}
        if question_type == "choice":
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": probabilities,
            }
        expected_score = sum(int(option_id) * probabilities[option_id] for option_id in option_ids)
        return {
            "type": "score",
            "score": expected_score,
            "legend": {str(index): str(label) for index, label in enumerate(criteria)},
            "probabilities": probabilities,
        }

    @torch.no_grad()
    def predict(self, request: dict[str, Any]) -> BosunPrediction:
        if "state" not in request or "questions" not in request:
            raise ValueError("request must contain state and questions")
        questions = request["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("request.questions must be a nonempty object")

        seed = int(request.get("seed", 0))
        answers: dict[str, Any] = {}
        representations: dict[str, BosunQuestionRepresentation] = {}
        input_tokens = 0
        started = time.perf_counter()

        for question_id, question in questions.items():
            question_type = question.get("type")
            if question_type not in QUESTION_TYPES:
                raise ValueError(f"unsupported question type for {question_id}: {question_type}")
            instructions = question.get("instructions")
            if not isinstance(instructions, str) or not instructions:
                raise ValueError(f"question {question_id} requires nonempty instructions")
            criteria = question.get("criteria")
            candidates = self._candidates(question_type, criteria)
            option_ids = [candidate["id"] for candidate in candidates]
            row_id = str(question_id)

            official = self.model.predict(
                state=request["state"],
                instructions=instructions,
                candidates=candidates,
                decision_type=question_type,
                seed=seed,
                row_id=row_id,
            )
            candidate_to_slot = {
                str(option_id): int(slot)
                for option_id, slot in official["candidate_to_slot"].items()
            }
            slot_logits = [float(value) for value in official["slot_logits"]]
            slot_probabilities = [
                float(value) for value in official["slot_probabilities"]
            ]
            caller_probabilities = [float(value) for value in official["probabilities"]]
            raw_logits = {
                option_id: slot_logits[candidate_to_slot[option_id]]
                for option_id in option_ids
            }
            distribution = {
                option_id: probability
                for option_id, probability in zip(
                    option_ids,
                    caller_probabilities,
                    strict=True,
                )
            }
            mappings = [
                {
                    "option_id": option_id,
                    "presented_slot": candidate_to_slot[option_id],
                    "decision_token": f"<|decision_{candidate_to_slot[option_id]:03d}|>",
                    "raw_logit": raw_logits[option_id],
                    "probability": distribution[option_id],
                }
                for option_id in option_ids
            ]
            representations[question_id] = BosunQuestionRepresentation(
                question_type=question_type,
                option_ids=option_ids,
                raw_logits=raw_logits,
                raw_probability_distribution=dict(distribution),
                calibrated_probability_distribution=dict(distribution),
                option_id_probability_mapping=[
                    {"option_id": option_id, "probability": distribution[option_id]}
                    for option_id in option_ids
                ],
                presented_slot_mapping=mappings,
                candidate_to_slot=candidate_to_slot,
                slot_logits=slot_logits,
                slot_probability_distribution=slot_probabilities,
                prompt_schema=self.model.config.prompt_schema,
                seed=seed,
                row_id=row_id,
            )
            answers[question_id] = self._typed_answer(
                question_type,
                option_ids,
                distribution,
                criteria,
            )
            encoded = self.model.tokenizer(str(official["prompt"]), return_tensors="pt")
            input_tokens += int(encoded["attention_mask"].sum())

        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        native_response = {
            "model": "Bosun-v3.1-1.7B",
            "answers": answers,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0,
            },
            "latency_ms": latency_ms,
        }
        return BosunPrediction(native_response, representations)


__all__ = [
    "BosunLocalRuntime",
    "BosunPrediction",
    "BosunQuestionRepresentation",
]
