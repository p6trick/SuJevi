"""Run the pinned Jeva checkpoint locally and preserve its decision outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from jeva import Choice, Jeva, Noul, Score


ROOT = Path(__file__).resolve().parents[1]
REVISION = "25ca454b3522615a243c06703ba410ecf152a128"
DEFAULT_CHECKPOINT = (
    ROOT / "models/huggingface/hub/models--LIQIIIII--Jeva-4B/snapshots" / REVISION
)


def make_question(question: dict):
    kind = question["type"]
    instructions = question["instructions"]
    if kind == "noul":
        return Noul(instructions=instructions)
    if kind == "choice":
        return Choice(instructions=instructions, criteria=question["criteria"], allow_null=False)
    if kind == "score":
        criteria = question["criteria"]
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria must be a list of at least two descriptions")
        return Score(instructions=instructions, criteria={str(i): v for i, v in enumerate(criteria)},
                     values=list(range(len(criteria))))
    raise ValueError(f"unsupported question type: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--representations-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.name != REVISION:
        raise ValueError(f"checkpoint snapshot must have the pinned revision name {REVISION}")
    for name in ("config.yaml", "head.pt", "backbone/config.json",
                 "backbone_adapter/adapter_config.json", "backbone_adapter/adapter_model.safetensors",
                 "tokenizer/tokenizer.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"missing checkpoint file: {name}")
    output_paths = (args.native_output, args.representations_output)
    if output_paths[0].resolve() == output_paths[1].resolve():
        raise ValueError("native and representation output paths must differ")
    for path in output_paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    questions = {key: make_question(value) for key, value in request["questions"].items()}
    if not questions:
        raise ValueError("request must contain at least one question")

    runtime = Jeva.from_pretrained(
        str(checkpoint), device=args.device, dtype=torch.bfloat16,
        merge_lora=False, cache="none", local_files_only=True, trust_remote_code=False,
    )
    original_decide = runtime.model.decide_from_option_cache
    captured = []

    def capture(*call_args, **call_kwargs):
        output = original_decide(*call_args, **call_kwargs)
        captured.append({key: output[key].detach().float().cpu().clone()
                         for key in ("scores", "decision_logits", "q", "tau_dec", "coverage_logit", "p_null")})
        return output

    with patch.object(runtime.model, "decide_from_option_cache", side_effect=capture):
        response = runtime.system_one(request["state"], questions)
    if len(captured) != len(questions):
        raise RuntimeError("unexpected Jeva decision count; refusing to associate outputs")

    representations = {}
    for (key, question), tensors in zip(questions.items(), captured, strict=True):
        answer = response.answers[key]
        ids = list(answer.probabilities)
        scores = tensors["scores"].reshape(-1)
        decision_logits = tensors["decision_logits"].reshape(-1)
        native_q = tensors["q"].reshape(-1)
        if len(ids) != len(scores) or not torch.isfinite(scores).all().item():
            raise RuntimeError(f"invalid scores or option count for {key}")
        raw_probs = torch.softmax(scores, dim=-1).tolist()
        native_probs = native_q.tolist()
        if not math.isclose(sum(native_probs), 1.0, abs_tol=1e-5):
            raise RuntimeError(f"native distribution for {key} is not normalized")
        if any(not math.isclose(answer.probabilities[i], p, abs_tol=1e-6)
               for i, p in zip(ids, native_probs, strict=True)):
            raise RuntimeError(f"native response differs from captured distribution for {key}")
        mapping = lambda values: dict(zip(ids, values, strict=True))
        representations[key] = {
            "question_type": request["questions"][key]["type"],
            "option_ids": ids,
            "raw_logits": mapping(scores.tolist()),
            "raw_probability_distribution": mapping(raw_probs),
            "decision_logits": mapping(decision_logits.tolist()),
            "native_probability_distribution": mapping(native_probs),
            "calibrated_probability_distribution": None,
            "calibration_status": "unverified_no_posthoc_calibration_identified",
            "learned_decision_temperature": float(tensors["tau_dec"].item()),
            "option_id_probability_mapping": [
                {"option_id": i, "probability": p} for i, p in zip(ids, native_probs, strict=True)
            ],
            "native_head_null_probability": float(tensors["p_null"].item()),
            "coverage_logit": float(tensors["coverage_logit"].item()),
            "null_included_in_distribution": False,
            "default_model_representation": mapping(native_probs),
        }
    artifact = {
        "model": "Jeva-4B", "checkpoint": str(checkpoint), "checkpoint_revision": REVISION,
        "dtype": "bfloat16", "quantized": False, "adapter_merged": runtime.lora_merged,
        "cache": "none", "default_model_representation": "native_probability_distribution",
        "questions": representations,
    }
    for path, payload in zip(output_paths, (response.to_dict(), artifact), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
    print(f"native_response={args.native_output}")
    print(f"representations={args.representations_output}")
    print(f"model={type(runtime.model).__module__}.{type(runtime.model).__name__}")
    print(f"backbone={type(runtime.model.backbone.model).__module__}.{type(runtime.model.backbone.model).__name__}")
    print(f"adapter_merged={runtime.lora_merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
