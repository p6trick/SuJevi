"""Direct BF16 text readout following the pinned OpenJev helper settings.

This is an adaptation of the helper's one-question text path, not the vLLM
server; numerical parity with the measured FP8 deployment is not established.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
REVISION = "b4a89f5884f3482a5e894b2e7ff01db8ea033726"
DEFAULT_CHECKPOINT = ROOT / "models/huggingface/hub/models--openjev--openjev/snapshots" / REVISION
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
READOUT_T = 0.85
NOUL_T = 1.829074
NOUL_BIAS = 0.0


def desc(value):
    return "" if value is None else value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def prepare(state, question):
    if not isinstance(state, (str, dict)):
        raise ValueError("state must be text or a JSON object")
    if isinstance(state, dict) and any(k in state for k in ("image", "screenshot")):
        raise ValueError("this adapter only supports text inputs")
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    instruction = question.get("instructions")
    if instruction is None:
        raise ValueError("instructions is required")
    instruction = instruction if isinstance(instruction, str) else str(instruction)
    kind = question["type"]
    if kind == "noul":
        criteria = question.get("criteria") or {}
        if not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object")
        options = [("yes", desc(criteria.get("true")) or "The statement is true."),
                   ("no", desc(criteria.get("false")) or "The statement is false.")]
        ids = ["true", "false"]
    elif kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice criteria must be a nonempty object")
        options = [(k, desc(v)) for k, v in criteria.items()]
        ids = list(criteria)
    elif kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria must contain at least two levels")
        options = [(str(i), desc(v)) for i, v in enumerate(criteria)]
        ids = [str(i) for i in range(len(criteria))]
        instruction += " Rate along the ordered levels below (lowest first)."
    else:
        raise ValueError(f"unsupported question type: {kind}")
    if len(options) > len(LETTERS):
        raise ValueError("this single-pass adapter supports at most 52 options")
    lines = "\n".join(f"[{LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(options))
    content = (f"State:\n{state_text}\n\nQuestion: {instruction}\nOptions:\n{lines}"
               "\n\nAnswer with the letter of the best option only.")
    return kind, ids, content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--representations-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.name != REVISION:
        raise ValueError(f"checkpoint must be the pinned snapshot {REVISION}")
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "helper/shim.py"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"missing checkpoint file: {name}")
    for i in range(1, 13):
        name = f"model-{i:05d}-of-00012.safetensors"
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"missing checkpoint shard: {name}")
    paths = (args.native_output, args.representations_output)
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("output paths must be distinct")
    for path in paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    questions = request["questions"]
    if len(questions) != 1:
        raise ValueError("this smoke runtime requires exactly one question")
    key, question = next(iter(questions.items()))
    kind, option_ids, content = prepare(request["state"], question)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    token_ids = []
    for letter in LETTERS[:len(option_ids)]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"option letter {letter!r} is not a single token")
        token_ids.append(encoded[0])
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("option letters do not map to unique token IDs")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this BF16 smoke runtime")
    model = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint), dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, logits_to_keep=1)
    logits = output.logits[0, -1].float().cpu()
    scores = logits[token_ids].to(torch.float64)
    if not torch.isfinite(scores).all().item():
        raise RuntimeError("non-finite option logits")
    raw = torch.softmax(scores, dim=0).tolist()
    intermediate = torch.softmax(scores / READOUT_T, dim=0).tolist()
    if kind == "noul":
        p = min(max(intermediate[0], 1e-4), 1 - 1e-4)
        z = math.log(p / (1 - p)) / NOUL_T + NOUL_BIAS
        calibrated_true = 1 / (1 + math.exp(-z))
        calibrated = [calibrated_true, 1 - calibrated_true]
        answer = {"type": "noul", "noul": round(calibrated_true, 4)}
    else:
        calibrated = intermediate
        winner = max(range(len(calibrated)), key=calibrated.__getitem__)
        if kind == "choice":
            n = len(calibrated)
            confidence = 1.0 if n == 1 else max(0.0, (max(calibrated) - 1 / n) / (1 - 1 / n))
            answer = {"type": "choice", "choice": option_ids[winner],
                      "probabilities": dict(zip(option_ids, map(lambda p: round(p, 4), calibrated), strict=True)),
                      "confidence": round(confidence, 4)}
        else:
            n = len(calibrated)
            distance = sum(p * abs(i - winner) for i, p in enumerate(calibrated))
            center = (n - 1) / 2
            uniform_mad = sum(abs(i - center) for i in range(n)) / n
            confidence = max(0.0, 1 - distance / uniform_mad)
            answer = {"type": "score", "score": round(sum(i * p for i, p in enumerate(calibrated)), 4),
                      "legend": dict(zip(option_ids, question["criteria"], strict=True)),
                      "probabilities": dict(zip(option_ids, map(lambda p: round(p, 4), calibrated), strict=True)),
                      "confidence": round(confidence, 4)}
    native = {"model": "OpenJev-27B-direct-bf16", "answers": {key: answer},
              "usage": {"input_tokens": inputs["input_ids"].shape[1], "output_tokens": 0}}
    mapping = lambda values: dict(zip(option_ids, values, strict=True))
    representation = {
        "question_type": kind, "option_ids": option_ids,
        "option_letters": mapping(list(LETTERS[:len(option_ids)])),
        "option_token_ids": mapping(token_ids), "raw_logits": mapping(scores.tolist()),
        "raw_probability_distribution": mapping(raw),
        "readout_temperature_probability_distribution": mapping(intermediate),
        "calibrated_probability_distribution": mapping(calibrated),
        "calibration_method": "official_helper_temperature_and_noul_slope" if kind == "noul" else "official_helper_temperature",
        "readout_temperature": READOUT_T, "noul_temperature": NOUL_T if kind == "noul" else None,
        "noul_bias": NOUL_BIAS if kind == "noul" else None,
        "option_id_probability_mapping": [{"option_id": i, "probability": p}
                                          for i, p in zip(option_ids, calibrated, strict=True)],
        "default_model_representation": mapping(calibrated),
    }
    artifact = {"model": "OpenJev-27B", "checkpoint": str(checkpoint), "checkpoint_revision": REVISION,
                "dtype": "bfloat16", "quantized": False, "transport": "direct_python",
                "official_vllm_fp8_parity": "not_verified", "helper_prompt_parity": "adapted_not_verified_against_server",
                "default_model_representation": "calibrated_probability_distribution", "questions": {key: representation}}
    for path, payload in zip(paths, (native, artifact), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(native, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"native_response={args.native_output}")
    print(f"representations={args.representations_output}")
    print(f"model={type(model).__module__}.{type(model).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
