"""Run one text decision using the pinned official Liquid d1 runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[1]
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
DEFAULT_CHECKPOINT = ROOT / "models/huggingface/hub/models--LiquidAI--d1-3B/snapshots" / REVISION


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
        raise ValueError(f"checkpoint must be the pinned snapshot {REVISION}")
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "modeling_d1.py", "lfm2_vl.py", "hybrid.py", "runner.py", "prompt.py", "api.py"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"missing checkpoint file: {name}")
    paths = (args.native_output, args.representations_output)
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("output paths must be distinct")
    for path in paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    questions = request["questions"]
    if len(questions) != 1:
        raise ValueError("this smoke runtime supports exactly one text question per request")
    key, question = next(iter(questions.items()))
    kind = question["type"]
    if kind not in {"noul", "choice", "score"}:
        raise ValueError(f"unsupported question type: {kind}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(str(checkpoint), trust_remote_code=True,
                                      local_files_only=True, dtype=dtype).to(args.device).eval()
    engine = model.engine
    if engine.calibration is not None:
        raise RuntimeError("unexpected calibration object; inspect its rule before continuing")
    prompt = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".prompt")
    captured = {}
    original_readout = engine._readout

    def capture_readout(q, logz):
        captured["question"] = q
        captured["logz"] = logz.detach().float().cpu().clone()
        return original_readout(q, logz)

    def capture_forward(_module, _args, output):
        captured["vocab_logits"] = output.logits[0, -1].detach().float().cpu().clone()

    hook = model.register_forward_hook(capture_forward)
    try:
        with patch.object(engine, "_readout", side_effect=capture_readout):
            native = model.system_one(request["state"], questions)
    finally:
        hook.remove()
    if set(captured) != {"question", "logz", "vocab_logits"}:
        raise RuntimeError("expected one-question forward/readout capture was not obtained")
    groups = prompt.readout_ids(engine.tokenizer, captured["question"])
    if kind == "noul":
        ids = ["true", "false"]  # official readout returns yes then no
    elif kind == "choice":
        ids = list(question["criteria"])
    else:
        ids = [str(i) for i in range(len(question["criteria"]))]
    if len(ids) != len(groups):
        raise RuntimeError("option-to-token group mismatch")
    logits = captured["vocab_logits"]
    logz = captured["logz"]
    scores = [max(float(logits[token]) for token in group) for group in groups]
    probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0).tolist()
    official_probs = original_readout(captured["question"], logz)
    if any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
           for a, b in zip(probabilities, official_probs, strict=True)):
        raise RuntimeError("raw-logit probabilities differ from official readout")
    answer = native["answers"][key]
    native_probs = ([answer["noul"], 1.0 - answer["noul"]] if kind == "noul"
                    else [answer["probabilities"][option] for option in ids])
    if any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
           for a, b in zip(official_probs, native_probs, strict=True)):
        raise RuntimeError("native response differs from official option readout")

    representation = {
        "question_type": kind,
        "option_ids": ids,
        "raw_logits": dict(zip(ids, scores, strict=True)),
        "raw_probability_distribution": dict(zip(ids, probabilities, strict=True)),
        "native_probability_distribution": dict(zip(ids, official_probs, strict=True)),
        "calibrated_probability_distribution": None,
        "calibration_status": "no_posthoc_calibration_object_in_official_default_runtime",
        "option_id_probability_mapping": [
            {"option_id": option, "probability": probability}
            for option, probability in zip(ids, official_probs, strict=True)
        ],
        "readout": "maximum raw logit within each official option token group",
        "option_token_ids": dict(zip(ids, groups, strict=True)),
        "option_token_logits": {option: {str(token): float(logits[token]) for token in group}
                                for option, group in zip(ids, groups, strict=True)},
        "default_model_representation": dict(zip(ids, official_probs, strict=True)),
    }
    artifact = {
        "model": "Liquid-d1-3B", "checkpoint": str(checkpoint), "checkpoint_revision": REVISION,
        "dtype": str(dtype).removeprefix("torch."), "quantized": False,
        "remote_code": True, "default_model_representation": "native_probability_distribution",
        "questions": {key: representation},
    }
    for path, payload in zip(paths, (native, artifact), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(native, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"native_response={args.native_output}")
    print(f"representations={args.representations_output}")
    print(f"official_model={type(model).__module__}.{type(model).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
