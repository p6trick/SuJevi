"""Run a pinned local Jeff checkpoint directly in Python, without HTTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sujevi.jeff_local import JeffLocalRuntime


ROOT = Path(__file__).resolve().parents[1]
REVISION = "5401940b5a8c959cba82187ebfa2dce5a21abc0a"
DEFAULT_CHECKPOINT = (
    ROOT
    / "models/huggingface/hub/models--jgeuter--Jeff-1.1-Qwen-4B/snapshots"
    / REVISION
)


def write_json(path: Path, payload: object, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--representations-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    runtime = JeffLocalRuntime(args.checkpoint, device=args.device)
    prediction = runtime.predict(request)

    representation_artifact = {
        "model": request.get("model", runtime.model.name),
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_revision": REVISION,
        "dtype": "bfloat16",
        "quantized": False,
        "reasoning": False,
        "noul_commitment": False,
        "serving_rule": runtime.serving_rule,
        "default_model_representation": "calibrated_probability_distribution",
        "questions": {
            key: value.as_dict()
            for key, value in prediction.representations.items()
        },
    }
    write_json(args.native_output, prediction.native_response, args.overwrite)
    write_json(args.representations_output, representation_artifact, args.overwrite)

    print(json.dumps(prediction.native_response, indent=2, ensure_ascii=False))
    print(f"native_response={args.native_output}")
    print(f"representations={args.representations_output}")
    print(f"backbone={type(runtime.backbone).__module__}.{type(runtime.backbone).__name__}")
    print(f"base_model={type(runtime.base_model).__module__}.{type(runtime.base_model).__name__}")
    print(f"decision_head={type(runtime.decision_head).__module__}.{type(runtime.decision_head).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
