"""Run the pinned local Laya checkpoint directly in Python, without HTTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sujevi.laya_local import LayaLocalRuntime


ROOT = Path(__file__).resolve().parents[1]
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
DEFAULT_CHECKPOINT = (
    ROOT
    / "models/huggingface/hub/models--convaiinnovations--laya/snapshots"
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
    runtime = LayaLocalRuntime(args.checkpoint, device=args.device)
    prediction = runtime.predict(request)

    representation_artifact = {
        "model": request.get("model", "Laya"),
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_revision": REVISION,
        "runtime_version": "0.4.24",
        "dtype": str(runtime.agent.dtype).removeprefix("torch."),
        "quantized": False,
        "backend": runtime.agent.backend,
        "temperature": runtime.agent.temperature,
        "temperature_by_options": runtime.agent.temperature_by_options,
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
    print(f"decision_head={type(runtime.decision_head).__module__}.{type(runtime.decision_head).__name__}")
    print(f"option_scorer={type(runtime.option_scorer).__module__}.{type(runtime.option_scorer).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
