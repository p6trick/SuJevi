"""Run pinned local Bosun artifacts directly in Python, without HTTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sujevi.bosun_local import BASE_REVISION, BOSUN_REVISION, BosunLocalRuntime


ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = ROOT / "models/huggingface/hub"
DEFAULT_CHECKPOINT = (
    HF_CACHE
    / "models--Hanno-Labs--bosun-v3.1-1.7b/snapshots"
    / BOSUN_REVISION
)
DEFAULT_BASE_CHECKPOINT = (
    HF_CACHE
    / "models--Qwen--Qwen3-1.7B/snapshots"
    / BASE_REVISION
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
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--representations-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    runtime = BosunLocalRuntime(
        args.checkpoint,
        args.base_checkpoint,
        cache_dir=HF_CACHE,
        device=args.device,
    )
    prediction = runtime.predict(request)

    representation_artifact = {
        "model": request.get("model", "Bosun-v3.1-1.7B"),
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_revision": BOSUN_REVISION,
        "base_checkpoint": str(runtime.base_checkpoint_path),
        "base_revision": BASE_REVISION,
        "dtype": str(runtime.dtype).removeprefix("torch."),
        "quantized": False,
        "adapter_merged": False,
        "remote_code": True,
        "decision_token_count": runtime.model.config.decision_token_count,
        "decision_token_assignment": runtime.model.config.decision_token_assignment,
        "probability_rule": "masked_softmax_over_presented_slots",
        "calibration_method": "identity",
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
    print(
        "official_model="
        f"{type(runtime.model).__module__}.{type(runtime.model).__name__}"
    )
    print(
        "adapter_model="
        f"{type(runtime.adapter_model).__module__}.{type(runtime.adapter_model).__name__}"
    )
    print(
        "base_model="
        f"{type(runtime.base_model).__module__}.{type(runtime.base_model).__name__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
