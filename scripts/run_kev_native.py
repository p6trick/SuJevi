"""Run the official Kev model components directly in Python, without HTTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sujevi.kev_local import KevLocalRuntime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "models/huggingface/hub/models--jaredpalmer--kev-4b/snapshots"
    / "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
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
    parser.add_argument(
        "--request",
        type=Path,
        default=ROOT / "tests/smoke_tests/kev_native_request.json",
    )
    parser.add_argument(
        "--native-output",
        type=Path,
        default=ROOT / "results/native_outputs/kev-4b.json",
    )
    parser.add_argument(
        "--probabilities-output",
        type=Path,
        default=ROOT / "results/native_outputs/kev-4b.probabilities.json",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    runtime = KevLocalRuntime(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    prediction = runtime.predict(request)

    distribution_artifact = {
        "model": request.get("model", "kev-latest"),
        "checkpoint": str(runtime.checkpoint_path),
        "base_model": runtime.metadata.base,
        "base_revision": runtime.metadata.base_revision,
        "temperature": runtime.model.head.temperature,
        "dtype": runtime.model.dtype,
        "adapter_merged": False,
        "probability_distributions": prediction.probability_distributions,
    }
    write_json(args.native_output, prediction.native_response, args.overwrite)
    write_json(args.probabilities_output, distribution_artifact, args.overwrite)

    print(json.dumps(prediction.native_response, indent=2, ensure_ascii=False))
    print(f"native_response={args.native_output}")
    print(f"probability_distributions={args.probabilities_output}")
    print(f"adapter_model={type(runtime.adapter_model).__module__}.{type(runtime.adapter_model).__name__}")
    print(f"backbone={type(runtime.backbone).__module__}.{type(runtime.backbone).__name__}")
    print(f"decision_head={type(runtime.decision_head).__module__}.{type(runtime.decision_head).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
