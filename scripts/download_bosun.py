"""Download the pinned Bosun package and its pinned Qwen base.

This script downloads files only. It does not import checkpoint remote code,
load either model, initialize CUDA, or run inference.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
BOSUN_REPO_ID = "Hanno-Labs/bosun-v3.1-1.7b"
BOSUN_REVISION = "1d8dc82a20e4a32ed60927a47272d6efff48eed2"
BASE_REPO_ID = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

BOSUN_REQUIRED_FILES = (
    "LICENSE",
    "NOTICE",
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
BASE_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
)


def download_snapshot(repo_id: str, revision: str, cache_dir: Path) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
        )
    ).resolve()
    if snapshot.name != revision:
        raise RuntimeError(
            f"unexpected snapshot revision for {repo_id}: "
            f"{snapshot.name} (expected {revision})"
        )
    return snapshot


def require_files(snapshot: Path, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete {label} snapshot; missing: {', '.join(missing)}")


def require_base_weights(snapshot: Path) -> None:
    shards = sorted(snapshot.glob("model-*.safetensors"))
    single_file = snapshot / "model.safetensors"
    if not shards and not single_file.is_file():
        raise FileNotFoundError("incomplete Qwen base snapshot; no Safetensors weights found")


def main() -> None:
    hf_home = Path(os.environ.get("HF_HOME", ROOT / "models" / "huggingface")).expanduser().resolve()
    cache_dir = hf_home / "hub"

    bosun_snapshot = download_snapshot(BOSUN_REPO_ID, BOSUN_REVISION, cache_dir)
    require_files(bosun_snapshot, BOSUN_REQUIRED_FILES, "Bosun")

    base_snapshot = download_snapshot(BASE_REPO_ID, BASE_REVISION, cache_dir)
    require_files(base_snapshot, BASE_REQUIRED_FILES, "Qwen base")
    require_base_weights(base_snapshot)

    print("✓ Downloaded")
    print(f"  bosun_repository: {BOSUN_REPO_ID}")
    print(f"  bosun_revision: {BOSUN_REVISION}")
    print(f"  bosun_path: {bosun_snapshot}")
    print(f"  base_repository: {BASE_REPO_ID}")
    print(f"  base_revision: {BASE_REVISION}")
    print(f"  base_path: {base_snapshot}")


if __name__ == "__main__":
    main()
