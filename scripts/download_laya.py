"""Download only the pinned English Laya checkpoint files.

This script does not load the model or use the GPU. It mirrors the allow-list
used by the official Laya Agent so sibling multilingual and typed-decisions
weights in the same Hub repository are not downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "convaiinnovations/laya"
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
ALLOW_PATTERNS = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer/*",
    "encoder/*",
)
REQUIRED_FILES = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
)


def main() -> None:
    hf_home = Path(os.environ.get("HF_HOME", ROOT / "models" / "huggingface")).expanduser().resolve()
    snapshot = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            cache_dir=hf_home / "hub",
            allow_patterns=list(ALLOW_PATTERNS),
        )
    ).resolve()

    missing = [name for name in REQUIRED_FILES if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Laya snapshot; missing: {', '.join(missing)}")
    if snapshot.name != REVISION:
        raise RuntimeError(f"unexpected snapshot revision: {snapshot.name} (expected {REVISION})")

    print("✓ Downloaded")
    print(f"  repository: {REPO_ID}")
    print(f"  revision: {REVISION}")
    print(f"  path: {snapshot}")


if __name__ == "__main__":
    main()
