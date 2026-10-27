"""Download model snapshots into the Git-ignored project models directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def download(repo_id: str, directory: str) -> None:
    target = MODELS / directory
    print(f"Downloading {repo_id} -> {target}")
    snapshot_download(repo_id, local_dir=target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onejev", action="store_true", help="download OmniJev/OneJev-4B only")
    parser.add_argument("--jev-omni", action="store_true", help="download akhilaaa3/Jev-Omni only")
    args = parser.parse_args()

    selected = args.onejev or args.jev_omni
    if args.onejev or not selected:
        download("OmniJev/OneJev-4B", "OneJev-4B")
    if args.jev_omni or not selected:
        download("akhilaaa3/Jev-Omni", "Jev-Omni")


if __name__ == "__main__":
    main()
