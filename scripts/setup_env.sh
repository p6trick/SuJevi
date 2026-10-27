#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${SUJEVI_ENV_NAME:-sujevi}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml"
else
  conda env update -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml"
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

case "$(uname -m)" in
  aarch64|arm64)
    echo "Installing the CUDA-enabled ARM64 PyTorch build for DGX Spark"
    python -m pip install --upgrade uv
    uv pip install --python "$CONDA_PREFIX/bin/python" --torch-backend=auto --prerelease=allow torch torchvision
    ;;
  x86_64)
    TORCH_INDEX="${SUJEVI_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
    python -m pip install torch torchvision --index-url "$TORCH_INDEX"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

python -m pip install -e "$PROJECT_ROOT[train]"
python -m pip install -e "$PROJECT_ROOT/third_party/OneJev[train]"
python -m pip check

python - <<'PY'
import platform
import torch
import transformers

print(f"architecture={platform.machine()}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the unified SuJevi environment")
print(f"device={torch.cuda.get_device_name(0)}")
PY

echo "Environment ready: conda activate $ENV_NAME"
