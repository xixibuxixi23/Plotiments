#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/0_DATA/2_Avatar/zhizhou_share/rcz
GAMMA_ROOT="$ROOT/Gamma-World"
VENV_DIR=${VENV_DIR:-"$ROOT/Plotiments/gammaworld/.venv"}
UV_BIN=${UV_BIN:-/private/software/conda/bin/uv}

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found or not executable: $UV_BIN" >&2
  exit 1
fi

# Gamma-World's lock file pins the CUDA 12.8 / Torch 2.7 binary stack.  Keeping
# this environment separate avoids loading extensions built for the host's
# Torch 2.8 environment.
UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV_BIN" sync \
  --project "$GAMMA_ROOT" \
  --locked \
  --extra cu128

"$VENV_DIR/bin/python" - <<'PY'
import torch
import torchvision
import flash_attn
import natten
import transformer_engine

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("flash_attn", flash_attn.__version__)
print("natten", natten.__version__)
print("transformer_engine", transformer_engine.__version__)
PY
