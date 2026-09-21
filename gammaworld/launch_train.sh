#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/0_DATA/2_Avatar/zhizhou_share/rcz
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
MAX_ITER=${MAX_ITER:-150000}
SAVE_ITER=${SAVE_ITER:-1000}
LOGGING_ITER=${LOGGING_ITER:-10}
NUM_WORKERS=${NUM_WORKERS:-2}
RUN_NAME=${RUN_NAME:-polis_causal_stage2_8gpu_$(date -u +%Y%m%d_%H%M%S)}

# Cap the PyTorch caching allocator at 72 GiB per rank.  The remaining 8 GiB
# below the requested 80 GiB ceiling is reserved for CUDA/NCCL allocations that
# are not managed by the PyTorch allocator.  The measured workload uses ~32 GiB.
PYTORCH_PROCESS_MEMORY_LIMIT_MIB=${PYTORCH_PROCESS_MEMORY_LIMIT_MIB:-73728}
TRAIN_SCRIPT="$ROOT/Plotiments/gammaworld/train_with_memory_cap.py"

IFS=',' read -r -a selected_gpus <<<"$GPU_IDS"
if [[ ${#selected_gpus[@]} -ne "$NPROC" ]]; then
  echo "GPU_IDS contains ${#selected_gpus[@]} devices but NPROC=$NPROC" >&2
  exit 1
fi

export PYTORCH_PROCESS_MEMORY_LIMIT_MIB
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

exec env \
  GPU_IDS="$GPU_IDS" \
  NPROC="$NPROC" \
  MAX_ITER="$MAX_ITER" \
  SAVE_ITER="$SAVE_ITER" \
  LOGGING_ITER="$LOGGING_ITER" \
  NUM_WORKERS="$NUM_WORKERS" \
  RUN_NAME="$RUN_NAME" \
  TRAIN_SCRIPT="$TRAIN_SCRIPT" \
  bash "$ROOT/Plotiments/gammaworld/launch_smoke.sh"
