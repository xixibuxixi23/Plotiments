#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET=${SSH_TARGET:-root@vr.turbo-ai.com}
SSH_PORT=${SSH_PORT:-20470}
GPU_IDS=${GPU_IDS:-0,1,2,3}
NPROC=${NPROC:-4}
MAX_ITER=${MAX_ITER:-10}
NUM_WORKERS=${NUM_WORKERS:-2}
RUN_NAME=${RUN_NAME:-polis_two_player_causal_stage2_remote_$(date -u +%Y%m%d_%H%M%S)}

# Positional arguments keep caller-provided values out of the remote shell
# program text.  SSH_TARGET may also be set to the local SSH-config alias
# "zhizhou-avgen-js-public".
ssh -p "$SSH_PORT" "$SSH_TARGET" bash -s -- \
  "$GPU_IDS" "$NPROC" "$MAX_ITER" "$NUM_WORKERS" "$RUN_NAME" <<'REMOTE'
set -euo pipefail

GPU_IDS=$1
NPROC=$2
MAX_ITER=$3
NUM_WORKERS=$4
RUN_NAME=$5
ROOT=/public/0_DATA/2_Avatar/zhizhou_share/rcz

cd "$ROOT"
GPU_IDS="$GPU_IDS" \
NPROC="$NPROC" \
MAX_ITER="$MAX_ITER" \
NUM_WORKERS="$NUM_WORKERS" \
RUN_NAME="$RUN_NAME" \
  bash Plotiments/gammaworld/launch_smoke.sh
REMOTE
