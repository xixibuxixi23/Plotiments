#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/0_DATA/2_Avatar/zhizhou_share/rcz
GAMMA_ROOT="$ROOT/Gamma-World"
DATA_ROOT="$ROOT/textagent/data/releases/polis_two_player_fixed_skins_complete_20260917_360p"
ADAPTER_ROOT="$ROOT"
PYTHON_BIN=${PYTHON_BIN:-"$ROOT/Plotiments/gammaworld/.venv/bin/python"}
NPROC=${NPROC:-4}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MAX_ITER=${MAX_ITER:-10}
NUM_WORKERS=${NUM_WORKERS:-2}
SAVE_ITER=${SAVE_ITER:-5}
LOGGING_ITER=${LOGGING_ITER:-1}
RUN_NAME=${RUN_NAME:-polis_two_player_causal_stage2_smoke_$(date -u +%Y%m%d_%H%M%S)}
DCP_ROOT=${DCP_ROOT:-"$ROOT/Plotiments/gammaworld/checkpoints/causal_dcp"}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"$GAMMA_ROOT/checkpoints/Gamma-World/causal/model.safetensors"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT/Plotiments/gammaworld/outputs"}
LOG_ROOT=${LOG_ROOT:-"$ROOT/Plotiments/gammaworld/logs"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$GAMMA_ROOT/scripts/train.py"}

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$(dirname "$DCP_ROOT")"
export PYTHONPATH="$ADAPTER_ROOT:$GAMMA_ROOT:$GAMMA_ROOT/packages/cosmos-oss${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

if [[ ! -f "$DCP_ROOT/.metadata" && ! -f "$DCP_ROOT/model/.metadata" ]]; then
  "$PYTHON_BIN" "$GAMMA_ROOT/scripts/convert_checkpoint_to_dcp.py" \
    --input "$SOURCE_CHECKPOINT" \
    --output "$DCP_ROOT"
fi

cd "$ROOT"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  "$TRAIN_SCRIPT" \
  --config Plotiments/gammaworld/config.py -- \
  experiment=polis_causal \
  dataloader_train.data_root="$DATA_ROOT" \
  dataloader_train.num_workers="$NUM_WORKERS" \
  checkpoint.load_path="$DCP_ROOT" \
  checkpoint.load_training_state=false \
  checkpoint.save_iter="$SAVE_ITER" \
  model.config.fsdp_shard_size="$NPROC" \
  trainer.max_iter="$MAX_ITER" \
  trainer.logging_iter="$LOGGING_ITER" \
  trainer.validation_iter=1000000 \
  trainer.run_validation=false \
  job.name="$RUN_NAME" \
  job.wandb_mode=disabled \
  model.config.text_encoder_config.ckpt_path="$GAMMA_ROOT/checkpoints/Cosmos-Reason1-7B" \
  model.config.tokenizer.vae_pth="$GAMMA_ROOT/checkpoints/Gamma-World/tokenizer.pth" \
  2>&1 | tee "$LOG_ROOT/$RUN_NAME.log"
