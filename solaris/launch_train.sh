#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/public/0_DATA/2_Avatar/zhizhou_share/rcz
SOLARIS_ROOT="$WORKSPACE/solaris"
EXPERIMENT_ROOT="$WORKSPACE/Plotiments/solaris"
DATA_ROOT="${DATA_ROOT:-$EXPERIMENT_ROOT/data}"
GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
GPU_COUNT=$(awk -F, '{print NF}' <<< "$GPU_ID")
BATCH_SIZE="${BATCH_SIZE:-$GPU_COUNT}"
TOTAL_STEPS="${TOTAL_STEPS:-60000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
RUN_NAME="${RUN_NAME:?RUN_NAME must be set for a resumable formal run}"
RUN_ROOT="$EXPERIMENT_ROOT/outputs/$RUN_NAME"
mkdir -p "$RUN_ROOT" "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/jax_cache"

VENV="$SOLARIS_ROOT/.venv"
NVIDIA_LIBS=$(find "$VENV/lib/python3.10/site-packages/nvidia" -maxdepth 3 -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$WORKSPACE:$SOLARIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HYDRA_FULL_ERROR=1

cd "$SOLARIS_ROOT"
exec "$VENV/bin/python" src/train.py \
  runner=trainer_mp_causal model=solaris dataset=duet device=gpu \
  experiment_name="$RUN_NAME" enable_jax_cache=true \
  +jax_cache_dir="$EXPERIMENT_ROOT/jax_cache" \
  dataset.class=Plotiments.solaris.polis_dataset.DatasetMultiplayerPolis \
  device.data_dir="$DATA_ROOT" \
  device.eval_data_dir="$DATA_ROOT" \
  device.pretrained_model_dir="$SOLARIS_ROOT/pretrained" \
  device.output_dir="$RUN_ROOT/eval" \
  device.checkpoint_dir="$RUN_ROOT/checkpoints" \
  device.jax_cache_dir="$EXPERIMENT_ROOT/jax_cache" \
  device.batch_size="$BATCH_SIZE" device.eval_num_samples="$BATCH_SIZE" device.num_workers=2 \
  runner.params.pretrained_model_path="$SOLARIS_ROOT/pretrained/mp_causal_60000.pt" \
  runner.params.total_steps="$TOTAL_STEPS" \
  runner.params.train_log_every_steps=10 \
  runner.params.test_log_every_steps=1000000 \
  runner.params.eval_every_steps=1000000 \
  runner.params.use_wandb=false \
  runner.params.checkpoint_config.params.save_interval_steps="$SAVE_EVERY" \
  runner.params.checkpoint_config.params.max_to_keep=3 \
  runner.params.save_model_state_to="$RUN_ROOT/final_model.pt" \
  '~eval_datasets.duet'
