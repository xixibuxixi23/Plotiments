# Gamma-World stage-2 causal fine-tuning on Polis

This directory contains the adapter and launch files for fine-tuning Gamma-World on
`polis_two_player_fixed_skins_complete_20260917_360p`.  Gamma-World itself is not modified.

## Data contract

- Input observations: synchronized `rgb_agent0.mp4` and `rgb_agent1.mp4`, 24 FPS.
- Polis transition contract: `observation[t] -> action[t] -> observation[t+1]` with `T+1`
  observations and `T` actions.
- Gamma-facing windows use a zero action at the first observation.  Frame `j>0` receives the
  transition(s) between sampled frames `j-1` and `j`.  Discrete keys are OR-reduced and mouse
  deltas are summed if temporal downsampling is enabled.
- Polis key names are mapped into the Solaris/VPT 23-slot keyboard order.  `aux1 -> sprint`,
  `dig -> attack`, and `place -> use`; `zoom` has no Solaris equivalent and is intentionally
  dropped.  Mouse x/y are emitted separately as Gamma `camera` using the released example's
  `raw * rad2deg / 15` conversion.
- The two agent streams are concatenated along Gamma's time/view axis as
  `[agent0 frames..., agent1 frames...]`, with matching view indices and action tensors.

## Validate one full sample

```bash
cd /public/0_DATA/2_Avatar/zhizhou_share/rcz
PYTHONPATH="$PWD:$PWD/Gamma-World:$PWD/Gamma-World/packages/cosmos-oss" \
  Plotiments/gammaworld/.venv/bin/python \
  Plotiments/gammaworld/inspect_dataset.py \
  --data-root "$PWD/textagent/data/releases/polis_two_player_fixed_skins_complete_20260917_360p"
```

## Four-GPU stage-2 causal smoke fine-tune

`launch_smoke.sh` fine-tunes the pre-DMD, block-causal student (the second stage in
`bidirectional teacher -> causal student -> DMD/few-step student`). It converts the released
`causal/model.safetensors` checkpoint to DCP once, then runs ten iterations by default. It never
kills existing GPU processes. Outputs follow Gamma's native layout:
`outputs/cosmos_v2_causal_av/causal_cosmos2/<RUN_NAME>`.

Create the isolated CUDA 12.8 / Torch 2.7 environment once if `.venv` is absent:

```bash
bash /public/0_DATA/2_Avatar/zhizhou_share/rcz/Plotiments/gammaworld/setup_env.sh
```

Launch on the current machine:

```bash
GPU_IDS=0,1,2,3 MAX_ITER=10 \
  bash /public/0_DATA/2_Avatar/zhizhou_share/rcz/Plotiments/gammaworld/launch_smoke.sh
```

Or launch through the requested SSH endpoint (`root@vr.turbo-ai.com:20470`):

```bash
GPU_IDS=0,1,2,3 MAX_ITER=10 \
  bash /public/0_DATA/2_Avatar/zhizhou_share/rcz/Plotiments/gammaworld/launch_remote_smoke.sh
```

If the caller has the SSH config entry, `SSH_TARGET=zhizhou-avgen-js-public` can be used instead.

Override `RUN_NAME`, `NUM_WORKERS`, `DCP_ROOT`, or `OUTPUT_ROOT` as needed.  The output and log
directories live below this directory.

## Eight-GPU formal training

`launch_train.sh` uses all eight GPUs, the official 150,000-step stage-2 schedule, and saves every
1,000 steps. Each rank's PyTorch allocator is capped at 72 GiB, leaving 8 GiB below the requested
80 GiB per-process ceiling for CUDA/NCCL allocations. The four-GPU smoke run measured only about
32 GiB per rank.

```bash
nohup bash /public/0_DATA/2_Avatar/zhizhou_share/rcz/Plotiments/gammaworld/launch_train.sh \
  > /public/0_DATA/2_Avatar/zhizhou_share/rcz/Plotiments/gammaworld/logs/formal_launcher.log 2>&1 &
```

Active formal run: `polis_causal_stage2_8gpu_formal_20260918_091728`. It started from the released
stage-2 causal checkpoint with strict weight loading, uses the native 24-latent-frame setting,
and has `max_iter=150000`, `save_iter=1000`, and `fsdp_shard_size=8`.

## Verified stage-2 causal smoke run

The remote four-GPU run `polis_causal_stage2_4gpu_smoke_remote_20260918` loaded all 624 causal
checkpoint tensors with no missing, unexpected, or incorrectly-shaped keys. It completed two
iterations with losses `0.1633` and `0.1722`, then saved `iter_000000002`. The steady second step
took `23.27s`; peak allocated memory was about `31.84GB` per rank. This verifies the training
path, not convergence or rollout quality.

## Earlier teacher smoke run

The earlier remote four-GPU run `polis_bidirectional_4gpu_smoke_remote_20260918_0810` completed two
training iterations and saved `iter_000000002`.  The losses were `0.1542` and `0.1902`; this
only verified the teacher path. It is retained for comparison and is not the stage-2 target used
by the current launch scripts.
