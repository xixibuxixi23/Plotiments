# Solaris Stage-3 fine-tuning on Polis

This directory keeps all release adapters and launch scripts outside the upstream
`solaris/` checkout. The target is the pre-distillation, causal multiplayer model
`solaris/pretrained/mp_causal_60000.pt`.

## Action and time mapping

- Polis `action[t]` is paired with `rgb_agent*[t+1]`. This matches Solaris'
  native `read_obs_slice_decord` shift.
- `backward -> back`, `aux1 -> sprint`, `dig -> attack`, `place -> use`.
- `inventory`, `drop`, movement, jump, sneak, and hotbar 1-9 are preserved.
- Polis camera radians are converted to degrees and then processed by Solaris'
  `CameraLinearConverterMatrixGame2` (clip to +/-20 degrees and divide by 15).
- `zoom` has no Solaris action dimension and is intentionally omitted.

## Smoke test

```bash
solaris/.venv/bin/python Plotiments/solaris/convert_polis_release.py \
  --output Plotiments/solaris/data_smoke --max-train 64 --max-test 8
GPU_ID=7 Plotiments/solaris/launch_smoke.sh
```

The launcher disables JAX's large up-front GPU allocation. It still performs a
real forward/backward/update and saves a checkpoint, so memory must be monitored
when it runs concurrently with other jobs.

The formal launcher defaults to one JAX process across GPUs 0-7, 60,000 steps, and a
checkpoint every 1,000 steps. `watch_process_gpu_memory.sh` watches total card
memory and terminates only Solaris if the card has less than 3 GiB free.
