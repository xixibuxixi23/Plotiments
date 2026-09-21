#!/usr/bin/env bash
set -euo pipefail

PID="${1:?usage: watch_process_gpu_memory.sh PID GPU_INDICES [minimum_free_mib] [log]}"
GPU_INDICES="${2:?usage: watch_process_gpu_memory.sh PID GPU_INDICES [minimum_free_mib] [log]}"
MINIMUM_FREE_MIB="${3:-3072}"
LOG="${4:-/dev/stdout}"

while kill -0 "$PID" 2>/dev/null; do
  process_state=$(ps -o stat= -p "$PID" 2>/dev/null | tr -d ' ')
  [[ "$process_state" == Z* ]] && exit 0
  IFS=, read -ra gpu_list <<< "$GPU_INDICES"
  for gpu_index in "${gpu_list[@]}"; do
    stats=$(nvidia-smi --id="$gpu_index" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits)
    IFS=, read -r used free total <<< "$stats"
    used=${used// /}; free=${free// /}; total=${total// /}
    printf '%s pid=%s gpu=%s used_mib=%s free_mib=%s total_mib=%s minimum_free_mib=%s\n' \
      "$(date -u +%FT%TZ)" "$PID" "$gpu_index" "$used" "$free" "$total" "$MINIMUM_FREE_MIB" >> "$LOG"
    if (( free < MINIMUM_FREE_MIB )); then
      printf '%s card safety margin exceeded; terminating Solaris only\n' "$(date -u +%FT%TZ)" >> "$LOG"
      kill -TERM "$PID" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$PID" 2>/dev/null || exit 0
        sleep 1
      done
      kill -KILL "$PID" 2>/dev/null || true
      exit 1
    fi
  done
  sleep 10
done
