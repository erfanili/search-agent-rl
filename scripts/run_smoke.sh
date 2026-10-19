#!/usr/bin/env bash
# Start the P0.7 smoke test in the background and follow its output.
#   bash scripts/run_smoke.sh
# Follows the log until the run ends. Ctrl-C stops following, not the run.
# To stop the run: bash scripts/stop_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh

# warn if another process already holds memory on one of our GPUs (the node is shared)
for g in "${GPU_LIST[@]}"; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( used > 1000 )); then
        echo "warning: GPU $g already has ${used} MiB in use by another process (see nvidia-smi)."
        echo "         Consider removing it from GPUS in configs/gpus.env."
    fi
done

LOGS=/data/erfan/logs   # all run output lives under /data/erfan
mkdir -p "$LOGS"
nohup bash scripts/p0_7_smoke_grpo_gsm8k.sh > "$LOGS/p0_7_launch.log" 2>&1 &
RUN_PID=$!
echo "smoke test started (pid $RUN_PID); output: $LOGS/p0_7_launch.log"
echo
tail -f --pid="$RUN_PID" "$LOGS/p0_7_launch.log"   # exits when the run does

OUT=$(ls -dt "$LOGS"/p0_7_smoke_* | head -1)
echo
echo "run finished; summary of $OUT:"
.venv-verl/bin/python scripts/summarize_smoke.py "$OUT" || true
