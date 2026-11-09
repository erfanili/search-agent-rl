#!/usr/bin/env bash
# Train the reward arms (agent/reward.py) one after another with the 60-step recipe; only REWARD changes.
#   nohup bash scripts/run_reward_arms.sh > $DATA/logs/reward_arms_<stamp>.log 2>&1 &
#   ARMS="em_hit f1" bash scripts/run_reward_arms.sh     # a subset, in this order
# Waits until every GPU in configs/gpus.env is free (another job may hold some), then starts.
# Before each arm it checks disk space: a step-60 checkpoint with optimizer state is ~40 GB.
# Stops at the first arm that fails. Cancel: pkill -f "[r]un_reward_arms.sh", then stop the current run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpus.sh

ARMS=${ARMS:-em_hit em_multi f1}
MIN_FREE_GB=${MIN_FREE_GB:-45}
log() { echo "[$(date '+%F %T')] $*"; }

busy_gpus() {
    for g in "${GPU_LIST[@]}"; do
        (( $(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) > 1000 )) && echo -n "$g "
    done
}
while [[ -n $(busy_gpus) ]]; do
    log "waiting for GPUs: $(busy_gpus)in use"
    sleep 600
done
curl -sf localhost:8000/health >/dev/null || { log "retrieval server not up: bash scripts/serve_retrieval.sh"; exit 1; }

for R in $ARMS; do
    free_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
    (( free_gb >= MIN_FREE_GB )) || { log "only ${free_gb} GB free on $DATA, need $MIN_FREE_GB for a checkpoint; stopping before $R"; exit 1; }
    log "arm $R: start (${free_gb} GB free)"
    # the last run's Ray workers take a while to exit; train_grpo_3b.sh refuses to start while any remain
    pkill -u "$USER" -f "^ray::" || true
    for _ in {1..60}; do pgrep -u "$USER" -f "^ray::" >/dev/null || break; sleep 2; done
    pgrep -u "$USER" -f "^ray::" >/dev/null && pkill -9 -u "$USER" -f "^ray::" && sleep 5
    # the run tees its own output to <run dir>/train.log
    if REWARD=$R TRAIN_BATCH=64 MINI_BATCH=32 TOTAL_STEPS=60 WARMUP_RATIO=0.1 TEST_FREQ=20 SAVE_FREQ=60 CKPT_KEEP=1 \
        bash scripts/train_grpo_3b.sh full > /dev/null; then status=done; else status=FAILED; fi
    log "arm $R: $status, $(ls -td "$DATA"/logs/grpo_qwen2.5-3b_full_"${R}"_* | head -1)"
    [[ $status == done ]] || exit 1
done
