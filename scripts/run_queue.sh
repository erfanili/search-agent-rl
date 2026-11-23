#!/usr/bin/env bash
# GPU queue for the plan's next experiments (notes/new_plan.md), on the GPUs in configs/gpus.env (4-7).
#   0. wait until NOT_BEFORE, then until every queue GPU has no process at all (checked every POLL seconds;
#      QUIET checks in a row)
#   1. E1, seed 1: scripts/run_e1.sh (pilot, forced 40, release to 60, free-depth evals)
#   2. free 60-step run, seed 2: the reference for E1's second seed
#   3. E1, seed 2: scripts/run_e1.sh with SEED=2 and the step-2 run as FREE_REF (skipped by run_e1.sh's disk
#      check if the data disk has under 90 GB free; prune old checkpoints, then run it by hand)
# Stops at the first failure.
#   nohup bash scripts/run_queue.sh > $DATA/logs/queue_<stamp>.log 2>&1 &
#   NOT_BEFORE="2026-11-16 18:00" ...       # change the start time
# Cancel: pkill -f "[r]un_queue.sh"; pkill -f "[r]un_e1.sh"; then pkill -u $USER -f '^ray::'
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpus.sh

NOT_BEFORE=${NOT_BEFORE:-now}
POLL=${POLL:-60}
QUIET=${QUIET:-1}
log() { echo "[$(date '+%F %T')] $*"; }
gpu_procs() {   # number of compute processes on the queue's GPUs, any user
    local n=0 g
    for g in "${GPU_LIST[@]}"; do
        n=$((n + $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)))
    done
    echo "$n"
}

log "queue on GPUs ${GPU_LIST[*]}; not before $NOT_BEFORE"
while (( $(date +%s) < $(date -d "$NOT_BEFORE" +%s) )); do sleep "$POLL"; done
quiet=0
while (( quiet < QUIET )); do
    if (( $(gpu_procs) == 0 )); then quiet=$((quiet + 1)); else quiet=0; fi
    (( quiet < QUIET )) && { (( SECONDS % 3600 < POLL )) && log "waiting for GPUs ${GPU_LIST[*]} to be empty ($(gpu_procs) processes)"; sleep "$POLL"; }
done

log "1. E1 seed 1"
bash scripts/run_e1.sh || { log "E1 seed 1 FAILED"; exit 1; }

log "2. free 60-step run, seed 2"
pkill -u "$USER" -f "^ray::" || true; sleep 10
env TRAIN_BATCH=64 MINI_BATCH=32 CKPT_KEEP=1 SEED=2 DEPTH_POLICY=free TOTAL_STEPS=60 WARMUP_RATIO=0.1 \
    TEST_FREQ=20 SAVE_FREQ=60 bash scripts/train_grpo_3b.sh full > /dev/null || { log "free seed-2 run FAILED"; pkill -u "$USER" -f "^ray::" || true; exit 1; }
FREE2=$(ls -td "$DATA"/logs/grpo_qwen2.5-3b_full_em_seed2_* | head -1)
[[ -d $FREE2/ckpt/global_step_60/actor ]] || { log "no step-60 checkpoint in $FREE2"; exit 1; }
log "free seed-2 run $FREE2"

log "3. E1 seed 2"
SEED=2 FREE_REF="$FREE2/ckpt/global_step_60" bash scripts/run_e1.sh || { log "E1 seed 2 FAILED or skipped"; exit 1; }
log "queue done"
