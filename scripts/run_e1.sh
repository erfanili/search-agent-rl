#!/usr/bin/env bash
# E1 trap test (notes/new_plan.md), first seed, as one queue:
#   1. wait for scripts/run_e0.sh to finish and the GPUs in configs/gpus.env to be free
#   2. pilot: 3 steps at DEPTH_POLICY=exact:$K, no checkpoint; gate = scripts/check_depth_pilot.py
#   3. forced: 40 steps at exact:$K (warmup 6 steps, as the free 60-step run), checkpoint at step 40
#   4. release: resume step 40 with DEPTH_POLICY=free, 20 more steps to 60
#   5. free-depth evals on $EVAL_SPLIT: forced step 40, release step 60, and the free run's step 60 (in parallel)
# Same recipe as the free 60-step run (64 x 5, mini 32, lr 1e-6, EM). Stops at the first failure.
#   nohup bash scripts/run_e1.sh > $DATA/logs/e1_<stamp>.log 2>&1 &
#   K=2 SEED=2 bash scripts/run_e1.sh            # second seed (FREE_REF must then be a seed-2 free run)
# Cancel: pkill -f "[r]un_e1.sh", then stop the current training run (pkill -u $USER -f '^ray::').
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpus.sh

K=${K:-2}
SEED=${SEED:-}
EVAL_SPLIT=${EVAL_SPLIT:-full}
FREE_REF=${FREE_REF:-$DATA/logs/grpo_qwen2.5-3b_full_20261019_183501/ckpt/global_step_60}
MIN_FREE_GB=${MIN_FREE_GB:-90}    # forced + release checkpoints, ~40 GB each, plus merged weights
RECIPE=(TRAIN_BATCH=64 MINI_BATCH=32 CKPT_KEEP=1)
export SEED
log() { echo "[$(date '+%F %T')] $*"; }
trap 'clear_ray' EXIT   # a failed run must not leave Ray workers holding the GPUs (the Nov 11 pilot did)

busy_gpus() {
    for g in "${GPU_LIST[@]}"; do
        (( $(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) > 1000 )) && echo -n "$g "
    done
}
ray_pids() {   # my Ray workers; under gq only this job's (GQ_JOB_ID in their environment)
    local p
    for p in $(pgrep -u "$USER" -f "^ray::"); do
        [[ -z ${GQ_JOB_ID:-} ]] || tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "GQ_JOB_ID=$GQ_JOB_ID" || continue
        echo "$p"
    done
}
clear_ray() {
    local pids
    pids=$(ray_pids); [[ -n $pids ]] && kill $pids 2>/dev/null
    for _ in {1..60}; do [[ -z $(ray_pids) ]] && return 0; sleep 2; done
    pids=$(ray_pids); [[ -n $pids ]] && kill -9 $pids 2>/dev/null; sleep 5
}
newest_run() { ls -td "$DATA"/logs/grpo_qwen2.5-3b_full_"$1"_* 2>/dev/null | head -1; }
train() {   # train <env assignments...>; prints nothing, the run tees to <run>/train.log
    env "${RECIPE[@]}" "$@" bash scripts/train_grpo_3b.sh full > /dev/null
}

while [[ -z ${GQ_JOB_ID:-} ]] && { pgrep -f "[r]un_e0.sh" >/dev/null || [[ -n $(busy_gpus) ]]; }; do   # gq already checked
    log "waiting: E0 running or GPUs busy ($(busy_gpus))"
    sleep 300
done
require_retrieval_server
free_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
(( free_gb >= MIN_FREE_GB )) || { log "only ${free_gb} GB free on $DATA, need $MIN_FREE_GB"; exit 1; }
TAG=em_exact${K}; [[ -z $SEED ]] || TAG+="_*seed$SEED"   # run-name pattern (scripts/train_grpo_3b.sh)

log "pilot: exact:$K, 3 steps"
clear_ray
train DEPTH_POLICY=exact:$K TOTAL_STEPS=3 WARMUP_RATIO=0.1 TEST_FREQ=-1 SAVE_FREQ=-1 || { log "pilot FAILED"; exit 1; }
PILOT=$(newest_run "$TAG")
log "pilot run $PILOT"
.venv/bin/python scripts/check_depth_pilot.py "$PILOT" --k "$K" || { log "pilot gate FAILED, see above"; exit 1; }

log "forced: exact:$K, 40 steps"
clear_ray
train DEPTH_POLICY=exact:$K TOTAL_STEPS=40 WARMUP_RATIO=0.15 TEST_FREQ=20 SAVE_FREQ=40 || { log "forced run FAILED: $(newest_run "$TAG")"; exit 1; }
FORCED=$(newest_run "$TAG")
[[ -d $FORCED/ckpt/global_step_40/actor ]] || { log "no step-40 checkpoint in $FORCED"; exit 1; }
log "forced run $FORCED"

log "release: free from step 40 to 60"
clear_ray
train DEPTH_POLICY=free RESUME_FROM="$FORCED/ckpt/global_step_40" TOTAL_STEPS=60 WARMUP_RATIO=0.1 TEST_FREQ=20 SAVE_FREQ=60 \
    || { log "release FAILED: $(newest_run "em_resume40")"; exit 1; }
RELEASE=$(newest_run "em_resume40")
log "release run $RELEASE"
clear_ray

log "free-depth evals on $EVAL_SPLIT"
pids=()
i=0
for CK in "$FORCED/ckpt/global_step_40" "$RELEASE/ckpt/global_step_60" "$FREE_REF"; do
    G=${WORK_GPUS[$i]}; i=$((i + 1))
    GPU=$G SPLIT=$EVAL_SPLIT bash scripts/eval_checkpoint.sh "$CK" > "$RELEASE/eval_$(basename "$(dirname "$(dirname "$CK")")")_$(basename "$CK").log" 2>&1 &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
log "E1 done (eval status $status). Depth dynamics:"
.venv/bin/python scripts/depth_dynamics.py "$FORCED" "$RELEASE" || true
.venv/bin/python scripts/depth_dynamics.py --split val "$FORCED" "$RELEASE" || true
log "scores: grep -A6 EM $DATA/logs/eval_*{exact${K},resume40}*global_step_*/scores.txt"
