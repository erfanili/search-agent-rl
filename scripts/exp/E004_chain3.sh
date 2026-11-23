#!/usr/bin/env bash
# E004 chain, version 3 (Erfan, 2026-11-21, after arm A's CUDA OOM at step 79).
#   T1=<tokens> T2=<tokens> bash scripts/exp/E004_chain3.sh > $DATA_ROOT/logs/E004/chain3.log 2>&1
# All training here: actor ppo_max_token_len_per_gpu = T1
# (MAX_TOKENS_PER_GPU; log-prob passes 4x), for both arms.
#  1. arm A: resume from ckpt/global_step_60 (segment logs of steps 61-79 already archived as *.part2.*).
#  2. per arm, on a CUDA OOM (OutOfMemoryError in train.log): archive the segment's logs as *.part<k>.*, trim
#     reward_steps.jsonl to the latest full checkpoint, resume ONCE from it with ppo_max_token_len_per_gpu = T2.
#     A second OOM stops that arm (its confirmed checkpoints are evaluated) and the chain continues.
#     A failure that is not an OOM stops the whole chain (not covered by Erfan's rule). Stall watchdog: train.log silent
#     for 30 min -> training killed, counted as a non-OOM failure (the 22:19 hang sat silent for 74 min).
#  3. arm A finished with hf/step_100 confirmed -> delete its global_step_100 shards + optimizer state (approved);
#     if arm A stopped early its latest full checkpoint is kept (no approval to delete it).
#  4. arm B (fresh), disk guard as in E004_chain2.sh (STOP_DISK -> chain stops, ask Erfan).
#  5. evaluation of every confirmed checkpoint (E004_eval_all.sh).
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
# PYTORCH_CUDA_ALLOC_CONF: taken from the environment (default unset). expandable_segments:True is NOT usable here:
# vLLM's sleep-mode CuMemAllocator asserts against it at rollout-engine load (attempt r3/part3, 2026-11-21 22:19, hung).
[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF || unset PYTORCH_CUDA_ALLOC_CONF
A_TAG=${A_TAG:-r4}
: ${T1:?set T1} ${T2:?set T2}
E=$DATA_ROOT/logs/E004
log() { echo "[$(date '+%F %T')] $*"; }
alive() { pgrep -u $USER -f "default_local_dir=$1/ckpt" >/dev/null; }
pid_alive() { local w; w=$(cat $1/ckpt_watch.pid 2>/dev/null); [ -n "$w" ] && kill -0 $w 2>/dev/null; }
free_gb() { df -BG --output=avail /data | tail -1 | tr -dc 0-9; }
CK=$(du -cBG $E/E004-A-s0/ckpt/global_step_60/actor/{model,optim,extra_state}_world_size_*.pt | tail -1 | tr -dc 0-9)
NEED=$((CK + 7 + 3))   # full checkpoint + merge (6.2 GB) + 3 GB
log "T1=$T1 T2=$T2 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset} NEED=${NEED} GB free=$(free_gb) GB"

gpus_free() {   # wait up to 5 min for GPUs 4-7 to drop below 1 GB used after a crash
  pkill -u $USER -f '^ray::' 2>/dev/null; .venv-verl/bin/ray stop --force >/dev/null 2>&1
  for i in $(seq 30); do
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>=4 && $2>1000' | wc -l)
    [ "$u" = 0 ] && return 0; sleep 10
  done
  log "GPUs 4-7 still in use after 5 min"; return 1
}

archive() {     # $1 run dir: move this segment's logs to *.part<k>.*, trim reward_steps to the latest full checkpoint
  local d=$1 k=1 L
  while [ -e $d/train.part$k.log ]; do k=$((k + 1)); done
  L=$(cat $d/ckpt/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
  cp -p $d/reward_steps.jsonl $d/reward_steps.part$k.jsonl
  python3 -c "
import json,sys
L=[l for l in open(sys.argv[1]) if not (json.loads(l)['split']=='train' and json.loads(l)['step']>int(sys.argv[2]))]
open(sys.argv[1],'w').writelines(L)" $d/reward_steps.jsonl $L
  mv $d/train.log $d/train.part$k.log; mv $d/config.txt $d/config.part$k.txt; mv $d/git.diff $d/git.part$k.diff 2>/dev/null
  mv $d/hydra $d/hydra.part$k; mv $d/rollouts $d/rollouts.part$k; mv $d/ckpt_watch.log $d/ckpt_watch.part$k.log
  log "archived $(basename $d) segment as part$k; reward_steps trimmed to step $L" >&2
  echo $L
}

guard_b() {     # $1 training pid: disk guard for arm B
  local d=$E/E004-B-s0 checked="" s F
  sleep 120
  while kill -0 $1 2>/dev/null; do
    for s in 20 40 60 80; do
      [[ " $checked " == *" $s "* ]] && continue
      [ -f $d/hf/step_$s/LOAD_OK.json ] || continue
      sleep 180; checked="$checked $s"; F=$(free_gb)
      log "arm B step $s confirmed; free ${F} GB; NEED ${NEED} GB"
      if [ "$F" -lt "$NEED" ]; then
        echo "after step $s: free ${F} GB < NEED ${NEED} GB (latest full checkpoint global_step_$s kept)" > $d/STOP_DISK
        pkill -u $USER -f "default_local_dir=$d/ckpt"; sleep 30; pkill -u $USER -f '^ray::'
        log "STOP_DISK: arm B training killed after step $s"; return 1
      fi
    done
    sleep 60
  done
  return 0
}

run_arm() {     # $1 arm (A|B), $2 first launch: "resume <step> <tag>" or "fresh". Returns 0 done, 2 stopped (2nd OOM), 1 chain stop.
  local arm=$1 d=$E/E004-$1-s0 attempt=1 T=$T1 launch=$2 tp rc=0 L k
  while true; do
    log "arm $arm attempt $attempt: $launch, MAX_TOKENS_PER_GPU=$T"
    if [ "$launch" = fresh ]; then
      MAX_TOKENS_PER_GPU=$T bash scripts/exp/E004_train.sh $arm > $E/E004-$arm-launch.log 2>&1 &
    else
      set -- $launch
      MAX_TOKENS_PER_GPU=$T bash scripts/exp/E004_train.sh resume $arm $2 $3 > $E/E004-$arm-$3-launch.log 2>&1 &
    fi
    tp=$!
    ( sleep 600; while kill -0 $tp 2>/dev/null; do   # stall watchdog: train.log silent for 30 min -> kill (non-OOM failure)
        if [ -f $d/train.log ] && [ $(( $(date +%s) - $(stat -c %Y $d/train.log) )) -gt 1800 ]; then
          log "arm $arm: train.log silent for 30 min; killing training (stall)"; echo "stall" > $d/STALL_attempt$attempt
          pkill -u $USER -f "default_local_dir=$d/ckpt"; break
        fi; sleep 60; done ) &
    if [ $arm = B ]; then guard_b $tp || { wait $tp; return 1; }; fi
    wait $tp; log "arm $arm attempt $attempt training exit $?"
    while pid_alive $d; do sleep 60; done
    log "arm $arm watcher finished:"; tail -2 $d/ckpt_watch.log
    gpus_free || return 1
    if [ -f $d/hf/step_100/LOAD_OK.json ]; then log "arm $arm done (hf/step_100 confirmed)"; return 0; fi
    if ! grep -q "OutOfMemoryError" $d/train.log; then log "arm $arm failed without CUDA OOM (stall or other); chain stops"; return 1; fi
    if [ $attempt = 2 ]; then log "arm $arm: second CUDA OOM; arm stopped; confirmed checkpoints: $(ls $d/hf | tr '\n' ' ')"; return 2; fi
    L=$(archive $d | tail -1)
    [ "$L" -gt 0 ] && [ -n "$(ls $d/ckpt/global_step_$L/actor/model_world_size_*.pt 2>/dev/null)" ] \
      || { log "arm $arm: no complete checkpoint to resume from; arm stopped"; return 2; }
    k=1; while [ -e $d/base_commit.r$k.txt ]; do k=$((k + 1)); done
    attempt=2; T=$T2; launch="resume $L r$k"
    log "arm $arm: CUDA OOM; resuming once from global_step_$L with MAX_TOKENS_PER_GPU=$T"
  done
}

gpus_free || exit 1
run_arm A "resume 60 $A_TAG"; ra=$?
[ $ra = 1 ] && { log "chain stopped after arm A"; exit 1; }
A=$E/E004-A-s0
if [ $ra = 0 ]; then
  files=$(ls $A/ckpt/global_step_100/actor/{model,optim,extra_state}_world_size_*.pt 2>/dev/null)
  if [ -n "$files" ]; then
    gb=$(du -cBG $files | tail -1 | tr -dc 0-9)
    log "deleting arm A global_step_100 FSDP shards + optimizer state ($(echo $files | wc -w) files, ${gb} GB); hf/step_100: $(tr -d '\n ' < $A/hf/step_100/LOAD_OK.json)"
    rm -f $files
  fi
else
  log "arm A stopped early; its latest full checkpoint is kept"
fi
F=$(free_gb); log "free /data ${F} GB; NEED ${NEED} GB"
[ "$F" -ge "$NEED" ] || { log "free < NEED before arm B; arm B NOT started"; echo "before B: free ${F} < ${NEED}" > $E/STOP_DISK_before_B; exit 0; }
run_arm B fresh; rb=$?
[ $rb = 1 ] && { log "chain stopped in arm B (see above); no evaluation"; exit 1; }
mkdir -p $E/eval
bash scripts/exp/E004_eval_all.sh 2>&1 | tee $E/eval/launch.log
log "E004 all done"
