#!/usr/bin/env bash
# E005 chain (card): arms O, S, C (100 steps each, GPUs 4-7), then evaluation. Started by E005_pilot.sh when all
# pilots pass.   T1=4608 T2=3072 bash scripts/exp/E005_chain.sh > $DATA_ROOT/logs/E005/chain.log 2>&1
# Per arm (E004_chain3.sh logic, generalized):
#  - training: MAX_TOKENS_PER_GPU = T1; PYTORCH_CUDA_ALLOC_CONF unset (deviation: expandable_segments breaks vLLM sleep
#    mode, E004 r3). Checkpoint watcher E004_ckpt_watch.py (started by E005_train.sh).
#  - stop rule: E005_stop_watch.py from step 10 (answer words > 2x own steps 1-3 mean per label, unclosed > 0.05);
#    a stop kills training -> arm stopped, its confirmed checkpoints are evaluated, chain continues.
#  - CUDA OOM: archive the segment's logs as *.part<k>.*, trim reward_steps.jsonl to the latest full checkpoint, resume
#    once with MAX_TOKENS_PER_GPU = T2; a second OOM -> arm stopped, chain continues.
#  - disk guard: after each confirmed checkpoint, free /data < NEED (full checkpoint + merge + 3 GB) -> STOP_DISK, training
#    killed, arm stopped, chain continues.
#  - another failure (not OOM / stop / disk), including a stall (train.log silent 30 min): the whole chain stops.
#  - arm finished with hf/step_100 confirmed -> its global_step_100 shards + optimizer state deleted (card). An arm
#    stopped early keeps its latest full checkpoint.
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
unset PYTORCH_CUDA_ALLOC_CONF
: ${T1:?set T1} ${T2:?set T2}
E=$DATA_ROOT/logs/E005
log() { echo "[$(date '+%F %T')] $*"; }
pid_alive() { local w; w=$(cat $1/ckpt_watch.pid 2>/dev/null); [ -n "$w" ] && kill -0 $w 2>/dev/null; }
free_gb() { df -BG --output=avail /data | tail -1 | tr -dc 0-9; }
NEED=$((36 + 7 + 3))   # E004 full checkpoint 36 GB (model + optimizer + extra state) + merge 6.2 GB + 3 GB
log "E005 chain: T1=$T1 T2=$T2 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset} NEED=${NEED} GB free=$(free_gb) GB"

gpus_free() {
  pkill -u $USER -f '^ray::' 2>/dev/null; .venv-verl/bin/ray stop --force >/dev/null 2>&1
  for i in $(seq 30); do
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>=4 && $2>1000' | wc -l)
    [ "$u" = 0 ] && return 0; sleep 10
  done
  log "GPUs 4-7 still in use after 5 min"; return 1
}

archive() {     # $1 run dir: this segment's logs -> *.part<k>.*, reward_steps trimmed to the latest full checkpoint
  local d=$1 k=1 L
  while [ -e $d/train.part$k.log ]; do k=$((k + 1)); done
  L=$(cat $d/ckpt/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
  cp -p $d/reward_steps.jsonl $d/reward_steps.part$k.jsonl
  python3 -c "
import json,sys
L=[l for l in open(sys.argv[1]) if not (json.loads(l)['split']=='train' and json.loads(l)['step']>int(sys.argv[2]))]
open(sys.argv[1],'w').writelines(L)" $d/reward_steps.jsonl $L
  mv $d/train.log $d/train.part$k.log; mv $d/config.txt $d/config.part$k.txt; mv $d/git.diff $d/git.part$k.diff 2>/dev/null
  mv $d/hydra $d/hydra.part$k 2>/dev/null; mv $d/rollouts $d/rollouts.part$k; mv $d/ckpt_watch.log $d/ckpt_watch.part$k.log
  mv $d/stop_watch.jsonl $d/stop_watch.part$k.jsonl 2>/dev/null; mv $d/stop_watch.log $d/stop_watch.part$k.log 2>/dev/null
  log "archived $(basename $d) segment as part$k; reward_steps trimmed to step $L" >&2
  echo $L
}

guard() {       # $1 run dir, $2 training pid: disk guard after each confirmed checkpoint
  local d=$1 checked="" s F
  sleep 120
  while kill -0 $2 2>/dev/null; do
    for s in 20 40 60 80; do
      [[ " $checked " == *" $s "* ]] && continue
      [ -f $d/hf/step_$s/LOAD_OK.json ] || continue
      sleep 180; checked="$checked $s"; F=$(free_gb)
      log "$(basename $d) step $s confirmed; free ${F} GB; NEED ${NEED} GB"
      if [ "$F" -lt "$NEED" ]; then
        echo "after step $s: free ${F} GB < NEED ${NEED} GB (latest full checkpoint kept)" > $d/STOP_DISK
        pkill -u $USER -f "default_local_dir=$d/ckpt"; sleep 30; pkill -u $USER -f '^ray::'
        log "STOP_DISK: $(basename $d) training killed after step $s"; return 1
      fi
    done
    sleep 60
  done
  return 0
}

run_arm() {     # $1 arm. Returns 0 done, 2 arm stopped (stop rule / 2nd OOM / disk), 1 chain stop.
  local arm=$1 d=$E/E005-$1-s0 attempt=1 T=$T1 launch="full" tp sw L k
  while true; do
    log "arm $arm attempt $attempt: $launch, MAX_TOKENS_PER_GPU=$T"
    if [ "$launch" = full ]; then
      MAX_TOKENS_PER_GPU=$T bash scripts/exp/E005_train.sh full $arm > $E/E005-$arm-launch.log 2>&1 &
    else
      set -- $launch
      MAX_TOKENS_PER_GPU=$T bash scripts/exp/E005_train.sh resume $arm $2 $3 > $E/E005-$arm-$3-launch.log 2>&1 &
    fi
    tp=$!
    sleep 20
    nohup .venv-verl/bin/python scripts/exp/E005_stop_watch.py --run $d > $d/stop_watch.log 2>&1 &
    sw=$!
    ( sleep 600; while kill -0 $tp 2>/dev/null; do   # stall watchdog
        if [ -f $d/train.log ] && [ $(( $(date +%s) - $(stat -c %Y $d/train.log) )) -gt 1800 ]; then
          log "arm $arm: train.log silent for 30 min; killing training (stall)"; echo "stall" > $d/STALL_attempt$attempt
          pkill -u $USER -f "default_local_dir=$d/ckpt"; break
        fi; sleep 60; done ) &
    guard $d $tp; gr=$?
    wait $tp; log "arm $arm attempt $attempt training exit $?"
    sleep 90; kill $sw 2>/dev/null
    while pid_alive $d; do sleep 60; done
    log "arm $arm watcher finished:"; tail -2 $d/ckpt_watch.log
    gpus_free || return 1
    if [ -f $d/hf/step_100/LOAD_OK.json ]; then log "arm $arm done (hf/step_100 confirmed)"; return 0; fi
    if [ -f $d/STOP ]; then log "arm $arm: stop rule: $(tr '\n' ' ' < $d/STOP); arm stopped; confirmed: $(ls $d/hf 2>/dev/null | tr '\n' ' ')"; return 2; fi
    if [ $gr = 1 ] || [ -f $d/STOP_DISK ]; then log "arm $arm: disk guard stop; confirmed: $(ls $d/hf 2>/dev/null | tr '\n' ' ')"; return 2; fi
    if ! grep -q "OutOfMemoryError" $d/train.log; then log "arm $arm failed without CUDA OOM (stall or other); chain stops"; return 1; fi
    if [ $attempt = 2 ]; then log "arm $arm: second CUDA OOM; arm stopped; confirmed: $(ls $d/hf 2>/dev/null | tr '\n' ' ')"; return 2; fi
    L=$(archive $d | tail -1)
    [ "$L" -gt 0 ] && [ -n "$(ls $d/ckpt/global_step_$L/actor/model_world_size_*.pt 2>/dev/null)" ] \
      || { log "arm $arm: no complete checkpoint to resume from; arm stopped"; return 2; }
    k=1; while [ -e $d/base_commit.r$k.txt ]; do k=$((k + 1)); done
    attempt=2; T=$T2; launch="resume $L r$k"
    log "arm $arm: CUDA OOM; resuming once from global_step_$L with MAX_TOKENS_PER_GPU=$T"
  done
}

gpus_free || exit 1
for arm in O S C; do
  F=$(free_gb)
  if [ "$F" -lt "$NEED" ]; then log "free ${F} GB < NEED ${NEED} GB before arm $arm; arm $arm NOT started"; echo "before $arm: free ${F}" > $E/STOP_DISK_before_$arm; continue; fi
  run_arm $arm; rc=$?
  [ $rc = 1 ] && { log "chain stopped in arm $arm; no evaluation"; exit 1; }
  d=$E/E005-$arm-s0
  if [ $rc = 0 ]; then
    files=$(ls $d/ckpt/global_step_100/actor/{model,optim,extra_state}_world_size_*.pt 2>/dev/null)
    if [ -n "$files" ]; then
      gb=$(du -cBG $files | tail -1 | tr -dc 0-9)
      log "deleting arm $arm global_step_100 shards + optimizer state ($(echo $files | wc -w) files, ${gb} GB); hf/step_100: $(tr -d '\n ' < $d/hf/step_100/LOAD_OK.json)"
      rm -f $files
    fi
  else
    log "arm $arm stopped early; its latest full checkpoint is kept"
  fi
  log "free /data $(free_gb) GB"
done
mkdir -p $E/eval
bash scripts/exp/E005_eval_all.sh 2>&1 | tee $E/eval/launch.log
log "E005 all done"
