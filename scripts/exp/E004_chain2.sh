#!/usr/bin/env bash
# E004 chain, version 2 (Erfan, 2026-11-21 disk decision). Takes over from E004_all_resume.sh while arm A (resumed) trains:
#  1. wait for arm A's training and checkpoint watcher to end;
#  2. require hf/step_100/LOAD_OK.json for arm A, then delete arm A's ckpt/global_step_100/actor/
#     {model,optim,extra_state}_world_size_*.pt (FSDP shards + optimizer state) BEFORE arm B starts (approved candidate 1);
#  3. require answer_check.json "stop": false (answer-span gate);
#  4. disk guard: NEED = arm A's step-100 full-checkpoint size + its merge size + 3 GB margin. Before arm B starts and
#     after each of arm B's confirmed checkpoints (once the watcher freed the earlier one), free space on /data must be
#     >= NEED for the next save + merge; otherwise kill arm B's training (the latest full checkpoint stays for resuming),
#     write E004-B-s0/STOP_DISK and stop the chain without evaluating (Erfan: stop and ask). No other deletion.
#  5. arm B's watcher done -> evaluation queue.
#   bash scripts/exp/E004_chain2.sh > $DATA_ROOT/logs/E004/chain2.log 2>&1
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
E=$DATA_ROOT/logs/E004; A=$E/E004-A-s0; B=$E/E004-B-s0
log() { echo "[$(date '+%F %T')] $*"; }
alive() { pgrep -u $USER -f "default_local_dir=$1/ckpt" >/dev/null; }
pid_alive() { local w; w=$(cat $1/ckpt_watch.pid 2>/dev/null); [ -n "$w" ] && kill -0 $w 2>/dev/null; }
free_gb() { df -BG --output=avail /data | tail -1 | tr -dc 0-9; }

log "waiting for arm A training + watcher"
while alive $A || pid_alive $A; do sleep 60; done
log "arm A done; watcher:"; tail -3 $A/ckpt_watch.log
pkill -u $USER -f '^ray::' 2>/dev/null; sleep 20
[ -f $A/hf/step_100/LOAD_OK.json ] || { log "arm A hf/step_100 not confirmed; nothing deleted; STOP"; exit 1; }
files=$(ls $A/ckpt/global_step_100/actor/{model,optim,extra_state}_world_size_*.pt 2>/dev/null)
[ -n "$files" ] || { log "no step-100 shard files found; STOP"; exit 1; }
CK=$(du -cBG $files | tail -1 | tr -dc 0-9); HFS=$(du -sBG $A/hf/step_100 | cut -f1 | tr -dc 0-9)
NEED=$((CK + HFS + 3))
log "deleting arm A global_step_100 FSDP shards + optimizer state ($(echo $files | wc -w) files, ${CK} GB); hf/step_100 confirmed: $(cat $A/hf/step_100/LOAD_OK.json | tr -d '\n ')"
rm -f $files
log "deleted; free /data $(free_gb) GB; NEED per save+merge ${NEED} GB"
grep -q '"stop": false' $A/answer_check.json 2>/dev/null || { log "answer check stop or missing; arm B NOT started"; exit 0; }
F=$(free_gb); [ "$F" -ge "$NEED" ] || { log "free ${F} GB < NEED ${NEED} GB before arm B; arm B NOT started; STOP_DISK"; echo "before start: free ${F} < ${NEED}" > $E/STOP_DISK_before_B; exit 0; }

log "start arm B"
bash scripts/exp/E004_train.sh B > $E/E004-B-launch.log 2>&1 &
TP=$!
sleep 120
checked=""
while kill -0 $TP 2>/dev/null; do
  for s in 20 40 60 80; do
    [[ " $checked " == *" $s "* ]] && continue
    [ -f $B/hf/step_$s/LOAD_OK.json ] || continue
    sleep 180   # the watcher deletes the earlier checkpoint's shards right after the load check
    checked="$checked $s"
    F=$(free_gb)
    log "arm B step $s confirmed; free ${F} GB; NEED ${NEED} GB"
    if [ "$F" -lt "$NEED" ]; then
      echo "after step $s: free ${F} GB < NEED ${NEED} GB (latest full checkpoint global_step_$s kept)" > $B/STOP_DISK
      pkill -u $USER -f "default_local_dir=$B/ckpt"; sleep 30; pkill -u $USER -f '^ray::'
      log "STOP_DISK: arm B training killed after step $s; chain stopped; nothing deleted"; exit 0
    fi
  done
  sleep 60
done
wait $TP; log "arm B training exit $?"
while pid_alive $B; do sleep 60; done
log "arm B watcher finished"; tail -3 $B/ckpt_watch.log
pkill -u $USER -f '^ray::' 2>/dev/null; sleep 20
mkdir -p $E/eval
bash scripts/exp/E004_eval_all.sh 2>&1 | tee $E/eval/launch.log
log "E004 all done"
