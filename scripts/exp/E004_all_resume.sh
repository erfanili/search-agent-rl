#!/usr/bin/env bash
# E004 chain after the answer-span request (Erfan, 2026-11-21): arm A resumed from global_step_20 with the logging,
# gated by E004_answer_check.py at the first logged step (21); arm B only if answer_check.json exists with
# "stop": false; then the evaluation queue. Replaces E004_all.sh, which was stopped during arm A step 21.
#   bash scripts/exp/E004_all_resume.sh > $DATA_ROOT/logs/E004/all_resume.log 2>&1
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
E=$DATA_ROOT/logs/E004
wait_watch() {
  local w; w=$(cat $E/E004-$1-s0/ckpt_watch.pid 2>/dev/null)
  while [ -n "$w" ] && kill -0 $w 2>/dev/null; do sleep 60; done
  echo "[$(date '+%F %T')] arm $1 checkpoint watcher finished"; tail -3 $E/E004-$1-s0/ckpt_watch.log
  pkill -u $USER -f '^ray::' 2>/dev/null; sleep 20
}
echo "[$(date '+%F %T')] resume arm A from global_step_20"
bash scripts/exp/E004_train.sh A-resume > $E/E004-A-resume-launch.log 2>&1
echo "[$(date '+%F %T')] arm A training exit $?"
wait_watch A
if [ -e $E/E004-A-s0/STOP ] || ! grep -q '"stop": false' $E/E004-A-s0/answer_check.json 2>/dev/null; then
  echo "[$(date '+%F %T')] answer check stop or missing; arm B NOT started"; cat $E/E004-A-s0/STOP 2>/dev/null; exit 0
fi
echo "[$(date '+%F %T')] start arm B"
bash scripts/exp/E004_train.sh B > $E/E004-B-launch.log 2>&1
echo "[$(date '+%F %T')] arm B training exit $?"
wait_watch B
mkdir -p $E/eval
bash scripts/exp/E004_eval_all.sh 2>&1 | tee $E/eval/launch.log
echo "[$(date '+%F %T')] E004 all done"
