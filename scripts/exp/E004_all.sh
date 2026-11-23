#!/usr/bin/env bash
# E004 after the pilot: arm A, then arm B (each waits for its checkpoint watcher), then the evaluation queue.
#   bash scripts/exp/E004_all.sh 2>&1 | tee $DATA_ROOT/logs/E004/all.log
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
for arm in A B; do
  echo "[$(date '+%F %T')] start arm $arm"
  bash scripts/exp/E004_train.sh $arm > $DATA_ROOT/logs/E004/E004-$arm-launch.log 2>&1
  echo "[$(date '+%F %T')] arm $arm training exit $?"
  w=$(cat $DATA_ROOT/logs/E004/E004-$arm-s0/ckpt_watch.pid 2>/dev/null)
  while [ -n "$w" ] && kill -0 $w 2>/dev/null; do sleep 60; done
  echo "[$(date '+%F %T')] arm $arm checkpoint watcher finished"; tail -3 $DATA_ROOT/logs/E004/E004-$arm-s0/ckpt_watch.log
  pkill -u $USER -f '^ray::' 2>/dev/null; sleep 20
done
mkdir -p $DATA_ROOT/logs/E004/eval
bash scripts/exp/E004_eval_all.sh 2>&1 | tee $DATA_ROOT/logs/E004/eval/launch.log
echo "[$(date '+%F %T')] E004 all done"
