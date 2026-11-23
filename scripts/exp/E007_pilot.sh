#!/usr/bin/env bash
# E007 pilots (card): 1 step of E007-N and E007-Sshuf only, no checkpoint, GPUs 4-7, then E006_pilot_summary.py per run
# (garbled share <= 0.02 and parse >= 0.95 -> "E006 PILOT FLAGS: none"). If both pass and CHAIN=1, starts E007_chain.sh.
#   CHAIN=1 nohup bash scripts/exp/E007_pilot.sh > $DATA_ROOT/logs/E007/pilot.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
unset PYTORCH_CUDA_ALLOC_CONF
E=$DATA_ROOT/logs/E007
log() { echo "[$(date '+%F %T')] $*"; }
gpus_free() {
  pkill -u $USER -f '^ray::' 2>/dev/null; .venv-verl/bin/ray stop --force >/dev/null 2>&1
  for i in $(seq 30); do
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>=4 && $2>1000' | wc -l)
    [ "$u" = 0 ] && return 0; sleep 10
  done; log "GPUs 4-7 still in use"; return 1
}
ok=1
for id in E007-N-s0 E007-Sshuf-s0; do
  p=${id%-s*}-pilot
  gpus_free || exit 1
  log "pilot $p start"
  MAX_TOKENS_PER_GPU=4608 timeout 3600 bash scripts/exp/E007_train.sh pilot $id > $E/$p-launch.log 2>&1
  log "pilot $p exit $?"
  .venv-verl/bin/python scripts/exp/E006_pilot_summary.py $E/$p > $E/$p/pilot_summary.md 2>&1
  tail -1 $E/$p/pilot_summary.md
  grep -q "^E006 PILOT FLAGS: none" $E/$p/pilot_summary.md || ok=0
done
gpus_free
if [ $ok = 1 ] && [ "${CHAIN:-0}" = 1 ]; then
  log "all pilots pass; starting chain"
  T1=4608 T2=3072 nohup bash scripts/exp/E007_chain.sh > $E/chain.log 2>&1 &
else
  log "pilots done; ok=$ok CHAIN=${CHAIN:-0}; chain not started"
fi
