#!/usr/bin/env bash
# E006 pilots: 1 step of each arm (O, S, C), no checkpoint, GPUs 4-7, then E006_pilot_summary.py per arm.
#   bash scripts/exp/E006_pilot.sh > $DATA_ROOT/logs/E006/pilot.log 2>&1
# MAX_TOKENS_PER_GPU=4608 (card), PREFIX_CACHING=False (E006_train.sh). PYTORCH_CUDA_ALLOC_CONF left unset (deviation: expandable_segments:True makes vLLM's
# sleep-mode CuMemAllocator assert at rollout-engine load; E004 attempt r3, 2026-11-21, hung 74 min).
# If every pilot has no E006 pilot flag (garbled share <= 0.02, parse >= 0.95) and CHAIN=1, starts E006_chain.sh (card: run the chain without waiting).
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
unset PYTORCH_CUDA_ALLOC_CONF
E=$DATA_ROOT/logs/E006
log() { echo "[$(date '+%F %T')] $*"; }
gpus_free() {
  pkill -u $USER -f '^ray::' 2>/dev/null; .venv-verl/bin/ray stop --force >/dev/null 2>&1
  for i in $(seq 30); do
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>=4 && $2>1000' | wc -l)
    [ "$u" = 0 ] && return 0; sleep 10
  done; log "GPUs 4-7 still in use"; return 1
}
ok=1
for arm in O S C; do
  gpus_free || exit 1
  log "pilot $arm start"
  MAX_TOKENS_PER_GPU=4608 timeout 3600 bash scripts/exp/E006_train.sh pilot $arm > $E/E006-$arm-pilot-launch.log 2>&1
  log "pilot $arm exit $?"
  .venv-verl/bin/python scripts/exp/E006_pilot_summary.py $E/E006-$arm-pilot > $E/E006-$arm-pilot/pilot_summary.md 2>&1
  tail -1 $E/E006-$arm-pilot/pilot_summary.md
  grep -q "^E006 PILOT FLAGS: none" $E/E006-$arm-pilot/pilot_summary.md || ok=0
done
gpus_free
if [ $ok = 1 ] && [ "${CHAIN:-0}" = 1 ]; then
  log "all pilots pass; starting chain"
  T1=4608 T2=3072 nohup bash scripts/exp/E006_chain.sh > $E/chain.log 2>&1 &
else
  log "pilots done; ok=$ok CHAIN=${CHAIN:-0}; chain not started"
fi
