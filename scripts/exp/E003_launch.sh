#!/usr/bin/env bash
# E003 full run: per model, E003_gen.py (vLLM) then E003_hidden.py (transformers). Shared queue, GPUs 4-7.
#   bash scripts/exp/E003_launch.sh 2>&1 | tee $DATA_ROOT/logs/E003/launch.log
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data} HF_HOME=${HF_HOME:-/data/erfan/search-agent-rl-data/hf_cache}
B=$DATA_ROOT/logs/E003
mkdir -p $B
Q=$B/queue.txt
printf "%s\n" qwen7b-instruct qwen3b-base qwen3b-instruct searchr1-3b grpo-e1a-s60 > $Q
worker() {
  local g=$1
  while true; do
    m=$(flock $Q.lock sh -c "head -n1 $Q; sed -i 1d $Q")
    [ -z "$m" ] && break
    mkdir -p $B/$m
    (CUDA_VISIBLE_DEVICES=$g .venv-verl/bin/python scripts/exp/E003_gen.py --model $m && \
     CUDA_VISIBLE_DEVICES=$g .venv-verl/bin/python scripts/exp/E003_hidden.py --model $m) > $B/$m.log 2>&1
    echo "[$(date '+%F %T')] GPU $g $m exit $?"
  done
}
for g in 4 5 6 7; do worker $g & done
wait
echo "[$(date '+%F %T')] all done"
