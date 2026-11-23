#!/usr/bin/env bash
# E001 full run: 3B models on GPUs 5, 6, 7 and qwen3b-base then qwen7b-instruct on GPU 4.
#   bash scripts/exp/E001_launch.sh            (add --pilot to rerun the pilot instead)
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/data/erfan/search-agent-rl-data/hf_cache VLLM_CACHE_ROOT=/data/erfan/search-agent-rl-data/vllm_cache
OUT=/data/erfan/search-agent-rl-data/logs/E001
[[ ${1:-} == --pilot ]] && OUT=$OUT/pilot2
mkdir -p "$OUT"
run() { local rc=0; CUDA_VISIBLE_DEVICES=$1 .venv-verl/bin/python scripts/exp/E001_run.py --model "$2" ${3:-} > "$OUT/$2.log" 2>&1 || rc=$?; echo "[$(date "+%F %T")] $2 exit $rc"; }
( run 4 qwen3b-base "${1:-}"; run 4 qwen7b-instruct "${1:-}" ) &
run 5 qwen3b-instruct "${1:-}" &
run 6 searchr1-3b "${1:-}" &
run 7 grpo-e1a-s60 "${1:-}" &
wait
echo "[$(date '+%F %T')] all done"
