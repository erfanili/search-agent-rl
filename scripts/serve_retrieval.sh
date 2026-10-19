#!/usr/bin/env bash
# Start the retrieval server (P0.6) on RETRIEVAL_GPU from configs/gpus.env.
# Usage: scripts/serve_retrieval.sh [index_dir]
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
INDEX=${1:-/data/erfan/search-agent-rl-data/index_e5_IVF4096_SQ8}
export HF_HOME=${HF_HOME:-/data/erfan/search-agent-rl-data/hf_cache}   # / is ~99% full; keep caches on /data
mkdir -p "$INDEX/logs"

CUDA_VISIBLE_DEVICES=$RETRIEVAL_GPU nohup taskset -c "$(cpus_for "$RETRIEVAL_GPU")" \
    .venv/bin/python retrieval/retrieval_server.py --index-dir "$INDEX" --nprobe "${NPROBE:-64}" \
    --threads "$(ncpus_for "$RETRIEVAL_GPU")" --port 8000 > "$INDEX/logs/server.log" 2>&1 &
echo "server starting on GPU $RETRIEVAL_GPU (pid $!); log: $INDEX/logs/server.log; check: curl localhost:8000/health"
