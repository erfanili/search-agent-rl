#!/usr/bin/env bash
# Start the retrieval server (P0.6). Usage: scripts/serve_retrieval.sh [index_dir] [gpu]
set -euo pipefail

cd "$(dirname "$0")/.."
INDEX=${1:-/data/erfan/index_e5_IVF4096_SQ8}
GPU=${2:-0}
export HF_HOME=${HF_HOME:-/data/erfan/hf_cache}   # / is 98% full; keep caches on /data
mkdir -p "$INDEX/logs"

CUDA_VISIBLE_DEVICES=$GPU nohup taskset -c 0-23,48-71 \
    .venv/bin/python retrieval_server.py --index-dir "$INDEX" --nprobe "${NPROBE:-64}" \
    --threads 48 --port 8000 > "$INDEX/logs/server.log" 2>&1 &
echo "server starting (pid $!); log: $INDEX/logs/server.log; check: curl localhost:8000/health"