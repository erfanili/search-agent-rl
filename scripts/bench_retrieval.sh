#!/usr/bin/env bash
# Benchmark the running retrieval server (P0.6). Usage: scripts/bench_retrieval.sh [tag]
# Writes /data/erfan/search-rl-agent-data/logs/p0.6_bench_<tag>.json (numbers) and .txt (full output, incl. the spot check). The exact-search check encodes queries on RETRIEVAL_GPU
# (configs/gpus.env), next to the server, so it doesn't touch training GPUs.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
TAG=${1:-ivf_nprobe64}
export HF_HOME=${HF_HOME:-/data/erfan/search-rl-agent-data/hf_cache}

CUDA_VISIBLE_DEVICES=$RETRIEVAL_GPU .venv/bin/python retrieval/bench_retrieval.py --url http://127.0.0.1:8000 \
    --n 1000 --topk 3 --concurrency 32 --emb-dir /data/erfan/search-rl-agent-data/emb_e5 --exact-n 200 \
    --out "/data/erfan/search-rl-agent-data/logs/p0.6_bench_${TAG}.json" 2>&1 | tee "/data/erfan/search-rl-agent-data/logs/p0.6_bench_${TAG}.txt"
