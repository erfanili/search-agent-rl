#!/usr/bin/env bash
# Benchmark the running retrieval server (P0.6). Usage: scripts/bench_retrieval.sh [tag]
# Writes logs/p0.6_bench_<tag>.json.
set -euo pipefail

cd "$(dirname "$0")/.."
TAG=${1:-ivf_nprobe64}
export HF_HOME=${HF_HOME:-/data/erfan/hf_cache}

CUDA_VISIBLE_DEVICES=${GPU:-1} .venv/bin/python bench_retrieval.py --url http://127.0.0.1:8000 \
    --n 1000 --topk 3 --concurrency 32 --emb-dir /data/erfan/emb_e5 --exact-n 200 \
    --out "logs/p0.6_bench_${TAG}.json"