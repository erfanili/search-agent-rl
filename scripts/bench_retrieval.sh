#!/usr/bin/env bash
# Benchmark the running retrieval server (P0.6). Usage: scripts/bench_retrieval.sh [tag]
# Writes $DATA/logs/p0.6_bench_<tag>.json (numbers) and .txt (full output, incl. the spot check). The exact-search check encodes queries on RETRIEVAL_GPU
# (configs/gpus.env), next to the server, so it doesn't touch training GPUs; on CPU when that is "cpu".
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
TAG=${1:-ivf_nprobe64}

CUDA_VISIBLE_DEVICES=${RETRIEVAL_GPU/#cpu/} .venv/bin/python retrieval/bench_retrieval.py --url http://127.0.0.1:8000 \
    --n 1000 --topk 3 --concurrency 32 --emb-dir $DATA/emb_e5 --exact-n 200 \
    --out "$DATA/logs/p0.6_bench_${TAG}.json" 2>&1 | tee "$DATA/logs/p0.6_bench_${TAG}.txt"
