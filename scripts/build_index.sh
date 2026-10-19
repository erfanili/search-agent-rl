#!/usr/bin/env bash
# Build the FAISS index + passage store from the E5 shards (P0.6).
# Run after: encode_corpus.py verify passes. Pass FACTORY=Flat to build the exact index instead.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
export HF_HOME=${HF_HOME:-/data/erfan/search-rl-agent-data/hf_cache}   # keep model downloads under /data/erfan/search-rl-agent-data, not ~/.cache
INPUT=/data/erfan/search-rl-agent-data/corpus/data/psgs_w100/no_embeddings
EMB=/data/erfan/search-rl-agent-data/emb_e5
FACTORY=${FACTORY:-IVF4096,SQ8}
OUT=/data/erfan/search-rl-agent-data/index_e5_${FACTORY//[^A-Za-z0-9]/_}

# k-means on every GPU in configs/gpus.env; CPU threads pinned to those GPUs' NUMA node(s)
CPUS=$(cpus_for "$GPUS")yes
echo "GPUs $GPUS, CPUs $CPUS"
CUDA_VISIBLE_DEVICES=$GPUS taskset -c "$CPUS" \
    .venv/bin/python retrieval/build_index.py --emb-dir "$EMB" --corpus "$INPUT" --out "$OUT" \
    --factory "$FACTORY" --train-size 1000000 --gpu-train --threads "$(ncpus_for "$GPUS")"
