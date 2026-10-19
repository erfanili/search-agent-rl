#!/usr/bin/env bash
# Encode the full corpus with encode_corpus.py, one process per GPU in configs/gpus.env.
# Safe to rerun, even with a different GPU count: finished shards are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
export HF_HOME=${HF_HOME:-/data/erfan/search-rl-agent-data/hf_cache}   # keep model downloads under /data/erfan/search-rl-agent-data, not ~/.cache
INPUT=/data/erfan/search-rl-agent-data/corpus/data/psgs_w100/no_embeddings
OUT=/data/erfan/search-rl-agent-data/emb_e5
mkdir -p "$OUT/logs"

for r in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$r]} nohup .venv/bin/python retrieval/encode_corpus.py encode \
        --input "$INPUT" --out "$OUT" --max-length 256 \
        --rank "$r" --world-size "$N_GPUS" > "$OUT/logs/encode_$r.log" 2>&1 &
done
echo "started $N_GPUS ranks on GPUs $GPUS; logs in $OUT/logs/"
