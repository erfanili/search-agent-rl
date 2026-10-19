#!/usr/bin/env bash
# Encode the full corpus with encode_corpus.py, one process per GPU in configs/gpus.env.
# Safe to rerun, even with a different GPU count: finished shards are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
INPUT=/data/erfan/corpus/data/psgs_w100/no_embeddings
OUT=/data/erfan/emb_e5
mkdir -p "$OUT/logs"

for r in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$r]} nohup .venv/bin/python encode_corpus.py encode \
        --input "$INPUT" --out "$OUT" --max-length 256 \
        --rank "$r" --world-size "$N_GPUS" > "$OUT/logs/encode_$r.log" 2>&1 &
done
echo "started $N_GPUS ranks on GPUs $GPUS; logs in $OUT/logs/"
