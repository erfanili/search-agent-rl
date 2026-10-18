#!/usr/bin/env bash
# Encode the full corpus with encode_corpus.py, one process per GPU (GPUs 0-5).
# Safe to rerun: finished shards are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
INPUT=/data/erfan/corpus/data/psgs_w100/no_embeddings
OUT=/data/erfan/emb_e5
mkdir -p "$OUT/logs"

for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$r nohup .venv/bin/python encode_corpus.py encode \
        --input "$INPUT" --out "$OUT" --max-length 256 \
        --rank $r --world-size 4 > "$OUT/logs/encode_$r.log" 2>&1 &
done
echo "started 4 ranks; logs in $OUT/logs/"
