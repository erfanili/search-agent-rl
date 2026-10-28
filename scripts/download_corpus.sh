#!/usr/bin/env bash
# Download the retrieval corpus: facebook/wiki_dpr, Wikipedia Dec 2018 in 100-word passages (D-03).
# Only no_embeddings/ (28 parquet shards, 21,015,300 passages, ~7 GB) is needed, plus the dummy shard
# for end-to-end tests. The nq/ and multiset/ dirs carry DPR's own embeddings (~80 GB each) and are skipped.
#   bash scripts/download_corpus.sh
# Writes $DATA/corpus/data/psgs_w100/{no_embeddings,dummy.no_embeddings}/, the input of scripts/encode_corpus.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/paths.sh
export HF_HOME=${HF_HOME:-$DATA/hf_cache}
HF=${HF:-.venv/bin/hf}
OUT=$DATA/corpus

$HF download facebook/wiki_dpr --repo-type dataset --local-dir "$OUT" \
    --include "data/psgs_w100/no_embeddings/*" --include "data/psgs_w100/dummy.no_embeddings/*" \
    --include "wiki_dpr.py" --include "README.md"

N=$(ls "$OUT"/data/psgs_w100/no_embeddings/*.parquet | wc -l)
echo "$N shards in $OUT/data/psgs_w100/no_embeddings (expected 28)"
.venv/bin/python -c "
import glob, pyarrow.parquet as pq
n = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob('$OUT/data/psgs_w100/no_embeddings/*.parquet'))
print(f'{n:,} passages (expected 21,015,300)')
"
