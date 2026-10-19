#!/usr/bin/env bash
# P0.8 done-check: build the frozen eval sets, run the unit tests, score a dummy predictions file.
# Usage (from anywhere): bash tests/check_eval_harness.sh
# All data stays under /data/erfan: eval sets in /data/erfan/eval, outputs in /data/erfan/logs/p0_8.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
export HF_HOME=${HF_HOME:-/data/erfan/hf_cache}   # / is nearly full; keep caches on /data
PY=.venv/bin/python       # retrieval/eval env; see notes/environments.md
GOLD=/data/erfan/eval
OUT=/data/erfan/logs/p0_8

[[ -f $GOLD/MANIFEST.json ]] || $PY tests/make_eval_sets.py --out "$GOLD" --subset 300
$PY -m pytest tests/ -q
mkdir -p "$OUT"
$PY tests/score_predictions.py --gold-dir "$GOLD" --make-dummy "$OUT/dummy_preds.jsonl" --split sub300
$PY tests/score_predictions.py --gold-dir "$GOLD" --preds "$OUT/dummy_preds.jsonl" --split sub300 --out "$OUT/dummy_scored"
echo "expected on each 300-row subset: EM 0.400, abort 0.100, calls 1.50 (Bamboogle, 125 rows: EM 0.400, abort 0.104, calls 1.50)"
