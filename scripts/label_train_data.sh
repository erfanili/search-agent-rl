#!/usr/bin/env bash
# Label the training questions for E2 (agent/label_train_data.py): closed_book_em (one GPU) and d1_hit (retrieval).
#   bash scripts/label_train_data.sh                      # both stages, all 169,615 training questions
#   STAGES=d1 bash scripts/label_train_data.sh            # one stage: closed_book | d1
#   LIMIT=2000 bash scripts/label_train_data.sh           # timing run on the first 2,000 questions
#   GPU=4 bash scripts/label_train_data.sh                # closed_book on another GPU (default: first in configs/gpus.env)
# closed_book: baselines/searchr1_infer.py --depth-policy exact:0 with the base model (the policy at step 0),
#   i.e. V(0) of E0 on the training questions. d1: top-3 retrieval for the question itself; resumable.
# Output: labels.parquet next to train.parquet; then agent/make_train_data.py --labels ... --filter ...
# Needs the retrieval server up (both stages: infer checks it at start).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpus.sh
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

DATA_DIR=${DATA_DIR:-$DATA/verl_data/nq_hotpotqa}
TRAIN=$DATA_DIR/train.parquet
MODEL=${MODEL:-Qwen/Qwen2.5-3B}
STAGES=${STAGES:-closed_book d1}
LIMIT=${LIMIT:-0}
[[ -f $TRAIN ]] || { echo "no $TRAIN: .venv-verl/bin/python agent/make_train_data.py --out $DATA_DIR" >&2; exit 1; }
require_retrieval_server
LIMIT_ARGS=(); (( LIMIT )) && LIMIT_ARGS=(--limit "$LIMIT")

for S in $STAGES; do
    case $S in
    closed_book)
        MODEL_PATH=$(hf_snapshot models "$MODEL"); MODEL_PATH=${MODEL_PATH:-$MODEL}
        OUT=$DATA/logs/label_closed_book_$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')_$(date +%Y%m%d_%H%M%S)
        GPU=${GPU:-${WORK_GPUS[0]:?no GPU in configs/gpus.env besides the retrieval one}}
        mkdir -p "$OUT"
        echo "closed_book: GPU $GPU, model $MODEL_PATH, output $OUT"
        CUDA_VISIBLE_DEVICES=$GPU taskset -c "$(cpus_for "$GPU")" \
            .venv-verl/bin/python baselines/searchr1_infer.py --model "$MODEL_PATH" --questions-parquet "$TRAIN" \
            --depth-policy exact:0 --gpu-mem "${GPU_MEM:-0.85}" "${LIMIT_ARGS[@]}" --out "$OUT/preds.jsonl" 2>&1 | tee "$OUT/infer.log"
        .venv/bin/python agent/label_train_data.py --train "$TRAIN" --closed-book-preds "$OUT/preds.jsonl"
        ;;
    d1)
        .venv/bin/python agent/label_train_data.py --train "$TRAIN" --d1 "${LIMIT_ARGS[@]}"
        ;;
    *) echo "unknown stage $S (closed_book | d1)" >&2; exit 1 ;;
    esac
done
