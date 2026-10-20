#!/usr/bin/env bash
# Evaluate a trained checkpoint with the same harness as the Search-R1 baseline (baselines/searchr1_infer.py
# + tests/score_predictions.py), so its numbers sit in the same table as P1.2. Then list Bamboogle misses.
#   bash scripts/eval_checkpoint.sh <run_dir>/ckpt/global_step_60       # verl FSDP checkpoint: merged to HF first
#   bash scripts/eval_checkpoint.sh /path/to/hf_model                   # any HF model dir or hub id
#   DATASETS=bamboogle bash scripts/eval_checkpoint.sh ...              # one set only (Bamboogle: ~1 min)
# Needs the retrieval server up. One GPU from configs/gpus.env (not the retrieval one); the merge runs on CPU.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
DATA=/data/erfan/search-agent-rl-data
export HF_HOME=$DATA/hf_cache
export VLLM_CACHE_ROOT=$DATA/vllm_cache
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
SRC=${1:?usage: eval_checkpoint.sh <ckpt/global_step_N | hf_model>}
SRC=${SRC%/}
SPLIT=${SPLIT:-sub300}
DATASETS=${DATASETS:-hotpotqa 2wiki musique bamboogle}

# verl FSDP checkpoint -> HF weights (CPU, once; written next to the shards)
if [[ -d $SRC/actor && -f $SRC/actor/fsdp_config.json ]]; then
    MODEL=$SRC/actor/hf_merged
    if [[ ! -f $MODEL/config.json ]]; then
        echo "merging $SRC/actor -> $MODEL"
        CUDA_VISIBLE_DEVICES= .venv-verl/bin/python -m verl.model_merger merge --backend fsdp \
            --local_dir "$SRC/actor" --target_dir "$MODEL"
    fi
    # tokenizer files live in actor/huggingface; make sure the merged dir has them
    [[ -f $MODEL/tokenizer_config.json ]] || cp -n "$SRC"/actor/huggingface/* "$MODEL"/
    TAG=$(basename "$(dirname "$(dirname "$SRC")")")_$(basename "$SRC")      # <run>_global_step_N
else
    MODEL=$SRC
    TAG=$(basename "$SRC")
fi
OUT=$DATA/logs/eval_${TAG}_$(date +%Y%m%d_%H%M%S)

GPU=
for g in "${GPU_LIST[@]}"; do [[ $g != "$RETRIEVAL_GPU" ]] && { GPU=$g; break; }; done
[[ -n $GPU ]] || { echo "no GPU in configs/gpus.env besides RETRIEVAL_GPU=$RETRIEVAL_GPU"; exit 1; }
curl -sf localhost:8000/health >/dev/null || { echo "retrieval server not up; run: bash scripts/serve_retrieval.sh"; exit 1; }

mkdir -p "$OUT"
echo "GPU $GPU, model $MODEL, $SPLIT [$DATASETS], output $OUT"
CUDA_VISIBLE_DEVICES=$GPU taskset -c "$(cpus_for "$GPU")" \
    .venv-verl/bin/python baselines/searchr1_infer.py --model "$MODEL" --split "$SPLIT" --datasets $DATASETS \
    --gpu-mem "${GPU_MEM:-0.5}" --out "$OUT/preds.jsonl" 2>&1 | tee "$OUT/infer.log"
.venv/bin/python tests/score_predictions.py --preds "$OUT/preds.jsonl" --split "$SPLIT" --datasets $DATASETS \
    --out "$OUT/scored" | tee "$OUT/scores.txt"

if [[ " $DATASETS " == *" bamboogle "* ]]; then
    .venv/bin/python scripts/show_misses.py "$OUT/scored/per_example.jsonl" --dataset bamboogle \
        | tee "$OUT/bamboogle_misses.txt"
fi
