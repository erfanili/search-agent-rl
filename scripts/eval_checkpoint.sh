#!/usr/bin/env bash
# Evaluate a trained checkpoint with the same harness as the Search-R1 baseline (baselines/searchr1_infer.py
# + tests/score_predictions.py), so its numbers sit in the same table as P1.2.
#   bash scripts/eval_checkpoint.sh <run_dir>/ckpt/global_step_60       # verl FSDP checkpoint: merged to HF first
#   bash scripts/eval_checkpoint.sh /path/to/hf_model                   # any HF model dir or hub id
#   DATASETS=bamboogle bash scripts/eval_checkpoint.sh ...              # one set only (Bamboogle: ~1 min)
#   FORCE_DEPTH=2 bash scripts/eval_checkpoint.sh ...                   # exactly 2 searches per question (E0)
#   DEPTH_POLICY=min:1 bash scripts/eval_checkpoint.sh ...              # free | exact:k | min:k (FORCE_DEPTH=k = exact:k)
# Needs the retrieval server up. One GPU from configs/gpus.env (not the retrieval one); the merge runs on CPU.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
SRC=${1:?usage: eval_checkpoint.sh <ckpt/global_step_N | hf_model>}
SRC=${SRC%/}
SPLIT=${SPLIT:-sub300}
DATASETS=${DATASETS:-hotpotqa 2wiki musique bamboogle}
if [[ -n ${FORCE_DEPTH:-} ]]; then
    [[ -z ${DEPTH_POLICY:-} || $DEPTH_POLICY == "exact:$FORCE_DEPTH" ]] || { echo "set FORCE_DEPTH or DEPTH_POLICY, not both" >&2; exit 1; }
    DEPTH_POLICY=exact:$FORCE_DEPTH
fi
DEPTH_POLICY=${DEPTH_POLICY:-free}

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
[[ $DEPTH_POLICY == free ]] || TAG=${TAG}_${DEPTH_POLICY/:/}                     # e.g. ..._exact2
OUT=$DATA/logs/eval_${TAG}_$(date +%Y%m%d_%H%M%S)

GPU=${GPU:-${WORK_GPUS[0]:-}}                                            # GPU=5 to pick one
[[ -n $GPU ]] || { echo "no GPU in configs/gpus.env besides RETRIEVAL_GPU=$RETRIEVAL_GPU" >&2; exit 1; }
require_retrieval_server

mkdir -p "$OUT"
echo "GPU $GPU, model $MODEL, $SPLIT [$DATASETS], depth $DEPTH_POLICY, output $OUT"
CUDA_VISIBLE_DEVICES=$GPU taskset -c "$(cpus_for "$GPU")" \
    .venv-verl/bin/python baselines/searchr1_infer.py --model "$MODEL" --split "$SPLIT" --datasets $DATASETS \
    --depth-policy "$DEPTH_POLICY" --gpu-mem "${GPU_MEM:-0.5}" --out "$OUT/preds.jsonl" 2>&1 | tee "$OUT/infer.log"
.venv/bin/python tests/score_predictions.py --preds "$OUT/preds.jsonl" --split "$SPLIT" --datasets $DATASETS \
    --out "$OUT/scored" | tee "$OUT/scores.txt"
if wandb_ready; then .venv-verl/bin/python scripts/wandb_log.py eval "$OUT" ${GROUP:+--group "$GROUP"} || echo "wandb logging failed (results are in $OUT)" >&2; fi

