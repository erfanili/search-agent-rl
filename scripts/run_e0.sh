#!/usr/bin/env bash
# E0 (notes/new_plan.md): depth-value curves V(k). Forces exactly k searches (FORCE_DEPTH) on each model,
# one k per GPU in parallel, models in sequence on each GPU. Needs the retrieval server up.
#   nohup bash scripts/run_e0.sh > $DATA/logs/e0_<stamp>.log 2>&1 &
#   KS="0 1 2 3" MODELS="m1 m2" SPLIT=full bash scripts/run_e0.sh
# GPUs: configs/gpus.env, one per k (needs at least as many GPUs as values of k).
# Results: $DATA/logs/eval_<model tag>_exact<k>_<stamp>/scores.txt; per-GPU logs in $DATA/logs/e0_<stamp>/.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpus.sh

KS=(${KS:-0 1 2 3})
MODELS=(${MODELS:-Qwen/Qwen2.5-3B PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo $DATA/logs/grpo_qwen2.5-3b_full_20261019_183501/ckpt/global_step_60})
(( ${#WORK_GPUS[@]} >= ${#KS[@]} )) || { echo "need ${#KS[@]} GPUs, configs/gpus.env has ${#WORK_GPUS[@]}" >&2; exit 1; }
require_retrieval_server
# merge FSDP checkpoints once, before the parallel jobs (eval_checkpoint.sh merges on first use)
for M in "${MODELS[@]}"; do
    [[ -d $M/actor && ! -f $M/actor/hf_merged/config.json ]] && { echo "merge $M first: bash scripts/eval_checkpoint.sh with DATASETS=bamboogle" >&2; exit 1; }
done

LOGDIR=$DATA/logs/e0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "E0: k=[${KS[*]}] on GPUs [${WORK_GPUS[*]:0:${#KS[@]}}], ${#MODELS[@]} models, split ${SPLIT:-sub300}; logs $LOGDIR"
pids=()
for i in "${!KS[@]}"; do
    K=${KS[$i]} G=${WORK_GPUS[$i]}
    (
        for M in "${MODELS[@]}"; do
            echo "[$(date '+%F %T')] k=$K GPU $G: $M"
            GPU=$G FORCE_DEPTH=$K bash scripts/eval_checkpoint.sh "$M" || echo "[$(date '+%F %T')] FAILED k=$K $M"
        done
        echo "[$(date '+%F %T')] k=$K done"
    ) > "$LOGDIR/k$K.log" 2>&1 &
    pids+=($!)
done
wait "${pids[@]}"
echo "E0 done: grep -H 'searches/q' $DATA/logs/eval_*_exact*/infer.log"
