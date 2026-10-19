#!/usr/bin/env bash
# Debug baseline: Search-R1 3B (EM, GRPO) on the frozen sub300 sets of HotpotQA / 2Wiki / MuSiQue,
# scored by tests/score_predictions.py. Tests the eval harness end to end against a known checkpoint.
#   bash scripts/run_baseline_searchr1.sh            # all 900 questions
#   LIMIT=20 bash scripts/run_baseline_searchr1.sh   # first 20 per dataset, quick check
# Needs the retrieval server up (bash scripts/serve_retrieval.sh). Runs on one GPU from configs/gpus.env,
# not the retrieval one; sharing a GPU with the index build is fine (vLLM takes GPU_MEM of the card).
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
export HF_HOME=/data/erfan/search-agent-rl-data/hf_cache
export VLLM_CACHE_ROOT=/data/erfan/search-agent-rl-data/vllm_cache   # vLLM compile cache, default ~/.cache/vllm
MODEL=${MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo}
SPLIT=${SPLIT:-sub300}
DATASETS=${DATASETS:-hotpotqa 2wiki musique}
OUT=/data/erfan/search-agent-rl-data/logs/baseline_searchr1_3b_$(date +%Y%m%d_%H%M%S)

GPU=
for g in "${GPU_LIST[@]}"; do [[ $g != "$RETRIEVAL_GPU" ]] && { GPU=$g; break; }; done
[[ -n $GPU ]] || { echo "no GPU in configs/gpus.env besides RETRIEVAL_GPU=$RETRIEVAL_GPU"; exit 1; }
curl -sf localhost:8000/health >/dev/null || { echo "retrieval server not up; run: bash scripts/serve_retrieval.sh"; exit 1; }

mkdir -p "$OUT"
echo "GPU $GPU, model $MODEL, $SPLIT [$DATASETS], output $OUT"
CUDA_VISIBLE_DEVICES=$GPU taskset -c "$(cpus_for "$GPU")" \
    .venv-verl/bin/python baselines/searchr1_infer.py --model "$MODEL" --split "$SPLIT" --datasets $DATASETS \
    --limit "${LIMIT:-0}" --gpu-mem "${GPU_MEM:-0.5}" --out "$OUT/preds.jsonl" 2>&1 | tee "$OUT/infer.log"

ALLOW=()
if [[ ${LIMIT:-0} != 0 ]]; then ALLOW=(--allow-missing); echo "LIMIT set: unanswered questions count as aborts, so scores are a pipeline check only"; fi
.venv/bin/python tests/score_predictions.py --preds "$OUT/preds.jsonl" --split "$SPLIT" --datasets $DATASETS \
    "${ALLOW[@]}" --out "$OUT/scored" | tee "$OUT/scores.txt"
