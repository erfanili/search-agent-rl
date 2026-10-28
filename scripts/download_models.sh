#!/usr/bin/env bash
# Download models and datasets from the HF Hub into $HF_HOME (default $DATA/hf_cache), at the revisions
# our results were produced with. The scripts find them there by repo id.
#   bash scripts/download_models.sh            # everything (~20 GB)
#   bash scripts/download_models.sh core       # only what the main pipeline needs: e5, Qwen2.5-3B, train/eval data
# Eval sets are frozen separately from FlashRAG: .venv/bin/python tests/make_eval_sets.py --subset 300
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/paths.sh
export HF_HOME=${HF_HOME:-$DATA/hf_cache}
HF=${HF:-.venv/bin/hf}
WHAT=${1:-all}
[[ $WHAT == all || $WHAT == core ]] || { echo "usage: bash scripts/download_models.sh [all|core]" >&2; exit 1; }

# kind repo revision  (what uses it)
CORE=(
    "model   intfloat/e5-base-v2                                        f52bf8ec8c7124536f0efb74aca902b2995e5bcd"  # retriever encoder
    "model   Qwen/Qwen2.5-3B                                            3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"  # policy, train_grpo_3b.sh
    "dataset PeterJinGo/nq_hotpotqa_train                               b7d80abfee334a7a91cb377544f09180d58b34f6"  # make_train_data.py
    "dataset RUC-NLPIR/FlashRAG_datasets                                bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"  # make_eval_sets.py
)
EXTRA=(
    "model   PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo   d4d2f71e39ff3e0ed4def1e8f860b87d9f6bfac7"  # baseline, run_baseline_searchr1.sh
    "model   Qwen/Qwen2.5-0.5B-Instruct                                 7ae557604adf67be50417f59c2c2f167def9a775"  # smoke test, rollout check, tests
    "dataset hotpotqa/hotpot_qa                                         1908d6afbbead072334abe2965f91bd2709910ab"  # bench_retrieval.py queries
    "dataset openai/gsm8k                                               740312add88f781978c0658806c59bc2815b9866"  # P0.7 smoke test
)
ITEMS=("${CORE[@]}")
[[ $WHAT == all ]] && ITEMS+=("${EXTRA[@]}")

for item in "${ITEMS[@]}"; do
    read -r kind repo rev <<<"$item"
    echo "== $kind $repo @ ${rev:0:8}"
    if [[ $repo == RUC-NLPIR/FlashRAG_datasets ]]; then
        # ~40 QA sets in this repo; take only the four we evaluate on
        $HF download "$repo" --repo-type dataset --revision "$rev" \
            --include "hotpotqa/*" --include "2wikimultihopqa/*" --include "musique/*" \
            --include "bamboogle/*" --include "README.md" >/dev/null
    elif [[ $repo == hotpotqa/hotpot_qa ]]; then
        $HF download "$repo" --repo-type dataset --revision "$rev" --include "distractor/*" --include "README.md" >/dev/null
    else
        $HF download "$repo" --repo-type "$kind" --revision "$rev" >/dev/null
    fi
done
echo "done: $HF_HOME/hub"
