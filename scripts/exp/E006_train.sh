#!/usr/bin/env bash
# E006 training (E005 rerun, prefix caching off): one arm, run FROM the worktree ../sar-alias-reward (uncommitted: alias_match_strict, arm terms C/S/O,
# component logging; E004's per-label stats, max:k, MAX_TURNS, RUN_DIR, VAL_BEFORE_TRAIN). GPUs 4-7 (configs/gpus.env).
#   bash scripts/exp/E006_train.sh pilot O          # 1 step, no checkpoint (E006 pilot) -> $DATA_ROOT/logs/E006/E006-O-pilot
#   bash scripts/exp/E006_train.sh full O           # 100 steps, seed 0        -> $DATA_ROOT/logs/E006/E006-O-s0
#   bash scripts/exp/E006_train.sh resume O 60 r1   # from ckpt/global_step_60; provenance base_commit.r1.txt,
#                                                   # code.r1.patch, git_status.r1.txt (caller archives old logs)
# E006 = E005_train.sh with: logs/E006, cost 0.20 for C, PREFIX_CACHING=False (agent/vllm_server.py
# AsyncvLLMServerNoPrefixCache), pilot = 1 step. Training also writes rollout_samples.jsonl (50 per step) and
# garble/ (per-server garble events and warnings), and capped answer statistics (worktree, 2026-11-22).
# Arms (card): reward = alias_match_strict + arm term; C: -0.20 x executed searches (E005 change, kept in E006); S: 0.3 (d(1-a) + (1-d)a),
# a = extra_info.sc_agreement (E005_sc.py); O: same with a = 1 knowable / 0 unknowable.
# Data: $DATA_ROOT/verl_data/e005/mix50sc = E004 MIX-50 rows (same order) + sc_agreement, for all three arms.
# Recipe = E004: GRPO 64 x 5, mini-batch 32, lr 1e-6, warmup ratio 0.06 (6 steps), KL 0.001 low_var_kl (launcher
# default), temperature 1.0, max:3 with max_turns 3, 2,048-token episodes, prompts <= 1,024, seed 0, no validation,
# checkpoints every 20 steps. MAX_TOKENS_PER_GPU and PYTORCH_CUDA_ALLOC_CONF pass through from the environment.
# Full arms also start E004_ckpt_watch.py (merge -> load check -> delete earlier shards + optimizer state).
set -euo pipefail
MODE=${1:?usage: E006_train.sh pilot|full|resume <C|S|O> [step tag]}
ARM=${2:?arm C|S|O}
[[ $ARM =~ ^[CSO]$ ]] || { echo "arm: C|S|O" >&2; exit 1; }
COST=0.20   # E006 card: cost = 0.20 (unused by O, S)
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WT=$(cd "$REPO/../sar-alias-reward" && pwd)
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf_cache}
case $MODE in
  pilot)  RUN=E006-$ARM-pilot; STEPS=1;   SAVE=-1 ;;
  full)   RUN=E006-$ARM-s0;    STEPS=100; SAVE=20 ;;
  resume) RUN=E006-$ARM-s0;    STEPS=100; SAVE=20; RSTEP=${3:?step}; RTAG=${4:?tag} ;;
  *) echo "mode: pilot|full|resume" >&2; exit 1 ;;
esac
OUT=$DATA_ROOT/logs/E006/$RUN
WATCH_STEPS="20 40 60 80 100"
prov() {  # $1 suffix ("" or ".tag")
  git -C "$WT" rev-parse HEAD > "$OUT/base_commit$1.txt"
  echo "worktree: $WT (branch $(git -C "$WT" branch --show-current))" >> "$OUT/base_commit$1.txt"
  echo "arm: $ARM  cost: $COST  PREFIX_CACHING=False  resume_from: ${RESUME_FROM:-none}  MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144 (default)}  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset}" >> "$OUT/base_commit$1.txt"
  git -C "$WT" diff > "$OUT/code$1.patch"
  git -C "$WT" status > "$OUT/git_status$1.txt"
}
if [[ $MODE == resume ]]; then
  [[ -f $OUT/ckpt/global_step_$RSTEP/data.pt && -n $(ls $OUT/ckpt/global_step_$RSTEP/actor/model_world_size_*.pt 2>/dev/null) ]] \
    || { echo "no complete checkpoint global_step_$RSTEP in $OUT" >&2; exit 1; }
  export RESUME_FROM=$OUT/ckpt/global_step_$RSTEP
  WATCH_STEPS=$(for s in 20 40 60 80 100; do (( s > RSTEP )) && echo -n "$s "; done)
  prov ".$RTAG"
else
  [[ -e $OUT ]] && { echo "$OUT exists" >&2; exit 1; }
  mkdir -p "$OUT"
  prov ""
fi
if [[ $MODE != pilot ]]; then
  nohup "$REPO/.venv-verl/bin/python" "$REPO/scripts/exp/E004_ckpt_watch.py" --run "$OUT" --steps $WATCH_STEPS \
    > "$OUT/ckpt_watch.log" 2>&1 &
  echo $! > "$OUT/ckpt_watch.pid"
fi
cd "$WT"
MODEL=Qwen/Qwen2.5-3B-Instruct MODEL_NAME=qwen2.5-3b-instruct DATA_DIR=$DATA_ROOT/verl_data/e005/mix50sc \
TRAIN_BATCH=64 MINI_BATCH=32 ROLLOUT_N=5 LR=1e-6 WARMUP_RATIO=0.06 TOTAL_STEPS=$STEPS \
MAX_PROMPT=1024 MAX_RESPONSE=2048 REWARD=alias_match_strict REWARD_ARM=$ARM ARM_LAM=0.3 ARM_COST=$COST \
DEPTH_POLICY=max:3 MAX_TURNS=3 SEED=0 PREFIX_CACHING=False \
TEST_FREQ=-1 VAL_BEFORE_TRAIN=False SAVE_FREQ=$SAVE CKPT_KEEP=null RUN_DIR=$OUT GROUP=E006 \
  bash scripts/train_grpo_3b.sh full
