#!/usr/bin/env bash
# E004 training: one arm, run FROM the worktree ../sar-alias-reward (uncommitted alias_match reward, per-label
# step stats, max:k / MAX_TURNS / RUN_DIR / VAL_BEFORE_TRAIN in train_grpo_3b.sh). GPUs 4-7 (configs/gpus.env).
#   bash scripts/exp/E004_train.sh pilot     # arm A, 3 steps, no checkpoint  -> $DATA_ROOT/logs/E004/E004-A-pilot
#   bash scripts/exp/E004_train.sh A         # MIX-50, seed 0, 100 steps      -> $DATA_ROOT/logs/E004/E004-A-s0
#   bash scripts/exp/E004_train.sh A-resume  # arm A from ckpt/global_step_20 with the answer-span logging (Erfan,
#                                            # 2026-11-21); part-1 logs kept as *.part1.*; also starts E004_answer_check
#   bash scripts/exp/E004_train.sh B         # MIX-20, seed 0, 100 steps      -> $DATA_ROOT/logs/E004/E004-B-s0
#   bash scripts/exp/E004_train.sh resume A 60 r3   # resume arm A/B from its ckpt/global_step_<step>; provenance files
#                                            # base_commit.<tag>.txt, code.<tag>.patch, git_status.<tag>.txt; the
#                                            # caller archives the previous segment's logs first (E004_chain3.sh)
# Memory settings pass through from the environment: MAX_TOKENS_PER_GPU (actor ppo_max_token_len_per_gpu; log-prob
# passes use 4x; train_grpo_3b.sh default 6144) and PYTORCH_CUDA_ALLOC_CONF.
# Full arms also start scripts/exp/E004_ckpt_watch.py (merge each checkpoint to HF, confirm it loads, then delete the
# FSDP shards + optimizer state of earlier checkpoints; the latest full checkpoint is kept).
# Recipe (card; L001 / L006): GRPO 64 prompts x 5, mini-batch 32, lr 1e-6, warmup 6 steps (ratio 0.06 of 100),
# KL loss 0.001 low_var_kl, temperature 1.0, max:3 with max_turns 3, 2,048-token episodes, prompts <= 1,024 tokens,
# reward alias_match, no validation, checkpoints every 20 steps.
set -euo pipefail
ARM=${1:?usage: E004_train.sh pilot|A|B|A-resume|resume <A|B> <step> <tag>}
if [[ $ARM == resume ]]; then RARM=${2:?arm}; RSTEP=${3:?step}; RTAG=${4:?tag}; ARM=$RARM-resume-generic; fi
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WT=$(cd "$REPO/../sar-alias-reward" && pwd)
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf_cache}
case $ARM in
  pilot) RUN=E004-A-pilot; MIX=mix50; STEPS=3;   SAVE=-1 ;;
  A)     RUN=E004-A-s0;    MIX=mix50; STEPS=100; SAVE=20 ;;
  A-resume) RUN=E004-A-s0; MIX=mix50; STEPS=100; SAVE=20 ;;
  B)     RUN=E004-B-s0;    MIX=mix20; STEPS=100; SAVE=20 ;;
  A-resume-generic) RUN=E004-A-s0; MIX=mix50; STEPS=100; SAVE=20 ;;
  B-resume-generic) RUN=E004-B-s0; MIX=mix20; STEPS=100; SAVE=20 ;;
  *) echo "arm: pilot|A|B" >&2; exit 1 ;;
esac
OUT=$DATA_ROOT/logs/E004/$RUN
WATCH_STEPS="20 40 60 80 100"
if [[ $ARM == *-resume-generic ]]; then
  [[ -f $OUT/ckpt/global_step_$RSTEP/data.pt && -n $(ls $OUT/ckpt/global_step_$RSTEP/actor/model_world_size_*.pt 2>/dev/null) ]] \
    || { echo "no complete checkpoint global_step_$RSTEP in $OUT" >&2; exit 1; }
  export RESUME_FROM=$OUT/ckpt/global_step_$RSTEP
  WATCH_STEPS=$(for s in 20 40 60 80 100; do (( s > RSTEP )) && echo -n "$s "; done)
  git -C "$WT" rev-parse HEAD > "$OUT/base_commit.$RTAG.txt"
  echo "worktree: $WT (branch $(git -C "$WT" branch --show-current))" >> "$OUT/base_commit.$RTAG.txt"
  echo "resume_from: $RESUME_FROM  MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144 (default)}  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}" >> "$OUT/base_commit.$RTAG.txt"
  git -C "$WT" diff > "$OUT/code.$RTAG.patch"
  git -C "$WT" status > "$OUT/git_status.$RTAG.txt"
elif [[ $ARM == A-resume ]]; then
  [[ -f $OUT/ckpt/global_step_20/data.pt && -f $OUT/train.part1.log ]] || { echo "no step-20 checkpoint / part-1 logs in $OUT" >&2; exit 1; }
  export RESUME_FROM=$OUT/ckpt/global_step_20
  WATCH_STEPS="40 60 80 100"
  git -C "$WT" rev-parse HEAD > "$OUT/base_commit.resume.txt"
  echo "worktree: $WT (branch $(git -C "$WT" branch --show-current))" >> "$OUT/base_commit.resume.txt"
  git -C "$WT" diff > "$OUT/code.resume.patch"
  git -C "$WT" status > "$OUT/git_status.resume.txt"
  nohup "$REPO/.venv-verl/bin/python" "$REPO/scripts/exp/E004_answer_check.py" --run $RUN --step 21 --watch --kill \
    > "$OUT/answer_check.log" 2>&1 &
else
  [[ -e $OUT ]] && { echo "$OUT exists" >&2; exit 1; }
  mkdir -p "$OUT"
  git -C "$WT" rev-parse HEAD > "$OUT/base_commit.txt"
  git -C "$WT" diff > "$OUT/code.patch"
  git -C "$WT" status > "$OUT/git_status.txt"
  echo "worktree: $WT (branch $(git -C "$WT" branch --show-current))" >> "$OUT/base_commit.txt"
  echo "MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144 (default)}  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}" >> "$OUT/base_commit.txt"
fi
if [[ $ARM != pilot ]]; then
  nohup "$REPO/.venv-verl/bin/python" "$REPO/scripts/exp/E004_ckpt_watch.py" --run "$OUT" --steps $WATCH_STEPS \
    > "$OUT/ckpt_watch.log" 2>&1 &
  echo $! > "$OUT/ckpt_watch.pid"
fi
cd "$WT"
MODEL=Qwen/Qwen2.5-3B-Instruct MODEL_NAME=qwen2.5-3b-instruct DATA_DIR=$DATA_ROOT/verl_data/e004/$MIX \
TRAIN_BATCH=64 MINI_BATCH=32 ROLLOUT_N=5 LR=1e-6 WARMUP_RATIO=0.06 TOTAL_STEPS=$STEPS \
MAX_PROMPT=1024 MAX_RESPONSE=2048 REWARD=alias_match DEPTH_POLICY=max:3 MAX_TURNS=3 SEED=0 \
TEST_FREQ=-1 VAL_BEFORE_TRAIN=False SAVE_FREQ=$SAVE CKPT_KEEP=null RUN_DIR=$OUT GROUP=E004 \
  bash scripts/train_grpo_3b.sh full
