#!/usr/bin/env bash
# E007 training: one run of the E007 queue, run FROM the worktree ../sar-alias-reward (as E006). GPUs 4-7.
#   bash scripts/exp/E007_train.sh pilot E007-N-s0              # 1 step, no checkpoint -> $DATA_ROOT/logs/E007/E007-N-pilot
#   bash scripts/exp/E007_train.sh full E007-S-s1               # 100 steps            -> $DATA_ROOT/logs/E007/E007-S-s1
#   bash scripts/exp/E007_train.sh resume E007-S-s1 60 r1       # from ckpt/global_step_60 (provenance *.r1.*)
# = E006_train.sh (alias_match_strict, PREFIX_CACHING=False, MAX_TOKENS_PER_GPU from the environment, recipe E004/E005/E006)
# with the run spec taken from the run ID (card):
#   E007-S-s1 / -s2, E007-O-s1 / -s2, E007-C-s1: arm, SEED 1 / 2 (data.seed and rollout.seed, train_grpo_3b.sh);
#   E007-N-s0: REWARD_ARM=none (outcome only, no decision term, no cost);
#   E007-Sshuf-s0: arm S on $DATA_ROOT/verl_data/e007/mix50sc_shuf (sc_agreement permuted, E007_shuf_data.py);
#   E007-S-l01-s0 / E007-S-l05-s0: arm S with ARM_LAM 0.1 / 0.5.
# Defaults: lam 0.3, cost 0.20 (C only), data e005/mix50sc, SEED 0.
set -euo pipefail
MODE=${1:?usage: E007_train.sh pilot|full|resume <run id> [step tag]}
ID=${2:?run id}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WT=$(cd "$REPO/../sar-alias-reward" && pwd)
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data}
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf_cache}
LAM=0.3; COST=0.20; DATA_DIR=$DATA_ROOT/verl_data/e005/mix50sc
case $ID in
  E007-S-s1) ARM=S; SEED=1 ;;          E007-S-s2) ARM=S; SEED=2 ;;
  E007-O-s1) ARM=O; SEED=1 ;;          E007-O-s2) ARM=O; SEED=2 ;;
  E007-C-s1) ARM=C; SEED=1 ;;
  E007-N-s0) ARM=none; SEED=0 ;;
  E007-Sshuf-s0) ARM=S; SEED=0; DATA_DIR=$DATA_ROOT/verl_data/e007/mix50sc_shuf ;;
  E007-S-l01-s0) ARM=S; SEED=0; LAM=0.1 ;;
  E007-S-l05-s0) ARM=S; SEED=0; LAM=0.5 ;;
  *) echo "unknown run id $ID" >&2; exit 1 ;;
esac
case $MODE in
  pilot)  RUN=${ID%-s*}-pilot; STEPS=1; SAVE=-1 ;;
  full)   RUN=$ID; STEPS=100; SAVE=20 ;;
  resume) RUN=$ID; STEPS=100; SAVE=20; RSTEP=${3:?step}; RTAG=${4:?tag} ;;
  *) echo "mode: pilot|full|resume" >&2; exit 1 ;;
esac
OUT=$DATA_ROOT/logs/E007/$RUN
WATCH_STEPS="20 40 60 80 100"
prov() {
  git -C "$WT" rev-parse HEAD > "$OUT/base_commit$1.txt"
  echo "worktree: $WT (branch $(git -C "$WT" branch --show-current))" >> "$OUT/base_commit$1.txt"
  echo "run: $ID  arm: $ARM  lam: $LAM  cost: $COST  seed: $SEED  data: $DATA_DIR  PREFIX_CACHING=False  resume_from: ${RESUME_FROM:-none}  MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144 (default)}  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset}" >> "$OUT/base_commit$1.txt"
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
MODEL=Qwen/Qwen2.5-3B-Instruct MODEL_NAME=qwen2.5-3b-instruct DATA_DIR=$DATA_DIR \
TRAIN_BATCH=64 MINI_BATCH=32 ROLLOUT_N=5 LR=1e-6 WARMUP_RATIO=0.06 TOTAL_STEPS=$STEPS \
MAX_PROMPT=1024 MAX_RESPONSE=2048 REWARD=alias_match_strict REWARD_ARM=$ARM ARM_LAM=$LAM ARM_COST=$COST \
DEPTH_POLICY=max:3 MAX_TURNS=3 SEED=$SEED PREFIX_CACHING=False \
TEST_FREQ=-1 VAL_BEFORE_TRAIN=False SAVE_FREQ=$SAVE CKPT_KEEP=null RUN_DIR=$OUT GROUP=E007 \
  bash scripts/train_grpo_3b.sh full
