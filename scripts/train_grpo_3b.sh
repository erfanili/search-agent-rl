#!/usr/bin/env bash
# GRPO on Qwen2.5-3B Base with Search-R1 rollouts (D-07..D-09), full fine-tuning.
# LoRA isn't supported: verl v0.5.0's async rollout can't serve adapters (D-08 and the challenges table).
#
#   bash scripts/train_grpo_3b.sh full                   # one-day run: 300 steps of 128 x 5
#   TOTAL_STEPS=3 bash scripts/train_grpo_3b.sh full     # pilot: step time and memory before a long run
#   DRY_RUN=1 bash scripts/train_grpo_3b.sh full         # print the resolved verl config, start nothing
#   REWARD=em_hit bash scripts/train_grpo_3b.sh full     # reward arm: em (default), f1, em_hit, em_multi (agent/reward.py)
#
# Hyperparameters follow Search-R1's train_grpo.sh (Qwen2.5-3B, GRPO, EM reward) unless noted:
#   5 samples per prompt, lr 1e-6 with 28.5% warmup, KL loss 0.001 low_var_kl, temperature 1,
#   actor param+optimizer offload, ref param offload, val every 50 steps.
# Defaults are sized to finish in under a day on 5 A5000s (D-12). Search-R1's full recipe:
#   TRAIN_BATCH=512 MINI_BATCH=256 TOTAL_STEPS=1005 MAX_RESPONSE=3072 bash scripts/train_grpo_3b.sh full
# Differences, all deliberate:
#   - max_turns 4 (Search-R1's script: 2): the same environment as our eval rollouts (configs/agent_loop.yaml).
#   - one response budget for the whole episode (MAX_RESPONSE) instead of 500 per turn: verl v0.5 sets
#     max_tokens itself (notes/p1.3_rollout_contract.md).
#   - dynamic micro-batches by token count instead of a fixed micro-batch of 64: episodes vary 150..3000+ tokens.
#   - validation on our frozen sub300 sets (agent/make_train_data.py), not Search-R1's test split.
# Every value below can be overridden from the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpus.sh

METHOD=${1:-}
[[ $METHOD == full ]] || { echo "usage: bash scripts/train_grpo_3b.sh full" >&2; exit 1; }

PY=$PWD/.venv-verl/bin/python
MODEL_PATH=${MODEL_PATH:-$(hf_snapshot models Qwen/Qwen2.5-3B)}
MODEL_PATH=${MODEL_PATH%/}   # verl's copy_to_local rejects a trailing slash
DATA_DIR=${DATA_DIR:-$DATA/verl_data/nq_hotpotqa}
verl_env

# ---- settings -------------------------------------------------------------------
TRAIN_BATCH=${TRAIN_BATCH:-128}        # prompts per step (Search-R1: 512)
ROLLOUT_N=${ROLLOUT_N:-5}              # samples per prompt (the GRPO group)
MINI_BATCH=${MINI_BATCH:-64}           # prompts per actor update -> 2 updates per step, as in Search-R1
MAX_PROMPT=${MAX_PROMPT:-512}          # prompts are ~160 tokens
MAX_RESPONSE=${MAX_RESPONSE:-2048}     # whole episode: model turns + retrieved passages; the tail sets step time
# Actor micro-batch size in tokens. The logits alone are tokens x 151,936 vocab: 12,288 tokens OOMed
# in backward on 24 GB (pilot 2026-10-19). Log-prob passes have no backward and take 4x this.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144}
VLLM_MEM=${VLLM_MEM:-0.7}              # vLLM has the GPU to itself during rollout (actor offloaded, KV freed after)
ENFORCE_EAGER=${ENFORCE_EAGER:-False}  # False: CUDA graphs, much faster decoding for a 3B model (verl default: True)
AGENT_WORKERS=${AGENT_WORKERS:-8}
TOTAL_STEPS=${TOTAL_STEPS:-300}
WARMUP_RATIO=${WARMUP_RATIO:-0.285}
TEST_FREQ=${TEST_FREQ:-50}
SAVE_FREQ=${SAVE_FREQ:-100}
CKPT_KEEP=${CKPT_KEEP:-3}              # a full fine-tuning checkpoint with optimizer state is ~40 GB
REWARD=${REWARD:-em}                   # agent/reward.py REWARDS; everything else is held fixed across arms
HIT_BONUS=${HIT_BONUS:-0.2}            # em_hit: reward for gold answer appearing in retrieved passages
MULTI_BONUS=${MULTI_BONUS:-0.2}        # em_multi: reward for a correct answer reached with >= 2 distinct queries
LOGGER=${LOGGER:-'["console"]'}        # '["console","wandb"]' once wandb is logged in

LR=${LR:-1e-6}
METHOD_ARGS=(
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)
# -------------------------------------------------------------------------------

[[ -n $MODEL_PATH && -f $MODEL_PATH/config.json ]] || { echo "Qwen2.5-3B not in $HF_HOME: hf download Qwen/Qwen2.5-3B" >&2; exit 1; }
[[ -f $DATA_DIR/train.parquet ]] || { echo "no training data: $PY agent/make_train_data.py --out $DATA_DIR" >&2; exit 1; }
DRY_RUN=${DRY_RUN:-0}
(( DRY_RUN )) || require_retrieval_server
if (( ! DRY_RUN )) && pgrep -u "$USER" -f "^ray::" >/dev/null; then
    echo "leftover Ray workers from an earlier run are holding GPU memory; stop them first:" >&2
    echo "  pkill -u \$USER -f '^ray::'" >&2
    exit 1
fi

N_TRAIN=${#WORK_GPUS[@]}
(( N_TRAIN > 0 )) || { echo "no training GPUs left after RETRIEVAL_GPU=$RETRIEVAL_GPU" >&2; exit 1; }
(( (TRAIN_BATCH * ROLLOUT_N) % N_TRAIN == 0 && (MINI_BATCH * ROLLOUT_N) % N_TRAIN == 0 )) || {
    echo "batch $TRAIN_BATCH x $ROLLOUT_N (mini $MINI_BATCH) doesn't split over $N_TRAIN GPUs" >&2; exit 1; }
GPUS=$(IFS=,; echo "${WORK_GPUS[*]}")

STAMP=$(date +%Y%m%d_%H%M%S)
NAME=grpo_qwen2.5-3b_${METHOD}_${REWARD}_${STAMP}
OUT=$DATA/logs/$NAME
HYDRA_EXTRA=()
if (( DRY_RUN )); then OUT=$(mktemp -d "$DATA/rt/dry_XXXX"); HYDRA_EXTRA=(--cfg job); fi
mkdir -p "$OUT/rollouts"
sed "s#^\(  dump_dir:\).*#\1 $OUT/rollouts#" configs/agent_loop.yaml > "$OUT/agent_loop.yaml"
if "$PY" -c "import flash_attn" 2>/dev/null; then REMOVE_PAD=True; else REMOVE_PAD=False; fi
{
    echo "date: $(date -Is)"; echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "method: $METHOD  lr: $LR  model: $MODEL_PATH"; echo "reward: $REWARD hit_bonus=$HIT_BONUS multi_bonus=$MULTI_BONUS"; echo "gpus: $GPUS (retrieval on $RETRIEVAL_GPU)"
    echo "batch=$TRAIN_BATCH n=$ROLLOUT_N mini=$MINI_BATCH max_response=$MAX_RESPONSE steps=$TOTAL_STEPS"
    echo "--- agent_loop.yaml"; cat "$OUT/agent_loop.yaml"
} > "$OUT/config.txt"
echo "training GPUs: $GPUS (retrieval server on $RETRIEVAL_GPU); output: $OUT"

CUDA_VISIBLE_DEVICES=$GPUS taskset -c "$(cpus_for "$GPUS")" "$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESPONSE \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=$REMOVE_PAD \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$WARMUP_RATIO \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_MEM \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_TOKENS_PER_GPU * 4)) \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$OUT/agent_loop.yaml" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_TOKENS_PER_GPU * 4)) \
    "${METHOD_ARGS[@]}" \
    custom_reward_function.path="$PWD/agent/reward.py" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.metric=$REWARD \
    +custom_reward_function.reward_kwargs.hit_bonus=$HIT_BONUS \
    +custom_reward_function.reward_kwargs.multi_bonus=$MULTI_BONUS \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name=search-agent-rl \
    trainer.experiment_name=$NAME \
    trainer.n_gpus_per_node=$N_TRAIN \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$CKPT_KEEP \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.default_local_dir="$OUT/ckpt" \
    hydra.run.dir="$OUT/hydra" \
    "${HYDRA_EXTRA[@]}" \
    2>&1 | tee "$OUT/train.log"
