#!/usr/bin/env bash
# GRPO on Qwen2.5-3B Base with Search-R1 rollouts (D-07..D-09), full fine-tuning.
# LoRA isn't supported: verl v0.5.0's async rollout can't serve adapters (D-08 and the challenges table).
#
#   bash scripts/train_grpo_3b.sh full                   # one-day run: 300 steps of 128 x 5
#   TOTAL_STEPS=3 bash scripts/train_grpo_3b.sh full     # pilot: step time and memory before a long run
#   DRY_RUN=1 bash scripts/train_grpo_3b.sh full         # print the resolved verl config, start nothing
#   REWARD=f1 bash scripts/train_grpo_3b.sh full         # reward: em (default) or f1 (agent/reward.py)
#   DEPTH_POLICY=exact:2 bash scripts/train_grpo_3b.sh full   # env writes the action tags: free (default), exact:k, min:k, max:k
#   MAX_TURNS=3 DEPTH_POLICY=max:3 ...                   # agent_loop.yaml max_turns override; max:k = forced answer after k searches
#   RUN_DIR=$DATA_ROOT/logs/E004/E004-A-s0 ...           # output directory (default: $DATA_ROOT/logs/<generated name>)
#   TEST_FREQ=-1 VAL_BEFORE_TRAIN=False ...              # no validation during training
#   SEED=2 bash scripts/train_grpo_3b.sh full            # data order + vLLM sampling seed (unset = earlier runs)
#   MODEL=Qwen/Qwen2.5-1.5B bash scripts/train_grpo_3b.sh full   # other model; its name goes in the run name
#   DEPTH_POLICY=free RESUME_FROM=<run>/ckpt/global_step_40 TOTAL_STEPS=60 bash scripts/train_grpo_3b.sh full
#                                                        # release: continue a forced run from step 40 in a new run dir
# Per-step depth statistics (searches, zero-variance groups, reward by depth): <run>/reward_steps.jsonl,
#   printed by .venv/bin/python scripts/depth_dynamics.py <run>
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
MODEL=${MODEL:-Qwen/Qwen2.5-3B}
MODEL_PATH=${MODEL_PATH:-$(hf_snapshot models "$MODEL")}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}   # qwen2.5-3b
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
PREFIX_CACHING=${PREFIX_CACHING:-True}   # False: agent/vllm_server.py AsyncvLLMServerNoPrefixCache (verl hard-codes True)
SERVER_ARGS=()
if [[ $PREFIX_CACHING == False ]]; then
  SERVER_ARGS=(actor_rollout_ref.rollout.agent.custom_async_server.path=pkg://agent.vllm_server
               actor_rollout_ref.rollout.agent.custom_async_server.name=AsyncvLLMServerNoPrefixCache)
fi
AGENT_WORKERS=${AGENT_WORKERS:-8}
TOTAL_STEPS=${TOTAL_STEPS:-300}
WARMUP_RATIO=${WARMUP_RATIO:-0.285}
TEST_FREQ=${TEST_FREQ:-50}
SAVE_FREQ=${SAVE_FREQ:-100}
CKPT_KEEP=${CKPT_KEEP:-3}              # a full fine-tuning checkpoint with optimizer state is ~40 GB
REWARD=${REWARD:-em}                   # agent/reward.py REWARDS
REWARD_ARM=${REWARD_ARM:-none}         # E005 arm term added to the outcome reward: none, C (cost), S, O (agent/reward.py)
ARM_LAM=${ARM_LAM:-0.3}                # decision-term weight (arms S, O)
ARM_COST=${ARM_COST:-0.15}             # cost per executed search (arm C)
MAX_TURNS=${MAX_TURNS:-}               # unset: configs/agent_loop.yaml (4)
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
RUN_DIR=${RUN_DIR:-}
DEPTH_POLICY=${DEPTH_POLICY:-free}     # agent/search_r1_rollout.py; validation runs at this depth too (eval_checkpoint.sh for free depth)
SEED=${SEED:-}                          # unset: verl defaults (data order seed 1, vLLM seed 0), as in the earlier runs;
                                       # set: data.seed and rollout.seed both = SEED
RESUME_FROM=${RESUME_FROM:-}           # a ckpt/global_step_N dir: load weights, optimizer, data order and step from it
LOGGER=${LOGGER:-}                     # default: console + wandb when logged in (`.venv-verl/bin/wandb login`), else console
GROUP=${GROUP:-}                       # W&B group: the question/experiment this run serves, e.g. E1
NOTES=${NOTES:-}                       # W&B notes: why this run (question, prediction, what each outcome decides)

LR=${LR:-1e-6}
METHOD_ARGS=(
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)
# -------------------------------------------------------------------------------

[[ -n $MODEL_PATH && -f $MODEL_PATH/config.json ]] || { echo "$MODEL not in $HF_HOME: hf download $MODEL" >&2; exit 1; }
[[ $DEPTH_POLICY =~ ^(free|exact:[0-9]+|min:[0-9]+|max:[0-9]+)$ ]] || { echo "DEPTH_POLICY=$DEPTH_POLICY: expected free, exact:k, min:k or max:k" >&2; exit 1; }
[[ -z $MAX_TURNS || $MAX_TURNS =~ ^[0-9]+$ ]] || { echo "MAX_TURNS=$MAX_TURNS: expected an integer" >&2; exit 1; }
STEP_OFFSET=0   # reward_steps.jsonl step numbers continue from a resumed checkpoint
SEED_ARGS=()
if [[ -n $SEED ]]; then
    [[ $SEED =~ ^[0-9]+$ ]] || { echo "SEED=$SEED: expected an integer" >&2; exit 1; }
    SEED_ARGS=(+data.seed=$SEED +actor_rollout_ref.rollout.seed=$SEED)
fi
RESUME_ARGS=(trainer.resume_mode=disable)
if [[ -n $RESUME_FROM ]]; then
    RESUME_FROM=$(realpath "$RESUME_FROM")
    [[ $RESUME_FROM == */global_step_* && -d $RESUME_FROM/actor ]] || { echo "RESUME_FROM=$RESUME_FROM is not a ckpt/global_step_N dir" >&2; exit 1; }
    RESUME_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="$RESUME_FROM")
    STEP_OFFSET=${RESUME_FROM##*global_step_}
fi
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
TAGS=$REWARD
[[ $REWARD_ARM == none ]] || TAGS+=_arm$REWARD_ARM
[[ $DEPTH_POLICY == free ]] || TAGS+=_${DEPTH_POLICY/:/}       # exact:2 -> exact2
[[ -n $RESUME_FROM ]] && TAGS+=_resume$(basename "$RESUME_FROM" | tr -dc 0-9)   # source run: config.txt
[[ -z $SEED ]] || TAGS+=_seed$SEED
NAME=grpo_${MODEL_NAME}_${METHOD}_${TAGS}_${STAMP}
OUT=${RUN_DIR:-$DATA/logs/$NAME}
HYDRA_EXTRA=()
if (( DRY_RUN )); then OUT=$(mktemp -d "$DATA/rt/dry_XXXX"); HYDRA_EXTRA=(--cfg job); fi
mkdir -p "$OUT/rollouts"
sed -e "s#^\(  dump_dir:\).*#\1 $OUT/rollouts\n  garble_dir: $OUT/garble#" -e "s#^\(  depth_policy:\) *[^ ]*#\1 $DEPTH_POLICY#" \
    configs/agent_loop.yaml > "$OUT/agent_loop.yaml"
if [[ -n $MAX_TURNS ]]; then
    sed -i "s#^\(  max_turns:\) *[0-9]*#\1 $MAX_TURNS#" "$OUT/agent_loop.yaml"
    grep -q "^  max_turns: $MAX_TURNS " "$OUT/agent_loop.yaml" || { echo "max_turns not set in $OUT/agent_loop.yaml" >&2; exit 1; }
fi
grep -q "^  depth_policy: $DEPTH_POLICY " "$OUT/agent_loop.yaml" || { echo "depth_policy not set in $OUT/agent_loop.yaml" >&2; exit 1; }
if "$PY" -c "import flash_attn" 2>/dev/null; then REMOVE_PAD=True; else REMOVE_PAD=False; fi
git diff HEAD > "$OUT/git.diff" 2>/dev/null || true   # uncommitted code this run used (with config.txt's commit)
git ls-files --others --exclude-standard -z | xargs -0 -r tar czf "$OUT/git_untracked.tgz" 2>/dev/null || true
if [[ -z $LOGGER ]]; then
    if wandb_ready; then LOGGER='["console","wandb"]'; else LOGGER='["console"]'; echo "wandb not logged in: console logging only" >&2; fi
fi
export WANDB_DIR=$OUT WANDB_JOB_TYPE=train WANDB_RUN_GROUP=${GROUP:-$METHOD} WANDB_NOTES=$NOTES
export WANDB_TAGS="reward:$REWARD,depth:${DEPTH_POLICY},seed:${SEED:-default},model:$MODEL_NAME${RESUME_FROM:+,resumed},git:$(git rev-parse --short HEAD 2>/dev/null || echo none)$(git diff --quiet HEAD 2>/dev/null || echo -dirty)"
{
    echo "date: $(date -Is)"; echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "method: $METHOD  lr: $LR  model: $MODEL_PATH"; echo "reward: $REWARD  arm: $REWARD_ARM  lam: $ARM_LAM  cost: $ARM_COST"; echo "depth_policy: $DEPTH_POLICY  seed: ${SEED:-default}  resume_from: ${RESUME_FROM:-none}"; echo "gpus: $GPUS (retrieval on $RETRIEVAL_GPU)"; echo "vllm: prefix_caching=$PREFIX_CACHING enforce_eager=$ENFORCE_EAGER"
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
    "${SEED_ARGS[@]}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_MEM \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    "${SERVER_ARGS[@]}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_TOKENS_PER_GPU * 4)) \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$OUT/agent_loop.yaml" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_TOKENS_PER_GPU * 4)) \
    "${METHOD_ARGS[@]}" \
    custom_reward_function.path="$PWD/agent/reward.py" \
    reward_model.reward_manager=batch \
    custom_reward_function.name=compute_score_batch \
    +custom_reward_function.reward_kwargs.log_path="$OUT/reward_steps.jsonl" \
    +custom_reward_function.reward_kwargs.step_offset=$STEP_OFFSET \
    +custom_reward_function.reward_kwargs.sample_path="$OUT/rollout_samples.jsonl" \
    +custom_reward_function.reward_kwargs.sample_n=${SAMPLE_N:-50} \
    +custom_reward_function.reward_kwargs.garble_dir="$OUT/garble" \
    +custom_reward_function.reward_kwargs.garble_warn=${GARBLE_WARN:-0.10} \
    +custom_reward_function.reward_kwargs.metric=$REWARD \
    +custom_reward_function.reward_kwargs.arm=$REWARD_ARM \
    +custom_reward_function.reward_kwargs.lam=$ARM_LAM \
    +custom_reward_function.reward_kwargs.cost=$ARM_COST \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name=search-agent-rl \
    trainer.experiment_name=$NAME \
    trainer.n_gpus_per_node=$N_TRAIN \
    trainer.nnodes=1 \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$CKPT_KEEP \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.default_local_dir="$OUT/ckpt" \
    "${RESUME_ARGS[@]}" \
    hydra.run.dir="$OUT/hydra" \
    "${HYDRA_EXTRA[@]}" \
    2>&1 | tee "$OUT/train.log"
