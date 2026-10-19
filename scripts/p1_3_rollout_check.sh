#!/usr/bin/env bash
# P1.3 rollout check: the search agent's ids and masks are right in the REAL pipeline.
#
#   1. contract tests     tests/test_rollout_contract.py with the real tokenizer (fake model, no GPU)
#   2. tiny dataset       64 HotpotQA eval questions in verl's format (the step uses 6; plumbing only, never training)
#   3. one verl step      GRPO, vLLM async rollout + our agent loop + live retrieval server;
#                         the agent loop writes every episode (ids, text, mask) to $OUT/rollouts/
#   4. report             scripts/view_rollouts.py: invariants on the real episodes + an HTML page
#                         with env tokens shaded, for reading by eye
#
# Not a training run: 1 step, Qwen2.5-0.5B-Instruct, 6 questions x 4 samples.
#
# Usage (from anywhere; retrieval server must be up: bash scripts/serve_retrieval.sh):
#   bash scripts/p1_3_rollout_check.sh
#   MODEL_PATH=/path/to/other/model bash scripts/p1_3_rollout_check.sh
# Output: /data/erfan/search-agent-rl-data/logs/p1_3_rollout_check_<timestamp>/{train.log,rollouts/,report.html,report.txt}
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpus.sh

# ---- settings -----------------------------------------------------------------
PY=$PWD/.venv-verl/bin/python
MODEL_PATH=${MODEL_PATH:-/data/erfan/search-agent-rl-data/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}
DATA_DIR=/data/erfan/search-agent-rl-data/verl_data/p1_3_check
export HF_HOME=${HF_HOME:-/data/erfan/search-agent-rl-data/hf_cache}
export RAY_TMPDIR=${RAY_TMPDIR:-/data/erfan/search-agent-rl-data/rt}  # short name: Ray AF_UNIX socket paths must be <=107 bytes
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/data/erfan/search-agent-rl-data/vllm_cache}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}   # P2P hangs on this machine (notes/smoke_test.md)
export VLLM_USE_V1=1   # verl async server needs vLLM V1; unset, vLLM falls back to V0 (engine built off the main thread)
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}  # Ray workers import agent/ from the repo root

TRAIN_BATCH=6          # prompts
ROLLOUT_N=4            # samples per prompt -> 24 episodes: splits evenly over 1-4 GPUs and 8 agent workers
MINI_BATCH=6
MICRO_PER_GPU=2
MAX_PROMPT=512
MAX_RESPONSE=2048      # total response budget per episode (model + env tokens)
VLLM_MEM=0.4
AGENT_WORKERS=8
# -------------------------------------------------------------------------------

# ---- retrieval server must be up, and training stays off its GPU -------------
curl -sf localhost:8000/health >/dev/null || { echo "retrieval server not up: bash scripts/serve_retrieval.sh" >&2; exit 1; }
CANDIDATES=()
for g in "${GPU_LIST[@]}"; do [[ $g == "$RETRIEVAL_GPU" ]] || CANDIDATES+=("$g"); done
fits() {
    local k=$1
    (( (TRAIN_BATCH * ROLLOUT_N) % k == 0 && (MINI_BATCH * ROLLOUT_N / k) % MICRO_PER_GPU == 0 ))
}
N_GPUS=0
for (( k=${#CANDIDATES[@]}; k>=1; k-- )); do if fits "$k"; then N_GPUS=$k; break; fi; done
(( N_GPUS > 0 )) || { echo "no usable GPU count from: ${CANDIDATES[*]:-none}" >&2; exit 1; }
GPUS=$(IFS=,; echo "${CANDIDATES[*]:0:N_GPUS}")
echo "training GPUs: $GPUS (retrieval server on $RETRIEVAL_GPU)"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=/data/erfan/search-agent-rl-data/logs/p1_3_rollout_check_${STAMP}
mkdir -p "$OUT/rollouts" "$RAY_TMPDIR"

# ---- 1. contract tests with the real tokenizer ---------------------------------
echo "== 1. contract tests"
TOKENIZER_PATH=$MODEL_PATH "$PY" -m pytest tests/test_rollout_contract.py tests/test_qa_metrics.py -q | tee "$OUT/contract_tests.txt"

# ---- 2. tiny dataset -------------------------------------------------------------
echo "== 2. data"
"$PY" agent/make_check_data.py --out "$DATA_DIR" --n 64

# ---- 3. one verl step ------------------------------------------------------------
# agent loop config for this run: the repo's defaults plus a dump directory
sed "s#^\(  dump_dir:\).*#\1 $OUT/rollouts#" configs/agent_loop.yaml > "$OUT/agent_loop.yaml"
if "$PY" -c "import flash_attn" 2>/dev/null; then REMOVE_PAD=True; else REMOVE_PAD=False; fi
{
    echo "date: $(date -Is)"; echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "model: $MODEL_PATH"; echo "gpus: $GPUS"; echo "batch=$TRAIN_BATCH n=$ROLLOUT_N max_response=$MAX_RESPONSE"
    echo "--- agent_loop.yaml"; cat "$OUT/agent_loop.yaml"
} > "$OUT/config.txt"

echo "== 3. one GRPO step (log: $OUT/train.log)"
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_MEM \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$OUT/agent_loop.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    custom_reward_function.path="$PWD/agent/reward.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=search-agent-rl \
    trainer.experiment_name=p1_3_rollout_check_${STAMP} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="$OUT/ckpt" \
    hydra.run.dir="$OUT/hydra" \
    2>&1 | tee "$OUT/train.log"

# ---- 4. report -----------------------------------------------------------------
echo "== 4. report"
"$PY" scripts/view_rollouts.py "$OUT/rollouts" --tokenizer "$MODEL_PATH" --html "$OUT/report.html" | tee "$OUT/report.txt"
echo "done: $OUT  (open report.html to read the episodes)"
