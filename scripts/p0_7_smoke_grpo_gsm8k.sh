#!/usr/bin/env bash
# P0.7 smoke test: verl GRPO on GSM8K, Qwen2.5-0.5B, 20 steps.
#
# Goal: prove verl + vLLM + FSDP run end to end on this server, and record
# per-step time and GPU memory. Not a training run: no validation, no checkpoints.
#
# GPUs come from configs/gpus.env (the only place to change them). verl needs every
# batch to split evenly across GPUs, so the script uses the largest number of GPUs from
# that list that works (e.g. 6 allowed -> 5 used, 3 -> 2) and says so. If the retrieval
# server is running, its GPU is left out.
#
# Usage (from the repo root; no venv activation needed):
#   bash scripts/p0_7_smoke_grpo_gsm8k.sh
#   .venv-verl/bin/python scripts/summarize_smoke.py /data/erfan/search-agent-rl-data/logs/p0_7_smoke_<timestamp>
#
# Outputs in /data/erfan/search-agent-rl-data/logs/p0_7_smoke_<timestamp>/ (all data stays under /data/erfan/search-agent-rl-data):
#   train.log    verl console output (one "step:N - key:value - ..." line per step)
#   gpu.csv      nvidia-smi samples every 2 s: true per-GPU memory, including vLLM's reserve
#   config.txt   every setting below, for the decision log

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpus.sh

# ---- settings (edit here, not in the command) --------------------------------
PY=$PWD/.venv-verl/bin/python   # training env; see notes/environments.md
VERL_DIR=${VERL_DIR:-/data/erfan/search-agent-rl-data/src/verl}
MODEL_PATH=${MODEL_PATH:-/data/erfan/search-agent-rl-data/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}
DATA_DIR=${DATA_DIR:-/data/erfan/search-agent-rl-data/verl_data/gsm8k}
export HF_HOME=${HF_HOME:-/data/erfan/search-agent-rl-data/hf_cache}       # / is ~99% full: keep caches on /data
export RAY_TMPDIR=${RAY_TMPDIR:-/data/erfan/search-agent-rl-data/rt}  # Ray logs and spill files; short name: AF_UNIX socket paths must be <=107 bytes
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/data/erfan/search-agent-rl-data/vllm_cache}  # vLLM compile cache, default ~/.cache/vllm
# GPU-to-GPU (P2P) copies appear to hang on this machine (AMD EPYC, IOMMU on): the first NCCL
# broadcast never finished and the run died after the 30-min timeout. Route transfers through
# host memory instead. Check with: bash scripts/nccl_check.sh
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

STEPS=20
TRAIN_BATCH=64          # prompts per step
ROLLOUT_N=5             # GRPO group size: responses per prompt -> 320 responses per step
MINI_BATCH=32           # prompts per PPO mini-batch -> 2 updates per step
MICRO_PER_GPU=8         # responses per forward/backward per GPU
LOGPROB_MICRO_PER_GPU=16
MAX_PROMPT=512
MAX_RESPONSE=512
VLLM_MEM=0.4            # fraction of each GPU vLLM may reserve for weights + KV cache
LR=1e-6
KL_COEF=0.001
# -------------------------------------------------------------------------------

# ---- pick GPUs from configs/gpus.env -------------------------------------------
CANDIDATES=("${GPU_LIST[@]}")
if server_running; then
    CANDIDATES=()
    for g in "${GPU_LIST[@]}"; do [[ $g == "$RETRIEVAL_GPU" ]] || CANDIDATES+=("$g"); done
    echo "retrieval server is running: leaving its GPU $RETRIEVAL_GPU out"
fi
# verl splits TRAIN_BATCH*ROLLOUT_N responses across GPUs, and each GPU's share of a
# mini-batch must be a whole number of micro-batches
fits() {
    local k=$1 per_gpu
    (( (TRAIN_BATCH * ROLLOUT_N) % k == 0 && (MINI_BATCH * ROLLOUT_N) % k == 0 )) || return 1
    per_gpu=$(( MINI_BATCH * ROLLOUT_N / k ))
    (( per_gpu % MICRO_PER_GPU == 0 ))
}
N_GPUS=0
for (( k=${#CANDIDATES[@]}; k>=1; k-- )); do
    if fits "$k"; then N_GPUS=$k; break; fi
done
if (( N_GPUS == 0 )); then
    echo "no usable GPU count from: ${CANDIDATES[*]:-none}" >&2
    exit 1
fi
GPUS=$(IFS=,; echo "${CANDIDATES[*]:0:N_GPUS}")
if (( N_GPUS < ${#CANDIDATES[@]} )); then
    echo "using $N_GPUS of ${#CANDIDATES[@]} available GPUs (${#CANDIDATES[@]} doesn't split the batch evenly)"
fi
echo "GPUs: $GPUS"
# -------------------------------------------------------------------------------

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=/data/erfan/search-agent-rl-data/logs/p0_7_smoke_${STAMP}
mkdir -p "$OUT" "$RAY_TMPDIR"
export CUDA_VISIBLE_DEVICES=$GPUS

# flash-attn is needed for use_remove_padding (packing without pad tokens)
if "$PY" -c "import flash_attn" 2>/dev/null; then REMOVE_PAD=True; else REMOVE_PAD=False; fi

# 1. data: verl's own GSM8K preprocessor writes train/test parquet with prompt + rule-based reward info
if [[ ! -f $DATA_DIR/train.parquet ]]; then
    "$PY" "$VERL_DIR/examples/data_preprocess/gsm8k.py" --local_dir "$DATA_DIR"
fi

# 2. record the configuration
{
    echo "date: $(date -Is)"
    echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "verl: $("$PY" -c 'import verl; print(verl.__version__)' 2>/dev/null || echo unknown)"
    echo "vllm: $("$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
    echo "torch: $("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    echo "gpus: $GPUS ($N_GPUS used; allowance in configs/gpus.env: ${GPU_LIST[*]})"
    echo "model: $MODEL_PATH"
    echo "steps=$STEPS train_batch=$TRAIN_BATCH n=$ROLLOUT_N mini_batch=$MINI_BATCH micro_per_gpu=$MICRO_PER_GPU"
    echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
    echo "max_prompt=$MAX_PROMPT max_response=$MAX_RESPONSE vllm_mem=$VLLM_MEM lr=$LR kl=$KL_COEF remove_padding=$REMOVE_PAD"
    nvidia-smi -i "$GPUS" --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$OUT/config.txt"

# 3. GPU memory sampler in the background, stopped on exit
nvidia-smi -i "$GPUS" --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,nounits -l 2 > "$OUT/gpu.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# 4. train
"$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESPONSE \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=$REMOVE_PAD \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_MEM \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_PER_GPU \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=search-agent-rl \
    trainer.experiment_name=p0_7_smoke_${STAMP} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$STEPS \
    trainer.default_local_dir="$OUT/ckpt" \
    hydra.run.dir="$OUT/hydra" \
    2>&1 | tee "$OUT/train.log"

echo "done: $OUT"
echo "next: $PY scripts/summarize_smoke.py $OUT"