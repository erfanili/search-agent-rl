#!/usr/bin/env bash
# P0.7 smoke test: verl GRPO on GSM8K, Qwen2.5-0.5B, 20 steps, 4 GPUs.
#
# Goal: prove verl + vLLM + FSDP run end to end on this server, and record
# per-step time and GPU memory. Not a training run: no validation, no checkpoints.
#
# Usage (from the repo root, inside .venv-verl):
#   VERL_DIR=/path/to/verl MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
#       bash scripts/p0_7_smoke_grpo_gsm8k.sh
#   python scripts/summarize_smoke.py logs/p0_7_smoke_<timestamp>
#
# Outputs in logs/p0_7_smoke_<timestamp>/:
#   train.log    verl console output (one "step:N - key:value - ..." line per step)
#   gpu.csv      nvidia-smi samples every 2 s: true per-GPU memory, including vLLM's reserve
#   config.txt   every setting below, for the decision log

set -euo pipefail

# ---- settings (edit here, not in the command) --------------------------------
VERL_DIR=${VERL_DIR:?set VERL_DIR to your verl clone (needs examples/data_preprocess/gsm8k.py)}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-/data/erfan/verl_data/gsm8k}
GPUS=${GPUS:-0,1,2,3}
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

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

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=logs/p0_7_smoke_${STAMP}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPUS

# flash-attn is needed for use_remove_padding (packing without pad tokens)
if python -c "import flash_attn" 2>/dev/null; then REMOVE_PAD=True; else REMOVE_PAD=False; fi

# 1. data: verl's own GSM8K preprocessor writes train/test parquet with prompt + rule-based reward info
if [[ ! -f $DATA_DIR/train.parquet ]]; then
    python "$VERL_DIR/examples/data_preprocess/gsm8k.py" --local_dir "$DATA_DIR"
fi

# 2. record the configuration
{
    echo "date: $(date -Is)"
    echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "verl: $(python -c 'import verl; print(verl.__version__)' 2>/dev/null || echo unknown)"
    echo "vllm: $(python -c 'import vllm; print(vllm.__version__)')"
    echo "torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    echo "gpus: $GPUS ($N_GPUS)"
    echo "model: $MODEL_PATH"
    echo "steps=$STEPS train_batch=$TRAIN_BATCH n=$ROLLOUT_N mini_batch=$MINI_BATCH micro_per_gpu=$MICRO_PER_GPU"
    echo "max_prompt=$MAX_PROMPT max_response=$MAX_RESPONSE vllm_mem=$VLLM_MEM lr=$LR kl=$KL_COEF remove_padding=$REMOVE_PAD"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$OUT/config.txt"

# 3. GPU memory sampler in the background, stopped on exit
nvidia-smi -i "$GPUS" --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,nounits -l 2 > "$OUT/gpu.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# 4. train
python -m verl.trainer.main_ppo \
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
    2>&1 | tee "$OUT/train.log"

echo "done: $OUT"
echo "next: python scripts/summarize_smoke.py $OUT"