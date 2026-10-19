#!/usr/bin/env bash
# Stop a running P0.7 smoke test, including the Ray workers verl started.
#   bash scripts/stop_smoke.sh
# (`ray stop` crashes in this env, so Ray's processes are stopped directly.)

# [x]-style patterns keep pkill from matching this script's own command line
pkill -f "[p]0_7_smoke_grpo_gsm8k.sh"
pkill -f "[v]erl.trainer.main_ppo"
pkill -f "[n]vidia-smi -i .*-l 2"
sleep 5
pkill -9 -f "[r]ay/dashboard|[r]ay/_private|[r]ay/core/src/ray|[d]efault_worker|[R]ay::"
sleep 1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
