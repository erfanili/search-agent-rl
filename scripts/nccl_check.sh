#!/usr/bin/env bash
# Check whether NCCL GPU-to-GPU transfers work on the GPUs in configs/gpus.env,
# with P2P on and with P2P off. Takes under 2 minutes, most of it the P2P-on timeout if it hangs.
#   bash scripts/nccl_check.sh
set -uo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
export CUDA_VISIBLE_DEVICES=$GPUS

for p2p_off in 0 1; do
    echo "== NCCL_P2P_DISABLE=$p2p_off on GPUs $GPUS"
    NCCL_P2P_DISABLE=$p2p_off timeout 90 .venv-verl/bin/python scripts/nccl_check.py 2>&1 \
        | grep -E "broadcast|Error|timeout|Timeout" | head -3
    [[ ${PIPESTATUS[0]} -eq 124 ]] && echo "   hung (killed after 90 s)"
done
echo
echo "If P2P on hangs and P2P off is OK, keep NCCL_P2P_DISABLE=1 (the smoke test's default)."
