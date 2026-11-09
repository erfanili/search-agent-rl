#!/usr/bin/env bash
# Start the retrieval server (P0.6). The e5 query encoder runs on RETRIEVAL_GPU from configs/gpus.env,
# or on CPU when it is "cpu"; FAISS search is on CPU either way.
# Usage: scripts/serve_retrieval.sh [index_dir]
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
INDEX=${1:-$DATA/index_e5_IVF4096_SQ8}
mkdir -p "$INDEX/logs"

if [[ $RETRIEVAL_GPU == cpu ]]; then
    # training pins itself to every CPU too; a fixed thread count keeps FAISS from oversubscribing
    DEVICE=cpu VISIBLE= CPUS=0-95 THREADS=${RETRIEVAL_THREADS:-32}
else
    DEVICE=cuda VISIBLE=$RETRIEVAL_GPU CPUS=$(cpus_for "$RETRIEVAL_GPU") THREADS=$(ncpus_for "$RETRIEVAL_GPU")
fi
CUDA_VISIBLE_DEVICES=$VISIBLE nohup taskset -c "$CPUS" \
    .venv/bin/python retrieval/retrieval_server.py --index-dir "$INDEX" --device $DEVICE --nprobe "${NPROBE:-64}" \
    --threads "$THREADS" --port 8000 > "$INDEX/logs/server.log" 2>&1 &
echo "server starting, encoder on $RETRIEVAL_GPU (pid $!); log: $INDEX/logs/server.log; check: curl localhost:8000/health"
