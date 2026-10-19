#!/usr/bin/env bash
# Wait for the index build, start the retrieval server, then run the Search-R1 baseline eval
# and the P0.6 retrieval benchmark (one after the other, so the benchmark's latency isn't skewed).
#   nohup bash scripts/queue_baseline_after_index.sh > /data/erfan/search-agent-rl-data/logs/queue_baseline.log 2>&1 &
# The server is left running afterwards; stop it with: pkill -f "[r]etrieval_server.py"
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gpus.sh
INDEX=/data/erfan/search-agent-rl-data/index_e5_IVF4096_SQ8
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for the index build to finish"
while pgrep -f "[b]uild_index.py" >/dev/null; do sleep 60; done
[[ -f $INDEX/index_meta.json ]] || { log "build exited but $INDEX/index_meta.json is missing: build failed, stopping"; exit 1; }
log "index ready: $(ls -sh "$INDEX/index.faiss")"

if server_running; then
    log "retrieval server already running"
else
    bash scripts/serve_retrieval.sh "$INDEX"
fi
for i in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 15; done   # up to 30 min
curl -sf localhost:8000/health || { log "server not healthy after 30 min; see $INDEX/logs/server.log"; exit 1; }
echo
log "server up; starting baseline eval"
bash scripts/run_baseline_searchr1.sh || log "baseline eval FAILED (see its infer.log); continuing to the benchmark"
echo
log "starting P0.6 retrieval benchmark"
bash scripts/bench_retrieval.sh || log "benchmark FAILED"
log "all done"
