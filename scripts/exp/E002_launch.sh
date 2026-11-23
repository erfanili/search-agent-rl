#!/usr/bin/env bash
# E002 launcher: one job queue per GPU (4-7), jobs run in sequence on each GPU.
#   bash scripts/exp/E002_launch.sh pilot     # 50 q x 5 models x P1, P2, P6
#   bash scripts/exp/E002_launch.sh full      # 1,000 q x 5 models x P1-P6, plus P2 forced f0 / f1
# Job = "<model> <cond> <depth>". Logs: $DATA_ROOT/logs/E002[/pilot]/<model>/<cond>.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/data/erfan/search-agent-rl-data/hf_cache VLLM_CACHE_ROOT=/data/erfan/search-agent-rl-data/vllm_cache
MODE=${1:?pilot or full}
BASE=/data/erfan/search-agent-rl-data/logs/E002; FLAG=""
[[ $MODE == pilot ]] && { BASE=$BASE/pilot; FLAG=--pilot; CONDS="P1 P2 P6"; DEPTHS=""; }
[[ $MODE == full ]] && { CONDS="P1 P2 P3 P4 P5 P6"; DEPTHS="f0 f1"; }
jobs_for() {  # model list -> job lines
    for m in "$@"; do
        for c in $CONDS; do echo "$m $c free"; done
        for d in $DEPTHS; do echo "$m P2 $d"; done
    done
}
queue() {  # gpu, then job lines on stdin
    local gpu=$1 m c d tag rc
    while read -r m c d; do
        tag=$c; [[ $d != free ]] && tag=$c-$d
        mkdir -p "$BASE/$m"
        rc=0
        CUDA_VISIBLE_DEVICES=$gpu .venv-verl/bin/python scripts/exp/E002_run.py --model "$m" --cond "$c" --depth "$d" $FLAG \
            > "$BASE/$m/$tag.log" 2>&1 || rc=$?
        echo "[$(date '+%F %T')] GPU $gpu $m $tag exit $rc"
    done
}
mkdir -p "$BASE"
if [[ $MODE == pilot ]]; then
    jobs_for qwen3b-base qwen7b-instruct | queue 4 &
    jobs_for qwen3b-instruct | queue 5 &
    jobs_for searchr1-3b | queue 6 &
    jobs_for grpo-e1a-s60 | queue 7 &
    wait
else
    # shared queue: each GPU worker takes the next job (7B first, it is slowest); flock guards the job file
    Q=$BASE/queue.txt; jobs_for qwen7b-instruct qwen3b-base qwen3b-instruct searchr1-3b grpo-e1a-s60 > "$Q"
    worker() {
        local gpu=$1 line
        while true; do
            line=$(flock "$Q.lock" sh -c 'l=$(head -n1 "$0"); [ -n "$l" ] && sed -i 1d "$0"; echo "$l"' "$Q")
            [[ -z $line ]] && break
            echo "$line" | queue "$gpu"
        done
    }
    for g in 4 5 6 7; do worker $g & done
    wait
fi
echo "[$(date '+%F %T')] all done"
