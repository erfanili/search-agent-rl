#!/usr/bin/env bash
# E005 garble diagnosis, item 4: arm O's recipe for 1 step, 3 times in a row, fresh processes each time, on GPUs 4-7.
# Before each launch: nvidia-smi (GPUs 4-7 memory + compute apps) and the ray/vllm/python processes of all users are
# written to garble/probe<i>_preflight.txt; a launch is skipped (and the script stops) if any compute app is on 4-7.
# After each launch: the same snapshot to probe<i>_post.txt, then 60 s pause.
#   bash scripts/exp/E005_garble_probe.sh      -> $DATA_ROOT/logs/E005/garble/E005-O-probe{1,2,3}/
#   PREFIX_CACHING=False bash scripts/exp/E005_garble_probe.sh np   -> .../E005-O-probenp{1,2,3}/ (only prefix caching off)
set -uo pipefail
P=${1:-}   # probe tag prefix
REPO=$(cd "$(dirname "$0")/../.." && pwd)
L=/data/erfan/search-agent-rl-data/logs/E005/garble
mkdir -p $L
snap() {
  { date '+%F %T'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    echo "-- compute apps (all GPUs): gpu_uuid, pid, name, used_memory"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    echo "-- gpu index -> uuid"
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader
    echo "-- ray / vllm / raylet / gcs processes (all users)"
    ps -eo user,pid,etime,cmd | grep -E "raylet|gcs_server|vllm|ray::|AsyncvLLM|WorkerDict|main_ppo" | grep -v grep
  } > "$1"
}
busy47() {  # count compute apps on GPUs 4-7
  local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1>=4 && $1<=7 {print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c -F -f <(echo "$u") || true
}
while pgrep -f "python scripts/exp/E005_garble_temp.py" > /dev/null; do sleep 30; done   # standalone runs finish first
for i in 1 2 3; do
  snap $L/probe${P}${i}_preflight.txt
  n=$(busy47)
  if [[ $n -gt 0 ]]; then echo "$(date '+%F %T') probe ${P}$i: $n compute apps on GPUs 4-7; stopping" >> $L/probe.log; exit 1; fi
  echo "$(date '+%F %T') probe ${P}$i start (GPUs 4-7 free)" >> $L/probe.log
  MAX_TOKENS_PER_GPU=4608 bash $REPO/scripts/exp/E005_train.sh probe O ${P}$i > $L/E005-O-probe${P}${i}-launch.log 2>&1
  echo "$(date '+%F %T') probe ${P}$i exit $?" >> $L/probe.log
  sleep 60
  snap $L/probe${P}${i}_post.txt
done
echo "$(date '+%F %T') probes ${P} done" >> $L/probe.log
