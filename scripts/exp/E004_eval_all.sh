#!/usr/bin/env bash
# E004 evaluation queue: every merged checkpoint (steps 20..100) of both arms, shared queue on GPUs 4-7.
# Per job: E004_eval.py (P2, P1, P6 rollouts + S5 greedy turn; --relabel at step 100), then E004_hidden.py.
# A job that writes its outputs but whose process does not exit is cut by `timeout` (vLLM shutdown hang seen in E003);
# the hidden stage runs whenever s5_turn1.jsonl and all three samples files exist.
#   bash scripts/exp/E004_eval_all.sh 2>&1 | tee $DATA_ROOT/logs/E004/eval/launch.log
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data} HF_HOME=${HF_HOME:-/data/erfan/search-agent-rl-data/hf_cache}
E=$DATA_ROOT/logs/E004/eval
mkdir -p $E
Q=$E/queue.txt
: > $Q
# only checkpoints whose merge was confirmed to load (an arm stopped early is evaluated on what exists; Erfan 2026-11-21)
for s in 100 20 40 60 80; do for r in E004-A-s0 E004-B-s0; do
  if [ -f $DATA_ROOT/logs/E004/$r/hf/step_$s/LOAD_OK.json ]; then echo "$r $s" >> $Q; else echo "skip $r step$s: no confirmed merge"; fi
done; done
worker() {
  local g=$1
  while true; do
    j=$(flock $Q.lock sh -c "head -n1 $Q; sed -i 1d $Q")
    [ -z "$j" ] && break
    set -- $j; r=$1; s=$2; d=$E/$r/step$s; mkdir -p $d
    rl=""; [ "$s" = 100 ] && rl=--relabel
    CUDA_VISIBLE_DEVICES=$g timeout 4h .venv-verl/bin/python scripts/exp/E004_eval.py --run $r --step $s $rl > $d/eval.log 2>&1
    e1=$?
    if [ -f $d/s5_turn1.jsonl ] && [ -f $d/P1/samples.jsonl ] && [ -f $d/P2/samples.jsonl ] && [ -f $d/P6/samples.jsonl ]; then
      CUDA_VISIBLE_DEVICES=$g .venv-verl/bin/python scripts/exp/E004_hidden.py --run $r --step $s > $d/hidden.log 2>&1
      e2=$?
    else e2=skipped; fi
    echo "[$(date '+%F %T')] GPU $g $r step$s eval exit $e1 hidden exit $e2"
  done
}
for g in 4 5 6 7; do worker $g & done
wait
echo "[$(date '+%F %T')] all done"
