#!/usr/bin/env bash
# E006 evaluation queue (E006_eval_all.sh with --exp E006 and --no-prefix-cache): every confirmed merged checkpoint (hf/step_<k>/LOAD_OK.json)
# of arms O, S, C, shared queue on GPUs 4-7. Per job: E004_eval.py --exp E006 (P2, P1, P6 rollouts on the E001 set,
# 5 samples, T=0.7; S5 greedy turn; --relabel at the arm's last confirmed step, step 100 when the arm finished), then
# E004_hidden.py --exp E006 (layer hidden states, S5 margin). Step 0 = E002/E003/E004 outputs (not rerun).
#   bash scripts/exp/E006_eval_all.sh 2>&1 | tee $DATA_ROOT/logs/E006/eval/launch.log
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data} HF_HOME=${HF_HOME:-/data/erfan/search-agent-rl-data/hf_cache}
E=$DATA_ROOT/logs/E006/eval
mkdir -p $E
Q=$E/queue.txt
: > $Q
declare -A LAST
for r in E006-O-s0 E006-S-s0 E006-C-s0; do
  for s in 20 40 60 80 100; do [ -f $DATA_ROOT/logs/E006/$r/hf/step_$s/LOAD_OK.json ] && LAST[$r]=$s; done
done
for s in 100 80 60 40 20; do for r in E006-O-s0 E006-S-s0 E006-C-s0; do
  if [ -f $DATA_ROOT/logs/E006/$r/hf/step_$s/LOAD_OK.json ]; then echo "$r $s ${LAST[$r]}" >> $Q; else echo "skip $r step$s: no confirmed merge"; fi
done; done
worker() {
  local g=$1
  while true; do
    j=$(flock $Q.lock sh -c "head -n1 $Q; sed -i 1d $Q")
    [ -z "$j" ] && break
    set -- $j; r=$1; s=$2; last=$3; d=$E/$r/step$s; mkdir -p $d
    rl=""; [ "$s" = "$last" ] && rl=--relabel
    CUDA_VISIBLE_DEVICES=$g timeout 4h .venv-verl/bin/python scripts/exp/E004_eval.py --exp E006 --no-prefix-cache --run $r --step $s $rl > $d/eval.log 2>&1
    e1=$?
    if [ -f $d/s5_turn1.jsonl ] && [ -f $d/P1/samples.jsonl ] && [ -f $d/P2/samples.jsonl ] && [ -f $d/P6/samples.jsonl ]; then
      CUDA_VISIBLE_DEVICES=$g .venv-verl/bin/python scripts/exp/E004_hidden.py --exp E006 --run $r --step $s > $d/hidden.log 2>&1
      e2=$?
    else e2=skipped; fi
    echo "[$(date '+%F %T')] GPU $g $r step$s eval exit $e1 hidden exit $e2"
  done
}
for g in 4 5 6 7; do worker $g & done
wait
echo "[$(date '+%F %T')] all done"
