#!/usr/bin/env bash
# E007 evaluation of the given runs (= E006_eval_all.sh with --exp E007 --no-prefix-cache): steps 20, 60, 100 (card) of
# each run's confirmed merges (hf/step_<k>/LOAD_OK.json); a run stopped early: 20 and 60 if confirmed, plus its last
# confirmed step. Shared queue on GPUs 4-7. Per job: E004_eval.py (P2, P1, P6 on the E001 set, 5 samples, T=0.7; S5
# greedy turn; --relabel at the run's last evaluated step, as E006), then E004_hidden.py. Step 0 reused (E002/E003).
#   bash scripts/exp/E007_eval.sh E007-S-s1 E007-S-s2 ... 2>&1 | tee -a $DATA_ROOT/logs/E007/eval/launch.log
set -u
cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-/data/erfan/search-agent-rl-data} HF_HOME=${HF_HOME:-/data/erfan/search-agent-rl-data/hf_cache}
E=$DATA_ROOT/logs/E007/eval
mkdir -p $E
Q=$E/queue.txt
: > $Q
declare -A LAST STEPS
for r in "$@"; do
  LAST[$r]=0
  for s in 20 40 60 80 100; do [ -f $DATA_ROOT/logs/E007/$r/hf/step_$s/LOAD_OK.json ] && LAST[$r]=$s; done
  st=""; for s in 20 60 100; do [ -f $DATA_ROOT/logs/E007/$r/hf/step_$s/LOAD_OK.json ] && st="$st $s"; done
  [ ${LAST[$r]} -gt 0 ] && [[ " $st " != *" ${LAST[$r]} "* ]] && st="$st ${LAST[$r]}"
  STEPS[$r]=$st
  echo "[$(date '+%F %T')] $r: steps to evaluate:${st:- none} (last confirmed ${LAST[$r]})"
done
for s in 100 80 60 40 20; do for r in "$@"; do
  [[ " ${STEPS[$r]} " == *" $s "* ]] && echo "$r $s ${LAST[$r]}" >> $Q
done; done
worker() {
  local g=$1
  while true; do
    j=$(flock $Q.lock sh -c "head -n1 $Q; sed -i 1d $Q")
    [ -z "$j" ] && break
    set -- $j; r=$1; s=$2; last=$3; d=$E/$r/step$s; mkdir -p $d
    rl=""; [ "$s" = "$last" ] && rl=--relabel
    CUDA_VISIBLE_DEVICES=$g timeout 4h .venv-verl/bin/python scripts/exp/E004_eval.py --exp E007 --no-prefix-cache --run $r --step $s $rl > $d/eval.log 2>&1
    e1=$?
    if [ -f $d/s5_turn1.jsonl ] && [ -f $d/P1/samples.jsonl ] && [ -f $d/P2/samples.jsonl ] && [ -f $d/P6/samples.jsonl ]; then
      CUDA_VISIBLE_DEVICES=$g .venv-verl/bin/python scripts/exp/E004_hidden.py --exp E007 --run $r --step $s > $d/hidden.log 2>&1
      e2=$?
    else e2=skipped; fi
    echo "[$(date '+%F %T')] GPU $g $r step$s eval exit $e1 hidden exit $e2"
  done
}
for g in 4 5 6 7; do worker $g & done
wait
echo "[$(date '+%F %T')] eval done: $*"
