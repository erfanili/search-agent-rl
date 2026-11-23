#!/usr/bin/env bash
# E005 garble setting sweep: 5 conditions x 6 standalone engine launches (seeds 0-5), fresh process each, GPUs 4-7
# (one job per GPU at a time; a free-GPU queue). -> $DATA_ROOT/logs/E005/garble/sweep/<cond>_s<seed>/
cd "$(dirname "$0")/../.."
L=/data/erfan/search-agent-rl-data/logs/E005/garble/sweep; mkdir -p $L
declare -A F=([a_baseline]="" [b_noprefix]="--no-prefix-cache" [c_eager]="--eager" [d_nochunk]="--no-chunked-prefill"
              [e_alloff]="--no-prefix-cache --eager --no-chunked-prefill")
JOBS=(); for s in 0 1 2 3 4 5; do for c in a_baseline b_noprefix c_eager d_nochunk e_alloff; do JOBS+=("$c $s"); done; done
declare -A PID
i=0
while (( i < ${#JOBS[@]} )) || (( ${#PID[@]} > 0 )); do
  for g in 4 5 6 7; do
    if [[ -n ${PID[$g]:-} ]] && ! kill -0 ${PID[$g]} 2>/dev/null; then unset PID[$g]; fi
    if [[ -z ${PID[$g]:-} ]] && (( i < ${#JOBS[@]} )); then
      read c s <<< "${JOBS[$i]}"; i=$((i+1))
      CUDA_VISIBLE_DEVICES=$g .venv-verl/bin/python scripts/exp/E005_garble_temp.py --cond $c --seed $s ${F[$c]} > $L/${c}_s$s.log 2>&1 &
      PID[$g]=$!; echo "$(date '+%F %T') GPU $g start $c s$s" >> $L/sweep.log
    fi
  done
  sleep 5
done
echo "$(date '+%F %T') sweep done" >> $L/sweep.log
