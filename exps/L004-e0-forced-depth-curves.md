# L004: E0 forced-depth curves V(k), k = 0..3

- **Question:** What EM does each of Qwen2.5-3B base, Search-R1-3B and the L001 step-60 checkpoint reach when exactly k ∈ {0, 1, 2, 3} searches are forced?
- **Parent:** L001, L002
- **Status:** done (2026-11-11)
- **Source:** reconstructed from `$DATA_ROOT/logs` (`e0_launch_20261111_220211.log`, `e0_20261111_220211/`, `eval_*_exact<k>_20261111_*`) and `scripts/e0_table.py`.

## Setup

- **Models:** `Qwen/Qwen2.5-3B`; `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`; `$DATA_ROOT/logs/grpo_qwen2.5-3b_full_20261019_183501/ckpt/global_step_60` (L001).
- **Forcing:** `FORCE_DEPTH=k` in `scripts/eval_checkpoint.sh` (exactly k executed searches per episode). Baseline harness (`baselines/searchr1_infer.py`), greedy.
- **Eval sets:** sub300 HotpotQA, 2Wiki, MuSiQue (n = 300) and Bamboogle (n = 125).
- **Hardware:** GPUs 4, 5, 6, 7, one k per GPU, models in sequence.
- **Metric:** EM ×100. V(k+1) − V(k) is the paired difference on the same questions ± SE.

## Commands

```bash
nohup bash scripts/run_e0.sh > $DATA_ROOT/logs/e0_launch_<stamp>.log 2>&1 &    # KS="0 1 2 3", default MODELS
.venv/bin/python scripts/e0_table.py
```

## Execution

- **Commit:** not recorded in the run outputs. Branch HEAD at launch time was ccd9ff6; `scripts/run_e0.sh` and `scripts/e0_table.py` were first committed in 3fc3d70 (2026-11-17), so the run used uncommitted code.
- **Run IDs / paths:** `$DATA_ROOT/logs/eval_<model tag>_exact<k>_<stamp>/` (`scores.txt`, `preds.jsonl`, `scored/per_example.jsonl`); per-GPU logs in `$DATA_ROOT/logs/e0_20261111_220211/k{0..3}.log`.
- **Wall time:** 2026-11-11 22:02 → 22:40 (38 min; k=3 finished last).
- **Forcing check:** executed searches/q equals k on every (model, set, k) cell.

## Result

**Qwen2.5-3B** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 9.3 | 14.3 | 17.3 | 17.3 | +5.0 ± 1.8* | +3.0 ± 1.4* | +0.0 ± 0.8 |
| 2wiki | 300 | 7.7 | 16.3 | 13.0 | 13.7 | +8.7 ± 2.3* | -3.3 ± 2.0 | +0.7 ± 1.5 |
| musique | 300 | 3.3 | 4.0 | 4.0 | 4.0 | +0.7 ± 0.8 | +0.0 ± 1.1 | +0.0 ± 1.1 |
| bamboogle | 125 | 8.0 | 14.4 | 12.0 | 13.6 | +6.4 ± 2.9* | -2.4 ± 1.8 | +1.6 ± 1.6 |

**Search-R1-3B** (`SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`) (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 14.3 | 27.3 | 28.3 | 30.3 | +13.0 ± 2.5* | +1.0 ± 1.1 | +2.0 ± 1.2 |
| 2wiki | 300 | 24.0 | 25.7 | 28.3 | 26.7 | +1.7 ± 2.2 | +2.7 ± 1.5 | -1.7 ± 1.2 |
| musique | 300 | 2.7 | 6.7 | 6.3 | 7.7 | +4.0 ± 1.4* | -0.3 ± 1.2 | +1.3 ± 0.9 |
| bamboogle | 125 | 11.2 | 10.4 | 12.0 | 11.2 | -0.8 ± 2.9 | +1.6 ± 2.0 | -0.8 ± 1.8 |

**L001 step 60** (`grpo_qwen2.5-3b_full_20261019_183501_global_step_60`) (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 15.3 | 23.3 | 26.3 | 26.7 | +8.0 ± 2.4* | +3.0 ± 1.7 | +0.3 ± 1.5 |
| 2wiki | 300 | 27.7 | 29.0 | 29.7 | 24.3 | +1.3 ± 2.3 | +0.7 ± 2.2 | -5.3 ± 2.0* |
| musique | 300 | 3.0 | 7.3 | 7.7 | 8.7 | +4.3 ± 1.4* | +0.3 ± 1.1 | +1.0 ± 0.7 |
| bamboogle | 125 | 6.4 | 4.8 | 13.6 | 16.0 | -1.6 ± 2.5 | +8.8 ± 2.5* | +2.4 ± 2.4 |

### Summary

All 12 cells per model have executed searches/q equal to k. Paired differences beyond 2 SE:
- **Qwen2.5-3B:** V(1)−V(0) on HotpotQA (+5.0 ± 1.8), 2Wiki (+8.7 ± 2.3) and Bamboogle (+6.4 ± 2.9); V(2)−V(1) on HotpotQA (+3.0 ± 1.4).
- **Search-R1-3B:** V(1)−V(0) on HotpotQA (+13.0 ± 2.5) and MuSiQue (+4.0 ± 1.4).
- **L001 step 60:** V(1)−V(0) on HotpotQA (+8.0 ± 2.4) and MuSiQue (+4.3 ± 1.4); V(2)−V(1) on Bamboogle (+8.8 ± 2.5); V(3)−V(2) on 2Wiki (−5.3 ± 2.0).

Free-depth EM for these checkpoints: Search-R1-3B in L002; L001 step 60 in L001; Qwen2.5-3B base in L006.
