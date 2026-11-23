# L006: E1 counterfactual floor, arms A / B / C (100 steps)

- **Question:** After 100 GRPO steps, what free-depth and forced-k EM do arm A (5 free rollouts), arm B (3 free + exact:0 + exact:2, forced tags masked) and arm C (same slots, forced tags trained) reach?
- **Parent:** L004
- **Status:** done (2026-11-18), with gaps listed under Execution
- **Source:** reconstructed from `$DATA_ROOT/logs`. The runs were launched from a different repo: `~/aget-tool-use` at commit 633dfa3 (`experiments/E1_counterfactual_floor.md` there). Tables come from `~/aget-tool-use/scripts/e0_table.py` (which adds V(free) and regret) and this repo's `scripts/compare_reward_arms.py`.

## Setup

- Qwen2.5-3B base, L001 recipe (EM, 64 prompts × 5, mini 32, lr 1e-6, KL 0.001, T = 1, 2,048-token episodes, ≤ 4 searches), 100 steps, warmup ratio 0.06, validation and checkpoint every 20 steps, `BALANCE_BATCH=False`, 1 seed per arm (+ one extra seed for A).

| arm | DEPTH_POLICY (5 rollouts per group) | TRAIN_FORCED_TAGS |
|---|---|---|
| A | free | 0 |
| B | slots:free,free,free,exact:0,exact:2 | 0 (forced tag segments masked) |
| C | slots:free,free,free,exact:0,exact:2 | 1 (forced tag segments are model tokens, mask 1) |

- Validation is always free depth.
- **E0 on checkpoints:** every kept checkpoint (steps 20–100) evaluated through the baseline harness with exactly k ∈ {0,1,2,3} searches and free depth, sub300 sets, greedy. regret = max_k V(k) − V(free).
- **Hardware:** training on GPUs 4–7 (retrieval on CPU); E0 evals on GPUs 0–3.

## Commands

```bash
# in ~/aget-tool-use
bash scripts/run_e1.sh   # smoke: 2 steps per arm, TRAIN_BATCH=8 (log e1_smoke_20261118_013117.log)
ARMS="C B A" nohup bash scripts/run_e1.sh > $DATA_ROOT/logs/e1_20261118_015458.log 2>&1 &
SEED=1 ARMS=A bash scripts/run_e1.sh      # E1As1 (launch log not found)
MODELS="$(ls -d $DATA_ROOT/logs/grpo_*_E1?_*/hf/*)" KS="0 1 2 3 free" bash scripts/run_e0.sh
python scripts/e0_table.py
```

## Execution

- **Commit:** 633dfa3 in `~/aget-tool-use` for A, B, C and the smoke runs, with an 18-line `git.diff` (README) and `git_untracked.tgz` saved in each run dir. E1As1: `git: none`, empty `git.diff`.

| run | run dir (`$DATA_ROOT/logs/`) | wall time | steps |
|---|---|---|---|
| A | `grpo_qwen2.5-3b_full_E1A_em_20261118_102220` | 10:22 → ~14:07 (≈3 h 45 min) | 100 |
| B | `grpo_qwen2.5-3b_full_E1B_em_slots-fffe0e2_20261118_063613` | 06:36 → 10:22 (3 h 46 min) | 100 |
| C | `grpo_qwen2.5-3b_full_E1C_em_slots-fffe0e2_ftag_20261118_015458` | 01:55 → 06:36 (4 h 41 min) | 100 |
| A seed 1 (E1As1) | `grpo_qwen2.5-3b_full_E1As1_em_seed1_20261118_141829` | 14:18 → 17:59 (3 h 41 min) | 100 |
| smoke A/B/C | `grpo_qwen2.5-3b_full_E1{A,B,C}smoke_*_20261118_01*` | ~7 min each | 2 |

- Merged HF weights: `<run>/hf/<run>_step{20,40,60,80,100}`. FSDP shards were deleted after each arm, except A, which still has `ckpt/global_step_{20,40,60}`.
- **E0 runs:** `e0_20261118_105943/` (B and C checkpoints + Qwen2.5-3B free; 10:59 → 12:54) and `e0_20261118_141316/` (A checkpoints; 14:13 → 15:11). Outputs `eval_<run>_step<N>_exact<k>_<stamp>/` and `eval_<run>_step<N>_<stamp>/` (free).

**What went wrong**
- **Smoke gate:** `e1_queue_20261118_013636.log` failed arm B (slot_match exact:2 = 0.75 at step 2) and launched only A. Per `e1_queue_verdict.txt`, the gate counted searches from text, which miscounted B. The full queue was relaunched by hand as `ARMS="C B A"` at 01:54.
- **A, step 100:** `train.log` ends during the step-100 validation output, so there is no step-100 validation line. The first merge of step 100 (14:07) failed with `OSError: Can't load the configuration of …/global_step_100/actor/huggingface`. `hf/…_step100` exists, written at 14:12, and was used by E0.
- **C, step 40, free-depth eval failed:** `requests.exceptions.ReadTimeout` against the retrieval server So V(free) and regret for C step 40 are missing. The failed eval dir was deleted on 2026-11-20.
- **E1As1:** trained 100 steps; no E0 or free-depth evals exist for its checkpoints.

## Result

### Training-time validation EM (%), free depth, steps 0 / 20 / 40 / 60 / 80 / 100

| run | HotpotQA | 2Wiki | MuSiQue | Bamboogle |
|---|---|---|---|---|
| A | 8.3 / 22.0 / 26.3 / 24.0 / 28.3 / — | 5.7 / 16.0 / 25.0 / 24.3 / 25.3 / — | 3.3 / 5.0 / 4.7 / 5.7 / 6.7 / — | 12.0 / 11.2 / 12.0 / 9.6 / 6.4 / — |
| A seed 1 | 6.7 / 21.3 / 24.3 / 25.0 / 27.7 / 26.7 | 8.0 / 20.3 / 24.7 / 24.7 / 26.3 / 25.0 | 3.7 / 4.0 / 4.3 / 6.7 / 7.0 / 6.3 | 11.2 / 12.8 / 4.0 / 7.2 / 9.6 / 10.4 |
| B | 8.3 / 18.7 / 24.3 / 26.7 / 25.0 / 25.3 | 7.0 / 20.7 / 27.7 / 26.7 / 29.3 / 26.3 | 4.3 / 5.0 / 4.3 / 5.7 / 5.7 / 6.3 | 12.0 / 12.0 / 7.2 / 9.6 / 8.8 / 11.2 |
| C | 8.0 / 23.3 / 27.0 / 24.3 / 26.7 / 25.3 | 6.0 / 16.7 / 28.0 / 19.7 / 23.3 / 23.0 | 3.0 / 4.7 / 5.3 / 5.0 / 4.3 / 4.0 | 11.2 / 12.0 / 22.4 / 9.6 / 9.6 / 9.6 |

Multi-hop average at step 100: A seed 1 19.3, B 19.3, C 17.4 (A: no step-100 line).

### Training-time validation searches/q, steps 0 / 20 / 40 / 60 / 80 / 100

| run | HotpotQA | 2Wiki | MuSiQue | Bamboogle |
|---|---|---|---|---|
| A | 1.03 / 1.00 / 1.01 / 1.00 / 1.00 / — | 0.83 / 0.98 / 1.04 / 1.01 / 1.00 / — | 1.08 / 0.99 / 1.00 / 1.00 / 1.00 / — | 1.10 / 0.99 / 1.00 / 1.00 / 1.00 / — |
| A seed 1 | 1.01 / 1.00 / 1.00 / 1.00 / 1.01 / 1.00 | 0.78 / 0.98 / 1.03 / 1.00 / 1.07 / 1.00 | 1.11 / 0.99 / 1.00 / 1.00 / 1.01 / 1.00 | 1.08 / 0.99 / 1.00 / 1.00 / 1.00 / 1.00 |
| B | 1.03 / 1.01 / 1.04 / 1.00 / 1.02 / 1.00 | 0.70 / 1.16 / 1.07 / 1.05 / 1.06 / 1.00 | 1.06 / 0.99 / 1.00 / 1.01 / 1.01 / 1.03 | 0.93 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| C | 1.01 / 1.02 / 2.03 / 2.00 / 2.01 / 2.00 | 0.76 / 1.05 / 2.08 / 2.00 / 2.01 / 2.00 | 1.03 / 1.00 / 2.00 / 2.00 / 2.00 / 2.00 | 1.00 / 1.00 / 2.00 / 2.00 / 2.00 / 2.00 |

### E0 on checkpoints (baseline harness, sub300)

**Qwen2.5-3B base (step 0 reference; V(k) from L004, V(free) from `eval_Qwen2.5-3B_20261118_130427`)** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 9.3 | 14.3 | 17.3 | 17.3 | +5.0 ± 1.8* | +3.0 ± 1.4* | +0.0 ± 0.8 | 9.7 [0.97] | +7.7 |
| 2wiki | 300 | 7.7 | 16.3 | 13.0 | 13.7 | +8.7 ± 2.3* | -3.3 ± 2.0 | +0.7 ± 1.5 | 9.3 [0.87] | +7.0 |
| musique | 300 | 3.3 | 4.0 | 4.0 | 4.0 | +0.7 ± 0.8 | +0.0 ± 1.1 | +0.0 ± 1.1 | 2.7 [1.07] | +1.3 |
| bamboogle | 125 | 8.0 | 14.4 | 12.0 | 13.6 | +6.4 ± 2.9* | -2.4 ± 1.8 | +1.6 ± 1.6 | 9.6 [0.96] | +4.8 |


**E1A step 20** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 15.7 | 21.7 | 18.3 | 19.7 | +6.0 ± 2.2* | -3.3 ± 1.9 | +1.3 ± 1.6 | 22.3 [1.01] | -0.7 |
| 2wiki | 300 | 19.7 | 16.0 | 21.0 | 16.7 | -3.7 ± 2.6 | +5.0 ± 2.3* | -4.3 ± 1.8* | 17.7 [1.04] | +3.3 |
| musique | 300 | 2.7 | 4.7 | 5.7 | 6.7 | +2.0 ± 1.2 | +1.0 ± 1.1 | +1.0 ± 1.0 | 4.7 [1.00] | +2.0 |
| bamboogle | 125 | 8.0 | 12.0 | 12.0 | 13.6 | +4.0 ± 2.6 | +0.0 ± 1.6 | +1.6 ± 1.6 | 12.8 [1.00] | +0.8 |


**E1A step 40** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 15.3 | 28.0 | 26.3 | 25.0 | +12.7 ± 2.4* | -1.7 ± 1.6 | -1.3 ± 1.3 | 27.3 [1.01] | +0.7 |
| 2wiki | 300 | 26.7 | 25.7 | 25.7 | 24.3 | -1.0 ± 2.5 | +0.0 ± 2.1 | -1.3 ± 1.4 | 25.0 [1.06] | +1.7 |
| musique | 300 | 4.0 | 5.0 | 7.0 | 8.3 | +1.0 ± 1.3 | +2.0 ± 1.2 | +1.3 ± 0.8 | 4.3 [1.01] | +4.0 |
| bamboogle | 125 | 5.6 | 12.0 | 16.8 | 16.0 | +6.4 ± 2.7* | +4.8 ± 2.7 | -0.8 ± 1.4 | 11.2 [1.00] | +5.6 |


**E1A step 60** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 16.0 | 25.7 | 25.0 | 24.7 | +9.7 ± 2.4* | -0.7 ± 1.8 | -0.3 ± 1.3 | 25.0 [1.00] | +0.7 |
| 2wiki | 300 | 19.3 | 24.3 | 27.7 | 26.7 | +5.0 ± 2.4* | +3.3 ± 2.1 | -1.0 ± 1.7 | 24.3 [1.01] | +3.3 |
| musique | 300 | 3.7 | 5.3 | 5.0 | 5.7 | +1.7 ± 1.4 | -0.3 ± 1.1 | +0.7 ± 0.8 | 5.7 [1.00] | +0.0 |
| bamboogle | 125 | 5.6 | 11.2 | 14.4 | 16.0 | +5.6 ± 2.9 | +3.2 ± 2.3 | +1.6 ± 1.6 | 11.2 [1.00] | +4.8 |


**E1A step 80** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 15.3 | 27.3 | 30.0 | 28.0 | +12.0 ± 2.4* | +2.7 ± 1.8 | -2.0 ± 1.3 | 27.3 [1.00] | +2.7 |
| 2wiki | 300 | 26.3 | 25.3 | 23.7 | 23.0 | -1.0 ± 2.5 | -1.7 ± 2.1 | -0.7 ± 1.7 | 26.0 [1.00] | +0.3 |
| musique | 300 | 4.0 | 7.0 | 6.7 | 6.7 | +3.0 ± 1.5 | -0.3 ± 1.2 | +0.0 ± 0.5 | 6.7 [1.00] | +0.3 |
| bamboogle | 125 | 5.6 | 7.2 | 12.0 | 13.6 | +1.6 ± 2.5 | +4.8 ± 2.2* | +1.6 ± 1.6 | 7.2 [1.00] | +6.4 |


**E1A step 100** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 17.0 | 27.7 | 29.3 | 29.3 | +10.7 ± 2.5* | +1.7 ± 1.5 | +0.0 ± 0.8 | 27.7 [1.00] | +1.7 |
| 2wiki | 300 | 27.3 | 24.3 | 27.7 | 26.0 | -3.0 ± 2.5 | +3.3 ± 1.8 | -1.7 ± 1.6 | 24.0 [1.02] | +3.7 |
| musique | 300 | 4.3 | 7.3 | 7.7 | 7.7 | +3.0 ± 1.6 | +0.3 ± 1.3 | +0.0 ± 0.7 | 7.3 [1.00] | +0.3 |
| bamboogle | 125 | 5.6 | 10.4 | 12.0 | 13.6 | +4.8 ± 2.7 | +1.6 ± 2.3 | +1.6 ± 1.6 | 9.6 [1.00] | +4.0 |


**E1B step 20** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 14.7 | 20.3 | 22.3 | 21.0 | +5.7 ± 2.2* | +2.0 ± 1.7 | -1.3 ± 1.4 | 18.7 [1.06] | +3.7 |
| 2wiki | 300 | 22.0 | 20.0 | 22.0 | 20.3 | -2.0 ± 2.5 | +2.0 ± 2.4 | -1.7 ± 1.9 | 21.3 [1.24] | +0.7 |
| musique | 300 | 1.3 | 4.0 | 5.3 | 4.7 | +2.7 ± 1.1* | +1.3 ± 0.8 | -0.7 ± 0.9 | 3.3 [1.02] | +2.0 |
| bamboogle | 125 | 7.2 | 10.4 | 16.0 | 15.2 | +3.2 ± 2.8 | +5.6 ± 2.4* | -0.8 ± 1.4 | 12.8 [1.00] | +3.2 |


**E1B step 40** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 14.7 | 24.0 | 24.3 | 24.0 | +9.3 ± 2.4* | +0.3 ± 1.7 | -0.3 ± 1.3 | 23.0 [1.03] | +1.3 |
| 2wiki | 300 | 24.7 | 26.3 | 25.3 | 26.0 | +1.7 ± 2.4 | -1.0 ± 2.2 | +0.7 ± 1.7 | 24.3 [1.10] | +2.0 |
| musique | 300 | 4.3 | 5.0 | 7.7 | 7.3 | +0.7 ± 1.3 | +2.7 ± 1.2* | -0.3 ± 1.0 | 4.3 [1.01] | +3.3 |
| bamboogle | 125 | 4.0 | 7.2 | 17.6 | 19.2 | +3.2 ± 2.5 | +10.4 ± 3.2* | +1.6 ± 2.5 | 7.2 [1.00] | +12.0 |


**E1B step 60** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 13.7 | 25.3 | 28.0 | 26.0 | +11.7 ± 2.4* | +2.7 ± 1.5 | -2.0 ± 1.2 | 26.0 [1.01] | +2.0 |
| 2wiki | 300 | 28.7 | 26.3 | 25.7 | 25.7 | -2.3 ± 2.2 | -0.7 ± 2.2 | +0.0 ± 1.6 | 26.3 [1.04] | +2.3 |
| musique | 300 | 2.3 | 5.3 | 8.0 | 7.3 | +3.0 ± 1.3* | +2.7 ± 1.1* | -0.7 ± 0.9 | 5.3 [1.01] | +2.7 |
| bamboogle | 125 | 2.4 | 8.8 | 16.8 | 16.0 | +6.4 ± 2.7* | +8.0 ± 2.7* | -0.8 ± 1.8 | 9.6 [1.00] | +7.2 |


**E1B step 80** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 16.3 | 25.0 | 28.7 | 29.3 | +8.7 ± 2.2* | +3.7 ± 1.8 | +0.7 ± 1.1 | 25.3 [1.04] | +4.0 |
| 2wiki | 300 | 28.3 | 27.7 | 28.7 | 29.7 | -0.7 ± 2.3 | +1.0 ± 2.1 | +1.0 ± 1.9 | 29.0 [1.07] | +0.7 |
| musique | 300 | 1.7 | 6.0 | 6.0 | 6.0 | +4.3 ± 1.4* | +0.0 ± 1.1 | +0.0 ± 0.9 | 6.0 [1.00] | +0.0 |
| bamboogle | 125 | 2.4 | 8.8 | 14.4 | 17.6 | +6.4 ± 2.7* | +5.6 ± 2.1* | +3.2 ± 1.9 | 8.8 [1.00] | +8.8 |


**E1B step 100** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 15.3 | 23.7 | 26.7 | 27.3 | +8.3 ± 2.3* | +3.0 ± 1.8 | +0.7 ± 1.3 | 25.0 [1.00] | +2.3 |
| 2wiki | 300 | 21.7 | 26.7 | 31.0 | 31.7 | +5.0 ± 2.4* | +4.3 ± 2.2 | +0.7 ± 1.7 | 26.3 [1.01] | +5.3 |
| musique | 300 | 2.0 | 6.0 | 6.7 | 8.0 | +4.0 ± 1.4* | +0.7 ± 1.3 | +1.3 ± 1.1 | 6.7 [1.01] | +1.3 |
| bamboogle | 125 | 4.0 | 11.2 | 12.0 | 16.0 | +7.2 ± 2.8* | +0.8 ± 2.1 | +4.0 ± 2.4 | 9.6 [1.01] | +6.4 |


**E1C step 20** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 15.0 | 22.0 | 21.3 | 22.3 | +7.0 ± 2.3* | -0.7 ± 1.7 | +1.0 ± 1.2 | 21.3 [1.05] | +1.0 |
| 2wiki | 300 | 20.3 | 18.7 | 20.7 | 17.7 | -1.7 ± 2.7 | +2.0 ± 2.3 | -3.0 ± 1.9 | 16.3 [1.22] | +4.3 |
| musique | 300 | 2.7 | 3.7 | 5.0 | 5.3 | +1.0 ± 1.2 | +1.3 ± 1.1 | +0.3 ± 0.3 | 4.0 [1.03] | +1.3 |
| bamboogle | 125 | 8.0 | 12.0 | 11.2 | 11.2 | +4.0 ± 2.6 | -0.8 ± 1.4 | +0.0 ± 1.6 | 11.2 [1.02] | +0.8 |


**E1C step 40** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 15.3 | 26.0 | 26.0 | 24.7 | +10.7 ± 2.3* | +0.0 ± 1.5 | -1.3 ± 1.5 |
| 2wiki | 300 | 24.3 | 23.3 | 27.0 | 26.3 | -1.0 ± 2.4 | +3.7 ± 2.2 | -0.7 ± 2.1 |
| musique | 300 | 2.3 | 5.0 | 5.0 | 5.0 | +2.7 ± 1.1* | +0.0 ± 0.9 | +0.0 ± 0.8 |
| bamboogle | 125 | 12.0 | 17.6 | 22.4 | 20.8 | +5.6 ± 2.6* | +4.8 ± 1.9* | -1.6 ± 2.3 |


**E1C step 60** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 13.7 | 23.3 | 24.7 | 24.3 | +9.7 ± 2.4* | +1.3 ± 1.6 | -0.3 ± 1.1 | 23.7 [2.08] | +1.0 |
| 2wiki | 300 | 27.0 | 22.3 | 21.3 | 19.0 | -4.7 ± 2.4 | -1.0 ± 2.2 | -2.3 ± 1.6 | 20.7 [2.01] | +6.3 |
| musique | 300 | 2.0 | 4.7 | 6.0 | 5.7 | +2.7 ± 1.2* | +1.3 ± 1.2 | -0.3 ± 0.7 | 5.3 [2.04] | +0.7 |
| bamboogle | 125 | 4.0 | 4.8 | 7.2 | 7.2 | +0.8 ± 2.1 | +2.4 ± 1.8 | +0.0 ± 2.0 | 7.2 [2.02] | +0.0 |


**E1C step 80** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 15.0 | 23.3 | 25.7 | 27.7 | +8.3 ± 2.3* | +2.3 ± 1.5 | +2.0 ± 0.9* | 25.0 [2.02] | +2.7 |
| 2wiki | 300 | 24.3 | 24.7 | 24.0 | 23.3 | +0.3 ± 2.4 | -0.7 ± 1.9 | -0.7 ± 1.3 | 24.3 [2.03] | +0.3 |
| musique | 300 | 2.3 | 3.3 | 4.3 | 5.7 | +1.0 ± 1.2 | +1.0 ± 0.7 | +1.3 ± 0.8 | 4.0 [2.01] | +1.7 |
| bamboogle | 125 | 4.0 | 8.0 | 10.4 | 9.6 | +4.0 ± 2.6 | +2.4 ± 2.1 | -0.8 ± 1.4 | 9.6 [2.02] | +0.8 |

**E1C step 100** (EM ×100; paired diff ± SE, * = beyond 2 SE)

| set | n | V(0) | V(1) | V(2) | V(3) | V(1)−V(0) | V(2)−V(1) | V(3)−V(2) | V(free) [searches/q] | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 300 | 12.7 | 25.7 | 25.7 | 25.7 | +13.0 ± 2.4* | +0.0 ± 1.6 | +0.0 ± 1.2 | 26.0 [2.01] | -0.3 |
| 2wiki | 300 | 24.0 | 24.3 | 24.0 | 25.0 | +0.3 ± 2.2 | -0.3 ± 1.9 | +1.0 ± 1.1 | 24.7 [2.02] | +0.3 |
| musique | 300 | 3.0 | 4.0 | 4.3 | 5.0 | +1.0 ± 1.3 | +0.3 ± 1.0 | +0.7 ± 0.8 | 4.7 [2.00] | +0.3 |
| bamboogle | 125 | 4.8 | 7.2 | 10.4 | 8.0 | +2.4 ± 2.7 | +3.2 ± 1.9 | -2.4 ± 1.8 | 9.6 [2.02] | +0.8 |


### Summary

n = 300 per set (Bamboogle 125), one seed per arm for E0. Free-depth searches/q in E0: A and B checkpoints 1.00–1.24 at every step. C is 1.02–1.22 at step 20 and 2.00–2.08 at steps 60–100 (step 40 missing). Training-time validation searches/q for C is 2.00–2.08 from step 40 on. Macro regret (mean over the 4 sets) at step 100: A 2.4, B 3.8, C 0.3. Qwen2.5-3B base: 5.2. Training-time step-100 multi-hop validation EM: A seed 1 19.3, B 19.3, C 17.4.
