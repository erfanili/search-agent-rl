# L001: GRPO, Qwen2.5-3B base, 60 steps

- **Question:** What validation EM does Search-R1-recipe GRPO on Qwen2.5-3B base reach after 60 steps (64 prompts × 5) on the sub300 sets?
- **Parent:** —
- **Status:** done (2026-10-19/20)
- **Source:** converted from `notes/results_grpo_3b_60step.md` (legacy).

## Setup

- **Recipe:** Search-R1's GRPO, with batch, steps and warmup shrunk: 64 prompts × 5 samples, mini-batch 32, lr 1e-6, 10% warmup, KL loss 0.001 (low_var_kl), temperature 1, 2,048-token episodes. Reward: EM.
- **Data:** NQ + HotpotQA. 3,840 training prompts in total (0.75% of Search-R1's 514,560).
- **Validation:** sub300 sets (Bamboogle is all 125 questions), greedy decoding, computed by verl inside the training run.
- **Hardware:** GPUs 0,1,2,4,5; retrieval on GPU 3.

## Commands

```bash
TRAIN_BATCH=64 MINI_BATCH=32 TOTAL_STEPS=60 WARMUP_RATIO=0.1 TEST_FREQ=20 SAVE_FREQ=20 CKPT_KEEP=1 bash scripts/train_grpo_3b.sh full
bash scripts/eval_checkpoint.sh $DATA_ROOT/logs/grpo_qwen2.5-3b_full_20261019_183501/ckpt/global_step_60
```

## Execution

- **Commit:** a38fa39 (from the run's `config.txt`). Dirty state not recorded.
- **Run dir:** `$DATA_ROOT/logs/grpo_qwen2.5-3b_full_20261019_183501/`. Checkpoint kept: `ckpt/global_step_60` (~36 GB).
- **Wall time:** 8 h 46 min of training (60 steps, 526 s/step mean), plus four validation passes.
- **Harness re-evals of step 60:** `$DATA_ROOT/logs/eval_grpo_qwen2.5-3b_full_20261019_183501_global_step_60_20261020_104937/` (HotpotQA, 2Wiki, MuSiQue, Bamboogle) and `$DATA_ROOT/logs/eval_grpo_qwen2.5-3b_full_20261019_183501_global_step_60_20261020_103802/` (Bamboogle).

## Result

### Validation EM (%)

| step | HotpotQA | 2Wiki | MuSiQue | Bamboogle | avg, 3 multi-hop | avg, 4 |
|---|---|---|---|---|---|---|
| 0 | 8.7 | 7.7 | 2.3 | 7.2 | 6.2 | 6.5 |
| 20 | 23.7 | 16.7 | 5.0 | 14.4 | 15.1 | 14.9 |
| 40 | 24.0 | 24.7 | 5.7 | 6.4 | 18.1 | 15.2 |
| **60** | **23.7** | **29.0** | **7.0** | **4.0** | **19.9** | **15.9** |
| Search-R1-3B released, our harness (L002) | 27.3 | 26.0 | 6.3 | — | 19.9 | — |
| Search-R1-3B, paper (2503.09516v5, Table 3) | 29.7 | 27.4 | 6.6 | — | 21.2 | — |

- n = 300 per set (SE ≈ 2.6 EM); Bamboogle n = 125.
- The step rows come from verl validation (our agent loop); the Search-R1 rows from `baselines/searchr1_infer.py`. Same prompt, retriever, turn limits and scorer; different code paths.

### Step 60 through the Search-R1 baseline harness

`scripts/eval_checkpoint.sh` → `baselines/searchr1_infer.py` + `tests/score_predictions.py` (same code path as L002).

| | ours EM | Search-R1-3B EM | both right | only ours | only Search-R1 | paired sign test p | query = question verbatim, ours / SR1 |
|---|---|---|---|---|---|---|---|
| HotpotQA | 23.7 | 27.3 | 53 | 18 | 29 | 0.14 | 29% / 14% |
| 2Wiki | 28.7 | 26.0 | 54 | 32 | 24 | 0.35 | 59% / 0% |
| MuSiQue | 7.7 | 6.3 | 12 | 11 | 7 | 0.48 | 40% / 1% |
| Bamboogle | 4.8 | — | | | | | 49% / — |
| **avg, 3 multi-hop** | **20.0** | **19.9** | | | | | |

- Macro F1 over the four sets: 0.234. Searches per question: 1.00–1.02. 1,024 of 1,025 episodes ended with an answer.
- Bamboogle-only harness eval (`…_20261020_103802`): EM 4.8, F1 0.149.

### Answered rate, EM among answered, searches, F1

| | answered, step 0 → 20 → 60 | EM among answered, step 0 → 20 → 60 | searches / q, step 0 → 60 | F1, step 0 → 60 |
|---|---|---|---|---|
| HotpotQA | 0.68 → 0.99 → 1.00 | 0.127 → 0.239 → 0.237 | 0.98 → 1.01 | 0.137 → 0.326 |
| 2Wiki | 0.53 → 0.97 → 1.00 | 0.145 → 0.172 → 0.290 | 0.79 → 1.01 | 0.104 → 0.358 |
| MuSiQue | 0.64 → 1.00 → 1.00 | 0.036 → 0.050 → 0.070 | 1.02 → 1.01 | 0.071 → 0.125 |
| Bamboogle | 0.65 → 0.99 → 1.00 | 0.111 → 0.145 → 0.040 | 1.12 → 1.00 | 0.112 → 0.141 |

### Training dynamics

| step | train reward | response tokens | at 2,048 cap | turns | entropy | KL loss | step time (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 1,156 | 29% | 5.6 | 1.77 | 0.0007 | 900 |
| 10 | 0.06 | 895 | 12% | 5.1 | 2.06 | 0.003 | 694 |
| 20 | 0.17 | 556 | 0.3% | 4.0 | 1.20 | 0.035 | 546 |
| 30 | 0.30 | 535 | 0 | 4.0 | 0.74 | 0.094 | 424 |
| 40 | 0.31 | 542 | 0 | 4.0 | 0.48 | 0.128 | 475 |
| 50 | 0.31 | 534 | 0 | 4.0 | 0.28 | 0.224 | 462 |
| 60 | 0.29 | 530 | 0 | 4.0 | 0.25 | 0.242 | 545 |

`turns` counts the prompt plus model and environment segments; 4.0 = one search then an answer.

### Summary

Multi-hop average validation EM went from 6.2 (step 0) to 19.9 (step 60), n = 300 per set. Through the baseline harness, step 60 scored 23.7 / 28.7 / 7.7 on HotpotQA / 2Wiki / MuSiQue, against 27.3 / 26.0 / 6.3 for Search-R1-3B. On the same questions, the paired sign test p-values were 0.14–0.48. Searches per question were 1.00–1.02 at step 60. Bamboogle validation EM was 14.4 at step 20 and 4.0 at step 60 (n = 125).
