# L002: Search-R1 3B checkpoint through our eval harness

- **Question:** What EM does the released Search-R1-3B checkpoint score through our eval harness on the sub300 multi-hop sets?
- **Parent:** —
- **Status:** done (2026-10-19)
- **Source:** converted from `notes/baseline_searchr1.md` (legacy).

## Setup

- **Checkpoint:** `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` (Qwen2.5-3B base; GRPO on NQ + HotpotQA with an EM reward).
- **Retrieval:** e5-base-v2 + IVF4096,SQ8 index, `nprobe=64`, top-3.
- **Rollout:** `baselines/searchr1_infer.py`, a copy of Search-R1's prompt and loop: greedy decoding, ≤ 500 tokens per turn, info blocks cut to 500 tokens, up to 4 search turns, then a final answer turn.
- **Eval sets:** frozen `sub300` of HotpotQA, 2Wiki and MuSiQue (900 questions).
- **Scorer:** `tests/score_predictions.py`, env-stripping (environment text removed before the answer is parsed). The first scoring, without stripping, is kept in `scores_v1_nostrip.txt`.

## Commands

```bash
bash scripts/serve_retrieval.sh                # if not already up
bash scripts/run_baseline_searchr1.sh          # LIMIT=20 for a quick check; SPLIT=full for full dev splits
```

## Execution

- **Commit:** not recorded.
- **Run dir:** `$DATA_ROOT/logs/baseline_searchr1_3b_20261019_004238/`
- **Hardware / wall time:** 1 GPU, 99 s.

## Result

| dataset | EM | F1 | ACC | abort | searches / question | paper EM, Search-R1-base GRPO 3B |
|---|---|---|---|---|---|---|
| HotpotQA | 0.273 | 0.374 | 0.293 | 0.003 | 1.01 | 0.297 (−2.4) |
| 2Wiki | 0.260 | 0.332 | 0.277 | 0.000 | 1.00 | 0.274 (−1.4) |
| MuSiQue | 0.063 | 0.130 | 0.073 | 0.000 | 1.00 | 0.066 (−0.3) |
| **macro** | **0.199** | **0.279** | **0.214** | **0.001** | **1.00** | |

- Paper numbers: arXiv 2503.09516v5, Table 3 (Qwen2.5-3B base, GRPO). The HF model card has no numbers. It is not confirmed that this checkpoint is the model in that table.
- n = 300 per set; SE of EM ≈ 2.6 points.
- 899 of 900 episodes ended with `<answer>`; 1 ran out of turns (scored as an abort).

**Re-scoring changes (3 of 900 rows; EM unchanged):**
- hotpotqa dev_6054 ran out of turns. The old scorer read it as answering "and", which came from the invalid-action message. It is now an abort.
- The invalid-action message contains `<search> and </search>`, which the old scorer counted as a search. Fixing this lowered search counts for dev_6054 (5 → 3), 2wiki dev_12108 (2 → 1) and musique dev_1471 (2 → 1).

### Summary

EM 0.273 / 0.260 / 0.063 on HotpotQA / 2Wiki / MuSiQue (n = 300 each, SE ≈ 0.026), against paper values that are 2.4, 1.4 and 0.3 points higher. Searches per question were 1.00–1.01. Full dev splits were not run.
