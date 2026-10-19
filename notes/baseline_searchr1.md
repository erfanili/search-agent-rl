# Baseline: Search-R1 3B checkpoint (eval-harness debug run)

**Purpose:** run a published checkpoint through our whole eval pipeline (retrieval server, rollout, `tests/score_predictions.py`) and check that the numbers land near the published ones. It is a harness test, not a result we report.

- **Checkpoint:** `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` (Qwen2.5-3B base; GRPO on NQ + HotpotQA with an EM reward).
- **Retrieval:** our e5-base-v2 + IVF4096,SQ8 index, `nprobe=64`, top-3 (D-05).
- **Rollout:** `baselines/searchr1_infer.py` copies Search-R1's prompt and loop:
  - greedy decoding
  - ≤ 500 tokens per turn
  - info blocks cut to 500 tokens
  - up to 4 search turns, then a final answer turn
- **Eval sets:** frozen `sub300` of HotpotQA, 2Wiki and MuSiQue (900 questions).
- **Run:** 2026-10-19 on 1 GPU, 99 s. Output in `/data/erfan/logs/baseline_searchr1_3b_20261019_004238/`.

## Results

| dataset | EM | F1 | ACC | abort | searches / question |
|---|---|---|---|---|---|
| HotpotQA | 0.273 | 0.374 | 0.297 | 0.000 | 1.01 |
| 2Wiki | 0.260 | 0.332 | 0.277 | 0.000 | 1.00 |
| MuSiQue | 0.063 | 0.130 | 0.073 | 0.000 | 1.00 |
| **macro** | **0.199** | **0.279** | **0.216** | | |

## Reading

- **The harness works:** every question got an answer (899 with `<answer>`, 1 ran out of turns). Trajectories have the expected `<think>` → `<search>` → `<information>` → `<answer>` shape.
- **Close to the paper:** the EM values are in the range Search-R1 reports for its 3B models. Check the exact table before citing these as a reproduction.
- **One search per question,** even on 2- and 4-hop questions. This is a property of the EM-trained 3B checkpoint, not of the harness: it learned to answer after one search. It's the behaviour that outcome-only rewards are suspected of causing, so it's a useful reference point for Phase 3.
- **Scores are lowest where more hops are needed:** HotpotQA (2 hops) > 2Wiki (2 to 4 hops) ≫ MuSiQue (2 to 4 hops, designed to resist shortcuts).

## Re-run

```bash
bash scripts/serve_retrieval.sh                # if not already up
bash scripts/run_baseline_searchr1.sh          # LIMIT=20 for a quick check
```
