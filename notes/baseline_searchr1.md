# Baseline: Search-R1 3B checkpoint (eval-harness debug run)

**Purpose:** run a published checkpoint through our whole eval pipeline (retrieval server, rollout, `tests/score_predictions.py`) and check that the numbers land near the published ones. It is a harness test, not a result we report.

- **Checkpoint:** `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` (Qwen2.5-3B base; GRPO on NQ + HotpotQA with an EM reward).
- **Retrieval:** our e5-base-v2 + IVF4096,SQ8 index, `nprobe=64`, top-3.
- **Rollout:** `baselines/searchr1_infer.py` copies Search-R1's prompt and loop:
  - greedy decoding
  - ≤ 500 tokens per turn
  - info blocks cut to 500 tokens
  - up to 4 search turns, then a final answer turn
- **Eval sets:** frozen `sub300` of HotpotQA, 2Wiki and MuSiQue (900 questions).
- **Run:** 2026-10-19 on 1 GPU, 99 s. Output in `/data/erfan/search-agent-rl-data/logs/baseline_searchr1_3b_20261019_004238/`.

## Results

Scored with the env-stripping scorer (environment text is removed before the answer is parsed). The first scoring read env text as model text; it's kept in `scores_v1_nostrip.txt`.

| dataset | EM | F1 | ACC | abort | searches / question | paper EM, Search-R1-base GRPO 3B |
|---|---|---|---|---|---|---|
| HotpotQA | 0.273 | 0.374 | 0.293 | 0.003 | 1.01 | 0.297 (−2.4) |
| 2Wiki | 0.260 | 0.332 | 0.277 | 0.000 | 1.00 | 0.274 (−1.4) |
| MuSiQue | 0.063 | 0.130 | 0.073 | 0.000 | 1.00 | 0.066 (−0.3) |
| **macro** | **0.199** | **0.279** | **0.214** | **0.001** | **1.00** | |

Paper numbers come from arXiv 2503.09516v5, Table 3 (Qwen2.5-3B base, GRPO). The HF model card has no numbers, and it isn't confirmed that this "preliminary" checkpoint is exactly the model in that table.

**Re-scoring changes (3 of 900 rows; EM unchanged):**
- **hotpotqa dev_6054:** this episode ran out of turns. The old scorer counted it as answering "and", which came from the invalid-action message. It is now an abort.
- **Search counts:** the invalid-action message contains `<search> and </search>`, which the old scorer counted as a search. Fixing this lowered the counts for dev_6054 (5 → 3), 2wiki dev_12108 (2 → 1) and musique dev_1471 (2 → 1).

**P1.2 status: not settled by this run.** All three datasets are within the 2.5-EM bar. But on 300 questions the standard error of EM is about 2.6 points, so the subset can't resolve that bar; HotpotQA at −2.4 could be noise or a real gap. Decide on the `full` dev splits (HotpotQA 7,405, 2Wiki 12,576, MuSiQue 2,417): `SPLIT=full bash scripts/run_baseline_searchr1.sh`.

## Reading

- **The harness works:** 899 of 900 episodes ended with `<answer>`, and 1 ran out of turns (now correctly an abort). Trajectories have the expected `<think>` → `<search>` → `<information>` → `<answer>` shape.
- **Close to the paper:** the EM values are in the range Search-R1 reports for its 3B models. Check the exact table before citing these as a reproduction.
- **One search per question,** even on 2- and 4-hop questions. This is a property of the EM-trained 3B checkpoint, not of the harness: it learned to answer after one search. It's the behaviour that outcome-only rewards are suspected of causing, so it's a useful reference point for Phase 3.
- **Scores are lowest where more hops are needed:** HotpotQA (2 hops) > 2Wiki (2 to 4 hops) ≫ MuSiQue (2 to 4 hops, designed to resist shortcuts).

## Re-run

```bash
bash scripts/serve_retrieval.sh                # if not already up
bash scripts/run_baseline_searchr1.sh          # LIMIT=20 for a quick check
```
