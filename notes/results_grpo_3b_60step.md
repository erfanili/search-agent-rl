# Results: GRPO, Qwen2.5-3B base, 60 steps (2026-10-19/20)

**Run:** `/data/erfan/search-agent-rl-data/logs/grpo_qwen2.5-3b_full_20261019_183501/`. It finished in 8 h 46 min of training (60 steps, 526 s/step on average), plus four validation passes. The one checkpoint kept is `ckpt/global_step_60`, about 36 GB.

**Command:**
```bash
TRAIN_BATCH=64 MINI_BATCH=32 TOTAL_STEPS=60 WARMUP_RATIO=0.1 TEST_FREQ=20 SAVE_FREQ=20 CKPT_KEEP=1 bash scripts/train_grpo_3b.sh full
```

**Setup:**
- **Recipe:** Search-R1's GRPO, with batch, steps and warmup shrunk to fit one day: 64 prompts × 5 samples, mini-batch 32, lr 1e-6, 10% warmup, KL loss 0.001 (low_var_kl), temperature 1, 2,048-token episodes.
- **Data:** NQ + HotpotQA. 3,840 training prompts in total, which is 0.75% of Search-R1's 514,560.
- **Validation:** the sub300 sets (Bamboogle is all 125 questions), greedy decoding, computed by verl inside the training run.

## Validation EM (%)

| step | HotpotQA | 2Wiki | MuSiQue | Bamboogle | avg, 3 multi-hop | avg, 4 |
|---|---|---|---|---|---|---|
| 0 | 8.7 | 7.7 | 2.3 | 7.2 | 6.2 | 6.5 |
| 20 | 23.7 | 16.7 | 5.0 | 14.4 | 15.1 | 14.9 |
| 40 | 24.0 | 24.7 | 5.7 | 6.4 | 18.1 | 15.2 |
| **60** | **23.7** | **29.0** | **7.0** | **4.0** | **19.9** | **15.9** |
| Search-R1-3B released, our harness (P1.2) | 27.3 | 26.0 | 6.3 | — | 19.9 | — |
| Search-R1-3B, paper (2503.09516v5, Table 3) | 29.7 | 27.4 | 6.6 | — | 21.2 | — |

Two things to keep in mind when reading this table:
- **Noise.** At n = 300 the standard error is about 2.6 EM per set, and more for Bamboogle (n = 125). This run's step 0 differs from the pilot's step 0 by up to 1.7 EM on the multi-hop sets and 4.8 on Bamboogle. Batched vLLM decoding isn't bit-for-bit deterministic, so differences of that size are noise.
- **Harness.** The Search-R1 rows come from `baselines/searchr1_infer.py`. The step rows come from verl validation, which runs our agent loop. Both use the same prompt, retriever, turn limits and scorer, but they are different code paths. `scripts/eval_checkpoint.sh` re-evaluates step 60 through the baseline harness so the comparison is exact.

## Step 60 through the Search-R1 baseline harness (exact comparison)

`scripts/eval_checkpoint.sh` ran the merged step-60 checkpoint through `baselines/searchr1_infer.py` and `tests/score_predictions.py`, the same code path as the Search-R1 row. Output: `/data/erfan/search-agent-rl-data/logs/eval_grpo_qwen2.5-3b_full_20261019_183501_global_step_60_20261020_104937/`.

| | ours EM | Search-R1-3B EM | both right | only ours | only Search-R1 | paired sign test p | query = question verbatim, ours / SR1 |
|---|---|---|---|---|---|---|---|
| HotpotQA | 23.7 | 27.3 | 53 | 18 | 29 | 0.14 | 29% / 14% |
| 2Wiki | 28.7 | 26.0 | 54 | 32 | 24 | 0.35 | 59% / 0% |
| MuSiQue | 7.7 | 6.3 | 12 | 11 | 7 | 0.48 | 40% / 1% |
| Bamboogle | 4.8 | — | | | | | 49% / — |
| **avg, 3 multi-hop** | **20.0** | **19.9** | | | | | |

- **The harnesses agree.** Through the baseline harness, step 60 scores 23.7, 28.7 and 7.7 on HotpotQA, 2Wiki and MuSiQue. verl's validation gave 23.7, 29.0 and 7.0. Macro F1 over all four sets is 0.234; searches per question are 1.00–1.02, and 1,024 of 1,025 episodes ended with an answer.
- **The match holds question by question.** On identical questions, the paired sign test finds no significant difference from Search-R1-3B on any set (p = 0.14–0.48). The two models get different questions right, though: on 2Wiki, 32 questions only we answer and 24 only Search-R1 answers. Our checkpoint is a different solution of similar quality, not a copy.
- **Same search count, different queries.** Both models search once. Ours pastes the question as the query far more often: 59% of the time on 2Wiki, where Search-R1 never does. Search-R1 rewrites the question into a query, which is the one query-level skill its longer training seems to have added.

## Format versus search

| | answered, step 0 → 20 → 60 | EM among answered, step 0 → 20 → 60 | searches / q, step 0 → 60 | F1, step 0 → 60 |
|---|---|---|---|---|
| HotpotQA | 0.68 → 0.99 → 1.00 | 0.127 → 0.239 → 0.237 | 0.98 → 1.01 | 0.137 → 0.326 |
| 2Wiki | 0.53 → 0.97 → 1.00 | 0.145 → 0.172 → 0.290 | 0.79 → 1.01 | 0.104 → 0.358 |
| MuSiQue | 0.64 → 1.00 → 1.00 | 0.036 → 0.050 → 0.070 | 1.02 → 1.01 | 0.071 → 0.125 |
| Bamboogle | 0.65 → 0.99 → 1.00 | 0.111 → 0.145 → 0.040 | 1.12 → 1.00 | 0.112 → 0.141 |

## Training dynamics

| step | train reward | response tokens | at 2,048 cap | turns | entropy | KL loss | step time (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 1,156 | 29% | 5.6 | 1.77 | 0.0007 | 900 |
| 10 | 0.06 | 895 | 12% | 5.1 | 2.06 | 0.003 | 694 |
| 20 | 0.17 | 556 | 0.3% | 4.0 | 1.20 | 0.035 | 546 |
| 30 | 0.30 | 535 | 0 | 4.0 | 0.74 | 0.094 | 424 |
| 40 | 0.31 | 542 | 0 | 4.0 | 0.48 | 0.128 | 475 |
| 50 | 0.31 | 534 | 0 | 4.0 | 0.28 | 0.224 | 462 |
| 60 | 0.29 | 530 | 0 | 4.0 | 0.25 | 0.242 | 545 |

`turns` counts the prompt plus the model and environment segments, so 4.0 means one search and then an answer.

## Findings

1. **The multi-hop average matches the released Search-R1-3B with under 1% of its training prompts:** 6.2 → 19.9 EM, against 19.9 for Search-R1. By set, HotpotQA is 3.6 points below it, while 2Wiki is 3.0 above and MuSiQue 0.7 above. All of these differences are within about 1.4 standard errors.
2. **Format is learned by step 20, and some of the gain goes beyond format.** The answered rate saturates at 97–100% by step 20. After that, EM keeps rising on 2Wiki (16.7 → 29.0) and MuSiQue (5.0 → 7.0), while HotpotQA plateaus at about 24. EM among answered episodes roughly doubles on HotpotQA, 2Wiki and MuSiQue.
3. **The agent did not learn multi-hop search.** It makes 1.0 search per question on every set at every checkpoint, including 4-hop MuSiQue, and the episode structure collapses to a single search followed by an answer (turns = 4.0 from step 20 on). Policy entropy falls from 1.8 to 0.25, and the KL to the reference policy grows steadily. The released Search-R1 3B lands at the same single-search policy (1.00 searches per question) and the same multi-hop EM. **Outcome-reward GRPO at 3B converges to "search once, answer", whether it gets 0.75% or 100% of the compute.**
4. **Bamboogle regressed after step 20:** 14.4 → 6.4 → 4.0, below step 0. Step 60 re-evaluated through the Search-R1 baseline harness gives 4.8 EM and 0.149 F1, consistent with verl's validation (`logs/eval_grpo_qwen2.5-3b_full_20261019_183501_global_step_60_20261020_103802/`). Analysing the misses rules out the phrasing-drift guess:
   - **The answers are short and well formed:** 1.7 words on average, the same as the correct ones. The 26 "near misses" are wrong entities that share a token with the gold answer (James Monroe vs. James Madison, Augustine vs. Lawrence Washington, a different date in the right year). They are wrong facts, not formatting problems.
   - **The policy is effectively single-shot RAG:** one search per episode in all 125. The query is the question verbatim in 60/125 episodes (48%), and 66% of queries reuse at least 80% of the question's words.
   - **It plans the hops but doesn't execute them:** in 17 episodes (14%) the `<think>` spells out a multi-step plan ("first find the year Citibank was founded, then the president in that year"). It then issues one combined query and answers straight from that one result. EM on those 17 is 0.
   - **So:** Bamboogle's compositional 2-hop questions need a second, dependent query, which this policy never makes. It guesses the second hop from the first retrieval. Why step 20 scored 14.4 is unknown, because that checkpoint wasn't kept; entropy was still 1.2 then, so its policy was less collapsed.
5. **Throughput improved as the policy converged.** Shorter episodes and no length-capped episodes cut step time from 900 s to about 450 s. Most of the early cost was the update pass over long, mostly environment-text sequences.
