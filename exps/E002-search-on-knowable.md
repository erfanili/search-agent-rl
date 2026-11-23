# E002: What makes the model search on questions it can answer from memory?
Question: Across models and prompts, how often does the agent search on knowable vs unknowable questions (E001 labels), and does its search decision separate the two?
Parent: E001

## Setup
- Models: the same 5 as E001 (qwen3b-base, qwen3b-instruct, searchr1-3b, grpo-e1a-s60, qwen7b-instruct). Same checkpoints, same tags.
- Data: data/e001_easy_1k.jsonl (sha256 1c45efef…), all 1,000 questions, with E001 labels and the ambiguous flag.
- Agent loop: our existing search rollout (agent/search_r1_rollout.py), with the retrieval server as in L001 (e5-base-v2, wiki-18, top-3). At most 3 searches per episode, then the answer is forced. Instruct models use their chat template.
- Prompts: 6 conditions = instruction variant × demo block. Save the exact strings in scripts/exp/E002_prompts.py.
  Instruction variants:
   I-SR1: Search-R1's original instruction text, verbatim.
   I-NEU: neutral. Describes the search tool and the format only. No advice on when to search.
   I-CONF: I-NEU + "If you are confident you know the answer, answer directly. Search only when you need information you do not know."
   I-COST: I-NEU + "Each search is slow and costly. Use it only when necessary."
  Demo blocks (2 demos, questions not in PopQA or TriviaQA; all demos use the identical think/search/answer format):
   D-MIX: one demo that answers directly, one that searches once and then answers.
   D-SRCH: both demos search once.
   D-DIR: both demos answer directly.
  Conditions: P1 = I-SR1 + D-MIX, P2 = I-NEU + D-MIX, P3 = I-CONF + D-MIX, P4 = I-COST + D-MIX, P5 = I-NEU + D-SRCH, P6 = I-NEU + D-DIR.
- Decoding: temperature 0.7, top_p 1.0, 5 samples per question, max 512 new tokens per turn, stop at </search> or </answer> as the rollout already does.
- Scoring: the same parse rule and correctness rule as E001.
- Forced-depth reference: for P2 only, all 5 models, also run FORCE_DEPTH=0 and FORCE_DEPTH=1 (5 samples each).

## Pilot first
Run 50 questions (25 PopQA + 25 TriviaQA, the same as E001 pilot 2) × 5 models × P1, P2 and P6. Report the parse rate and the mean number of searches. Stop if any 3B model parses below 95%, or if a condition shows malformed search calls (e.g. a search tag with an empty query) above 5%. Show 2 examples if so.

## Output wanted
For each model × condition (30 cells), computed on knowable vs unknowable questions with ambiguous questions excluded, and again on all questions:
1. Table A: P(≥1 search | knowable), P(≥1 search | unknowable), P(≥1 search | uncertain), mean searches per episode, parse rate.
2. Table B: AUROC of the per-question search rate (out of 5 samples) for knowable vs unknowable, with a bootstrap 95% CI (1,000 resamples, seed 0).
3. Table C: accuracy overall, on knowable, and on unknowable (± SE).
4. Table D (P2 forced depth): accuracy at forced 0 vs forced 1 searches, on knowable and on unknowable, per model, with a paired sign test on knowable.
5. Table E: Table A's knowable column again, restricted to the shared knowable set (knowable for all four 3B models), for the 3B models.
6. Files: $DATA_ROOT/logs/E002/<model tag>/<condition>/samples.jsonl, plus one tidy CSV with per-question, per-cell search rate and correctness.

## Execution   ← you fill

### Pre-run checks (2026-11-21)
- **Data:** `$DATA_ROOT/eval/e001/e001_easy_1k.jsonl` (sha256 `1c45efef…`; card path `data/e001_easy_1k.jsonl`). E001 labels and the ambiguous flag are in `$DATA_ROOT/eval/e001/e001_easy_1k_labels.jsonl`.
- **Checkpoints:** all 5 E001 checkpoints exist (paths in the E001 card).
- **Retrieval:** a server is already running on :8000 (pid 1133585, started 2026-10-30 by erfan): `$DATA_ROOT/index_e5_IVF4096_SQ8`, e5 encoder on CPU, nprobe 64, top-3, 21,015,300 passages. This is the same index as L001. Measured throughput: ~69 qps at concurrency 32 (P0.6 benchmark).
- **GPUs:** 4–7 are free; 0–3 are in use by another user (~17.6 GB each).
- **Rollout:** `agent/search_r1_rollout.run_episode` can be driven from `scripts/exp/E002_run.py` with vLLM and the HTTP retriever, without changing shared code. Settings:
  - the 512-token per-turn cap goes in the script's `generate`;
  - `max_turns=3`;
  - FORCE_DEPTH=0/1 = `depth_policy` `exact:0` / `exact:1`.
- **Prompts:** drafted in `scripts/exp/E002_prompts.py` and awaiting review.
- **Decisions (Erfan, 2026-11-21):**
  - **Forced answer:** a new depth policy `max:k` (free until k searches, then the environment writes `<answer>`), used at `max:3` with `max_turns=3`. It is in worktree `../sar-max-depth`, branch `feat/max-depth`, uncommitted: changes to `agent/search_r1_rollout.py`, `baselines/searchr1_infer.py` (help/docstring) and `tests/test_rollout_contract.py` (2 new tests plus updated parse assertions). 23/23 rollout-contract tests pass. Waiting for Erfan's merge before the pilot.
  - **Labels:** each model's own E001 labels; the shared 3B knowable set is used for Table E only.
  - **Malformed searches:** counted separately as empty query, `<search>` never closed, >1 `<search>` in one turn, and `<search>` + `<answer>` in one turn.
  - **Length cap:** 4,096-token total response cap kept. The cap-hit rate is reported per cell; the pilot stops if it exceeds 2% for any 3B model.
  - **Prompts:** revised to the review criteria (same 2 easy demo questions in every block, matched think length, I-CONF and I-COST = I-NEU + one sentence). Awaiting Erfan's review before the pilot.
### Merge (2026-11-21)
At Erfan's request ("Merge it. I can't"), I committed `feat/max-depth` in `../sar-max-depth` (359ea18) and fast-forwarded local `main` to it. Not pushed. 23/23 rollout-contract tests pass on main.

### Pilot (2026-11-21)
- 50 questions (the E001 pilot-2 subset: 13 PopQA bin 4, 12 PopQA bin 5, 25 TriviaQA) × 5 models × P1, P2, P6 × 5 samples.
- `bash scripts/exp/E002_launch.sh pilot` (log `$DATA_ROOT/logs/E002/pilot/launch.log`). GPUs 4–7, 2026-11-21 01:25–01:35, 55–154 s per cell. Code at commit 359ea18 + uncommitted `scripts/exp/E002_*`.
- Outputs: `$DATA_ROOT/logs/E002/pilot/<model>/<cond>/{samples.jsonl, run.json}`. Summary: `scripts/exp/E002_pilot_summary.py` → `$DATA_ROOT/logs/E002/pilot/summary.md`.
- Malformed rates are the share of episodes with ≥ 1 free turn of that type; the turn count is in brackets. "forced answer" = episodes that reached 3 searches and got `<answer>` from the environment.

| model | cond | episodes | parse rate | untagged | mean searches | P(≥1 search) | cap-hit | forced answer | empty_query (eps %, turns) | unclosed_search (eps %, turns) | multi_search (eps %, turns) | search_and_answer (eps %, turns) | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b-base | P1 | 250 | 0.972 | 7 | 1.05 | 0.95 | 0.000 | 4 | 0.0 (0) | 2.8 (7) | 0.0 (0) | 0.8 (2) | 0.312 |
| qwen3b-base | P2 | 250 | 0.992 | 2 | 0.94 | 0.91 | 0.000 | 0 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.8 (2) | 0.352 |
| qwen3b-base | P6 | 250 | 1.000 | 0 | 0.06 | 0.05 | 0.000 | 0 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.232 |
| qwen3b-instruct | P1 | 250 | 0.968 | 8 | 0.91 | 0.77 | 0.000 | 3 | 0.0 (0) | 2.0 (5) | 1.2 (3) | 2.0 (5) | 0.408 |
| qwen3b-instruct | P2 | 250 | 0.972 | 7 | 0.59 | 0.52 | 0.000 | 1 | 0.0 (0) | 3.6 (9) | 0.4 (1) | 2.4 (6) | 0.288 |
| qwen3b-instruct | P6 | 250 | 0.976 | 6 | 0.35 | 0.31 | 0.000 | 0 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.4 (1) | 0.352 |
| searchr1-3b | P1 | 250 | 0.992 | 2 | 0.96 | 0.90 | 0.000 | 3 | 0.0 (0) | 0.4 (1) | 0.4 (1) | 0.8 (2) | 0.352 |
| searchr1-3b | P2 | 250 | 0.992 | 2 | 0.82 | 0.81 | 0.000 | 0 | 0.0 (0) | 0.4 (1) | 0.0 (0) | 0.4 (1) | 0.364 |
| searchr1-3b | P6 | 250 | 1.000 | 0 | 0.25 | 0.23 | 0.000 | 0 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.232 |
| grpo-e1a-s60 | P1 | 250 | 0.996 | 1 | 1.06 | 0.98 | 0.000 | 2 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.8 (2) | 0.400 |
| grpo-e1a-s60 | P2 | 250 | 1.000 | 0 | 1.00 | 0.95 | 0.000 | 0 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.4 (1) | 0.356 |
| grpo-e1a-s60 | P6 | 250 | 1.000 | 0 | 0.10 | 0.10 | 0.000 | 0 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.256 |
| qwen7b-instruct | P1 | 250 | 0.980 | 5 | 1.14 | 0.85 | 0.000 | 20 | 0.0 (0) | 1.2 (3) | 0.0 (0) | 0.8 (2) | 0.536 |
| qwen7b-instruct | P2 | 250 | 0.992 | 2 | 0.12 | 0.11 | 0.000 | 0 | 0.0 (0) | 0.4 (1) | 0.0 (0) | 0.4 (1) | 0.352 |
| qwen7b-instruct | P6 | 250 | 1.000 | 0 | 0.04 | 0.04 | 0.000 | 0 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.360 |

no stop flags

- **Stop rules:** none met. 3B parse rates are 0.968–1.000, no 3B episode hit the 4,096-token cap, and no malformed type exceeds 5% in any cell (maximum: unclosed_search 3.6%, qwen3b-instruct P2). Proceeding to the full run as the card specifies.

### Full run (2026-11-21)
- 1,000 questions × 5 models × (P1–P6 free, plus P2 at forced depth 0 and 1) × 5 samples = 40 cells, 5,000 episodes each.
- `bash scripts/exp/E002_launch.sh full` (shared queue, 7B first, one worker per GPU on 4–7). Log `$DATA_ROOT/logs/E002/launch_full.log`: 2026-11-21 01:36–02:41:47 (66 min wall); all 40 jobs exit 0. Per-cell wall time 118–1,078 s (incl. model load), 13,981 GPU-s (3.9 GPU-h) in total.
- Code: commit 359ea18, 27 dirty/untracked paths (`git_dirty_files` in each run.json), including the uncommitted `scripts/exp/E002_*`.
- Outputs: `$DATA_ROOT/logs/E002/<model>/<cond>/{samples.jsonl, run.json}` (cond = P1..P6, P2-f0, P2-f1); per-job logs `$DATA_ROOT/logs/E002/<model>/<cond>.log`.
- Tables: `.venv/bin/python scripts/exp/E002_tables.py` → `$DATA_ROOT/logs/E002/e002_tables.md`; tidy CSV `$DATA_ROOT/logs/E002/e002_tidy.csv` (40,000 rows: question × model × condition).
- Checkpoints used, all confirmed to exist after the run:
  - qwen3b-base: `$DATA_ROOT/hf_cache/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f19…`
  - qwen3b-instruct: `…/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e7253…`
  - searchr1-3b: `…/models--PeterJinGo--SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo/snapshots/d4d2f71e…`
  - grpo-e1a-s60: `$DATA_ROOT/logs/grpo_qwen2.5-3b_full_E1A_em_20261118_102220/hf/grpo_qwen2.5-3b_full_E1A_em_20261118_102220_step60`
  - qwen7b-instruct: `…/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a3545…`
- Run health (all 40 cells): no episode hit the 4,096-token cap. Highest malformed rate per type (share of episodes): empty_query 0.02%, unclosed_search 1.94%, multi_search 0.36%, search_and_answer 1.36%, all in grpo-e1a-s60 P3. Untagged episodes per free cell: 0–97 of 5,000 (max qwen3b-instruct P6). Episodes that reached 3 searches and got the forced `<answer>`: 1–296 of 5,000 per free cell (max qwen7b-instruct P1).
- Forced-depth cells: n_search = 0 in every P2-f0 episode and 1 in every P2-f1 episode. Forced turns in which the policy had written a different tag or none (n_overridden): P2-f0 758 (7B) to 4,776 (grpo) of 5,000 forced turns; P2-f1 233 (grpo) to 4,558 (7B) of 10,000.
- Implementation notes: Table D accuracy for f0/f1 is computed on each model's own knowable / unknowable questions; the sign test compares per-question accuracy (out of 5) f1 vs f0, ties dropped.

## Result      ← you fill (facts only)

Accuracy = cover match (alias in answer), mean over questions of n_correct/5. SE over questions. Labels = each model's own E001 label. n per label is in Table A.

### Ambiguous questions excluded

#### Table A: search rates by E001 label (own model's labels)

| model | cond | n know / unk / unc | P(≥1 search ∣ knowable) | P(≥1 search ∣ unknowable) | P(≥1 search ∣ uncertain) | mean searches | parse rate | cap-hit | malformed eps |
|---|---|---|---|---|---|---|---|---|---|
| qwen3b-base | P1 | 264 / 484 / 139 | 0.936 | 0.963 | 0.931 | 1.03 | 0.988 | 0.000 | 0.003 |
| qwen3b-base | P2 | 264 / 484 / 139 | 0.882 | 0.930 | 0.886 | 0.92 | 0.998 | 0.000 | 0.002 |
| qwen3b-base | P3 | 264 / 484 / 139 | 0.881 | 0.915 | 0.873 | 0.92 | 0.999 | 0.000 | 0.001 |
| qwen3b-base | P4 | 264 / 484 / 139 | 0.889 | 0.937 | 0.889 | 0.93 | 1.000 | 0.000 | 0.001 |
| qwen3b-base | P5 | 264 / 484 / 139 | 0.986 | 0.995 | 0.988 | 1.00 | 0.998 | 0.000 | 0.001 |
| qwen3b-base | P6 | 264 / 484 / 139 | 0.025 | 0.044 | 0.035 | 0.04 | 0.999 | 0.000 | 0.000 |
| qwen3b-instruct | P1 | 278 / 466 / 143 | 0.673 | 0.837 | 0.726 | 0.85 | 0.998 | 0.000 | 0.002 |
| qwen3b-instruct | P2 | 278 / 466 / 143 | 0.489 | 0.609 | 0.512 | 0.57 | 0.998 | 0.000 | 0.004 |
| qwen3b-instruct | P3 | 278 / 466 / 143 | 0.461 | 0.695 | 0.538 | 0.62 | 0.999 | 0.000 | 0.000 |
| qwen3b-instruct | P4 | 278 / 466 / 143 | 0.491 | 0.682 | 0.537 | 0.62 | 0.998 | 0.000 | 0.002 |
| qwen3b-instruct | P5 | 278 / 466 / 143 | 0.922 | 0.912 | 0.916 | 0.97 | 0.989 | 0.000 | 0.000 |
| qwen3b-instruct | P6 | 278 / 466 / 143 | 0.245 | 0.376 | 0.241 | 0.35 | 0.981 | 0.000 | 0.003 |
| searchr1-3b | P1 | 273 / 499 / 115 | 0.876 | 0.933 | 0.936 | 0.94 | 0.998 | 0.000 | 0.002 |
| searchr1-3b | P2 | 273 / 499 / 115 | 0.757 | 0.847 | 0.800 | 0.82 | 0.999 | 0.000 | 0.003 |
| searchr1-3b | P3 | 273 / 499 / 115 | 0.755 | 0.835 | 0.805 | 0.82 | 1.000 | 0.000 | 0.002 |
| searchr1-3b | P4 | 273 / 499 / 115 | 0.804 | 0.887 | 0.847 | 0.87 | 1.000 | 0.000 | 0.000 |
| searchr1-3b | P5 | 273 / 499 / 115 | 0.713 | 0.714 | 0.703 | 0.75 | 0.997 | 0.000 | 0.001 |
| searchr1-3b | P6 | 273 / 499 / 115 | 0.087 | 0.214 | 0.143 | 0.17 | 1.000 | 0.000 | 0.000 |
| grpo-e1a-s60 | P1 | 275 / 485 / 127 | 0.979 | 0.993 | 0.984 | 1.04 | 0.990 | 0.000 | 0.011 |
| grpo-e1a-s60 | P2 | 275 / 485 / 127 | 0.913 | 0.953 | 0.932 | 0.96 | 0.994 | 0.000 | 0.011 |
| grpo-e1a-s60 | P3 | 275 / 485 / 127 | 0.868 | 0.920 | 0.883 | 0.92 | 0.989 | 0.000 | 0.026 |
| grpo-e1a-s60 | P4 | 275 / 485 / 127 | 0.943 | 0.969 | 0.951 | 0.96 | 1.000 | 0.000 | 0.000 |
| grpo-e1a-s60 | P5 | 275 / 485 / 127 | 0.997 | 0.998 | 0.998 | 1.00 | 1.000 | 0.000 | 0.000 |
| grpo-e1a-s60 | P6 | 275 / 485 / 127 | 0.045 | 0.084 | 0.052 | 0.07 | 1.000 | 0.000 | 0.000 |
| qwen7b-instruct | P1 | 451 / 323 / 113 | 0.762 | 0.947 | 0.936 | 1.04 | 0.995 | 0.000 | 0.000 |
| qwen7b-instruct | P2 | 451 / 323 / 113 | 0.099 | 0.223 | 0.145 | 0.17 | 1.000 | 0.000 | 0.000 |
| qwen7b-instruct | P3 | 451 / 323 / 113 | 0.075 | 0.209 | 0.085 | 0.14 | 0.999 | 0.000 | 0.001 |
| qwen7b-instruct | P4 | 451 / 323 / 113 | 0.107 | 0.284 | 0.159 | 0.20 | 0.999 | 0.000 | 0.001 |
| qwen7b-instruct | P5 | 451 / 323 / 113 | 0.078 | 0.161 | 0.110 | 0.12 | 0.998 | 0.000 | 0.000 |
| qwen7b-instruct | P6 | 451 / 323 / 113 | 0.026 | 0.097 | 0.041 | 0.06 | 0.998 | 0.000 | 0.000 |

#### Table B: AUROC of per-question search rate, unknowable vs knowable (95% bootstrap CI)

| model | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| qwen3b-base | 0.550 [0.519, 0.580] | 0.579 [0.541, 0.614] | 0.556 [0.516, 0.595] | 0.570 [0.532, 0.609] | 0.516 [0.502, 0.532] | 0.545 [0.520, 0.572] |
| qwen3b-instruct | 0.635 [0.596, 0.670] | 0.587 [0.546, 0.626] | 0.674 [0.634, 0.710] | 0.642 [0.602, 0.678] | 0.488 [0.457, 0.519] | 0.622 [0.580, 0.662] |
| searchr1-3b | 0.568 [0.534, 0.600] | 0.575 [0.537, 0.613] | 0.569 [0.531, 0.606] | 0.580 [0.544, 0.618] | 0.498 [0.458, 0.538] | 0.640 [0.605, 0.673] |
| grpo-e1a-s60 | 0.527 [0.509, 0.547] | 0.559 [0.528, 0.590] | 0.569 [0.534, 0.603] | 0.554 [0.525, 0.584] | 0.503 [0.496, 0.512] | 0.573 [0.543, 0.603] |
| qwen7b-instruct | 0.659 [0.627, 0.689] | 0.620 [0.588, 0.653] | 0.630 [0.597, 0.662] | 0.670 [0.637, 0.704] | 0.606 [0.575, 0.637] | 0.595 [0.570, 0.619] |

#### Table C: accuracy (alias in answer), % ± SE over questions

| model | cond | overall | knowable | unknowable |
|---|---|---|---|---|
| qwen3b-base | P1 | 48.5 ± 1.3 | 68.1 ± 2.1 | 36.9 ± 1.7 |
| qwen3b-base | P2 | 48.5 ± 1.3 | 70.8 ± 2.1 | 36.0 ± 1.7 |
| qwen3b-base | P3 | 48.5 ± 1.3 | 71.1 ± 2.0 | 35.2 ± 1.7 |
| qwen3b-base | P4 | 47.6 ± 1.3 | 70.0 ± 2.1 | 34.7 ± 1.7 |
| qwen3b-base | P5 | 50.6 ± 1.4 | 71.4 ± 2.2 | 38.5 ± 1.8 |
| qwen3b-base | P6 | 41.8 ± 1.4 | 84.8 ± 1.6 | 14.8 ± 1.2 |
| qwen3b-instruct | P1 | 58.4 ± 1.4 | 81.4 ± 1.8 | 42.3 ± 1.9 |
| qwen3b-instruct | P2 | 52.1 ± 1.4 | 79.1 ± 1.9 | 33.7 ± 1.7 |
| qwen3b-instruct | P3 | 54.6 ± 1.4 | 80.4 ± 1.9 | 37.3 ± 1.8 |
| qwen3b-instruct | P4 | 53.7 ± 1.4 | 80.1 ± 1.9 | 35.8 ± 1.8 |
| qwen3b-instruct | P5 | 59.1 ± 1.4 | 80.9 ± 2.0 | 44.1 ± 2.0 |
| qwen3b-instruct | P6 | 47.8 ± 1.3 | 82.9 ± 1.6 | 24.2 ± 1.4 |
| searchr1-3b | P1 | 56.6 ± 1.5 | 80.2 ± 2.0 | 43.1 ± 2.0 |
| searchr1-3b | P2 | 53.6 ± 1.4 | 77.8 ± 1.9 | 39.5 ± 1.8 |
| searchr1-3b | P3 | 54.5 ± 1.4 | 79.9 ± 1.9 | 39.7 ± 1.8 |
| searchr1-3b | P4 | 55.0 ± 1.4 | 79.7 ± 2.0 | 40.5 ± 1.9 |
| searchr1-3b | P5 | 47.3 ± 1.4 | 73.4 ± 2.0 | 32.8 ± 1.7 |
| searchr1-3b | P6 | 42.6 ± 1.4 | 82.6 ± 1.6 | 18.6 ± 1.4 |
| grpo-e1a-s60 | P1 | 54.2 ± 1.4 | 71.9 ± 2.2 | 43.9 ± 2.0 |
| grpo-e1a-s60 | P2 | 52.6 ± 1.4 | 73.5 ± 2.1 | 40.5 ± 1.9 |
| grpo-e1a-s60 | P3 | 49.8 ± 1.4 | 69.4 ± 2.2 | 38.6 ± 1.9 |
| grpo-e1a-s60 | P4 | 54.5 ± 1.4 | 75.4 ± 2.1 | 42.4 ± 1.9 |
| grpo-e1a-s60 | P5 | 56.6 ± 1.5 | 77.3 ± 2.1 | 44.0 ± 2.0 |
| grpo-e1a-s60 | P6 | 43.1 ± 1.4 | 84.1 ± 1.6 | 16.9 ± 1.3 |
| qwen7b-instruct | P1 | 71.4 ± 1.3 | 90.0 ± 1.1 | 48.6 ± 2.5 |
| qwen7b-instruct | P2 | 51.2 ± 1.5 | 83.0 ± 1.5 | 13.3 ± 1.4 |
| qwen7b-instruct | P3 | 51.0 ± 1.5 | 83.2 ± 1.5 | 12.8 ± 1.4 |
| qwen7b-instruct | P4 | 52.9 ± 1.5 | 83.6 ± 1.5 | 16.8 ± 1.6 |
| qwen7b-instruct | P5 | 49.3 ± 1.5 | 81.6 ± 1.6 | 11.3 ± 1.4 |
| qwen7b-instruct | P6 | 48.6 ± 1.6 | 82.3 ± 1.6 | 9.0 ± 1.3 |

#### Table D: P2 forced depth, accuracy % ± SE (forced 0 vs forced 1 searches)

| model | knowable n | knowable f0 | knowable f1 | sign test knowable (f1>f0 / f1<f0, p) | unknowable n | unknowable f0 | unknowable f1 |
|---|---|---|---|---|---|---|---|
| qwen3b-base | 264 | 75.5 ± 1.8 | 69.5 ± 2.1 | 72 / 108, p = 0.0089 | 484 | 9.8 ± 1.0 | 38.3 ± 1.7 |
| qwen3b-instruct | 278 | 73.1 ± 2.0 | 80.8 ± 1.9 | 88 / 37, p = 5.83e-06 | 466 | 7.5 ± 0.9 | 41.9 ± 1.9 |
| searchr1-3b | 273 | 85.1 ± 1.6 | 78.7 ± 2.0 | 46 / 78, p = 0.00516 | 499 | 14.1 ± 1.3 | 42.2 ± 1.9 |
| grpo-e1a-s60 | 275 | 78.4 ± 1.8 | 76.9 ± 2.1 | 70 / 78, p = 0.565 | 485 | 11.1 ± 1.2 | 43.4 ± 2.0 |
| qwen7b-instruct | 451 | 81.2 ± 1.6 | 89.0 ± 1.3 | 101 / 53, p = 0.000136 | 323 | 4.4 ± 0.9 | 44.7 ± 2.6 |

#### Table E: P(≥1 search ∣ knowable) on the shared 3B knowable set

| model | n | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|---|
| qwen3b-base | 200 | 0.931 | 0.870 | 0.871 | 0.877 | 0.985 | 0.023 |
| qwen3b-instruct | 200 | 0.653 | 0.472 | 0.445 | 0.479 | 0.927 | 0.243 |
| searchr1-3b | 200 | 0.855 | 0.717 | 0.721 | 0.774 | 0.682 | 0.080 |
| grpo-e1a-s60 | 200 | 0.975 | 0.909 | 0.863 | 0.936 | 0.996 | 0.044 |

### All questions

#### Table A: search rates by E001 label (own model's labels)

| model | cond | n know / unk / unc | P(≥1 search ∣ knowable) | P(≥1 search ∣ unknowable) | P(≥1 search ∣ uncertain) | mean searches | parse rate | cap-hit | malformed eps |
|---|---|---|---|---|---|---|---|---|---|
| qwen3b-base | P1 | 265 / 590 / 145 | 0.936 | 0.964 | 0.932 | 1.03 | 0.988 | 0.000 | 0.003 |
| qwen3b-base | P2 | 265 / 590 / 145 | 0.882 | 0.932 | 0.886 | 0.93 | 0.997 | 0.000 | 0.002 |
| qwen3b-base | P3 | 265 / 590 / 145 | 0.881 | 0.915 | 0.873 | 0.92 | 0.999 | 0.000 | 0.001 |
| qwen3b-base | P4 | 265 / 590 / 145 | 0.889 | 0.934 | 0.890 | 0.93 | 1.000 | 0.000 | 0.001 |
| qwen3b-base | P5 | 265 / 590 / 145 | 0.986 | 0.994 | 0.989 | 1.00 | 0.998 | 0.000 | 0.001 |
| qwen3b-base | P6 | 265 / 590 / 145 | 0.025 | 0.052 | 0.037 | 0.05 | 0.999 | 0.000 | 0.000 |
| qwen3b-instruct | P1 | 278 / 575 / 147 | 0.673 | 0.830 | 0.729 | 0.85 | 0.997 | 0.000 | 0.002 |
| qwen3b-instruct | P2 | 278 / 575 / 147 | 0.489 | 0.574 | 0.502 | 0.56 | 0.998 | 0.000 | 0.004 |
| qwen3b-instruct | P3 | 278 / 575 / 147 | 0.461 | 0.674 | 0.537 | 0.62 | 0.999 | 0.000 | 0.000 |
| qwen3b-instruct | P4 | 278 / 575 / 147 | 0.491 | 0.651 | 0.532 | 0.61 | 0.998 | 0.000 | 0.002 |
| qwen3b-instruct | P5 | 278 / 575 / 147 | 0.922 | 0.870 | 0.910 | 0.95 | 0.984 | 0.000 | 0.000 |
| qwen3b-instruct | P6 | 278 / 575 / 147 | 0.245 | 0.353 | 0.242 | 0.34 | 0.981 | 0.000 | 0.003 |
| searchr1-3b | P1 | 275 / 605 / 120 | 0.876 | 0.936 | 0.933 | 0.94 | 0.998 | 0.000 | 0.002 |
| searchr1-3b | P2 | 275 / 605 / 120 | 0.756 | 0.848 | 0.797 | 0.83 | 0.999 | 0.000 | 0.002 |
| searchr1-3b | P3 | 275 / 605 / 120 | 0.753 | 0.832 | 0.803 | 0.82 | 1.000 | 0.000 | 0.001 |
| searchr1-3b | P4 | 275 / 605 / 120 | 0.803 | 0.883 | 0.850 | 0.87 | 1.000 | 0.000 | 0.000 |
| searchr1-3b | P5 | 275 / 605 / 120 | 0.713 | 0.717 | 0.690 | 0.75 | 0.997 | 0.000 | 0.001 |
| searchr1-3b | P6 | 275 / 605 / 120 | 0.088 | 0.230 | 0.147 | 0.19 | 1.000 | 0.000 | 0.000 |
| grpo-e1a-s60 | P1 | 276 / 592 / 132 | 0.979 | 0.993 | 0.983 | 1.04 | 0.991 | 0.000 | 0.010 |
| grpo-e1a-s60 | P2 | 276 / 592 / 132 | 0.913 | 0.957 | 0.930 | 0.96 | 0.995 | 0.000 | 0.010 |
| grpo-e1a-s60 | P3 | 276 / 592 / 132 | 0.868 | 0.928 | 0.877 | 0.93 | 0.990 | 0.000 | 0.024 |
| grpo-e1a-s60 | P4 | 276 / 592 / 132 | 0.943 | 0.969 | 0.948 | 0.96 | 1.000 | 0.000 | 0.000 |
| grpo-e1a-s60 | P5 | 276 / 592 / 132 | 0.997 | 0.999 | 0.998 | 1.00 | 1.000 | 0.000 | 0.000 |
| grpo-e1a-s60 | P6 | 276 / 592 / 132 | 0.045 | 0.094 | 0.055 | 0.08 | 1.000 | 0.000 | 0.000 |
| qwen7b-instruct | P1 | 451 / 434 / 115 | 0.762 | 0.950 | 0.936 | 1.08 | 0.995 | 0.000 | 0.001 |
| qwen7b-instruct | P2 | 451 / 434 / 115 | 0.099 | 0.206 | 0.144 | 0.17 | 1.000 | 0.000 | 0.000 |
| qwen7b-instruct | P3 | 451 / 434 / 115 | 0.075 | 0.296 | 0.092 | 0.20 | 0.998 | 0.000 | 0.001 |
| qwen7b-instruct | P4 | 451 / 434 / 115 | 0.107 | 0.297 | 0.165 | 0.22 | 0.999 | 0.000 | 0.001 |
| qwen7b-instruct | P5 | 451 / 434 / 115 | 0.078 | 0.151 | 0.108 | 0.12 | 0.998 | 0.000 | 0.000 |
| qwen7b-instruct | P6 | 451 / 434 / 115 | 0.026 | 0.122 | 0.040 | 0.07 | 0.998 | 0.000 | 0.000 |

#### Table B: AUROC of per-question search rate, unknowable vs knowable (95% bootstrap CI)

| model | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| qwen3b-base | 0.555 [0.525, 0.587] | 0.584 [0.546, 0.621] | 0.558 [0.521, 0.597] | 0.569 [0.533, 0.607] | 0.513 [0.497, 0.528] | 0.559 [0.534, 0.583] |
| qwen3b-instruct | 0.628 [0.592, 0.670] | 0.563 [0.522, 0.606] | 0.659 [0.618, 0.701] | 0.620 [0.579, 0.662] | 0.446 [0.419, 0.480] | 0.603 [0.565, 0.644] |
| searchr1-3b | 0.572 [0.541, 0.605] | 0.576 [0.536, 0.617] | 0.566 [0.526, 0.606] | 0.575 [0.540, 0.611] | 0.499 [0.462, 0.540] | 0.654 [0.621, 0.686] |
| grpo-e1a-s60 | 0.528 [0.511, 0.547] | 0.566 [0.538, 0.598] | 0.582 [0.549, 0.618] | 0.556 [0.530, 0.585] | 0.504 [0.497, 0.512] | 0.585 [0.555, 0.615] |
| qwen7b-instruct | 0.663 [0.636, 0.689] | 0.615 [0.586, 0.645] | 0.688 [0.657, 0.719] | 0.696 [0.665, 0.728] | 0.597 [0.568, 0.627] | 0.628 [0.604, 0.652] |

#### Table C: accuracy (alias in answer), % ± SE over questions

| model | cond | overall | knowable | unknowable |
|---|---|---|---|---|
| qwen3b-base | P1 | 45.1 ± 1.3 | 67.8 ± 2.2 | 33.4 ± 1.5 |
| qwen3b-base | P2 | 45.0 ± 1.3 | 70.5 ± 2.1 | 32.4 ± 1.5 |
| qwen3b-base | P3 | 44.8 ± 1.2 | 70.9 ± 2.0 | 31.6 ± 1.5 |
| qwen3b-base | P4 | 44.2 ± 1.2 | 69.7 ± 2.1 | 31.3 ± 1.5 |
| qwen3b-base | P5 | 47.0 ± 1.3 | 71.1 ± 2.2 | 34.8 ± 1.7 |
| qwen3b-base | P6 | 37.4 ± 1.3 | 84.5 ± 1.6 | 12.4 ± 1.0 |
| qwen3b-instruct | P1 | 54.0 ± 1.3 | 81.4 ± 1.8 | 37.7 ± 1.7 |
| qwen3b-instruct | P2 | 47.3 ± 1.3 | 79.1 ± 1.9 | 29.1 ± 1.5 |
| qwen3b-instruct | P3 | 49.8 ± 1.3 | 80.4 ± 1.9 | 32.5 ± 1.6 |
| qwen3b-instruct | P4 | 49.0 ± 1.3 | 80.1 ± 1.9 | 31.4 ± 1.6 |
| qwen3b-instruct | P5 | 54.0 ± 1.4 | 80.9 ± 2.0 | 38.3 ± 1.8 |
| qwen3b-instruct | P6 | 43.3 ± 1.3 | 82.9 ± 1.6 | 21.0 ± 1.2 |
| searchr1-3b | P1 | 52.4 ± 1.4 | 79.7 ± 2.0 | 38.6 ± 1.8 |
| searchr1-3b | P2 | 49.6 ± 1.3 | 77.3 ± 2.0 | 35.5 ± 1.6 |
| searchr1-3b | P3 | 50.4 ± 1.3 | 79.3 ± 1.9 | 35.6 ± 1.6 |
| searchr1-3b | P4 | 51.0 ± 1.4 | 79.3 ± 2.0 | 36.4 ± 1.7 |
| searchr1-3b | P5 | 43.6 ± 1.3 | 72.9 ± 2.1 | 29.4 ± 1.5 |
| searchr1-3b | P6 | 38.3 ± 1.3 | 82.0 ± 1.7 | 15.9 ± 1.2 |
| grpo-e1a-s60 | P1 | 50.5 ± 1.4 | 71.7 ± 2.2 | 39.6 ± 1.8 |
| grpo-e1a-s60 | P2 | 48.9 ± 1.3 | 73.2 ± 2.1 | 36.5 ± 1.7 |
| grpo-e1a-s60 | P3 | 46.4 ± 1.3 | 69.1 ± 2.2 | 35.0 ± 1.7 |
| grpo-e1a-s60 | P4 | 50.5 ± 1.4 | 75.1 ± 2.1 | 38.0 ± 1.7 |
| grpo-e1a-s60 | P5 | 52.4 ± 1.4 | 77.0 ± 2.1 | 39.5 ± 1.8 |
| grpo-e1a-s60 | P6 | 38.9 ± 1.3 | 83.9 ± 1.6 | 14.8 ± 1.1 |
| qwen7b-instruct | P1 | 66.4 ± 1.3 | 90.0 ± 1.1 | 42.9 ± 2.1 |
| qwen7b-instruct | P2 | 45.9 ± 1.4 | 83.0 ± 1.5 | 10.8 ± 1.1 |
| qwen7b-instruct | P3 | 46.8 ± 1.4 | 83.2 ± 1.5 | 13.1 ± 1.2 |
| qwen7b-instruct | P4 | 47.7 ± 1.4 | 83.6 ± 1.5 | 14.5 ± 1.3 |
| qwen7b-instruct | P5 | 44.2 ± 1.4 | 81.6 ± 1.6 | 9.5 ± 1.1 |
| qwen7b-instruct | P6 | 43.7 ± 1.5 | 82.3 ± 1.6 | 8.1 ± 1.0 |

#### Table D: P2 forced depth, accuracy % ± SE (forced 0 vs forced 1 searches)

| model | knowable n | knowable f0 | knowable f1 | sign test knowable (f1>f0 / f1<f0, p) | unknowable n | unknowable f0 | unknowable f1 |
|---|---|---|---|---|---|---|---|
| qwen3b-base | 265 | 75.2 ± 1.8 | 69.2 ± 2.1 | 72 / 109, p = 0.00728 | 590 | 8.2 ± 0.9 | 34.5 ± 1.6 |
| qwen3b-instruct | 278 | 73.1 ± 2.0 | 80.8 ± 1.9 | 88 / 37, p = 5.83e-06 | 575 | 6.1 ± 0.8 | 37.1 ± 1.7 |
| searchr1-3b | 275 | 84.7 ± 1.6 | 78.3 ± 2.0 | 46 / 79, p = 0.00402 | 605 | 11.8 ± 1.1 | 37.8 ± 1.7 |
| grpo-e1a-s60 | 276 | 78.2 ± 1.8 | 76.6 ± 2.1 | 70 / 79, p = 0.512 | 592 | 9.3 ± 1.0 | 38.9 ± 1.8 |
| qwen7b-instruct | 451 | 81.2 ± 1.6 | 89.0 ± 1.3 | 101 / 53, p = 0.000136 | 434 | 3.7 ± 0.7 | 39.1 ± 2.2 |

#### Table E: P(≥1 search ∣ knowable) on the shared 3B knowable set

| model | n | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|---|
| qwen3b-base | 200 | 0.931 | 0.870 | 0.871 | 0.877 | 0.985 | 0.023 |
| qwen3b-instruct | 200 | 0.653 | 0.472 | 0.445 | 0.479 | 0.927 | 0.243 |
| searchr1-3b | 200 | 0.855 | 0.717 | 0.721 | 0.774 | 0.682 | 0.080 |
| grpo-e1a-s60 | 200 | 0.975 | 0.909 | 0.863 | 0.936 | 0.996 | 0.044 |

### Summary of numbers (ambiguous excluded)
- P(≥1 search | knowable) across P1–P4: qwen3b-base 0.88–0.94, grpo-e1a-s60 0.87–0.98, searchr1-3b 0.76–0.88, qwen3b-instruct 0.46–0.67, qwen7b-instruct 0.08–0.11 (P2–P4) and 0.76 (P1). Under P5 (both demos search): 0.71 (searchr1-3b) to 0.997 (grpo) for 3B models, 0.08 for the 7B. Under P6 (both demos direct): 0.03–0.25.
- P(≥1 search | unknowable) exceeds P(≥1 search | knowable) in 29 of 30 cells (exception: qwen3b-instruct P5, 0.912 vs 0.922).
- AUROC (Table B) ranges 0.488–0.674 across the 30 cells (n = 264–451 knowable, 323–499 unknowable per model). The 95% CI includes 0.5 in 3 cells (P5 for qwen3b-instruct, searchr1-3b, grpo-e1a-s60) and excludes it in the other 27. Highest per model: qwen3b-base 0.579 (P2), qwen3b-instruct 0.674 (P3), searchr1-3b 0.640 (P6), grpo-e1a-s60 0.573 (P6), qwen7b-instruct 0.670 (P4).
- Accuracy on knowable questions under P6 (mostly no search): 82.3–84.8% for all models; under P1–P5: 68.1–81.4% for 3B models, 81.6–90.0% for the 7B.
- Table D (P2 forced depth, knowable): f1 − f0 = −6.0 pts qwen3b-base (72/108, p = 0.0089), +7.7 qwen3b-instruct (88/37, p = 5.8e-6), −6.4 searchr1-3b (46/78, p = 0.0052), −1.5 grpo-e1a-s60 (70/78, p = 0.57), +7.8 qwen7b-instruct (101/53, p = 1.4e-4). Unknowable: f0 4.4–14.1%, f1 38.3–44.7%.
- Table E (shared 3B knowable set, n = 200): values within 0.031 of the Table A knowable column in every cell (largest gap: searchr1-3b P5, 0.682 vs 0.713).

