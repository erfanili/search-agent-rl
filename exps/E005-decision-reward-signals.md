# E005: Which training signal connects the search decision to the model's knowledge?
Question: Starting from the E004 setup (qwen3b-instruct, P2 context, MIX-50 data), does adding (C) a per-search cost, (S) a decision reward from the start model's own self-consistency, or (O) a decision reward from oracle knowable/unknowable labels make the search decision separate knowable from unknowable questions on held-out data, hold under prompt shift, and track the model's internal knowledge signal beyond the labels?
Parent: E004

## Setup
- Start model and context: Qwen/Qwen2.5-3B-Instruct, the exact E002 P2 prompt in the chat template (same as E004).
- Training data: E004's MIX-50 set ($DATA_ROOT/verl_data/e004/mix50/, 3,200 rows, labels included). No new sampling of questions.
- Self-consistency targets (precompute before the pilot): for each of the 3,200 training questions, 5 new closed-book samples from the start model with the exact E001 rev-2 prompt and decoding (T=0.7) but seed 1. agreement a = share of samples matching the most common normalized answer (as E003 S2; untagged samples never agree). Store a in extra_info. Report the AUROC of (1 − a) for unknowable vs knowable on the training set, and the histogram of a by label.
- Outcome reward, all arms: alias_match_strict (new, replacing alias_match for training). Score 1 only if:
  1. the first <answer> span is closed, and
  2. after normalize_answer, a gold alias equals the span, OR a gold alias is a substring of the span AND the span has at most (alias word count + 3) words.
  Otherwise 0.
- Decision variable: d = 1 if the episode made at least 1 executed search, else 0.
- Arms (reward = outcome + arm term):
  - C (cost): − 0.15 × number of executed searches.
  - S (self-consistency): + λ · [ d·(1 − a) + (1 − d)·a ], with λ = 0.3. Searching earns more when agreement is low; answering directly earns more when it is high.
  - O (oracle): the same formula with a = 1 for knowable and a = 0 for unknowable, λ = 0.3.
  Run IDs: E005-C-s0, E005-S-s0, E005-O-s0. Order: O, S, C.
- Recipe: E004's exactly (GRPO 64×5, mini-batch 32, lr 1e-6, warmup 6 steps, KL 0.001 low_var_kl, rollout T=1.0, max 3 searches with forced answer, 2,048-token episodes, MAX_PROMPT 1024, in-training validation off), with ppo_max_token_len_per_gpu=4608 and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. 100 steps per arm.
- Code: build the reward changes on top of the existing uncommitted worktree ../sar-alias-reward (don't commit or merge). Add alias_match_strict, the decision term, the cost term, tests for each, and per-step logging of each reward component. Save base commit, code.patch and git status in every run directory, as in E004.

## Training-time logging and stop rules (all arms, every step)
- Per label: share of rollouts with ≥1 search; mean outcome reward, mean arm term and mean total reward, each at 0 searches vs ≥1 search; groups mixed in search; groups mixed in search and total reward; answer-span mean and p95 words; unclosed share.
- Stop rule, checked every step from step 10: an arm stops if its mean answer-span words exceed 2× its own steps 1–3 mean for either label, or its unclosed share exceeds 0.05. Record the stop, evaluate that arm's existing checkpoints, and continue the chain with the next arm.
- On CUDA OOM: resume once from the latest full checkpoint with ppo_max_token_len_per_gpu=3072. On a second OOM, stop that arm, evaluate what exists, and continue the chain.

## Checkpoints and disk
As in E004: save at 20, 40, 60, 80, 100. Merge each, confirm it loads, then delete the FSDP shards and optimizer state of earlier checkpoints, keeping only the latest full checkpoint for resuming. After an arm finishes and its hf/step_100 loads, delete that arm's step-100 shards and optimizer state too. Never delete merged weights, and never touch any other run. Disk guard as in E004: if free space can't fit the next checkpoint plus its merge, stop that arm, evaluate what exists, and continue.

## Pilot first
3 steps of each arm, no checkpoint. Report per arm: step time, peak memory, share of rollouts with ≥1 search at step 1 by label, the mean of each reward component at step 1 by label and by search/no-search, parse rate, malformed-search rate and truncation rate. Stop if the step-1 search share is outside 0.3–0.7, the parse rate is below 95% or truncation is above 2%. If the pilot passes, run the chain (O, S, C, then evaluation) without waiting for me.

## Evaluation (steps 0, 20, …, 100, every arm)
- Step 0: reuse E002/E003/E004 step-0 results (same model, prompt, decoding).
- Data and labels: data/e001_easy_1k.jsonl, ambiguous excluded, qwen3b-instruct's E001 labels. Prompts: P2, plus P1 and P6. T=0.7, 5 samples.
- Per cell: P(≥1 search | knowable), P(≥1 search | unknowable), AUROC of the per-question search rate [bootstrap 95% CI], accuracy overall / knowable / unknowable under BOTH the E001 rule and alias_match_strict (± SE), mean answer-span words.
- Under P2: S5 margin AUROC and the layer-24 probe AUROC (E003 method, refit per checkpoint).
- Use of the internal signal (the key analysis), on held-out evaluation questions under P2 at each checkpoint:
  - Fixed knowledge score k per question = the start model's out-of-fold layer-24 probe prediction from E003 (no refitting; uncertain questions scored by the fold models).
  - Logistic regression of each sample's search decision on [label, k], over knowable + unknowable questions. Report the k coefficient with a question-level bootstrap 95% CI.
  - Spearman correlation between the S5 margin and k, within knowable and within unknowable separately.
  - Uncertain questions (E001 label uncertain): search rate binned by the start model's E003 S2 agreement (bins 0.2–0.4, 0.6, 0.8–1.0).
- Final relabel at step 100 per arm (E001 procedure), reporting the share of closed-book samples with a <search> tag, with no <answer>, and the mean answer-span length, alongside the start model's E001 run.

## Output wanted
1. Table A (per arm): rows = steps 0, 20, …, 100; columns = P(s|kn), P(s|unk), AUROC [CI], accuracy kn / unk (strict rule), mean answer words. Under P2.
2. Table B (prompt shift): AUROC, P(s|kn) and P(s|unk) under P1, P2 and P6 at steps 0 and 100, per arm.
3. Table C (per arm, by step): S5 margin AUROC, probe AUROC, k coefficient [CI], Spearman(margin, k) within knowable and within unknowable.
4. Table D (training dynamics, per arm, every 10 steps): the logged quantities above.
5. Table E: uncertain-question search rate by agreement bin, steps 0 and 100, per arm.
6. Table F: final relabel diagnostics per arm vs the start model.
7. Plot: AUROC and both search rates against step, all 3 arms plus E004 arm A on one figure.
8. Files: $DATA_ROOT/logs/E005/<run id>/ and $DATA_ROOT/logs/E005/eval/<run id>/step<k>/<prompt>/, plus one tidy CSV of all evaluation cells.

## Execution   ← you fill

### Code and provenance (2026-11-22)
- Worktree `../sar-alias-reward` (branch feat/alias-reward, base 359ea18), uncommitted, on top of the E004 changes:
  - `agent/reward.py`: `alias_match_strict`, `executed_searches`, `arm_term(arm, n_exec, extra_info, lam, cost)`; `compute_score` takes `arm` (none/C/S/O), `lam`, `cost` and returns `score` = outcome + arm term, plus `outcome`, `arm_term`, `n_exec_search`, `alias_match_strict`; `compute_score_batch` logs `components` per step.
  - `agent/step_stats.py`: `component_stats` (all rollouts and per e004_label: executed-search share; mean outcome / arm term / total over all, at 0 and at ≥1 executed searches; groups; groups mixed in search; groups mixed in search and total).
  - `scripts/train_grpo_3b.sh`: `REWARD_ARM`, `ARM_LAM`, `ARM_COST` → `reward_kwargs.arm/lam/cost`.
  - `tests/test_reward.py`: 8 new tests (strict rule, executed-search count, cost term, S term at a ∈ {0.2, 0.6, 1.0}, O term, component logging). `.venv`: 64/64 pass; `.venv-verl` (test_reward + test_rollout_contract): 47/47 pass.
- Main repo `scripts/exp/` (untracked): `E005_sc.py`, `E005_train.sh`, `E005_pilot.sh`, `E005_pilot_summary.py`, `E005_stop_watch.py`, `E005_chain.sh`, `E005_eval_all.sh`, `E005_tables.py`. `E004_eval.py` and `E004_hidden.py` got an `--exp` flag (default E004, behaviour unchanged) so E005 reuses them with outputs under `$DATA_ROOT/logs/E005/eval/`.
- Every run directory holds `base_commit.txt` (worktree HEAD, arm, MAX_TOKENS_PER_GPU, PYTORCH_CUDA_ALLOC_CONF), `code.patch` (worktree diff), `git_status.txt`; resumed segments add `.<tag>` versions.

### Self-consistency targets (2026-11-22 11:31–11:35, GPU 4)
- `E005_sc.py gen`: 3,200 MIX-50 questions × 5 samples, E001 rev-2 prompt in the chat template, T=0.7, top_p 1.0, max 512, stop `</answer>`, vLLM engine and request seed 1; E001 parse; a = E003_tables.agreement (untagged never agree). Wall 208 s. Untagged samples: 10 of 16,000.
- Data: `$DATA_ROOT/verl_data/e005/mix50sc/{train,val}.parquet` = E004 MIX-50 parquet, same rows and order, with `extra_info.sc_agreement` added (used by all three arms). Samples and report: `$DATA_ROOT/verl_data/e005/{sc_samples.jsonl, sc_agreement.jsonl, sc_run.json, sc_report.md}`.
- AUROC of (1 − a), unknowable vs knowable, training set: **0.750 [0.733, 0.765]** (n 1,600 / 1,600). Mean a: knowable 0.690, unknowable 0.417.

| a | knowable n (share) | unknowable n (share) |
|---|---|---|
| 0.2 | 223 (0.139) | 727 (0.454) |
| 0.4 | 261 (0.163) | 393 (0.246) |
| 0.6 | 278 (0.174) | 223 (0.139) |
| 0.8 | 253 (0.158) | 130 (0.081) |
| 1.0 | 585 (0.366) | 127 (0.079) |

### Implementation choices and deviations
1. **PYTORCH_CUDA_ALLOC_CONF left unset** (card: expandable_segments:True). In E004 (attempt r3, 2026-11-21 22:19) this setting made vLLM's sleep-mode CuMemAllocator assert at rollout-engine load and the run hung 74 min; E004 arm A r4 and arm B then ran 100 steps without it at ppo_max_token_len_per_gpu=4608.
2. Executed searches = `<information>` blocks the environment inserted in the trajectory (one per executed search; malformed or over-limit searches insert none). d = [executed ≥ 1]. The pilot search-share rule uses this d.
3. alias_match_strict: word counts are taken after normalize_answer (span and alias); "substring" = character substring of the normalized span, as cover_match.
4. Stop rule: per label, mean answer-span words > 2 × the arm's own steps 1–3 mean (pooled over the answered rollouts of steps 1–3); unclosed share > 0.05 checked per label and over all rollouts (as E004's step-20 check). Watcher `E005_stop_watch.py` checks every logged step ≥ 10, writes `stop_watch.jsonl` and on a stop `STOP`, and kills training.
5. A failure that is neither CUDA OOM, the stop rule nor the disk guard (including a stall: train.log silent 30 min) stops the whole chain, as in E004 chain3 (the card does not cover it).
6. Relabel: at step 100, or at an arm's last confirmed step if it stopped early.
7. Step-0 strict accuracy and answer words are recomputed from the E002 qwen3b-instruct samples (E001-rule step-0 numbers reproduce E004: P2 AUROC 0.587, accuracy 52.1%).
8. k = E003 out-of-fold layer-24 probe P(unknowable) (higher = less knowledge): knowable/unknowable from E003 `signals.csv` (S4_fixed_L24); uncertain = mean of the 5 rebuilt E003 fold models (rebuilt folds reproduce signals.csv exactly, max |diff| 0). Cached in `$DATA_ROOT/logs/E005/k_scores.jsonl`. Logistic regression: sklearn, no penalty, regressors [unknowable indicator, k] on raw scale, one row per P2 episode; CI from 1,000 question-level bootstrap resamples.
9. Uncertain-question agreement bins use the start model's E003 S2 agreement (s2_samples.jsonl).
### Pilots (2026-11-22 11:36–12:09, GPUs 4–7, 3 steps each, no checkpoint, MAX_TOKENS_PER_GPU=4608)
Run dirs `$DATA_ROOT/logs/E005/E005-{O,S,C}-pilot/` (summary in `pilot_summary.md`). Search share = executed searches (d); parse / malformed / finish reasons from the 960 dumped episodes per pilot; truncation = verl response_length/clip_ratio.

| arm | step time s (1/2/3) | peak mem alloc / reserved GB | step-1 search share all / kn / unk | parse | truncation max | malformed: unclosed / multi / search+answer | stop flags |
|---|---|---|---|---|---|---|---|
| O | 173 / 174 / 144 | 32.7 / 37.5 | 0.509 / 0.488 / 0.523 | 0.998 | 0.003 | 0.006 / 0.001 / 0.007 | none |
| S | 160 / 176 / 179 | 33.1 / 37.6 | 0.512 / 0.488 / 0.528 | 0.985 | 0.009 | 0.020 / 0.011 / 0.020 | none |
| C | 165 / 148 / 147 | 33.2 / 37.6 | 0.522 / 0.512 / 0.528 | 0.998 | 0.000 | 0.015 / 0.002 / 0.009 | none |

Step-1 reward components (mean outcome, arm term, total; at 0 executed searches → at ≥1):

| arm | label | outcome 0s → ≥1s | arm term 0s → ≥1s | total 0s → ≥1s |
|---|---|---|---|---|
| O | knowable | 0.812 → 0.934 | 0.300 → 0.000 | 1.112 → 0.934 |
| O | unknowable | 0.054 → 0.284 | 0.000 → 0.300 | 0.054 → 0.584 |
| S | knowable | 0.781 → 0.918 | 0.222 → 0.086 | 1.003 → 1.004 |
| S | unknowable | 0.033 → 0.340 | 0.104 → 0.183 | 0.136 → 0.523 |
| C | knowable | 0.705 → 0.859 | 0.000 → −0.152 | 0.705 → 0.707 |
| C | unknowable | 0.033 → 0.320 | 0.000 → −0.167 | 0.033 → 0.153 |

All three pilots passed; `E005_pilot.sh` started `E005_chain.sh` at 12:09:11 (arm O attempt 1 at 12:09:14, free /data 203 GB).

### Change: arm C cost 0.15 → 0.20 per executed search (Erfan, 2026-11-22, after the pilots, before arm C started)
- Reason (Erfan): at 0.15, pilot C's step-1 knowable totals were 0.705 (no search) vs 0.707 (search), so skipping was not favored on knowable questions; at 0.20 the step-1 numbers favor skipping on knowable (0.705 vs 0.659) and searching on unknowable (0.033 vs 0.120). No new pilot.
- Implementation (~12:20, arm O training): `scripts/exp/E005_train.sh` replaced by a new file (new inode, so arm O's running launcher kept reading the old one) that sets `ARM_COST=0.20` for arm C only (O and S do not use the cost) and writes the cost into `base_commit*.txt`. The running `E005_chain.sh` was not edited; it calls `E005_train.sh` fresh for each arm and for any resume, so arm C (including an OOM resume) uses 0.20. Arm O's config.txt records cost 0.15 (unused by its O term). The pilot C row above was run at 0.15.

### Chain (2026-11-22 12:09–15:31, GPUs 4–7)
Commit 359ea18 + uncommitted worktree `/home/erfan/sar-alias-reward` (feat/alias-reward); `code.patch`, `git.diff`, `git_status.txt` (10 dirty entries) and `base_commit.txt` per run dir. Log: `$DATA_ROOT/logs/E005/chain.log`. All three arms ended by the answer-length stop rule on knowable questions; no CUDA OOM, no disk-guard stop, no 3072 resume. Training exit 134 = killed by the stop watcher.

| arm | run id | cost / λ | wall | steps logged | stop | knowable mean words at stop / 2 × steps 1–3 mean | unclosed share at stop (all) | knowable words, 3 steps before stop | checkpoints |
|---|---|---|---|---|---|---|---|---|---|
| O | E005-O-s0 | λ 0.3 | 12:09–13:14 | 23 | step 23 | 17.59 / 14.05 | 0.003 | 3.13 / 1.93 (steps 21, 22); 1.86 (step 20) | hf/step_20 (load ok), ckpt/global_step_20 |
| S | E005-S-s0 | λ 0.3 | 13:14–14:29 | 25 | step 25 | 15.93 / 9.90 | 0.003 | 2.03 / 2.00 / 2.35 (steps 22–24) | hf/step_20 (load ok), ckpt/global_step_20 |
| C | E005-C-s0 | cost 0.20 | 14:29–15:06 | 13 | step 13 | 20.27 / 16.04 | 0.003 | 5.07 / 1.95 (steps 11, 12) | none (stopped before step 20) |

- Paths: `$DATA_ROOT/logs/E005/E005-{O,S,C}-s0/` (train.log, reward_steps.jsonl, stop_watch.jsonl, STOP, rollouts/). Merged weights: `E005-O-s0/hf/step_20`, `E005-S-s0/hf/step_20` (6.4 GB each); FSDP shards + optimizer `E005-{O,S}-s0/ckpt/global_step_20` (36 GB each) kept as each arm's latest full checkpoint. Nothing deleted (no arm reached step 40 or 100). All four exist after the run (checked 2026-11-22 21:20). Free /data 119 GB.
- Evaluation (15:07–15:31, GPUs 4–5): `E005_eval_all.sh`; O and S at step 20 (P1/P2/P6, S5, hidden states, relabel), exit 0. C has no checkpoint beyond step 0, so its rows are step 0 only. Steps 40–100 skipped for all arms (no confirmed merge). Eval dirs `$DATA_ROOT/logs/E005/eval/E005-{O,S}-s0/step20/`.
- Tables: `.venv-verl/bin/python scripts/exp/E005_tables.py` (2026-11-22 21:19) → `$DATA_ROOT/logs/E005/{e005_tables.md, e005_tidy.csv (15 cells), e005_plot.png, k_scores.jsonl}`.
- Card deviation (Table B, E): "step 100" = the arm's last evaluated step (20 for O and S; none for C).

### Defect found: relabel keeps fewer than 5 samples (Table F, and E004's relabel)
`E004_eval.py --relabel` requests n=5 closed-book samples in one vLLM AsyncLLM request and keeps only the last streamed output (`gen()`), which held 1–5 of the 5 completions. Samples per question in `relabel/samples.jsonl`: E005-O step 20 1:720, 2:99, 3:46, 4:43, 5:92 (1,000 questions); E005-S similar; E004-A step 100 1:682, 2:105, 3:57, 4:45, 5:111; E004-B step 100 1:741, 2:80, 3:46, 4:41, 5:92. Labels (`labels.jsonl`) were computed from these partial sets, so the Table F label counts here and E004's reported relabel counts (A 100 / 545, B 78 / 538) are not from 5 samples per question. P1/P2/P6 evaluation episodes use one request per episode (n=1) and have 5 samples for all 1,000 questions (checked for E005 O/S step 20 and E004 A/B step 100). Not rerun.

### Relabel fix (2026-11-22, requested by Erfan)
- **Fix (2026-11-22, `scripts/exp/E004_eval.py`, untracked main-repo experiment script):** the relabel now sends 5 separate n=1 requests per question (same sampling: E001 prompt, T 0.7, top_p 1.0, max 512, stop `</answer>`; per-sample seed = md5("<id>|relabel|<k>"), replacing the single request seed 0) and asserts exactly 5 samples per question before writing. New flags `--only-relabel` (skip P1/P2/P6 and S5) and `--model-path`. Test `scripts/exp/test_E004_relabel.py` (3 passed: 5 n=1 requests with distinct seeds per question; the old single n=5 call fails the count; the output check flags a dir with one question at 1 sample). Output check `scripts/exp/E004_check_relabel.py` (every question exactly 5 samples, labels cover the same ids, n_correct = sum of correct).
- **Rerun (2026-11-22 21:41–21:46, GPUs 4–7):** E004-A step 100, E004-B step 100, E005-O step 20, E005-S step 20, and the start model (Qwen2.5-3B-Instruct snapshot aa8e725…, the E001 model path) → `$DATA_ROOT/logs/E005/eval/start-model/step0/relabel/`. Logs `$DATA_ROOT/logs/E005/relabel_fix/`. All exit 0; check passes for all five (1,000 questions × 5 samples). Old outputs moved (not deleted) to `.../relabel_invalid_streamcut/`.
- **Start-model check against E001** (887 non-ambiguous questions): label counts knowable / unknowable / uncertain E001 278 / 466 / 143, rerun 278 / 466 / 143; per-question label agreement 0.825 (E001 knowable → rerun 239 / 6 / 33; unknowable → 1 / 424 / 41; uncertain → 38 / 36 / 69); mean correct share 0.323 vs 0.320; Spearman of n_correct 0.900. Sample-level identity is not expected (E001 used vLLM's offline LLM with request seed 1).
- **Tables rebuilt** with `scripts/exp/E005_tables.py` (old: `e005_tables_before_relabel_fix.md`, `e005_tidy_before_relabel_fix.csv`). Only Table F changed. Corrected Table F rows (non-ambiguous questions, 887 × 5 samples):

| model | step | samples | share with <search> | share with no <answer> | mean answer words | knowable | unknowable | uncertain |
|---|---|---|---|---|---|---|---|---|
| start model (E001 run) | 0 | 4435 | 0.000 | 0.000 | 9.09 | 278 | 466 | 143 |
| start model (rerun, fixed relabel) | 0 | 4435 | 0.000 | 0.000 | 7.77 | 278 | 466 | 143 |
| E005-O-s0 | 20 | 4435 | 0.000 | 0.000 | 5.87 | 272 | 475 | 140 |
| E005-S-s0 | 20 | 4435 | 0.000 | 0.000 | 4.99 | 289 | 451 | 147 |

### Diagnosis requested 2026-11-22 (after the chain)
**(a) Stop steps.** Per-rollout text and word counts exist only in the rollout dumps, which hold the first 200 rollouts per worker (8 workers × 40 rollouts per step = steps 1–5). No per-rollout data exist for steps 6 onward, so for the stop steps the median, max, share > 20 words and example answers cannot be computed; available are the per-step `reward_steps.jsonl` aggregates (mean, linear-interpolated p95, unclosed share, strict outcome). Max lower bound = (n·mean − (n−k)·p95)/k with k = ⌈0.05 n⌉ answers allowed above p95.

| arm | step | label | n answers | mean words | p95 | unclosed | max ≥ | strict outcome | total |
|---|---|---|---|---|---|---|---|---|---|
| O | 20 / 21 / 22 | knowable | 185 / 160 / 145 | 1.86 / 3.12 / 1.93 | 4.0 / 6.0 / 3.0 | 0 / 0 / 0 | | 0.805 / 0.812 / 0.752 | 0.951 / 0.929 / 0.886 |
| O | **23** | knowable | 135 | 17.59 | 6.3 | 0.0074 (1 of 135) | 224 | 0.770 | 0.919 |
| O | 20 / 21 / 22 | unknowable | 135 / 160 / 175 | 3.96 / 5.39 / 4.73 | 13.3 / 27.0 / 18.9 | 0 / 0 / 0 | | 0.378 / 0.381 / 0.394 | |
| O | **23** | unknowable | 185 | 4.44 | 11.0 | 0 | | 0.281 | 0.495 |
| S | 22 / 23 / 24 | knowable | 145 / 135 / 165 | 2.03 / 2.00 / 2.35 | 4.0 / 3.3 / 3.8 | 0 / 0 / 0 | | 0.676 / 0.837 / 0.861 | 0.810 / 0.968 / 0.985 |
| S | **25** | knowable | 165 | 15.93 | 24.0 | 0.0061 (1 of 165) | 24 | 0.661 | 0.779 |
| S | 22 / 23 / 24 | unknowable | 175 / 185 / 155 | 4.97 / 4.92 / 5.32 | 28.6 / 20.4 / 23.3 | 0 / 0 / 0 | | 0.446 / 0.330 / 0.394 | |
| S | **25** | unknowable | 155 | 7.07 | 44.3 | 0 | 44 | 0.381 | 0.562 |
| C | 10 / 11 / 12 | knowable | 135 / 144 / 140 | 2.10 / 5.07 / 1.95 | 5.3 / 24.6 / 3.0 | 0 / 0 / 0 | | 0.770 / 0.662 / 0.779 | 0.713 / 0.610 / 0.727 |
| C | **13** | knowable | 169 | 20.27 | 37.6 | 0.0059 (1 of 169) | 38 | 0.700 | 0.641 |
| C | 10 / 11 / 12 | unknowable | 182 / 173 / 180 | 4.48 / 13.65 / 2.54 | 15.0 / 32.2 / 6.0 | 0 / 0.0116 / 0 | | 0.205 / 0.154 / 0.272 | |
| C | **13** | unknowable | 145 | 6.70 | 36.2 | 0 | 36 | 0.133 | 0.056 |

Each stop step has exactly one unclosed answer on knowable (all-rollout unclosed share 0.003125 = 1/320 in O, S and C); unclosed spans are counted to the end of the text (environment text stripped). The O step-23 knowable mean (17.59) is above its p95 (6.3).

**(b) Pilot C vs chain C.** Differences in the recorded launch config (`config.txt`, `base_commit.txt`, `agent_loop.yaml`, `code.patch`, `git_status.txt`, hydra log): cost 0.15 vs 0.20; total training steps 3 vs 100; dump_dir; launch time (11:58 vs 14:28, chain C launched 2 min after arm S's training was killed). Code, data, seed, model and all other settings identical. The step-1 batch is the same 64 questions (checked by id from the dumps). Step-1 rollouts are generated before any update, so cost and total steps do not enter step-1 generation or outcome.
Rescoring the step-1 dumps (first 40 lines per worker file) with alias_match_strict reproduces the logged values exactly. Step-1 rollouts with garbled model text (regex `</\s*\.|<\s+/|</ |< answer|</\.` on model segments) and with ≥1 invalid action:

| run | garbled | garbled in turn 1 | ≥1 invalid action | mean model words | kn searched outcome: clean (n) / garbled (n) |
|---|---|---|---|---|---|
| O pilot | 0.069 | 0.056 | 0.037 | 77.3 | 0.948 (58) / 0.667 (3) |
| S pilot | 0.019 | 0.019 | 0.006 | 68.4 | 0.915 (59) / 1.000 (2) |
| C pilot | 0.141 | 0.103 | 0.087 | 78.6 | 0.927 (55) / 0.444 (9) |
| O chain | 0.225 | 0.175 | 0.122 | 115.3 | 0.868 (38) / 0.312 (16) |
| S chain | 0.169 | 0.113 | 0.113 | 97.2 | 0.891 (46) / 0.267 (15) |
| C chain | 0.441 | 0.281 | 0.306 | 151.5 | 0.786 (28) / 0.295 (44) |

Same regex on standalone-vLLM evaluation samples: start model E002 P2 0.007 (5,000 samples); E005-O step 20 P2 0.003. Chain C's step-1 searched-knowable outcome 0.486 (n 72) = 28 clean rollouts at 0.786 and 44 garbled at 0.295; pilot C's 0.859 (n 64) = 55 clean at 0.927 and 9 garbled at 0.444. E004 step-1 dumps on the same regex: A pilot 0.284, B 0.087.

**Table B at step 20 (P1 / P6).** O: P1 AUROC 0.693 [0.656, 0.729], P(s|kn) 0.657, P(s|unk) 0.882; P6 0.660 [0.623, 0.698], 0.437, 0.679. S: P1 0.664 [0.627, 0.698], 0.713, 0.915; P6 0.630 [0.592, 0.669], 0.591, 0.793. (Step 0: P1 0.635 [0.596, 0.670], 0.673, 0.837; P6 0.622 [0.580, 0.662], 0.245, 0.376.)

### Garble diagnosis (requested 2026-11-22, after the relabel fix; no training beyond 1-step probes)
Garbled = regex `</\s*\.|<\s+/|</ |< answer|</\.` on any model segment of a rollout (as in "Diagnosis" above). Outputs under `$DATA_ROOT/logs/E005/garble/`.

**1. Temperature control (standalone vLLM, `scripts/exp/E005_garble_temp.py`, 21:54–21:56, GPUs 4/5/6).** Start model, the 64 step-1 questions × 5 (training prompt from mix50sc), worktree `run_episode` with the training RolloutConfig, same retrieval server; T 1.0, top_p 1.0, top_k −1, max_tokens = 3072 − context (verl rule), per-request md5 seed; engine: AsyncLLMEngine bf16, max_model_len 8192, gpu mem 0.85, prefix caching, engine seed = run seed, no sleep mode.

| seed | garbled | garbled in first model segment | ≥1 invalid | garbled kn / unk | questions with ≥1 garbled (of 64) |
|---|---|---|---|---|---|
| 0 | 0.066 | 0.025 | 0.025 | 0.048 / 0.077 | 16 |
| 1 | 0.059 | 0.028 | 0.044 | 0.048 / 0.067 | 16 |
| 2 | 0.272 | 0.059 | 0.191 | 0.136 / 0.359 | 39 |

(n = 320 each. Seed-2 examples: coherent first turn, then word salad after the information block, e.g. `<think to think id group</th) ... </灵活回答`.)

**2. verl rollout vs standalone engine settings.** verl values from E005-O-s0 train.log config and `verl/workers/rollout/vllm_rollout/vllm_async_server.py`; standalone = E004_eval.py / E005_garble_temp.py (vLLM 0.8.5 V1 log). Same: dtype bfloat16, top_k −1, top_p 1.0, repetition_penalty 1.0, enforce_eager False (CUDA graphs + torch.compile), prefix caching on, chunked prefill on, TP 1, engine seed 0 (E004_eval), PYTORCH_CUDA_ALLOC_CONF unset. Different:

| setting | verl training rollout | standalone |
|---|---|---|
| temperature | 1.0 | 0.7 (E002/E004 eval); 1.0 in item 1 |
| per-request seed | none | md5 per (id, sample, turn) |
| max_model_len | 3072 (prompt 1024 + response 2048) | 8192 |
| per-turn max_tokens | 3072 − context | min(512, 8192 − context) in eval; 3072 − context in item 1 |
| sleep mode (free_cache_engine) | on: wake_up / sleep every step; prefix cache reset before sleep | off |
| weights | load_format dummy, then synced from the FSDP actor (fp32 params → bf16, 512 MB buckets) via ExternalZeroMQDistributedExecutor; model runs in the colocated FSDP worker process | loaded from the safetensors snapshot by vLLM's own worker |
| gpu_memory_utilization | 0.7 (GPU shared with the offloaded FSDP actor) | 0.85 |
| max_num_batched_tokens | 8192 | 2048 (vLLM default for this GPU) |
| max_num_seqs | 1024 | 256 (vLLM default for this GPU) |
| disable_custom_all_reduce | True | False |
| engines | 4 AsyncLLM servers (1 per GPU), 8 agent-loop workers route episodes by least-requests | 1 engine |
| logprobs | 0 (override_generation_config) | none |

**3. Per-worker breakdown (step-1 dumps, first 40 lines per agent-loop worker; workers are CPU processes that send episodes to any of the 4 servers; old dumps do not record the server).** Garbled per worker of 40, chi-square p over workers:

| launch | total | per worker | p |
|---|---|---|---|
| E004-A pilot | 0.284 | 15 10 8 11 11 8 16 12 | 0.39 |
| E005-O pilot | 0.069 | 3 3 3 3 3 2 3 2 | 1.00 |
| E005-S pilot | 0.019 | 0 1 1 1 1 1 0 1 | 0.96 |
| E005-C pilot | 0.141 | 8 6 4 7 5 7 3 5 | 0.77 |
| E005-O chain | 0.225 | 11 6 7 6 10 12 9 11 | 0.57 |
| E005-S chain | 0.169 | 9 5 8 6 5 8 7 6 | 0.91 |
| E005-C chain | 0.441 | 20 18 16 18 14 19 16 20 | 0.86 |

**4. Clean-launch test (`scripts/exp/E005_garble_probe.sh`, `E005_train.sh probe O <i>`: arm O recipe, 1 step, no checkpoint, MAX_TOKENS_PER_GPU=4608, fresh processes, 60 s between).** Preflight and post snapshots (`probe<i>_{preflight,post}.txt`): all 8 GPUs 10 MiB, no compute apps, no ray / vLLM processes, before and after every launch. All exit 0. Rollouts now record the vLLM server dp rank (`server_ranks`; rank r = `async_llm_server_r`, its FSDP worker rank r, i.e. GPU 4+r if ranks map to devices in order — not verified).

| probe | time | garbled (n 320) | garbled first segment | ≥1 invalid | garbled by server rank 0 / 1 / 2 / 3 (of 80 each) | per-step samples garbled (of 50) | kn searched outcome |
|---|---|---|---|---|---|---|---|
| 1 | 21:57–22:03 | 0.181 | 0.103 | 0.116 | 0 / 58 / 0 / 0 | 7 | 0.696 |
| 2 | 22:04–22:10 | 0.325 | 0.209 | 0.244 | 41 / 0 / 0 / 63 | 18 | 0.438 |
| 3 | 22:11–22:16 | 0.125 | 0.103 | 0.059 | 0 / 3 / 0 / 37 | 4 | 0.870 |

**5. Per-step rollout samples (worktree, uncommitted).** `agent/reward.py`: `compute_score_batch(..., sample_path, sample_n=50)` appends 50 rollouts per reward call, chosen with `numpy.default_rng(step)` without replacement, to `<run>/rollout_samples.jsonl` (step, split, row, id, index, labels, data source, score / outcome / arm term, executed searches, answer words, unclosed, full response text); failures are caught and printed. `scripts/train_grpo_3b.sh` passes `sample_path=$OUT/rollout_samples.jsonl`, `sample_n=${SAMPLE_N:-50}`. `agent/verl_agent_loop.py`: the first-200-per-worker dump also records `server_ranks`. Tests: 2 new in `tests/test_reward.py` (`.venv` 66 passed; `.venv-verl` test_reward 26 passed). Probes 1–3 wrote 50 samples each at step 1.

### Garble setting sweep (requested 2026-11-22; standalone, no training except 1-step probes)
Versions: vLLM 0.8.5 (V1 engine), torch 2.6.0+cu124 (CUDA 12.4, cuDNN 9.1), transformers 4.51.3, flash-attn 2.7.4.post1, xformers 0.0.29.post2, triton 3.2.0, Ray 2.47.1, verl 0.5.0 (8fdc4d3f), Python 3.10.12; 4× RTX A5000, driver 550.120.

**Standalone sweep** (`scripts/exp/E005_garble_sweep.sh` → `E005_garble_temp.py --cond`, 22:23–22:37, GPUs 4–7, one job per GPU, fresh process per launch, seeds 0–5 = engine seed and request-seed base; T 1.0, the 64 step-1 questions × 5, training rollout loop and retrieval). Outputs `$DATA_ROOT/logs/E005/garble/sweep/<cond>_s<seed>/`. Bad launch = garbled share > 0.10. Throughput = 320 episodes / generation wall time (engine start excluded; 4 jobs shared the CPU retrieval server, so throughput is noisy). Settings as resolved by vLLM (`summary.json` vllm_config).

| condition | resolved settings (prefix cache / chunked prefill / eager) | garbled share, seeds 0–5 | bad launches | garbled in first model segment, range | episodes/s mean (range) | engine start s |
|---|---|---|---|---|---|---|
| a baseline | on / on / off | 0.309 0.091 0.097 0.078 0.134 0.094 | 2 / 6 | 0.016–0.050 | 9.60 (7.1–13.4) | 64 |
| b prefix caching off | off / on / off | 0.000 ×6 | 0 / 6 | 0.000 | 6.90 (6.3–7.4) | 62 |
| c enforce_eager | on / on / on | 0.100 0.094 0.097 0.069 0.072 0.100 | 0 / 6 | 0.016–0.044 | 11.96 (6.1–15.4) | 13 |
| d chunked prefill off (requested) | on / **on** / off | 0.197 0.084 0.075 0.191 0.200 0.203 | 4 / 6 | 0.025–0.044 | 10.57 (7.7–14.0) | 60 |
| e all three off (requested) | off / **on** / on | 0.000 ×6 | 0 / 6 | 0.000 | 6.35 (5.2–6.8) | 14 |

Deviation: `enable_chunked_prefill=False` is not applied by vLLM 0.8.5's V1 engine (log: "Chunked prefill is enabled with max_num_batched_tokens=2048"; resolved `chunked_prefill_enabled=True`), so d equals the baseline settings and e = prefix caching off + eager.

**verl probes with prefix caching off only** (`PREFIX_CACHING=False bash scripts/exp/E005_garble_probe.sh np`, arm O recipe, 1 step, 22:36–22:55). verl 0.5.0 hard-codes `enable_prefix_caching=True` in `AsyncvLLMServer`; the worktree adds `agent/vllm_server.py` `AsyncvLLMServerNoPrefixCache` (same server, engine arg set False before verl builds the vLLM config; its print appears in each launch log), selected through verl's `rollout.agent.custom_async_server` by `PREFIX_CACHING=False` in `scripts/train_grpo_3b.sh` (config.txt line `vllm: prefix_caching=False enforce_eager=False`). Pre/post snapshots: GPUs 0–7 10 MiB, no compute apps, no ray / vLLM processes. All exit 0.

| probe | garbled (n 320) | per server rank 0 / 1 / 2 / 3 (of 80) | new step metric (same) | ≥1 invalid | per-step samples garbled (of 50) | kn searched outcome | timing_s/gen | timing_s/step |
|---|---|---|---|---|---|---|---|---|
| np1 | 0.000 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0, no warning | 0.019 | 0 | 0.925 | 25.6 | 166.1 |
| np2 | 0.000 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0, no warning | 0.006 | 0 | 0.962 | 23.5 | 162.4 |
| np3 | 0.000 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0, no warning | 0.019 | 0 | 0.893 | 23.2 | 161.9 |

Prefix caching on (probes 1–3 above, same recipe): timing_s/gen 44.7 / 47.1 / 22.9, timing_s/step 193.6 / 209.7 / 166.4.

**Per-server garble metric (worktree, uncommitted).** `agent/garble.py` (regex, events, per-server aggregation); `agent/verl_agent_loop.py` appends one event per episode (server dp rank, garbled, garbled in first model segment) to `<run>/garble/events_<pid>.jsonl` (agent_loop.yaml `garble_dir`); `agent/reward.py` `compute_score_batch(garble_dir, garble_warn=0.10)` reads the events since its last call (byte-offset cursor `<run>/garble/_cursor.json`), logs `garble` {all, by_server: n / garbled / share / garbled_first, warn_servers} in `reward_steps.jsonl`, and if a server's share > 0.10 prints `GARBLE WARNING step <s>: ...` and appends it to `<run>/garble/warnings.log` (no stop). Launcher passes both (`GARBLE_WARN`, default 0.10). Test added (`tests/test_reward.py`; `.venv` 67 passed, `.venv-verl` test_reward 27 passed). Verified in probes np1–3 (320 events per step, per-server counts equal to the dump counts).

## Result      ← you fill (facts only)

Tables copied from `$DATA_ROOT/logs/E005/e005_tables.md`. Plot: `$DATA_ROOT/logs/E005/e005_plot.png`. Tidy CSV: `$DATA_ROOT/logs/E005/e005_tidy.csv`. n = 278 knowable / 466 unknowable / 143 uncertain questions (E001 labels), 5 episodes each. Arm C has step-0 rows only. Table F E005 rows are INVALID (relabel defect, Execution); old tables kept as `e005_tables_before_relabel_fix.md`.

Questions: E001 set, ambiguous excluded; labels = qwen3b-instruct E001 labels. Step 0 = start model (E002 / E003 outputs). AUROC: unknowable = positive, 95% bootstrap CI. Accuracy ± SE over questions.

### Table A (E005-O-s0, P2)

| step | P(s∣kn) | P(s∣unk) | AUROC [CI] | acc kn % (strict) | acc unk % (strict) | acc overall % (strict) | acc kn % (E001) | acc unk % (E001) | acc overall % (E001) | mean answer words | mean searches | parse | cap-hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.489 | 0.609 | 0.587 [0.546, 0.626] | 77.0 ± 2.1 | 32.3 ± 1.7 | 50.4 ± 1.4 | 79.1 ± 1.9 | 33.7 ± 1.7 | 52.1 ± 1.4 | 3.22 | 0.57 | 0.998 | 0.000 |
| 20 | 0.466 | 0.814 | 0.741 [0.706, 0.775] | 81.7 ± 1.9 | 43.8 ± 2.0 | 59.2 ± 1.4 | 84.3 ± 1.7 | 45.4 ± 2.0 | 61.2 ± 1.4 | 3.00 | 0.70 | 0.998 | 0.000 |

### Table A (E005-S-s0, P2)

| step | P(s∣kn) | P(s∣unk) | AUROC [CI] | acc kn % (strict) | acc unk % (strict) | acc overall % (strict) | acc kn % (E001) | acc unk % (E001) | acc overall % (E001) | mean answer words | mean searches | parse | cap-hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.489 | 0.609 | 0.587 [0.546, 0.626] | 77.0 ± 2.1 | 32.3 ± 1.7 | 50.4 ± 1.4 | 79.1 ± 1.9 | 33.7 ± 1.7 | 52.1 ± 1.4 | 3.22 | 0.57 | 0.998 | 0.000 |
| 20 | 0.761 | 0.927 | 0.633 [0.602, 0.664] | 81.1 ± 2.0 | 46.2 ± 2.1 | 61.0 ± 1.4 | 83.9 ± 1.8 | 48.9 ± 2.1 | 63.6 ± 1.4 | 3.92 | 0.93 | 0.999 | 0.000 |

### Table A (E005-C-s0, P2)

| step | P(s∣kn) | P(s∣unk) | AUROC [CI] | acc kn % (strict) | acc unk % (strict) | acc overall % (strict) | acc kn % (E001) | acc unk % (E001) | acc overall % (E001) | mean answer words | mean searches | parse | cap-hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.489 | 0.609 | 0.587 [0.546, 0.626] | 77.0 ± 2.1 | 32.3 ± 1.7 | 50.4 ± 1.4 | 79.1 ± 1.9 | 33.7 ± 1.7 | 52.1 ± 1.4 | 3.22 | 0.57 | 0.998 | 0.000 |

### Table B (prompt shift; step 100 = the arm's last evaluated step if it stopped early)

| run | step | prompt | P(s∣kn) | P(s∣unk) | AUROC [CI] |
|---|---|---|---|---|---|
| E005-O-s0 | 0 | P1 | 0.673 | 0.837 | 0.635 [0.596, 0.670] |
| E005-O-s0 | 0 | P2 | 0.489 | 0.609 | 0.587 [0.546, 0.626] |
| E005-O-s0 | 0 | P6 | 0.245 | 0.376 | 0.622 [0.580, 0.662] |
| E005-O-s0 | 20 | P1 | 0.657 | 0.882 | 0.693 [0.656, 0.729] |
| E005-O-s0 | 20 | P2 | 0.466 | 0.814 | 0.741 [0.706, 0.775] |
| E005-O-s0 | 20 | P6 | 0.437 | 0.679 | 0.660 [0.623, 0.698] |
| E005-S-s0 | 0 | P1 | 0.673 | 0.837 | 0.635 [0.596, 0.670] |
| E005-S-s0 | 0 | P2 | 0.489 | 0.609 | 0.587 [0.546, 0.626] |
| E005-S-s0 | 0 | P6 | 0.245 | 0.376 | 0.622 [0.580, 0.662] |
| E005-S-s0 | 20 | P1 | 0.713 | 0.915 | 0.664 [0.627, 0.698] |
| E005-S-s0 | 20 | P2 | 0.761 | 0.927 | 0.633 [0.602, 0.664] |
| E005-S-s0 | 20 | P6 | 0.591 | 0.793 | 0.630 [0.592, 0.669] |
| E005-C-s0 | 0 | P1 | 0.673 | 0.837 | 0.635 [0.596, 0.670] |
| E005-C-s0 | 0 | P2 | 0.489 | 0.609 | 0.587 [0.546, 0.626] |
| E005-C-s0 | 0 | P6 | 0.245 | 0.376 | 0.622 [0.580, 0.662] |

### Table C (E005-O-s0, P2)

| step | S5 margin AUROC [CI] (n) | probe L24 AUROC [CI] (n) | k coefficient [CI] (episodes, P(search), dropped resamples) | Spearman(margin, k) knowable (n) | Spearman(margin, k) unknowable (n) |
|---|---|---|---|---|---|
| 0 | 0.604 [0.560, 0.644] (744) | 0.850 [0.821, 0.876] (744) | 0.56 [0.10, 1.03] (3720, 0.564, 0) | 0.058 (278) | 0.030 (466) |
| 20 | 0.709 [0.669, 0.747] (744) | 0.854 [0.825, 0.880] (744) | 2.42 [1.89, 2.97] (3720, 0.684, 0) | 0.262 (278) | 0.311 (466) |

### Table C (E005-S-s0, P2)

| step | S5 margin AUROC [CI] (n) | probe L24 AUROC [CI] (n) | k coefficient [CI] (episodes, P(search), dropped resamples) | Spearman(margin, k) knowable (n) | Spearman(margin, k) unknowable (n) |
|---|---|---|---|---|---|
| 0 | 0.604 [0.560, 0.644] (744) | 0.850 [0.821, 0.876] (744) | 0.56 [0.10, 1.03] (3720, 0.564, 0) | 0.058 (278) | 0.030 (466) |
| 20 | 0.619 [0.578, 0.660] (744) | 0.850 [0.819, 0.875] (744) | 1.89 [1.16, 2.59] (3720, 0.865, 0) | 0.087 (278) | 0.055 (466) |

### Table C (E005-C-s0, P2)

| step | S5 margin AUROC [CI] (n) | probe L24 AUROC [CI] (n) | k coefficient [CI] (episodes, P(search), dropped resamples) | Spearman(margin, k) knowable (n) | Spearman(margin, k) unknowable (n) |
|---|---|---|---|---|---|
| 0 | 0.604 [0.560, 0.644] (744) | 0.850 [0.821, 0.876] (744) | 0.56 [0.10, 1.03] (3720, 0.564, 0) | 0.058 (278) | 0.030 (466) |

### Table D (E005-O-s0, training rollouts by training label; search = ≥1 executed search)

| step | label | rollouts | P(search) | outcome 0s / ≥1s | arm term 0s / ≥1s | total 0s / ≥1s | groups | mixed search | mixed search & total | answer words mean | p95 | unclosed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | knowable | 125 | 0.432 | 0.704 / 0.704 | 0.300 / 0.000 | 1.004 / 0.704 | 25 | 14 | 14 | 10.85 | 86.25 | 0.008 |
| 1 | unknowable | 195 | 0.574 | 0.048 / 0.179 | 0.000 / 0.300 | 0.048 / 0.479 | 39 | 27 | 27 | 25.01 | 42.00 | 0.010 |
| 10 | knowable | 135 | 0.489 | 0.754 / 0.848 | 0.300 / 0.000 | 1.054 / 0.848 | 27 | 18 | 18 | 2.39 | 4.30 | 0.000 |
| 10 | unknowable | 185 | 0.622 | 0.100 / 0.374 | 0.000 / 0.300 | 0.100 / 0.674 | 37 | 20 | 20 | 3.74 | 15.80 | 0.000 |
| 20 | knowable | 185 | 0.514 | 0.811 / 0.800 | 0.300 / 0.000 | 1.111 / 0.800 | 37 | 20 | 20 | 1.86 | 4.00 | 0.000 |
| 20 | unknowable | 135 | 0.770 | 0.290 / 0.404 | 0.000 / 0.300 | 0.290 / 0.704 | 27 | 12 | 12 | 3.96 | 13.30 | 0.000 |

### Table D (E005-S-s0, training rollouts by training label; search = ≥1 executed search)

| step | label | rollouts | P(search) | outcome 0s / ≥1s | arm term 0s / ≥1s | total 0s / ≥1s | groups | mixed search | mixed search & total | answer words mean | p95 | unclosed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | knowable | 125 | 0.488 | 0.766 / 0.738 | 0.218 / 0.081 | 0.983 / 0.818 | 25 | 12 | 12 | 4.17 | 19.70 | 0.008 |
| 1 | unknowable | 195 | 0.559 | 0.023 / 0.294 | 0.115 / 0.193 | 0.138 / 0.486 | 39 | 26 | 26 | 7.12 | 30.00 | 0.005 |
| 10 | knowable | 135 | 0.593 | 0.818 / 0.812 | 0.199 / 0.065 | 1.017 / 0.878 | 27 | 20 | 20 | 2.64 | 4.00 | 0.000 |
| 10 | unknowable | 185 | 0.616 | 0.042 / 0.289 | 0.117 / 0.192 | 0.160 / 0.481 | 37 | 21 | 21 | 5.37 | 22.00 | 0.000 |
| 20 | knowable | 185 | 0.676 | 0.783 / 0.728 | 0.218 / 0.107 | 1.001 / 0.835 | 37 | 14 | 14 | 3.38 | 15.70 | 0.000 |
| 20 | unknowable | 135 | 0.859 | 0.105 / 0.414 | 0.199 / 0.164 | 0.304 / 0.578 | 27 | 10 | 10 | 6.11 | 31.30 | 0.000 |

### Table D (E005-C-s0, training rollouts by training label; search = ≥1 executed search)

| step | label | rollouts | P(search) | outcome 0s / ≥1s | arm term 0s / ≥1s | total 0s / ≥1s | groups | mixed search | mixed search & total | answer words mean | p95 | unclosed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | knowable | 125 | 0.576 | 0.604 / 0.486 | 0.000 / -0.258 | 0.604 / 0.228 | 25 | 17 | 17 | 14.70 | 73.05 | 0.018 |
| 1 | unknowable | 195 | 0.549 | 0.023 / 0.168 | 0.000 / -0.239 | 0.023 / -0.071 | 39 | 28 | 28 | 18.33 | 73.50 | 0.000 |
| 10 | knowable | 135 | 0.259 | 0.770 / 0.771 | 0.000 / -0.223 | 0.770 / 0.549 | 27 | 13 | 13 | 2.10 | 5.30 | 0.000 |
| 10 | unknowable | 185 | 0.411 | 0.110 / 0.342 | 0.000 / -0.221 | 0.110 / 0.121 | 37 | 22 | 22 | 4.48 | 14.95 | 0.000 |

### Table E (uncertain questions, P2 search rate by start-model E003 S2 agreement)

| run | step | 0.2–0.4 (n) | 0.6 (n) | 0.8–1.0 (n) |
|---|---|---|---|---|
| E005-O-s0 | 0 | 0.535 (101) | 0.514 (28) | 0.343 (14) |
| E005-O-s0 | 20 | 0.620 (101) | 0.679 (28) | 0.386 (14) |
| E005-S-s0 | 0 | 0.535 (101) | 0.514 (28) | 0.343 (14) |
| E005-S-s0 | 20 | 0.850 (101) | 0.729 (28) | 0.671 (14) |
| E005-C-s0 | 0 | 0.535 (101) | 0.514 (28) | 0.343 (14) |

### Table F (final relabel, E001 closed-book procedure; non-ambiguous questions) — INVALID rows marked; corrected rows under Execution, "Relabel fix"

| model | step | samples | share with <search> | share with no <answer> | mean answer words | knowable | unknowable | uncertain |
|---|---|---|---|---|---|---|---|---|
| start model (E001 run) | 0 | 4435 | 0.000 | 0.000 | 9.09 | 278 | 466 | 143 |
| E005-O-s0 (INVALID) | 20 | 1550 | 0.000 | 0.000 | 8.78 | 72 | 566 | 249 |
| E005-S-s0 (INVALID) | 20 | 1151 | 0.000 | 0.002 | 7.64 | 33 | 567 | 287 |

### Summary (facts)
- All three arms stopped early on the knowable answer-length rule: O at step 23, S at step 25, C (cost 0.20) at step 13. Evaluated checkpoints: O and S at step 20; C none.
- P2, step 0 → 20. O: P(s|kn) 0.489 → 0.466, P(s|unk) 0.609 → 0.814, AUROC 0.587 [0.546, 0.626] → 0.741 [0.706, 0.775]. S: P(s|kn) 0.489 → 0.761, P(s|unk) 0.609 → 0.927, AUROC → 0.633 [0.602, 0.664].
- Strict accuracy, overall, step 20: O 59.2 ± 1.4 %, S 61.0 ± 1.4 % (step 0: 50.4 ± 1.4 %).
- k coefficient (P2): step 0 0.56 [0.10, 1.03]; step 20 O 2.42 [1.89, 2.97], S 1.89 [1.16, 2.59] (3,720 episodes, 0 dropped resamples).
- Spearman(margin, k), step 20: O 0.262 (kn, n 278) / 0.311 (unk, n 466); S 0.087 / 0.055 (step 0: 0.058 / 0.030). Probe L24 AUROC: 0.850 (step 0), 0.854 (O), 0.850 (S).
- Arm C training rollouts, step 1 → 10: P(search) knowable 0.576 → 0.259, unknowable 0.549 → 0.411.
