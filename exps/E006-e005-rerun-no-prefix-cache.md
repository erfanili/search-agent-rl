# E006: E005 rerun with prefix caching off
Question: Same as E005: does (C) a per-search cost, (S) a self-consistency decision reward or (O) an oracle decision reward connect the search decision to the model's knowledge, now trained on clean rollouts?
Parent: E005

## Setup
Identical to E005 (start model, P2 context, MIX-50 with sc_agreement, alias_match_strict, arm terms with λ = 0.3, cost = 0.20, recipe, MAX_TOKENS_PER_GPU=4608, PYTORCH_CUDA_ALLOC_CONF unset, checkpoints, disk rules, evaluation, outputs), with these changes:
1. PREFIX_CACHING=False for all training rollouts (agent/vllm_server.py). Record the vLLM, torch and verl versions in the card.
2. Train from scratch. Do NOT resume from E005's step-20 checkpoints.
3. New stop rule, checked from step 10: stop an arm only if, for 3 consecutive steps, either the MEDIAN answer-span words exceed 2× its own steps 1–3 median for either label, or the strict outcome reward is more than 0.15 below its steps 1–3 mean. Cap unclosed answers at 50 words in the length statistics.
4. Garble monitor: log per-server garbled share every step (already built). If any server exceeds 0.10 for 3 consecutive steps, stop the arm and report: that means prefix caching isn't the only cause.
5. Arm C regime check: at arm C's step 1, compute total reward by label and by search choice. If skipping does not score higher than searching on knowable questions, or searching does not score higher than skipping on unknowable questions, stop arm C and report the table. Don't adjust the cost yourself.
6. Use rollout_samples.jsonl (50 per step) for any stop diagnosis.

## Pilot
1 step per arm (garbling and the payoff table are already characterized). Stop if the garbled share is above 0.02 or the parse rate is below 0.95. Otherwise run the chain O → S → C → evaluation without waiting for me.

## Output wanted
Same tables as E005 (A–F and the plot), with E005's step-20 rows included as a reference. Add the per-step garble share (max over servers) to Table D.

## Execution   ← you fill
### Setup and code (2026-11-22)
- **Versions:** vLLM 0.8.5 (V1 engine), torch 2.6.0+cu124 (CUDA 12.4), verl 0.5.0 (`$DATA_ROOT/src/verl`, commit 8fdc4d3f, unmodified), transformers 4.51.3, Ray 2.47.1, flash-attn 2.7.4.post1; 4× RTX A5000 (GPUs 4–7), driver 550.120.
- **Base commit** 359ea18 plus the uncommitted worktree `/home/erfan/sar-alias-reward` (feat/alias-reward); per run `base_commit.txt` (with `PREFIX_CACHING=False`), `code.patch`, `git_status.txt`. Main-repo experiment scripts untracked.
- **Worktree changes since E005** (uncommitted): `agent/vllm_server.py` (AsyncvLLMServerNoPrefixCache, selected by `PREFIX_CACHING=False` in `scripts/train_grpo_3b.sh` through verl's `rollout.agent.custom_async_server`); `agent/garble.py` + `agent/verl_agent_loop.py` + `agent/reward.py` (per-server garble events → `reward_steps.jsonl` `garble`, `GARBLE WARNING` lines in `<run>/garble/warnings.log`); `agent/reward.py` `rollout_samples.jsonl` (50 random rollouts per step); `agent/step_stats.py` answer cells gain `median_words`, `max_words`, `share_over_20`, and `answer_capped` (unclosed spans capped at 50 words) plus per-label capped word lists `answer_words_capped`. Tests: `.venv` 68 passed, `.venv-verl` test_reward 28 passed.
- **Scripts** (`scripts/exp/`): `E006_train.sh` (= E005_train.sh, logs/E006, cost 0.20, `PREFIX_CACHING=False`, pilot = 1 step), `E006_pilot.sh` + `E006_pilot_summary.py`, `E006_stop_watch.py`, `E006_chain.sh`, `E006_eval_all.sh` (`E004_eval.py --exp E006 --no-prefix-cache`), `E006_tables.py`. `E004_eval.py` gains `--no-prefix-cache` (recorded in each run.json as `prefix_caching`).
- **Data / model:** `$DATA_ROOT/verl_data/e005/mix50sc/{train,val}.parquet` (E005), Qwen2.5-3B-Instruct snapshot aa8e725…; retrieval server up (IVF4096,SQ8, 21,015,300 passages).
- **Disk:** 119 GB free at the start; E006 needs ≈170 GB at arm C's peak. Erfan approved deleting (2026-11-22 23:08, logged in `$DATA_ROOT/logs/E005/deletions.log`): E005-O-s0 and E005-S-s0 `ckpt/global_step_20` (FSDP shards + optimizer, 36 GB each) and E004-B-s0 `ckpt/global_step_100` (36 GB). Merged weights (`hf/step_*`) kept. Free after: 226 GB.

### Implementation choices and decisions
1. **Stop rule (card item 3):** a step ≥ 10 is flagged if (a) the median capped answer-span words for knowable or unknowable > 2 × that label's steps 1–3 median, pooled over the answered rollouts of steps 1–3; or (b) the mean strict outcome over all rollouts < the steps 1–3 mean − 0.15 (Erfan: all rollouts). Stop after 3 consecutive flagged steps (either clause on each). Capped words: unclosed spans count min(words, 50). Exit → arm stopped, chain continues (as E005).
2. **Garble stop (item 4):** a step is garble-flagged if any server's garbled share > 0.10 (checked from step 1); 3 consecutive → STOP_GARBLE; the remaining arms are not run and what exists is evaluated (Erfan).
3. **Arm C regime check (item 5):** at chain arm C's step 1 (`--regime`), table of total reward by label × search choice (0 vs ≥ 1 executed searches); if knowable total(0) ≤ total(≥1) or unknowable total(≥1) ≤ total(0): STOP_REGIME, arm C stopped, then evaluation. Not applied to pilot C (reported only).
4. **Evaluation with prefix caching off** (Erfan): `E004_eval.py --no-prefix-cache`. Step 0 and the E005 reference rows are reused from their earlier runs (prefix caching on).
5. **Pilot rule:** garbled share = reward_steps `garble.all.share` at step 1; parse rate from the dumped episodes (E004_pilot_summary definition).
6. Everything else as E005 (deviations 1–9 there carry over; PYTORCH_CUDA_ALLOC_CONF unset).

### Pilots (2026-11-22 23:11–23:28, GPUs 4–7, 1 step each, no checkpoint, prefix caching off)
Run dirs `$DATA_ROOT/logs/E006/E006-{O,S,C}-pilot/` (`pilot_summary.md`). Garbled share from reward_steps `garble` (320 rollouts, 80 per server); parse rate from the 320 dumped episodes.

| arm | step time s | peak mem alloc / reserved GB | truncation | executed-search share all | garbled (per server 0/1/2/3) | parse | capped answer words median kn / unk (max) | flags |
|---|---|---|---|---|---|---|---|---|
| O | 168.6 | 32.7 / 37.6 | 0.000 | 0.544 | 0 / 320 (0/0/0/0) | 1.000 | 2 / 2 (21 / 58) | none |
| S | 163.1 | 32.6 / 37.1 | 0.000 | 0.541 | 0 / 320 (0/0/0/0) | 0.997 | 2 / 2 (30 / 54) | none |
| C | 163.8 | 32.6 / 37.5 | 0.000 | 0.537 | 0 / 320 (0/0/0/0) | 1.000 | 2 / 2 (92 / 35) | none |

Step-1 total reward, 0 executed searches / ≥ 1 (n): O knowable 1.090 (62) / 0.921 (63), unknowable 0.048 (84) / 0.579 (111); S knowable 1.014 (66) / 0.972 (59), unknowable 0.132 (81) / 0.555 (114); C knowable 0.790 (62) / 0.686 (63), unknowable 0.070 (86) / 0.112 (109) (pilot C: skip higher on knowable, search higher on unknowable; the regime check itself applies to the chain's arm C).
All pilots passed; `E006_pilot.sh` started `E006_chain.sh` at 23:28:00 (arm O attempt 1 at 23:28:02, free /data 251 GB).

### Training chain (2026-11-22 23:28 – 2026-11-23 11:31, GPUs 4–7, `E006_chain.sh`, log `$DATA_ROOT/logs/E006/chain.log`)
| run | start | training exit | wall | attempts | stop rule (flagged steps, longest streak) | garbled rollouts (of 32,000), max server share |
|---|---|---|---|---|---|---|
| E006-O-s0 | 23:28:02 | 03:55:22, exit 0 | 4 h 27 min | 1 (MAX_TOKENS_PER_GPU 4608) | 0, 0 | 0, 0.000 |
| E006-S-s0 | 03:57:05 | 08:24:26, exit 0 | 4 h 27 min | 1 (4608) | 0, 0 | 1 (step 29, server 3: 1/80), 0.0125 |
| E006-C-s0 | 08:26:07 | 11:29:28, exit 0 | 3 h 03 min | 1 (4608) | 6 (steps 18, 27, 29, 44, 65, 85; outcome clause, 0.341–0.362 < 0.363), 1 | 0, 0.000 |

- All 3 arms ran 100 steps; no STOP, STOP_GARBLE or STOP_REGIME. Per-step watcher records: `<run>/stop_watch.jsonl`; no `GARBLE WARNING` lines.
- Arm C regime check at step 1 (`E006-C-s0/stop_watch.jsonl`, step 1): total reward 0 / ≥ 1 searches: knowable 0.819 (72) / 0.706 (53), unknowable 0.079 (89) / 0.175 (106); passed.
- Checkpoints: merged weights `$DATA_ROOT/logs/E006/E006-{O,S,C}-s0/hf/step_{20,40,60,80,100}` (32 GB per arm), all exist after the run (checked 2026-11-23 14:20); step-100 merges: finite logits, 3,085,938,688 params, bf16. FSDP shards + optimizer state of every saved step deleted by the chain after the merge was confirmed (as E005); `ckpt/` keeps 76 MB per arm (no shards).
- Run dirs record base commit 359ea18 + worktree feat/alias-reward (uncommitted; `code.patch`, `git_status.txt` 16 lines), `PREFIX_CACHING=False`.

### Evaluation (2026-11-23 11:31–14:04, GPUs 4–7, `E006_eval_all.sh`, prefix caching off)
- 15 checkpoints (3 arms × steps 20–100): `E004_eval.py --exp E006 --no-prefix-cache` (P1/P2/P6, S5, hidden states, relabel) and `E004_hidden.py`, all exit 0. Eval dirs `$DATA_ROOT/logs/E006/eval/E006-{O,S,C}-s0/step{20,40,60,80,100}/`.
- Tables: `.venv-verl/bin/python scripts/exp/E006_tables.py` → `$DATA_ROOT/logs/E006/{e006_tables.md, e006_tidy.csv, e006_plot.png}`.
- Step 0 and E005 step-20 reference rows reused from their earlier runs (prefix caching on in those evaluations; E006 evaluations off). E005-C has no reference row (no step-20 checkpoint).

## Result      ← you fill (facts only)

Tables copied from `$DATA_ROOT/logs/E006/e006_tables.md`. Plot: `$DATA_ROOT/logs/E006/e006_plot.png`. Tidy CSV: `$DATA_ROOT/logs/E006/e006_tidy.csv`. n = 278 knowable / 466 unknowable / 143 uncertain questions (E001 labels), 5 episodes each. Table D shows every 10th step; garbled share there is the step's max over the 4 servers (80 rollouts each).

Questions: E001 set, ambiguous excluded; labels = qwen3b-instruct E001 labels. Step 0 = start model (E002 / E003 outputs). AUROC: unknowable = positive, 95% bootstrap CI. Accuracy ± SE over questions.

### Table A (E006-O-s0, P2)

| step | P(s∣kn) | P(s∣unk) | AUROC [CI] | acc kn % (strict) | acc unk % (strict) | acc overall % (strict) | acc kn % (E001) | acc unk % (E001) | acc overall % (E001) | mean answer words | mean searches | parse | cap-hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.489 | 0.609 | 0.587 [0.546, 0.626] | 77.0 ± 2.1 | 32.3 ± 1.7 | 50.4 ± 1.4 | 79.1 ± 1.9 | 33.7 ± 1.7 | 52.1 ± 1.4 | 3.22 | 0.57 | 0.998 | 0.000 |
| 20 | 0.385 | 0.781 | 0.779 [0.746, 0.810] | 81.4 ± 1.9 | 40.9 ± 1.9 | 57.3 ± 1.4 | 83.8 ± 1.7 | 42.9 ± 1.9 | 59.4 ± 1.4 | 2.74 | 0.67 | 1.000 | 0.000 |
| 40 | 0.255 | 0.779 | 0.829 [0.801, 0.858] | 83.5 ± 1.9 | 41.8 ± 2.0 | 58.7 ± 1.4 | 85.5 ± 1.7 | 43.3 ± 2.0 | 60.4 ± 1.4 | 2.52 | 0.63 | 1.000 | 0.000 |
| 60 | 0.283 | 0.843 | 0.832 [0.803, 0.860] | 84.7 ± 1.8 | 44.1 ± 2.0 | 60.8 ± 1.4 | 86.0 ± 1.7 | 45.5 ± 2.0 | 62.1 ± 1.4 | 2.51 | 0.70 | 1.000 | 0.000 |
| 80 | 0.312 | 0.854 | 0.821 [0.790, 0.848] | 84.9 ± 1.8 | 45.8 ± 2.1 | 62.2 ± 1.4 | 85.5 ± 1.8 | 47.0 ± 2.1 | 63.2 ± 1.4 | 2.07 | 0.70 | 1.000 | 0.000 |
| 100 | 0.301 | 0.848 | 0.808 [0.778, 0.837] | 86.0 ± 1.8 | 46.5 ± 2.1 | 62.8 ± 1.5 | 86.3 ± 1.7 | 47.1 ± 2.1 | 63.4 ± 1.5 | 1.98 | 0.67 | 1.000 | 0.000 |
| E005 ref 20 | 0.466 | 0.814 | 0.741 [0.706, 0.775] | 81.7 ± 1.9 | 43.8 ± 2.0 | 59.2 ± 1.4 | 84.3 ± 1.7 | 45.4 ± 2.0 | 61.2 ± 1.4 | 3.00 | 0.70 | 0.998 | 0.000 |

### Table A (E006-S-s0, P2)

| step | P(s∣kn) | P(s∣unk) | AUROC [CI] | acc kn % (strict) | acc unk % (strict) | acc overall % (strict) | acc kn % (E001) | acc unk % (E001) | acc overall % (E001) | mean answer words | mean searches | parse | cap-hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.489 | 0.609 | 0.587 [0.546, 0.626] | 77.0 ± 2.1 | 32.3 ± 1.7 | 50.4 ± 1.4 | 79.1 ± 1.9 | 33.7 ± 1.7 | 52.1 ± 1.4 | 3.22 | 0.57 | 0.998 | 0.000 |
| 20 | 0.594 | 0.870 | 0.718 [0.682, 0.752] | 81.6 ± 2.0 | 44.4 ± 2.0 | 60.1 ± 1.4 | 84.0 ± 1.8 | 46.5 ± 2.0 | 62.3 ± 1.4 | 2.99 | 0.80 | 0.998 | 0.000 |
| 40 | 0.446 | 0.873 | 0.770 [0.738, 0.804] | 84.7 ± 1.8 | 45.4 ± 2.0 | 61.3 ± 1.4 | 86.8 ± 1.6 | 47.0 ± 2.0 | 63.0 ± 1.4 | 2.46 | 0.77 | 0.997 | 0.000 |
| 60 | 0.391 | 0.875 | 0.797 [0.765, 0.827] | 83.2 ± 1.9 | 47.2 ± 2.0 | 62.3 ± 1.4 | 85.6 ± 1.8 | 49.1 ± 2.0 | 64.4 ± 1.4 | 2.77 | 0.78 | 1.000 | 0.000 |
| 80 | 0.286 | 0.824 | 0.816 [0.782, 0.845] | 84.5 ± 1.9 | 44.8 ± 2.0 | 61.1 ± 1.5 | 86.1 ± 1.8 | 46.3 ± 2.0 | 62.5 ± 1.4 | 2.22 | 0.64 | 1.000 | 0.000 |
| 100 | 0.213 | 0.759 | 0.818 [0.787, 0.846] | 86.0 ± 1.7 | 44.5 ± 2.0 | 61.5 ± 1.4 | 88.1 ± 1.6 | 46.1 ± 2.0 | 63.2 ± 1.4 | 2.41 | 0.57 | 1.000 | 0.000 |
| E005 ref 20 | 0.761 | 0.927 | 0.633 [0.602, 0.664] | 81.1 ± 2.0 | 46.2 ± 2.1 | 61.0 ± 1.4 | 83.9 ± 1.8 | 48.9 ± 2.1 | 63.6 ± 1.4 | 3.92 | 0.93 | 0.999 | 0.000 |

### Table A (E006-C-s0, P2)

| step | P(s∣kn) | P(s∣unk) | AUROC [CI] | acc kn % (strict) | acc unk % (strict) | acc overall % (strict) | acc kn % (E001) | acc unk % (E001) | acc overall % (E001) | mean answer words | mean searches | parse | cap-hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.489 | 0.609 | 0.587 [0.546, 0.626] | 77.0 ± 2.1 | 32.3 ± 1.7 | 50.4 ± 1.4 | 79.1 ± 1.9 | 33.7 ± 1.7 | 52.1 ± 1.4 | 3.22 | 0.57 | 0.998 | 0.000 |
| 20 | 0.013 | 0.065 | 0.586 [0.564, 0.608] | 82.4 ± 1.8 | 13.4 ± 1.2 | 41.4 ± 1.4 | 83.8 ± 1.7 | 13.7 ± 1.2 | 42.1 ± 1.4 | 2.13 | 0.04 | 1.000 | 0.000 |
| 40 | 0.012 | 0.005 | 0.492 [0.480, 0.501] | 83.8 ± 1.9 | 10.8 ± 1.2 | 40.3 ± 1.5 | 85.0 ± 1.8 | 10.9 ± 1.2 | 40.9 ± 1.5 | 1.95 | 0.01 | 1.000 | 0.000 |
| 60 | 0.000 | 0.000 | 0.500 [0.500, 0.500] | 86.1 ± 1.6 | 10.6 ± 1.2 | 41.3 ± 1.5 | 87.2 ± 1.5 | 10.6 ± 1.2 | 41.7 ± 1.5 | 2.00 | 0.00 | 1.000 | 0.000 |
| 80 | 0.000 | 0.000 | 0.500 [0.500, 0.500] | 87.1 ± 1.6 | 11.7 ± 1.3 | 42.7 ± 1.5 | 88.4 ± 1.5 | 11.8 ± 1.3 | 43.3 ± 1.5 | 2.04 | 0.00 | 1.000 | 0.000 |
| 100 | 0.000 | 0.000 | 0.500 [0.500, 0.500] | 87.6 ± 1.6 | 12.5 ± 1.3 | 43.5 ± 1.5 | 89.0 ± 1.5 | 12.6 ± 1.3 | 44.3 ± 1.5 | 2.16 | 0.00 | 1.000 | 0.000 |

### Table B (prompt shift; step 100 = the arm's last evaluated step if it stopped early)

| run | step | prompt | P(s∣kn) | P(s∣unk) | AUROC [CI] |
|---|---|---|---|---|---|
| E006-O-s0 | 0 | P1 | 0.673 | 0.837 | 0.635 [0.596, 0.670] |
| E006-O-s0 | 0 | P2 | 0.489 | 0.609 | 0.587 [0.546, 0.626] |
| E006-O-s0 | 0 | P6 | 0.245 | 0.376 | 0.622 [0.580, 0.662] |
| E006-O-s0 | 100 | P1 | 0.462 | 0.917 | 0.788 [0.758, 0.819] |
| E006-O-s0 | 100 | P2 | 0.301 | 0.848 | 0.808 [0.778, 0.837] |
| E006-O-s0 | 100 | P6 | 0.152 | 0.682 | 0.817 [0.788, 0.845] |
| E006-S-s0 | 0 | P1 | 0.673 | 0.837 | 0.635 [0.596, 0.670] |
| E006-S-s0 | 0 | P2 | 0.489 | 0.609 | 0.587 [0.546, 0.626] |
| E006-S-s0 | 0 | P6 | 0.245 | 0.376 | 0.622 [0.580, 0.662] |
| E006-S-s0 | 100 | P1 | 0.486 | 0.896 | 0.770 [0.737, 0.801] |
| E006-S-s0 | 100 | P2 | 0.213 | 0.759 | 0.818 [0.787, 0.846] |
| E006-S-s0 | 100 | P6 | 0.148 | 0.544 | 0.776 [0.743, 0.808] |
| E006-C-s0 | 0 | P1 | 0.673 | 0.837 | 0.635 [0.596, 0.670] |
| E006-C-s0 | 0 | P2 | 0.489 | 0.609 | 0.587 [0.546, 0.626] |
| E006-C-s0 | 0 | P6 | 0.245 | 0.376 | 0.622 [0.580, 0.662] |
| E006-C-s0 | 100 | P1 | 0.004 | 0.019 | 0.526 [0.511, 0.540] |
| E006-C-s0 | 100 | P2 | 0.000 | 0.000 | 0.500 [0.500, 0.500] |
| E006-C-s0 | 100 | P6 | 0.001 | 0.000 | 0.499 [0.495, 0.503] |
| E005-O-s0 | 20 | P1 | 0.657 | 0.882 | 0.693 [0.656, 0.729] |
| E005-O-s0 | 20 | P2 | 0.466 | 0.814 | 0.741 [0.706, 0.775] |
| E005-O-s0 | 20 | P6 | 0.437 | 0.679 | 0.660 [0.623, 0.698] |
| E005-S-s0 | 20 | P1 | 0.713 | 0.915 | 0.664 [0.627, 0.698] |
| E005-S-s0 | 20 | P2 | 0.761 | 0.927 | 0.633 [0.602, 0.664] |
| E005-S-s0 | 20 | P6 | 0.591 | 0.793 | 0.630 [0.592, 0.669] |

### Table C (E006-O-s0, P2)

| step | S5 margin AUROC [CI] (n) | probe L24 AUROC [CI] (n) | k coefficient [CI] (episodes, P(search), dropped resamples) | Spearman(margin, k) knowable (n) | Spearman(margin, k) unknowable (n) |
|---|---|---|---|---|---|
| 0 | 0.604 [0.560, 0.644] (744) | 0.850 [0.821, 0.876] (744) | 0.56 [0.10, 1.03] (3720, 0.564, 0) | 0.058 (278) | 0.030 (466) |
| 20 | 0.760 [0.725, 0.795] (744) | 0.856 [0.827, 0.881] (744) | 2.60 [2.11, 3.08] (3720, 0.633, 0) | 0.303 (278) | 0.454 (466) |
| 40 | 0.801 [0.767, 0.832] (744) | 0.854 [0.826, 0.881] (744) | 4.16 [3.56, 4.78] (3720, 0.583, 0) | 0.460 (278) | 0.516 (466) |
| 60 | 0.800 [0.769, 0.833] (744) | 0.852 [0.822, 0.878] (744) | 4.62 [4.01, 5.31] (3720, 0.634, 0) | 0.445 (278) | 0.466 (466) |
| 80 | 0.789 [0.754, 0.825] (744) | 0.856 [0.826, 0.881] (744) | 4.75 [4.06, 5.52] (3720, 0.651, 0) | 0.482 (278) | 0.470 (466) |
| 100 | 0.793 [0.758, 0.828] (744) | 0.860 [0.833, 0.885] (744) | 4.89 [4.12, 5.75] (3720, 0.644, 0) | 0.519 (278) | 0.373 (466) |
| E005 ref 20 | 0.709 [0.669, 0.747] (744) | 0.854 [0.825, 0.880] (744) | 2.42 [1.89, 2.97] (3720, 0.684, 0) | 0.262 (278) | 0.311 (466) |

### Table C (E006-S-s0, P2)

| step | S5 margin AUROC [CI] (n) | probe L24 AUROC [CI] (n) | k coefficient [CI] (episodes, P(search), dropped resamples) | Spearman(margin, k) knowable (n) | Spearman(margin, k) unknowable (n) |
|---|---|---|---|---|---|
| 0 | 0.604 [0.560, 0.644] (744) | 0.850 [0.821, 0.876] (744) | 0.56 [0.10, 1.03] (3720, 0.564, 0) | 0.058 (278) | 0.030 (466) |
| 20 | 0.670 [0.629, 0.712] (743) | 0.858 [0.828, 0.883] (744) | 2.54 [1.98, 3.07] (3720, 0.767, 0) | 0.201 (278) | 0.378 (465) |
| 40 | 0.696 [0.656, 0.740] (743) | 0.855 [0.825, 0.881] (744) | 5.09 [4.47, 5.78] (3720, 0.713, 0) | 0.516 (278) | 0.202 (465) |
| 60 | 0.736 [0.695, 0.775] (744) | 0.866 [0.839, 0.890] (744) | 4.64 [4.04, 5.29] (3720, 0.694, 0) | 0.412 (278) | -0.019 (466) |
| 80 | 0.803 [0.768, 0.838] (743) | 0.860 [0.831, 0.886] (744) | 4.53 [3.95, 5.17] (3720, 0.623, 0) | 0.449 (278) | 0.394 (465) |
| 100 | 0.816 [0.782, 0.849] (743) | 0.857 [0.828, 0.883] (744) | 4.96 [4.30, 5.72] (3720, 0.555, 0) | 0.441 (278) | 0.385 (465) |
| E005 ref 20 | 0.619 [0.578, 0.660] (744) | 0.850 [0.819, 0.875] (744) | 1.89 [1.16, 2.59] (3720, 0.865, 0) | 0.087 (278) | 0.055 (466) |

### Table C (E006-C-s0, P2)

| step | S5 margin AUROC [CI] (n) | probe L24 AUROC [CI] (n) | k coefficient [CI] (episodes, P(search), dropped resamples) | Spearman(margin, k) knowable (n) | Spearman(margin, k) unknowable (n) |
|---|---|---|---|---|---|
| 0 | 0.604 [0.560, 0.644] (744) | 0.850 [0.821, 0.876] (744) | 0.56 [0.10, 1.03] (3720, 0.564, 0) | 0.058 (278) | 0.030 (466) |
| 20 | 0.724 [0.687, 0.759] (744) | 0.860 [0.830, 0.884] (744) | 1.71 [0.78, 2.82] (3720, 0.046, 0) | 0.292 (278) | 0.385 (466) |
| 40 | 0.460 [0.413, 0.507] (740) | 0.861 [0.833, 0.886] (744) | -1.48 [-3.15, 0.51] (3720, 0.008, 0) | -0.139 (276) | 0.025 (464) |
| 60 | 0.673 [0.634, 0.712] (744) | 0.861 [0.832, 0.885] (744) | — [—, —] (3720, 0.000, 1000) | 0.309 (278) | 0.173 (466) |
| 80 | 0.677 [0.639, 0.716] (744) | 0.856 [0.827, 0.882] (744) | — [—, —] (3720, 0.000, 1000) | 0.299 (278) | 0.180 (466) |
| 100 | 0.718 [0.681, 0.755] (744) | 0.859 [0.831, 0.885] (744) | — [—, —] (3720, 0.000, 1000) | 0.319 (278) | 0.230 (466) |

### Table D (E006-O-s0, training rollouts by training label; search = ≥1 executed search)

| step | label | rollouts | P(search) | outcome 0s / ≥1s | arm term 0s / ≥1s | total 0s / ≥1s | groups | mixed search | mixed search & total | answer words mean | p95 | capped median | unclosed | garbled share, max over servers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | knowable | 125 | 0.464 | 0.806 / 0.914 | 0.300 / 0.000 | 1.106 / 0.914 | 25 | 13 | 13 | 2.09 | 3.00 | 2.00 | 0.000 | 0.000 |
| 1 | unknowable | 195 | 0.595 | 0.038 / 0.353 | 0.000 / 0.300 | 0.038 / 0.653 | 39 | 29 | 29 | 3.67 | 13.00 | 2.00 | 0.000 | 0.000 |
| 10 | knowable | 135 | 0.533 | 0.825 / 0.819 | 0.300 / 0.000 | 1.125 / 0.819 | 27 | 19 | 19 | 1.85 | 4.00 | 1.00 | 0.000 | 0.000 |
| 10 | unknowable | 185 | 0.622 | 0.171 / 0.339 | 0.000 / 0.300 | 0.171 / 0.639 | 37 | 20 | 20 | 4.13 | 13.80 | 2.00 | 0.000 | 0.000 |
| 20 | knowable | 185 | 0.459 | 0.850 / 0.694 | 0.300 / 0.000 | 1.150 / 0.694 | 37 | 27 | 27 | 1.99 | 4.00 | 1.00 | 0.000 | 0.000 |
| 20 | unknowable | 135 | 0.689 | 0.238 / 0.452 | 0.000 / 0.300 | 0.238 / 0.752 | 27 | 14 | 14 | 3.70 | 13.60 | 2.00 | 0.000 | 0.000 |
| 30 | knowable | 145 | 0.352 | 0.904 / 0.745 | 0.300 / 0.000 | 1.204 / 0.745 | 29 | 16 | 16 | 1.74 | 4.80 | 1.00 | 0.000 | 0.000 |
| 30 | unknowable | 175 | 0.806 | 0.176 / 0.440 | 0.000 / 0.300 | 0.176 / 0.740 | 35 | 17 | 17 | 2.79 | 6.00 | 2.00 | 0.000 | 0.000 |
| 40 | knowable | 190 | 0.411 | 0.929 / 0.705 | 0.300 / 0.000 | 1.229 / 0.705 | 38 | 14 | 14 | 2.23 | 3.00 | 2.00 | 0.000 | 0.000 |
| 40 | unknowable | 130 | 0.654 | 0.267 / 0.435 | 0.000 / 0.300 | 0.267 / 0.735 | 26 | 13 | 13 | 2.61 | 6.00 | 2.00 | 0.000 | 0.000 |
| 50 | knowable | 155 | 0.297 | 0.881 / 0.543 | 0.300 / 0.000 | 1.181 / 0.543 | 31 | 14 | 14 | 2.92 | 5.00 | 1.00 | 0.000 | 0.000 |
| 50 | unknowable | 165 | 0.679 | 0.245 / 0.402 | 0.000 / 0.300 | 0.245 / 0.702 | 33 | 11 | 11 | 4.64 | 31.80 | 2.00 | 0.006 | 0.000 |
| 60 | knowable | 205 | 0.317 | 0.871 / 0.692 | 0.300 / 0.000 | 1.171 / 0.692 | 41 | 11 | 11 | 1.96 | 4.00 | 1.00 | 0.000 | 0.000 |
| 60 | unknowable | 115 | 0.930 | 0.125 / 0.589 | 0.000 / 0.300 | 0.125 / 0.889 | 23 | 5 | 5 | 2.24 | 4.00 | 2.00 | 0.000 | 0.000 |
| 70 | knowable | 170 | 0.294 | 0.925 / 0.640 | 0.300 / 0.000 | 1.225 / 0.640 | 34 | 9 | 9 | 1.96 | 5.00 | 1.00 | 0.000 | 0.000 |
| 70 | unknowable | 150 | 0.827 | 0.192 / 0.516 | 0.000 / 0.300 | 0.192 / 0.816 | 30 | 4 | 4 | 4.29 | 6.00 | 2.00 | 0.000 | 0.000 |
| 80 | knowable | 140 | 0.371 | 0.920 / 0.577 | 0.300 / 0.000 | 1.220 / 0.577 | 28 | 5 | 5 | 1.80 | 3.00 | 2.00 | 0.000 | 0.000 |
| 80 | unknowable | 180 | 0.717 | 0.216 / 0.434 | 0.000 / 0.300 | 0.216 / 0.734 | 36 | 10 | 10 | 2.02 | 3.00 | 2.00 | 0.000 | 0.000 |
| 90 | knowable | 165 | 0.261 | 0.836 / 0.721 | 0.300 / 0.000 | 1.136 / 0.721 | 33 | 6 | 6 | 1.85 | 5.00 | 1.00 | 0.000 | 0.000 |
| 90 | unknowable | 155 | 0.877 | 0.158 / 0.412 | 0.000 / 0.300 | 0.158 / 0.712 | 31 | 6 | 6 | 2.21 | 5.00 | 2.00 | 0.000 | 0.000 |
| 100 | knowable | 180 | 0.356 | 0.888 / 0.766 | 0.300 / 0.000 | 1.188 / 0.766 | 36 | 10 | 10 | 1.68 | 3.00 | 1.00 | 0.000 | 0.000 |
| 100 | unknowable | 140 | 0.843 | 0.000 / 0.441 | 0.000 / 0.300 | 0.000 / 0.741 | 28 | 4 | 4 | 2.20 | 5.00 | 2.00 | 0.000 | 0.000 |
| E005 ref 20 | knowable | 185 | 0.514 | 0.811 / 0.800 | 0.300 / 0.000 | 1.111 / 0.800 | 37 | 20 | 20 | 1.86 | 4.00 | — | 0.000 | — |
| E005 ref 20 | unknowable | 135 | 0.770 | 0.290 / 0.404 | 0.000 / 0.300 | 0.290 / 0.704 | 27 | 12 | 12 | 3.96 | 13.30 | — | 0.000 | — |

### Table D (E006-S-s0, training rollouts by training label; search = ≥1 executed search)

| step | label | rollouts | P(search) | outcome 0s / ≥1s | arm term 0s / ≥1s | total 0s / ≥1s | groups | mixed search | mixed search & total | answer words mean | p95 | capped median | unclosed | garbled share, max over servers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | knowable | 125 | 0.472 | 0.773 / 0.864 | 0.213 / 0.075 | 0.985 / 0.940 | 25 | 12 | 12 | 1.96 | 3.00 | 2.00 | 0.000 | 0.000 |
| 1 | unknowable | 195 | 0.538 | 0.011 / 0.314 | 0.098 / 0.178 | 0.109 / 0.493 | 39 | 23 | 23 | 3.61 | 10.90 | 2.00 | 0.000 | 0.000 |
| 10 | knowable | 135 | 0.489 | 0.710 / 0.788 | 0.209 / 0.068 | 0.919 / 0.856 | 27 | 20 | 20 | 2.80 | 4.00 | 1.00 | 0.000 | 0.000 |
| 10 | unknowable | 185 | 0.622 | 0.100 / 0.357 | 0.120 / 0.193 | 0.220 / 0.550 | 37 | 21 | 21 | 4.86 | 19.60 | 2.00 | 0.000 | 0.000 |
| 20 | knowable | 185 | 0.611 | 0.778 / 0.699 | 0.205 / 0.101 | 0.983 / 0.801 | 37 | 20 | 20 | 2.54 | 5.80 | 1.00 | 0.000 | 0.000 |
| 20 | unknowable | 135 | 0.807 | 0.115 / 0.431 | 0.180 / 0.164 | 0.295 / 0.595 | 27 | 13 | 13 | 4.81 | 18.70 | 2.00 | 0.000 | 0.000 |
| 30 | knowable | 145 | 0.290 | 0.942 / 0.738 | 0.215 / 0.113 | 1.157 / 0.851 | 29 | 12 | 12 | 1.67 | 3.80 | 1.00 | 0.000 | 0.000 |
| 30 | unknowable | 175 | 0.743 | 0.200 / 0.485 | 0.191 / 0.188 | 0.391 / 0.673 | 35 | 18 | 18 | 2.58 | 6.00 | 2.00 | 0.000 | 0.000 |
| 40 | knowable | 190 | 0.474 | 0.900 / 0.656 | 0.218 / 0.099 | 1.118 / 0.754 | 38 | 14 | 14 | 2.65 | 6.00 | 2.00 | 0.000 | 0.000 |
| 40 | unknowable | 130 | 0.746 | 0.303 / 0.454 | 0.155 / 0.164 | 0.458 / 0.618 | 26 | 6 | 6 | 2.68 | 6.00 | 2.00 | 0.000 | 0.000 |
| 50 | knowable | 155 | 0.316 | 0.896 / 0.592 | 0.212 / 0.164 | 1.108 / 0.756 | 31 | 5 | 5 | 1.81 | 3.00 | 1.00 | 0.000 | 0.000 |
| 50 | unknowable | 165 | 0.636 | 0.250 / 0.457 | 0.149 / 0.197 | 0.399 / 0.654 | 33 | 7 | 7 | 2.73 | 4.00 | 2.00 | 0.000 | 0.000 |
| 60 | knowable | 205 | 0.429 | 0.872 / 0.670 | 0.228 / 0.130 | 1.100 / 0.800 | 41 | 14 | 14 | 1.81 | 4.80 | 1.00 | 0.000 | 0.000 |
| 60 | unknowable | 115 | 0.887 | 0.462 / 0.529 | 0.120 / 0.204 | 0.582 / 0.733 | 23 | 3 | 3 | 2.19 | 5.00 | 2.00 | 0.000 | 0.000 |
| 70 | knowable | 170 | 0.359 | 0.862 / 0.689 | 0.199 / 0.109 | 1.061 / 0.798 | 34 | 17 | 17 | 2.05 | 5.00 | 1.00 | 0.000 | 0.000 |
| 70 | unknowable | 150 | 0.947 | 0.125 / 0.585 | 0.135 / 0.189 | 0.260 / 0.774 | 30 | 5 | 5 | 4.10 | 7.00 | 2.00 | 0.000 | 0.000 |
| 80 | knowable | 140 | 0.386 | 0.872 / 0.704 | 0.202 / 0.154 | 1.074 / 0.858 | 28 | 6 | 6 | 2.19 | 5.00 | 2.00 | 0.000 | 0.000 |
| 80 | unknowable | 180 | 0.667 | 0.350 / 0.500 | 0.215 / 0.175 | 0.565 / 0.675 | 36 | 9 | 9 | 2.47 | 3.00 | 2.00 | 0.000 | 0.000 |
| 90 | knowable | 165 | 0.085 | 0.868 / 0.571 | 0.200 / 0.141 | 1.067 / 0.713 | 33 | 6 | 6 | 1.84 | 5.00 | 1.00 | 0.000 | 0.000 |
| 90 | unknowable | 155 | 0.710 | 0.178 / 0.436 | 0.189 / 0.189 | 0.367 / 0.626 | 31 | 5 | 5 | 2.25 | 5.00 | 2.00 | 0.000 | 0.000 |
| 100 | knowable | 180 | 0.133 | 0.853 / 0.708 | 0.202 / 0.148 | 1.054 / 0.856 | 36 | 5 | 5 | 1.76 | 4.00 | 1.00 | 0.000 | 0.000 |
| 100 | unknowable | 140 | 0.636 | 0.216 / 0.416 | 0.156 / 0.187 | 0.372 / 0.603 | 28 | 11 | 11 | 2.88 | 6.00 | 2.00 | 0.000 | 0.000 |
| E005 ref 20 | knowable | 185 | 0.676 | 0.783 / 0.728 | 0.218 / 0.107 | 1.001 / 0.835 | 37 | 14 | 14 | 3.38 | 15.70 | — | 0.000 | — |
| E005 ref 20 | unknowable | 135 | 0.859 | 0.105 / 0.414 | 0.199 / 0.164 | 0.304 / 0.578 | 27 | 10 | 10 | 6.11 | 31.30 | — | 0.000 | — |

### Table D (E006-C-s0, training rollouts by training label; search = ≥1 executed search)

| step | label | rollouts | P(search) | outcome 0s / ≥1s | arm term 0s / ≥1s | total 0s / ≥1s | groups | mixed search | mixed search & total | answer words mean | p95 | capped median | unclosed | garbled share, max over servers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | knowable | 125 | 0.424 | 0.819 / 0.906 | 0.000 / -0.200 | 0.819 / 0.706 | 25 | 14 | 14 | 2.10 | 3.80 | 2.00 | 0.000 | 0.000 |
| 1 | unknowable | 195 | 0.544 | 0.079 / 0.387 | 0.000 / -0.211 | 0.079 / 0.175 | 39 | 23 | 23 | 4.08 | 14.20 | 2.00 | 0.000 | 0.000 |
| 10 | knowable | 135 | 0.193 | 0.789 / 0.923 | 0.000 / -0.215 | 0.789 / 0.708 | 27 | 10 | 10 | 2.14 | 4.00 | 1.00 | 0.000 | 0.000 |
| 10 | unknowable | 185 | 0.314 | 0.150 / 0.379 | 0.000 / -0.207 | 0.150 / 0.172 | 37 | 24 | 24 | 3.30 | 11.00 | 2.00 | 0.000 | 0.000 |
| 20 | knowable | 185 | 0.016 | 0.808 / 0.000 | 0.000 / -0.200 | 0.808 / -0.200 | 37 | 2 | 2 | 2.04 | 4.00 | 1.00 | 0.000 | 0.000 |
| 20 | unknowable | 135 | 0.059 | 0.094 / 0.375 | 0.000 / -0.225 | 0.094 / 0.150 | 27 | 5 | 5 | 2.42 | 5.30 | 2.00 | 0.000 | 0.000 |
| 30 | knowable | 145 | 0.000 | 0.793 / — | 0.000 / — | 0.793 / — | 29 | 0 | 0 | 1.52 | 3.00 | 1.00 | 0.000 | 0.000 |
| 30 | unknowable | 175 | 0.006 | 0.075 / 1.000 | 0.000 / -0.200 | 0.075 / 0.800 | 35 | 1 | 1 | 2.15 | 4.00 | 2.00 | 0.000 | 0.000 |
| 40 | knowable | 190 | 0.000 | 0.821 / — | 0.000 / — | 0.821 / — | 38 | 0 | 0 | 1.77 | 4.00 | 1.00 | 0.000 | 0.000 |
| 40 | unknowable | 130 | 0.000 | 0.154 / — | 0.000 / — | 0.154 / — | 26 | 0 | 0 | 2.38 | 5.00 | 2.00 | 0.000 | 0.000 |
| 50 | knowable | 155 | 0.000 | 0.806 / — | 0.000 / — | 0.806 / — | 31 | 0 | 0 | 1.82 | 4.00 | 1.00 | 0.000 | 0.000 |
| 50 | unknowable | 165 | 0.000 | 0.127 / — | 0.000 / — | 0.127 / — | 33 | 0 | 0 | 2.18 | 4.00 | 2.00 | 0.000 | 0.000 |
| 60 | knowable | 205 | 0.000 | 0.805 / — | 0.000 / — | 0.805 / — | 41 | 0 | 0 | 1.80 | 5.00 | 1.00 | 0.000 | 0.000 |
| 60 | unknowable | 115 | 0.009 | 0.149 / 1.000 | 0.000 / -0.200 | 0.149 / 0.800 | 23 | 1 | 1 | 2.10 | 5.00 | 2.00 | 0.000 | 0.000 |
| 70 | knowable | 170 | 0.000 | 0.847 / — | 0.000 / — | 0.847 / — | 34 | 0 | 0 | 1.75 | 3.00 | 1.50 | 0.000 | 0.000 |
| 70 | unknowable | 150 | 0.000 | 0.073 / — | 0.000 / — | 0.073 / — | 30 | 0 | 0 | 2.42 | 5.00 | 2.00 | 0.000 | 0.000 |
| 80 | knowable | 140 | 0.000 | 0.829 / — | 0.000 / — | 0.829 / — | 28 | 0 | 0 | 2.22 | 6.00 | 2.00 | 0.000 | 0.000 |
| 80 | unknowable | 180 | 0.000 | 0.178 / — | 0.000 / — | 0.178 / — | 36 | 0 | 0 | 2.29 | 3.00 | 2.00 | 0.000 | 0.000 |
| 90 | knowable | 165 | 0.000 | 0.782 / — | 0.000 / — | 0.782 / — | 33 | 0 | 0 | 1.98 | 6.00 | 1.00 | 0.000 | 0.000 |
| 90 | unknowable | 155 | 0.000 | 0.097 / — | 0.000 / — | 0.097 / — | 31 | 0 | 0 | 2.46 | 5.30 | 2.00 | 0.000 | 0.000 |
| 100 | knowable | 180 | 0.000 | 0.817 / — | 0.000 / — | 0.817 / — | 36 | 0 | 0 | 1.86 | 4.00 | 2.00 | 0.000 | 0.000 |
| 100 | unknowable | 140 | 0.000 | 0.093 / — | 0.000 / — | 0.093 / — | 28 | 0 | 0 | 2.11 | 5.00 | 2.00 | 0.000 | 0.000 |

### Table E (uncertain questions, P2 search rate by start-model E003 S2 agreement)

| run | step | 0.2–0.4 (n) | 0.6 (n) | 0.8–1.0 (n) |
|---|---|---|---|---|
| E006-O-s0 | 0 | 0.535 (101) | 0.514 (28) | 0.343 (14) |
| E006-O-s0 | 100 | 0.572 (101) | 0.436 (28) | 0.314 (14) |
| E006-S-s0 | 0 | 0.535 (101) | 0.514 (28) | 0.343 (14) |
| E006-S-s0 | 100 | 0.436 (101) | 0.257 (28) | 0.357 (14) |
| E006-C-s0 | 0 | 0.535 (101) | 0.514 (28) | 0.343 (14) |
| E006-C-s0 | 100 | 0.000 (101) | 0.000 (28) | 0.000 (14) |
| E005-O-s0 | 20 | 0.620 (101) | 0.679 (28) | 0.386 (14) |
| E005-S-s0 | 20 | 0.850 (101) | 0.729 (28) | 0.671 (14) |

### Table F (final relabel, E001 closed-book procedure; non-ambiguous questions)

| model | step | samples | share with <search> | share with no <answer> | mean answer words | knowable | unknowable | uncertain |
|---|---|---|---|---|---|---|---|---|
| start model (E001 run) | 0 | 4435 | 0.000 | 0.000 | 9.09 | 278 | 466 | 143 |
| E006-O-s0 | 100 | 4435 | 0.000 | 0.000 | 4.06 | 266 | 512 | 109 |
| E006-S-s0 | 100 | 4435 | 0.000 | 0.002 | 5.07 | 264 | 517 | 106 |
| E006-C-s0 | 100 | 4435 | 0.000 | 0.000 | 4.69 | 285 | 483 | 119 |
| E005-O-s0 | 20 | 4435 | 0.000 | 0.000 | 5.87 | 272 | 475 | 140 |
| E005-S-s0 | 20 | 4435 | 0.000 | 0.000 | 4.99 | 289 | 451 | 147 |

### Summary (facts)
- All 3 arms trained 100 steps with no stop; 1 garbled rollout in 96,000 training rollouts (S step 29).
- P2 AUROC at step 100 [95% CI]: O 0.808 [0.778, 0.837], S 0.818 [0.787, 0.846], C 0.500 (no searches); step 0 0.587 [0.546, 0.626]; E005 step-20 refs O 0.741, S 0.633. Peak over steps: O 0.832 (step 60), S 0.818 (step 100).
- P2 P(search) knowable / unknowable at step 100: O 0.301 / 0.848, S 0.213 / 0.759, C 0.000 / 0.000 (C ≤ 0.012 from step 40 on).
- P2 overall strict accuracy at step 100 (± SE): O 62.8 ± 1.5, S 61.5 ± 1.4, C 43.5 ± 1.5; step 0 50.4 ± 1.4. Unknowable strict: O 46.5 ± 2.1, S 44.5 ± 2.0, C 12.5 ± 1.3 (step 0 32.3 ± 1.7).
- Prompt shift (Table B), step 100 AUROC P1 / P6: O 0.788 / 0.817, S 0.770 / 0.776, C 0.526 / 0.499.
- Table C at step 100: S5 margin AUROC O 0.793, S 0.816, C 0.718 (step 0 0.604); probe L24 AUROC 0.857–0.860 for all arms (step 0 0.850); k coefficient O 4.89 [4.12, 5.75], S 4.96 [4.30, 5.72], C undefined (P(search) 0).
- Answer length: capped median answer words 1–2 for both labels at every reported step in all arms; P2 mean answer words at step 100: O 1.98, S 2.41, C 2.16.
- Table F (final relabel, step 100) knowable / unknowable / uncertain: O 266 / 512 / 109, S 264 / 517 / 106, C 285 / 483 / 119; start model 278 / 466 / 143.
