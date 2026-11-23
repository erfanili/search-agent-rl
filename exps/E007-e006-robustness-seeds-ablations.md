# E007: Robustness of E006: seeds and ablations
Question: Do E006's results (S and O raise held-out decision AUROC to about 0.81 and hold under prompt shift; C collapses) replicate across seeds, and do they depend on the per-question knowledge signal itself, the decision-reward strength, or the strict outcome reward alone?
Parent: E006

## Setup
Everything as in E006 (start model, P2 context, MIX-50 with sc_agreement, alias_match_strict, recipe, PREFIX_CACHING=False, MAX_TOKENS_PER_GPU=4608, stop rule, garble monitor, checkpoint/disk rules), except as listed per run. Run from the same worktree. Save base commit, code.patch and git status per run as before. SEED sets data order and rollout sampling (as in train_grpo_3b.sh).

## Runs, in this order
Group 1, seeds (main result):
 1. E007-S-s1: arm S, seed 1
 2. E007-S-s2: arm S, seed 2
 3. E007-O-s1: arm O, seed 1
 4. E007-O-s2: arm O, seed 2
Group 2, controls:
 5. E007-N-s0: outcome only (alias_match_strict, no decision term, no cost; λ = 0). The clean outcome-only control. E004 used the lenient reward and had garbling.
 6. E007-Sshuf-s0: arm S with sc_agreement randomly permuted across training questions (seed 0; save the permutation). Same reward structure and the same distribution of targets, but the per-question link to knowledge is broken.
Group 3, strength:
 7. E007-S-l01-s0: arm S with λ = 0.1
 8. E007-S-l05-s0: arm S with λ = 0.5
Group 4, optional, only if time remains:
 9. E007-C-s1: arm C, seed 1

Each run is 100 steps. If any run stops (stop rule, garble, OOM after retry), record it, evaluate what exists, and continue with the next run.

## Checkpoints
Save at 20, 40, 60, 80, 100, but keep merged weights only for steps 20, 60 and 100 (delete the merged 40 and 80 once step 100 is confirmed). Delete shards and optimizer state as in E006.

## Evaluation (steps 0, 20, 60, 100 per run; step 0 reused)
Same as E006, with prefix caching off: P2, P1 and P6 at T=0.7 with 5 samples; P(s|kn), P(s|unk), AUROC [bootstrap CI], strict accuracy overall / knowable / unknowable; under P2 also the S5 margin AUROC, probe L24 AUROC, k coefficient [CI], and Spearman(margin, k) within knowable and within unknowable.
Also report the Spearman(margin, k) values for E006 O, S and C at steps 20–100. They were computed but not reported.

## Analysis-only baselines (no training; existing outputs)
 A. SC-gated retrieval: on the E001 eval set, the qwen3b-instruct policy "search iff start-model held-out agreement (E003 S2) < τ". Use the E002 P2 forced-depth runs (forced 0 and forced 1, 5 samples each) for per-question accuracy under each action. For τ ∈ {0.4, 0.6, 0.8, 1.0}, report the search rate on knowable and unknowable, the AUROC, and the expected strict accuracy. Note: this policy costs 5 extra closed-book generations per question.
 B. Probe-gated retrieval: the same, using the start model's out-of-fold layer-24 probe score from E003, with thresholds set to match each τ's overall search rate.
 C. Matched comparison: for E006-S, E006-O and every E007 S/O run at step 100, the SC-gated and probe-gated accuracy at the SAME overall search rate as that run.

## Output wanted
1. Table A (seeds): for S and O, rows = seeds 0 (E006), 1, 2, and mean ± SD and the 95% t-interval across seeds. Columns at step 100: P(s|kn), P(s|unk), AUROC under P2 / P1 / P6, strict accuracy overall / unknowable, within-label Spearman kn / unk. The same for the peak AUROC step.
2. Table B (controls and strength): rows = E006-S, N, Sshuf, S-λ0.1, S-λ0.5 (plus E006-O and E006-C for reference); the same columns, at steps 20, 60 and 100.
3. Table C (baselines): A and B over τ; C as the matched comparison.
4. Table D (training dynamics): per run, every 10 steps: search share by label, mean outcome and total reward by label and search choice, the stop-rule statistics, and the max per-server garble share.
5. Plot: held-out P2 AUROC against step, all runs; seeds of the same arm in the same color.
6. Files: $DATA_ROOT/logs/E007/<run id>/, $DATA_ROOT/logs/E007/eval/, and a tidy CSV.

## Pilot
1 step of N and of Sshuf only (the new reward variants). Stop if garbled > 0.02 or parse < 0.95. Otherwise run the whole queue in the order above without waiting for me.

## Execution   ← you fill
### Setup (2026-11-23)
- Code: worktree `/home/erfan/sar-alias-reward` (feat/alias-reward), as E006; per run `base_commit.txt`, `code.patch`, `git_status.txt`. Runs launched before Erfan's commit of the worktree record base 359ea18 + patch; later runs record the new HEAD (b5a683e, if committed then) and its diff.
- Scripts (`scripts/exp/`): `E007_train.sh` (run spec from the run ID: arm, SEED, λ, data), `E007_shuf_data.py`, `E007_pilot.sh`, `E007_chain.sh`, `E007_eval.sh`, `E007_tables.py`. Stop rules: `E006_stop_watch.py` (unchanged).
- Inputs present: E005 mix50sc data; E002 qwen3b-instruct P2-f0 / P2-f1 samples (1,000 questions, 5 samples each); E003 S2 samples; k = E003 out-of-fold L24 probe scores (`$DATA_ROOT/logs/E006/k_scores.jsonl`). Retrieval server up (21,015,300 passages). Free /data 392 GB.
- Sshuf data: `$DATA_ROOT/verl_data/e007/mix50sc_shuf/` (`E007_shuf_data.py`): sc_agreement permuted over the 3,200 training rows, numpy default_rng(0).permutation(3200) (2 fixed points, 2,513 rows changed value); permutation saved in `permutation.jsonl`. AUROC of (1 − a), unknowable vs knowable: 0.750 original, 0.498 permuted; mean a knowable / unknowable 0.690 / 0.417 → 0.553 / 0.554; value multiset identical.

### Decisions and implementation choices
1. Run C-s1 (optional) is queued after run 8 (Erfan, 2026-11-23).
2. Evaluation after each group (Erfan, 2026-11-23): G1 (S-s1, S-s2, O-s1, O-s2), G2 (N, Sshuf), G3 (S-l01, S-l05), G4 (C-s1); GPUs 4–7 shared queue; `E004_eval.py --exp E007 --no-prefix-cache` + `E004_hidden.py`; relabel at the last evaluated step (as E006).
3. Seeds: SEED = 1 / 2 sets verl `data.seed` and `rollout.seed` (train_grpo_3b.sh); seed-0 runs use SEED=0 as E006.
4. N: `REWARD_ARM=none` (outcome alias_match_strict only; arm term 0).
5. A stopped run (stop rule, garble, 2nd OOM, disk) is recorded and the chain continues (card); unlike E006, a garble stop does not stop the chain. A stopped run keeps all merged steps; evaluated: 20 and 60 if present plus its last confirmed step.
6. Checkpoints: after hf/step_100 is confirmed, global_step_100 shards + optimizer state and merged hf/step_40, hf/step_80 are deleted (card).
7. Arm C regime check (E006 item 5) applied to C-s1 at step 1.
8. Table A peak AUROC step: chosen among steps 20, 60, 100 for every run, E006 seeds included (the steps evaluated in E007).
9. Baselines: gated-policy AUROC = AUROC of the 0/1 search decision; expected strict accuracy = per question acc(forced 1) if search else acc(forced 0), strict = alias_match_strict on E002 P2-f0 / P2-f1 samples; matched comparison: ties at the agreement boundary search with a common fractional probability so the expected overall rate equals the run's P2 rate (probe: fractional boundary question).
10. Card note: E006 Spearman(margin, k) values were already in E006's Table C; repeated here as Table E.

### Pilots (2026-11-23 17:40–17:51, GPUs 4–7, 1 step, no checkpoint)
| run | garbled (per server 0/1/2/3) | parse | capped answer words median kn / unk (max) | total reward 0 / ≥1 searches: kn; unk (n) | flags |
|---|---|---|---|---|---|
| E007-N-pilot | 0 / 320 (0/0/0/0) | 1.000 | 2 / 2 (43 / 81) | 0.824 (51) / 0.892 (74); 0.056 (90) / 0.362 (105) | none |
| E007-Sshuf-pilot | 0 / 320 (0/0/0/0) | 1.000 | 2 / 2 (7 / 19) | 0.928 (70) / 1.041 (55); 0.226 (92) / 0.540 (103) | none |

Both passed; `E007_pilot.sh` started `E007_chain.sh` at 17:51:41 (E007-S-s1 attempt 1 at 17:51:42, free /data 392 GB).

## Result      ← you fill (facts only)
