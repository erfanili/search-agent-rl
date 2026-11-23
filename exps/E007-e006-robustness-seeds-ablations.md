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
## Result      ← you fill (facts only)
