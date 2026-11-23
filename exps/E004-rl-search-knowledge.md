# E004: Can RL connect the search decision to the model's knowledge signal?
Question: When qwen3b-instruct is trained with outcome-reward GRPO on easy QA that is balanced between knowable and unknowable questions, starting from a context where p(search) ≈ 0.5, does the search decision's AUROC for knowable vs unknowable rise over training, does it hold under prompt shift, and how does the knowable/unknowable mix change the overall search rate?
Parent: E003

## Setup
- Start model: Qwen/Qwen2.5-3B-Instruct. Training context: the exact E002 P2 prompt (I-NEU + D-MIX), in the chat template.
- Training data (disjoint from the E001 evaluation set):
  1. Candidate pool: 4,000 PopQA questions from quintiles 4–5 that are NOT in data/e001_easy_1k.jsonl, plus 4,000 unique TriviaQA questions from rc.nocontext train split. Seed 0.
  2. Label them with the E001 procedure on the start model (the exact closed-book prompt and decoding of E001 rev 2, 5 samples at T=0.7): knowable = ≥3/5, unknowable = 0/5. Drop uncertain questions, and drop questions where any sample asks for clarification.
  3. Build two training sets from the labelled pool:
     - MIX-50: 50% knowable / 50% unknowable, balanced within each source as far as possible.
     - MIX-20: 20% knowable / 80% unknowable.
     Same size for both (use the largest size the pool supports for MIX-50, up to 3,200; MIX-20 is the same size). Save as data/e004_train_mix50.jsonl and data/e004_train_mix20.jsonl, with the label in each row.
- Reward: binary correctness using the E001 rule (normalized alias match, exact or substring, on the parsed <answer> span; no answer = 0). Add it to agent/reward.py as a new reward name (e.g. alias_match) in a worktree (../sar-alias-reward) with a test, for me to merge before the pilot. No search cost, no format bonus.
- Recipe: as L001 / L006: GRPO, 64 prompts × 5 rollouts, mini-batch 32, lr 1e-6, warmup 6 steps, KL loss 0.001 (low_var_kl), rollout temperature 1.0, at most 3 searches with the forced answer after the 3rd (the max-depth setting from ../sar-max-depth), retrieval as in E002. 100 steps.
- Arms:
  - A: MIX-50, seed 0 (main arm)
  - B: MIX-20, seed 0 (data-mix arm)
  Run IDs: E004-A-s0, E004-B-s0. Run them in parallel if the GPUs allow; otherwise A first.
- Checkpoints: save at steps 20, 40, 60, 80 and 100. After each save, merge to HF weights and confirm the merged weights load. You may delete that checkpoint's optimizer state only after the merge is confirmed. Never delete merged weights.
- Training-time logging (per step, split by the row's knowable/unknowable label): share of rollouts with ≥1 search; mean reward at 0 searches vs ≥1 search; number of groups mixed in search (some rollouts search, some don't); number of groups mixed in both search and reward.

## Evaluation (step 0 = the start model, then each saved checkpoint, for both arms)
- Data: data/e001_easy_1k.jsonl, ambiguous excluded. Labels: qwen3b-instruct's E001 labels. Also relabel at step 100 with the E001 procedure on each arm's final model, and report the final-step tables under both labelings.
- Prompts: P2 (training context), plus P1 and P6 (prompt shift). T=0.7, 5 samples, as in E002.
- Metrics per cell: P(≥1 search | knowable), P(≥1 search | unknowable), AUROC of the per-question search rate with bootstrap 95% CI, accuracy overall / knowable / unknowable (± SE).
- Under P2 only: the S5 margin AUROC (E003 method), and the S4 probe AUROC at layer 24 (E003 method, all-questions version). Run the probe at steps 0 and 100 at least, and at every checkpoint if it's cheap.

## Pilot first
3 training steps of arm A, no checkpoint. Report: step time, peak GPU memory, share of rollouts with ≥1 search at step 1 (split by label), parse rate, and malformed-search rate. Stop if the step-1 search share is outside 0.3–0.7 overall, or if parse rate is below 95%.

## Output wanted
1. Table A (per arm): rows = steps 0, 20, …, 100; columns = P(search|knowable), P(search|unknowable), AUROC [CI], accuracy knowable, accuracy unknowable, all under P2.
2. Table B (prompt shift, per arm): AUROC and P(search|knowable) / P(search|unknowable) at steps 0 and 100 under P1, P2 and P6.
3. Table C (per arm): S5 margin AUROC and S4 probe AUROC by step.
4. Table D (training dynamics, per arm, every 10 steps): the logged quantities above, split by label.
5. Plots: Table A's AUROC and both search rates against step, both arms on one figure.
6. Files: $DATA_ROOT/logs/E004/<run id>/ (training logs, merged checkpoints), $DATA_ROOT/logs/E004/eval/<run id>/step<k>/<prompt>/samples.jsonl, and one tidy CSV of all evaluation cells.

## Execution   ← you fill
### Pre-run checks (2026-11-21)
- Start model `Qwen/Qwen2.5-3B-Instruct@aa8e7253` (local). PopQA test (rev 098765c7) and TriviaQA rc.nocontext (rev 0f7faf33) are in the HF cache. The retrieval server is up (:8000, same index as E002).
- Recipe reference: L006 arm A (`$DATA_ROOT/logs/grpo_qwen2.5-3b_full_E1A_em_20261118_102220`): 64×5, mini-batch 32, GPUs 4–7, ~210 s per step, 34 GB peak allocated.
- GPUs: only 4–7 are Erfan's, so the arms run one after the other (A, then B).
- Disk: 138 GB free on `/data`. A full FSDP checkpoint is ~40 GB, merged HF weights ~6.2 GB.

### Decisions (Erfan, 2026-11-21)
1. Shared-code changes are built in worktree `../sar-alias-reward` (branch `feat/alias-reward`, base 359ea18) and left uncommitted. The pilot and both arms run FROM that worktree. Each run directory holds `base_commit.txt`, `code.patch` (`git diff` of the worktree) and `git_status.txt`. Erfan merges after the runs.
2. After a merged checkpoint is confirmed to load, the FSDP shards and optimizer state of earlier checkpoints are deleted. The latest full checkpoint is kept for resuming. Merged weights are never deleted.
3. Training uses 2,048 total tokens per episode (as in L006), with no per-turn cap and MAX_PROMPT = 1,024. Evaluation uses E002's settings (4,096 total, 512 per turn). The pilot also reports truncation (episodes at 2,048) and stops above 2%.
4. In-training validation is off (TEST_FREQ = −1, VAL_BEFORE_TRAIN = False).
5. MIX-50: per source, equal knowable and unknowable counts; PopQA up to 800 + 800, TriviaQA fills the rest. MIX-20: the same per-source sizes, 20/80 within each source.
6. 100 steps × 64 prompts = 2 passes over 3,200 rows (intended).
7. Step 0 reuses E002 (qwen3b-instruct P1/P2/P6 samples) and E003 (qwen3b-instruct S5 margin; the layer-24 probe is refit with the same method). Same model, prompts, decoding and code.
8. The layer-24 probe runs at every checkpoint, refit with the E003 method (all-questions version).

### Shared-code changes (worktree `../sar-alias-reward`, uncommitted)
- `agent/reward.py`: new reward `alias_match` = the E001 rule. Environment text stripped → FIRST `<answer>` span (an unclosed `<answer>` → text to the end) → cover match; no `<answer>` = 0. Logged as `alias_match` under every metric. `e004_label` added to `LABEL_KEYS`.
- `agent/step_stats.py`: per-label `share_search`, `mean_score_0search`, `mean_score_search`, `n_groups`, `n_groups_mixed_search`, `n_groups_mixed_search_and_score`.
- `scripts/train_grpo_3b.sh`: accepts `DEPTH_POLICY=max:k`; new overrides `MAX_TURNS`, `RUN_DIR`, `VAL_BEFORE_TRAIN`.
- `tests/test_reward.py`: `test_alias_match_e001_rule`, `test_step_stats_per_label_search_and_group_counts`, `test_batch_logs_e004_label`. The existing no-answer test skips the unclosed-`<answer>` case for `alias_match`, since the E001 rule pays it. 53/53 tests pass (`.venv`); `test_reward.py` + `test_rollout_contract.py` 36/36 (`.venv-verl`).
- Note: main's working tree has uncommitted edits to `agent/reward.py`, `scripts/train_grpo_3b.sh` and `tests/test_reward.py` (the restored reward arms `acc`, `em_hit`, `em_multi`). The worktree is based on HEAD and does not contain them, so merging needs reconciling with those edits.
- The worktree has symlinks `.venv` / `.venv-verl` → the main repo's venvs and a copy of `configs/paths.local.env`, all gitignored.

### Training data (2026-11-21)
- `scripts/exp/E004_build_data.py pool | label | mix`. Card paths `data/e004_train_mix{50,20}.jsonl` → `$DATA_ROOT/verl_data/e004/mix50/e004_train_mix50.jsonl` and `.../mix20/e004_train_mix20.jsonl`, with verl `train.parquet` / `val.parquet` next to them (val = the first 64 train rows; required by the launcher, never used).
- Pool (`$DATA_ROOT/verl_data/e004/pool.jsonl`, sha256 0b8051fa…):
  - 4,000 PopQA from quintiles 4–5 (E001 edges), drawn from the 5,207 not in the E001 set;
  - 4,000 TriviaQA rc.nocontext train questions (first row per question_id; 76,523 unique);
  - all drawn with `default_rng(0)`.
- Labels: the E001 rev-2 procedure on the start model (GPU 4, 410 s), in `pool_labels.jsonl` / `pool_samples.jsonl`. Counts after dropping 1,095 uncertain and 96 knowable/unknowable questions with a clarification sample (E001 `CLARIFY_RE`):
  - PopQA: 714 knowable, 2,799 unknowable;
  - TriviaQA: 1,517 knowable, 1,779 unknowable.
- MIX-50 (n = 3,200, sha256 182b3254…): PopQA 714 knowable + 714 unknowable, TriviaQA 886 + 886. PopQA knowable is the limit.
- MIX-20 (n = 3,200, sha256 f64d3dda…): PopQA 286 + 1,142, TriviaQA 354 + 1,418.
- Training prompt: the E002 P2 text as one user message. The agent loop's token ids equal `tok.encode(E002 make_prompt("P2", chat))` for all 3,200 MIX-50 rows; 615–695 tokens.

### Scripts
- `scripts/exp/E004_train.sh pilot|A|B`: runs `scripts/train_grpo_3b.sh` from the worktree with the card recipe and writes the provenance files. Full arms also start `scripts/exp/E004_ckpt_watch.py`.
- `scripts/exp/E004_ckpt_watch.py`: merge → load check (bf16 on CPU, parameter count, finite logits; `hf/step_N/LOAD_OK.json`) → delete earlier shards and optimizer state.
- `scripts/exp/E004_pilot_summary.py`: step time, memory, truncation, search share by label, parse and malformed rates (from the agent-loop dumps).
### Pilot (2026-11-21)
- `bash scripts/exp/E004_train.sh pilot`: arm A recipe, 3 steps, no checkpoint, GPUs 4–7, run from `../sar-alias-reward`. Run dir `$DATA_ROOT/logs/E004/E004-A-pilot/`, with `base_commit.txt` (359ea18 + worktree path), `code.patch`, `git_status.txt`, `train.log`, `reward_steps.jsonl`, `rollouts/`, and the launcher's own `git.diff` / `config.txt`. Summary: `scripts/exp/E004_pilot_summary.py`.

| step | step time s | max mem allocated GB (verl metric) | max mem reserved GB | truncation (episodes at 2,048) | P(≥1 search) | knowable | unknowable | mean reward | reward knowable | reward unknowable |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 199.7 | 34.44 | 37.47 | 0.016 | 0.534 | 0.496 | 0.559 | 0.356 | 0.664 | 0.159 |
| 2 | 163.4 | 34.48 | 37.47 | 0.006 | 0.519 | 0.561 | 0.479 | 0.494 | 0.755 | 0.248 |
| 3 | 152.3 | 34.48 | 37.50 | 0.000 | 0.491 | 0.419 | 0.558 | 0.588 | 0.826 | 0.364 |

- Memory: verl's `perf/max_memory_*` metric, the same metric L006 logged (34.2 GB). It exceeds the 24 GB card memory; the actor's parameters and optimizer are offloaded.
- Rollout dumps (all 960 pilot episodes):
  - parse rate 0.961 (`<answer>` in model text, E002 rule);
  - finish: answer 921, turn_budget 32, length 7;
  - malformed (share of episodes with ≥1 free turn of that type): empty_query 0.002, unclosed_search 0.026, multi_search 0.005, search_and_answer 0.020.
- Mean response length 495 / 413 / 371 tokens; rollout generation 47 / 43 / 36 s; actor update 120 / 96 / 93 s per step.
- Stop rules: none met (step-1 search share 0.534 within 0.3–0.7; parse rate 0.961 ≥ 0.95; maximum truncation 0.016 ≤ 0.02).

### Full runs
- Started 2026-11-21 with `bash scripts/exp/E004_all.sh` (log `$DATA_ROOT/logs/E004/all.log`): arm A (`E004-A-s0`), then arm B (`E004-B-s0`), each with its checkpoint watcher; then the evaluation queue `scripts/exp/E004_eval_all.sh` (log `$DATA_ROOT/logs/E004/eval/launch.log`).

### Answer-span logging and step-20 check (Erfan, 2026-11-21)
- **Request:** log the mean and 95th-percentile word count of the parsed answer span per step, plus the share of answers that are unclosed, split by label. Check at step 20 against steps 1–3. Stop arm A and report before arm B if the mean answer length has more than doubled or the unclosed share is above 5%. Add these to Table D.
- **Timing:** the request arrived after arm A had finished step 20 (16:17). The checkpoint `ckpt/global_step_20` was saved and its merge confirmed (`hf/step_20/LOAD_OK.json`, 16:18). The running process could not start the logging. Its rollout dumps hold only the first 200 episodes per worker, about steps 1–5, so steps 6–20 had no rollout text.
- **Code change (worktree, uncommitted):**
  - `agent/reward.py`: `answer_span_info` measures the span alias_match reads. That is the first `<answer>` span with environment text stripped; an unclosed answer runs to the end of the text. Words are counted by whitespace split. "Unclosed" means no `</answer>` after the first `<answer>`.
  - `compute_score_batch` now writes an `answer` field to `reward_steps.jsonl` through `agent/step_stats.py:answer_stats`. It holds n_answer, n_no_answer, mean_words, p95_words (linear interpolation) and share_unclosed (over rollouts with an `<answer>`), for all rollouts and per `e004_label`.
  - New test `test_batch_logs_answer_span_stats`. Tests pass: 54/54 in `.venv`, 37/37 in `.venv-verl` (test_reward + test_rollout_contract).
- **Deviation, resume:** at 16:20 I stopped arm A (`E004_all.sh`, the training and the watcher). It was in step 21, and that step's update was discarded. Arm A was then resumed from `ckpt/global_step_20` with the logging (`RESUME_FROM`: weights, optimizer, LR schedule and dataloader state from `data.pt`).
  - The resumed run writes reward_steps step numbers from 21.
  - Steps 1–20 have no answer-span log.
  - Rollout sampling after the resume is not bit-identical to an uninterrupted run.
- **Part-1 files kept in `$DATA_ROOT/logs/E004/E004-A-s0/`:**
  - `train.part1.log`, `config.part1.txt`, `git.part1.diff`, `hydra.part1/`, `rollouts.part1/`, `ckpt_watch.part1.log`;
  - `reward_steps.part1.jsonl`: all 21 lines, including the step-21 line scored before the stop;
  - `reward_steps.jsonl` keeps steps 1–20 and gets the resumed steps appended.
- **Resume provenance:** `base_commit.resume.txt`, `code.resume.patch`, `git_status.resume.txt`. The original `base_commit.txt`, `code.patch` and `git_status.txt` describe steps 1–20.
- **Check:** `scripts/exp/E004_answer_check.py`, run with `--watch --kill`. Output in `answer_check.json`, log in `answer_check.log`.
  - **Checked step:** 21, the first logged step. Its rollouts come from the step-20 checkpoint.
  - **Steps 1–3 reference:** all 960 dumped episodes of `E004-A-pilot`, the same recipe and seed covering exactly steps 1–3. They are pooled, since the dumps carry no step number. Labels are joined by question from mix50. The reference uses the same span rule applied to the dump text with special tokens removed.
  - **Reference values:**

    | label | mean words | p95 words | unclosed share | answers |
    |---|---|---|---|---|
    | all | 11.42 | 33.9 | 0.002 | 923 |
    | knowable | 6.43 | 21.0 | 0.000 | 421 |
    | unknowable | 15.60 | 35.9 | 0.004 | 502 |

  - **Cross-check:** arm A's own part-1 dumps (1,600 episodes, about steps 1–5): mean 9.93 / 8.54 / 11.11 words (all / knowable / unknowable); unclosed 0.004 / 0.006 / 0.002.
  - **Stop rule as implemented:** mean words above 2× the reference, or unclosed share above 0.05. It is checked separately for all rollouts, knowable and unknowable; any one triggers it. On a stop, the arm A training is killed and `STOP` is written.
- **Chain:** `scripts/exp/E004_all_resume.sh` (log `$DATA_ROOT/logs/E004/all_resume.log`) replaces `E004_all.sh`. It runs `E004_train.sh A-resume`, which also starts the checkpoint watcher for steps 40–100 and the check. Arm B starts only if `answer_check.json` exists with `"stop": false`; the evaluation queue follows.
- **Table D:** `E004_tables.py` adds answers, answer words mean, answer words p95 and unclosed share. Arm A gets a "1–3 (pilot dumps)" reference row and a step-21 row; steps 1–20 show "—". Arm B logs from step 1.
- **Disk (projection, 2026-11-21 16:25):** 96 GB free on `/data`. A full checkpoint is about 36 GB and a merge about 6.2 GB.
  - After arm A, about 71 GB will be free (step-100 full checkpoint plus 5 merges kept).
  - Arm B's step-40 save would need 36 GB with about 29 GB free, because step 20's shards are kept until step 40's merge is confirmed.
  - Raised with Erfan; nothing deleted.
- **Step-21 check result (2026-11-21 16:26): passed, no flags.** Values at step 21 (n = 320 rollouts):

  | label | answers | mean words | p95 words | unclosed share |
  |---|---|---|---|---|
  | all | 319 | 4.53 | 24.0 | 0.000 |
  | knowable | 159 | 3.18 | 6.3 | 0.000 |
  | unknowable | 160 | 5.87 | 31.2 | 0.000 |

### Disk decision (Erfan, 2026-11-21)
- **Approved:** candidate 1. After arm A's `hf/step_100` is confirmed to load, delete arm A's `ckpt/global_step_100/actor/{model,optim,extra_state}_world_size_*.pt` before arm B starts, automatically in the chain.
- **Not to be deleted:** any L006-era run. `grpo_qwen2.5-3b_full_E1A_em_20261118_102220` holds the grpo-e1a-s60 checkpoint used in E001–E003, and the others back L cards.
- **If disk runs short during arm B:** stop and ask.
- **Implementation:** `scripts/exp/E004_chain2.sh` (log `$DATA_ROOT/logs/E004/chain2.log`) took over at 17:04 from `E004_all_resume.sh`. I killed only the chain's bash process; arm A's training and watcher were not touched. The chain:
  - waits for arm A's training and watcher to end;
  - requires `hf/step_100/LOAD_OK.json`, then deletes the step-100 shards and optimizer state, logging the file count and GB;
  - requires the answer check's `"stop": false`.
- **Disk guard:** NEED = the size of arm A's step-100 shards plus the size of its merged model, plus 3 GB.
  - Before arm B: if free space is below NEED, arm B is not started and `STOP_DISK_before_B` is written.
  - After each of arm B's confirmed checkpoints at steps 20–80, 3 minutes after the load check (when the watcher frees the earlier one): if free space is below NEED, arm B's training is killed, keeping its latest full checkpoint, `E004-B-s0/STOP_DISK` is written and the chain stops without evaluation.
  - Nothing else is deleted.
- **Projection (96 GB free at 17:04):**
  - About 107 GB free when arm B starts.
  - Free space after each of arm B's cleanups: about 65, 59, 52 and 46 GB (steps 20, 40, 60, 80).
  - NEED is about 45 GB, so the step-80 check has about 1 GB of slack. If other users write to `/data`, the guard may stop arm B after step 80.
  - Update (2026-11-21 17:15): Erfan deleted an unrelated project folder (`/data/erfan/wams`, 171 GB). `/data` now has 263 GB free, so the guard has about 165 GB more slack than projected.

### Arm A crash at step 79 (2026-11-21 19:22)
- **What happened:** the resumed arm A died with a CUDA out-of-memory error in the actor update (backward pass) of step 79: "Tried to allocate 1.82 GiB; GPU 0 total 23.67 GiB, 1.28 GiB free".
  - The last complete step in `train.log` is 78. `reward_steps.jsonl` has rollouts up to step 79.
  - Peak allocated memory was flat at 35.2 GB over steps 70–78; maximum response length 1,809–2,010 tokens.
  - Settings: dynamic batch size on, `ppo_max_token_len_per_gpu` 32,768, the same as the pilot and L006.
- **State:**
  - Checkpoints 20, 40 and 60 are merged and confirmed to load (`hf/step_{20,40,60}`).
  - `ckpt/global_step_60` is complete, with shards, optimizer state and `data.pt`.
  - The shards of global_step_20 and 40 were deleted after their merges loaded.
  - No step-80 or step-100 checkpoint exists.
- **Chain:** `E004_chain2.sh` found no confirmed `hf/step_100` and stopped as designed (19:23). Nothing was deleted and arm B was not started. GPUs 4–7 are idle.
- Waiting for Erfan's decision.
- **Decision (Erfan, 2026-11-21):** option 2. Resume arm A from step 60 with `ppo_max_token_len_per_gpu=24576` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, with the same settings for arm B. Keep steps 61–79 as `*.part2.*` and trim `reward_steps.jsonl` to step 60. On a second CUDA OOM in either arm, resume once from its latest full checkpoint with 16384. On another crash, stop that arm, evaluate the checkpoints that exist and continue the chain.
- **Done:**
  - Steps 61–79 are archived in `E004-A-s0/` as `train.part2.log`, `config.part2.txt`, `git.part2.diff`, `hydra.part2/`, `rollouts.part2/`, `ckpt_watch.part2.log`, `reward_steps.part2.jsonl` (all 79 steps), and `{base_commit,code,git_status}.part2.*` (provenance of the step-20 resume).
  - `reward_steps.jsonl` is trimmed to steps 1–60.
  - New scripts:
    - `E004_train.sh` gets a `resume <arm> <step> <tag>` mode and records `MAX_TOKENS_PER_GPU` in the provenance files;
    - `E004_eval_all.sh` queues only checkpoints with `LOAD_OK.json`;
    - `scripts/exp/E004_chain3.sh` implements the decision with T1/T2 as parameters.
- **Not launched: the setting as written contradicts its purpose.** The actor's `ppo_max_token_len_per_gpu` was 6144 (the `MAX_TOKENS_PER_GPU` default in `train_grpo_3b.sh`; `train.part2.log` config dump). The 32768 reported earlier is the critic's value, which GRPO does not use; that report was wrong. 24576 and 16384 would raise the actor's per-GPU micro-batch above the setting that crashed. Waiting for Erfan to confirm the values.
- **Values confirmed (Erfan, 2026-11-21):** T1 = 4608 and T2 = 3072, for the actor `ppo_max_token_len_per_gpu`. The log-prob passes use 4×, so 18432 and 12288.
- **Launched 22:17:** `T1=4608 T2=3072 bash scripts/exp/E004_chain3.sh` (log `$DATA_ROOT/logs/E004/chain3.log`).
  - Arm A resumed from `ckpt/global_step_60`. Provenance: `E004-A-s0/{base_commit,code,git_status}.r3.*`.
  - Checked in the running job: `train.log` shows `ppo_max_token_len_per_gpu: 4608` and `log_prob_max_token_len_per_gpu: 18432`; `config.txt` shows resume_from global_step_60; the Ray WorkerDict process environment has `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Arm B (fresh) will use the same settings.
- **Chain3 rules:**
  - On a CUDA OOM, an arm resumes once from its latest full checkpoint with 3072. Logs are archived as `*.part<k>.*`, `reward_steps.jsonl` is trimmed to that checkpoint, and provenance goes in `*.r<k>.*`.
  - A second OOM stops that arm; its confirmed checkpoints are evaluated and the chain continues.
  - A failure that is not an OOM stops the chain.
  - If arm B crashes before its first checkpoint, there is nothing to resume from and the arm stops.
  - If arm A stops early, its latest full checkpoint is kept.
  - Disk guard: NEED = 46 GB (36 GB of checkpoint shards + 7 + 3).
- **Attempt r3 hung (22:19–23:31):** vLLM's sleep-mode memory pool (`CuMemAllocator`) refuses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It failed at rollout-engine load with "AssertionError: Expandable segments are not compatible with memory pool". verl did not exit, so the run sat idle for 74 min with no step.
  - At 23:31 I killed the run, its watcher and Ray, plus chain3's bash process.
  - The attempt's files are kept as `*.part3.*`: `train.part3.log`, `config.part3.txt`, `git.part3.diff`, `hydra.part3/`, `rollouts.part3/`, `ckpt_watch.part3.log`, `{base_commit,code,git_status}.part3.*`.
  - `reward_steps.jsonl` was unchanged (steps 1–60). The first `chain3.log` (2 lines, 22:17) was overwritten by the relaunch.
- **Deviation from the decision: `expandable_segments` dropped,** because it cannot be used with vLLM sleep mode in this setup. T1 = 4608 and T2 = 3072 are kept.
  - `E004_chain3.sh` now takes `PYTORCH_CUDA_ALLOC_CONF` from the environment (unset) and the arm A tag from `A_TAG`.
  - It also adds a stall watchdog: if `train.log` is silent for 30 min, training is killed and the chain treats it as a non-OOM failure, so the chain stops.
- **Relaunched 23:33:** `T1=4608 T2=3072 A_TAG=r4 bash scripts/exp/E004_chain3.sh`. Provenance: `E004-A-s0/{base_commit,code,git_status}.r4.*`. `train.log` shows `ppo_max_token_len_per_gpu: 4608`. Steps 61 and 62 completed by 23:42.

### Full-run outcome (2026-11-22)
- **Arm A (`E004-A-s0`), 100 steps in segments:**

  | segment | time | steps | notes |
  |---|---|---|---|
  | part 1 | 2026-11-21 15:22–16:20 | 1–20 | |
  | part 2 | 16:22–19:22 | 21–79 | CUDA OOM at step 79 |
  | part 3 | 22:17–23:31 | none | hung on `expandable_segments` |
  | r4 | 23:33 to 2026-11-22 01:55 | 61–100 | actor tokens 4608; mean step 209.7 s (n = 40) |

  - No OOM in r4, so the 3072 retry was never used.
- **Arm B (`E004-B-s0`, fresh):** 2026-11-22 01:57–07:23, 100 steps, actor tokens 4608, mean step 194.5 s (n = 100). No OOM.
  - Disk guard: free space after its checkpoints was 231, 224, 218 and 211 GB at steps 20, 40, 60 and 80 (NEED 46).
- **Evaluation:** `E004_eval_all.sh` ran 07:26–08:52. All 10 jobs (steps 20–100 × 2 arms) exited 0 for both the eval and hidden stages. Log: `$DATA_ROOT/logs/E004/chain3.log`, also `eval/launch.log`.
- **Merged weights:** `$DATA_ROOT/logs/E004/E004-{A,B}-s0/hf/step_{20,40,60,80,100}`. All 10 exist with `LOAD_OK.json` (3,085,938,688 params, bf16, finite logits), confirmed 2026-11-22 08:55; 32 GB per arm.
- **Full checkpoints and deletions:**
  - Remaining full checkpoint: `E004-B-s0/ckpt/global_step_100` (14 files, 36 GB).
  - Deleted by the watcher after each later merge was confirmed: arm A global_step_{20,40,60,80} (part-2 and r4 watcher logs) and arm B global_step_{20,40,60,80}; 12 files and 38.3 GB each.
  - Deleted by chain3 at 01:56 after `hf/step_100` loaded: arm A global_step_100 (12 files, 36 GB; approved).
  - The other `global_step_*` directories keep only `data.pt` and `actor/huggingface` (16 MB each).
- **Provenance:**
  - Arm A: `base_commit.txt`, `code.patch`, `git_status.txt` (steps 1–20); `*.part2.*` (steps 21–79); `*.part3.*` (hung attempt); `*.r4.*` (steps 61–100).
  - Arm B: `base_commit.txt`, `code.patch`, `git_status.txt`.
  - Base commit 359ea18 plus the uncommitted worktree `../sar-alias-reward`. Main-repo experiment scripts are untracked.
- **Outputs:** `$DATA_ROOT/logs/E004/{e004_tables.md, e004_tidy.csv, e004_plot.png}` from `scripts/exp/E004_tables.py`; eval samples in `$DATA_ROOT/logs/E004/eval/<run>/step<k>/<prompt>/samples.jsonl`. `/data` had 204 GB free at the end.
- **Recipe deviations, recorded above:** arm A was resumed twice (after step 20, for the answer-span logging; after step 60, for the OOM). Actor `ppo_max_token_len_per_gpu` was 6144 for arm A steps 1–60 and 4608 for arm A steps 61–100 and all of arm B.

### Relabel fix (2026-11-22, requested by Erfan after E005)
- **Defect:** the step-100 relabel kept the last streamed output of a single n=5 request, so questions had 1–5 samples (arm A 682 / 105 / 57 / 45 / 111 questions with 1 / 2 / 3 / 4 / 5 samples; arm B 741 / 80 / 46 / 41 / 92). All "final labels" numbers in the first version of Result are invalid.
- **Fix (2026-11-22, `scripts/exp/E004_eval.py`, untracked main-repo experiment script):** the relabel now sends 5 separate n=1 requests per question (same sampling: E001 prompt, T 0.7, top_p 1.0, max 512, stop `</answer>`; per-sample seed = md5("<id>|relabel|<k>"), replacing the single request seed 0) and asserts exactly 5 samples per question before writing. New flags `--only-relabel` (skip P1/P2/P6 and S5) and `--model-path`. Test `scripts/exp/test_E004_relabel.py` (3 passed: 5 n=1 requests with distinct seeds per question; the old single n=5 call fails the count; the output check flags a dir with one question at 1 sample). Output check `scripts/exp/E004_check_relabel.py` (every question exactly 5 samples, labels cover the same ids, n_correct = sum of correct).
- **Rerun (2026-11-22 21:41–21:46, GPUs 4–7):** E004-A step 100, E004-B step 100, E005-O step 20, E005-S step 20, and the start model (Qwen2.5-3B-Instruct snapshot aa8e725…, the E001 model path) → `$DATA_ROOT/logs/E005/eval/start-model/step0/relabel/`. Logs `$DATA_ROOT/logs/E005/relabel_fix/`. All exit 0; check passes for all five (1,000 questions × 5 samples). Old outputs moved (not deleted) to `.../relabel_invalid_streamcut/`.
- **Start-model check against E001** (887 non-ambiguous questions): label counts knowable / unknowable / uncertain E001 278 / 466 / 143, rerun 278 / 466 / 143; per-question label agreement 0.825 (E001 knowable → rerun 239 / 6 / 33; unknowable → 1 / 424 / 41; uncertain → 38 / 36 / 69); mean correct share 0.323 vs 0.320; Spearman of n_correct 0.900. Sample-level identity is not expected (E001 used vLLM's offline LLM with request seed 1).
- **Tables rebuilt** with `scripts/exp/E004_tables.py` (old: `e004_tables_before_relabel_fix.md`, `e004_tidy_before_relabel_fix.csv`). Only the "final labels" rows changed. Corrected values (step 100, final labels):

| arm | kn / unk | P2 P(s∣kn) | P2 P(s∣unk) | P2 AUROC [CI] | acc kn % | acc unk % | P1 AUROC [CI] | P6 AUROC [CI] (P(s∣kn) / P(s∣unk)) | S5 margin AUROC (n) | probe L24 AUROC (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 290 / 464 | 0.969 | 0.985 | 0.521 [0.498, 0.543] | 79.9 ± 1.7 | 48.0 ± 2.0 | 0.511 [0.493, 0.529] | 0.619 [0.579, 0.658] (0.690 / 0.855) | 0.516 [0.479, 0.556] (753) | 0.833 [0.804, 0.861] (754) |
| B | 318 / 455 | 1.000 | 0.999 | 0.499 [0.497, 0.500] | 95.3 ± 1.0 | 66.5 ± 2.0 | 0.507 [0.498, 0.517] | 0.561 [0.538, 0.585] (0.913 / 0.975) | 0.388 [0.346, 0.431] (772) | 0.837 [0.807, 0.864] (773) |

## Result      ← you fill (facts only)

> **INVALID (2026-11-22): every "final labels" row and number below.** The step-100 relabel (`E004_eval.py --relabel`) kept only the last streamed output of one n=5 request, so most questions had 1–4 of the 5 closed-book samples (arm A: 682 questions with 1 sample, 105 with 2, 57 with 3, 45 with 4, 111 with 5; arm B: 741 / 80 / 46 / 41 / 92). The labels, and the "final labels" rows of Tables A–C and the relabel counts (A 100 / 545, B 78 / 538), are not from 5 samples per question. Old outputs moved to `$DATA_ROOT/logs/E004/eval/E004-{A,B}-s0/step100/relabel_invalid_streamcut/`; old tables kept as `e004_tables_before_relabel_fix.md` / `e004_tidy_before_relabel_fix.csv`. Corrected values: Execution, "Relabel fix" (arm A 290 / 464, arm B 318 / 455).
Tables are copied from `$DATA_ROOT/logs/E004/e004_tables.md`. Questions: E001 set, ambiguous excluded. Labels: qwen3b-instruct's E001 labels, n = 278 knowable / 466 unknowable, unless marked "final" (relabel on the arm's step-100 model). Step 0 is the start model (E002/E003 outputs). AUROC: unknowable = positive class, 95% stratified bootstrap CI. Accuracy is mean ± SE over questions. Plot: `$DATA_ROOT/logs/E004/e004_plot.png`. Tidy CSV: `$DATA_ROOT/logs/E004/e004_tidy.csv`.

Questions: E001 set, ambiguous excluded. Labels: qwen3b-instruct E001 labels unless marked final (relabel of the arm's step-100 model). Step 0 = start model (E002 / E003 outputs). AUROC: unknowable = positive, 95% bootstrap CI.

#### Table A (E004-A-s0, P2)

| step | n know / unk | P(search ∣ knowable) | P(search ∣ unknowable) | AUROC [CI] | acc knowable % | acc unknowable % | acc overall % | mean searches | parse | cap-hit |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 278 / 466 | 0.489 | 0.609 | 0.587 [0.546, 0.626] | 79.1 ± 1.9 | 33.7 ± 1.7 | 52.1 ± 1.4 | 0.57 | 0.998 | 0.000 |
| 20 | 278 / 466 | 0.839 | 0.914 | 0.582 [0.547, 0.618] | 69.1 ± 1.9 | 39.1 ± 1.8 | 51.3 ± 1.3 | 1.03 | 0.951 | 0.000 |
| 40 | 278 / 466 | 0.976 | 0.995 | 0.513 [0.501, 0.526] | 83.6 ± 1.9 | 52.7 ± 2.1 | 66.1 ± 1.4 | 1.09 | 0.999 | 0.000 |
| 60 | 278 / 466 | 0.983 | 0.999 | 0.517 [0.506, 0.529] | 85.3 ± 1.8 | 51.8 ± 2.1 | 66.6 ± 1.4 | 1.14 | 0.999 | 0.000 |
| 80 | 278 / 466 | 0.995 | 1.000 | 0.506 [0.500, 0.514] | 84.7 ± 1.8 | 51.3 ± 2.1 | 66.1 ± 1.4 | 1.27 | 0.987 | 0.000 |
| 100 | 278 / 466 | 0.962 | 0.985 | 0.533 [0.510, 0.555] | 79.2 ± 1.8 | 48.0 ± 2.0 | 61.5 ± 1.4 | 1.25 | 0.946 | 0.000 |
| 100 (final labels, INVALID) | 100 / 545 | 0.960 | 0.980 | 0.549 [0.511, 0.590] | 82.2 ± 2.8 | 49.8 ± 1.9 | 61.5 ± 1.4 | 1.25 | 0.946 | 0.000 |

#### Table A (E004-B-s0, P2)

| step | n know / unk | P(search ∣ knowable) | P(search ∣ unknowable) | AUROC [CI] | acc knowable % | acc unknowable % | acc overall % | mean searches | parse | cap-hit |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 278 / 466 | 0.489 | 0.609 | 0.587 [0.546, 0.626] | 79.1 ± 1.9 | 33.7 ± 1.7 | 52.1 ± 1.4 | 0.57 | 0.998 | 0.000 |
| 20 | 278 / 466 | 0.818 | 0.895 | 0.594 [0.554, 0.631] | 67.4 ± 2.0 | 39.4 ± 1.9 | 51.4 ± 1.4 | 0.98 | 0.958 | 0.000 |
| 40 | 278 / 466 | 0.976 | 0.988 | 0.510 [0.496, 0.524] | 87.7 ± 1.7 | 54.5 ± 2.2 | 68.7 ± 1.4 | 1.06 | 0.991 | 0.000 |
| 60 | 278 / 466 | 0.996 | 0.998 | 0.505 [0.497, 0.514] | 90.5 ± 1.6 | 58.1 ± 2.2 | 72.0 ± 1.4 | 1.02 | 0.999 | 0.000 |
| 80 | 278 / 466 | 0.984 | 0.984 | 0.500 [0.485, 0.519] | 89.6 ± 1.5 | 57.5 ± 2.1 | 71.3 ± 1.4 | 1.05 | 0.997 | 0.000 |
| 100 | 278 / 466 | 1.000 | 0.998 | 0.498 [0.495, 0.500] | 94.4 ± 1.2 | 65.4 ± 2.0 | 78.1 ± 1.3 | 1.35 | 1.000 | 0.000 |
| 100 (final labels, INVALID) | 78 / 538 | 0.992 | 0.999 | 0.505 [0.497, 0.519] | 94.1 ± 2.2 | 68.3 ± 1.9 | 78.1 ± 1.3 | 1.35 | 1.000 | 0.000 |

#### Table B (E004-A-s0, prompt shift)

| step | prompt | P(search ∣ knowable) | P(search ∣ unknowable) | AUROC [CI] |
|---|---|---|---|---|
| 0 | P1 | 0.673 | 0.837 | 0.635 [0.596, 0.670] |
| 0 | P2 | 0.489 | 0.609 | 0.587 [0.546, 0.626] |
| 0 | P6 | 0.245 | 0.376 | 0.622 [0.580, 0.662] |
| 100 | P1 | 0.968 | 0.991 | 0.521 [0.503, 0.540] |
| 100 | P2 | 0.962 | 0.985 | 0.533 [0.510, 0.555] |
| 100 | P6 | 0.696 | 0.862 | 0.620 [0.578, 0.654] |
| 100 (final labels, INVALID) | P1 | 0.966 | 0.985 | 0.521 [0.493, 0.548] |
| 100 (final labels, INVALID) | P2 | 0.960 | 0.980 | 0.549 [0.511, 0.590] |
| 100 (final labels, INVALID) | P6 | 0.696 | 0.831 | 0.605 [0.548, 0.667] |

#### Table B (E004-B-s0, prompt shift)

| step | prompt | P(search ∣ knowable) | P(search ∣ unknowable) | AUROC [CI] |
|---|---|---|---|---|
| 0 | P1 | 0.673 | 0.837 | 0.635 [0.596, 0.670] |
| 0 | P2 | 0.489 | 0.609 | 0.587 [0.546, 0.626] |
| 0 | P6 | 0.245 | 0.376 | 0.622 [0.580, 0.662] |
| 100 | P1 | 0.988 | 0.997 | 0.511 [0.500, 0.523] |
| 100 | P2 | 1.000 | 0.998 | 0.498 [0.495, 0.500] |
| 100 | P6 | 0.898 | 0.975 | 0.576 [0.550, 0.602] |
| 100 (final labels, INVALID) | P1 | 0.995 | 0.995 | 0.498 [0.487, 0.512] |
| 100 (final labels, INVALID) | P2 | 0.992 | 0.999 | 0.505 [0.497, 0.519] |
| 100 (final labels, INVALID) | P6 | 0.923 | 0.964 | 0.543 [0.500, 0.588] |

#### Table C (E004-A-s0, P2)

| step | S5 margin AUROC [CI] (n) | S4 probe layer 24 AUROC [CI] (n) |
|---|---|---|
| 0 | 0.604 [0.560, 0.644] (744) | 0.850 [0.821, 0.876] (744) |
| 20 | 0.623 [0.582, 0.664] (744) | 0.854 [0.825, 0.880] (744) |
| 40 | 0.522 [0.480, 0.564] (744) | 0.855 [0.826, 0.879] (744) |
| 60 | 0.579 [0.537, 0.620] (744) | 0.858 [0.829, 0.883] (744) |
| 80 | 0.554 [0.513, 0.597] (744) | 0.859 [0.830, 0.884] (744) |
| 100 | 0.522 [0.480, 0.563] (743) | 0.853 [0.825, 0.879] (744) |
| 100 (final labels, INVALID) | 0.526 [0.467, 0.592] (645) | 0.818 [0.773, 0.858] (645) |

#### Table C (E004-B-s0, P2)

| step | S5 margin AUROC [CI] (n) | S4 probe layer 24 AUROC [CI] (n) |
|---|---|---|
| 0 | 0.604 [0.560, 0.644] (744) | 0.850 [0.821, 0.876] (744) |
| 20 | 0.549 [0.507, 0.589] (740) | 0.856 [0.826, 0.882] (744) |
| 40 | 0.489 [0.446, 0.530] (743) | 0.856 [0.827, 0.882] (744) |
| 60 | 0.515 [0.473, 0.554] (744) | 0.850 [0.822, 0.876] (744) |
| 80 | 0.442 [0.401, 0.492] (743) | 0.853 [0.826, 0.880] (744) |
| 100 | 0.380 [0.339, 0.426] (743) | 0.861 [0.834, 0.886] (744) |
| 100 (final labels, INVALID) | 0.393 [0.334, 0.460] (615) | 0.801 [0.740, 0.858] (616) |

#### Table D (E004-A-s0, training rollouts, by training label)

| step | label | rollouts | P(≥1 search) | mean reward | reward at 0 searches | reward at ≥1 search | groups | groups mixed in search | mixed in search and reward | answers | answer words mean | answer words p95 | unclosed share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1–3 (pilot dumps) | knowable | — | — | — | — | — | — | — | — | 421 | 6.43 | 21.0 | 0.000 |
| 1–3 (pilot dumps) | unknowable | — | — | — | — | — | — | — | — | 502 | 15.60 | 35.9 | 0.004 |
| 1 | knowable | 125 | 0.528 | 0.632 | 0.763 | 0.515 | 25 | 17 | 15 | — | — | — | — |
| 1 | unknowable | 195 | 0.564 | 0.144 | 0.035 | 0.227 | 39 | 30 | 13 | — | — | — | — |
| 10 | knowable | 135 | 0.622 | 0.763 | 0.765 | 0.762 | 27 | 16 | 9 | — | — | — | — |
| 10 | unknowable | 185 | 0.692 | 0.297 | 0.123 | 0.375 | 37 | 21 | 9 | — | — | — | — |
| 20 | knowable | 185 | 0.919 | 0.800 | 1.000 | 0.782 | 37 | 7 | 2 | — | — | — | — |
| 20 | unknowable | 135 | 0.985 | 0.400 | 0.000 | 0.406 | 27 | 2 | 1 | — | — | — | — |
| 21 | knowable | 160 | 0.856 | 0.838 | 0.870 | 0.832 | 32 | 11 | 5 | 159 | 3.18 | 6.3 | 0.000 |
| 21 | unknowable | 160 | 0.944 | 0.406 | 0.222 | 0.417 | 32 | 8 | 3 | 160 | 5.87 | 31.2 | 0.000 |
| 30 | knowable | 145 | 0.903 | 0.841 | 0.643 | 0.863 | 29 | 2 | 0 | 145 | 1.49 | 3.0 | 0.000 |
| 30 | unknowable | 175 | 0.983 | 0.440 | 0.000 | 0.448 | 35 | 2 | 0 | 175 | 4.43 | 13.6 | 0.000 |
| 40 | knowable | 190 | 0.953 | 0.826 | 0.889 | 0.823 | 38 | 3 | 1 | 190 | 2.32 | 3.0 | 0.000 |
| 40 | unknowable | 130 | 0.992 | 0.538 | 0.000 | 0.543 | 26 | 1 | 1 | 130 | 3.04 | 6.0 | 0.000 |
| 50 | knowable | 155 | 1.000 | 0.781 | — | 0.781 | 31 | 0 | 0 | 155 | 3.57 | 4.3 | 0.006 |
| 50 | unknowable | 165 | 1.000 | 0.515 | — | 0.515 | 33 | 0 | 0 | 165 | 3.62 | 5.0 | 0.000 |
| 60 | knowable | 205 | 0.990 | 0.844 | 1.000 | 0.842 | 41 | 2 | 0 | 205 | 1.86 | 4.0 | 0.000 |
| 60 | unknowable | 115 | 0.991 | 0.530 | 1.000 | 0.526 | 23 | 1 | 0 | 114 | 2.58 | 5.7 | 0.000 |
| 70 | knowable | 185 | 1.000 | 0.897 | — | 0.897 | 37 | 0 | 0 | 185 | 3.00 | 5.0 | 0.000 |
| 70 | unknowable | 135 | 1.000 | 0.481 | — | 0.481 | 27 | 0 | 0 | 135 | 5.06 | 23.0 | 0.000 |
| 80 | knowable | 145 | 1.000 | 0.966 | — | 0.966 | 29 | 0 | 0 | 145 | 2.03 | 5.6 | 0.000 |
| 80 | unknowable | 175 | 0.994 | 0.417 | 0.000 | 0.420 | 35 | 1 | 1 | 175 | 5.70 | 23.0 | 0.000 |
| 90 | knowable | 190 | 0.979 | 0.826 | 0.500 | 0.833 | 38 | 3 | 2 | 189 | 4.92 | 23.6 | 0.000 |
| 90 | unknowable | 130 | 0.962 | 0.531 | 0.000 | 0.552 | 26 | 4 | 3 | 130 | 5.37 | 22.6 | 0.000 |
| 100 | knowable | 155 | 1.000 | 0.845 | — | 0.845 | 31 | 0 | 0 | 155 | 6.25 | 25.6 | 0.006 |
| 100 | unknowable | 165 | 0.994 | 0.509 | 0.000 | 0.512 | 33 | 1 | 0 | 163 | 8.88 | 31.9 | 0.012 |

#### Table D (E004-B-s0, training rollouts, by training label)

| step | label | rollouts | P(≥1 search) | mean reward | reward at 0 searches | reward at ≥1 search | groups | groups mixed in search | mixed in search and reward | answers | answer words mean | answer words p95 | unclosed share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | knowable | 55 | 0.509 | 0.800 | 0.704 | 0.893 | 11 | 5 | 1 | 55 | 2.35 | 3.3 | 0.000 |
| 1 | unknowable | 265 | 0.536 | 0.268 | 0.146 | 0.373 | 53 | 34 | 14 | 263 | 6.05 | 23.0 | 0.000 |
| 10 | knowable | 50 | 0.420 | 0.860 | 0.828 | 0.905 | 10 | 4 | 2 | 50 | 2.42 | 3.0 | 0.000 |
| 10 | unknowable | 270 | 0.641 | 0.393 | 0.144 | 0.532 | 54 | 30 | 14 | 270 | 3.01 | 11.5 | 0.000 |
| 20 | knowable | 50 | 0.940 | 0.900 | 1.000 | 0.894 | 10 | 3 | 0 | 50 | 9.38 | 50.7 | 0.000 |
| 20 | unknowable | 270 | 0.952 | 0.437 | 0.154 | 0.451 | 54 | 8 | 3 | 269 | 7.06 | 42.0 | 0.000 |
| 30 | knowable | 65 | 0.985 | 0.877 | 1.000 | 0.875 | 13 | 1 | 0 | 65 | 11.11 | 41.4 | 0.000 |
| 30 | unknowable | 255 | 0.984 | 0.537 | 0.000 | 0.546 | 51 | 3 | 1 | 254 | 10.08 | 42.4 | 0.004 |
| 40 | knowable | 80 | 0.863 | 0.912 | 0.909 | 0.913 | 16 | 5 | 1 | 80 | 12.06 | 32.0 | 0.000 |
| 40 | unknowable | 240 | 0.979 | 0.575 | 0.000 | 0.587 | 48 | 5 | 4 | 239 | 14.32 | 54.2 | 0.000 |
| 50 | knowable | 75 | 1.000 | 0.827 | — | 0.827 | 15 | 0 | 0 | 75 | 11.01 | 27.2 | 0.000 |
| 50 | unknowable | 245 | 1.000 | 0.535 | — | 0.535 | 49 | 0 | 0 | 245 | 15.41 | 49.6 | 0.004 |
| 60 | knowable | 80 | 1.000 | 0.762 | — | 0.762 | 16 | 0 | 0 | 80 | 18.41 | 56.3 | 0.000 |
| 60 | unknowable | 240 | 1.000 | 0.575 | — | 0.575 | 48 | 0 | 0 | 240 | 22.26 | 59.1 | 0.000 |
| 70 | knowable | 70 | 1.000 | 0.914 | — | 0.914 | 14 | 0 | 0 | 70 | 32.39 | 87.8 | 0.000 |
| 70 | unknowable | 250 | 1.000 | 0.396 | — | 0.396 | 50 | 0 | 0 | 250 | 33.64 | 88.2 | 0.000 |
| 80 | knowable | 65 | 0.985 | 0.923 | 1.000 | 0.922 | 13 | 1 | 0 | 65 | 33.88 | 93.2 | 0.000 |
| 80 | unknowable | 255 | 1.000 | 0.557 | — | 0.557 | 51 | 0 | 0 | 255 | 44.64 | 111.3 | 0.000 |
| 90 | knowable | 55 | 1.000 | 1.000 | — | 1.000 | 11 | 0 | 0 | 55 | 25.27 | 61.5 | 0.000 |
| 90 | unknowable | 265 | 1.000 | 0.600 | — | 0.600 | 53 | 0 | 0 | 265 | 69.07 | 232.2 | 0.030 |
| 100 | knowable | 45 | 1.000 | 0.889 | — | 0.889 | 9 | 0 | 0 | 45 | 28.29 | 97.8 | 0.000 |
| 100 | unknowable | 275 | 0.989 | 0.622 | 0.000 | 0.629 | 55 | 1 | 0 | 275 | 92.28 | 266.6 | 0.058 |


**Summary of the numbers:**
- **P2 search rates:** in both arms, P(≥1 search) rose from 0.489 (knowable) / 0.609 (unknowable) at step 0 to ≥ 0.96 on both labels from step 40 on. At step 100: arm A 0.962 / 0.985, arm B 1.000 / 0.998.
- **P2 search-rate AUROC:**
  - arm A: 0.587 [0.546, 0.626] at step 0; 0.582 at step 20; 0.506–0.517 at steps 40–80; 0.533 [0.510, 0.555] at step 100 (0.549 [0.511, 0.590] with final labels — INVALID, relabel defect);
  - arm B: 0.594 at step 20; 0.498 [0.495, 0.500] at step 100 (0.505 with final labels — INVALID, relabel defect).
- **P2 accuracy, overall:** step 0 52.1 ± 1.4%; step 100 arm A 61.5 ± 1.4%, arm B 78.1 ± 1.3%.
  - knowable: 79.1 → 79.2 (A) and 94.4 (B);
  - unknowable: 33.7 → 48.0 (A) and 65.4 (B).
- **Prompt shift, step 100:**
  - P6 AUROC: arm A 0.620 [0.578, 0.654], P(search) 0.696 / 0.862; arm B 0.576 [0.550, 0.602], P(search) 0.898 / 0.975.
  - P1 AUROC: 0.521 (A), 0.511 (B).
  - Step 0 values: P1 0.635, P6 0.622.
- **Internal signals (Table C):**
  - S4 layer-24 probe AUROC stayed within 0.850–0.861 at every checkpoint of both arms (0.818 / 0.801 with final labels — INVALID, relabel defect, n = 645 / 616).
  - S5 margin AUROC: arm A 0.604 → 0.522 at step 100; arm B 0.604 → 0.380 [0.339, 0.426] at step 100.
- **Relabel at step 100 (E001 procedure on the final model), INVALID (relabel defect):** arm A 100 knowable / 545 unknowable; arm B 78 / 538 (start model: 278 / 466).
- **Training rollouts (Table D):**
  - Groups mixed in search fell from 17/25 (knowable) and 30/39 (unknowable) at arm A step 1 to 0–4 per label from step 30 on. Arm B is similar.
  - Answer span, arm A: mean words 1.5–8.9 at steps 21–100 (pilot steps 1–3: 6.43 / 15.60), unclosed share ≤ 0.012.
  - Answer span, arm B: mean words rose from 2.35 (knowable) / 6.05 (unknowable) at step 1 to 28.29 / 92.28 at step 100, with p95 97.8 / 266.6; unknowable unclosed share 0.030 at step 90 and 0.058 at step 100.
  - The stop rule was specified for arm A at step 20 only and was not applied to arm B.
