# E003: Does the model represent what it knows, even though its search decision doesn't use it?
Question: For each model, how well do signals that use no gold answers (self-consistency, answer log-probability, a linear probe on hidden states) separate knowable from unknowable questions, compared with the search decision measured in E002?
Parent: E002

## Setup
- Models: the same 5 as E001/E002, same checkpoints and tags.
- Data: data/e001_easy_1k.jsonl. Each model's own E001 labels: knowable vs unknowable, ambiguous excluded, uncertain excluded.
- Signals (all computed without gold answers):
  S1. Self-consistency, in-sample: from the 5 E001 closed-book samples, the share agreeing with the most common normalized answer (normalize_answer). No new runs. Partly circular, because the labels come from the same samples.
  S2. Self-consistency, held out: 5 new closed-book samples with the exact E001 prompt and decoding (T=0.7), but seed 1. Same agreement measure. This is the main version of S1.
  S3. Answer log-probability: one greedy closed-book generation with the E001 prompt. Mean token log-prob of the text inside <answer>…</answer>.
  S4. Linear probe: run the P2 prompt from E002 (exact string from scripts/exp/E002_prompts.py) through the model up to and including "Question: <q>\n", with no generation. Take the last-token hidden state at every 4th layer, plus the final layer. Fit an L2 logistic regression predicting knowable vs unknowable, with 5-fold cross-validation split by question (seed 0) and the regularization strength chosen by inner CV. Report the held-out AUROC per layer.
  S5. Behavioral margin (reference): in one greedy P2 rollout, at the position right before the model emits its first <search> or <answer> tag, compute log p("<search") − log p("<answer") using the first token of each tag. This is a continuous version of the E002 decision. Report its AUROC, with higher margin meaning unknowable.
- Reference from E002: the P2 per-question search rate AUROC for each model (copy it, don't recompute).

## Source confound control
Source (PopQA vs TriviaQA) and popularity bin correlate with the labels, so a signal could just detect the dataset. For every signal, report AUROC on all questions AND within each source separately (PopQA only, TriviaQA only). For S4, also fit a baseline probe on source + popularity bin alone, and report its AUROC.

## Pilot first
S2, S3 and S4 on 100 questions for qwen3b-base and qwen7b-instruct. Check that S3 finds the <answer> span in at least 95% of generations, and that S4's hidden-state extraction gives the expected shapes. Report and stop if anything breaks.

## Output wanted
1. Table A: rows = models; columns = AUROC for S1, S2, S3, S4 (best layer, with its index), S5, and the E002 P2 search decision, each with a bootstrap 95% CI (1,000 resamples, seed 0). Three versions: all questions, PopQA only, TriviaQA only.
2. Table B: S4 held-out AUROC by layer, rows = layers, columns = models.
3. Table C: S4 source-only baseline AUROC per model.
4. Files: $DATA_ROOT/logs/E003/<model tag>/ with per-question signals (one CSV: id, label, source, S1–S5, search rate from E002 P2), the probe weights for the best layer, and the new samples.

## Execution   ← you fill
### Pre-run checks (2026-11-21)
- **Data:** `$DATA_ROOT/eval/e001/e001_easy_1k.jsonl` (card path `data/e001_easy_1k.jsonl`); labels and ambiguous flag in `$DATA_ROOT/eval/e001/e001_easy_1k_labels.jsonl`. PopQA rows are in popularity bins 4 and 5 (250 each); TriviaQA rows have no bin.
- **Inputs:** E001 samples `$DATA_ROOT/logs/E001/E001-<model>-s0/samples.jsonl` (S1); E002 P2 samples `$DATA_ROOT/logs/E002/<model>/P2/samples.jsonl` (reference). All 5 checkpoints exist (paths in the E002 card).
- **Tokenization:** in both tokenizers `<search>` → `<`, `search`, `>` and `<answer>` → `<`, `answer`, `>`, so the first token of the two tags is identical.
- **Layers:** Qwen2.5-3B models 36 layers (hidden 2048); Qwen2.5-7B-Instruct 28 layers (hidden 3584).
- **E001 closed-book parse rate:** 3B models 0.997–1.000, qwen7b-instruct 0.941.
- **Decisions (Erfan, 2026-11-21):**
  1. S5 = log p(`search`) − log p(`answer`) at the token right after the generated `<` that opens the first tag (equal to log p("<search") − log p("<answer") over the full strings).
  2. scikit-learn installed into `.venv-verl`.
  3. S3 95% span check applies to the 3B models only. For qwen7b-instruct, report S3 coverage; questions with no answer span are missing for S3.
  4. S4 input = the full P2 prompt including the chat template's generation prompt (instruct models: last token is the one right before the reply; base models: the `\n` after the question).
  5. S4 layers: 4, 8, …, 36 (3B) and 4, 8, …, 28 (7B); the final layer (after the final norm) is in both lists.
  6. S4 fit: StandardScaler + L2 logistic regression, C ∈ logspace(−4, 2, 7) chosen by inner 5-fold CV (AUROC); outer stratified 5-fold, seed 0. Per-source AUROCs use the out-of-fold predictions of the all-question probe. Best layer chosen on the all-question AUROC (its CI is optimistic). Also reported: a layer fixed in advance at ~2/3 depth, layer 24 (3B) and layer 20 (7B).
  7. S1/S2: each untagged sample counts as its own answer. S3 missing when there is no answer span; count reported.
  8. AUROC: unknowable = positive class for all signals (S1–S3 negated; S4 = P(unknowable); S5 margin as is).
  9. Source-only baseline features: one-hot {PopQA bin 4, PopQA bin 5, TriviaQA}.
  10. E002 reference: all-question AUROC copied from the E002 card; per-source AUROCs and CIs computed from the existing E002 P2 samples with the same code (no new runs).
  11. Pilot: 25 PopQA from bin 4 and 25 from bin 5 (random, seed 0) plus the first 50 TriviaQA = 100 questions.
### Pilot (2026-11-21)
- 100 questions (25 PopQA bin 4 + 25 PopQA bin 5, `random.Random(0).sample`; first 50 TriviaQA) × qwen3b-base (GPU 4), qwen7b-instruct (GPU 5).
- Scripts: `scripts/exp/E003_gen.py` (vLLM: S2 samples, S3 greedy with token log-probs, S5 greedy P2 first turn), `scripts/exp/E003_hidden.py` (transformers forward passes: S4 hidden states, S5 margin), `scripts/exp/E003_tables.py` (signals, probes, tables). Commit 359ea18 + 27 dirty/untracked paths. Generation 107–126 s, forward pass 18–34 s per model.
- Outputs: `$DATA_ROOT/logs/E003/pilot/<model>/`, logs `$DATA_ROOT/logs/E003/pilot/<model>.log`, tables `$DATA_ROOT/logs/E003/pilot/e003_tables.md`.
- **S3 answer span:** qwen3b-base 100/100 closed spans; qwen7b-instruct 95/100 (5 untagged, all asking for clarification; 1 generation hit 512 tokens).
- **S4 shapes:** qwen3b-base H = [100, 9, 2048], layers 4…36; qwen7b-instruct H = [100, 7, 3584], layers 4…28. Prompt lengths 587–646 / 616–675 tokens.
- **S5:** tag-name token ids search = 1836, answer = 9217 (same in both tokenizers). The token after `<` was the tag-name token in every case. qwen3b-base 100/100 defined (98 first tag search, 2 answer); qwen7b-instruct 99/100 (90 answer, 9 search, 1 with no tag in turn 1).
- **S2 parse:** qwen3b-base 499/500 tagged; qwen7b-instruct 480/500.
- **Analysis smoke test** (`E003_tables.py --pilot`, n = 78 per model after excluding ambiguous/uncertain) ran end to end. The source-only baseline's TriviaQA-only AUROC came out 0.409 / 0.423, not 0.5: the TriviaQA rows get a different constant out-of-fold prediction in each fold, so within-source AUROCs of out-of-fold predictions include between-fold offsets. The same applies to the S4 per-source AUROCs.
- Stop rules: none met. Full run not started; awaiting Erfan.

### Analysis change after the pilot (Erfan, 2026-11-21)
- **Change:** per-source columns (PopQA only, TriviaQA only) of S4 (best and fixed layer) and of the source-only baseline now use probes fit within that source only, with the same nested CV (outer StratifiedKFold(5, seed 0), inner C selection). The all-questions column keeps the all-questions probe. The per-source S4 probes use the layers chosen on all questions (best layer, fixed layer 24/20). Table B stays all-questions only.
- **Reason:** pooled out-of-fold predictions carry between-fold offsets. In the pilot, the source-only baseline's TriviaQA-only AUROC was 0.409 / 0.423 instead of 0.5.
- **Implementation detail:** fitting within source alone still gave 0.460 / 0.471 for the TriviaQA baseline. With constant features, each fold's intercept follows that fold's class balance. A probe whose features are all constant now returns a constant score (AUROC 0.5). This affects only the TriviaQA-only baseline cell. Non-constant within-source probes (S4, PopQA baseline) still pool out-of-fold predictions across 5 folds.
- `signals.csv` keeps the all-questions S4 predictions and adds `S4_best_L*_within_source` / `S4_fixed_L*_within_source`.

### Full run (2026-11-21)
- Started 2026-11-21 ~11:40 with `bash scripts/exp/E003_launch.sh` (log `$DATA_ROOT/logs/E003/launch.log`); GPUs 4–7, shared queue, 7B first. Commit 359ea18, 27 dirty/untracked paths.
- Per model (generation s / forward-pass s, GPU): qwen7b-instruct 252.6 / 276.8 (5), qwen3b-instruct 194.3 / 158.3 (6), searchr1-3b 170.8 / 150.3 (7), grpo-e1a-s60 155.6 / 151.3 (7), qwen3b-base 109.5 / 144.2 (4).
- **Incident (qwen3b-base):** `E003_gen.py` wrote all outputs and `gen_run.json` at 11:44:07, but the vLLM process did not exit, so the launcher never started `E003_hidden.py`. Found at 13:09; I terminated the hung process (pid 1248427) and ran `E003_hidden.py --model qwen3b-base` by hand on GPU 4 (log `$DATA_ROOT/logs/E003/qwen3b-base.hidden.log`, exit 0). The launcher's "qwen3b-base exit 0" line at 13:09:37 reflects the terminated process. Only the manual forward pass wrote `hidden.npz` / `s5_margin.jsonl`.
- Analysis: `.venv-verl/bin/python scripts/exp/E003_tables.py` → `$DATA_ROOT/logs/E003/e003_tables.md` (13:17). The recomputed E002 P2 all-question AUROCs equal the E002 card values (3 dp) for all 5 models.
- Outputs per model in `$DATA_ROOT/logs/E003/<model>/`: `signals.csv` (id, label, source, bin, S1–S5, E002 P2 search rate, within-source S4), `probe_best.npz` / `probe_fixed.npz` (coef, intercept, C, scaler, layer; fit on all questions), `probe_layers.json`, new samples `s2_samples.jsonl`, `s3_greedy.jsonl`, `s5_turn1.jsonl`, `s5_margin.jsonl`, `hidden.npz`. 424 MB in total.
- Coverage on all 1,000 questions:
  - S3 answer span (3B 95% check): qwen3b-base 1000/1000, qwen3b-instruct 1000 (4 unclosed), searchr1-3b 1000 (1 unclosed), grpo-e1a-s60 1000/1000 — all pass. qwen7b-instruct 933/1000 (67 untagged; missing for S3).
  - S2 tagged samples of 5,000: 4,985 / 4,992 / 4,993 / 5,000 / 4,699 (base, instruct, searchr1, grpo, 7B).
  - S5 defined: 1000 / 994 / 987 / 1000 / 996 (the rest: no tag in the greedy first turn).
  - S4: H = [1000, 9, 2048] for the 3B models, [1000, 7, 3584] for the 7B.
- Checkpoints used, all confirmed to exist after the run: the HF snapshots `Qwen2.5-3B@3aab1f19`, `Qwen2.5-3B-Instruct@aa8e7253`, `SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo@d4d2f71e`, `Qwen2.5-7B-Instruct@a09a3545` under `$DATA_ROOT/hf_cache/hub/`, and `$DATA_ROOT/logs/grpo_qwen2.5-3b_full_E1A_em_20261118_102220/hf/grpo_qwen2.5-3b_full_E1A_em_20261118_102220_step60`.

## Result      ← you fill (facts only)

AUROC, unknowable = positive; 95% bootstrap CI. Own E001 labels, ambiguous and uncertain excluded.

#### Table A (all questions)

| model | n unk / know | S1 in-sample consistency | S2 held-out consistency | S3 answer log-prob | S4 best layer | S4 fixed layer | S5 margin | E002 P2 search rate |
|---|---|---|---|---|---|---|---|---|
| qwen3b-base | 484 / 264 | 0.882 [0.858, 0.904] | 0.869 [0.842, 0.893] | 0.882 [0.856, 0.908] | 0.840 [0.811, 0.869] (L20) | 0.836 [0.805, 0.866] (L24) | 0.503 [0.462, 0.547] | 0.579 [0.541, 0.614] |
| qwen3b-instruct | 466 / 278 | 0.761 [0.728, 0.795] | 0.764 [0.731, 0.799] | 0.808 [0.777, 0.839] | 0.856 [0.827, 0.881] (L28) | 0.850 [0.821, 0.876] (L24) | 0.604 [0.560, 0.644] | 0.587 [0.546, 0.626] |
| searchr1-3b | 499 / 273 | 0.851 [0.822, 0.875] | 0.841 [0.811, 0.866] | 0.883 [0.857, 0.907] | 0.845 [0.815, 0.872] (L24) | 0.845 [0.815, 0.872] (L24) | 0.624 [0.582, 0.668] | 0.575 [0.537, 0.613] |
| grpo-e1a-s60 | 485 / 275 | 0.860 [0.834, 0.884] | 0.852 [0.825, 0.878] | 0.875 [0.848, 0.900] | 0.830 [0.800, 0.859] (L20) | 0.822 [0.792, 0.850] (L24) | 0.516 [0.474, 0.560] | 0.559 [0.528, 0.590] |
| qwen7b-instruct | 323 / 451 | 0.815 [0.785, 0.845] | 0.816 [0.789, 0.847] | 0.505 [0.466, 0.545] | 0.860 [0.836, 0.887] (L20) | 0.860 [0.836, 0.887] (L20) | 0.741 [0.706, 0.778] | 0.620 [0.588, 0.653] |

#### Table A (PopQA only)

| model | n unk / know | S1 in-sample consistency | S2 held-out consistency | S3 answer log-prob | S4 best layer | S4 fixed layer | S5 margin | E002 P2 search rate |
|---|---|---|---|---|---|---|---|---|
| qwen3b-base | 256 / 84 | 0.914 [0.882, 0.940] | 0.888 [0.848, 0.921] | 0.904 [0.862, 0.937] | 0.871 [0.827, 0.912] (L20) | 0.874 [0.831, 0.914] (L24) | 0.596 [0.525, 0.664] | 0.574 [0.517, 0.632] |
| qwen3b-instruct | 252 / 86 | 0.757 [0.701, 0.813] | 0.755 [0.697, 0.814] | 0.811 [0.762, 0.859] | 0.862 [0.818, 0.902] (L28) | 0.853 [0.808, 0.894] (L24) | 0.563 [0.492, 0.636] | 0.535 [0.469, 0.604] |
| searchr1-3b | 265 / 88 | 0.879 [0.839, 0.916] | 0.865 [0.825, 0.902] | 0.896 [0.855, 0.932] | 0.874 [0.831, 0.914] (L24) | 0.874 [0.831, 0.914] (L24) | 0.728 [0.666, 0.788] | 0.523 [0.458, 0.588] |
| grpo-e1a-s60 | 261 / 87 | 0.882 [0.842, 0.916] | 0.859 [0.814, 0.900] | 0.898 [0.863, 0.931] | 0.874 [0.833, 0.912] (L20) | 0.865 [0.815, 0.907] (L24) | 0.598 [0.532, 0.663] | 0.561 [0.514, 0.611] |
| qwen7b-instruct | 183 / 148 | 0.806 [0.757, 0.851] | 0.800 [0.752, 0.844] | 0.507 [0.445, 0.566] | 0.850 [0.809, 0.892] (L20) | 0.850 [0.809, 0.892] (L20) | 0.742 [0.681, 0.796] | 0.619 [0.565, 0.669] |

#### Table A (TriviaQA only)

| model | n unk / know | S1 in-sample consistency | S2 held-out consistency | S3 answer log-prob | S4 best layer | S4 fixed layer | S5 margin | E002 P2 search rate |
|---|---|---|---|---|---|---|---|---|
| qwen3b-base | 228 / 180 | 0.857 [0.821, 0.893] | 0.847 [0.811, 0.884] | 0.862 [0.824, 0.898] | 0.788 [0.741, 0.831] (L20) | 0.790 [0.744, 0.834] (L24) | 0.415 [0.362, 0.473] | 0.551 [0.500, 0.599] |
| qwen3b-instruct | 214 / 192 | 0.740 [0.695, 0.791] | 0.744 [0.697, 0.791] | 0.795 [0.750, 0.835] | 0.821 [0.781, 0.860] (L28) | 0.815 [0.777, 0.854] (L24) | 0.627 [0.571, 0.678] | 0.590 [0.535, 0.641] |
| searchr1-3b | 234 / 185 | 0.816 [0.779, 0.855] | 0.815 [0.775, 0.856] | 0.869 [0.831, 0.901] | 0.795 [0.750, 0.835] (L24) | 0.795 [0.750, 0.835] (L24) | 0.555 [0.500, 0.613] | 0.598 [0.546, 0.648] |
| grpo-e1a-s60 | 224 / 188 | 0.835 [0.799, 0.873] | 0.835 [0.800, 0.873] | 0.853 [0.816, 0.889] | 0.770 [0.727, 0.816] (L20) | 0.771 [0.728, 0.816] (L24) | 0.431 [0.372, 0.488] | 0.538 [0.495, 0.578] |
| qwen7b-instruct | 140 / 303 | 0.782 [0.735, 0.829] | 0.796 [0.745, 0.840] | 0.493 [0.438, 0.549] | 0.844 [0.801, 0.883] (L20) | 0.844 [0.801, 0.883] (L20) | 0.710 [0.659, 0.756] | 0.608 [0.561, 0.653] |

#### Table B: S4 out-of-fold AUROC by layer (all questions)

| layer | qwen3b-base | qwen3b-instruct | searchr1-3b | grpo-e1a-s60 | qwen7b-instruct |
|---|---|---|---|---|---|
| 4 | 0.742 | 0.786 | 0.733 | 0.744 | 0.734 |
| 8 | 0.776 | 0.797 | 0.776 | 0.762 | 0.773 |
| 12 | 0.792 | 0.807 | 0.789 | 0.762 | 0.818 |
| 16 | 0.809 | 0.803 | 0.822 | 0.810 | 0.842 |
| 20 | 0.840 | 0.835 | 0.840 | 0.830 | 0.860 |
| 24 | 0.836 | 0.850 | 0.845 | 0.822 | 0.855 |
| 28 | 0.840 | 0.856 | 0.840 | 0.830 | 0.849 |
| 32 | 0.830 | 0.852 | 0.828 | 0.830 | — |
| 36 | 0.816 | 0.849 | 0.814 | 0.816 | — |

#### Table C: source-only baseline probe AUROC (features: PopQA bin 4, PopQA bin 5, TriviaQA)

| model | all | PopQA only | TriviaQA only |
|---|---|---|---|
| qwen3b-base | 0.625 [0.588, 0.666] | 0.585 [0.520, 0.652] | 0.500 [0.500, 0.500] |
| qwen3b-instruct | 0.633 [0.595, 0.672] | 0.595 [0.523, 0.669] | 0.500 [0.500, 0.500] |
| searchr1-3b | 0.621 [0.581, 0.660] | 0.589 [0.532, 0.654] | 0.500 [0.500, 0.500] |
| grpo-e1a-s60 | 0.621 [0.583, 0.662] | 0.573 [0.504, 0.642] | 0.500 [0.500, 0.500] |
| qwen7b-instruct | 0.649 [0.610, 0.686] | 0.671 [0.614, 0.728] | 0.500 [0.500, 0.500] |

Notes:
- qwen3b-base: n = 748 (484 unknowable, 264 knowable); S3 missing 0; S5 missing 0
- qwen3b-instruct: n = 744 (466 unknowable, 278 knowable); S3 missing 0; S5 missing 0
- searchr1-3b: n = 772 (499 unknowable, 273 knowable); S3 missing 0; S5 missing 10
- grpo-e1a-s60: n = 760 (485 unknowable, 275 knowable); S3 missing 0; S5 missing 0
- qwen7b-instruct: n = 774 (323 unknowable, 451 knowable); S3 missing 10; S5 missing 3

### Summary of numbers (all questions unless stated; n per model in the notes above)
- S4 probe: best-layer AUROC 0.830–0.860 (best layers 20, 28, 24, 20, 20); fixed layer (24 for 3B, 20 for 7B) 0.822–0.860. In Table B, layer 4 gives 0.733–0.786; the maximum over layers is at layers 20–28 for every model.
- S2 held-out consistency 0.764–0.869; S1 in-sample consistency 0.761–0.882; S1 − S2 between −0.003 and +0.013 per model.
- S3 answer log-prob: 3B models 0.808–0.883; qwen7b-instruct 0.505 [0.466, 0.545].
- S5 margin: 0.503–0.741. The CI includes 0.5 for qwen3b-base (0.503) and grpo-e1a-s60 (0.516).
- E002 P2 search rate: 0.559–0.620. S4 best layer minus E002: +0.261 (base), +0.269 (instruct), +0.270 (searchr1), +0.271 (grpo), +0.240 (7B); the CIs do not overlap for any model.
- Highest all-question AUROC per model: qwen3b-base S1 and S3 0.882; qwen3b-instruct S4 0.856; searchr1-3b S3 0.883; grpo-e1a-s60 S3 0.875; qwen7b-instruct S4 0.860.
- Within source (S4 from within-source probes): PopQA S4 best 0.850–0.874, TriviaQA 0.770–0.844. S2: PopQA 0.755–0.888, TriviaQA 0.744–0.847. E002: PopQA 0.523–0.619, TriviaQA 0.538–0.608.
- S5 TriviaQA-only is below 0.5 for qwen3b-base 0.415 [0.362, 0.473] and grpo-e1a-s60 0.431 [0.372, 0.488].
- Source-only baseline (Table C): all questions 0.621–0.649; PopQA only (bin) 0.573–0.671; TriviaQA only 0.500 by construction.
