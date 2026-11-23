# E001: Closed-book knowledge labels on PopQA
Question: For each model, which PopQA questions can it answer from memory (knowable) and which can't it (unknowable), and how does closed-book accuracy vary with entity popularity?
Parent: none

## Setup
*(Revision 2, 2026-11-20, after pilot 1. Revision 1 is kept under Execution.)*
- Models/checkpoints:
  1. Qwen/Qwen2.5-3B (base)
  2. Qwen/Qwen2.5-3B-Instruct
  3. PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo (Search-R1, RL)
  4. grpo-e1a-s60: L006 run grpo_qwen2.5-3b_full_E1A_em_20261118_102220, step 60 (merged HF weights at $DATA_ROOT/logs/grpo_qwen2.5-3b_full_E1A_em_20261118_102220/hf/grpo_qwen2.5-3b_full_E1A_em_20261118_102220_step60). Replaces the L001 step-60 checkpoint ($DATA_ROOT/logs/grpo_qwen2.5-3b_full_20261019_183501/ckpt/global_step_60), which was deleted between 2026-11-11 and 2026-11-17. Recipe differences from L001: 100-step schedule instead of 60 (same 6 warmup steps), 4 GPUs instead of 5, BALANCE_BATCH=False, launched from ~/aget-tool-use at 633dfa3. Not retrained (Erfan, 2026-11-21).
  5. Qwen/Qwen2.5-7B-Instruct (scale check)
- Data: rebuild the frozen set as data/e001_easy_1k.jsonl:
  - 500 PopQA questions from the top 2 quintiles of log10(s_pop), 250 per quintile, seed 0
  - 500 TriviaQA questions (mandarjoshi/trivia_qa, config rc.nocontext, validation split), seed 0, with answer aliases as gold answers
  - Fields: id, source, question, answers, s_pop (PopQA only), bin.
- Procedure: closed-book only. No search tool in the prompt and no retrieval server. Prompt: a 3-shot demonstration before the question, for all models (instruct models: in the chat template as one user turn). 3 simple factual QA examples NOT drawn from the evaluation set, each in the exact format:
  ```
  Question: <q>
  <think></think>
  <answer><short entity></answer>
  ```
  Then the test question, ending with "Question: <q>\n". Saved in scripts/exp/E001_prompts.py.
- Decoding: max 512 new tokens, stop sequence "</answer>" (kept in the output), temperature 0.7, top_p 1.0, 5 samples, vLLM.
- Parse rule: text inside <answer>…</answer>. If <answer> is opened but not closed, take the text from <answer> to the end. No tags = wrong (counted separately as untagged). Do not match against the full output.
- Metrics: Correct = any alias in the gold answers matches the parsed answer after normalize_answer (agent/qa_metrics.py), exactly or as a substring (alias in answer). Report EM (exact only) as well.
- Labels per question per model: knowable = ≥3/5 correct, unknowable = 0/5, uncertain = 1–2/5.
- Untagged samples are classified as clarification (asks for context or calls the question ambiguous) or other untagged, and counted separately per model in Table A. Per-question flag ambiguous = true if any model asks for clarification in any sample; written to data/e001 labels. Label counts are reported twice: all questions, and with ambiguous questions excluded.
- Reuse: none.

## Pilot first
Rerun the pilot (50 questions: 25 PopQA + 25 TriviaQA) × all 5 models. Report: parse rate, untagged count, token-limit hits, and knowable / unknowable / uncertain per model, split by source. Stop if any parse rate is below 95% or any 3B model has fewer than 10 knowable out of 50.

Full run (Erfan, 2026-11-21): all 1,000 questions × 5 models. qwen7b-instruct's parse rate is accepted, with no prompt or stop-rule change. The 95% parse rule applies to the 3B models only: stop only if a 3B model drops below 95%.

## Output wanted
1. Table A, label counts: rows = models; columns = n knowable, n unknowable, n uncertain, parse rate.
2. Table B, closed-book accuracy (mean over 5 samples) by popularity bin: rows = bins 1–5 (with the s_pop range of each), columns = models. Include standard errors.
3. Table C, overlap: for each pair of models, the share of questions knowable to both.
4. Files: $DATA_ROOT/logs/E001/<model tag>/samples.jsonl (all 5 raw outputs per question) and labels.jsonl (id, n_correct, label), plus data/e001_easy_1k.jsonl.

## Execution   ← you fill

### Full run (2026-11-21)
- **Command:** `bash scripts/exp/E001_launch.sh` (log `$DATA_ROOT/logs/E001/launch_full.log`). qwen3b-base and then qwen7b-instruct ran on GPU 4; qwen3b-instruct, searchr1-3b and grpo-e1a-s60 on GPUs 5, 6 and 7. Started 2026-11-21 ~00:12 and ended 00:17:25.
- **Commit:** 0f7325e, dirty: 27 uncommitted paths (the depth-study restore, `exps/`, `scripts/exp/`, CLAUDE.md, README.md).
- **Set:** `$DATA_ROOT/eval/e001/e001_easy_1k.jsonl`, sha256 `1c45efef…`, unchanged from pilot 2.

| run ID | model | GPU | wall (s, incl. load) | output |
|---|---|---|---|---|
| E001-qwen3b-base-s0 | Qwen/Qwen2.5-3B @ 3aab1f19 | 4 | 69.9 | `$DATA_ROOT/logs/E001/E001-qwen3b-base-s0/` |
| E001-qwen3b-instruct-s0 | Qwen/Qwen2.5-3B-Instruct @ aa8e7253 | 5 | 91.1 | `$DATA_ROOT/logs/E001/E001-qwen3b-instruct-s0/` |
| E001-searchr1-3b-s0 | PeterJinGo/SearchR1-…-em-grpo @ d4d2f71e | 6 | 69.8 | `$DATA_ROOT/logs/E001/E001-searchr1-3b-s0/` |
| E001-grpo-e1a-s60-s0 | grpo-e1a-s60 (path below) | 7 | 66.3 | `$DATA_ROOT/logs/E001/E001-grpo-e1a-s60-s0/` |
| E001-qwen7b-instruct-s0 | Qwen/Qwen2.5-7B-Instruct @ a09a3545 | 4 | 193.7 | `$DATA_ROOT/logs/E001/E001-qwen7b-instruct-s0/` |

- **Per-run files:** `samples.jsonl` (raw, 5 outputs per question), `labels.jsonl` (id, n_correct, n_em, n_untagged, n_clarification, n_other_untagged, label, ambiguous), `untagged.jsonl` (every untagged sample with its class) and `run.json`.
- **Cross-model labels:** `$DATA_ROOT/eval/e001/e001_easy_1k_labels.jsonl` (card: "data/e001 labels"). Fields: id, source, bin, ambiguous, clarification_models, labels per model, n_correct per model.
- **Tables:** `.venv/bin/python scripts/exp/E001_tables.py`, output saved to `$DATA_ROOT/logs/E001/tables.md`.
- **Clarification classifier (my implementation):** a case-insensitive regex (`CLARIFY_RE` in `scripts/exp/E001_tables.py`), applied to untagged samples only. It matches "more/additional context|information|details", "enough context|information", "specify", "clarif", "ambiguous", "incomplete", "could/can/may/might refer to", "referring to" and "which one|version|work". Manual review:
  - All 24 untagged pilot-2 samples of qwen7b-instruct (18 + 3 borderline ones counted as clarification, all of which ask for context) and all 6 of qwen3b-instruct.
  - A full-run spot check of 8 clarification and 10 other samples from qwen7b-instruct, plus all 2 from qwen3b-base and 4 from qwen3b-instruct.
  - Under the card's definition, "clarification" includes replies that say the model lacks information and ask for more context about an unambiguous entity (e.g. "I don't have specific information about Richard Sandrak's occupation. Could you provide more context…").
- **Checkpoints after the run (all exist, checked 2026-11-21):**
  - `$DATA_ROOT/logs/grpo_qwen2.5-3b_full_E1A_em_20261118_102220/hf/grpo_qwen2.5-3b_full_E1A_em_20261118_102220_step60/` (safetensors 4.98 GB + 1.81 GB)
  - `$DATA_ROOT/hf_cache/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`
  - `$DATA_ROOT/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1`
  - `$DATA_ROOT/hf_cache/hub/models--PeterJinGo--SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo/snapshots/d4d2f71e39ff3e0ed4def1e8f860b87d9f6bfac7`
  - `$DATA_ROOT/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`
- **Stop rule:** the 3B parse rates were 0.997–1.000, so no stop.

### Decisions after pilot 2 (Erfan, 2026-11-21)
- **Model 4 = grpo-e1a-s60** (L006 run `grpo_qwen2.5-3b_full_E1A_em_20261118_102220`, step 60). The specified L001 checkpoint was deleted, and grpo-e1a-s60 differs from it in recipe as listed in Setup. Not retrained.
- **qwen7b-instruct parse rate (0.904 in pilot 2) accepted.** No prompt or stop-rule change. Untagged samples are classified as clarification or other, an `ambiguous` per-question flag is added, and label counts are reported with and without ambiguous questions. The 95% parse rule applies to 3B models only.

### Revision 2 (2026-11-20): changes after pilot 1 and their reason
Reason (Erfan): in pilot 1, 4 of 5 models parsed below 95% (see Revision 1 below).
- **Prompt:** 3-shot demonstrations added, all models. The questions ask for a William Shakespeare, a France and a Leonardo da Vinci answer. None occurs, exact or as a substring, in PopQA test or TriviaQA rc.nocontext validation. "What is the capital of Japan?" was rejected because it is a PopQA question. *My choice:* the instruction line comes first, then a blank line between the instruction, each demo and the test question.
- **Decoding:** 512 tokens, stop `</answer>` included in the output, temperature 0.7.
- **Parse:** closed block, or open `<answer>` to the end; untagged = wrong.
- **Data:** new set `$DATA_ROOT/eval/e001/e001_easy_1k.jsonl` (card path `data/e001_easy_1k.jsonl`; all data under `$DATA_ROOT` per CLAUDE.md). sha256 `1c45efefa89ef73f08564ed25ab002440a74050b3fd245d8654a3fd09ebf179b`.
  - PopQA revision `098765c7`. Bins 4 and 5 use the same quintile edges as revision 1, computed over all 14,267 rows. Sampled s_pop ranges: bin 4 1,845–9,700, bin 5 9,764–1,629,691. IDs are prefixed `popqa_`.
  - TriviaQA revision `0f7faf33`. rc.nocontext validation has 17,944 rows but only 9,960 unique `question_id`s, because each question repeats once per evidence document. *My choice:* keep the first row per `question_id`, then sample 500 with a separate `default_rng(0)`. Gold answers = `answer.aliases`. IDs are prefixed `tqa_`, and `bin`/`s_pop` = null.
- **Labels:** knowable ≥ 3/5.
- **Pilot subset (my choice):** the first 13 PopQA bin-4, first 12 PopQA bin-5 and first 25 TriviaQA rows in file order.
- **Model 4 (unresolved):** the specified checkpoint `…_20261019_183501/ckpt/global_step_60` does not exist. It was deleted before this session; the run dir holds only logs and config. The pilot still runs the substitute `grpo-e1a-s60` (E1 arm A step 60, see L006) and labels it as such.
- Revision-1 outputs stay in place: `$DATA_ROOT/eval/popqa/popqa_e001_1k.jsonl` and `$DATA_ROOT/logs/E001/pilot/`. Revision-2 pilot outputs go to `$DATA_ROOT/logs/E001/pilot2/`.

**Pilot 2 (2026-11-20, revision 2):** 50 questions (25 PopQA + 25 TriviaQA) × 5 models × 5 samples. GPUs 4–7, 100–127 s per model including load. Outputs in `$DATA_ROOT/logs/E001/pilot2/E001-<tag>-s0/`. The first launch failed for three models because of a quoting bug in my launch command (empty log path; nothing ran). They were rerun unchanged.

| model | source | n | knowable | unknowable | uncertain | parse rate | untagged | open, not closed | token-limit hits |
|---|---|---|---|---|---|---|---|---|---|
| qwen3b-base | PopQA | 25 | 4 | 20 | 1 | 1.000 | 0/125 | 0 | 0 |
| qwen3b-base | TriviaQA | 25 | 9 | 13 | 3 | 1.000 | 0/125 | 0 | 0 |
| qwen3b-instruct | PopQA | 25 | 4 | 19 | 2 | 0.992 | 1/125 | 0 | 0 |
| qwen3b-instruct | TriviaQA | 25 | 7 | 14 | 4 | 0.992 | 1/125 | 1 | 0 |
| searchr1-3b | PopQA | 25 | 4 | 20 | 1 | 1.000 | 0/125 | 0 | 0 |
| searchr1-3b | TriviaQA | 25 | 7 | 12 | 6 | 1.000 | 0/125 | 0 | 0 |
| grpo-e1a-s60 (substitute) | PopQA | 25 | 4 | 20 | 1 | 1.000 | 0/125 | 0 | 0 |
| grpo-e1a-s60 (substitute) | TriviaQA | 25 | 9 | 14 | 2 | 1.000 | 0/125 | 0 | 0 |
| qwen7b-instruct | PopQA | 25 | 7 | 17 | 1 | 0.840 | 20/125 | 0 | 0 |
| qwen7b-instruct | TriviaQA | 25 | 9 | 10 | 6 | 0.968 | 4/125 | 0 | 1 |

Knowable out of 50: qwen3b-base 13, qwen3b-instruct 11, searchr1-3b 11, grpo-e1a-s60 13, qwen7b-instruct 16. qwen7b-instruct's overall parse rate is 0.904 (24/250 untagged). Its 20 untagged PopQA samples fall on 8 questions. In 18 of the 20 the model says the title is ambiguous or asks for more context (e.g. "Who was the producer of Loose?" → "Could you please provide more details…"); of the other 2, one answers in plain text without tags and one says no widely known work by that name exists. The 4 untagged TriviaQA samples (4 questions) are plain-text answers or reasoning without an `<answer>` tag. Stop rule met (parse rate < 95%); the full run was not started.

### Revision 1
Setup as first given: PopQA only, 1,000 questions (5 quintiles of log10(s_pop) × 200, seed 0). 0-shot prompt "Answer the given question. Think inside <think></think> if needed, then give the final answer inside <answer></answer>. Question: …" (chat template for instruct models). Temperature 1.0, top_p 1.0, 5 samples, 256 tokens, no stop. Correct = alias in parsed answer. Labels: knowable ≥ 4/5, unknowable 0/5, uncertain 1–3/5.


**Deviations and choices (approved by Erfan 2026-11-20 unless marked):**
- **Model 4:** the L001 step-60 checkpoint no longer exists (`…_20261019_183501/ckpt/` gone). Substituted: E1 arm A step 60, `$DATA_ROOT/logs/grpo_qwen2.5-3b_full_E1A_em_20261118_102220/hf/grpo_qwen2.5-3b_full_E1A_em_20261118_102220_step60` (tag `grpo-e1a-s60`).
- **Frozen set location:** `$DATA_ROOT/eval/popqa/popqa_e001_1k.jsonl` (not repo `data/`), sha256 `a33c6ffdfd889c7392847af5486a1259b3b929a05440c0ae83eb410d534a3566`; metadata in `popqa_e001_1k.meta.json`.
- **Correct:** a normalized alias is a substring of the normalized parsed answer (`agent.qa_metrics.cover_match`), so exact matches count too. EM: `exact_match`.
- **Prompt:** the card's string with the PopQA question appended verbatim. Nothing after it.
- *Not specified in the card (my choice):* the parsed answer is the **first** complete `<answer>…</answer>` block. No stop strings. Raw text and `n_answer_blocks` are saved, so the last-block rule can be recomputed.
- *Not specified in the card (my choice):* vLLM `seed=0`, bf16, `max_model_len=1024`. The instruct chat templates include Qwen's default system prompt.
- *Pilot subset (my choice):* the first 10 questions of each bin, in file order (50).

**Frozen set:** PopQA test, revision `098765c79ea10a2cb19c828324e33281b8336ec0`, 14,267 rows. Bins are quintiles of log10(s_pop), with source bin sizes 2,847–2,858. Sampled s_pop ranges: bin 1: 14–175, bin 2: 179–552, bin 3: 555–1,838, bin 4: 1,848–9,441, bin 5: 9,786–706,284.

**Models (local snapshots):** Qwen2.5-3B `3aab1f19`, Qwen2.5-3B-Instruct `aa8e7253`, Search-R1-3B `d4d2f71e`, Qwen2.5-7B-Instruct `a09a3545`, grpo-e1a-s60 (path above).

**Code:** `scripts/exp/E001_prompts.py`, `E001_build_set.py`, `E001_run.py`, `E001_tables.py`. Commit 0f7325e, dirty (27 uncommitted paths, mostly the depth-study restore and the new `exps/` files).

**Pilot (2026-11-20):** 50 questions × 5 models × 5 samples, one model per GPU (4–7; 7B on GPU 4 after the 3B runs). Outputs in `$DATA_ROOT/logs/E001/pilot/E001-<tag>-s0/{samples,labels,run}.jsonl`, logs in `$DATA_ROOT/logs/E001/pilot/<tag>.log`. Wall time 112–168 s per model, including model load.

| model | parse rate | unparsed / 250 | knowable | unknowable | uncertain | outputs that hit the 256-token cap |
|---|---|---|---|---|---|---|
| qwen3b-base | 0.664 | 84 | 0 | 46 | 4 | 20 |
| qwen3b-instruct | 0.920 | 20 | 0 | 42 | 8 | 6 |
| searchr1-3b | 0.896 | 26 | 0 | 49 | 1 | 48 |
| grpo-e1a-s60 | 0.932 | 17 | 0 | 47 | 3 | 0 |
| qwen7b-instruct | 0.996 | 1 | 11 | 38 | 1 | 0 |

Four of five models have a parse rate below 95%. Per the card, the run stopped before the full set.

## Result      ← you fill (facts only)

Full run: 1,000 questions (500 PopQA from s_pop quintiles 4–5, 500 TriviaQA) × 5 samples per model. Knowable = ≥3/5 correct; correct = a normalized gold alias is a substring of the normalized parsed answer.

**Ambiguous questions:** 113 / 1,000 (PopQA 108, TriviaQA 5). Flagged by qwen7b-instruct only: 101; qwen3b-instruct only: 4; qwen3b-instruct and qwen7b-instruct: 6; qwen3b-base only: 1; qwen3b-base and qwen7b-instruct: 1.

### Table A1: parsing (all questions)
| model | source | n | parse rate | untagged | clarification | other untagged | open, not closed | token-limit hits |
|---|---|---|---|---|---|---|---|---|
| qwen3b-base | popqa | 500 | 1.000 | 0/2500 | 0 | 0 | 1 | 0 |
| qwen3b-base | triviaqa | 500 | 0.994 | 15/2500 | 2 | 13 | 2 | 3 |
| qwen3b-base | all | 1000 | 0.997 | 15/5000 | 2 | 13 | 3 | 3 |
| qwen3b-instruct | popqa | 500 | 0.996 | 9/2500 | 9 | 0 | 14 | 0 |
| qwen3b-instruct | triviaqa | 500 | 0.998 | 5/2500 | 2 | 3 | 8 | 0 |
| qwen3b-instruct | all | 1000 | 0.997 | 14/5000 | 11 | 3 | 22 | 0 |
| searchr1-3b | popqa | 500 | 1.000 | 0/2500 | 0 | 0 | 3 | 3 |
| searchr1-3b | triviaqa | 500 | 0.998 | 5/2500 | 0 | 5 | 1 | 1 |
| searchr1-3b | all | 1000 | 0.999 | 5/5000 | 0 | 5 | 4 | 4 |
| grpo-e1a-s60 | popqa | 500 | 1.000 | 0/2500 | 0 | 0 | 1 | 0 |
| grpo-e1a-s60 | triviaqa | 500 | 1.000 | 1/2500 | 0 | 1 | 1 | 0 |
| grpo-e1a-s60 | all | 1000 | 1.000 | 1/5000 | 0 | 1 | 2 | 0 |
| qwen7b-instruct | popqa | 500 | 0.891 | 272/2500 | 252 | 20 | 1 | 0 |
| qwen7b-instruct | triviaqa | 500 | 0.991 | 22/2500 | 3 | 19 | 1 | 0 |
| qwen7b-instruct | all | 1000 | 0.941 | 294/5000 | 255 | 39 | 2 | 0 |

### Table A2: label counts
All questions:

| model | source | n | knowable | unknowable | uncertain |
|---|---|---|---|---|---|
| qwen3b-base | popqa | 500 | 85 | 358 | 57 |
| qwen3b-base | triviaqa | 500 | 180 | 232 | 88 |
| qwen3b-base | all | 1000 | 265 | 590 | 145 |
| qwen3b-instruct | popqa | 500 | 86 | 357 | 57 |
| qwen3b-instruct | triviaqa | 500 | 192 | 218 | 90 |
| qwen3b-instruct | all | 1000 | 278 | 575 | 147 |
| searchr1-3b | popqa | 500 | 90 | 367 | 43 |
| searchr1-3b | triviaqa | 500 | 185 | 238 | 77 |
| searchr1-3b | all | 1000 | 275 | 605 | 120 |
| grpo-e1a-s60 | popqa | 500 | 88 | 364 | 48 |
| grpo-e1a-s60 | triviaqa | 500 | 188 | 228 | 84 |
| grpo-e1a-s60 | all | 1000 | 276 | 592 | 132 |
| qwen7b-instruct | popqa | 500 | 148 | 290 | 62 |
| qwen7b-instruct | triviaqa | 500 | 303 | 144 | 53 |
| qwen7b-instruct | all | 1000 | 451 | 434 | 115 |

Ambiguous questions excluded:

| model | source | n | knowable | unknowable | uncertain |
|---|---|---|---|---|---|
| qwen3b-base | popqa | 392 | 84 | 256 | 52 |
| qwen3b-base | triviaqa | 495 | 180 | 228 | 87 |
| qwen3b-base | all | 887 | 264 | 484 | 139 |
| qwen3b-instruct | popqa | 392 | 86 | 252 | 54 |
| qwen3b-instruct | triviaqa | 495 | 192 | 214 | 89 |
| qwen3b-instruct | all | 887 | 278 | 466 | 143 |
| searchr1-3b | popqa | 392 | 88 | 265 | 39 |
| searchr1-3b | triviaqa | 495 | 185 | 234 | 76 |
| searchr1-3b | all | 887 | 273 | 499 | 115 |
| grpo-e1a-s60 | popqa | 392 | 87 | 261 | 44 |
| grpo-e1a-s60 | triviaqa | 495 | 188 | 224 | 83 |
| grpo-e1a-s60 | all | 887 | 275 | 485 | 127 |
| qwen7b-instruct | popqa | 392 | 148 | 183 | 61 |
| qwen7b-instruct | triviaqa | 495 | 303 | 140 | 52 |
| qwen7b-instruct | all | 887 | 451 | 323 | 113 |

Table B: closed-book accuracy (alias in answer), %, mean over 5 samples ± SE over questions (all questions)

| group | s_pop range | n | qwen3b-base | qwen3b-instruct | searchr1-3b | grpo-e1a-s60 | qwen7b-instruct |
|---|---|---|---|---|---|---|---|
| PopQA bin 4 | 1845–9700 | 250 | 7.3 ± 1.3 | 8.0 ± 1.5 | 7.4 ± 1.4 | 7.4 ± 1.4 | 13.8 ± 1.9 |
| PopQA bin 5 | 9764–1629691 | 250 | 28.8 ± 2.5 | 28.7 ± 2.5 | 29.0 ± 2.6 | 28.9 ± 2.5 | 45.7 ± 2.7 |
| TriviaQA |  | 500 | 36.0 ± 1.8 | 39.1 ± 1.9 | 37.0 ± 1.9 | 37.6 ± 1.8 | 59.9 ± 2.0 |
| all |  | 1000 | 27.0 ± 1.2 | 28.7 ± 1.2 | 27.6 ± 1.2 | 27.9 ± 1.2 | 44.8 ± 1.4 |

Table B: closed-book EM, %, mean over 5 samples ± SE over questions (all questions)

| group | s_pop range | n | qwen3b-base | qwen3b-instruct | searchr1-3b | grpo-e1a-s60 | qwen7b-instruct |
|---|---|---|---|---|---|---|---|
| PopQA bin 4 | 1845–9700 | 250 | 5.8 ± 1.2 | 4.0 ± 1.0 | 6.3 ± 1.3 | 6.3 ± 1.2 | 9.4 ± 1.6 |
| PopQA bin 5 | 9764–1629691 | 250 | 27.3 ± 2.5 | 18.1 ± 2.1 | 27.5 ± 2.5 | 27.4 ± 2.5 | 31.4 ± 2.6 |
| TriviaQA |  | 500 | 30.7 ± 1.7 | 27.3 ± 1.7 | 33.0 ± 1.8 | 33.2 ± 1.8 | 48.0 ± 2.0 |
| all |  | 1000 | 23.6 ± 1.1 | 19.2 ± 1.1 | 25.0 ± 1.2 | 25.0 ± 1.2 | 34.2 ± 1.3 |

Table C: share of questions knowable to both (all questions; diagonal = share knowable)

| | qwen3b-base | qwen3b-instruct | searchr1-3b | grpo-e1a-s60 | qwen7b-instruct |
|---|---|---|---|---|---|
| qwen3b-base | 0.265 | 0.212 | 0.251 | 0.257 | 0.237 |
| qwen3b-instruct | 0.212 | 0.278 | 0.214 | 0.218 | 0.253 |
| searchr1-3b | 0.251 | 0.214 | 0.275 | 0.252 | 0.244 |
| grpo-e1a-s60 | 0.257 | 0.218 | 0.252 | 0.276 | 0.247 |
| qwen7b-instruct | 0.237 | 0.253 | 0.244 | 0.247 | 0.451 |

### Table B: closed-book accuracy by group (%, mean over 5 samples ± SE over questions; all questions)
| group | s_pop range | n | qwen3b-base | qwen3b-instruct | searchr1-3b | grpo-e1a-s60 | qwen7b-instruct |
|---|---|---|---|---|---|---|---|
| PopQA bin 4 | 1845–9700 | 250 | 7.3 ± 1.3 | 8.0 ± 1.5 | 7.4 ± 1.4 | 7.4 ± 1.4 | 13.8 ± 1.9 |
| PopQA bin 5 | 9764–1629691 | 250 | 28.8 ± 2.5 | 28.7 ± 2.5 | 29.0 ± 2.6 | 28.9 ± 2.5 | 45.7 ± 2.7 |
| TriviaQA |  | 500 | 36.0 ± 1.8 | 39.1 ± 1.9 | 37.0 ± 1.9 | 37.6 ± 1.8 | 59.9 ± 2.0 |
| all |  | 1000 | 27.0 ± 1.2 | 28.7 ± 1.2 | 27.6 ± 1.2 | 27.9 ± 1.2 | 44.8 ± 1.4 |

EM (exact only):

| group | s_pop range | n | qwen3b-base | qwen3b-instruct | searchr1-3b | grpo-e1a-s60 | qwen7b-instruct |
|---|---|---|---|---|---|---|---|
| PopQA bin 4 | 1845–9700 | 250 | 5.8 ± 1.2 | 4.0 ± 1.0 | 6.3 ± 1.3 | 6.3 ± 1.2 | 9.4 ± 1.6 |
| PopQA bin 5 | 9764–1629691 | 250 | 27.3 ± 2.5 | 18.1 ± 2.1 | 27.5 ± 2.5 | 27.4 ± 2.5 | 31.4 ± 2.6 |
| TriviaQA |  | 500 | 30.7 ± 1.7 | 27.3 ± 1.7 | 33.0 ± 1.8 | 33.2 ± 1.8 | 48.0 ± 2.0 |
| all |  | 1000 | 23.6 ± 1.1 | 19.2 ± 1.1 | 25.0 ± 1.2 | 25.0 ± 1.2 | 34.2 ± 1.3 |

### Table C: share of questions knowable to both (all 1,000 questions; diagonal = share knowable to that model)
| | qwen3b-base | qwen3b-instruct | searchr1-3b | grpo-e1a-s60 | qwen7b-instruct |
|---|---|---|---|---|---|
| qwen3b-base | 0.265 | 0.212 | 0.251 | 0.257 | 0.237 |
| qwen3b-instruct | 0.212 | 0.278 | 0.214 | 0.218 | 0.253 |
| searchr1-3b | 0.251 | 0.214 | 0.275 | 0.252 | 0.244 |
| grpo-e1a-s60 | 0.257 | 0.218 | 0.252 | 0.276 | 0.247 |
| qwen7b-instruct | 0.237 | 0.253 | 0.244 | 0.247 | 0.451 |

### Summary
- **Parse rates:** 0.997–1.000 for the four 3B models; 0.941 for qwen7b-instruct (294/5,000 untagged: 255 clarification, 39 other). qwen7b-instruct's untagged samples are mostly on PopQA (272/2,500, 252 of them clarification). Token-limit hits: 0–4 per model.
- **Knowable / unknowable / uncertain of 1,000:**

  | model | knowable | unknowable | uncertain |
  |---|---|---|---|
  | qwen3b-base | 265 | 590 | 145 |
  | qwen3b-instruct | 278 | 575 | 147 |
  | searchr1-3b | 275 | 605 | 120 |
  | grpo-e1a-s60 | 276 | 592 | 132 |
  | qwen7b-instruct | 451 | 434 | 115 |

  Excluding the 113 ambiguous questions (n = 887), knowable counts change by 0–2 per model. Unknowable counts fall by 106–111 per model.
- **Accuracy, all 1,000 questions:**
  - 3B models: 27.0–28.7% (SE 1.2).
  - qwen7b-instruct: 44.8% (SE 1.4).
  - PopQA bin 4 (s_pop 1,845–9,700, n = 250): 7.3–8.0% for 3B, 13.8% for 7B.
  - PopQA bin 5 (s_pop 9,764–1,629,691, n = 250): 28.7–29.0% for 3B, 45.7% for 7B.
  - TriviaQA (n = 500): 36.0–39.1% for 3B, 59.9% for 7B.
- **EM, all questions:** 19.2% (qwen3b-instruct) to 25.0% (searchr1-3b, grpo-e1a-s60) for 3B; 34.2% for 7B.
- **Overlap:** the share of questions knowable to both models in a 3B pair ranges from 0.212 (qwen3b-base & qwen3b-instruct) to 0.257 (qwen3b-base & grpo-e1a-s60).
