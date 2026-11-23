# search-agent-rl

Training LLM search agents with reinforcement learning. A Qwen2.5 model learns to write `<search>` queries against a 21M-passage Wikipedia index and answer multi-hop questions, using GRPO with an exact-match reward, in the style of [Search-R1](https://arxiv.org/abs/2503.09516).

The pipeline runs on modest hardware (5 × 24 GB GPUs for training, one more for retrieval):
- a dense retrieval service (e5-base-v2, FAISS IVF-SQ8);
- multi-turn rollouts on vLLM through verl's async agent loop, with the loss mask built on token ids;
- one scorer shared by the training reward and evaluation.

Experiment registry (setup, commands, run paths, result tables): [`exps/INDEX.md`](exps/INDEX.md).

## How it works

```
question ─► policy (vLLM) ─► <think>…</think><search>q</search> ─► retrieval service ─► <information>top-3 passages</information>
                 ▲                                                                              │
                 └──────────────── next turn (up to 4 searches) ◄───────────────────────────────┘
                                   … <answer>a</answer> ─► EM reward ─► GRPO update (verl, FSDP)
```

- **Protocol:** Search-R1's prompt and tags, unchanged. Up to 4 search turns plus a final answer turn, top-3 passages per search, retrieved text capped at 500 tokens, 2,048-token episodes.
- **Retrieval** (`retrieval/`): Wikipedia Dec 2018 (`facebook/wiki_dpr`, 21,015,300 passages of 100 words), encoded with `intfloat/e5-base-v2`.
  - The index is FAISS `IVF4096,SQ8` at `nprobe=64`: 16 GB instead of about 65 GB for an exact index, at 0.885 recall@10 relative to exact search.
  - It is served over HTTP, so training and retrieval keep separate dependency stacks.
- **Rollouts** (`agent/`): each episode is a coroutine in verl's async agent loop.
  - The episode is assembled from token-id segments and never re-tokenized from text.
  - The loss mask is 1 on sampled tokens and 0 on retrieved text.
  - Environment text is stripped before the answer is parsed. Without that, text the environment wrote could earn reward.
- **Training:** GRPO with group size 5, lr 1e-6, KL loss 0.001 (low-variance estimator), temperature 1, on Search-R1's NQ + HotpotQA training mix (169,615 questions).
- **Evaluation** (`tests/score_predictions.py`): EM, token F1, cover-match ACC, answer rate and searches per question. The same function computes the training reward.

## Reproduce

Requirements: Linux, CUDA 12.4-capable driver, 6 GPUs with 24 GB (5 train + 1 retrieval), about 90 GB of disk (corpus 7 GB, models and datasets about 20 GB, embeddings 31 GB, index and passage store 28 GB), plus about 50 GB per full checkpoint, and about 20 GB of RAM for the retrieval server. GPUs are assigned in `configs/gpus.env`.

**Data location.** Everything the pipeline writes (corpus, embeddings, index, eval sets, HF/vLLM/Ray caches, training data, run logs) goes under one root, `DATA_ROOT`. It defaults to `./data`. To put it elsewhere, export `DATA_ROOT` or set it in `configs/paths.local.env` (gitignored):
```bash
echo 'DATA_ROOT=/big/disk/search-agent-rl-data' > configs/paths.local.env
```

**Environments.** Retrieval and training need conflicting pins (transformers 5.x / numpy 2 vs. transformers 4.51 / numpy < 2), so there are two venvs:
```bash
uv venv .venv --python 3.10      && uv pip install --python .venv/bin/python      -r requirements.txt        # retrieval, eval
uv venv .venv-verl --python 3.10 && uv pip install --python .venv-verl/bin/python -r requirements-verl.txt   # verl 0.5.0 + vLLM 0.8.5
```
`requirements-verl.txt` installs verl v0.5.0 (commit `8fdc4d3f`) editable from `$DATA_ROOT/src/verl`, so clone it there first, and keep `DATA_ROOT` exported when installing:
```bash
git clone https://github.com/volcengine/verl $DATA_ROOT/src/verl && git -C $DATA_ROOT/src/verl checkout 8fdc4d3f
```

**Pipeline**, one command per stage:
```bash
bash scripts/download_corpus.sh                                   # wiki_dpr passages (no_embeddings/, 21M rows)
bash scripts/download_models.sh                                   # e5, Qwen2.5-3B, Search-R1-3B, train data (pinned revisions)
.venv/bin/python tests/make_eval_sets.py --subset 300            # freeze eval sets (FlashRAG versions) + sha256 manifest
bash scripts/encode_corpus.sh                                     # e5 embeddings of 21M passages, resumable shards
bash scripts/build_index.sh                                       # FAISS IVF4096,SQ8 index + passage store
bash scripts/serve_retrieval.sh                                   # retrieval service on localhost:8000
bash scripts/run_baseline_searchr1.sh                             # released Search-R1-3B through the harness
.venv-verl/bin/python agent/make_train_data.py --out $DATA_ROOT/verl_data/nq_hotpotqa
TRAIN_BATCH=64 MINI_BATCH=32 TOTAL_STEPS=60 WARMUP_RATIO=0.1 TEST_FREQ=20 SAVE_FREQ=20 CKPT_KEEP=1 \
  bash scripts/train_grpo_3b.sh full                              # 60-step GRPO run, about 9 h on 5 × A5000
bash scripts/eval_checkpoint.sh $DATA_ROOT/logs/<run>/ckpt/global_step_60   # merge FSDP shards, evaluate
```

`DRY_RUN=1 bash scripts/train_grpo_3b.sh full` prints the resolved verl config without starting anything. Search-R1's full recipe is `TRAIN_BATCH=512 MINI_BATCH=256 TOTAL_STEPS=1005 MAX_RESPONSE=3072`.

## Layout

| path | contents |
|---|---|
| `retrieval/` | corpus encoding, index build, retrieval server, benchmark |
| `agent/` | rollout protocol (`search_r1_rollout.py`), verl agent loop, reward, QA metrics, training data |
| `baselines/` | Search-R1-style inference for evaluating any checkpoint |
| `tests/` | eval-set builder, scorer, unit tests for the metrics and the rollout contract (loss mask, tags, reward) |
| `scripts/` | launch scripts for every stage |
| `configs/` | GPU assignment, agent-loop settings |
| `exps/` | experiment cards and index |
