# search-agent-rl

Training LLM search agents with reinforcement learning. A Qwen2.5 model learns to write `<search>` queries against a 21M-passage Wikipedia index and answer multi-hop questions, using GRPO with an exact-match reward, in the style of [Search-R1](https://arxiv.org/abs/2503.09516).

The pipeline runs on modest hardware (5 × 24 GB GPUs for training, one more for retrieval):
- a dense retrieval service (e5-base-v2, FAISS IVF-SQ8);
- multi-turn rollouts on vLLM through verl's async agent loop, with the loss mask built on token ids;
- one scorer shared by the training reward and evaluation.

## Results

Qwen2.5-3B (base), GRPO for 60 steps: 64 questions × 5 samples per step, 8.8 h on 5 × RTX A5000. That is 3,840 training questions, 0.75% of Search-R1's 514,560. Every row below is evaluated with the same harness: same prompt, retriever, turn limits, greedy decoding and scorer. EM is in %, on fixed 300-question dev subsets (Bamboogle: all 125 questions).

| model | HotpotQA | 2Wiki | MuSiQue | Bamboogle | avg, 3 multi-hop | searches / q |
|---|---|---|---|---|---|---|
| Qwen2.5-3B base, prompted | 8.7 | 7.7 | 2.3 | 7.2 | 6.2 | 0.8–1.1 |
| **+ GRPO, 60 steps (this repo)** | 23.7 | 28.7 | 7.7 | 4.8 | **20.0** | 1.0 |
| Search-R1-3B, released checkpoint | 27.3 | 26.0 | 6.3 | — | 19.9 | 1.0 |

- **Ties the released Search-R1-3B with 0.75% of its training data.** A paired sign test on identical questions finds no significant difference on any set (p = 0.14–0.48). Search-R1's own reported numbers for this checkpoint are 29.7, 27.4 and 6.6 ([2503.09516](https://arxiv.org/abs/2503.09516) v5, Table 3); our harness lands within 0.3–2.4 EM of them.
- **Format is learned in about 20 steps.** By step 20, 97–100% of episodes end with an answer, up from 53–68%. Accuracy among answered episodes keeps rising after that on 2Wiki and MuSiQue.
- **Neither agent learns multi-hop search.** Both make one search per question, even on 4-hop MuSiQue. Our agent often uses the question itself as the query: verbatim in 29–59% of episodes. Bamboogle, whose questions need a second, dependent search, drops below the base model.

Full per-step results, training dynamics and the error analysis: [`notes/results_grpo_3b_60step.md`](notes/results_grpo_3b_60step.md). Evaluating the released Search-R1 checkpoint: [`notes/baseline_searchr1.md`](notes/baseline_searchr1.md).

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

Requirements: Linux, CUDA 12.4-capable driver, 6 GPUs with 24 GB (5 train + 1 retrieval), about 300 GB of disk (the `wiki_dpr` download is about 240 GB, embeddings 31 GB, index and passage store 28 GB), plus about 50 GB per full checkpoint, and about 20 GB of RAM for the retrieval server. Paths default to `/data/erfan/search-agent-rl-data`; edit `DATA` in the scripts for your machine. GPUs are assigned in `configs/gpus.env`.

**Environments.** Retrieval and training need conflicting pins (transformers 5.x / numpy 2 vs. transformers 4.51 / numpy < 2), so there are two venvs:
```bash
uv venv .venv --python 3.10      && uv pip install --python .venv/bin/python      -r requirements.txt        # retrieval, eval
uv venv .venv-verl --python 3.10 && uv pip install --python .venv-verl/bin/python -r requirements-verl.txt   # verl 0.5.0 + vLLM 0.8.5
```
`requirements-verl.txt` installs verl v0.5.0 (commit `8fdc4d3f`) from a local checkout.

**Pipeline**, one command per stage:
```bash
.venv/bin/python tests/make_eval_sets.py --subset 300            # freeze eval sets (FlashRAG versions) + sha256 manifest
bash scripts/encode_corpus.sh                                     # e5 embeddings of 21M passages, resumable shards
bash scripts/build_index.sh                                       # FAISS IVF4096,SQ8 index + passage store
bash scripts/serve_retrieval.sh                                   # retrieval service on localhost:8000
bash scripts/run_baseline_searchr1.sh                             # released Search-R1-3B through the harness
.venv-verl/bin/python agent/make_train_data.py --out $DATA/verl_data/nq_hotpotqa
TRAIN_BATCH=64 MINI_BATCH=32 TOTAL_STEPS=60 WARMUP_RATIO=0.1 TEST_FREQ=20 SAVE_FREQ=20 CKPT_KEEP=1 \
  bash scripts/train_grpo_3b.sh full                              # 60-step GRPO run, about 9 h on 5 × A5000
bash scripts/eval_checkpoint.sh $DATA/logs/<run>/ckpt/global_step_60   # merge FSDP shards, evaluate, list misses
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

## Limitations

- **One training seed**, evaluated on 300-question subsets. The standard error is about 2.6 EM per set.
- **Short training:** 60 steps covers the early regime only.
- **Approximate retrieval:** the index is IVF-SQ8, which lowers absolute numbers slightly relative to exact search.
