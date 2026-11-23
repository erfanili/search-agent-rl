#!/usr/bin/env python
"""E002: search rollouts on the E001 easy set for one model x one condition.

  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E002_run.py --model qwen3b-base --cond P2 [--depth free] [--pilot]

Rollout: agent.search_r1_rollout.run_episode, unchanged, driven by vLLM's AsyncLLMEngine and the HTTP retrieval
server (localhost:8000, top-3). RolloutConfig(max_turns=3, topk=3, max_obs_tokens=500, response_length=4096,
invalid_action="retry", depth_policy=...):
  --depth free  -> depth_policy "max:3": the policy chooses; after 3 searches the environment writes <answer>
  --depth f0/f1 -> "exact:0" / "exact:1" (card: FORCE_DEPTH=0/1; P2 only)
Per turn: temperature 0.7, top_p 1.0, max_tokens = min(512, max_model_len - context), stop strings from the rollout
(</search>, </answer>; forced turns: opening tags / the forced closing tag), included in the output. A fresh vLLM
request id per turn (see agent/verl_agent_loop.py). 5 episodes per question, request seed = hash of
(question id, sample, turn) for repeatability of the request stream.

Scoring (E001 rule): the response text with environment text removed (qa_metrics.strip_environment_text; this
keeps environment-forced <answer> tags, since they open the answer), then the FIRST <answer>...</answer>; an
<answer> opened but never closed -> text to the end; no <answer> -> untagged. correct = cover_match, em = exact_match.

Malformed search calls, per free (unforced) model turn, from the raw generated text of that turn:
  empty_query      a complete <search></search> with an empty/whitespace query
  unclosed_search  <search> with no </search> in the turn
  multi_search     more than one <search> in the turn
  search_and_answer  both <search> and <answer> in the turn

Output: $DATA_ROOT/logs/E002[/pilot]/<model>/<cond>/{samples.jsonl, run.json}; cond = P1..P6, or P2-f0 / P2-f1.
"""
import argparse, asyncio, hashlib, json, os, re, subprocess, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import requests  # noqa: E402
from agent.paths import DATA, hf_snapshot  # noqa: E402
from agent.qa_metrics import cover_match, exact_match, strip_environment_text  # noqa: E402
from agent.search_r1_rollout import RolloutConfig, run_episode, parse_depth_policy  # noqa: E402
from E001_prompts import MODELS  # noqa: E402  (same tags and checkpoints as E001)
from E001_run import pilot_rows  # noqa: E402  (25 PopQA + 25 TriviaQA, as E001 pilot 2)
from E002_prompts import CONDITIONS, make_prompt  # noqa: E402

SET = os.path.join(DATA, "eval", "e001", "e001_easy_1k.jsonl")
DEPTH = {"free": "max:3", "f0": "exact:0", "f1": "exact:1"}
URL = "http://127.0.0.1:8000"
N = 5
MAX_MODEL_LEN = 8192


def parse(text):
    i = text.find("<answer>")
    if i < 0:
        return None, "untagged"
    rest = text[i + len("<answer>"):]
    j = rest.find("</answer>")
    return (rest[:j].strip(), "closed") if j >= 0 else (rest.strip(), "open")


def malformed(raw):
    n_open = raw.count("<search>")
    m = re.search(r"<search>(.*?)</search>", raw, re.S)
    return {"empty_query": bool(m and not m.group(1).strip()),
            "unclosed_search": n_open > 0 and "</search>" not in raw,
            "multi_search": n_open > 1,
            "search_and_answer": n_open > 0 and "<answer>" in raw}


class BatchRetriever:
    """Collects concurrent queries for a few ms and sends them in one POST (the server batches the e5 encode)."""

    def __init__(self, topk=3, window=0.02, max_batch=256):
        self.topk, self.window, self.max_batch = topk, window, max_batch
        self.pending, self.task, self.n_calls, self.n_queries = [], None, 0, 0

    async def __call__(self, query, topk):
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((query, fut))
        if self.task is None:
            self.task = asyncio.create_task(self._flush())
        return await fut

    async def _flush(self):
        await asyncio.sleep(self.window)
        batch, self.pending, self.task = self.pending[: self.max_batch], self.pending[self.max_batch:], None
        if self.pending:
            self.task = asyncio.create_task(self._flush())
        qs = [q for q, _ in batch]
        for attempt in range(5):
            try:
                r = await asyncio.to_thread(requests.post, f"{URL}/retrieve",
                                            json={"queries": qs, "topk": self.topk, "return_scores": False}, timeout=600)
                r.raise_for_status()
                res = r.json()["result"]
                break
            except requests.RequestException as e:
                err = e
                await asyncio.sleep(2 ** attempt)
        else:
            for _, f in batch:
                f.set_exception(RuntimeError(f"retrieval failed: {err}"))
            return
        self.n_calls += 1
        self.n_queries += len(qs)
        for (_, f), hits in zip(batch, res):
            f.set_result(hits)


async def main_async(a):
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    rows = [json.loads(l) for l in open(SET)]
    if a.pilot:
        rows = pilot_rows(rows)
    cond = a.cond if a.depth == "free" else f"{a.cond}-{a.depth}"
    out = os.path.join(DATA, "logs", "E002", "pilot" if a.pilot else "", a.model, cond)
    os.makedirs(out, exist_ok=True)
    policy = DEPTH[a.depth]
    parse_depth_policy(policy)  # max:3 needs the feat/max-depth rollout change; fails loudly otherwise
    cfg = RolloutConfig(max_turns=3, topk=3, max_obs_tokens=500, response_length=4096, invalid_action="retry",
                        depth_policy=policy)

    src, kind = MODELS[a.model]
    path = src if os.path.isdir(src) else hf_snapshot("models", src)
    assert path and os.path.isdir(path), src
    print("retrieval server:", requests.get(f"{URL}/health", timeout=10).json(), flush=True)
    t0 = time.time()
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=path, dtype="bfloat16", seed=0, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.85,
        enable_prefix_caching=True))
    tok = await engine.get_tokenizer()
    retriever = BatchRetriever(topk=3)
    sem = asyncio.Semaphore(a.concurrency)
    counter = [0]

    async def episode(r, k, prompt_ids):
        raw_turns = []

        async def generate(context_ids, stop):
            counter[0] += 1
            turn = len(raw_turns)
            seed = int(hashlib.md5(f"{r['id']}|{k}|{turn}".encode()).hexdigest()[:8], 16)
            sp = SamplingParams(temperature=0.7, top_p=1.0, seed=seed, stop=list(stop), include_stop_str_in_output=True,
                                max_tokens=max(1, min(512, MAX_MODEL_LEN - len(context_ids))))
            final = None
            async for o in engine.generate({"prompt_token_ids": list(context_ids)}, sp, request_id=f"req-{counter[0]}"):
                final = o
            c = final.outputs[0]
            raw_turns.append({"text": c.text, "stop": list(stop), "finish_reason": c.finish_reason,
                              "free": list(stop) == ["</search>", "</answer>"]})
            return list(c.token_ids)

        async with sem:
            ep = await run_episode(prompt_ids, generate, retriever, tok, cfg)
        text = tok.decode(ep.response_ids, skip_special_tokens=True)
        ans, status = parse(strip_environment_text(text))
        mal = {key: sum(malformed(t["text"])[key] for t in raw_turns if t["free"]) for key in
               ("empty_query", "unclosed_search", "multi_search", "search_and_answer")}
        return {"text": text, "finish": ep.finish, "n_search": ep.n_search, "queries": ep.queries,
                "n_invalid": ep.n_invalid, "n_forced": ep.n_forced, "n_overridden": ep.n_overridden,
                "n_response_tokens": len(ep.response_ids), "cap_hit": ep.finish == "length",
                "n_free_turns": sum(t["free"] for t in raw_turns), "malformed": mal, "raw_turns": raw_turns,
                "answer": ans, "parse": status,
                "correct": int(ans is not None and cover_match(ans, r["answers"]) == 1.0),
                "em": int(ans is not None and exact_match(ans, r["answers"]) == 1.0)}

    async def question(r):
        prompt = make_prompt(a.cond, kind, tok, r["question"])
        ids = tok.encode(prompt, add_special_tokens=False)
        samples = await asyncio.gather(*[episode(r, k, ids) for k in range(N)])
        return {"id": r["id"], "source": r["source"], "bin": r["bin"], "question": r["question"],
                "answers": r["answers"], "prompt_tokens": len(ids), "samples": samples}

    results = await asyncio.gather(*[question(r) for r in rows])
    with open(f"{out}/samples.jsonl", "w") as f:
        for q in results:
            f.write(json.dumps(q) + "\n")
    git = lambda *c: subprocess.run(["git", "-C", ROOT, *c], capture_output=True, text=True).stdout.strip()
    meta = {"model_tag": a.model, "model_path": path, "prompt_type": kind, "condition": a.cond,
            "instruction_demos": CONDITIONS[a.cond], "depth": a.depth, "depth_policy": policy, "cfg": cfg.__dict__,
            "n_questions": len(rows), "n_samples": N, "pilot": a.pilot, "set": SET,
            "example_prompt": make_prompt(a.cond, kind, tok, rows[0]["question"]),
            "retrieval_calls": retriever.n_calls, "retrieval_queries": retriever.n_queries,
            "wall_s": round(time.time() - t0, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty_files": len(git("status", "--porcelain").splitlines())}
    json.dump(meta, open(f"{out}/run.json", "w"), indent=2)
    print(json.dumps({k: meta[k] for k in ("model_tag", "condition", "depth", "n_questions", "wall_s", "retrieval_queries")}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--cond", required=True, choices=list(CONDITIONS))
    p.add_argument("--depth", default="free", choices=list(DEPTH))
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--concurrency", type=int, default=1024, help="episodes in flight")
    a = p.parse_args()
    if a.depth != "free" and a.cond != "P2":
        p.error("forced depth is run for P2 only (card)")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
