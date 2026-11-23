#!/usr/bin/env python
"""E004 evaluation of one merged checkpoint: E002 rollouts under P1/P2/P6, S5 greedy P2 first turn, optional relabel.

  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E004_eval.py --run E004-A-s0 --step 20 [--relabel]

Model: $DATA_ROOT/logs/E004/<run>/hf/step_<k> (must have LOAD_OK.json), chat template (qwen3b-instruct).
Rollouts: exactly E002_run's episode settings (agent.search_r1_rollout.run_episode, RolloutConfig(max_turns=3, topk=3,
  max_obs_tokens=500, response_length=4096, invalid_action="retry", depth_policy="max:3"), per turn temperature 0.7,
  top_p 1.0, max_tokens = min(512, 8192 - context), md5 seed per (id, sample, turn), 5 episodes per question, E002
  parse and malformed definitions, BatchRetriever) on all 1,000 questions of the E001 set; one AsyncLLMEngine
  (max_model_len 8192, gpu mem 0.85, prefix caching, seed 0) serves P1, P2, P6 in turn.
S5 generation: greedy P2 first turn (temperature 0, max 512, stop </search> / </answer>), as E003_gen.py.
--relabel: E001 rev-2 closed-book procedure (E001 prompt in the chat template, n=5, temperature 0.7, top_p 1.0,
  max 512, stop </answer>, request seed 0; E001 parse, cover match, knowable >= 3 / unknowable 0 / uncertain 1-2;
  clarification flag from E001 CLARIFY_RE). Served by the same async engine instead of vLLM's offline LLM (E001):
  same sampling settings.
  Fix 2026-11-22: the relabel used to send one n=5 request and keep only the last streamed RequestOutput, which in
  vLLM 0.8.5's async engine held 1-5 of the 5 completions. It now sends 5 separate n=1 requests per question (seed =
  md5 of "<id>|relabel|<k>", k = 0..4) and asserts exactly 5 samples per question before writing
  (collect_closed_samples; test: scripts/exp/test_E004_relabel.py; output check: E004_check_relabel.py).
--no-prefix-cache: engine with enable_prefix_caching=False (E006; vLLM 0.8.5 prefix caching garbled some rollouts).
--only-relabel: skip the P1/P2/P6 rollouts and the S5 turn. --model-path: evaluate this model instead of
  hf/step_<k> (start model check: --model-path <Qwen2.5-3B-Instruct snapshot> --run start --step 0).
Outputs: $DATA_ROOT/logs/E004/eval/<run>/step<k>/{P1,P2,P6}/{samples.jsonl, run.json}, .../step<k>/s5_turn1.jsonl,
  .../step<k>/relabel/{samples.jsonl, labels.jsonl}
"""
import argparse, asyncio, hashlib, json, os, subprocess, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import requests  # noqa: E402
from agent.paths import DATA  # noqa: E402
from agent.qa_metrics import cover_match, exact_match, strip_environment_text  # noqa: E402
from agent.search_r1_rollout import RolloutConfig, run_episode  # noqa: E402
from E001_prompts import make_prompt as e001_prompt  # noqa: E402
from E001_run import label as e001_label, parse as e001_parse  # noqa: E402
from E001_tables import uclass  # noqa: E402
from E002_prompts import CONDITIONS, make_prompt  # noqa: E402
from E002_run import SET, URL, N, MAX_MODEL_LEN, BatchRetriever, malformed, parse  # noqa: E402

PROMPTS = ["P2", "P1", "P6"]


N_RELABEL = 5


def relabel_seed(qid, k):
    return int(hashlib.md5(f"{qid}|relabel|{k}".encode()).hexdigest()[:8], 16)


async def collect_closed_samples(gen, prompt_ids, qid, make_sp, n=N_RELABEL):
    """n separate n=1 requests (one per sample, seed relabel_seed(qid, k)); returns the n completion outputs."""
    outs = await asyncio.gather(*[gen(prompt_ids, make_sp(relabel_seed(qid, k))) for k in range(n)])
    cs = [o.outputs[0] for o in outs]
    assert len(cs) == n, (qid, len(cs))
    return cs


async def main_async(a):
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    rows = [json.loads(l) for l in open(SET)]
    if a.model_path:
        path = a.model_path
    else:
        path = os.path.join(DATA, "logs", a.exp, a.run, "hf", f"step_{a.step}")
        assert os.path.exists(os.path.join(path, "LOAD_OK.json")), path
    base = os.path.join(DATA, "logs", a.exp, "eval", a.run, f"step{a.step}")
    os.makedirs(base, exist_ok=True)
    print("retrieval server:", requests.get(f"{URL}/health", timeout=10).json(), flush=True)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=path, dtype="bfloat16", seed=0, max_model_len=MAX_MODEL_LEN,
                                                             gpu_memory_utilization=0.85, enable_prefix_caching=not a.no_prefix_cache))
    tok = await engine.get_tokenizer()
    counter = [0]
    git = lambda *c: subprocess.run(["git", "-C", ROOT, *c], capture_output=True, text=True).stdout.strip()

    async def gen(ids, sp):
        counter[0] += 1
        final = None
        async for o in engine.generate({"prompt_token_ids": list(ids)}, sp, request_id=f"req-{counter[0]}"):
            final = o
        return final

    for cond in ([] if a.only_relabel else PROMPTS):
        out = os.path.join(base, cond)
        if os.path.exists(f"{out}/run.json"):
            continue
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        cfg = RolloutConfig(max_turns=3, topk=3, max_obs_tokens=500, response_length=4096, invalid_action="retry",
                            depth_policy="max:3")
        retriever = BatchRetriever(topk=3)
        sem = asyncio.Semaphore(a.concurrency)

        async def episode(r, k, prompt_ids):
            raw_turns = []

            async def generate(context_ids, stop):
                turn = len(raw_turns)
                seed = int(hashlib.md5(f"{r['id']}|{k}|{turn}".encode()).hexdigest()[:8], 16)
                sp = SamplingParams(temperature=0.7, top_p=1.0, seed=seed, stop=list(stop), include_stop_str_in_output=True,
                                    max_tokens=max(1, min(512, MAX_MODEL_LEN - len(context_ids))))
                c = (await gen(context_ids, sp)).outputs[0]
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
            ids = tok.encode(make_prompt(cond, "chat", tok, r["question"]), add_special_tokens=False)
            samples = await asyncio.gather(*[episode(r, k, ids) for k in range(N)])
            return {"id": r["id"], "source": r["source"], "bin": r["bin"], "question": r["question"],
                    "answers": r["answers"], "prompt_tokens": len(ids), "samples": samples}

        results = await asyncio.gather(*[question(r) for r in rows])
        with open(f"{out}/samples.jsonl", "w") as f:
            for q in results:
                f.write(json.dumps(q) + "\n")
        meta = {"run": a.run, "step": a.step, "model_path": path, "prefix_caching": not a.no_prefix_cache, "condition": cond, "instruction_demos": CONDITIONS[cond],
                "depth_policy": "max:3", "cfg": cfg.__dict__, "n_questions": len(rows), "n_samples": N, "set": SET,
                "retrieval_queries": retriever.n_queries, "wall_s": round(time.time() - t0, 1),
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "git_commit": git("rev-parse", "--short", "HEAD"),
                "git_dirty_files": len(git("status", "--porcelain").splitlines())}
        json.dump(meta, open(f"{out}/run.json", "w"), indent=2)
        print(json.dumps({k: meta[k] for k in ("run", "step", "condition", "wall_s")}), flush=True)

    # S5 generation: greedy P2 first turn
    if not a.only_relabel and not os.path.exists(f"{base}/s5_turn1.jsonl"):
        sp = SamplingParams(temperature=0.0, max_tokens=512, stop=["</search>", "</answer>"], include_stop_str_in_output=True)

        async def turn1(r):
            pr = make_prompt("P2", "chat", tok, r["question"])
            c = (await gen(tok.encode(pr, add_special_tokens=False), sp)).outputs[0]
            return {"id": r["id"], "prompt": pr, "text": c.text, "token_ids": list(c.token_ids), "finish_reason": c.finish_reason}

        res = await asyncio.gather(*[turn1(r) for r in rows])
        with open(f"{base}/s5_turn1.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x) + "\n")

    if (a.relabel or a.only_relabel) and not os.path.exists(f"{base}/relabel/labels.jsonl"):
        os.makedirs(f"{base}/relabel", exist_ok=True)
        make_sp = lambda seed: SamplingParams(n=1, temperature=0.7, top_p=1.0, max_tokens=512, seed=seed,
                                              stop=["</answer>"], include_stop_str_in_output=True)

        async def closed(r):
            cs = await collect_closed_samples(gen, tok.encode(e001_prompt("chat", tok, r["question"]), add_special_tokens=False),
                                              r["id"], make_sp)
            S = []
            for c in cs:
                ans, st = e001_parse(c.text)
                s = {"text": c.text, "answer": ans, "parse": st,
                     "correct": int(ans is not None and cover_match(ans, r["answers"]) == 1.0)}
                s["untagged_class"] = uclass(s)
                S.append(s)
            return r, S

        res = await asyncio.gather(*[closed(r) for r in rows])
        assert len(res) == len(rows) and all(len(S) == N_RELABEL for _, S in res), "relabel: not 5 samples per question"
        with open(f"{base}/relabel/samples.jsonl", "w") as fs, open(f"{base}/relabel/labels.jsonl", "w") as fl:
            for r, S in res:
                n = sum(s["correct"] for s in S)
                fs.write(json.dumps({"id": r["id"], "samples": S}) + "\n")
                fl.write(json.dumps({"id": r["id"], "n_correct": n, "label": e001_label(n),
                                     "clarification": any(s["untagged_class"] == "clarification" for s in S)}) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--exp", default="E004")  # E005 (2026-11-22): same evaluation, runs under $DATA_ROOT/logs/<exp>/
    p.add_argument("--relabel", action="store_true")
    p.add_argument("--only-relabel", action="store_true")
    p.add_argument("--model-path", default=None)
    p.add_argument("--no-prefix-cache", action="store_true")  # E006 (2026-11-22): evaluation with prefix caching off
    p.add_argument("--concurrency", type=int, default=1024)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
