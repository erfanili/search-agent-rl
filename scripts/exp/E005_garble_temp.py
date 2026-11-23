#!/usr/bin/env python
"""E005 garble diagnosis, item 1: the start model in standalone vLLM on the 320 step-1 training rollouts (64 questions x 5)
at temperature 1.0, with the training rollout loop and retrieval.

  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E005_garble_temp.py --seed 0 [--temperature 1.0]

Questions: the 64 step-1 questions of the E005 chain (ids read from E005-C-s0's step-1 rollout dumps, first 40 lines per
  worker file; the same 64 in every E005 launch), each 5 times. Prompt: the training prompt (mix50sc train.parquet
  "prompt" messages, chat template via agent.search_r1_rollout.prompt_ids_for), as the verl agent loop builds it.
Rollout loop: the worktree's agent.search_r1_rollout.run_episode with the training RolloutConfig (max_turns 3, topk 3,
  max_obs_tokens 500, response_length 2048, invalid_action retry, depth_policy max:3); retrieval POST /retrieve on the
  same server, topk 3.
Sampling per turn: temperature (default 1.0), top_p 1.0, top_k -1, repetition_penalty 1.0, stop strings from the loop,
  max_tokens = 3072 - context length (verl server rule: max_model_len = prompt 1024 + response 2048), request seed =
  md5("<seed>|<id>|<k>|<turn>").
Engine (standalone settings, as E004_eval.py): vLLM AsyncLLMEngine, bfloat16, max_model_len 8192, gpu_memory_utilization
  0.85, prefix caching, engine seed = --seed; no sleep mode.
Garbled = regex  </\\s*\\.|<\\s+/|</ |< answer|</\\.  on any model segment (the E005 diagnosis regex).
Conditions (setting sweep, 2026-11-22): --no-prefix-cache (enable_prefix_caching=False), --eager (enforce_eager=True,
  no CUDA graphs / compile), --no-chunked-prefill (enable_chunked_prefill=False); --cond names the output directory.
  Throughput = episodes / generation wall time (engine start excluded); the engine's printed config line is saved.
Outputs: $DATA_ROOT/logs/E005/garble/temp<T>_s<seed>/{rollouts.jsonl, summary.json}, or garble/sweep/<cond>_s<seed>/
"""
import argparse, asyncio, glob, hashlib, json, os, re, sys, time

WT = "/home/erfan/sar-alias-reward"
sys.path.insert(0, WT)
import requests  # noqa: E402
from agent.paths import DATA  # noqa: E402
from agent.search_r1_rollout import RolloutConfig, prompt_ids_for, run_episode  # noqa: E402

G = re.compile(r"</\s*\.|<\s+/|</ |< answer|</\.")
URL = "http://127.0.0.1:8000"
MODEL = os.path.join(DATA, "hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
TRAIN_MAX_MODEL_LEN = 3072


def step1_questions():
    import pandas as pd
    d = pd.read_parquet(os.path.join(DATA, "verl_data/e005/mix50sc/train.parquet"))
    byq = {}
    for _, r in d.iterrows():
        e = r["extra_info"]
        byq[e["question"]] = (e["id"], e["e004_label"], [dict(m) for m in r["prompt"]])
    qs = []
    for f in sorted(glob.glob(os.path.join(DATA, "logs/E005/E005-C-s0/rollouts/*.jsonl"))):
        for l in list(open(f))[:40]:
            q = [x.strip() for x in re.findall(r"Question: ([^\n<]*)", json.loads(l)["prompt_text"])][-1]
            if q not in qs:
                qs.append(q)
    assert len(qs) == 64, len(qs)
    return [(q, *byq[q]) for q in qs]


async def main_async(a):
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    out = (os.path.join(DATA, "logs/E005/garble/sweep", f"{a.cond}_s{a.seed}") if a.cond
           else os.path.join(DATA, "logs/E005/garble", f"temp{a.temperature:g}_s{a.seed}"))
    os.makedirs(out, exist_ok=True)
    print("retrieval:", requests.get(f"{URL}/health", timeout=10).json(), flush=True)
    t_init = time.time()
    eargs = dict(model=MODEL, dtype="bfloat16", seed=a.seed, max_model_len=8192, gpu_memory_utilization=0.85,
                 enable_prefix_caching=not a.no_prefix_cache, enforce_eager=a.eager)
    if a.no_chunked_prefill:
        eargs["enable_chunked_prefill"] = False
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**eargs))
    t_init = time.time() - t_init
    tok = await engine.get_tokenizer()
    cfg = RolloutConfig(max_turns=3, topk=3, max_obs_tokens=500, response_length=2048, invalid_action="retry",
                        depth_policy="max:3")
    counter = [0]

    async def retrieve(query, topk):
        def post():
            r = requests.post(f"{URL}/retrieve", json={"queries": [query], "topk": topk, "return_scores": False}, timeout=120)
            r.raise_for_status()
            return r.json()["result"][0]
        return await asyncio.to_thread(post)

    async def episode(qid, k, pids):
        turn = [0]

        async def generate(context_ids, stop):
            seed = int(hashlib.md5(f"{a.seed}|{qid}|{k}|{turn[0]}".encode()).hexdigest()[:8], 16)
            turn[0] += 1
            counter[0] += 1
            sp = SamplingParams(temperature=a.temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0, seed=seed,
                                stop=list(stop), include_stop_str_in_output=True,
                                max_tokens=max(1, TRAIN_MAX_MODEL_LEN - len(context_ids)))
            final = None
            async for o in engine.generate({"prompt_token_ids": list(context_ids)}, sp, request_id=f"r{counter[0]}"):
                final = o
            return list(final.outputs[0].token_ids)

        return await run_episode(pids, generate, retrieve, tok, cfg)

    Q = step1_questions()
    t0 = time.time()
    jobs = []
    for q, qid, lab, msgs in Q:
        pids = prompt_ids_for(tok, msgs, True)
        for k in range(5):
            jobs.append((qid, lab, k, episode(qid, k, pids)))
    eps = await asyncio.gather(*[j[3] for j in jobs])
    n_g = n_g1 = n_inv = 0
    by = {"knowable": [0, 0], "unknowable": [0, 0]}
    with open(f"{out}/rollouts.jsonl", "w") as f:
        for (qid, lab, k, _), ep in zip(jobs, eps):
            segs = [{"role": s.role, "kind": s.kind, "text": tok.decode(s.ids, skip_special_tokens=False)} for s in ep.segments]
            ms = [s["text"] for s in segs if s["role"] == "model"]
            g = any(G.search(t) for t in ms)
            g1 = bool(ms) and bool(G.search(ms[0]))
            n_g += g; n_g1 += g1; n_inv += ep.n_invalid > 0
            by[lab][0] += 1; by[lab][1] += g
            f.write(json.dumps({"id": qid, "label": lab, "k": k, "finish": ep.finish, "n_search": ep.n_search,
                                "n_invalid": ep.n_invalid, "garbled": g, "garbled_turn1": g1, "segments": segs}) + "\n")
    n = len(eps)
    summ = {"temperature": a.temperature, "seed": a.seed, "n": n, "garbled": n_g / n, "garbled_turn1": n_g1 / n,
            "any_invalid": n_inv / n, "garbled_by_label": {k: v[1] / v[0] for k, v in by.items()},
            "n_requests": counter[0], "wall_s": round(time.time() - t0, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cond": a.cond, "engine_args": eargs, "engine_init_s": round(t_init, 1),
            "episodes_per_s": round(n / (time.time() - t0), 3),
            "vllm_config": {k: getattr(engine.vllm_config.cache_config, "enable_prefix_caching", None) if k == "prefix_caching"
                            else getattr(engine.vllm_config.scheduler_config, "chunked_prefill_enabled", None) if k == "chunked_prefill"
                            else engine.vllm_config.model_config.enforce_eager for k in ("prefix_caching", "chunked_prefill", "enforce_eager")}}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=2)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cond", default=None)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--no-chunked-prefill", action="store_true")
    asyncio.run(main_async(ap.parse_args()))
