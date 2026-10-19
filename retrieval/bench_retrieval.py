#!/usr/bin/env python
"""
Benchmark and sanity-check a running retrieval server. Step P0.6.

Reports, on N HotpotQA dev questions:
  latency    p50 / p95 / p99 of single-query requests, sent one at a time
  throughput queries/s with --concurrency parallel clients (rollouts hit the server like this)
  hit@k      share of questions where at least one / both gold supporting-fact titles appear
             in the top-k titles (a cheap retrieval-quality number, no labels needed)
  spot check top-k titles for --spot questions, printed for reading by eye
  recall@10  (optional, --emb-dir) approximate index vs exact brute-force search over the raw
             shards, on --exact-n queries: the cost of IVF-SQ8 vs a flat index

Example: scripts/bench_retrieval.sh
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests


def load_hotpot(n, seed):
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    rows = [ds[int(i)] for i in idx]
    return [r["question"] for r in rows], [sorted(set(r["supporting_facts"]["title"])) for r in rows]


def title_of(contents):
    return contents.split("\n", 1)[0].strip().strip('"')


def norm(t):
    return " ".join(t.lower().split())


def call(url, queries, topk):
    r = requests.post(f"{url}/retrieve", json={"queries": queries, "topk": topk, "return_scores": True}, timeout=60)
    r.raise_for_status()
    return r.json()["result"]


def pct(xs, p):
    return float(np.percentile(xs, p))


def exact_recall(url, questions, emb_dir, model, k):
    """Brute-force top-k over the float16 shards; compare ids with the server's answers."""
    from encode_corpus import check_gpus, shard_paths
    from retrieval_server import E5QueryEncoder

    with open(os.path.join(emb_dir, "meta.json")) as f:
        meta = json.load(f)
    n_shards = (meta["n_rows"] + meta["shard_size"] - 1) // meta["shard_size"]
    device = "cuda" if _has_cuda() else "cpu"
    if device == "cuda":
        check_gpus()
    enc = E5QueryEncoder(model, device)
    q = enc.encode(questions)  # [m, d] float32
    best_s = np.full((len(q), k), -np.inf, dtype=np.float32)
    best_id = np.zeros((len(q), k), dtype=np.int64)
    for s in range(n_shards):
        ids_path, emb_path = shard_paths(emb_dir, s)
        emb = np.load(emb_path, mmap_mode="r").astype(np.float32)
        ids = np.load(ids_path)
        sc = q @ emb.T
        top = np.argpartition(-sc, k, axis=1)[:, :k]
        cand_s = np.concatenate([best_s, np.take_along_axis(sc, top, 1)], 1)
        cand_id = np.concatenate([best_id, ids[top]], 1)
        order = np.argsort(-cand_s, axis=1)[:, :k]
        best_s, best_id = np.take_along_axis(cand_s, order, 1), np.take_along_axis(cand_id, order, 1)
    approx = call(url, questions, k)
    recalls = [len({int(h["document"]["id"]) for h in hits} & set(ex.tolist())) / k for hits, ex in zip(approx, best_id)]
    return float(np.mean(recalls))


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--spot", type=int, default=10)
    p.add_argument("--emb-dir", help="enable exact-search recall check")
    p.add_argument("--exact-n", type=int, default=200)
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the summary as JSON here")
    a = p.parse_args()

    health = requests.get(f"{a.url}/health", timeout=10).json()
    print("server:", health)
    questions, gold = load_hotpot(a.n, a.seed)

    call(a.url, questions[:8], a.topk)  # warm-up (CUDA kernels, page cache)

    lat, results = [], []
    for qtext in questions:
        t0 = time.perf_counter()
        results.append(call(a.url, [qtext], a.topk)[0])
        lat.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(lambda qtext: call(a.url, [qtext], a.topk), questions))
    qps = len(questions) / (time.perf_counter() - t0)

    any_hit = all_hit = 0
    for hits, g in zip(results, gold):
        got = {norm(title_of(h["document"]["contents"])) for h in hits}
        found = [norm(t) in got for t in g]
        any_hit += any(found)
        all_hit += all(found)

    summary = {
        "n": len(questions),
        "topk": a.topk,
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99)},
        f"qps_at_concurrency_{a.concurrency}": qps,
        f"hotpot_any_gold_title_at_{a.topk}": any_hit / len(questions),
        f"hotpot_all_gold_titles_at_{a.topk}": all_hit / len(questions),
        "server": health,
    }
    if a.emb_dir:
        summary["recall_at_10_vs_exact"] = exact_recall(a.url, questions[: a.exact_n], a.emb_dir, a.model, 10)
        summary["exact_n"] = min(a.exact_n, len(questions))

    print(json.dumps({k: v for k, v in summary.items() if k != "server"}, indent=2))

    print(f"\n--- spot check: top-{a.topk} titles for {a.spot} questions ---")
    for qtext, g, hits in list(zip(questions, gold, results))[: a.spot]:
        print(f"\nQ: {qtext}\n   gold: {g}")
        for h in hits:
            print(f"   {h['score']:.3f}  {title_of(h['document']['contents'])}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()