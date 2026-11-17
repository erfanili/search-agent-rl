#!/usr/bin/env python
"""
Label the GRPO training questions by what the policy can do before training (new_plan.md E2, repo item 4).

  closed_book_em  1 if the model answers correctly with zero searches. Read from predictions of
                  baselines/searchr1_infer.py --depth-policy exact:0 --questions-parquet train.parquet
                  (the V(0) measurement of E0, on the training questions); scripts/label_train_data.sh runs it.
  d1_hit          1 if a gold answer appears (whole words, normalized) in the top-k passages retrieved for
                  the question itself used as the query: one obvious search finds the evidence. Same test
                  as the evidence_hit training metric (agent/qa_metrics.answer_in_information).

Writes <out>/labels.parquet: id, data_source, and every label computed so far. Each stage can run on its
own; a later run merges into the existing file. Retrieval results are cached in <out>/d1_cache.jsonl, so an
interrupted d1 run resumes where it stopped.

Then build a filtered training set with agent/make_train_data.py --labels <out>/labels.parquet --filter "...".

Runs in .venv (CPU); d1 needs the retrieval server.
  .venv/bin/python agent/label_train_data.py --train $DATA_ROOT/verl_data/nq_hotpotqa/train.parquet --d1
  .venv/bin/python agent/label_train_data.py --train ... --closed-book-preds <eval dir>/preds.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qa_metrics import answer_in_text, score_output  # noqa: E402
from agent.reward import golds_from  # noqa: E402
from agent.search_r1_rollout import passages_to_string  # noqa: E402


def d1_labels(rows, url, topk, batch, cache_path, limit):
    import requests

    done = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r
    todo = [r for r in rows if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"d1: {len(done)} cached, {len(todo)} to retrieve (top-{topk})", flush=True)
    t0 = time.time()
    with open(cache_path, "a") as f:
        for s in range(0, len(todo), batch):
            chunk = todo[s : s + batch]
            resp = requests.post(f"{url}/retrieve", json={"queries": [r["question"] for r in chunk], "topk": topk,
                                                          "return_scores": False}, timeout=600)
            resp.raise_for_status()
            for r, hits in zip(chunk, resp.json()["result"]):
                rec = {"id": r["id"], "d1_hit": answer_in_text(passages_to_string(hits), r["golds"]),
                       "d1_doc_ids": [str(h.get("id", "")) for h in hits]}
                done[r["id"]] = rec
                f.write(json.dumps(rec) + "\n")
            f.flush()
            n = s + len(chunk)
            rate = n / max(time.time() - t0, 1e-9)
            print(f"d1: {n}/{len(todo)} ({rate:.0f} q/s, {(len(todo) - n) / rate / 60:.0f} min left)", flush=True)
    return {i: r["d1_hit"] for i, r in done.items()}


def closed_book_labels(rows, preds_path):
    golds = {r["id"]: r["golds"] for r in rows}
    out = {}
    with open(preds_path) as f:
        for line in f:
            p = json.loads(line)
            if p["id"] in golds:
                if p.get("n_search", 0):
                    sys.exit(f"{preds_path}: {p['id']} has {p['n_search']} searches; run with --depth-policy exact:0")
                out[p["id"]] = score_output(p["output"], golds[p["id"]])["em"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="verl train.parquet (agent/make_train_data.py)")
    p.add_argument("--out", default=None, help="directory for labels.parquet (default: next to --train)")
    p.add_argument("--d1", action="store_true", help="compute d1_hit with the retrieval server")
    p.add_argument("--closed-book-preds", default=None, help="preds.jsonl from searchr1_infer.py --depth-policy exact:0")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--topk", type=int, default=3, help="passages per search, as in training")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="d1: retrieve at most N new questions (timing)")
    a = p.parse_args()
    if not (a.d1 or a.closed_book_preds):
        p.error("nothing to do: pass --d1 and/or --closed-book-preds")

    import pandas as pd

    out_dir = a.out or os.path.dirname(os.path.abspath(a.train))
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(a.train, columns=["data_source", "reward_model", "extra_info"])
    rows = [{"id": e["id"], "data_source": str(ds), "question": e["question"], "golds": golds_from(rm["ground_truth"])}
            for ds, rm, e in zip(df.data_source, df.reward_model, df.extra_info)]
    assert len({r["id"] for r in rows}) == len(rows), "extra_info.id is not unique"

    path = os.path.join(out_dir, "labels.parquet")
    labels = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame({"id": [r["id"] for r in rows],
                                                                              "data_source": [r["data_source"] for r in rows]})
    labels = labels.set_index("id")
    if a.closed_book_preds:
        cb = closed_book_labels(rows, a.closed_book_preds)
        labels["closed_book_em"] = pd.Series(cb, dtype="float64").reindex(labels.index)
        print(f"closed_book_em: {len(cb)}/{len(rows)} questions labelled", flush=True)
    if a.d1:
        d1 = d1_labels(rows, a.url, a.topk, a.batch, os.path.join(out_dir, "d1_cache.jsonl"), a.limit)
        labels["d1_hit"] = pd.Series(d1, dtype="float64").reindex(labels.index)
        print(f"d1_hit: {len(d1)}/{len(rows)} questions labelled", flush=True)
    labels.reset_index().to_parquet(path)

    cols = [c for c in ("closed_book_em", "d1_hit") if c in labels]
    print(f"\nwrote {path}\nshare = 1 by source (labelled rows only):")
    print(labels.groupby("data_source")[cols].agg(["mean", "count"]).round(3).to_string())
    if len(cols) == 2:
        both = labels.dropna(subset=cols)
        print("\njoint counts (closed_book_em x d1_hit):")
        print(pd.crosstab([both.data_source, both.closed_book_em], both.d1_hit).to_string())


if __name__ == "__main__":
    main()
