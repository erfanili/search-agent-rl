#!/usr/bin/env python
"""Check relabel outputs: every question of the E001 set has exactly 5 closed-book samples, and labels.jsonl covers the
same ids with n_correct = number of correct samples.

  python scripts/exp/E004_check_relabel.py <relabel dir> [<relabel dir> ...]     # exit 1 if any check fails
"""
import collections, json, sys


def check(d, n_questions=1000, n=5):
    S = [json.loads(l) for l in open(f"{d}/samples.jsonl")]
    L = {r["id"]: r for r in map(json.loads, open(f"{d}/labels.jsonl"))}
    hist = dict(sorted(collections.Counter(len(r["samples"]) for r in S).items()))
    ids = [r["id"] for r in S]
    ok = (len(S) == n_questions and len(set(ids)) == n_questions and set(ids) == set(L) and hist == {n: n_questions}
          and all(L[r["id"]]["n_correct"] == sum(s["correct"] for s in r["samples"]) for r in S))
    return {"dir": d, "ok": ok, "n_questions": len(S), "count_hist": hist}


if __name__ == "__main__":
    res = [check(d) for d in sys.argv[1:]]
    for r in res:
        print(json.dumps(r))
    sys.exit(0 if all(r["ok"] for r in res) else 1)
