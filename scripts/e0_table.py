#!/usr/bin/env python
"""
E0 table (notes/new_plan.md): depth-value curves V(k) = EM with exactly k searches forced, per model and set.

Reads $DATA/logs/eval_<model tag>_exact<k>_<stamp>/scored/per_example.jsonl (scripts/run_e0.sh); for each
(model, k) the newest run counts. Also prints, per model and set, V(k+1) - V(k) as a PAIRED difference on
the same questions with its standard error, which is much tighter than comparing two EMs with ~2.6 EM
error each. "searches" are the executed searches from preds.jsonl and must equal k exactly; anything else
means the forcing broke.

  .venv/bin/python scripts/e0_table.py                    # all models found
  .venv/bin/python scripts/e0_table.py --metric f1 --split sub300
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent.paths import DATA  # noqa: E402

RUN_RE = re.compile(r"^eval_(?P<model>.+)_exact(?P<k>\d+)_(?P<stamp>\d{8}_\d{6})$")
ORDER = ["hotpotqa", "2wiki", "musique", "bamboogle"]


def load(split):
    runs = {}
    for d in glob.glob(os.path.join(DATA, "logs", "eval_*_exact*")):
        m = RUN_RE.match(os.path.basename(d))
        path = os.path.join(d, "scored", "per_example.jsonl")
        meta = os.path.join(d, "scored", "metrics.json")
        if not m or not os.path.exists(path):
            continue
        if split and os.path.exists(meta) and json.load(open(meta)).get("split") != split:
            continue
        key = (m["model"], int(m["k"]))
        if key not in runs or m["stamp"] > runs[key][0]:
            runs[key] = (m["stamp"], path)
    data = {}
    for key, (_, path) in runs.items():
        rows = {(r["dataset"], r["id"]): r for r in map(json.loads, open(path))}
        preds = os.path.join(os.path.dirname(os.path.dirname(path)), "preds.jsonl")
        if os.path.exists(preds):  # executed searches; the scorer counts <search> text, which can include junk
            for p in map(json.loads, open(preds)):
                if (p["dataset"], p["id"]) in rows and "n_search" in p:
                    rows[(p["dataset"], p["id"])]["n_search"] = p["n_search"]
        data[key] = rows
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", default="em", choices=["em", "f1", "acc"])
    p.add_argument("--split", default="sub300")
    a = p.parse_args()

    data = load(a.split)
    if not data:
        sys.exit(f"no eval_*_exact* runs with scores for split {a.split} in {DATA}/logs")
    models = sorted({m for m, _ in data})
    ks = sorted({k for _, k in data})
    for model in models:
        print(f"\n{model}   ({a.metric.upper()} x 100; searches/q in brackets)")
        print(f"{'set':<10} {'n':>4} " + " ".join(f"{'V(' + str(k) + ')':>13}" for k in ks)
              + "   " + " ".join(f"{'V(' + str(k + 1) + ')-V(' + str(k) + ')':>16}" for k in ks[:-1]))
        sets = sorted({ds for k in ks if (model, k) in data for ds, _ in data[(model, k)]},
                      key=lambda s: (ORDER.index(s) if s in ORDER else 99, s))
        for ds in sets:
            cells, n = [], None
            per_k = {}
            for k in ks:
                rows = {i: r for (d, i), r in data.get((model, k), {}).items() if d == ds}
                per_k[k] = rows
                if rows:
                    n = len(rows)
                    v = 100 * sum(r[a.metric] for r in rows.values()) / len(rows)
                    srch = sum(r["n_search"] for r in rows.values()) / len(rows)
                    flag = "" if abs(srch - k) < 1e-9 else "!"
                    cells.append(f"{v:6.1f} [{srch:.2f}{flag}]")
                else:
                    cells.append(f"{'-':>13}")
            diffs = []
            for k in ks[:-1]:
                common = set(per_k[k]) & set(per_k.get(k + 1, {}))
                if len(common) < 2:
                    diffs.append(f"{'-':>16}")
                    continue
                d = [per_k[k + 1][i][a.metric] - per_k[k][i][a.metric] for i in common]
                mu = sum(d) / len(d)
                se = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d))
                diffs.append(f"{100 * mu:+7.1f} ± {100 * se:4.1f}{'*' if abs(mu) > 2 * se else ' '}")
            print(f"{ds:<10} {n or 0:>4} " + " ".join(cells) + "   " + " ".join(diffs))
    print("\n[x.xx!] = searches/q differs from k (forcing failed somewhere). * = paired difference beyond 2 SE.")


if __name__ == "__main__":
    main()
