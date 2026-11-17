#!/usr/bin/env python
"""
Why a forced extra search helps or hurts (E0, notes/new_plan.md).

Decoding is greedy, so the exact:k and exact:k+1 runs of a model share the first k searches and differ only
from the next turn on: the k run must answer, the k+1 run must search once more and then answer. Per model
and set, for the k+1-th search of the exact:k+1 run:
  repeat    the query equals an earlier query (after normalization)
  overlap   mean share of the query's words already used in earlier queries
  new_docs  mean share of its top-3 passages (by title) not retrieved before
  new_gold  share of questions where a gold answer appears in this search's passages and in none before
  gain/loss questions right only with the extra search / right only without it (EM)
and gain/loss split by repeat vs new query, and by new_gold.

  .venv/bin/python scripts/e0_second_search.py                 # k=1 -> 2 and k=2 -> 3, all models
  .venv/bin/python scripts/e0_second_search.py --k 2 --examples 3 --model step_60 --dataset 2wiki
  .venv/bin/python scripts/e0_second_search.py --csv out.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent.qa_metrics import normalize_answer  # noqa: E402
from scripts.e0_table import ORDER, load  # noqa: E402

TURN_RE = re.compile(r"<search>(.*?)</search>\s*<information>(.*?)</information>", re.DOTALL)
TITLE_RE = re.compile(r'Doc \d+\(Title: "?(.*?)"?\)')
SHORT = {"Qwen2.5-3B": "Base", "SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo": "Search-R1",
         "grpo_qwen2.5-3b_full_20261019_183501_global_step_60": "Ours step 60"}


def turns(output):
    """[(normalized query, set of titles, normalized passage text)] for each executed search."""
    return [(normalize_answer(q), set(TITLE_RE.findall(info)), normalize_answer(info))
            for q, info in TURN_RE.findall(output)]


def preds_for(model, k, split):
    from agent.paths import DATA
    import glob
    best = None
    for d in glob.glob(os.path.join(DATA, "logs", f"eval_{model}_exact{k}_*")):
        meta = os.path.join(d, "scored", "metrics.json")
        if os.path.exists(meta) and json.load(open(meta)).get("split") != split:
            continue
        if os.path.exists(os.path.join(d, "preds.jsonl")) and (best is None or d > best):
            best = d
    return {(r["dataset"], r["id"]): r for r in map(json.loads, open(os.path.join(best, "preds.jsonl")))} if best else {}


def analyse(data, model, k, split):
    """Rows per dataset for the step k -> k+1."""
    if (model, k) not in data or (model, k + 1) not in data:
        return [], []
    before, after = data[(model, k)], data[(model, k + 1)]
    preds = preds_for(model, k + 1, split)
    acc = defaultdict(lambda: defaultdict(float))
    examples = []
    for key in before.keys() & after.keys() & preds.keys():
        ds = key[0]
        ts = turns(preds[key]["output"])
        if len(ts) < k + 1:
            continue
        q, titles, text = ts[k]
        earlier = ts[:k]
        prev_words = {w for eq, _, _ in earlier for w in eq.split()}
        prev_titles = set().union(*(t for _, t, _ in earlier)) if earlier else set()
        golds = [normalize_answer(g) for g in after[key]["golden_answers"]]
        golds = [g for g in golds if g]
        in_now = any(g in text for g in golds)
        in_before = any(g in et for g in golds for _, _, et in earlier)
        repeat = any(q == eq for eq, _, _ in earlier)
        new_gold = in_now and not in_before
        delta = after[key]["em"] - before[key]["em"]
        a = acc[ds]
        a["n"] += 1
        a["repeat"] += repeat
        a["overlap"] += len(set(q.split()) & prev_words) / max(1, len(set(q.split())))
        a["new_docs"] += len(titles - prev_titles) / max(1, len(titles))
        a["new_gold"] += new_gold
        a["gold_before"] += in_before
        a["gain"] += delta > 0
        a["loss"] += delta < 0
        a["gain_repeat"] += delta > 0 and repeat
        a["loss_repeat"] += delta < 0 and repeat
        a["gain_new_gold"] += delta > 0 and new_gold
        a["loss_new_gold"] += delta < 0 and new_gold
        if delta:
            examples.append({"dataset": ds, "id": key[1], "delta": delta, "repeat": repeat, "new_gold": new_gold,
                             "queries": [eq for eq, _, _ in ts[:k + 1]], "gold": after[key]["golden_answers"],
                             "answer_before": before[key]["answer"], "answer_after": after[key]["answer"]})
    rows = []
    for ds in sorted(acc, key=lambda s: (ORDER.index(s) if s in ORDER else 99, s)):
        a = acc[ds]
        n = a["n"]
        rows.append({"model": SHORT.get(model, model), "step": f"{k}->{k + 1}", "dataset": ds, "n": int(n),
                     "repeat": a["repeat"] / n, "overlap": a["overlap"] / n, "new_docs": a["new_docs"] / n,
                     "gold_before": a["gold_before"] / n, "new_gold": a["new_gold"] / n,
                     "gain": int(a["gain"]), "loss": int(a["loss"]),
                     "gain_repeat": int(a["gain_repeat"]), "loss_repeat": int(a["loss_repeat"]),
                     "gain_new_gold": int(a["gain_new_gold"]), "loss_new_gold": int(a["loss_new_gold"])})
    return rows, examples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, nargs="*", default=[1, 2], help="steps k -> k+1 to analyse")
    p.add_argument("--split", default="sub300")
    p.add_argument("--model", default="", help="substring of the model tag or short name")
    p.add_argument("--dataset", default="")
    p.add_argument("--examples", type=int, default=0, help="print this many gains and losses per model/set")
    p.add_argument("--csv", default="")
    a = p.parse_args()

    data = load(a.split)
    all_rows = []
    for model in sorted({m for m, _ in data}):
        if a.model and a.model not in model and a.model.lower() not in SHORT.get(model, "").lower():
            continue
        for k in a.k:
            rows, examples = analyse(data, model, k, a.split)
            rows = [r for r in rows if not a.dataset or r["dataset"] == a.dataset]
            if not rows:
                continue
            all_rows += rows
            print(f"\n{SHORT.get(model, model)}: search {k + 1} forced (exact:{k} -> exact:{k + 1})")
            print(f"{'set':<10} {'n':>4} {'repeat':>7} {'overlap':>8} {'new docs':>9} {'gold before':>12} "
                  f"{'new gold':>9} {'gain':>5} {'loss':>5}   {'gain|loss on repeats':>21}   {'gain|loss w/ new gold':>22}")
            for r in rows:
                print(f"{r['dataset']:<10} {r['n']:>4} {100 * r['repeat']:>6.0f}% {100 * r['overlap']:>7.0f}% "
                      f"{100 * r['new_docs']:>8.0f}% {100 * r['gold_before']:>11.0f}% {100 * r['new_gold']:>8.0f}% "
                      f"{r['gain']:>5} {r['loss']:>5}   {r['gain_repeat']:>10} | {r['loss_repeat']:<9}"
                      f"   {r['gain_new_gold']:>10} | {r['loss_new_gold']:<9}")
            if a.examples:
                for sign, name in ((1, "gains"), (-1, "losses")):
                    ex = [e for e in examples if e["delta"] * sign > 0 and (not a.dataset or e["dataset"] == a.dataset)]
                    for e in ex[:a.examples]:
                        print(f"  [{name[:-1]}] {e['dataset']}/{e['id']} gold={e['gold']} "
                              f"before={e['answer_before']!r} after={e['answer_after']!r} repeat={e['repeat']} "
                              f"new_gold={e['new_gold']}\n      queries: {e['queries']}")
    if a.csv and all_rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n{a.csv}: {len(all_rows)} rows")
    print("\nrepeat/overlap: the extra query against earlier ones. new gold: a gold answer first appears in the extra "
          "search's passages. gain/loss: questions EM flips to right/wrong with the extra search.")


if __name__ == "__main__":
    main()
