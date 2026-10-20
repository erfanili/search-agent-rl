#!/usr/bin/env python
"""
Summarise wrong answers in a scored per_example.jsonl (tests/score_predictions.py output), for error analysis.

Buckets each EM miss by what kind of miss it is:
  no_answer   the episode produced no <answer>
  near_miss   F1 > 0 or ACC = 1: overlapping or containing the gold answer (format / phrasing drift)
  wrong       no token overlap with any gold answer
and prints answer-length stats plus examples of each bucket. Standard library only.

Usage:
  .venv/bin/python scripts/show_misses.py <eval_dir>/scored/per_example.jsonl --dataset bamboogle
"""

import argparse
import json
from collections import Counter


def bucket(r):
    if not r["answered"]:
        return "no_answer"
    if r["f1"] > 0 or r["acc"] > 0:
        return "near_miss"
    return "wrong"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("per_example")
    p.add_argument("--dataset", default=None)
    p.add_argument("--examples", type=int, default=15, help="examples printed per bucket")
    a = p.parse_args()

    rows = [json.loads(line) for line in open(a.per_example)]
    if a.dataset:
        rows = [r for r in rows if r["dataset"] == a.dataset]
    misses = [r for r in rows if r["em"] == 0]
    counts = Counter(bucket(r) for r in misses)
    words = [len((r["answer"] or "").split()) for r in rows if r["answered"]]
    hit_words = [len(r["answer"].split()) for r in rows if r["em"] == 1]

    print(f"{a.dataset or 'all'}: {len(rows)} questions, EM {sum(r['em'] for r in rows) / len(rows):.3f}, "
          f"{len(misses)} misses: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    if words:
        print(f"answer length (words): mean {sum(words) / len(words):.1f}, max {max(words)}"
              + (f"; correct answers mean {sum(hit_words) / len(hit_words):.1f}" if hit_words else ""))
    for b in ("near_miss", "wrong", "no_answer"):
        ex = [r for r in misses if bucket(r) == b][: a.examples]
        if not ex:
            continue
        print(f"\n--- {b} ({counts[b]}) ---")
        for r in ex:
            print(f"{r['id']:>10}  f1={r['f1']:.2f} searches={r['n_search']}  answer={r['answer']!r}  gold={r['golden_answers']}")


if __name__ == "__main__":
    main()
