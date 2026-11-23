#!/usr/bin/env python
"""E002 pilot summary: per model x condition, from $DATA_ROOT/logs/E002/pilot/<model>/<cond>/samples.jsonl.

Columns: episodes, parse rate (share with an <answer> tag), untagged, mean searches per episode, P(>=1 search),
cap-hit rate (finish == "length", the 4,096-token cap), forced answers (n_forced > 0, i.e. max:3 reached),
and for each malformed type the share of EPISODES with >= 1 such free turn (stop threshold 5%), plus the turn count.
Stop flags (card + Erfan 2026-11-21): 3B parse < 95%; any malformed type > 5% in a cell; 3B cap-hit > 2%.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA  # noqa: E402
from E001_prompts import MODELS  # noqa: E402

TYPES = ("empty_query", "unclosed_search", "multi_search", "search_and_answer")
base = os.path.join(DATA, "logs", "E002", "pilot")
print("| model | cond | episodes | parse rate | untagged | mean searches | P(≥1 search) | cap-hit | forced answer | "
      + " | ".join(f"{t} (eps %, turns)" for t in TYPES) + " | accuracy |")
print("|---|---|---|---|---|---|---|---|---|" + "---|" * len(TYPES) + "---|")
flags = []
for m in MODELS:
    for c in ("P1", "P2", "P6"):
        f = os.path.join(base, m, c, "samples.jsonl")
        if not os.path.exists(f):
            print(f"| {m} | {c} | missing |"); continue
        S = [s for l in open(f) for s in json.loads(l)["samples"]]
        n = len(S)
        parse = 1 - sum(s["parse"] == "untagged" for s in S) / n
        cap = sum(s["cap_hit"] for s in S) / n
        mal = {t: (sum(s["malformed"][t] > 0 for s in S) / n, sum(s["malformed"][t] for s in S)) for t in TYPES}
        print(f"| {m} | {c} | {n} | {parse:.3f} | {sum(s['parse'] == 'untagged' for s in S)} | "
              f"{sum(s['n_search'] for s in S) / n:.2f} | {sum(s['n_search'] > 0 for s in S) / n:.2f} | {cap:.3f} | "
              f"{sum(s['n_forced'] > 0 for s in S)} | " + " | ".join(f"{100 * mal[t][0]:.1f} ({mal[t][1]})" for t in TYPES)
              + f" | {sum(s['correct'] for s in S) / n:.3f} |")
        is3b = m != "qwen7b-instruct"
        if is3b and parse < 0.95: flags.append(f"{m} {c}: parse {parse:.3f} < 0.95")
        if is3b and cap > 0.02: flags.append(f"{m} {c}: cap-hit {cap:.3f} > 0.02")
        for t in TYPES:
            if mal[t][0] > 0.05: flags.append(f"{m} {c}: {t} {100 * mal[t][0]:.1f}% of episodes > 5%")
print("\nSTOP FLAGS:" if flags else "\nno stop flags", *flags, sep="\n  ")
