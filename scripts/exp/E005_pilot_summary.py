#!/usr/bin/env python
"""E005 pilot summary for one run directory.

  .venv-verl/bin/python scripts/exp/E005_pilot_summary.py $DATA_ROOT/logs/E005/E005-O-pilot

Part 1 = E004_pilot_summary.py unchanged (step time, peak memory, truncation = response_length/clip_ratio, P(>=1
  <search> tag) by label, parse rate and malformed-search rates from the agent-loop dumps, E004 stop flags).
Part 2 (E005): reward components at step 1 from reward_steps.jsonl "components" (agent/step_stats.component_stats):
  by e004_label, outcome (alias_match_strict), arm term and total, over all rollouts and at 0 vs >= 1 executed
  searches (d = an <information> block was inserted), with the executed-search share.
E005 stop rule (card): step-1 executed-search share outside [0.3, 0.7] overall, parse rate < 0.95, truncation > 0.02.
  The parse and truncation flags come from part 1; the search-share flag is recomputed on executed searches.
"""
import json, os, subprocess, sys

run = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
p1 = subprocess.run([sys.executable, os.path.join(here, "E004_pilot_summary.py"), run], capture_output=True, text=True)
print(p1.stdout.strip())
if p1.returncode:
    print(p1.stderr[-2000:])
rs = [json.loads(l) for l in open(os.path.join(run, "reward_steps.jsonl")) if json.loads(l)["split"] == "train"]
f = lambda x: "—" if x is None else f"{x:.3f}"
for r in rs:
    c = r["components"]
    print(f"\n### step {r['step']}: reward components (executed-search decision)")
    print("| label | n | share ≥1 search | outcome | arm | total | outcome 0s | arm 0s | total 0s | outcome ≥1s | arm ≥1s | total ≥1s | groups mixed search | mixed search & total |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cells = [("all", c["all"])] + list(c.get("by_label", {}).get("e004_label", {}).items())
    for name, x in cells:
        print(f"| {name} | {x['n']} | {f(x['share_search'])} | {f(x['mean_outcome'])} | {f(x['mean_arm'])} | {f(x['mean_total'])} | "
              f"{f(x['mean_outcome_0search'])} | {f(x['mean_arm_0search'])} | {f(x['mean_total_0search'])} | "
              f"{f(x['mean_outcome_search'])} | {f(x['mean_arm_search'])} | {f(x['mean_total_search'])} | "
              f"{x['n_groups_mixed_search']}/{x['n_groups']} | {x['n_groups_mixed_search_and_total']}/{x['n_groups']} |")
s1 = rs[0]["components"]["all"]["share_search"]
flags = [l for l in p1.stdout.splitlines() if l.startswith("STOP FLAGS")]
e5 = [x for x in (flags[0].split(": ", 1)[1].split("; ") if flags else []) if x != "none" and "search share" not in x]
if not 0.3 <= s1 <= 0.7:
    e5.append(f"step-1 executed-search share {s1:.3f} outside 0.3-0.7")
print("\nE005 STOP FLAGS: " + ("; ".join(e5) if e5 else "none"))
