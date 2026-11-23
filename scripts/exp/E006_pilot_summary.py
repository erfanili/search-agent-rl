#!/usr/bin/env python
"""E006 pilot summary (1 step) for one run directory.

  .venv-verl/bin/python scripts/exp/E006_pilot_summary.py $DATA_ROOT/logs/E006/E006-O-pilot

Part 1 = E005_pilot_summary.py (E004 part: step time, memory, truncation, parse rate from the dumps; step-1 reward
  components by label and search choice). Part 2 (E006): garbled share at step 1 from reward_steps "garble" (per vLLM
  server and all rollouts; agent/garble.py regex) and from the rollout dumps, capped answer statistics, and arm C's
  regime table (total reward, 0 vs >= 1 executed searches, by label).
E006 pilot rule (card): stop if the garbled share > 0.02 or the parse rate < 0.95.
"""
import json, os, re, subprocess, sys, glob

run = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
p = subprocess.run([sys.executable, os.path.join(here, "E005_pilot_summary.py"), run], capture_output=True, text=True)
print(p.stdout.strip())
m = re.search(r"parse rate ([0-9.]+)", p.stdout)
parse = float(m.group(1)) if m else None
r = [json.loads(l) for l in open(os.path.join(run, "reward_steps.jsonl")) if json.loads(l)["split"] == "train"][0]
g = r.get("garble") or {}
sys.path.insert(0, "/home/erfan/sar-alias-reward")
from agent.garble import is_garbled  # noqa: E402
D = [json.loads(l) for f in glob.glob(os.path.join(run, "rollouts", "*.jsonl")) for l in open(f)]
gd = sum(is_garbled([s["text"] for s in d["segments"] if s["role"] == "model"]) for d in D)
print("\n### E006: garble at step 1")
print(f"metric: all {g.get('all')}; by server {json.dumps({k: (v['garbled'], v['n']) for k, v in g.get('by_server', {}).items()})}; warn {g.get('warn_servers')}")
print(f"dumps: {gd} / {len(D)} garbled")
ac = r["answer_capped"]["by_label"]["e004_label"]
print("capped answer words: " + "; ".join(f"{k}: median {v['median_words']}, mean {v['mean_words']:.2f}, max {v['max_words']}, >20 {v['share_over_20']:.3f}" for k, v in ac.items()))
c = r["components"]["by_label"]["e004_label"]
print("regime table (total reward 0 searches / >=1 search, n): " + "; ".join(
    f"{k}: {v['mean_total_0search']:.3f} ({v['n_0search']}) / {v['mean_total_search']:.3f} ({v['n_search']})" for k, v in c.items()))
share = g.get("all", {}).get("share")
flags = []
if share is None or share > 0.02:
    flags.append(f"garbled share {share} > 0.02")
if parse is None or parse < 0.95:
    flags.append(f"parse rate {parse} < 0.95")
print("E006 PILOT FLAGS: " + ("; ".join(flags) if flags else "none"))
