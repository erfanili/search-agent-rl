#!/usr/bin/env python
"""E004 pilot / training summary from a run directory.

  .venv-verl/bin/python scripts/exp/E004_pilot_summary.py $DATA_ROOT/logs/E004/E004-A-pilot

From train.log (verl console metrics, per step): timing_s/step, perf/max_memory_allocated_gb,
  perf/max_memory_reserved_gb, response_length/clip_ratio (share of episodes at the 2,048-token cap = truncation).
From reward_steps.jsonl (agent/step_stats.py): P(>=1 search) overall (share_gt["0"]) and per e004_label, mean score.
From rollouts/*.jsonl (agent loop dumps; at most 200 episodes per worker process, the first ones of the run):
  parse rate = episodes whose model text (environment text stripped) has an <answer> (E001/E002 parse rule; an
  environment-forced <answer> counts, as in E002); malformed rate = episodes with >= 1 free model turn of each E002
  type (empty_query, unclosed_search, multi_search, search_and_answer), applied to the kept text of free turns
  (model segments of kind search / answer / invalid; forced-turn segments excluded).
Stop rules (card + Erfan): step-1 P(>=1 search) outside [0.3, 0.7]; parse rate < 0.95; truncation > 0.02.
"""
import glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.qa_metrics import strip_environment_text  # noqa: E402
from E002_run import malformed, parse  # noqa: E402

run = sys.argv[1]
SPECIAL = re.compile(r"<\|(im_end|endoftext|im_start)\|>")
steps = {}
for line in open(os.path.join(run, "train.log"), errors="replace"):
    m = re.search(r"step:(\d+) - ", line)
    if not m or "timing_s/step:" not in line:
        continue
    kv = dict(re.findall(r"([\w/@\-.]+):(-?[\d.e+-]+)(?: -|$)", line))
    steps[int(m.group(1))] = kv
rs = [json.loads(l) for l in open(os.path.join(run, "reward_steps.jsonl")) if json.loads(l)["split"] == "train"]

print("| step | step time s | max mem alloc GB | max mem reserved GB | truncation (clip ratio) | P(≥1 search) | knowable | unknowable | mean reward | mean reward knowable | mean reward unknowable |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rs:
    s = r["step"]
    kv = steps.get(s, {})
    bl = r.get("by_label", {}).get("e004_label", {})
    f = lambda k: f"{float(kv[k]):.3f}" if k in kv else "—"
    print(f"| {s} | {f('timing_s/step')} | {f('perf/max_memory_allocated_gb')} | {f('perf/max_memory_reserved_gb')} | "
          f"{f('response_length/clip_ratio')} | {r['share_gt']['0']:.3f} | {bl.get('knowable', {}).get('share_search', float('nan')):.3f} | "
          f"{bl.get('unknowable', {}).get('share_search', float('nan')):.3f} | {r['mean_score']:.3f} | "
          f"{bl.get('knowable', {}).get('mean_score', float('nan')):.3f} | {bl.get('unknowable', {}).get('mean_score', float('nan')):.3f} |")

eps = [json.loads(l) for p in glob.glob(os.path.join(run, "rollouts", "*.jsonl")) for l in open(p)]
T = ("empty_query", "unclosed_search", "multi_search", "search_and_answer")
n_parse, mal, fin = 0, {t: 0 for t in T}, {}
for e in eps:
    text = SPECIAL.sub("", "".join(s["text"] for s in e["segments"]))
    n_parse += parse(strip_environment_text(text))[1] != "untagged"
    segs = e["segments"]
    free = [s for k, s in enumerate(segs) if s["role"] == "model" and s["kind"] in ("search", "answer", "invalid")
            and not (k > 0 and segs[k - 1]["kind"].startswith("forced_"))]
    for t in T:
        mal[t] += any(malformed(SPECIAL.sub("", s["text"]))[t] for s in free)
    fin[e["finish"]] = fin.get(e["finish"], 0) + 1
n = len(eps)
print(f"\ndumped episodes: {n}; parse rate {n_parse / n:.3f}; finish reasons {fin}")
print("malformed (share of episodes): " + ", ".join(f"{t} {mal[t] / n:.3f}" for t in T))
s1 = rs[0]
flags = []
if not 0.3 <= s1["share_gt"]["0"] <= 0.7:
    flags.append(f"step-1 search share {s1['share_gt']['0']:.3f} outside 0.3-0.7")
if n_parse / n < 0.95:
    flags.append(f"parse rate {n_parse / n:.3f} < 0.95")
tr = [float(steps[s]["response_length/clip_ratio"]) for s in steps if "response_length/clip_ratio" in steps[s]]
if tr and max(tr) > 0.02:
    flags.append(f"truncation {max(tr):.3f} > 0.02")
print("\nSTOP FLAGS: " + ("; ".join(flags) if flags else "none"))
