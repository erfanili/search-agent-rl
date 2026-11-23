#!/usr/bin/env python
"""
Compare reward arms on verl validation metrics, read from each run's train.log.
Prints one table per metric: per val set, the values at each validation step, plus the multi-hop average
(HotpotQA, 2Wiki, MuSiQue) at the last step. Metrics a run didn't log print as nan.

  .venv/bin/python scripts/compare_reward_arms.py                         # default runs below
  .venv/bin/python scripts/compare_reward_arms.py em=<run_dir> f1=<run_dir> ...
"""
import os
import re
import sys
from glob import glob

LOGS = os.path.join(os.environ.get("DATA_ROOT", "/data/erfan/search-agent-rl-data"), "logs")
DEFAULT_RUNS = {
    "em": "grpo_qwen2.5-3b_full_20261019_183501",
    "em_hit": "grpo_qwen2.5-3b_full_em_hit_2026*",
    "em_multi": "grpo_qwen2.5-3b_full_em_multi_2026*",
    "f1": "grpo_qwen2.5-3b_full_f1_2026*",
}
DATASETS = ["hotpotqa", "2wiki", "musique", "bamboogle"]
MULTI_HOP = DATASETS[:3]
METRICS = ["em", "f1", "answered", "n_search", "n_distinct_search", "evidence_hit", "correct_multi"]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
KV = re.compile(r"'?val-aux/(\w+)/(\w+)/mean@1'?:\s*([0-9.eE+-]+)")


def parse(run_dir):
    """{step: {(dataset, metric): value}} from the initial validation dict and the step lines."""
    text = ANSI.sub("", open(os.path.join(run_dir, "train.log"), errors="ignore").read())
    out = {}
    i = text.find("Initial validation metrics")
    if i >= 0:
        block = text[i:i + 50000].split("step:")[0]
        out[0] = {(d, m): float(v) for d, m, v in KV.findall(block)}
    for match in re.finditer(r"step:(\d+) - (.*)", text):
        vals = {(d, m): float(v) for d, m, v in KV.findall(match.group(2))}
        if vals:
            out[int(match.group(1))] = vals
    return out


def resolve(pattern):
    hits = sorted(glob(pattern if os.path.isabs(pattern) else os.path.join(LOGS, pattern)))
    if not hits:
        sys.exit(f"no run matches {pattern}")
    return hits[-1]


def main():
    specs = dict(a.split("=", 1) for a in sys.argv[1:]) or DEFAULT_RUNS
    runs = {arm: parse(resolve(p)) for arm, p in specs.items()}
    steps = sorted(set().union(*runs.values()))
    last = steps[-1]
    nan = float("nan")
    for metric in METRICS:
        print(f"\n## {metric}  (steps {'/'.join(map(str, steps))})")
        print("arm".ljust(10) + "".join(d.ljust(8 * len(steps) + 2) for d in DATASETS) + f"multi-hop@{last}")
        for arm, res in runs.items():
            row = arm.ljust(10)
            for d in DATASETS:
                row += "/".join(f"{res.get(s, {}).get((d, metric), nan):.3f}" for s in steps).ljust(8 * len(steps) + 2)
            mh = sum(res.get(last, {}).get((d, metric), nan) for d in MULTI_HOP) / len(MULTI_HOP)
            print(row + f"{mh:.3f}")


if __name__ == "__main__":
    main()
