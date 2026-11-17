#!/usr/bin/env python
"""
Print the per-step depth statistics of one or more training runs (<run>/reward_steps.jsonl, written by
agent/reward.py:compute_score_batch; fields in agent/step_stats.py).

  .venv/bin/python scripts/depth_dynamics.py <run_dir> [<run_dir> ...]          # training steps
  .venv/bin/python scripts/depth_dynamics.py --split val <run_dir>              # validation passes
  .venv/bin/python scripts/depth_dynamics.py --labels <run_dir>                 # also by data label

Columns: searches per rollout, share with >= 1 / >= 2 searches, continuation p1 = P(2nd | 1st),
mean score at 0 / 1 / 2+ searches, zero-variance groups (all 0 / all 1), groups mixed in depth, and
groups mixed in depth AND score (the only ones that move the stop decision).
"""

import argparse
import json
import os
import sys


def fmt(x, w=6, p=2):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def score_2plus(r):
    h, s = r["depth_hist"], r["score_by_depth"]
    num = sum(h[k] * s[k] for k in h if int(k) >= 2 and k in s)
    den = sum(h[k] for k in h if int(k) >= 2)
    return num / den if den else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--labels", action="store_true")
    a = p.parse_args()

    for run in a.runs:
        path = run if run.endswith(".jsonl") else os.path.join(run, "reward_steps.jsonl")
        if not os.path.exists(path):
            print(f"{path}: not found", file=sys.stderr)
            continue
        rows = [r for r in map(json.loads, open(path)) if r["split"] == a.split]
        print(f"\n{run}  ({len(rows)} {a.split} lines)")
        print(f"{'step':>5} {'score':>6} {'srch':>6} {'>=1':>6} {'>=2':>6} {'p1':>6} {'s@0':>6} {'s@1':>6} {'s@2+':>6}"
              f" {'zvar':>6} {'all0':>6} {'all1':>6} {'mixD':>6} {'mixDS':>6}")
        for r in rows:
            g = r["share_gt"]
            sd = r["score_by_depth"]
            p1 = g["1"] / g["0"] if g["0"] else None
            print(f"{r['step']:>5} {fmt(r['mean_score'])} {fmt(r['mean_search'])} {fmt(g['0'])} {fmt(g['1'])} {fmt(p1)}"
                  f" {fmt(sd.get('0'))} {fmt(sd.get('1'))} {fmt(score_2plus(r))} {fmt(r['zero_var_groups'])}"
                  f" {fmt(r['all_zero_groups'])} {fmt(r['all_one_groups'])} {fmt(r['mixed_depth_groups'])}"
                  f" {fmt(r['depth_contrast_groups'])}")
            if a.labels:
                for key, vals in (r.get("by_label") or {}).items():
                    cells = "  ".join(f"{key}={v}: n {s['n']} score {fmt(s['mean_score'], 4)} srch {fmt(s['mean_search'], 4)}"
                                      f" >=2 {fmt(s['share_gt1'], 4)}" for v, s in vals.items())
                    print(f"{'':>5}   {cells}")


if __name__ == "__main__":
    main()
