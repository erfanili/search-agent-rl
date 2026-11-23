#!/usr/bin/env python
"""
Gate for a forced-depth training pilot (E1): the rollouts verl trained on obey depth_policy exact:k.

Reads <run>/rollouts/*.jsonl (episode dumps, agent/verl_agent_loop.py) and <run>/reward_steps.jsonl.
Checks, on every dumped episode that finished with an answer:
  - exactly k executed searches (n_search == k) and k + 1 forced tags
  - every forced_* segment is env (mask 0); the policy's own tokens never contain an opening action tag
  - the response ends on a model token (mask 1), where the reward goes
and on the run: >= --min-answer of episodes answered, training steps logged with mean searches >= k - 0.05.
Prints a few episodes in short form. Exit 0 = pass, 1 = fail.

  .venv/bin/python scripts/check_depth_pilot.py <run_dir> --k 2
"""

import argparse
import glob
import json
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--min-answer", type=float, default=0.9)
    p.add_argument("--show", type=int, default=3)
    a = p.parse_args()

    eps = [json.loads(line) for f in sorted(glob.glob(os.path.join(a.run, "rollouts", "*.jsonl"))) for line in open(f)]
    fails = []
    if not eps:
        sys.exit(f"FAIL: no episode dumps in {a.run}/rollouts")
    answered = [e for e in eps if e["finish"] == "answer"]
    bad = []
    for e in answered:
        segs = e["segments"]
        mask_ok = all(s["role"] == "env" for s in segs if s["kind"].startswith("forced_"))
        model_text = "".join(s["text"] for s in segs if s["role"] == "model")
        no_tag = "<search>" not in model_text and "<answer>" not in model_text
        ok = (e["n_search"] == a.k and e.get("n_forced") == a.k + 1 and mask_ok and no_tag
              and e["response_mask"][-1] == 1 and segs[-1]["role"] == "model")
        if not ok:
            bad.append(e)
    share = len(answered) / len(eps)
    print(f"episodes dumped {len(eps)}, answered {share:.3f}, contract violations among answered {len(bad)}")
    print(f"finish reasons: { {f: sum(e['finish'] == f for e in eps) for f in sorted({e['finish'] for e in eps})} }")
    ov = sum(e.get("n_overridden", 0) for e in eps) / max(1, sum(e.get("n_forced", 0) for e in eps))
    print(f"forced turns where the policy had chosen otherwise: {ov:.3f}")
    if bad:
        e = bad[0]
        fails.append(f"{len(bad)} answered episodes break the contract, e.g. n_search={e['n_search']} n_forced={e.get('n_forced')} "
                     f"kinds={[(s['role'], s['kind']) for s in e['segments']]}")
    if share < a.min_answer:
        fails.append(f"answered share {share:.3f} < {a.min_answer}")

    steps_path = os.path.join(a.run, "reward_steps.jsonl")
    train = [r for r in map(json.loads, open(steps_path)) if r["split"] == "train"] if os.path.exists(steps_path) else []
    if not train:
        fails.append(f"no training steps in {steps_path}")
    for r in train:
        print(f"step {r['step']}: score {r['mean_score']:.3f} searches {r['mean_search']:.3f} zero-var groups {r['zero_var_groups']:.2f}")
        if r["mean_search"] < a.k - 0.05:
            fails.append(f"step {r['step']}: mean searches {r['mean_search']:.3f} < {a.k} - 0.05")

    for e in answered[: a.show]:
        print("\n---- episode:", " | ".join(f"[{s['role'][0]}:{s['kind']}] {s['text'][:90]!r}" for s in e["segments"]))

    if fails:
        print("\nFAIL:\n  " + "\n  ".join(fails))
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
