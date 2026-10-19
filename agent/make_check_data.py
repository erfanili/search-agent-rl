#!/usr/bin/env python
"""
Write a tiny dataset in verl's format for the P1.3 rollout check (plumbing only, never training).

Takes the first --n questions of one frozen eval set (default HotpotQA sub300) and writes
train.parquet and val.parquet with the columns verl's RLHFDataset and our agent loop read:

  data_source   "hotpotqa"                      -> reward function argument, metric grouping
  agent_name    "search_r1_agent"               -> which agent loop runs the row (configs/agent_loop.yaml)
  prompt        [{"role": "user", "content": Search-R1 instruction + question}]
  ability       "multihop_qa"
  reward_model  {"style": "rule", "ground_truth": {"target": [gold answers]}}
  extra_info    {"index", "id", "question"}

These are EVAL questions: fine for checking that ids and masks are right, wrong for training.
P1.5 builds the real training set (NQ + HotpotQA train, decision D-09) in this same format.

Usage (training env):
  .venv-verl/bin/python agent/make_check_data.py --out /data/erfan/search-rl-agent-data/verl_data/p1_3_check --n 64
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.search_r1_rollout import make_question_prompt  # noqa: E402

AGENT_NAME = "search_r1_agent"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-dir", default="/data/erfan/search-rl-agent-data/eval")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--split", default="sub300")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import pandas as pd

    rows = []
    with open(os.path.join(a.gold_dir, a.dataset, f"{a.split}.jsonl")) as f:
        for i, line in enumerate(f):
            if i >= a.n:
                break
            r = json.loads(line)
            rows.append({
                "data_source": a.dataset,
                "agent_name": AGENT_NAME,
                "prompt": [{"role": "user", "content": make_question_prompt(r["question"])}],
                "ability": "multihop_qa",
                "reward_model": {"style": "rule", "ground_truth": {"target": list(r["golden_answers"])}},
                "extra_info": {"index": i, "id": r["id"], "question": r["question"]},
            })
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    for name in ("train", "val"):
        df.to_parquet(os.path.join(a.out, f"{name}.parquet"))
    print(f"wrote {len(df)} rows to {a.out}/{{train,val}}.parquet (from {a.dataset}/{a.split}; plumbing check only)")


if __name__ == "__main__":
    main()