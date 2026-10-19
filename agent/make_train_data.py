#!/usr/bin/env python
"""
Build the GRPO training and validation sets in verl's format (P1.5, decision D-09).

train.parquet  Search-R1's NQ + HotpotQA train mix (PeterJinGo/nq_hotpotqa_train, 169,615 rows:
               NQ 79,168, HotpotQA 90,447), the data the released Search-R1 checkpoints trained on.
val.parquet    our frozen eval subsets (default: sub300 of hotpotqa, 2wiki, musique, plus bamboogle),
               so validation during training uses the same questions as tests/score_predictions.py.

Same columns as agent/make_check_data.py (data_source, agent_name, prompt, ability, reward_model,
extra_info). The prompt is rebuilt with make_question_prompt; the script checks it equals the prompt
stored in Search-R1's parquet, so the task text is exactly theirs. Search-R1's ids restart per source
(nq and hotpotqa both have train_0), so extra_info.id is "<source>_<id>".

Usage (training env; the dataset must be in the HF cache: hf download PeterJinGo/nq_hotpotqa_train --repo-type dataset):
  .venv-verl/bin/python agent/make_train_data.py --out /data/erfan/search-agent-rl-data/verl_data/nq_hotpotqa
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.make_check_data import AGENT_NAME  # noqa: E402
from agent.search_r1_rollout import make_question_prompt  # noqa: E402

DATA = "/data/erfan/search-agent-rl-data"
SRC_GLOB = f"{DATA}/hf_cache/hub/datasets--PeterJinGo--nq_hotpotqa_train/snapshots/*/train.parquet"


def row(data_source, question, answers, index, qid):
    return {
        "data_source": data_source,
        "agent_name": AGENT_NAME,
        "prompt": [{"role": "user", "content": make_question_prompt(question)}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": [str(a) for a in answers]}},
        "extra_info": {"index": index, "id": qid, "question": question},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=None, help=f"Search-R1 train.parquet (default: {SRC_GLOB})")
    p.add_argument("--gold-dir", default=f"{DATA}/eval")
    p.add_argument("--val-datasets", nargs="+", default=["hotpotqa", "2wiki", "musique", "bamboogle"])
    p.add_argument("--val-split", default="sub300")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import pandas as pd

    src = a.src or (sorted(glob.glob(SRC_GLOB)) or [None])[-1]
    if not src or not os.path.exists(src):
        sys.exit(f"Search-R1 train set not found ({SRC_GLOB}); download it first (see --help)")

    df = pd.read_parquet(src)
    rows, mismatched = [], 0
    for i, r in enumerate(df.itertuples(index=False)):
        out = row(r.data_source, r.question, list(r.golden_answers), i, f"{r.data_source}_{r.id}")
        mismatched += out["prompt"][0]["content"] != r.prompt[0]["content"]
        rows.append(out)
    if mismatched:
        sys.exit(f"{mismatched} prompts differ from Search-R1's; make_question_prompt no longer matches their task")
    train = pd.DataFrame(rows)

    val_rows = []
    for ds in a.val_datasets:
        with open(os.path.join(a.gold_dir, ds, f"{a.val_split}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                val_rows.append(row(ds, r["question"], r["golden_answers"], len(val_rows), r["id"]))
    val = pd.DataFrame(val_rows)

    os.makedirs(a.out, exist_ok=True)
    train.to_parquet(os.path.join(a.out, "train.parquet"))
    val.to_parquet(os.path.join(a.out, "val.parquet"))
    print(f"train: {len(train)} rows {train.data_source.value_counts().to_dict()} (prompts identical to Search-R1's)")
    print(f"val:   {len(val)} rows {val.data_source.value_counts().to_dict()} ({a.val_split})")
    print(f"wrote {a.out}/{{train,val}}.parquet")


if __name__ == "__main__":
    main()
