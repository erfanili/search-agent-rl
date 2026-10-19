#!/usr/bin/env python
"""
Build the GRPO training set: NQ train + HotpotQA train, Search-R1's mix (decision D-09).

Source: RUC-NLPIR/FlashRAG_datasets, the same preprocessed copies Search-R1 merged
(scripts/data_process/qa_search_train_merge.py: nq + hotpotqa, train split). That is about
79k NQ (single-hop) + 90k HotpotQA (two-hop) questions.

Writes, under --out (verl format, same columns as agent/make_check_data.py):
  train.parquet    every training question except the held-out ones
  dev.parquet      --dev-size questions held out from the SAME mix, for choosing checkpoints
  MANIFEST.json    sizes per source, seed, overlaps removed, sha256 of both files

Why a held-out dev split: checkpoints must be selected on questions that are neither trained on
nor reported. 2609.28765 selected on its reported eval suite and lists that as a limitation.
Here: train on train.parquet, pick checkpoints on dev.parquet, report on the frozen eval sets.

Safety check: any training question whose normalized text also appears in the frozen eval sets
(/data/erfan/eval/*/full.jsonl) is dropped and counted. HotpotQA train and dev are different
splits, so this should be ~0; the check makes it a measured fact instead of an assumption.

Usage (training env; downloads ~20 MB once):
  HF_HOME=/data/erfan/hf_cache .venv-verl/bin/python agent/make_train_data.py --out /data/erfan/verl_data/nq_hotpotqa
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qa_metrics import normalize_answer  # noqa: E402
from agent.search_r1_rollout import make_question_prompt  # noqa: E402

REPO = "RUC-NLPIR/FlashRAG_datasets"
SOURCES = ("nq", "hotpotqa")
AGENT_NAME = "search_r1_agent"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_questions(gold_dir):
    if not os.path.isdir(gold_dir):
        sys.exit(f"{gold_dir} not found: build the frozen eval sets first (tests/make_eval_sets.py)")
    seen = set()
    for name in sorted(os.listdir(gold_dir)):
        path = os.path.join(gold_dir, name, "full.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                seen.update(normalize_answer(json.loads(line)["question"]) for line in f if line.strip())
    return seen


def to_row(source, r, i):
    return {
        "data_source": source,
        "agent_name": AGENT_NAME,
        "prompt": [{"role": "user", "content": make_question_prompt(r["question"])}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": [str(g) for g in r["golden_answers"]]}},
        "extra_info": {"index": i, "id": str(r["id"]), "question": r["question"], "source": source},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--gold-dir", default="/data/erfan/eval")
    p.add_argument("--dev-size", type=int, default=1000, help="held out, split evenly between the sources")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import pandas as pd
    from datasets import load_dataset

    banned = eval_questions(a.gold_dir)
    print(f"{len(banned):,} eval questions loaded for the overlap check")

    rows, stats = [], {}
    for source in SOURCES:
        ds = load_dataset(REPO, source, split="train")
        kept = dropped_overlap = dropped_empty = 0
        for r in ds:
            if not r["question"].strip() or not any(str(g).strip() for g in r["golden_answers"]):
                dropped_empty += 1
                continue
            if normalize_answer(r["question"]) in banned:
                dropped_overlap += 1
                continue
            rows.append((source, r))
            kept += 1
        stats[source] = {"kept": kept, "dropped_eval_overlap": dropped_overlap, "dropped_empty": dropped_empty}
        print(f"{source:9s} {kept:7,} kept, {dropped_overlap} overlap with eval sets, {dropped_empty} empty")

    rng = random.Random(a.seed)
    rng.shuffle(rows)  # mix the sources; verl also shuffles per epoch, this fixes the dev draw
    per_source = a.dev_size // len(SOURCES)
    dev_idx, counts = set(), {s: 0 for s in SOURCES}
    for i, (source, _) in enumerate(rows):
        if counts[source] < per_source:
            dev_idx.add(i)
            counts[source] += 1
    train = [to_row(s, r, i) for i, (s, r) in enumerate(rows) if i not in dev_idx]
    dev = [to_row(s, r, i) for i, (s, r) in enumerate(rows) if i in dev_idx]

    os.makedirs(a.out, exist_ok=True)
    paths = {name: os.path.join(a.out, f"{name}.parquet") for name in ("train", "dev")}
    pd.DataFrame(train).to_parquet(paths["train"])
    pd.DataFrame(dev).to_parquet(paths["dev"])
    manifest = {
        "source": REPO,
        "sources": stats,
        "seed": a.seed,
        "n_train": len(train),
        "n_dev": len(dev),
        "dev_per_source": counts,
        "sha256": {k: sha256(v) for k, v in paths.items()},
    }
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"train {len(train):,}  dev {len(dev):,}  -> {a.out}")


if __name__ == "__main__":
    main()