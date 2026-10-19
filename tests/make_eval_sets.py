#!/usr/bin/env python
"""
Download the eval sets once and freeze them as files (P0.8, feeds D-10).

Source: RUC-NLPIR/FlashRAG_datasets on Hugging Face, the preprocessed versions Search-R1
and most follow-up papers evaluate on. Test labels are hidden for HotpotQA / 2Wiki / MuSiQue,
so the dev split is the evaluation set; Bamboogle only has its 125-question test split.

Writes, under --out (default /data/erfan/eval/; all data lives under /data/erfan):
  <name>/full.jsonl        every dev row: {"id", "question", "golden_answers"}
  <name>/sub<N>.jsonl      fixed random subset of N rows (seed 0, kept in original order),
                           used for fast iteration; full.jsonl is for final numbers
  MANIFEST.json            source, split, sizes, seed, and sha256 of every file

Every later eval reads these files, never the Hub. That way an upstream dataset update
cannot silently change the numbers, and the sha256 in MANIFEST.json proves which version
a result was computed on.

Usage:
  .venv/bin/python tests/make_eval_sets.py --subset 300
"""

import argparse
import hashlib
import json
import os
import random

SETS = {  # short name -> (FlashRAG config, split)
    "hotpotqa": ("hotpotqa", "dev"),
    "2wiki": ("2wikimultihopqa", "dev"),
    "musique": ("musique", "dev"),
    "bamboogle": ("bamboogle", "test"),
}
REPO = "RUC-NLPIR/FlashRAG_datasets"
DEFAULT_OUT = "/data/erfan/eval"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--subset", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="*", choices=list(SETS), help="default: all four")
    a = p.parse_args()

    from datasets import load_dataset

    manifest = {"source": REPO, "seed": a.seed, "subset_size": a.subset, "sets": {}}
    for name in a.only or SETS:
        config, split = SETS[name]
        ds = load_dataset(REPO, config, split=split)
        rows = [{"id": str(r["id"]), "question": r["question"].strip(), "golden_answers": list(r["golden_answers"])} for r in ds]

        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{name}: duplicate ids")
        empty = [r["id"] for r in rows if not any(g.strip() for g in r["golden_answers"])]
        if empty:
            raise SystemExit(f"{name}: {len(empty)} rows with no gold answer, e.g. {empty[:3]}")

        k = min(a.subset, len(rows))
        keep = sorted(random.Random(a.seed).sample(range(len(rows)), k))
        subset = [rows[i] for i in keep]

        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        full_path, sub_path = os.path.join(d, "full.jsonl"), os.path.join(d, f"sub{a.subset}.jsonl")
        write_jsonl(full_path, rows)
        write_jsonl(sub_path, subset)
        manifest["sets"][name] = {
            "config": config,
            "split": split,
            "n_full": len(rows),
            "n_subset": k,
            "full_sha256": sha256(full_path),
            f"sub{a.subset}_sha256": sha256(sub_path),
            "golds_per_question_mean": round(sum(len(r["golden_answers"]) for r in rows) / len(rows), 2),
        }
        print(f"{name:10s} {config}/{split}: {len(rows):6,} rows, subset {k}")

    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {a.out}/MANIFEST.json")


if __name__ == "__main__":
    main()