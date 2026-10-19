#!/usr/bin/env python
"""
Score a predictions file against the frozen eval sets (P0.8). Standard library only.

Every method in the baseline table (no retrieval, naive RAG, prompted agent, RL agent)
writes the same predictions format, and this script is the only thing that turns predictions
into numbers. One scorer for all rows is what makes the table comparable.

Predictions: JSONL, one line per question, with exactly one of "output" or "prediction":
  {"dataset": "hotpotqa", "id": "...", "output": "<think>..</think><search>..</search>...<answer>X</answer>"}
  {"dataset": "hotpotqa", "id": "...", "prediction": "X"}      # bare answer; null = no answer
"output" is a generated agent trajectory: the answer is parsed from its last <answer> block
and searches are counted from <search> blocks. "prediction" is used as the answer directly.

Writes to --out:
  metrics.json        per-dataset EM / F1 / ACC / answer rate / abort rate / mean searches,
                      macro average, plus sha256 of the predictions and gold files used
  per_example.jsonl   one scored row per question, for error analysis (P2.8)

Checks, each a hard error: unknown dataset, id not in the gold set, duplicate (dataset, id),
a row with both or neither of output/prediction, and questions with no prediction. Use
--allow-missing to score missing questions as aborts (0 on every metric) instead.

Usage:
  .venv/bin/python tests/score_predictions.py --preds runs/naive_rag/preds.jsonl --split sub300 --out runs/naive_rag/
  .venv/bin/python tests/score_predictions.py --make-dummy /data/erfan/logs/dummy_preds.jsonl --split sub300
"""

import argparse
import hashlib
import json
import os
import sys

from qa_metrics import aggregate, score_output, score_prediction

DEFAULT_GOLD = "/data/erfan/eval"  # written by make_eval_sets.py


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield n, json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{n}: not valid JSON ({e})")


def load_gold(gold_dir, split, datasets=None):
    """{dataset: {id: golden_answers}} for every set listed in MANIFEST.json."""
    with open(os.path.join(gold_dir, "MANIFEST.json")) as f:
        manifest = json.load(f)
    gold, files = {}, {}
    for name in sorted(manifest["sets"]):
        if datasets and name not in datasets:
            continue
        path = os.path.join(gold_dir, name, f"{split}.jsonl")
        if not os.path.exists(path):
            sys.exit(f"missing {path}; run tests/make_eval_sets.py")
        gold[name] = {r["id"]: r["golden_answers"] for _, r in read_jsonl(path)}
        files[name] = {"path": path, "sha256": sha256(path)}
    return gold, files


def make_dummy(path, gold):
    """Predictions with known scores, to test the pipeline end to end.
    Per dataset, by row index i: i % 10 == 0 -> no answer (abort); else even i -> correct
    answer with one search; else odd i -> wrong answer with two searches."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for name, rows in gold.items():
            for i, (qid, golds) in enumerate(rows.items()):
                if i % 10 == 0:
                    out = "<think>out of budget</think><search>q</search>"
                elif i % 2 == 0:
                    out = f"<search>q1</search><information>doc</information><answer>{golds[0]}</answer>"
                else:
                    out = "<search>q1</search><search>q2</search><answer>zzqq wrong</answer>"
                f.write(json.dumps({"dataset": name, "id": qid, "output": out}) + "\n")
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds")
    p.add_argument("--gold-dir", default=DEFAULT_GOLD)
    p.add_argument("--split", default="sub300", help="sub300 for iteration, full for final numbers")
    p.add_argument("--datasets", nargs="*", help="restrict to these sets (default: all in MANIFEST)")
    p.add_argument("--out", help="directory for metrics.json and per_example.jsonl")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--make-dummy", metavar="PATH")
    a = p.parse_args()

    gold, gold_files = load_gold(a.gold_dir, a.split, a.datasets)
    if a.make_dummy:
        make_dummy(a.make_dummy, gold)
        return
    if not a.preds:
        p.error("--preds is required (or --make-dummy)")

    rows, seen, errors = [], set(), []
    for n, r in read_jsonl(a.preds):
        name, qid = r.get("dataset"), str(r.get("id"))
        if name not in gold:
            errors.append(f"line {n}: dataset {name!r} not in {sorted(gold)}")
            continue
        if qid not in gold[name]:
            errors.append(f"line {n}: id {qid!r} not in {name}/{a.split}")
            continue
        if (name, qid) in seen:
            errors.append(f"line {n}: duplicate {name}/{qid}")
            continue
        has_out, has_pred = "output" in r, "prediction" in r
        if has_out == has_pred:
            errors.append(f"line {n}: needs exactly one of 'output' or 'prediction'")
            continue
        seen.add((name, qid))
        golds = gold[name][qid]
        s = score_output(r["output"], golds) if has_out else score_prediction(r["prediction"], golds)
        rows.append({"dataset": name, "id": qid, "golden_answers": golds, **s})

    missing = [(name, qid) for name in gold for qid in gold[name] if (name, qid) not in seen]
    if missing and not a.allow_missing:
        errors.append(f"{len(missing)} questions have no prediction, e.g. {missing[:3]} (or pass --allow-missing)")
    if errors:
        sys.exit("\n".join(["scoring refused:"] + errors[:20] + ([f"... {len(errors) - 20} more"] if len(errors) > 20 else [])))
    for name, qid in missing:
        rows.append({"dataset": name, "id": qid, "golden_answers": gold[name][qid], "answer": None,
                     "answered": False, "n_search": 0, "em": 0.0, "f1": 0.0, "acc": 0.0, "missing": True})

    metrics = aggregate(rows)
    result = {
        "split": a.split,
        "predictions": {"path": a.preds, "sha256": sha256(a.preds)},
        "gold": gold_files,
        "n_missing_scored_as_abort": len(missing),
        "metrics": metrics,
    }

    print(f"{'dataset':12s} {'n':>6s} {'EM':>6s} {'F1':>6s} {'ACC':>6s} {'abort':>6s} {'calls':>6s}")
    for name, m in metrics.items():
        n = m.get("n", m.get("n_datasets"))
        print(f"{name:12s} {n:6d} {m['em']:6.3f} {m['f1']:6.3f} {m['acc']:6.3f} {m['abort_rate']:6.3f} {m['mean_searches']:6.2f}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(a.out, "per_example.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {a.out}/metrics.json and per_example.jsonl")


if __name__ == "__main__":
    main()