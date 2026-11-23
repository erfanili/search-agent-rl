#!/usr/bin/env python
"""
Export the project's results as tidy CSVs for plotting (the results dashboard).

  .venv/bin/python scripts/export_results.py --out <dir>

Writes:
  train_steps.csv  arm, step, reward, entropy, response_length, response_clip, num_turns, kl_loss, step_time
                   (verl per-step training metrics from each run's train.log)
  val_steps.csv    arm, step, dataset, em, f1, answered, n_search, n_distinct_search, evidence_hit, correct_multi
                   (verl validation on the sub300 sets, steps 0/20/40/60)
  e0_curves.csv    model, k, dataset, n, em, f1, answered, searches, overridden  (E0: exactly k searches forced;
                   searches = executed searches per question; overridden = share of forced turns in which the
                   model had written the other tag or none)
  e0_steps.csv     model, dataset, k, diff, se  (paired EM difference V(k) - V(k-1) on the same questions)
  queries.csv      model, dataset, n, verbatim, overlap  (E0 k=1 queries: share identical to the question after
                   normalization; mean share of question words reused in the query)
  overlap.csv      dataset, outcome, n  (E0 k=1, the same questions: right for both, only ours, only Search-R1, neither)
  harness.csv      model, dataset, em  (free-depth EM through the baseline harness; paper numbers from
                   arXiv 2503.09516v5 Table 3)
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent.paths import DATA  # noqa: E402
from agent.qa_metrics import SEARCH_RE, normalize_answer, strip_environment_text  # noqa: E402

LOGS = os.path.join(DATA, "logs")
ARMS = {
    "em": "grpo_qwen2.5-3b_full_20261019_183501",
    "em_hit": "grpo_qwen2.5-3b_full_em_hit_2026*",
    "em_multi": "grpo_qwen2.5-3b_full_em_multi_2026*",
    "f1": "grpo_qwen2.5-3b_full_f1_2026*",
}
MODELS = {  # eval dir tag -> short name
    "Qwen2.5-3B": "Base (step 0)",
    "SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo": "Search-R1-3B",
    "grpo_qwen2.5-3b_full_20261019_183501_global_step_60": "Ours (step 60)",
}
PAPER = {"hotpotqa": 0.297, "2wiki": 0.274, "musique": 0.066}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
VAL = re.compile(r"'?val-aux/(\w+)/(\w+)/mean@1'?:\s*([0-9.eE+-]+)")
TRAIN_KEYS = {
    "reward": "critic/score/mean", "entropy": "actor/entropy", "response_length": "response_length/mean",
    "response_clip": "response_length/clip_ratio", "num_turns": "num_turns/mean", "kl_loss": "actor/kl_loss",
    "step_time": "timing_s/step",
}
VAL_METRICS = ["em", "f1", "answered", "n_search", "n_distinct_search", "evidence_hit", "correct_multi"]


def write(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 5) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    print(f"{path}: {len(rows)} rows")


def resolve(pattern):
    hits = sorted(glob.glob(os.path.join(LOGS, pattern)))
    return hits[-1] if hits else None


def parse_train_log(path):
    text = ANSI.sub("", open(path, errors="ignore").read())
    train, val = [], {}
    i = text.find("Initial validation metrics")
    if i >= 0:
        val[0] = {(d, m): float(v) for d, m, v in VAL.findall(text[i:i + 50000].split("step:")[0])}
    for m in re.finditer(r"step:(\d+) - (.*)", text):
        step, line = int(m.group(1)), m.group(2)
        kv = dict(re.findall(r"([\w/@.-]+):([-0-9.eE+]+)(?: -|$)", line))
        row = {"step": step}
        for name, key in TRAIN_KEYS.items():
            if key in kv:
                row[name] = float(kv[key])
        if len(row) > 1:
            train.append(row)
        v = {(d, mm): float(x) for d, mm, x in VAL.findall(line)}
        if v:
            val[step] = v
    return train, val


def latest_eval_runs():
    runs = {}
    for d in glob.glob(os.path.join(LOGS, "eval_*_exact*")):
        m = re.match(r"^eval_(.+)_exact(\d+)_(\d{8}_\d{6})$", os.path.basename(d))
        if not m or m.group(1) not in MODELS or not os.path.exists(os.path.join(d, "scored", "per_example.jsonl")):
            continue
        key = (MODELS[m.group(1)], int(m.group(2)))
        if key not in runs or m.group(3) > runs[key][0]:
            runs[key] = (m.group(3), d)
    return {k: v[1] for k, v in runs.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train_rows, val_rows = [], []
    for arm, pat in ARMS.items():
        run = resolve(pat)
        if not run:
            continue
        train, val = parse_train_log(os.path.join(run, "train.log"))
        train_rows += [{"arm": arm, **r} for r in train]
        for step, v in sorted(val.items()):
            for ds in sorted({d for d, _ in v}):
                val_rows.append({"arm": arm, "step": step, "dataset": ds, **{m: v.get((ds, m)) for m in VAL_METRICS}})
    write(os.path.join(a.out, "train_steps.csv"), train_rows, ["arm", "step", *TRAIN_KEYS])
    write(os.path.join(a.out, "val_steps.csv"), val_rows, ["arm", "step", "dataset", *VAL_METRICS])

    runs = latest_eval_runs()
    per = {}
    curves = []
    for (model, k), d in sorted(runs.items()):
        ex = {(r["dataset"], r["id"]): r for r in map(json.loads, open(os.path.join(d, "scored", "per_example.jsonl")))}
        preds = {(r["dataset"], r["id"]): r for r in map(json.loads, open(os.path.join(d, "preds.jsonl")))}
        per[(model, k)] = (ex, preds)
        for ds in sorted({x for x, _ in ex}):
            keys = [key for key in ex if key[0] == ds]
            n = len(keys)
            forced = sum(preds[key].get("n_forced", 0) for key in keys)
            curves.append({
                "model": model, "k": k, "dataset": ds, "n": n,
                "em": sum(ex[key]["em"] for key in keys) / n, "f1": sum(ex[key]["f1"] for key in keys) / n,
                "answered": sum(ex[key]["answered"] for key in keys) / n,
                "searches": sum(preds[key].get("n_search", 0) for key in keys) / n,
                "overridden": sum(preds[key].get("n_overridden", 0) for key in keys) / forced if forced else None,
            })
    write(os.path.join(a.out, "e0_curves.csv"), curves,
          ["model", "k", "dataset", "n", "em", "f1", "answered", "searches", "overridden"])

    steps = []
    for (model, k), (ex, _) in sorted(per.items()):
        if (model, k - 1) not in per:
            continue
        prev = per[(model, k - 1)][0]
        for ds in sorted({x for x, _ in ex}):
            common = [key for key in ex if key[0] == ds and key in prev]
            if len(common) < 2:
                continue
            diffs = [ex[key]["em"] - prev[key]["em"] for key in common]
            mu = sum(diffs) / len(diffs)
            se = math.sqrt(sum((x - mu) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
            steps.append({"model": model, "dataset": ds, "k": k, "diff": mu, "se": se})
    write(os.path.join(a.out, "e0_steps.csv"), steps, ["model", "dataset", "k", "diff", "se"])

    qrows = []
    for (model, k), (ex, preds) in sorted(per.items()):
        if k != 1:
            continue
        gold = {}
        for ds in {x for x, _ in ex}:
            for split in ("sub300",):
                path = os.path.join(DATA, "eval", ds, f"{split}.jsonl")
                if os.path.exists(path):
                    for r in map(json.loads, open(path)):
                        gold[(ds, r["id"])] = r["question"]
        by = defaultdict(list)
        for key, pr in preds.items():
            qs = SEARCH_RE.findall(strip_environment_text(pr["output"]))
            if not qs or key not in gold:
                continue
            q, question = normalize_answer(qs[0]), normalize_answer(gold[key])
            qw, sw = set(question.split()), set(q.split())
            by[key[0]].append((float(q == question), len(qw & sw) / max(1, len(qw))))
        for ds, xs in sorted(by.items()):
            qrows.append({"model": model, "dataset": ds, "n": len(xs), "verbatim": sum(x for x, _ in xs) / len(xs),
                          "overlap": sum(y for _, y in xs) / len(xs)})
    write(os.path.join(a.out, "queries.csv"), qrows, ["model", "dataset", "n", "verbatim", "overlap"])

    orows = []
    a_, b_ = per.get(("Ours (step 60)", 1)), per.get(("Search-R1-3B", 1))
    if a_ and b_:
        for ds in sorted({x for x, _ in a_[0]}):
            keys = [key for key in a_[0] if key[0] == ds and key in b_[0]]
            cnt = defaultdict(int)
            for key in keys:
                o, s_ = a_[0][key]["em"] == 1, b_[0][key]["em"] == 1
                cnt["Both right" if o and s_ else "Only ours" if o else "Only Search-R1" if s_ else "Neither"] += 1
            orows += [{"dataset": ds, "outcome": k_, "n": cnt[k_]} for k_ in ("Both right", "Only ours", "Only Search-R1", "Neither")]
    write(os.path.join(a.out, "overlap.csv"), orows, ["dataset", "outcome", "n"])

    hrows = [{"model": "Search-R1-3B (paper)", "dataset": ds, "em": v} for ds, v in PAPER.items()]
    for label, pat in [("Search-R1-3B (our harness)", "baseline_searchr1_3b_2026*"),
                       ("Ours step 60 (our harness)", "eval_grpo_qwen2.5-3b_full_20261019_183501_global_step_60_20261020_104937")]:
        d = resolve(pat)
        m = json.load(open(os.path.join(d, "scored", "metrics.json"))) if d and os.path.exists(os.path.join(d, "scored", "metrics.json")) else None
        if m:
            agg = m.get("metrics", m.get("aggregate", {}))
            for ds in ("hotpotqa", "2wiki", "musique", "bamboogle"):
                if ds in agg:
                    hrows.append({"model": label, "dataset": ds, "em": agg[ds]["em"]})
    write(os.path.join(a.out, "harness.csv"), hrows, ["model", "dataset", "em"])


if __name__ == "__main__":
    main()
