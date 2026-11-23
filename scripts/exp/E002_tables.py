#!/usr/bin/env python
"""E002 tables and tidy CSV.

  .venv/bin/python scripts/exp/E002_tables.py [--pilot]

Reads $DATA_ROOT/logs/E002[/pilot]/<model>/<cond>/samples.jsonl and the E001 labels
($DATA_ROOT/eval/e001/e001_easy_1k_labels.jsonl). Labels are each model's OWN E001 label (Erfan, 2026-11-21);
the shared knowable set (knowable for all four 3B models) is used for Table E only.
Every table is computed twice: ambiguous questions excluded, and all questions.

Definitions
  search episode      an episode with n_search >= 1 (executed searches)
  per-question search rate   share of its 5 episodes with n_search >= 1
  Table A  P(>=1 search | label) = share of episodes, pooled over the questions with that label; mean searches per
           episode; parse rate (episodes with an <answer> tag); cap-hit rate (4,096-token cap); malformed episodes
           (>= 1 free turn of any malformed type).
  Table B  AUROC of the per-question search rate, unknowable (positive) vs knowable (negative); ties count 1/2.
           95% CI: 1,000 bootstrap resamples (numpy default_rng(0)), questions resampled with replacement within
           each label class; percentile interval.
  Table C  accuracy (cover match) = mean over questions of n_correct/5, ± SE over questions; overall (all labels),
           knowable, unknowable.
  Table D  P2 forced depth (exact:0 vs exact:1): accuracy on knowable and on unknowable ± SE; paired sign test on
           knowable questions: per-question accuracy f0 vs f1, ties dropped, two-sided exact binomial p.
  Table E  P(>=1 search | knowable) restricted to the shared 3B knowable set, 3B models.
Tidy CSV ($DATA_ROOT/logs/E002[/pilot]/e002_tidy.csv): one row per question x model x condition (incl. P2-f0/f1).
"""
import argparse, csv, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA  # noqa: E402
from E001_prompts import MODELS  # noqa: E402

CONDS = ["P1", "P2", "P3", "P4", "P5", "P6"]
FORCED = ["P2-f0", "P2-f1"]
M3B = ["qwen3b-base", "qwen3b-instruct", "searchr1-3b", "grpo-e1a-s60"]
TYPES = ("empty_query", "unclosed_search", "multi_search", "search_and_answer")


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def auroc_ci(pos, neg, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    bs = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def sign_test(diffs):
    n_pos, n_neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = n_pos + n_neg
    if n == 0:
        return n_pos, n_neg, 1.0
    k = min(n_pos, n_neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n_pos, n_neg, min(1.0, 2 * p)


def f(x, d=3):
    return "—" if x != x else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true"); a = ap.parse_args()
    base = os.path.join(DATA, "logs", "E002", "pilot" if a.pilot else "")
    lab = {r["id"]: r for r in map(json.loads, open(os.path.join(DATA, "eval", "e001", "e001_easy_1k_labels.jsonl")))}
    shared = {i for i, r in lab.items() if all(r["labels"][m] == "knowable" for m in M3B)}

    data = {}  # (model, cond) -> {id: question record}
    for m in MODELS:
        for c in CONDS + FORCED:
            p = os.path.join(base, m, c, "samples.jsonl")
            if os.path.exists(p):
                data[(m, c)] = {q["id"]: q for q in map(json.loads, open(p))}

    # tidy CSV
    with open(os.path.join(base, "e002_tidy.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source", "bin", "model", "condition", "depth", "e001_label", "ambiguous", "shared_3b_knowable",
                    "search_rate", "mean_searches", "n_correct", "accuracy", "n_em", "n_untagged", "n_cap_hit",
                    "n_malformed_episodes", "n_forced_answer"])
        for (m, c), qs in data.items():
            for i, q in qs.items():
                S = q["samples"]
                w.writerow([i, q["source"], q["bin"], m, c.split("-")[0], c.split("-")[1] if "-" in c else "free",
                            lab[i]["labels"][m], lab[i]["ambiguous"], i in shared,
                            sum(s["n_search"] > 0 for s in S) / len(S), sum(s["n_search"] for s in S) / len(S),
                            sum(s["correct"] for s in S), sum(s["correct"] for s in S) / len(S), sum(s["em"] for s in S),
                            sum(s["parse"] == "untagged" for s in S), sum(s["cap_hit"] for s in S),
                            sum(any(s["malformed"][t] for t in TYPES) for s in S),
                            sum(s["n_forced"] > 0 for s in S)])  # free cells: max:3 reached; forced cells: every episode

    for scope, keep in (("ambiguous questions excluded", lambda i: not lab[i]["ambiguous"]), ("all questions", lambda i: True)):
        print(f"\n## {scope}\n")
        print("### Table A: search rates by E001 label (own model's labels)\n")
        print("| model | cond | n know / unk / unc | P(≥1 search ∣ knowable) | P(≥1 search ∣ unknowable) | P(≥1 search ∣ uncertain) "
              "| mean searches | parse rate | cap-hit | malformed eps |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for m in MODELS:
            for c in CONDS:
                if (m, c) not in data:
                    continue
                qs = {i: q for i, q in data[(m, c)].items() if keep(i)}
                by = {L: [s for i, q in qs.items() if lab[i]["labels"][m] == L for s in q["samples"]] for L in ("knowable", "unknowable", "uncertain")}
                S = [s for q in qs.values() for s in q["samples"]]
                ns = {L: sum(1 for i in qs if lab[i]["labels"][m] == L) for L in by}
                pr = lambda L: sum(s["n_search"] > 0 for s in by[L]) / len(by[L]) if by[L] else float("nan")
                print(f"| {m} | {c} | {ns['knowable']} / {ns['unknowable']} / {ns['uncertain']} | {f(pr('knowable'))} | "
                      f"{f(pr('unknowable'))} | {f(pr('uncertain'))} | {sum(s['n_search'] for s in S) / len(S):.2f} | "
                      f"{1 - sum(s['parse'] == 'untagged' for s in S) / len(S):.3f} | {sum(s['cap_hit'] for s in S) / len(S):.3f} | "
                      f"{sum(any(s['malformed'][t] for t in TYPES) for s in S) / len(S):.3f} |")

        print("\n### Table B: AUROC of per-question search rate, unknowable vs knowable (95% bootstrap CI)\n")
        print("| model | " + " | ".join(CONDS) + " |"); print("|---|" + "---|" * len(CONDS))
        for m in MODELS:
            cells = []
            for c in CONDS:
                if (m, c) not in data:
                    cells.append("—"); continue
                qs = {i: q for i, q in data[(m, c)].items() if keep(i)}
                rate = {i: sum(s["n_search"] > 0 for s in q["samples"]) / len(q["samples"]) for i, q in qs.items()}
                pos = [rate[i] for i in qs if lab[i]["labels"][m] == "unknowable"]
                neg = [rate[i] for i in qs if lab[i]["labels"][m] == "knowable"]
                lo, hi = auroc_ci(pos, neg)
                cells.append(f"{f(auroc(pos, neg))} [{f(lo)}, {f(hi)}]")
            print(f"| {m} | " + " | ".join(cells) + " |")

        print("\n### Table C: accuracy (alias in answer), % ± SE over questions\n")
        print("| model | cond | overall | knowable | unknowable |"); print("|---|---|---|---|---|")
        for m in MODELS:
            for c in CONDS:
                if (m, c) not in data:
                    continue
                qs = {i: q for i, q in data[(m, c)].items() if keep(i)}
                acc = {i: sum(s["correct"] for s in q["samples"]) / len(q["samples"]) for i, q in qs.items()}
                cell = lambda xs: "{:.1f} ± {:.1f}".format(*(100 * v for v in mean_se(xs)))
                print(f"| {m} | {c} | {cell(list(acc.values()))} | {cell([acc[i] for i in acc if lab[i]['labels'][m] == 'knowable'])} | "
                      f"{cell([acc[i] for i in acc if lab[i]['labels'][m] == 'unknowable'])} |")

        print("\n### Table D: P2 forced depth, accuracy % ± SE (forced 0 vs forced 1 searches)\n")
        print("| model | knowable n | knowable f0 | knowable f1 | sign test knowable (f1>f0 / f1<f0, p) | unknowable n | unknowable f0 | unknowable f1 |")
        print("|---|---|---|---|---|---|---|---|")
        for m in MODELS:
            if (m, "P2-f0") not in data or (m, "P2-f1") not in data:
                continue
            q0, q1 = data[(m, "P2-f0")], data[(m, "P2-f1")]
            ids = [i for i in q0 if i in q1 and keep(i)]
            acc = lambda q, i: sum(s["correct"] for s in q[i]["samples"]) / len(q[i]["samples"])
            kn = [i for i in ids if lab[i]["labels"][m] == "knowable"]
            un = [i for i in ids if lab[i]["labels"][m] == "unknowable"]
            cell = lambda xs: "{:.1f} ± {:.1f}".format(*(100 * v for v in mean_se(xs)))
            npos, nneg, p = sign_test([acc(q1, i) - acc(q0, i) for i in kn])
            print(f"| {m} | {len(kn)} | {cell([acc(q0, i) for i in kn])} | {cell([acc(q1, i) for i in kn])} | {npos} / {nneg}, p = {p:.3g} | "
                  f"{len(un)} | {cell([acc(q0, i) for i in un])} | {cell([acc(q1, i) for i in un])} |")

        print(f"\n### Table E: P(≥1 search ∣ knowable) on the shared 3B knowable set\n")
        print("| model | n | " + " | ".join(CONDS) + " |"); print("|---|---|" + "---|" * len(CONDS))
        for m in M3B:
            ids = [i for i in shared if keep(i)]
            cells = []
            for c in CONDS:
                if (m, c) not in data:
                    cells.append("—"); continue
                S = [s for i in ids if i in data[(m, c)] for s in data[(m, c)][i]["samples"]]
                cells.append(f(sum(s["n_search"] > 0 for s in S) / len(S)) if S else "—")
            n = len([i for i in ids if (m, CONDS[0]) in data and i in data[(m, CONDS[0])]])
            print(f"| {m} | {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
