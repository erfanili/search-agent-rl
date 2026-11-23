#!/usr/bin/env python
"""E001 (revision 2): untagged classification, ambiguous flags, label files and tables.

  .venv/bin/python scripts/exp/E001_tables.py [--pilot] [--no-write]

Reads $DATA_ROOT/logs/E001[/pilot2]/E001-<model>-s0/samples.jsonl (raw, never modified).
Untagged samples (no <answer>) are classified with CLARIFY_RE (case-insensitive) on the sample text:
  clarification = asks for context/details or calls the question ambiguous/incomplete; otherwise other_untagged.
ambiguous (per question) = any model has >= 1 clarification sample on it.
Writes (unless --no-write):
  <run dir>/labels.jsonl        id, n_correct, n_em, n_untagged, n_clarification, n_other_untagged, label, ambiguous
  <run dir>/untagged.jsonl      id, sample, class, text  (every untagged sample)
  $DATA_ROOT/eval/e001/e001_easy_1k_labels.jsonl  (full run only) id, source, bin, ambiguous, clarification_models,
                                                  labels {model: label}, n_correct {model: n}
Tables: A labels/parsing by source (all questions and ambiguous excluded); B accuracy and EM by group ± SE;
C share of questions knowable to both, per model pair (all questions).
"""
import argparse, json, math, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA  # noqa: E402
from E001_prompts import MODELS  # noqa: E402

K = 3  # knowable threshold
CLARIFY_RE = re.compile(
    r"more (context|information|details?|specific)|additional (context|information|details?)"
    r"|enough (context|information)|\bspecify\b|\bspecified\b|clarif|ambiguous|incomplete"
    r"|(could|can|may|might) refer to|referring to|which (one|version|work)", re.I)


def label(n):
    return "knowable" if n >= K else "unknowable" if n == 0 else "uncertain"


def mean_se(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def uclass(s):
    if s["parse"] != "untagged":
        return None
    return "clarification" if CLARIFY_RE.search(s["text"]) else "other_untagged"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--pilot", action="store_true"); p.add_argument("--no-write", action="store_true")
    a = p.parse_args()
    base = os.path.join(DATA, "logs", "E001", "pilot2" if a.pilot else "")
    meta = json.load(open(os.path.join(DATA, "eval", "e001", "e001_easy_1k.meta.json")))
    data = {m: [json.loads(l) for l in open(f)] for m in MODELS
            if os.path.exists(f := os.path.join(base, f"E001-{m}-s0", "samples.jsonl"))}
    models = list(data)
    ids = [r["id"] for r in data[models[0]]]
    assert all([r["id"] for r in data[m]] == ids for m in models)
    clar_models = {i: [] for i in ids}
    for m in models:
        for r in data[m]:
            if any(uclass(s) == "clarification" for s in r["samples"]):
                clar_models[r["id"]].append(m)
    amb = {i: bool(v) for i, v in clar_models.items()}
    nc = {m: {r["id"]: sum(s["correct"] for s in r["samples"]) for r in data[m]} for m in models}

    if not a.no_write:
        for m in models:
            d = os.path.join(base, f"E001-{m}-s0")
            with open(f"{d}/labels.jsonl", "w") as fl, open(f"{d}/untagged.jsonl", "w") as fu:
                for r in data[m]:
                    cls = [uclass(s) for s in r["samples"]]
                    fl.write(json.dumps({"id": r["id"], "n_correct": nc[m][r["id"]], "n_em": sum(s["em"] for s in r["samples"]),
                                         "n_untagged": sum(c is not None for c in cls),
                                         "n_clarification": cls.count("clarification"), "n_other_untagged": cls.count("other_untagged"),
                                         "label": label(nc[m][r["id"]]), "ambiguous": amb[r["id"]]}) + "\n")
                    for k, (s, c) in enumerate(zip(r["samples"], cls)):
                        if c:
                            fu.write(json.dumps({"id": r["id"], "sample": k, "class": c, "text": s["text"]}) + "\n")
        if not a.pilot:
            with open(os.path.join(DATA, "eval", "e001", "e001_easy_1k_labels.jsonl"), "w") as f:
                for r in data[models[0]]:
                    i = r["id"]
                    f.write(json.dumps({"id": i, "source": r["source"], "bin": r["bin"], "ambiguous": amb[i],
                                        "clarification_models": clar_models[i], "labels": {m: label(nc[m][i]) for m in models},
                                        "n_correct": {m: nc[m][i] for m in models}}) + "\n")

    print(f"ambiguous questions: {sum(amb.values())} / {len(ids)} "
          f"(PopQA {sum(amb[r['id']] for r in data[models[0]] if r['source'] == 'popqa')}, "
          f"TriviaQA {sum(amb[r['id']] for r in data[models[0]] if r['source'] == 'triviaqa')})\n")
    print("Table A1: parsing, by source (all questions)\n")
    print("| model | source | n | parse rate | untagged | clarification | other untagged | open, not closed | token-limit hits |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in models:
        for src in ("popqa", "triviaqa", "all"):
            S = [s for r in data[m] if src == "all" or r["source"] == src for s in r["samples"]]
            cl = [uclass(s) for s in S]
            unt = sum(c is not None for c in cl)
            print(f"| {m} | {src} | {len(S) // 5} | {1 - unt / len(S):.3f} | {unt}/{len(S)} | {cl.count('clarification')} | "
                  f"{cl.count('other_untagged')} | {sum(s['parse'] == 'open' for s in S)} | {sum(s['finish_reason'] == 'length' for s in S)} |")
    for title, keep in (("all questions", lambda i: True), ("ambiguous questions excluded", lambda i: not amb[i])):
        print(f"\nTable A2: knowable / unknowable / uncertain, {title}\n")
        print("| model | source | n | knowable | unknowable | uncertain |"); print("|---|---|---|---|---|---|")
        for m in models:
            for src in ("popqa", "triviaqa", "all"):
                c = [nc[m][r["id"]] for r in data[m] if (src == "all" or r["source"] == src) and keep(r["id"])]
                print(f"| {m} | {src} | {len(c)} | {sum(x >= K for x in c)} | {sum(x == 0 for x in c)} | {sum(0 < x < K for x in c)} |")

    groups = [("PopQA bin 4", lambda r: r["source"] == "popqa" and r["bin"] == 4),
              ("PopQA bin 5", lambda r: r["source"] == "popqa" and r["bin"] == 5),
              ("TriviaQA", lambda r: r["source"] == "triviaqa"), ("all", lambda r: True)]
    rng = meta["popqa"]["spop_range_sampled"]
    for metric, name in (("correct", "accuracy (alias in answer)"), ("em", "EM")):
        print(f"\nTable B: closed-book {name}, %, mean over 5 samples ± SE over questions (all questions)\n")
        print("| group | s_pop range | n | " + " | ".join(models) + " |"); print("|---|---|---|" + "---|" * len(models))
        for g, f in groups:
            cells = []
            for m in models:
                mu, se = mean_se([sum(s[metric] for s in r["samples"]) / 5 for r in data[m] if f(r)])
                cells.append(f"{100 * mu:.1f} ± {100 * se:.1f}")
            r_ = f"{rng['4'][0]}–{rng['4'][1]}" if g.endswith("4") else f"{rng['5'][0]}–{rng['5'][1]}" if g.endswith("5") else ""
            print(f"| {g} | {r_} | {sum(1 for r in data[models[0]] if f(r))} | " + " | ".join(cells) + " |")

    know = {m: {i for i in ids if nc[m][i] >= K} for m in models}
    print("\nTable C: share of questions knowable to both (all questions; diagonal = share knowable)\n")
    print("| | " + " | ".join(models) + " |"); print("|---|" + "---|" * len(models))
    for x in models:
        print(f"| {x} | " + " | ".join(f"{len(know[x] & know[y]) / len(ids):.3f}" for y in models) + " |")


if __name__ == "__main__":
    main()
