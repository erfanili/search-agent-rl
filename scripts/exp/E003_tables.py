#!/usr/bin/env python
"""E003 signals, probes, tables and per-question CSVs.

  .venv-verl/bin/python scripts/exp/E003_tables.py [--pilot] [--models m1 m2 ...]

Population per model: its own E001 label knowable or unknowable, ambiguous excluded (uncertain excluded).
Positive class = unknowable for every AUROC. Ties count 1/2. CI: 1,000 bootstrap resamples (default_rng(0)),
resampled within each class, percentile interval (E002_tables.auroc_ci). A signal's AUROC uses the questions
where it is defined (n reported).
  S1  E001 samples (seed 0): share of the 5 samples equal to the most common normalize_answer(answer); each
      untagged sample is its own answer. AUROC on -S1.
  S2  same on E003 s2_samples.jsonl (seed 1). AUROC on -S2.
  S3  greedy mean answer-token log-prob (E003_gen.py). AUROC on -S3. Missing when no answer span.
  S4  out-of-fold P(unknowable) of a StandardScaler + L2 LogisticRegressionCV probe (C in logspace(-4,2,7),
      inner StratifiedKFold(5, shuffle, seed 0), scoring roc_auc), outer StratifiedKFold(5, shuffle, seed 0),
      one probe per layer, fit on all questions. Per-source columns: separate probes fit within that source
      only (same nested CV) at the same best / fixed layer; Table B is all-questions probes only.
      Best layer = highest all-question AUROC (selection on the evaluation data). Fixed layer: 24 (36-layer
      models), 20 (28-layer).
  S5  log p(search) - log p(answer) after the "<" of the first tag in the greedy P2 first turn (E003_hidden.py).
  E002  per-question P2 search rate (share of 5 episodes with >= 1 search), $DATA_ROOT/logs/E002/<m>/P2.
  Source-only baseline: same probe on one-hot {PopQA bin 4, PopQA bin 5, TriviaQA}; per-source columns fit
      within source (TriviaQA: constant features, AUROC 0.5 by construction).
Outputs ($DATA_ROOT/logs/E003[/pilot]/): <model>/signals.csv, <model>/probe_best.npz, <model>/probe_fixed.npz,
<model>/probe_layers.json, e003_tables.md
"""
import argparse, collections, csv, json, os, sys

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA  # noqa: E402
from agent.qa_metrics import normalize_answer  # noqa: E402
from E001_prompts import MODELS  # noqa: E402
from E002_tables import auroc, auroc_ci  # noqa: E402

FIXED = {36: 24, 28: 20}
E002_CARD_P2 = {"qwen3b-base": 0.579, "qwen3b-instruct": 0.587, "searchr1-3b": 0.575, "grpo-e1a-s60": 0.559,
                "qwen7b-instruct": 0.620}  # E002 card, Table B, ambiguous excluded, P2


def agreement(samples):
    keys = [normalize_answer(s["answer"]) if s["parse"] != "untagged" else f"__untagged_{k}" for k, s in enumerate(samples)]
    return collections.Counter(keys).most_common(1)[0][1] / len(keys)


def probe():
    inner = StratifiedKFold(5, shuffle=True, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=inner, scoring="roc_auc",
                                                                penalty="l2", max_iter=5000, n_jobs=8))


def oof(X, y):
    p = np.zeros(len(y))
    if np.all(X.std(0) == 0):  # no input variation: a constant score (AUROC 0.5), not per-fold intercept offsets
        return p
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p[te] = probe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def save_probe(path, X, y, layer):
    m = probe().fit(X, y)
    sc, lr = m[0], m[1]
    np.savez(path, layer=layer, coef=lr.coef_[0], intercept=lr.intercept_, C=lr.C_, scaler_mean=sc.mean_,
             scaler_scale=sc.scale_, positive_class="unknowable")


def cell(score, y, mask):
    s, yy = score[mask], y[mask]
    ok = ~np.isnan(s)
    s, yy = s[ok], yy[ok]
    pos, neg = s[yy == 1], s[yy == 0]
    a = auroc(pos, neg)
    lo, hi = auroc_ci(pos, neg)
    return a, lo, hi, len(pos), len(neg)


def fmt(c):
    a, lo, hi, npos, nneg = c
    return "—" if a != a else f"{a:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    base = os.path.join(DATA, "logs", "E003", "pilot" if a.pilot else "")
    lab = {r["id"]: r for r in map(json.loads, open(os.path.join(DATA, "eval", "e001", "e001_easy_1k_labels.jsonl")))}
    load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
    res, layer_auc, base_auc, notes = {}, {}, {}, []

    for m in a.models:
        d = os.path.join(base, m)
        s2, s3, s5 = load(f"{d}/s2_samples.jsonl"), load(f"{d}/s3_greedy.jsonl"), load(f"{d}/s5_margin.jsonl")
        e1 = load(os.path.join(DATA, "logs", "E001", f"E001-{m}-s0", "samples.jsonl"))
        e2 = load(os.path.join(DATA, "logs", "E002", m, "P2", "samples.jsonl"))
        hz = np.load(f"{d}/hidden.npz")
        hid = {i: k for k, i in enumerate(hz["ids"].tolist())}
        ids = [i for i in hz["ids"].tolist() if not lab[i]["ambiguous"] and lab[i]["labels"][m] in ("knowable", "unknowable")]
        y = np.array([lab[i]["labels"][m] == "unknowable" for i in ids], int)
        src = np.array([lab[i]["source"] for i in ids])
        nan = float("nan")
        S1 = np.array([agreement(e1[i]["samples"]) for i in ids])
        S2 = np.array([agreement(s2[i]["samples"]) for i in ids])
        S3 = np.array([s3[i]["s3"] if s3[i]["s3"] is not None else nan for i in ids])
        S5 = np.array([s5[i]["s5"] if s5[i]["s5"] is not None else nan for i in ids])
        E2 = np.array([sum(s["n_search"] > 0 for s in e2[i]["samples"]) / len(e2[i]["samples"]) for i in ids])

        layers = hz["layers"].tolist()
        H = hz["H"][[hid[i] for i in ids]]
        P = {l: oof(H[:, k], y) for k, l in enumerate(layers)}
        layer_auc[m] = {l: auroc(P[l][y == 1], P[l][y == 0]) for l in layers}
        best = max(layers, key=lambda l: layer_auc[m][l])
        fixed = FIXED[max(layers)]
        json.dump({"layers": layers, "oof_auroc": layer_auc[m], "best": best, "fixed": fixed}, open(f"{d}/probe_layers.json", "w"), indent=2)
        save_probe(f"{d}/probe_best.npz", H[:, layers.index(best)], y, best)
        save_probe(f"{d}/probe_fixed.npz", H[:, layers.index(fixed)], y, fixed)
        B = np.array([[lab[i]["source"] == "popqa" and lab[i]["bin"] == 4, lab[i]["source"] == "popqa" and lab[i]["bin"] == 5,
                       lab[i]["source"] == "triviaqa"] for i in ids], float)
        sig = {"S1": -S1, "S2": -S2, "S3": -S3, "S5": S5, "E002": E2}
        masks = {"all": np.ones(len(ids), bool), "popqa": src == "popqa", "triviaqa": src == "triviaqa"}
        res[m] = {(k, sc): cell(v, y, mk) for k, v in sig.items() for sc, mk in masks.items()}
        res[m]["best"], res[m]["fixed"] = best, fixed
        # S4 and the source-only baseline: the all-questions column uses the all-questions probe; each per-source
        # column uses probes fit within that source only (same nested CV), at the same best / fixed layer
        # (Erfan, 2026-11-21: out-of-fold predictions pooled across folds carry between-fold offsets).
        P_src = {}
        for sc, mk in masks.items():
            yy = y[mk]
            if sc == "all":
                p_best, p_fixed, p_base = P[best], P[fixed], oof(B, y)
            else:
                p_best, p_fixed = oof(H[mk][:, layers.index(best)], yy), oof(H[mk][:, layers.index(fixed)], yy)
                p_base = oof(B[mk], yy)
            P_src[sc] = (p_best, p_fixed)
            allm = np.ones(mk.sum(), bool)
            res[m][("S4best", sc)] = cell(p_best, yy, allm)
            res[m][("S4fixed", sc)] = cell(p_fixed, yy, allm)
            base_auc.setdefault(m, {})[sc] = cell(p_base, yy, allm)
        e2a = res[m][("E002", "all")][0]
        if not a.pilot and round(e2a, 3) != E002_CARD_P2[m]:
            notes.append(f"{m}: recomputed E002 P2 AUROC {e2a:.3f} != card {E002_CARD_P2[m]:.3f}")
        notes.append(f"{m}: n = {len(ids)} ({y.sum()} unknowable, {len(y) - y.sum()} knowable); S3 missing "
                     f"{int(np.isnan(S3).sum())}; S5 missing {int(np.isnan(S5).sum())}")

        Wb, Wf = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
        for sc in ("popqa", "triviaqa"):
            Wb[masks[sc]], Wf[masks[sc]] = P_src[sc]
        with open(f"{d}/signals.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "label", "source", "bin", "S1", "S2", "S3", f"S4_best_L{best}", f"S4_fixed_L{fixed}", "S5",
                        "e002_p2_search_rate", f"S4_best_L{best}_within_source", f"S4_fixed_L{fixed}_within_source"])
            for k, i in enumerate(ids):
                w.writerow([i, lab[i]["labels"][m], lab[i]["source"], lab[i]["bin"], S1[k], S2[k],
                            "" if S3[k] != S3[k] else S3[k], P[best][k], P[fixed][k], "" if S5[k] != S5[k] else S5[k], E2[k], Wb[k], Wf[k]])

    L = []
    L.append("AUROC, unknowable = positive; 95% bootstrap CI. Own E001 labels, ambiguous and uncertain excluded.\n")
    for sc, name in (("all", "all questions"), ("popqa", "PopQA only"), ("triviaqa", "TriviaQA only")):
        L.append(f"### Table A ({name})\n")
        L.append("| model | n unk / know | S1 in-sample consistency | S2 held-out consistency | S3 answer log-prob | S4 best layer | "
                 "S4 fixed layer | S5 margin | E002 P2 search rate |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for m in a.models:
            r = res[m]
            n = r[("S1", sc)]
            L.append(f"| {m} | {n[3]} / {n[4]} | {fmt(r[('S1', sc)])} | {fmt(r[('S2', sc)])} | {fmt(r[('S3', sc)])} | "
                     f"{fmt(r[('S4best', sc)])} (L{r['best']}) | {fmt(r[('S4fixed', sc)])} (L{r['fixed']}) | "
                     f"{fmt(r[('S5', sc)])} | {fmt(r[('E002', sc)])} |")
        L.append("")
    L.append("### Table B: S4 out-of-fold AUROC by layer (all questions)\n")
    L.append("| layer | " + " | ".join(a.models) + " |")
    L.append("|---|" + "---|" * len(a.models))
    for l in sorted({l for m in a.models for l in layer_auc[m]}):
        L.append(f"| {l} | " + " | ".join(f"{layer_auc[m][l]:.3f}" if l in layer_auc[m] else "—" for m in a.models) + " |")
    L.append("\n### Table C: source-only baseline probe AUROC (features: PopQA bin 4, PopQA bin 5, TriviaQA)\n")
    L.append("| model | all | PopQA only | TriviaQA only |")
    L.append("|---|---|---|---|")
    for m in a.models:
        L.append(f"| {m} | " + " | ".join(fmt(base_auc[m][sc]) for sc in ("all", "popqa", "triviaqa")) + " |")
    L.append("\nNotes:\n" + "\n".join(f"- {x}" for x in notes))
    txt = "\n".join(L)
    open(os.path.join(base, "e003_tables.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
