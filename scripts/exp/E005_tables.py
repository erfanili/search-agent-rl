#!/usr/bin/env python
"""E005 tables A-F, tidy CSV and plot.

  .venv-verl/bin/python scripts/exp/E005_tables.py [--runs E005-O-s0 E005-S-s0 E005-C-s0]

Questions: the E001 set, ambiguous excluded. Labels: qwen3b-instruct's E001 labels (knowable / unknowable / uncertain);
AUROC positive class = unknowable, knowable vs unknowable only. Step 0 = start model: E002 rollouts
(logs/E002/qwen3b-instruct/{P1,P2,P6}), E003 S5 margin and hidden states (logs/E003/qwen3b-instruct).
Evaluation cells (E004_eval.py --exp E005): per question 5 episodes, T=0.7.
  P(>=1 search | label) = share of episodes with n_search >= 1 (executed searches); AUROC of the per-question search
  rate, 1,000-resample bootstrap 95% CI (E002_tables.auroc_ci); accuracy = mean over questions of (correct / 5) +- SE
  over questions, under the E001 rule (cover match on the first <answer> span, unclosed -> to the end; the samples'
  "correct") and under alias_match_strict (= sar-alias-reward agent/reward.py: first span closed AND normalized alias
  equals the span or is a substring of it with span words <= alias words + 3); overall = all non-ambiguous questions.
  Mean answer words = whitespace words of the first <answer> span (environment text stripped; unclosed -> to the end)
  over episodes with an <answer>.
Table C: S5 margin AUROC; S4 = out-of-fold layer-24 probe AUROC refit per checkpoint (E003 method, E003_tables.oof).
  k = the start model's out-of-fold layer-24 probe prediction from E003 (P(unknowable); signals.csv S4_fixed_L24 for
  knowable / unknowable; uncertain questions = mean of the 5 E003 fold models' predictions, the folds rebuilt with
  E003's StratifiedKFold(5, seed 0) on the same rows; the rebuilt out-of-fold values are checked against signals.csv).
  k coefficient: logistic regression (no penalty, sklearn lbfgs) of each P2 episode's search decision (n_search >= 1)
  on [unknowable indicator, k], knowable + unknowable questions; 95% CI from 1,000 question-level bootstrap resamples
  (default_rng(0)); resamples where the fit fails or the outcome is constant are dropped (count reported).
  Spearman(S5 margin, k) within knowable and within unknowable (questions with a margin).
Table D: reward_steps.jsonl, steps 1, 10, ..., 100, by e004_label ("components" + "answer" blocks).
Table E: uncertain questions (E001 label uncertain, non-ambiguous), P2 search rate (share of episodes) by the start
  model's E003 S2 agreement: bins 0.2-0.4, 0.6, 0.8-1.0; steps 0 and the arm's last evaluated step.
Table F: relabel samples (E001 procedure on the arm's last evaluated step) vs the start model's E001 samples: share
  of closed-book samples whose text contains "<search>", share with no "<answer>", mean answer-span words; label counts.
Outputs: $DATA_ROOT/logs/E005/{e005_tables.md, e005_tidy.csv, e005_plot.png, k_scores.jsonl}
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA  # noqa: E402
from agent.qa_metrics import normalize_answer, strip_environment_text  # noqa: E402
from E002_tables import auroc, auroc_ci, mean_se  # noqa: E402
from E003_tables import agreement, oof, probe  # noqa: E402

M = "qwen3b-instruct"
STEPS = [0, 20, 40, 60, 80, 100]
PROMPTS = ["P2", "P1", "P6"]
B = os.path.join(DATA, "logs", "E005")
load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
nan = float("nan")


def strict(text, golds):  # = sar-alias-reward agent/reward.py:alias_match_strict
    t = strip_environment_text(text)
    i = t.find("<answer>")
    if i < 0:
        return 0
    rest = t[i + len("<answer>"):]
    j = rest.find("</answer>")
    if j < 0:
        return 0
    span = normalize_answer(rest[:j])
    n = len(span.split())
    for g in map(normalize_answer, golds):
        if g and (g == span or (g in span and n <= len(g.split()) + 3)):
            return 1
    return 0


def span_words(text):
    t = strip_environment_text(text)
    i = t.find("<answer>")
    if i < 0:
        return None
    rest = t[i + len("<answer>"):]
    j = rest.find("</answer>")
    return len((rest[:j] if j >= 0 else rest).split())


def cell_dir(run, step):
    if step == 0:
        return None
    if run.startswith("E004"):
        return os.path.join(DATA, "logs", "E004", "eval", run, f"step{step}")
    return os.path.join(B, "eval", run, f"step{step}")


def samples(run, step, prompt):
    p = (os.path.join(DATA, "logs", "E002", M, prompt, "samples.jsonl") if step == 0
         else os.path.join(cell_dir(run, step), prompt, "samples.jsonl"))
    return load(p) if os.path.exists(p) else None


def ci(pos, neg):
    a = auroc(pos, neg)
    lo, hi = auroc_ci(pos, neg)
    return a, lo, hi


def metrics(qs, lab, ids):
    S = lambda L: [s for i in ids if lab[i] == L for s in qs[i]["samples"]]
    ps = lambda L: (sum(s["n_search"] > 0 for s in S(L)) / len(S(L))) if S(L) else nan
    rate = {i: np.mean([s["n_search"] > 0 for s in qs[i]["samples"]]) for i in ids}
    acc = {i: np.mean([s["correct"] for s in qs[i]["samples"]]) for i in ids}
    accs = {i: np.mean([strict(s["text"], qs[i]["answers"]) for s in qs[i]["samples"]]) for i in ids}
    a, lo, hi = ci([rate[i] for i in ids if lab[i] == "unknowable"], [rate[i] for i in ids if lab[i] == "knowable"])
    allS = [s for i in ids for s in qs[i]["samples"]]
    w = [x for x in (span_words(s["text"]) for s in allS) if x is not None]
    out = {"n_knowable": sum(lab[i] == "knowable" for i in ids), "n_unknowable": sum(lab[i] == "unknowable" for i in ids),
           "n_questions": len(ids), "p_search_knowable": ps("knowable"), "p_search_unknowable": ps("unknowable"),
           "p_search_uncertain": ps("uncertain"), "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
           "mean_searches": float(np.mean([s["n_search"] for s in allS])),
           "parse_rate": 1 - sum(s["parse"] == "untagged" for s in allS) / len(allS),
           "cap_hit": float(np.mean([s["cap_hit"] for s in allS])), "mean_answer_words": float(np.mean(w)) if w else nan}
    for rule, A in (("e001", acc), ("strict", accs)):
        for name, sel in (("overall", ids), ("knowable", [i for i in ids if lab[i] == "knowable"]),
                          ("unknowable", [i for i in ids if lab[i] == "unknowable"])):
            m, se = mean_se([A[i] for i in sel])
            out[f"acc_{rule}_{name}"], out[f"acc_{rule}_{name}_se"] = m, se
    return out


def k_scores(lab_rows):
    """Start-model fixed knowledge score k (E003 out-of-fold layer-24 probe, P(unknowable)); cached in k_scores.jsonl."""
    path = os.path.join(B, "k_scores.jsonl")
    if os.path.exists(path):
        return {i: r["k"] for i, r in load(path).items()}
    from sklearn.model_selection import StratifiedKFold
    d = os.path.join(DATA, "logs", "E003", M)
    hz = np.load(f"{d}/hidden.npz")
    hid = {i: k for k, i in enumerate(hz["ids"].tolist())}
    L = hz["layers"].tolist().index(24)
    ids = [i for i in hz["ids"].tolist() if not lab_rows[i]["ambiguous"] and lab_rows[i]["labels"][M] in ("knowable", "unknowable")]
    unc = [i for i in hz["ids"].tolist() if not lab_rows[i]["ambiguous"] and lab_rows[i]["labels"][M] == "uncertain"]
    y = np.array([lab_rows[i]["labels"][M] == "unknowable" for i in ids], int)
    X, Xu = hz["H"][[hid[i] for i in ids], L], hz["H"][[hid[i] for i in unc], L]
    P, Pu = np.zeros(len(ids)), np.zeros(len(unc))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = probe().fit(X[tr], y[tr])
        P[te] = m.predict_proba(X[te])[:, 1]
        Pu += m.predict_proba(Xu)[:, 1] / 5
    sig = {r["id"]: float(r["S4_fixed_L24"]) for r in csv.DictReader(open(f"{d}/signals.csv"))}
    dev = max(abs(P[k] - sig[i]) for k, i in enumerate(ids))
    print(f"k: rebuilt out-of-fold vs E003 signals.csv max |diff| = {dev:.2e} (n {len(ids)}); uncertain n {len(unc)}")
    assert dev < 1e-6, dev
    with open(path, "w") as f:
        for k, i in enumerate(ids):
            f.write(json.dumps({"id": i, "label": lab_rows[i]["labels"][M], "k": sig[i], "source": "E003 signals.csv"}) + "\n")
        for k, i in enumerate(unc):
            f.write(json.dumps({"id": i, "label": "uncertain", "k": float(Pu[k]), "source": "mean of 5 E003 fold models"}) + "\n")
    return {i: r["k"] for i, r in load(path).items()}


def k_coef(qs, lab, ids, K, n_boot=1000):
    from sklearn.linear_model import LogisticRegression
    ids = [i for i in ids if lab[i] in ("knowable", "unknowable")]
    per = {i: ([float(lab[i] == "unknowable"), K[i]], [int(s["n_search"] > 0) for s in qs[i]["samples"]]) for i in ids}

    def fit(sel):
        X = np.array([per[i][0] for i in sel for _ in per[i][1]])
        y = np.array([v for i in sel for v in per[i][1]])
        if y.min() == y.max():
            return None
        try:
            return LogisticRegression(penalty=None, max_iter=5000).fit(X, y).coef_[0][1]
        except Exception:
            return None
    c = fit(ids)
    rng = np.random.default_rng(0)
    bs = [fit(list(rng.choice(ids, len(ids)))) for _ in range(n_boot)]
    ok = [b for b in bs if b is not None]
    return {"k_coef": nan if c is None else float(c), "k_lo": float(np.percentile(ok, 2.5)) if ok else nan,
            "k_hi": float(np.percentile(ok, 97.5)) if ok else nan, "k_boot_dropped": n_boot - len(ok),
            "k_n_episodes": sum(len(per[i][1]) for i in ids),
            "k_share_search": float(np.mean([v for i in ids for v in per[i][1]]))}


def signals(run, step, lab, keep, K):
    from scipy.stats import spearmanr
    d = os.path.join(DATA, "logs", "E003", M) if step == 0 else cell_dir(run, step)
    if not os.path.exists(f"{d}/s5_margin.jsonl") or not os.path.exists(f"{d}/hidden.npz"):
        return {}
    s5 = load(f"{d}/s5_margin.jsonl")
    hz = np.load(f"{d}/hidden.npz")
    hid = {i: k for k, i in enumerate(hz["ids"].tolist())}
    ids = [i for i in hz["ids"].tolist() if keep(i) and lab[i] in ("knowable", "unknowable")]
    y = np.array([lab[i] == "unknowable" for i in ids], int)
    m5 = [(s5[i]["s5"], yy) for i, yy in zip(ids, y) if s5[i]["s5"] is not None]
    a5 = ci([v for v, yy in m5 if yy == 1], [v for v, yy in m5 if yy == 0])
    L = hz["layers"].tolist().index(24)
    P = oof(hz["H"][[hid[i] for i in ids], L], y)
    a4 = ci(P[y == 1], P[y == 0])
    out = {"s5_auroc": a5[0], "s5_lo": a5[1], "s5_hi": a5[2], "s5_n": len(m5),
           "s4_l24_auroc": a4[0], "s4_lo": a4[1], "s4_hi": a4[2], "s4_n": len(ids)}
    for L_ in ("knowable", "unknowable"):
        pairs = [(s5[i]["s5"], K[i]) for i in ids if lab[i] == L_ and s5[i]["s5"] is not None]
        r = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else None
        out[f"spearman_{L_}"] = float(r.statistic) if r is not None else nan
        out[f"spearman_{L_}_n"] = len(pairs)
    return out


def relabel_diag(rows, golds_lab=None):
    S = [s for r in rows.values() for s in r["samples"]]
    w = [x for x in (span_words(s["text"]) for s in S) if x is not None]
    return {"n_samples": len(S), "share_search_tag": float(np.mean(["<search>" in s["text"] for s in S])),
            "share_no_answer": float(np.mean(["<answer>" not in s["text"] for s in S])),
            "mean_answer_words": float(np.mean(w)) if w else nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["E005-O-s0", "E005-S-s0", "E005-C-s0"])
    a = ap.parse_args()
    lab_rows = load(os.path.join(DATA, "eval", "e001", "e001_easy_1k_labels.jsonl"))
    keep = lambda i: not lab_rows[i]["ambiguous"]
    lab = {i: r["labels"][M] for i, r in lab_rows.items()}
    ids_all = [i for i in lab if keep(i)]
    K = k_scores(lab_rows)
    s2 = load(os.path.join(DATA, "logs", "E003", M, "s2_samples.jsonl"))
    AG = {i: agreement(s2[i]["samples"]) for i in ids_all}
    cells, sig, last = [], {}, {}
    cache0 = {}
    for run in a.runs:
        for step in STEPS:
            for prompt in PROMPTS:
                if step == 0 and prompt in cache0:
                    c = dict(cache0[prompt])
                else:
                    qs = samples(run, step, prompt)
                    if qs is None:
                        continue
                    c = metrics(qs, lab, ids_all)
                    if prompt == "P2":
                        c.update(k_coef(qs, lab, ids_all, K))
                        for b, (lo, hi) in {"0.2-0.4": (0.2, 0.4), "0.6": (0.6, 0.6), "0.8-1.0": (0.8, 1.0)}.items():
                            sel = [i for i in ids_all if lab[i] == "uncertain" and lo - 1e-9 <= AG[i] <= hi + 1e-9]
                            S = [s for i in sel for s in qs[i]["samples"]]
                            c[f"unc_search_{b}"] = float(np.mean([s["n_search"] > 0 for s in S])) if S else nan
                            c[f"unc_n_{b}"] = len(sel)
                    if step == 0:
                        cache0[prompt] = dict(c)
                cells.append({"run": run, "step": step, "prompt": prompt, **c})
                if step > 0:
                    last[run] = max(last.get(run, 0), step)
            if step == 0 and 0 in {k[1] for k in sig}:
                sig[(run, 0)] = sig[next(k for k in sig if k[1] == 0)]
            else:
                sig[(run, step)] = signals(run, step, lab, keep, K)
    for c in cells:
        c.update({k: v for k, v in sig.get((c["run"], c["step"]), {}).items()} if c["prompt"] == "P2" else {})
    with open(os.path.join(B, "e005_tidy.csv"), "w", newline="") as fh:
        keys = sorted({k for c in cells for k in c}, key=lambda k: (k not in ("run", "step", "prompt"), k))
        keys = ["run", "step", "prompt"] + [k for k in keys if k not in ("run", "step", "prompt")]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(cells)

    f3 = lambda x: "—" if x is None or x != x else f"{x:.3f}"
    f2 = lambda x: "—" if x is None or x != x else f"{x:.2f}"
    pct = lambda c, k: f"{100 * c[k]:.1f} ± {100 * c[k + '_se']:.1f}"
    get = lambda run, step, prompt: next((c for c in cells if (c["run"], c["step"], c["prompt"]) == (run, step, prompt)), None)
    L = ["Questions: E001 set, ambiguous excluded; labels = qwen3b-instruct E001 labels. Step 0 = start model (E002 / E003 outputs). "
         "AUROC: unknowable = positive, 95% bootstrap CI. Accuracy ± SE over questions.\n"]
    for run in a.runs:
        L += [f"### Table A ({run}, P2)\n",
              "| step | P(s∣kn) | P(s∣unk) | AUROC [CI] | acc kn % (strict) | acc unk % (strict) | acc overall % (strict) | acc kn % (E001) | acc unk % (E001) | acc overall % (E001) | mean answer words | mean searches | parse | cap-hit |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for step in STEPS:
            c = get(run, step, "P2")
            if c:
                L.append(f"| {step} | {f3(c['p_search_knowable'])} | {f3(c['p_search_unknowable'])} | {f3(c['auroc'])} [{f3(c['auroc_lo'])}, {f3(c['auroc_hi'])}] | "
                         f"{pct(c, 'acc_strict_knowable')} | {pct(c, 'acc_strict_unknowable')} | {pct(c, 'acc_strict_overall')} | "
                         f"{pct(c, 'acc_e001_knowable')} | {pct(c, 'acc_e001_unknowable')} | {pct(c, 'acc_e001_overall')} | "
                         f"{f2(c['mean_answer_words'])} | {c['mean_searches']:.2f} | {c['parse_rate']:.3f} | {c['cap_hit']:.3f} |")
        L.append("")
    L += ["### Table B (prompt shift; step 100 = the arm's last evaluated step if it stopped early)\n",
          "| run | step | prompt | P(s∣kn) | P(s∣unk) | AUROC [CI] |", "|---|---|---|---|---|---|"]
    for run in a.runs:
        for step in sorted({0, last.get(run, 0)}):
            for prompt in ("P1", "P2", "P6"):
                c = get(run, step, prompt)
                if c:
                    L.append(f"| {run} | {step} | {prompt} | {f3(c['p_search_knowable'])} | {f3(c['p_search_unknowable'])} | "
                             f"{f3(c['auroc'])} [{f3(c['auroc_lo'])}, {f3(c['auroc_hi'])}] |")
    L.append("")
    for run in a.runs:
        L += [f"### Table C ({run}, P2)\n",
              "| step | S5 margin AUROC [CI] (n) | probe L24 AUROC [CI] (n) | k coefficient [CI] (episodes, P(search), dropped resamples) | Spearman(margin, k) knowable (n) | Spearman(margin, k) unknowable (n) |",
              "|---|---|---|---|---|---|"]
        for step in STEPS:
            s, c = sig.get((run, step)), get(run, step, "P2")
            if s and c:
                L.append(f"| {step} | {f3(s['s5_auroc'])} [{f3(s['s5_lo'])}, {f3(s['s5_hi'])}] ({s['s5_n']}) | "
                         f"{f3(s['s4_l24_auroc'])} [{f3(s['s4_lo'])}, {f3(s['s4_hi'])}] ({s['s4_n']}) | "
                         f"{f2(c.get('k_coef'))} [{f2(c.get('k_lo'))}, {f2(c.get('k_hi'))}] ({c.get('k_n_episodes')}, {f3(c.get('k_share_search'))}, {c.get('k_boot_dropped')}) | "
                         f"{f3(s['spearman_knowable'])} ({s['spearman_knowable_n']}) | {f3(s['spearman_unknowable'])} ({s['spearman_unknowable_n']}) |")
        L.append("")
    for run in a.runs:
        p = os.path.join(B, run, "reward_steps.jsonl")
        if not os.path.exists(p):
            continue
        rs = {r["step"]: r for r in map(json.loads, open(p)) if r["split"] == "train"}
        L += [f"### Table D ({run}, training rollouts by training label; search = ≥1 executed search)\n",
              "| step | label | rollouts | P(search) | outcome 0s / ≥1s | arm term 0s / ≥1s | total 0s / ≥1s | groups | mixed search | mixed search & total | answer words mean | p95 | unclosed |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for step in [1] + list(range(10, 101, 10)):
            if step not in rs:
                continue
            for lb in ("knowable", "unknowable"):
                c = rs[step].get("components", {}).get("by_label", {}).get("e004_label", {}).get(lb)
                an = rs[step].get("answer", {}).get("by_label", {}).get("e004_label", {}).get(lb)
                if c:
                    L.append(f"| {step} | {lb} | {c['n']} | {f3(c['share_search'])} | {f3(c['mean_outcome_0search'])} / {f3(c['mean_outcome_search'])} | "
                             f"{f3(c['mean_arm_0search'])} / {f3(c['mean_arm_search'])} | {f3(c['mean_total_0search'])} / {f3(c['mean_total_search'])} | "
                             f"{c['n_groups']} | {c['n_groups_mixed_search']} | {c['n_groups_mixed_search_and_total']} | "
                             f"{f2(an['mean_words']) if an else '—'} | {f2(an['p95_words']) if an else '—'} | {f3(an['share_unclosed']) if an else '—'} |")
        L.append("")
    L += ["### Table E (uncertain questions, P2 search rate by start-model E003 S2 agreement)\n",
          "| run | step | 0.2–0.4 (n) | 0.6 (n) | 0.8–1.0 (n) |", "|---|---|---|---|---|"]
    for run in a.runs:
        for step in sorted({0, last.get(run, 0)}):
            c = get(run, step, "P2")
            if c and "unc_search_0.6" in c:
                L.append(f"| {run} | {step} | " + " | ".join(f"{f3(c[f'unc_search_{b}'])} ({c[f'unc_n_{b}']})" for b in ("0.2-0.4", "0.6", "0.8-1.0")) + " |")
    L.append("")
    L += ["### Table F (final relabel, E001 closed-book procedure; non-ambiguous questions)\n",
          "| model | step | samples | share with <search> | share with no <answer> | mean answer words | knowable | unknowable | uncertain |",
          "|---|---|---|---|---|---|---|---|---|"]
    e1 = {i: r for i, r in load(os.path.join(DATA, "logs", "E001", f"E001-{M}-s0", "samples.jsonl")).items() if keep(i)}
    d0 = relabel_diag(e1)
    cnt = lambda labs: [sum(labs[i] == v for i in ids_all) for v in ("knowable", "unknowable", "uncertain")]
    L.append(f"| start model (E001 run) | 0 | {d0['n_samples']} | {f3(d0['share_search_tag'])} | {f3(d0['share_no_answer'])} | {f2(d0['mean_answer_words'])} | "
             + " | ".join(map(str, cnt(lab))) + " |")
    for run in a.runs:
        s = last.get(run)
        rp = os.path.join(B, "eval", run, f"step{s}", "relabel") if s else None
        if rp and os.path.exists(f"{rp}/labels.jsonl"):
            rows = {i: r for i, r in load(f"{rp}/samples.jsonl").items() if keep(i)}
            dd = relabel_diag(rows)
            rl = {i: r["label"] for i, r in load(f"{rp}/labels.jsonl").items()}
            L.append(f"| {run} | {s} | {dd['n_samples']} | {f3(dd['share_search_tag'])} | {f3(dd['share_no_answer'])} | {f2(dd['mean_answer_words'])} | "
                     + " | ".join(map(str, cnt(rl))) + " |")
    txt = "\n".join(L)
    open(os.path.join(B, "e005_tables.md"), "w").write(txt + "\n")
    print(txt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    runs = a.runs + ["E004-A-s0"]
    for run, col in zip(runs, ("#1f77b4", "#2ca02c", "#d62728", "#7f7f7f")):
        cs = [(s, samples(run, s, "P2")) for s in STEPS]
        pts = [(s, metrics(q, lab, ids_all) if run.startswith("E004") else get(run, s, "P2")) for s, q in cs if q is not None]
        pts = [(s, c) for s, c in pts if c]
        if not pts:
            continue
        st = [s for s, _ in pts]
        ax[0].errorbar(st, [c["auroc"] for _, c in pts], yerr=[[c["auroc"] - c["auroc_lo"] for _, c in pts], [c["auroc_hi"] - c["auroc"] for _, c in pts]],
                       color=col, marker="o", capsize=3, label=run)
        ax[1].plot(st, [c["p_search_knowable"] for _, c in pts], color=col, marker="o", label=run)
        ax[2].plot(st, [c["p_search_unknowable"] for _, c in pts], color=col, marker="s", label=run)
    ax[0].axhline(0.5, color="grey", lw=0.8)
    ax[0].set(xlabel="training step", ylabel="AUROC (search rate, unknowable vs knowable)", title="P2: AUROC (95% CI)")
    ax[1].set(xlabel="training step", ylabel="P(≥1 search | knowable)", title="P2: search rate, knowable", ylim=(0, 1.02))
    ax[2].set(xlabel="training step", ylabel="P(≥1 search | unknowable)", title="P2: search rate, unknowable", ylim=(0, 1.02))
    for x in ax:
        x.legend(fontsize=8)
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(B, "e005_plot.png"), dpi=150)


if __name__ == "__main__":
    main()
