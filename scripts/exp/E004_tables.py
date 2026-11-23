#!/usr/bin/env python
"""E004 tables, tidy CSV and plot.

  .venv-verl/bin/python scripts/exp/E004_tables.py [--runs E004-A-s0 E004-B-s0]

Questions: the E001 set, ambiguous excluded (E001 flag). Labels: qwen3b-instruct's E001 labels ("e001"); at step 100
also each arm's relabel with the E001 procedure on its final model ("final"; eval/<run>/step100/relabel/labels.jsonl).
Positive class for every AUROC = unknowable; knowable vs unknowable only (uncertain excluded from AUROC).
Step 0 (both arms) = the start model, reused from E002 (logs/E002/qwen3b-instruct/{P1,P2,P6}) and E003
(logs/E003/qwen3b-instruct/{s5_margin.jsonl, hidden.npz}).
Cell metrics: P(>=1 search | knowable / unknowable) = share of episodes; AUROC of the per-question search rate
  (share of its 5 episodes with >= 1 search) with a 1,000-resample stratified bootstrap 95% CI (default_rng(0));
  accuracy (cover match) overall / knowable / unknowable = mean over questions of n_correct/5 +- SE over questions;
  overall = all non-ambiguous questions (incl. uncertain). Also parse rate, cap-hit rate, mean searches.
S5: AUROC of the margin (E003 method). S4: out-of-fold AUROC of the E003 probe (StandardScaler + L2
  LogisticRegressionCV, C in logspace(-4,2,7), inner/outer StratifiedKFold(5, seed 0)) at layer 24, all questions.
Table D: reward_steps.jsonl (training rollouts), steps 1, 10, 20, ..., 100, split by e004_label. Answer-span columns
  (Erfan, 2026-11-21; "answer" field from the worktree's step_stats): mean / p95 words of the parsed answer span and
  the unclosed share over rollouts with an <answer>. Arm A logged them from step 21 only (resumed from global_step_20
  with the logging; steps 1-20 show "—"); its steps 1-3 reference row is the pilot's 960 dumped episodes
  (E004-A-s0/answer_check.json), and step 21 (= rollouts of the step-20 checkpoint, the checked step) is shown too.
Outputs: $DATA_ROOT/logs/E004/{e004_tables.md, e004_tidy.csv, e004_plot.png}
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA  # noqa: E402
from E002_tables import auroc, auroc_ci, mean_se  # noqa: E402
from E003_tables import oof  # noqa: E402

M = "qwen3b-instruct"
STEPS = [0, 20, 40, 60, 80, 100]
PROMPTS = ["P2", "P1", "P6"]
B = os.path.join(DATA, "logs", "E004")
load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}


def cell_dir(run, step):
    return None if step == 0 else os.path.join(B, "eval", run, f"step{step}")


def samples(run, step, prompt):
    p = (os.path.join(DATA, "logs", "E002", M, prompt, "samples.jsonl") if step == 0
         else os.path.join(cell_dir(run, step), prompt, "samples.jsonl"))
    return load(p) if os.path.exists(p) else None


def ci(pos, neg):
    a = auroc(pos, neg)
    lo, hi = auroc_ci(pos, neg)
    return a, lo, hi


def metrics(qs, lab, keep):
    ids = [i for i in qs if keep(i)]
    S = lambda L: [s for i in ids if lab[i] == L for s in qs[i]["samples"]]
    ps = lambda L: (sum(s["n_search"] > 0 for s in S(L)) / len(S(L))) if S(L) else float("nan")
    rate = {i: sum(s["n_search"] > 0 for s in qs[i]["samples"]) / len(qs[i]["samples"]) for i in ids}
    acc = {i: sum(s["correct"] for s in qs[i]["samples"]) / len(qs[i]["samples"]) for i in ids}
    a, lo, hi = ci([rate[i] for i in ids if lab[i] == "unknowable"], [rate[i] for i in ids if lab[i] == "knowable"])
    allS = [s for i in ids for s in qs[i]["samples"]]
    out = {"n_knowable": sum(lab[i] == "knowable" for i in ids), "n_unknowable": sum(lab[i] == "unknowable" for i in ids),
           "n_questions": len(ids), "p_search_knowable": ps("knowable"), "p_search_unknowable": ps("unknowable"),
           "auroc": a, "auroc_lo": lo, "auroc_hi": hi, "mean_searches": sum(s["n_search"] for s in allS) / len(allS),
           "parse_rate": 1 - sum(s["parse"] == "untagged" for s in allS) / len(allS),
           "cap_hit": sum(s["cap_hit"] for s in allS) / len(allS)}
    for name, sel in (("overall", ids), ("knowable", [i for i in ids if lab[i] == "knowable"]),
                      ("unknowable", [i for i in ids if lab[i] == "unknowable"])):
        m, se = mean_se([acc[i] for i in sel])
        out[f"acc_{name}"], out[f"acc_{name}_se"] = m, se
    return out


def s5_s4(run, step, lab, keep):
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
    return {"s5_auroc": a5[0], "s5_lo": a5[1], "s5_hi": a5[2], "s5_n": len(m5),
            "s4_l24_auroc": a4[0], "s4_lo": a4[1], "s4_hi": a4[2], "s4_n": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["E004-A-s0", "E004-B-s0"])
    a = ap.parse_args()
    lab_rows = load(os.path.join(DATA, "eval", "e001", "e001_easy_1k_labels.jsonl"))
    keep = lambda i: not lab_rows[i]["ambiguous"]
    e001 = {i: r["labels"][M] for i, r in lab_rows.items()}
    cells, sig = [], {}
    for run in a.runs:
        for step in STEPS:
            labelings = [("e001", e001)]
            rl = os.path.join(B, "eval", run, f"step{step}", "relabel", "labels.jsonl")
            if step == 100 and os.path.exists(rl):
                labelings.append(("final", {i: r["label"] for i, r in load(rl).items()}))
            for lname, lab in labelings:
                for prompt in PROMPTS:
                    qs = samples(run, step, prompt)
                    if qs is None:
                        continue
                    cells.append({"run": run, "step": step, "prompt": prompt, "labels": lname, **metrics(qs, lab, keep)})
                sig[(run, step, lname)] = s5_s4(run, step, lab, keep)

    with open(os.path.join(B, "e004_tidy.csv"), "w", newline="") as fh:
        keys = list(cells[0].keys())
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(cells)

    f3 = lambda x: "—" if x != x else f"{x:.3f}"
    pct = lambda c, k: f"{100 * c[k]:.1f} ± {100 * c[k + '_se']:.1f}"
    get = lambda run, step, prompt, lname="e001": next((c for c in cells if (c["run"], c["step"], c["prompt"], c["labels"]) == (run, step, prompt, lname)), None)
    L = ["Questions: E001 set, ambiguous excluded. Labels: qwen3b-instruct E001 labels unless marked final (relabel of the arm's step-100 model). Step 0 = start model (E002 / E003 outputs). AUROC: unknowable = positive, 95% bootstrap CI.\n"]
    for run in a.runs:
        L += [f"### Table A ({run}, P2)\n", "| step | n know / unk | P(search ∣ knowable) | P(search ∣ unknowable) | AUROC [CI] | acc knowable % | acc unknowable % | acc overall % | mean searches | parse | cap-hit |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for step in STEPS:
            for lname in ("e001", "final"):
                c = get(run, step, "P2", lname)
                if c:
                    L.append(f"| {step}{' (final labels)' if lname == 'final' else ''} | {c['n_knowable']} / {c['n_unknowable']} | {f3(c['p_search_knowable'])} | "
                             f"{f3(c['p_search_unknowable'])} | {f3(c['auroc'])} [{f3(c['auroc_lo'])}, {f3(c['auroc_hi'])}] | {pct(c, 'acc_knowable')} | "
                             f"{pct(c, 'acc_unknowable')} | {pct(c, 'acc_overall')} | {c['mean_searches']:.2f} | {c['parse_rate']:.3f} | {c['cap_hit']:.3f} |")
        L.append("")
    for run in a.runs:
        L += [f"### Table B ({run}, prompt shift)\n", "| step | prompt | P(search ∣ knowable) | P(search ∣ unknowable) | AUROC [CI] |", "|---|---|---|---|---|"]
        for step in (0, 100):
            for lname in ("e001", "final"):
                for prompt in ("P1", "P2", "P6"):
                    c = get(run, step, prompt, lname)
                    if c:
                        L.append(f"| {step}{' (final labels)' if lname == 'final' else ''} | {prompt} | {f3(c['p_search_knowable'])} | {f3(c['p_search_unknowable'])} | "
                                 f"{f3(c['auroc'])} [{f3(c['auroc_lo'])}, {f3(c['auroc_hi'])}] |")
        L.append("")
    for run in a.runs:
        L += [f"### Table C ({run}, P2)\n", "| step | S5 margin AUROC [CI] (n) | S4 probe layer 24 AUROC [CI] (n) |", "|---|---|---|"]
        for step in STEPS:
            for lname in ("e001", "final"):
                s = sig.get((run, step, lname))
                if s:
                    L.append(f"| {step}{' (final labels)' if lname == 'final' else ''} | {f3(s['s5_auroc'])} [{f3(s['s5_lo'])}, {f3(s['s5_hi'])}] ({s['s5_n']}) | "
                             f"{f3(s['s4_l24_auroc'])} [{f3(s['s4_lo'])}, {f3(s['s4_hi'])}] ({s['s4_n']}) |")
        L.append("")
    for run in a.runs:
        p = os.path.join(B, run, "reward_steps.jsonl")
        if not os.path.exists(p):
            continue
        rs = {r["step"]: r for r in map(json.loads, open(p)) if r["split"] == "train"}
        L += [f"### Table D ({run}, training rollouts, by training label)\n",
              "| step | label | rollouts | P(≥1 search) | mean reward | reward at 0 searches | reward at ≥1 search | groups | groups mixed in search | mixed in search and reward | answers | answer words mean | answer words p95 | unclosed share |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        ans_cols = lambda c: (f"{c['n_answer']} | {c['mean_words']:.2f} | {c['p95_words']:.1f} | {f3(c['share_unclosed'])}"
                              if c and c.get("mean_words") is not None else "— | — | — | —")
        chk = os.path.join(B, run, "answer_check.json")
        if os.path.exists(chk):
            ref = json.load(open(chk))["ref_steps1_3_pilot"]
            for lab in ("knowable", "unknowable"):
                L.append(f"| 1–3 (pilot dumps) | {lab} | — | — | — | — | — | — | — | — | {ans_cols(ref[lab])} |")
        steps_d = sorted(set([1] + list(range(10, 101, 10)) + ([21] if os.path.exists(chk) else [])))
        for step in steps_d:
            if step not in rs:
                continue
            for lab in ("knowable", "unknowable"):
                d = rs[step].get("by_label", {}).get("e004_label", {}).get(lab)
                ac = rs[step].get("answer", {}).get("by_label", {}).get("e004_label", {}).get(lab)
                if d:
                    L.append(f"| {step} | {lab} | {d['n']} | {f3(d['share_search'])} | {f3(d['mean_score'])} | "
                             f"{f3(d['mean_score_0search'] if d['mean_score_0search'] is not None else float('nan'))} | "
                             f"{f3(d['mean_score_search'] if d['mean_score_search'] is not None else float('nan'))} | {d['n_groups']} | "
                             f"{d['n_groups_mixed_search']} | {d['n_groups_mixed_search_and_score']} | {ans_cols(ac)} |")
        L.append("")
    txt = "\n".join(L)
    open(os.path.join(B, "e004_tables.md"), "w").write(txt + "\n")
    print(txt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for run, col in zip(a.runs, ("#1f77b4", "#d62728")):
        cs = [get(run, s, "P2") for s in STEPS]
        st = [s for s, c in zip(STEPS, cs) if c]
        cs = [c for c in cs if c]
        if not cs:
            continue
        ax[0].errorbar(st, [c["auroc"] for c in cs], yerr=[[c["auroc"] - c["auroc_lo"] for c in cs], [c["auroc_hi"] - c["auroc"] for c in cs]],
                       color=col, marker="o", capsize=3, label=run)
        ax[1].plot(st, [c["p_search_knowable"] for c in cs], color=col, marker="o", label=f"{run} knowable")
        ax[1].plot(st, [c["p_search_unknowable"] for c in cs], color=col, marker="s", ls="--", label=f"{run} unknowable")
    ax[0].axhline(0.5, color="grey", lw=0.8)
    ax[0].set(xlabel="training step", ylabel="AUROC (search rate, unknowable vs knowable)", title="P2: AUROC (95% CI)")
    ax[1].set(xlabel="training step", ylabel="P(≥1 search)", title="P2: search rate by E001 label", ylim=(0, 1))
    for x in ax:
        x.legend(fontsize=8)
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(B, "e004_plot.png"), dpi=150)


if __name__ == "__main__":
    main()
