#!/usr/bin/env python
"""E007 tables A-D, baselines, tidy CSV and plot.

  .venv-verl/bin/python scripts/exp/E007_tables.py [--runs ...]     # E007 runs present under $DATA_ROOT/logs/E007/eval

Evaluation cells: E006_tables.metrics / k_coef / signals (same definitions as E006: E001 set, ambiguous excluded,
  qwen3b-instruct E001 labels; P(s|label) = share of episodes with >= 1 executed search; AUROC of the per-question search
  rate, unknowable positive, 1,000-resample bootstrap 95% CI; strict accuracy = alias_match_strict, mean over questions
  +- SE; S5 margin AUROC, out-of-fold probe L24 AUROC, k coefficient [question bootstrap CI], Spearman(S5 margin, k)
  within knowable / unknowable). E006 runs from $DATA_ROOT/logs/E006/eval (steps 20-100), E007 runs from
  $DATA_ROOT/logs/E007/eval (steps 20, 60, 100). Step 0 = start model (E002 / E003 outputs).
Table A (seeds): S and O, seeds 0 (E006-<arm>-s0), 1, 2 (E007-<arm>-s1/-s2); step 100 and each run's peak P2 AUROC step
  among steps 20, 60, 100 (the steps evaluated for every run). Mean, SD (ddof 1) and 95% t-interval (t, df = n - 1)
  over the seeds present.
Table B (controls and strength): E006-S, E007-N, E007-Sshuf, E007-S-l01, E007-S-l05, E006-O, E006-C at steps 20, 60, 100.
Table C (analysis-only baselines, start model, no training):
  per question strict accuracy under each action from E002 qwen3b-instruct P2-f0 (forced 0 searches) and P2-f1 (forced
  1 search), 5 samples each (mean of alias_match_strict over the 5).
  A. SC-gated: search iff E003 S2 agreement a < tau (tau in 0.4, 0.6, 0.8, 1.0). Per tau: search rate on knowable /
     unknowable / all questions, AUROC of the binary decision (unknowable positive, bootstrap CI), expected strict
     accuracy overall / knowable / unknowable (+- SE over questions). Costs 5 extra closed-book generations per question.
  B. Probe-gated: search iff k (E003 out-of-fold layer-24 probe P(unknowable); k_scores.jsonl, uncertain = mean of the 5
     fold models) is among the top n questions, n = round(overall search rate of the matching tau x 887).
  C. Matched: for each S / O run at step 100, the run's overall P2 search rate r (share of episodes, all 887 questions);
     SC-gated and probe-gated policies searching on exactly r x 887 questions in expectation: questions sorted by a
     ascending (resp. k descending); at the boundary a level the tied questions search with the fractional probability
     that makes the expected rate equal r (probe: the fractional question at the boundary). Expected strict accuracy
     overall / knowable / unknowable and search rate by label, next to the run's own P2 strict accuracy.
Table D (training dynamics, E007 runs): reward_steps.jsonl + stop_watch.jsonl at steps 1, 10, ..., 100: search share by
  label, mean outcome and total reward by label and search choice (0 / >= 1 executed searches), stop-rule statistics
  (median capped answer words by label, mean strict outcome over all rollouts, flags, streak) and the max per-server
  garbled share.
Table E: Spearman(S5 margin, k) for E006 O, S, C at steps 20-100 (card).
Outputs: $DATA_ROOT/logs/E007/{e007_tables.md, e007_tidy.csv, e007_baselines.csv, e007_plot.png}
"""
import argparse, csv, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import E006_tables as E6  # noqa: E402
from agent.paths import DATA  # noqa: E402
from E002_tables import auroc_ci, auroc, mean_se  # noqa: E402
from E003_tables import agreement  # noqa: E402

M = E6.M
B = os.path.join(DATA, "logs", "E007")
PROMPTS = ["P2", "P1", "P6"]
E7_RUNS = ["E007-S-s1", "E007-S-s2", "E007-O-s1", "E007-O-s2", "E007-N-s0", "E007-Sshuf-s0", "E007-S-l01-s0",
           "E007-S-l05-s0", "E007-C-s1"]
E6_RUNS = ["E006-S-s0", "E006-O-s0", "E006-C-s0"]
nan = float("nan")
load = E6.load


def cell_dir(run, step):
    if step == 0:
        return None
    return os.path.join(DATA, "logs", run[:4], "eval", run, f"step{step}")


E6.cell_dir = cell_dir  # E6.samples / E6.signals resolve cell directories through this


def steps_of(run):
    return [0, 20, 40, 60, 80, 100] if run.startswith("E006") else [0, 20, 60, 100]


def f3(x):
    return "—" if x is None or x != x else f"{x:.3f}"


def pct(c, k):
    return "—" if c is None or k not in c else f"{100 * c[k]:.1f} ± {100 * c[k + '_se']:.1f}"


def tint(v):
    v = [x for x in v if x == x]
    if len(v) < 2:
        return (v[0] if v else nan), nan, nan, nan
    from scipy.stats import t
    m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    h = t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v))
    return m, sd, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    a = ap.parse_args()
    runs7 = a.runs if a.runs is not None else [r for r in E7_RUNS if os.path.isdir(os.path.join(B, "eval", r))]
    lab_rows = load(os.path.join(DATA, "eval", "e001", "e001_easy_1k_labels.jsonl"))
    keep = lambda i: not lab_rows[i]["ambiguous"]
    lab = {i: r["labels"][M] for i, r in lab_rows.items()}
    ids = [i for i in lab if keep(i)]
    K = {i: r["k"] for i, r in load(os.path.join(DATA, "logs", "E006", "k_scores.jsonl")).items()}
    s2 = load(os.path.join(DATA, "logs", "E003", M, "s2_samples.jsonl"))
    AG = {i: agreement(s2[i]["samples"]) for i in ids}

    # ---------- evaluation cells
    cells, cache0 = {}, {}
    for run in E6_RUNS + runs7:
        for step in steps_of(run):
            if step == 0 and cache0:
                for p, c in cache0.items():
                    cells[(run, 0, p)] = c
                continue
            for p in PROMPTS:
                qs = E6.samples(run, step, p)
                if qs is None:
                    continue
                c = E6.metrics(qs, lab, ids)
                c["p_search_all"] = float(np.mean([s["n_search"] > 0 for i in ids for s in qs[i]["samples"]]))
                if p == "P2":
                    c.update(E6.k_coef(qs, lab, ids, K))
                    c.update(E6.signals(run, step, lab, keep, K))
                cells[(run, step, p)] = c
                if step == 0:
                    cache0[p] = c
    get = lambda r, s, p: cells.get((r, s, p))
    with open(os.path.join(B, "e007_tidy.csv"), "w", newline="") as fh:
        rows = [{"run": r, "step": s, "prompt": p, **c} for (r, s, p), c in cells.items()]
        keys = ["run", "step", "prompt"] + sorted({k for x in rows for k in x} - {"run", "step", "prompt"})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def peak(run):
        cand = [(get(run, s, "P2")["auroc"], s) for s in (20, 60, 100) if get(run, s, "P2")]
        return max(cand)[1] if cand else None

    COLS = [("P(s∣kn)", lambda r, s: (get(r, s, "P2") or {}).get("p_search_knowable", nan)),
            ("P(s∣unk)", lambda r, s: (get(r, s, "P2") or {}).get("p_search_unknowable", nan)),
            ("AUROC P2", lambda r, s: (get(r, s, "P2") or {}).get("auroc", nan)),
            ("AUROC P1", lambda r, s: (get(r, s, "P1") or {}).get("auroc", nan)),
            ("AUROC P6", lambda r, s: (get(r, s, "P6") or {}).get("auroc", nan)),
            ("strict acc overall", lambda r, s: (get(r, s, "P2") or {}).get("acc_strict_overall", nan)),
            ("strict acc unk", lambda r, s: (get(r, s, "P2") or {}).get("acc_strict_unknowable", nan)),
            ("Spearman kn", lambda r, s: (get(r, s, "P2") or {}).get("spearman_knowable", nan)),
            ("Spearman unk", lambda r, s: (get(r, s, "P2") or {}).get("spearman_unknowable", nan))]

    def fmt(name, v):
        return "—" if v != v else (f"{100 * v:.1f}" if name.startswith("strict") else f"{v:.3f}")

    def row(run, step):
        c = get(run, step, "P2")
        if c is None:
            return f"| {run} | {step} | " + " | ".join("—" for _ in COLS) + " |"
        out = []
        for n, fn in COLS:
            v = fn(run, step)
            if n == "AUROC P2":
                out.append(f"{f3(v)} [{f3(c['auroc_lo'])}, {f3(c['auroc_hi'])}]")
            elif n.startswith("strict"):
                k = "acc_strict_overall" if "overall" in n else "acc_strict_unknowable"
                out.append(pct(c, k))
            else:
                out.append(fmt(n, v))
        return f"| {run} | {step} | " + " | ".join(out) + " |"

    hdr = "| run | step | " + " | ".join(n + (" %" if n.startswith("strict") else "") for n, _ in COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 2)
    L = ["Questions: E001 set, ambiguous excluded (n = 887; 278 knowable / 466 unknowable / 143 uncertain), qwen3b-instruct "
         "E001 labels, 5 episodes per question, T = 0.7, prefix caching off (E006 / E007 evaluations). AUROC: unknowable "
         "positive, 95% bootstrap CI. Accuracy ± SE over questions. Spearman = Spearman(S5 margin, k) within label (P2).\n"]

    # ---------- Table A
    for arm in ("S", "O"):
        seeds = [f"E006-{arm}-s0", f"E007-{arm}-s1", f"E007-{arm}-s2"]
        for what in ("step 100", "peak AUROC step (of 20, 60, 100)"):
            L += [f"### Table A ({arm}, seeds, {what})\n", hdr, sep]
            pts = {}
            for r in seeds:
                s = 100 if what == "step 100" else peak(r)
                if s is None or get(r, s, "P2") is None:
                    L.append(f"| {r} | — | " + " | ".join("—" for _ in COLS) + " |")
                    continue
                pts[r] = s
                L.append(row(r, s))
            stats = [tint([fn(r, s) for r, s in pts.items()]) for _, fn in COLS]
            L.append(f"| mean ± SD (n = {len(pts)}) | | " + " | ".join(
                "—" if m != m else (f"{100 * m:.1f} ± {100 * sd:.1f}" if n.startswith("strict") else f"{m:.3f} ± {sd:.3f}")
                for (n, _), (m, sd, lo, hi) in zip(COLS, stats)) + " |")
            L.append(f"| 95% t-interval | | " + " | ".join(
                "—" if lo != lo else (f"[{100 * lo:.1f}, {100 * hi:.1f}]" if n.startswith("strict") else f"[{lo:.3f}, {hi:.3f}]")
                for (n, _), (m, sd, lo, hi) in zip(COLS, stats)) + " |")
            L.append("")

    # ---------- Table B
    L += ["### Table B (controls and strength)\n", hdr, sep]
    for r in ["E006-S-s0", "E007-N-s0", "E007-Sshuf-s0", "E007-S-l01-s0", "E007-S-l05-s0", "E006-O-s0", "E006-C-s0"]:
        for s in (20, 60, 100):
            L.append(row(r, s))
    L.append("")

    # ---------- Table C: baselines
    f0 = load(os.path.join(DATA, "logs", "E002", M, "P2-f0", "samples.jsonl"))
    f1 = load(os.path.join(DATA, "logs", "E002", M, "P2-f1", "samples.jsonl"))
    acc0 = {i: float(np.mean([E6.strict(s["text"], f0[i]["answers"]) for s in f0[i]["samples"]])) for i in ids}
    acc1 = {i: float(np.mean([E6.strict(s["text"], f1[i]["answers"]) for s in f1[i]["samples"]])) for i in ids}
    nS = {i: (len(f0[i]["samples"]), len(f1[i]["samples"])) for i in ids}
    N = len(ids)

    def policy_eval(p):   # p: question -> probability of searching
        out = {}
        for name, sel in (("all", ids), ("knowable", [i for i in ids if lab[i] == "knowable"]),
                          ("unknowable", [i for i in ids if lab[i] == "unknowable"])):
            out[f"rate_{name}"] = float(np.mean([p[i] for i in sel]))
            m, se = mean_se([p[i] * acc1[i] + (1 - p[i]) * acc0[i] for i in sel])
            out[f"acc_{name}"], out[f"acc_{name}_se"] = m, se
        pos = [p[i] for i in ids if lab[i] == "unknowable"]
        neg = [p[i] for i in ids if lab[i] == "knowable"]
        out["auroc"] = auroc(pos, neg)
        out["auroc_lo"], out["auroc_hi"] = auroc_ci(pos, neg)
        return out

    def sc_rate(r):      # SC-gated at expected overall rate r
        order = sorted(ids, key=lambda i: AG[i])
        n = r * N
        p, done = {}, 0
        for lvl in sorted(set(AG.values())):
            grp = [i for i in ids if abs(AG[i] - lvl) < 1e-9]
            take = min(len(grp), max(0.0, n - done))
            for i in grp:
                p[i] = take / len(grp)
            done += take
        return p

    def probe_rate(r):
        order = sorted(ids, key=lambda i: -K[i])
        n = r * N
        return {i: min(1.0, max(0.0, n - k)) for k, i in enumerate(order)}

    base_rows = []
    L += ["### Table C (analysis-only baselines on the start model; no training)\n",
          "Per-question strict accuracy under each action: E002 qwen3b-instruct P2-f0 (forced 0 searches) and P2-f1 (forced 1 "
          f"search), 5 samples each; no-search accuracy overall {100 * np.mean([acc0[i] for i in ids]):.1f}%, "
          f"always-search {100 * np.mean([acc1[i] for i in ids]):.1f}%. SC-gated costs 5 extra closed-book generations "
          "per question. Probe-gated thresholds matched to each τ's overall search rate.\n",
          "| policy | τ | search rate kn | search rate unk | search rate all | AUROC [CI] | exp. strict acc overall % | kn % | unk % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for tau in (0.4, 0.6, 0.8, 1.0):
        pa = {i: float(AG[i] < tau - 1e-9) for i in ids}
        A_ = policy_eval(pa)
        n_s = round(A_["rate_all"] * N)
        order = sorted(ids, key=lambda i: -K[i])
        pb = {i: float(k < n_s) for k, i in enumerate(order)}
        B_ = policy_eval(pb)
        for nm, R in (("A SC-gated", A_), ("B probe-gated", B_)):
            base_rows.append({"table": nm, "tau": tau, **R})
            L.append(f"| {nm} | {tau} | {f3(R['rate_knowable'])} | {f3(R['rate_unknowable'])} | {f3(R['rate_all'])} | "
                     f"{f3(R['auroc'])} [{f3(R['auroc_lo'])}, {f3(R['auroc_hi'])}] | {pct(R, 'acc_all')} | {pct(R, 'acc_knowable')} | {pct(R, 'acc_unknowable')} |")
    L += ["", "C. Matched comparison (step 100; same expected overall search rate as the run, P2):\n",
          "| run | overall search rate | run strict acc overall % | SC-gated acc overall % | probe-gated acc overall % | run unk % | SC unk % | probe unk % | run P(s∣kn) / P(s∣unk) | SC rate kn / unk | probe rate kn / unk |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ["E006-S-s0", "E006-O-s0"] + [x for x in runs7 if x.split("-")[1] in ("S", "O", "Sshuf")]:
        c = get(r, 100, "P2")
        if c is None:
            L.append(f"| {r} | — (no step 100) | | | | | | | | | |")
            continue
        rr = c["p_search_all"]
        A_, B_ = policy_eval(sc_rate(rr)), policy_eval(probe_rate(rr))
        base_rows += [{"table": "C SC-gated", "run": r, "rate": rr, **A_}, {"table": "C probe-gated", "run": r, "rate": rr, **B_}]
        L.append(f"| {r} | {f3(rr)} | {pct(c, 'acc_strict_overall')} | {pct(A_, 'acc_all')} | {pct(B_, 'acc_all')} | "
                 f"{pct(c, 'acc_strict_unknowable')} | {pct(A_, 'acc_unknowable')} | {pct(B_, 'acc_unknowable')} | "
                 f"{f3(c['p_search_knowable'])} / {f3(c['p_search_unknowable'])} | {f3(A_['rate_knowable'])} / {f3(A_['rate_unknowable'])} | "
                 f"{f3(B_['rate_knowable'])} / {f3(B_['rate_unknowable'])} |")
    L.append("")
    with open(os.path.join(B, "e007_baselines.csv"), "w", newline="") as fh:
        keys = sorted({k for x in base_rows for k in x})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(base_rows)

    # ---------- Table D
    for r in runs7:
        rs = os.path.join(B, r, "reward_steps.jsonl")
        if not os.path.exists(rs):
            continue
        R = {x["step"]: x for x in map(json.loads, open(rs)) if x["split"] == "train"}
        swp = os.path.join(B, r, "stop_watch.jsonl")
        W = {x["step"]: x for x in map(json.loads, open(swp))} if os.path.exists(swp) else {}
        L += [f"### Table D ({r}, training rollouts; search = ≥ 1 executed search; 320 rollouts per step)\n",
              "| step | search share kn / unk | outcome kn 0s / ≥1s | outcome unk 0s / ≥1s | total kn 0s / ≥1s | total unk 0s / ≥1s | capped median words kn / unk | mean strict outcome all | stop flags (streak) | max server garble share |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in [1] + list(range(10, 101, 10)):
            if s not in R:
                continue
            c = R[s]["components"]["by_label"]["e004_label"]
            g = R[s].get("garble", {}).get("by_server", {})
            gm = max([v["share"] for v in g.values()] or [nan])
            w_ = W.get(s, {})
            mc = w_.get("median_capped") or {k: R[s]["answer_capped"]["by_label"]["e004_label"][k]["median_words"] for k in ("knowable", "unknowable")}
            fl = "; ".join(w_.get("flags", [])) or "none"
            L.append(f"| {s} | {f3(c['knowable']['share_search'])} / {f3(c['unknowable']['share_search'])} | "
                     f"{f3(c['knowable']['mean_outcome_0search'])} / {f3(c['knowable']['mean_outcome_search'])} | "
                     f"{f3(c['unknowable']['mean_outcome_0search'])} / {f3(c['unknowable']['mean_outcome_search'])} | "
                     f"{f3(c['knowable']['mean_total_0search'])} / {f3(c['knowable']['mean_total_search'])} | "
                     f"{f3(c['unknowable']['mean_total_0search'])} / {f3(c['unknowable']['mean_total_search'])} | "
                     f"{mc['knowable']} / {mc['unknowable']} | {f3(R[s]['components']['all']['mean_outcome'])} | {fl} ({w_.get('streak', '—')}) | {f3(gm)} |")
        st = [x for x in W.values() if x.get("flags")]
        L.append(f"\nflagged steps: {len(st)}; longest streak {max([x['streak'] for x in W.values()] or [0])}; "
                 f"steps trained: {max(R)}; stop files: {', '.join(f for f in ('STOP', 'STOP_GARBLE', 'STOP_REGIME', 'STOP_DISK') if os.path.exists(os.path.join(B, r, f))) or 'none'}\n")

    # ---------- Table E
    L += ["### Table E (Spearman(S5 margin, k) within label, E006 runs, P2)\n",
          "| run | step | knowable (n) | unknowable (n) |", "|---|---|---|---|"]
    for r in E6_RUNS:
        for s in (20, 40, 60, 80, 100):
            c = get(r, s, "P2") or {}
            L.append(f"| {r} | {s} | {f3(c.get('spearman_knowable', nan))} ({c.get('spearman_knowable_n', '—')}) | "
                     f"{f3(c.get('spearman_unknowable', nan))} ({c.get('spearman_unknowable_n', '—')}) |")
    txt = "\n".join(L)
    open(os.path.join(B, "e007_tables.md"), "w").write(txt + "\n")
    print(txt)

    # ---------- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"S": "#1f77b4", "O": "#d62728", "C": "#7f7f7f", "N": "#2ca02c", "Sshuf": "#9467bd", "S-l01": "#17becf", "S-l05": "#bcbd22"}
    ls = {"s0": "-", "s1": "--", "s2": ":"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in E6_RUNS + runs7:
        parts = r.split("-")
        kind, seed = "-".join(parts[1:-1]), parts[-1]
        pts = [(s, get(r, s, "P2")["auroc"]) for s in steps_of(r) if get(r, s, "P2")]
        if len(pts) < 2:
            continue
        ax.plot(*zip(*pts), ls[seed], marker="o", ms=3, color=col.get(kind, "k"), label=r)
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out P2 AUROC (search rate, unknowable vs knowable)")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(B, "e007_plot.png"), dpi=150)


if __name__ == "__main__":
    main()
