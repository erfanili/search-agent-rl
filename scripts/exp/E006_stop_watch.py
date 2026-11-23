#!/usr/bin/env python
"""E006 stop rules (card), checked every training step, for one run.

  .venv-verl/bin/python scripts/exp/E006_stop_watch.py --run $DATA_ROOT/logs/E006/E006-O-s0 [--regime]

Reads reward_steps.jsonl (worktree agent/reward.py; train split).
1. Length / outcome rule, from step 10. Step s is flagged if
   (a) for knowable or unknowable, the median capped answer-span words (answer_capped; unclosed spans count at most 50
       words) > 2 x the arm's own steps 1-3 median for that label (pooled over the answered rollouts of steps 1-3,
       answer_words_capped lists), or
   (b) the mean strict outcome over all rollouts (components.all.mean_outcome) < steps 1-3 mean - 0.15 (steps 1-3 mean =
       mean of the three per-step means; 320 rollouts each).
   The arm stops when 3 consecutive steps (all >= 10) are flagged: STOP (flags), training killed, exit 3.
2. Garble monitor, from step 1: a step is garble-flagged if any vLLM server's garbled share > 0.10 (garble.by_server).
   3 consecutive garble-flagged steps: STOP_GARBLE, training killed, exit 4 (the chain then stops).
3. --regime (arm C), at step 1: total reward by label and search choice (components; search = >= 1 executed search).
   If knowable total(0 searches) <= total(>= 1) or unknowable total(>= 1) <= total(0): STOP_REGIME (table), training
   killed, exit 5.
Every checked step is appended to stop_watch.jsonl. Exits 0 when training has gone and no new step appears.
"""
import argparse, json, os, statistics, subprocess, time

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--from-step", type=int, default=10)
ap.add_argument("--poll", type=int, default=30)
ap.add_argument("--regime", action="store_true")
a = ap.parse_args()
RS, CK = os.path.join(a.run, "reward_steps.jsonl"), os.path.join(a.run, "ckpt")
LABELS = ("knowable", "unknowable")
OUT = os.path.join(a.run, "stop_watch.jsonl")


def alive():
    return subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", f"default_local_dir={CK}"], capture_output=True).returncode == 0


def kill():
    subprocess.run(["pkill", "-u", str(os.getuid()), "-f", f"default_local_dir={CK}"])


def rows():
    out = {}
    if os.path.exists(RS):
        for line in open(RS):
            r = json.loads(line)
            if r["split"] == "train":
                out[r["step"]] = r
    return out


def stop(fname, lines, code):
    open(os.path.join(a.run, fname), "w").write("\n".join(lines) + "\n")
    kill()
    print(fname, lines, flush=True)
    raise SystemExit(code)


def regime_table(r):
    c = r["components"]["by_label"]["e004_label"]
    return {lab: {"n_0search": c[lab]["n_0search"], "n_search": c[lab]["n_search"],
                  "total_0search": c[lab]["mean_total_0search"], "total_search": c[lab]["mean_total_search"],
                  "outcome_0search": c[lab]["mean_outcome_0search"], "outcome_search": c[lab]["mean_outcome_search"],
                  "arm_0search": c[lab]["mean_arm_0search"], "arm_search": c[lab]["mean_arm_search"]} for lab in LABELS}


done = {json.loads(l)["step"] for l in open(OUT)} if os.path.exists(OUT) else set()
_last = [json.loads(l) for l in open(OUT)][-1:] if os.path.exists(OUT) else []
streak, gstreak, idle = (_last[0]["streak"], _last[0]["garble"]["streak"], 0) if _last else (0, 0, 0)
while True:
    R = rows()
    new = sorted(s for s in R if s not in done)
    if new:
        ref = None
        if all(s in R for s in (1, 2, 3)):
            ref = {"median": {lab: statistics.median(sum((R[s]["answer_words_capped"].get(lab, []) for s in (1, 2, 3)), []))
                              for lab in LABELS},
                   "outcome": sum(R[s]["components"]["all"]["mean_outcome"] for s in (1, 2, 3)) / 3}
        for s in new:
            r = R[s]
            rec = {"step": s, "time": time.strftime("%F %T"), "ref_steps1_3": ref}
            # garble
            g = r.get("garble")
            gflag = bool(g and g["warn_servers"])
            gstreak = gstreak + 1 if gflag else 0
            rec["garble"] = {"by_server": {k: v["share"] for k, v in g["by_server"].items()} if g else None,
                             "max_share": max((v["share"] or 0) for v in g["by_server"].values()) if g and g["by_server"] else None,
                             "flag": gflag, "streak": gstreak}
            # regime (arm C, step 1)
            if a.regime and s == 1:
                t = regime_table(r)
                ok_kn = t["knowable"]["total_0search"] > t["knowable"]["total_search"]
                ok_unk = t["unknowable"]["total_search"] > t["unknowable"]["total_0search"]
                rec["regime"] = {"table": t, "skip_wins_knowable": ok_kn, "search_wins_unknowable": ok_unk}
            # length / outcome
            ac = r["answer_capped"]["by_label"]["e004_label"]
            rec["median_capped"] = {lab: ac[lab]["median_words"] for lab in LABELS}
            rec["mean_outcome_all"] = r["components"]["all"]["mean_outcome"]
            flags = []
            if s >= a.from_step and ref is not None:
                for lab in LABELS:
                    m = ac[lab]["median_words"]
                    if m is not None and m > 2 * ref["median"][lab]:
                        flags.append(f"step {s} {lab}: median capped words {m:g} > 2 x {ref['median'][lab]:g}")
                if rec["mean_outcome_all"] < ref["outcome"] - 0.15:
                    flags.append(f"step {s}: strict outcome {rec['mean_outcome_all']:.3f} < {ref['outcome']:.3f} - 0.15")
                streak = streak + 1 if flags else 0
            rec["flags"], rec["streak"] = flags, streak
            open(OUT, "a").write(json.dumps(rec) + "\n")
            done.add(s)
            if a.regime and s == 1 and not (rec["regime"]["skip_wins_knowable"] and rec["regime"]["search_wins_unknowable"]):
                stop("STOP_REGIME", [json.dumps(rec["regime"])], 5)
            if gstreak >= 3:
                stop("STOP_GARBLE", [f"steps {s - 2}-{s}: server garbled share > 0.10 on 3 consecutive steps; last: {rec['garble']}"], 4)
            if streak >= 3:
                prev = [json.loads(l) for l in open(OUT)][-3:]
                stop("STOP", [f for p in prev for f in p["flags"]], 3)
        idle = 0
    else:
        idle = idle + 1 if not alive() else 0
        if idle >= 4:
            print("training gone; exiting", flush=True)
            raise SystemExit(0)
    time.sleep(a.poll)
