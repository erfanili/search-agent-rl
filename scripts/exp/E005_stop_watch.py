#!/usr/bin/env python
"""E005 answer-span stop rule (card), checked every training step from step 10, for one run.

  .venv-verl/bin/python scripts/exp/E005_stop_watch.py --run $DATA_ROOT/logs/E005/E005-O-s0

Reads reward_steps.jsonl ("answer" block, agent/step_stats.answer_stats; parsed answer span = first <answer> span of
the model text, environment text stripped, unclosed -> to the end; words = whitespace split).
Reference = the arm's own steps 1-3: per e004_label, mean words pooled over the answered rollouts of steps 1-3
(sum of mean_words x n_answer / sum of n_answer).
Stop at step s >= 10 if, for knowable or unknowable, mean words > 2 x its reference, or the unclosed share > 0.05
(checked per label and over all rollouts, as E004's step-20 check).
On a stop: writes STOP (the flags), kills the run's training (pkill on default_local_dir=<run>/ckpt), exits 3.
Every checked step is appended to stop_watch.jsonl. Exits 0 when training has gone and no new step appears.
"""
import argparse, json, os, subprocess, time

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--from-step", type=int, default=10)
ap.add_argument("--poll", type=int, default=30)
a = ap.parse_args()
RS, CK = os.path.join(a.run, "reward_steps.jsonl"), os.path.join(a.run, "ckpt")
LABELS = ("knowable", "unknowable")


def alive():
    return subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", f"default_local_dir={CK}"], capture_output=True).returncode == 0


def rows():
    if not os.path.exists(RS):
        return {}
    out = {}
    for line in open(RS):
        r = json.loads(line)
        if r["split"] == "train" and "answer" in r:
            out[r["step"]] = r["answer"]
    return out


done = set()
if os.path.exists(os.path.join(a.run, "stop_watch.jsonl")):
    done = {json.loads(l)["step"] for l in open(os.path.join(a.run, "stop_watch.jsonl"))}
idle = 0
while True:
    R = rows()
    new = sorted(s for s in R if s >= a.from_step and s not in done)
    if new and all(s in R for s in (1, 2, 3)):
        ref = {}
        for lab in LABELS:
            c = [R[s]["by_label"]["e004_label"][lab] for s in (1, 2, 3)]
            n = sum(x["n_answer"] for x in c)
            ref[lab] = sum(x["mean_words"] * x["n_answer"] for x in c if x["n_answer"]) / n
        for s in new:
            cur = {"all": R[s]["all"], **{lab: R[s]["by_label"]["e004_label"][lab] for lab in LABELS}}
            flags = []
            for lab in LABELS:
                m = cur[lab]["mean_words"]
                if m is not None and m > 2 * ref[lab]:
                    flags.append(f"step {s} {lab}: mean words {m:.2f} > 2 x {ref[lab]:.2f}")
            for k in ("all",) + LABELS:
                u = cur[k]["share_unclosed"]
                if u is not None and u > 0.05:
                    flags.append(f"step {s} {k}: unclosed share {u:.3f} > 0.05")
            rec = {"step": s, "time": time.strftime("%F %T"), "ref_mean_words_steps1_3": ref,
                   "mean_words": {k: cur[k]["mean_words"] for k in cur}, "share_unclosed": {k: cur[k]["share_unclosed"] for k in cur},
                   "flags": flags}
            open(os.path.join(a.run, "stop_watch.jsonl"), "a").write(json.dumps(rec) + "\n")
            done.add(s)
            if flags:
                open(os.path.join(a.run, "STOP"), "w").write("\n".join(flags) + "\n")
                subprocess.run(["pkill", "-u", str(os.getuid()), "-f", f"default_local_dir={CK}"])
                print("STOP:", flags, flush=True)
                raise SystemExit(3)
        idle = 0
    else:
        idle = idle + 1 if not alive() else 0
        if idle >= 4:
            print("training gone; exiting", flush=True)
            raise SystemExit(0)
    time.sleep(a.poll)
