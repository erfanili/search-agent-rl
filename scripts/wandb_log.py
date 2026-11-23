#!/usr/bin/env python
"""
Log finished runs to W&B (project search-agent-rl), so every result lives in one searchable table.

  eval <eval dir>...        an eval_checkpoint.sh output dir: per-set EM/F1/answered/searches, as a run of
                            job_type "eval" (scripts/eval_checkpoint.sh calls this at the end)
  train <run dir>...        a finished training run's per-step metrics, parsed from train.log (backfill for runs
                            trained before W&B logging was on; new runs log live)
  backfill                  every eval and training run under $DATA/logs not yet logged (marker: .wandb_logged)

  .venv-verl/bin/python scripts/wandb_log.py backfill
  .venv-verl/bin/python scripts/wandb_log.py eval $DATA/logs/eval_..._20261111_221812 --group E0
Needs `wandb login` once. Re-logging a dir is skipped unless --force.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent.paths import DATA  # noqa: E402

PROJECT = "search-agent-rl"
LOGS = os.path.join(DATA, "logs")
MARK = ".wandb_logged"
EVAL_RE = re.compile(r"^eval_(?P<model>.+?)(?:_(?P<depth>exact\d+|min\d+))?_(?P<stamp>\d{8}_\d{6})$")


def git_state(path):
    """Commit recorded by the run (config.txt) or the repo's current one."""
    cfg = os.path.join(path, "config.txt")
    if os.path.exists(cfg):
        for line in open(cfg):
            if line.startswith("git:"):
                return line.split(":", 1)[1].strip()
    return None


def log_eval(d, group=None, force=False):
    import wandb
    met = os.path.join(d, "scored", "metrics.json")
    if not os.path.exists(met) or (os.path.exists(os.path.join(d, MARK)) and not force):
        return False
    m = json.load(open(met))
    agg = m.get("metrics", m.get("aggregate", {}))
    name = os.path.basename(d)
    g = EVAL_RE.match(name)
    model = g["model"] if g else name
    depth = (g["depth"] if g and g["depth"] else "free")
    if group is None:
        group = "E0" if depth.startswith("exact") else "eval"
    step = re.search(r"global_step_(\d+)$", model)
    run = wandb.init(project=PROJECT, name=name, job_type="eval", group=group, dir=d, reinit="finish_previous",
                     tags=[f"depth:{depth}", f"split:{m.get('split')}"],
                     config={"model": model, "depth_policy": depth, "split": m.get("split"),
                             "train_run": re.sub(r"_global_step_\d+$", "", model) if step else None,
                             "ckpt_step": int(step.group(1)) if step else None, "eval_dir": d})
    summary = {}
    for ds, v in agg.items():
        for k in ("em", "f1", "acc", "answer_rate", "mean_searches", "n"):
            if k in v:
                summary[f"{ds}/{k}"] = v[k]
    rows = [[ds, v.get("n"), v.get("em"), v.get("f1"), v.get("answer_rate"), v.get("mean_searches")] for ds, v in agg.items()]
    run.log({"results": wandb.Table(columns=["dataset", "n", "em", "f1", "answered", "searches"], data=rows)})
    run.summary.update(summary)
    for f in ("scores.txt", "infer.log"):
        if os.path.exists(os.path.join(d, f)):
            run.save(os.path.join(d, f), base_path=d, policy="now")
    run.finish()
    run.url and open(os.path.join(d, MARK), "w").write(f"{run.url}\n")
    print(f"eval  {name} -> {run.url}")
    return True


def log_train(d, group=None, force=False):
    import wandb
    from scripts.export_results import parse_train_log
    log = os.path.join(d, "train.log")
    if not os.path.exists(log) or (os.path.exists(os.path.join(d, MARK)) and not force):
        return False
    if glob.glob(os.path.join(d, "wandb", "run-*")):   # logged live
        return False
    train, val = parse_train_log(log)
    if not train:
        return False
    name = os.path.basename(d)
    cfg = {}
    if os.path.exists(os.path.join(d, "config.txt")):
        for line in open(os.path.join(d, "config.txt")):
            for k, v in re.findall(r"(\w+)[:=] *([^ \n]+)", line):
                cfg.setdefault(k, v)
    os.makedirs(os.path.join(d, "wandb_backfill"), exist_ok=True)   # not d/wandb, which marks a live-logged run
    run = wandb.init(project=PROJECT, name=name, job_type="train", group=group or "train",
                     dir=os.path.join(d, "wandb_backfill"), reinit="finish_previous",
                     tags=["backfilled", f"git:{git_state(d) or 'unknown'}"], config={**cfg, "run_dir": d})
    vsteps = dict(val)
    for r in train:
        row = {f"train/{k}": v for k, v in r.items() if k != "step"}
        for (ds, mm), x in vsteps.get(r["step"], {}).items():
            row[f"val/{ds}/{mm}"] = x
        run.log(row, step=r["step"])
    if 0 in vsteps:
        run.summary.update({f"val0/{ds}/{mm}": x for (ds, mm), x in vsteps[0].items()})
    for f in ("config.txt", "git.diff", "reward_steps.jsonl"):
        if os.path.exists(os.path.join(d, f)):
            run.save(os.path.join(d, f), base_path=d, policy="now")
    run.finish()
    run.url and open(os.path.join(d, MARK), "w").write(f"{run.url}\n")
    print(f"train {name} -> {run.url}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["eval", "train", "backfill"])
    p.add_argument("dirs", nargs="*")
    p.add_argument("--group")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    if a.what == "eval":
        for d in a.dirs:
            log_eval(d, a.group, a.force)
    elif a.what == "train":
        for d in a.dirs:
            log_train(d, a.group, a.force)
    else:
        n = 0
        for d in sorted(glob.glob(os.path.join(LOGS, "grpo_*"))):
            n += log_train(d, "reward_arms" if re.search(r"_full_(em_hit|em_multi|f1)_", d) else "train", a.force)
        for d in sorted(glob.glob(os.path.join(LOGS, "eval_*")) + glob.glob(os.path.join(LOGS, "baseline_*"))):
            n += log_eval(d, None, a.force)
        print(f"backfill: {n} runs logged")


if __name__ == "__main__":
    main()
