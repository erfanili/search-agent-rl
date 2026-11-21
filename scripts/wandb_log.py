#!/usr/bin/env python
"""
Log finished eval runs to W&B (project search-agent-rl), so every result lives in one searchable table.
Training runs log live (scripts/train_grpo_3b.sh); this covers scripts/eval_checkpoint.sh outputs, which
call it at the end: per-set EM/F1/answered/searches, as a run of job_type "eval".

  .venv-verl/bin/python scripts/wandb_log.py eval $DATA/logs/eval_..._20261111_221812 --group <experiment>
Needs `wandb login` once. Re-logging a dir is skipped unless --force (marker: .wandb_logged).
"""

import argparse
import json
import os
import re

PROJECT = "search-agent-rl"
MARK = ".wandb_logged"
EVAL_RE = re.compile(r"^eval_(?P<model>.+?)(?:_(?P<depth>exact\d+|min\d+))?_(?P<stamp>\d{8}_\d{6})$")


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
    group = group or "eval"
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["eval"])
    p.add_argument("dirs", nargs="*")
    p.add_argument("--group")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    for d in a.dirs:
        log_eval(d, a.group, a.force)


if __name__ == "__main__":
    main()
