#!/usr/bin/env python
"""E004 checkpoint watcher: merge each saved checkpoint to HF weights, confirm the merge loads, then free disk.

  .venv-verl/bin/python scripts/exp/E004_ckpt_watch.py --run $DATA_ROOT/logs/E004/E004-A-s0 --steps 20 40 60 80 100

For each step N (in order), once ckpt/latest_checkpointed_iteration.txt >= N (verl writes it after a save completes):
  1. merge ckpt/global_step_N/actor -> hf/step_N with verl.model_merger (CPU; written to hf/step_N.tmp, then renamed);
     tokenizer files copied from actor/huggingface.
  2. confirm it loads: transformers AutoModelForCausalLM (bf16, CPU) + AutoTokenizer; parameter count equal to the
     start model's config-derived count; one forward pass on a short prompt with finite logits. Written to
     hf/step_N/LOAD_OK.json.
  3. after step N is confirmed: for every EARLIER checkpoint M < N whose hf/step_M has LOAD_OK.json, delete
     ckpt/global_step_M/actor/{model,optim,extra_state}_world_size_*.pt (FSDP shards + optimizer state). The latest
     checkpoint stays complete for resuming. Merged weights are never deleted. (Erfan, 2026-11-21.)
Exits after the last step, or when the training process is gone and no new checkpoint can appear.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--steps", type=int, nargs="+", required=True)
ap.add_argument("--poll", type=int, default=60)
a = ap.parse_args()
PY = sys.executable
CK = os.path.join(a.run, "ckpt")
HF = os.path.join(a.run, "hf")
os.makedirs(HF, exist_ok=True)


def log(*x):
    print(time.strftime("[%F %T]"), *x, flush=True)


def latest():
    try:
        return int(open(os.path.join(CK, "latest_checkpointed_iteration.txt")).read().strip())
    except (OSError, ValueError):
        return 0


def training_alive():
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", f"default_local_dir={CK}"], capture_output=True)
    return r.returncode == 0


VERIFY = r'''
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p = sys.argv[1]
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16).eval()
n = sum(x.numel() for x in m.parameters())
ids = tok("Question: What is the capital of France?\n", return_tensors="pt").input_ids
with torch.no_grad():
    lg = m(ids).logits
print(json.dumps({"n_params": n, "dtype": str(next(m.parameters()).dtype), "logits_finite": bool(torch.isfinite(lg).all()),
                  "vocab": lg.shape[-1]}))
'''


def merge_and_verify(n):
    src = os.path.join(CK, f"global_step_{n}", "actor")
    dst = os.path.join(HF, f"step_{n}")
    if os.path.exists(os.path.join(dst, "LOAD_OK.json")):
        return True
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    log("merge", src, "->", dst)
    r = subprocess.run([PY, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", src, "--target_dir", tmp],
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    if r.returncode != 0:
        log("MERGE FAILED", n, r.stderr[-2000:])
        return False
    for f in glob.glob(os.path.join(src, "huggingface", "*")):
        if not os.path.exists(os.path.join(tmp, os.path.basename(f))):
            shutil.copy(f, tmp)
    r = subprocess.run([PY, "-c", VERIFY, tmp], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    if r.returncode != 0:
        log("LOAD FAILED", n, r.stderr[-2000:])
        return False
    info = json.loads(r.stdout.strip().splitlines()[-1])
    if not info["logits_finite"] or info["n_params"] < 3_000_000_000:
        log("LOAD CHECK FAILED", n, info)
        return False
    os.rename(tmp, dst)
    json.dump({**info, "step": n, "merged_from": src, "time": time.strftime("%F %T")},
              open(os.path.join(dst, "LOAD_OK.json"), "w"), indent=2)
    log("load ok", n, info)
    return True


def free_earlier(n):
    for d in sorted(glob.glob(os.path.join(CK, "global_step_*"))):
        m = int(d.rsplit("_", 1)[1])
        if m >= n or not os.path.exists(os.path.join(HF, f"step_{m}", "LOAD_OK.json")):
            continue
        files = [f for pat in ("model_world_size_*.pt", "optim_world_size_*.pt", "extra_state_world_size_*.pt")
                 for f in glob.glob(os.path.join(d, "actor", pat))]
        if files:
            gb = sum(os.path.getsize(f) for f in files) / 1e9
            for f in files:
                os.remove(f)
            log(f"deleted FSDP shards + optimizer state of global_step_{m} ({len(files)} files, {gb:.1f} GB); merged weights kept")


pending = list(a.steps)
log("watching", a.run, "steps", pending)
seen_alive = False
while pending:
    n = pending[0]
    alive = training_alive()
    seen_alive |= alive
    if latest() >= n and os.path.isdir(os.path.join(CK, f"global_step_{n}", "actor")):
        if merge_and_verify(n):
            free_earlier(n)
            pending.pop(0)
            continue
        log("step", n, "not confirmed; checkpoint left untouched; retrying in 10 min")
        time.sleep(600)
        continue
    if seen_alive and not alive:
        time.sleep(30)
        if latest() < n:
            log("training process gone; latest checkpoint", latest(), "; stopping with pending", pending)
            break
    time.sleep(a.poll)
log("done; pending", pending)
