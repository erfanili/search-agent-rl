#!/usr/bin/env python
"""
Summarize a P0.7 smoke run: step time breakdown, GPU memory, and whether reward moved.

Usage:
  .venv-verl/bin/python scripts/summarize_smoke.py /data/erfan/logs/p0_7_smoke_<timestamp>

Reads train.log (verl console lines "step:N - key:value - ...") and gpu.csv (nvidia-smi samples).
Writes summary.json and summary.md into the same directory and prints the Markdown.

Two memory numbers, because they measure different things:
  verl perf/max_memory_allocated_gb  peak PyTorch allocation in the training process
                                     (FSDP actor + ref); excludes vLLM's reserved KV cache
  nvidia-smi peak memory.used        everything on the card: what actually limits batch size
"""

import csv
import json
import os
import re
import statistics
import sys

STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")

# verl metric keys; a missing key is reported as absent, not as an error
KEYS = {
    "step_s": "timing_s/step",
    "gen_s": "timing_s/gen",
    "old_log_prob_s": "timing_s/old_log_prob",
    "ref_s": "timing_s/ref",
    "adv_s": "timing_s/adv",
    "update_actor_s": "timing_s/update_actor",
    "reward_mean": "critic/score/mean",
    "response_len_mean": "response_length/mean",
    "response_clip_ratio": "response_length/clip_ratio",
    "entropy": "actor/entropy",
    "kl_loss": "actor/kl_loss",
    "grad_norm": "actor/grad_norm",
    "max_mem_allocated_gb": "perf/max_memory_allocated_gb",
    "max_mem_reserved_gb": "perf/max_memory_reserved_gb",
    "throughput_tok_per_s": "perf/throughput",
}


def parse_train_log(path):
    steps = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            metrics = {}
            for part in m.group(2).split(" - "):
                k, sep, v = part.strip().rpartition(":")
                if sep and NUM_RE.match(v.strip()):
                    metrics[k.strip()] = float(v)
            if metrics:
                steps[int(m.group(1))] = metrics
    return [steps[k] | {"step": k} for k in sorted(steps)]


def parse_gpu_csv(path):
    peak, util = {}, {}
    if not os.path.exists(path):
        return peak, util
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            try:
                gpu, mem, u = int(row[1]), float(row[2]), float(row[3])
            except ValueError:
                continue
            peak[gpu] = max(peak.get(gpu, 0.0), mem)
            util.setdefault(gpu, []).append(u)
    return {g: round(v / 1024, 2) for g, v in sorted(peak.items())}, {
        g: round(statistics.mean(v), 1) for g, v in sorted(util.items())
    }


def col(rows, key):
    return [r[key] for r in rows if key in r]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    rows = parse_train_log(os.path.join(out, "train.log"))
    if not rows:
        sys.exit("no 'step:N - ...' lines in train.log: the run failed before step 1, read the log tail")
    peak_gb, util = parse_gpu_csv(os.path.join(out, "gpu.csv"))

    steady = rows[1:] if len(rows) > 1 else rows  # step 1 includes warm-up (CUDA graphs, compile)
    summary = {"steps_logged": len(rows), "first_step": rows[0]["step"], "last_step": rows[-1]["step"]}
    for name, key in KEYS.items():
        vals = col(steady, key)
        if vals:
            summary[f"{name}_mean_excl_step1"] = round(statistics.mean(vals), 3)
        allv = col(rows, key)
        if allv and name in {"reward_mean", "response_len_mean", "entropy"}:
            summary[f"{name}_first"] = round(allv[0], 4)
            summary[f"{name}_last"] = round(allv[-1], 4)
        if allv and name.startswith("max_mem"):
            summary[f"{name}_peak"] = round(max(allv), 2)
    if col(rows[:1], KEYS["step_s"]):
        summary["step1_s"] = round(rows[0][KEYS["step_s"]], 1)
    summary["nvidia_smi_peak_used_gb"] = peak_gb
    summary["nvidia_smi_mean_util_pct"] = util

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    s = summary
    def g(k, fmt="{:.2f}"):
        return fmt.format(s[k]) if k in s else "n/a"

    step = s.get("step_s_mean_excl_step1")
    parts = ["gen_s", "old_log_prob_s", "ref_s", "update_actor_s"]
    breakdown = " | ".join(
        f"{p[:-2]} {s[p + '_mean_excl_step1']:.1f}s ({100 * s[p + '_mean_excl_step1'] / step:.0f}%)"
        for p in parts
        if step and p + "_mean_excl_step1" in s
    )
    md = [
        f"## P0.7 smoke run: {os.path.basename(out.rstrip('/'))}",
        "",
        f"- Steps logged: {s['steps_logged']} ({s['first_step']}-{s['last_step']})",
        f"- Step time: step 1 {g('step1_s', '{:.1f}')} s, then {g('step_s_mean_excl_step1', '{:.1f}')} s mean",
        f"- Breakdown: {breakdown or 'n/a'}",
        f"- Throughput: {g('throughput_tok_per_s_mean_excl_step1', '{:.0f}')} tokens/s per GPU (verl perf/throughput)",
        f"- Peak torch allocated: {g('max_mem_allocated_gb_peak')} GB; reserved: {g('max_mem_reserved_gb_peak')} GB",
        f"- Peak nvidia-smi used per GPU (GB): {peak_gb or 'n/a'}",
        f"- Mean GPU utilization (%): {util or 'n/a'}",
        f"- Reward: {g('reward_mean_first', '{:.3f}')} -> {g('reward_mean_last', '{:.3f}')}",
        f"- Response length: {g('response_len_mean_first', '{:.0f}')} -> {g('response_len_mean_last', '{:.0f}')} tokens;"
        f" clip ratio {g('response_clip_ratio_mean_excl_step1', '{:.2f}')}",
        f"- Entropy: {g('entropy_first', '{:.3f}')} -> {g('entropy_last', '{:.3f}')}",
    ]
    text = "\n".join(md) + "\n"
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()