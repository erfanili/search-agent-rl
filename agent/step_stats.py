"""
Per-step depth statistics for GRPO training (new_plan.md repo item 5).

agent/reward.py:compute_score_batch scores a whole training step at once (verl's "batch" reward
manager), calls step_stats on it, and appends one JSON line per step to reward_steps.jsonl in the
run directory. scripts/depth_dynamics.py prints the table.

One line holds, for one call of the reward function:
  split              "train" (one GRPO step) or "val" (a validation pass)
  step               verl's global step: the training step just scored, or the step a validation follows
  n, n_groups        rollouts and prompts (groups) scored
  mean_score, mean_em
  mean_search        mean executed searches per rollout
  depth_hist         {"0": share, "1": share, ...} share of rollouts by search count (4 = 4 or more)
  share_gt           {"0": P(n_search > 0), "1": P(n_search > 1), ...}; p_k = share_gt[k] / share_gt[k-1]
  score_by_depth     {"0": mean score of rollouts with 0 searches, ...} (absent depths left out)
  zero_var_groups    share of groups whose rollouts all got the same score: no GRPO gradient
  all_zero_groups, all_one_groups   the zero-variance groups split by value (score 0 / score >= 1)
  mixed_depth_groups share of groups whose rollouts differ in search count
  depth_contrast_groups  share of groups that differ in search count AND in score: the only groups
                     whose advantage can move the stop-or-continue decision (lock-in mechanism)
  by_label           the same mean_score / mean_search / share_gt["1"] per training-data label value
                     (closed_book_em, d1_hit in extra_info; agent/label_train_data.py), when present

Standard library only; process-global counters live here (not in reward.py) because verl loads
reward.py twice by path, once for training and once for validation, and both copies must share them.
"""

import json
import os
from collections import defaultdict

MAX_DEPTH_BIN = 4
STATE = {"train_calls": 0}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _bin(d):
    return str(min(int(d), MAX_DEPTH_BIN))


def step_stats(scores, ems, n_search, group_keys, labels=None):
    """scores, ems, n_search, group_keys: one entry per rollout. labels: optional list of dicts."""
    n = len(scores)
    out = {"n": n, "mean_score": _mean(scores), "mean_em": _mean(ems), "mean_search": _mean(n_search)}

    hist = defaultdict(int)
    by_depth = defaultdict(list)
    for s, d in zip(scores, n_search):
        hist[_bin(d)] += 1
        by_depth[_bin(d)].append(s)
    out["depth_hist"] = {k: hist[k] / n for k in sorted(hist)}
    out["share_gt"] = {str(k): sum(d > k for d in n_search) / n for k in range(MAX_DEPTH_BIN)}
    out["score_by_depth"] = {k: _mean(v) for k, v in sorted(by_depth.items())}

    groups = defaultdict(list)
    for s, d, g in zip(scores, n_search, group_keys):
        groups[g].append((s, d))
    G = len(groups)
    zero_var = [g for g in groups.values() if len({s for s, _ in g}) == 1]
    out["n_groups"] = G
    out["group_size"] = n / G if G else None
    out["zero_var_groups"] = len(zero_var) / G if G else None
    out["all_zero_groups"] = sum(g[0][0] == 0 for g in zero_var) / G if G else None
    out["all_one_groups"] = sum(g[0][0] >= 1 for g in zero_var) / G if G else None
    mixed = [g for g in groups.values() if len({d for _, d in g}) > 1]
    out["mixed_depth_groups"] = len(mixed) / G if G else None
    out["depth_contrast_groups"] = sum(len({s for s, _ in g}) > 1 for g in mixed) / G if G else None

    if labels and any(labels):
        by_label = {}
        for key in sorted({k for lab in labels if lab for k in lab}):
            vals = defaultdict(list)
            for i, lab in enumerate(labels):
                if lab and lab.get(key) is not None:
                    vals[str(lab[key])].append(i)
            by_label[key] = {
                v: {"n": len(ix), "mean_score": _mean([scores[i] for i in ix]),
                    "mean_search": _mean([n_search[i] for i in ix]),
                    "share_gt1": _mean([float(n_search[i] > 1) for i in ix])}
                for v, ix in sorted(vals.items())
            }
        out["by_label"] = by_label
    return out


def log_call(log_path, split, step_offset, stats):
    """Append one line; split "train" advances the step counter."""
    if split == "train":
        STATE["train_calls"] += 1
    rec = {"split": split, "step": step_offset + STATE["train_calls"], **stats}
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec
