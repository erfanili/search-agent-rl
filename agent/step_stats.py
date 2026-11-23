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
  by_label           per training-data label value (closed_book_em, d1_hit in extra_info, agent/label_train_data.py;
                     e004_label, scripts/exp/E004_build_data.py), when present: n, mean_score, mean_search,
                     share_gt1 (P(n_search > 1)), share_search (P(n_search >= 1)), mean_score_0search /
                     mean_score_search (mean score of rollouts with 0 / >= 1 searches), n_groups,
                     n_groups_mixed_search (groups where some rollouts search and some don't),
                     n_groups_mixed_search_and_score (of those, groups whose scores also differ)
  answer             (when compute_score_batch passes answer spans; E004) parsed answer span = the first <answer>
                     span of the model text (environment text stripped; unclosed -> text to the end), the span
                     the alias_match reward reads: {"all": {...}, "by_label": {key: {value: {...}}}} with
                     n_answer (rollouts with an <answer>), n_no_answer, mean_words / p95_words (whitespace word
                     count of the span; p95 linear interpolation), share_unclosed (answers with no </answer> after
                     the first <answer>, over n_answer)
  components         (E005; compute_score_batch) reward components by executed-search decision (d = at least one
                     <information> block): {"all": cell, "by_label": {key: {value: cell}}}, cell = n, share_search,
                     mean_{outcome,arm,total} over all rollouts and _0search / _search (0 vs >= 1 executed searches),
                     n_0search, n_search, n_groups, n_groups_mixed_search, n_groups_mixed_search_and_total (of those,
                     groups whose total rewards also differ)

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


def _label_stats(ix, scores, n_search, group_keys):
    """Rollouts ix of one label value. Groups = the GRPO groups those rollouts belong to."""
    s0 = [scores[i] for i in ix if n_search[i] == 0]
    s1 = [scores[i] for i in ix if n_search[i] > 0]
    groups = defaultdict(list)
    for i in ix:
        groups[group_keys[i]].append((scores[i], n_search[i] > 0))
    mixed = [g for g in groups.values() if len({srch for _, srch in g}) > 1]
    return {"n": len(ix), "mean_score": _mean([scores[i] for i in ix]), "mean_search": _mean([n_search[i] for i in ix]),
            "share_gt1": _mean([float(n_search[i] > 1) for i in ix]),
            "share_search": _mean([float(n_search[i] > 0) for i in ix]),
            "mean_score_0search": _mean(s0), "mean_score_search": _mean(s1),
            "n_groups": len(groups), "n_groups_mixed_search": len(mixed),
            "n_groups_mixed_search_and_score": sum(len({sc for sc, _ in g}) > 1 for g in mixed)}


def _p95(xs):
    """95th percentile, linear interpolation between order statistics (numpy's default)."""
    if not xs:
        return None
    v = sorted(xs)
    h = (len(v) - 1) * 0.95
    lo = int(h)
    return v[lo] + (h - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])


def _median(xs):
    if not xs:
        return None
    v = sorted(xs)
    m = len(v) // 2
    return float(v[m]) if len(v) % 2 else (v[m - 1] + v[m]) / 2


def _answer_cell(ix, words, unclosed):
    w = [words[i] for i in ix if words[i] is not None]
    u = [float(unclosed[i]) for i in ix if words[i] is not None]
    return {"n_answer": len(w), "n_no_answer": len(ix) - len(w), "mean_words": _mean(w), "p95_words": _p95(w),
            "median_words": _median(w), "max_words": max(w) if w else None,
            "share_over_20": _mean([float(x > 20) for x in w]), "share_unclosed": _mean(u)}


UNCLOSED_CAP = 50  # E006: unclosed answer spans count at most 50 words in the capped length statistics


def capped_words(words, unclosed, cap=UNCLOSED_CAP):
    return [None if w is None else (min(w, cap) if u else w) for w, u in zip(words, unclosed)]


def answer_stats(words, unclosed, labels=None):
    """words[i]: word count of rollout i's parsed answer span (None = no <answer>); unclosed[i]: bool."""
    out = {"all": _answer_cell(range(len(words)), words, unclosed)}
    if labels and any(labels):
        by_label = {}
        for key in sorted({k for lab in labels if lab for k in lab}):
            vals = defaultdict(list)
            for i, lab in enumerate(labels):
                if lab and lab.get(key) is not None:
                    vals[str(lab[key])].append(i)
            by_label[key] = {v: _answer_cell(ix, words, unclosed) for v, ix in sorted(vals.items())}
        out["by_label"] = by_label
    return out


def _component_cell(ix, outcome, term, total, n_exec, group_keys):
    srch = {i: n_exec[i] >= 1 for i in ix}
    c = {"n": len(ix), "share_search": _mean([float(srch[i]) for i in ix])}
    for name, v in (("outcome", outcome), ("arm", term), ("total", total)):
        c[f"mean_{name}"] = _mean([v[i] for i in ix])
        c[f"mean_{name}_0search"] = _mean([v[i] for i in ix if not srch[i]])
        c[f"mean_{name}_search"] = _mean([v[i] for i in ix if srch[i]])
    c["n_0search"] = sum(not srch[i] for i in ix)
    c["n_search"] = sum(srch[i] for i in ix)
    groups = defaultdict(list)
    for i in ix:
        groups[group_keys[i]].append((total[i], srch[i]))
    mixed = [g for g in groups.values() if len({d for _, d in g}) > 1]
    c["n_groups"] = len(groups)
    c["n_groups_mixed_search"] = len(mixed)
    c["n_groups_mixed_search_and_total"] = sum(len({t for t, _ in g}) > 1 for g in mixed)
    return c


def component_stats(outcome, term, total, n_exec, group_keys, labels=None):
    """One entry per rollout: outcome reward, arm term, total reward, executed searches, GRPO group key."""
    out = {"all": _component_cell(range(len(total)), outcome, term, total, n_exec, group_keys)}
    if labels and any(labels):
        by_label = {}
        for key in sorted({k for lab in labels if lab for k in lab}):
            vals = defaultdict(list)
            for i, lab in enumerate(labels):
                if lab and lab.get(key) is not None:
                    vals[str(lab[key])].append(i)
            by_label[key] = {v: _component_cell(ix, outcome, term, total, n_exec, group_keys) for v, ix in sorted(vals.items())}
        out["by_label"] = by_label
    return out


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
            by_label[key] = {v: _label_stats(ix, scores, n_search, group_keys) for v, ix in sorted(vals.items())}
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
