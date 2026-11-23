"""
GRPO reward for the search agent (P1.3), loaded by verl as a custom reward function:

  custom_reward_function.path=agent/reward.py
  custom_reward_function.name=compute_score
  +custom_reward_function.reward_kwargs.metric=em      # one of REWARDS below

verl's NaiveRewardManager decodes each trajectory's response (model AND env tokens, special
tokens skipped) and calls compute_score once per trajectory. The scalar goes on the last response
token. The other keys are logged per trajectory (verl's reward_extra_info), which gives search
counts and answer rates during training for free.

Scoring is agent/qa_metrics.score_output, the same function tests/score_predictions.py uses for
evaluation: environment text is stripped, the last <answer> block is the answer, no answer = 0.

Rewards (evaluation is EM either way):
  em          exact match (Search-R1)
  f1          token F1: partial credit for getting part of a multi-hop answer
  alias_match the E001 correctness rule (exps/E001, E004): environment text stripped, the FIRST <answer>
              span (an <answer> that never closes -> the text to the end), then cover match (a normalized
              gold alias equals or is a substring of the normalized span); no <answer> = 0. Unlike em/f1
              it reads the first answer, not the last, and pays an unclosed answer.
  alias_match_strict (E005) 1 only if the first <answer> span is closed AND, after normalize_answer, a gold
              alias equals the span, or a gold alias is a (character) substring of the span and the span has at
              most (alias word count + 3) words. Unclosed or no <answer> = 0.

E005 arm terms (reward_kwargs arm=C|S|O, plus cost / lam), added to the outcome reward:
  executed searches n = <information> blocks the environment inserted (one per executed search); d = [n >= 1]
  C  -cost * n                                       (cost 0.15)
  S  lam * (d * (1 - a) + (1 - d) * a)               a = extra_info["sc_agreement"] (scripts/exp/E005_sc.py), lam 0.3
  O  the S formula with a = 1 for e004_label knowable and a = 0 for unknowable
  score = outcome + arm term; outcome and arm_term are logged per rollout and per step (step_stats "components").
Shaped reward arms (em_hit, em_multi) are on the archive/depth-study branch.

Batch entry point. compute_score_batch (verl's "batch" reward manager, the launcher's default) scores a
whole step with compute_score, row by row, so rewards are identical; it also writes one line of depth
statistics per training step and per validation pass to log_path (agent/step_stats.py), which needs
the whole GRPO group at once.

verl loads this file by path (importlib), not as part of a package, so it puts the repo root on
sys.path itself.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qa_metrics import (  # noqa: E402
    INFO_RE, answer_in_information, normalize_answer, cover_match, distinct_searches, score_output, strip_environment_text,
)
from agent.step_stats import answer_stats, capped_words, component_stats, log_call, step_stats  # noqa: E402

REWARDS = ("em", "f1", "alias_match", "alias_match_strict")
ARMS = (None, "C", "S", "O")


def first_answer_span(text):
    """E001 parse rule: the first <answer>...</answer>; <answer> never closed -> text to the end; None if no <answer>."""
    i = text.find("<answer>")
    if i < 0:
        return None
    rest = text[i + len("<answer>"):]
    j = rest.find("</answer>")
    return (rest[:j] if j >= 0 else rest).strip()


def answer_span_info(solution_str):
    """(word count of the parsed answer span, unclosed) for the span alias_match reads; (None, None) if no <answer>."""
    text = strip_environment_text(solution_str)
    i = text.find("<answer>")
    if i < 0:
        return None, None
    return len(first_answer_span(text).split()), text.find("</answer>", i + len("<answer>")) < 0


def alias_match(solution_str, golds):
    span = first_answer_span(strip_environment_text(solution_str))
    return 0.0 if span is None else cover_match(span, golds)


def alias_match_strict(solution_str, golds):
    text = strip_environment_text(solution_str)
    i = text.find("<answer>")
    if i < 0:
        return 0.0
    rest = text[i + len("<answer>"):]
    j = rest.find("</answer>")
    if j < 0:
        return 0.0
    span = normalize_answer(rest[:j])
    n_words = len(span.split())
    for gold in golds:
        g = normalize_answer(gold)
        if g and (g == span or (g in span and n_words <= len(g.split()) + 3)):
            return 1.0
    return 0.0


def executed_searches(solution_str):
    """Searches the environment executed = <information> blocks it inserted into the trajectory."""
    return len(INFO_RE.findall(solution_str or ""))


def arm_term(arm, n_exec, extra_info, lam=0.3, cost=0.15):
    if arm is None:
        return 0.0
    if arm == "C":
        return -cost * n_exec
    d = 1.0 if n_exec >= 1 else 0.0
    if arm == "S":
        a = float(extra_info["sc_agreement"])
    elif arm == "O":
        a = {"knowable": 1.0, "unknowable": 0.0}[extra_info["e004_label"]]
    else:
        raise ValueError(f"arm={arm!r}; expected one of {ARMS}")
    return lam * (d * (1.0 - a) + (1.0 - d) * a)


def golds_from(ground_truth):
    """Rows store reward_model.ground_truth = {"target": [answers...]}; parquet may hand back a numpy array."""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, str):
        return [ground_truth]
    return [str(g) for g in ground_truth]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, metric="em", arm=None, lam=0.3, cost=0.15):
    if metric not in REWARDS:
        raise ValueError(f"metric={metric!r}; expected one of {REWARDS}")
    if arm in ("", "none", "None"):
        arm = None
    if arm not in ARMS:
        raise ValueError(f"arm={arm!r}; expected one of {ARMS}")
    golds = golds_from(ground_truth)
    s = score_output(solution_str, golds)
    hit = answer_in_information(solution_str, golds)
    n_distinct = distinct_searches(strip_environment_text(solution_str))
    am = alias_match(solution_str, golds)
    ams = alias_match_strict(solution_str, golds)
    outcome = {"alias_match": am, "alias_match_strict": ams}.get(metric, s.get(metric))
    n_exec = executed_searches(solution_str)
    term = arm_term(arm, n_exec, extra_info or {}, lam=float(lam), cost=float(cost))
    return {
        "score": outcome + term,
        "outcome": outcome,
        "arm_term": term,
        "n_exec_search": float(n_exec),
        "alias_match_strict": ams,
        "em": s["em"],
        "f1": s["f1"],
        "acc": s["acc"],
        "answered": float(s["answered"]),
        "n_search": float(s["n_search"]),
        "n_distinct_search": float(n_distinct),
        "evidence_hit": hit,
        "alias_match": am,
    }


LABEL_KEYS = ("closed_book_em", "d1_hit",   # training-data labels in extra_info (agent/label_train_data.py)
              "e004_label")                # knowable / unknowable (scripts/exp/E004_build_data.py)


def _extra(e):
    return e if isinstance(e, dict) else {}


def write_step_samples(sample_path, step, split, data_sources, solution_strs, extras, results, n=50):
    """Append n randomly chosen rollouts of this reward call (all if fewer) to sample_path, one JSON line each:
    step, split, row position, id / label / index, data source, full response text, score components and answer-span
    info. Random choice: numpy default_rng(step) over row positions, without replacement (E005 diagnosis request,
    2026-11-22: rollout text at every step, so stop-rule steps can be inspected)."""
    import json, os
    import numpy as np
    m = len(solution_strs)
    pick = sorted(np.random.default_rng(int(step)).choice(m, size=min(n, m), replace=False).tolist())
    os.makedirs(os.path.dirname(sample_path) or ".", exist_ok=True)
    with open(sample_path, "a") as f:
        for i in pick:
            e, r = extras[i], results[i]
            words, unclosed = answer_span_info(solution_strs[i])
            f.write(json.dumps({
                "step": int(step), "split": split, "row": i, "id": e.get("id"), "index": e.get("index"),
                "label": {k: e[k] for k in LABEL_KEYS if k in e}, "data_source": str(data_sources[i]),
                "score": r["score"], "outcome": r.get("outcome"), "arm_term": r.get("arm_term"),
                "n_exec_search": r.get("n_exec_search"), "answer_words": words, "answer_unclosed": unclosed,
                "text": solution_strs[i]}, ensure_ascii=False) + "\n")


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None, log_path=None, step_offset=0,
                        sample_path=None, sample_n=50, garble_dir=None, garble_warn=0.10, **kwargs):
    """verl BatchRewardManager entry: compute_score per row, plus one step_stats line in log_path.
    Rollouts of one prompt share extra_info["index"] (verl repeats each row n times), which defines the
    GRPO group. split: extra_info["split"] if set, else "train" when some prompt has more than one rollout."""
    extras = [_extra(e) for e in (extra_infos if extra_infos is not None else [None] * len(solution_strs))]
    results = [compute_score(ds, s, gt, e, **kwargs) for ds, s, gt, e in zip(data_sources, solution_strs, ground_truths, extras)]
    keys = [e.get("index", e.get("id", i)) for i, e in enumerate(extras)]
    split = next((e["split"] for e in extras if e.get("split")), None)
    if split is None:
        split = "train" if len(set(keys)) < len(keys) else "val"
    try:
        labels = [{k: e[k] for k in LABEL_KEYS if k in e} for e in extras]
        stats = step_stats([r["score"] for r in results], [r["em"] for r in results],
                           [r["n_search"] for r in results], keys, labels)
        info = [answer_span_info(s) for s in solution_strs]
        stats["answer"] = answer_stats([w for w, _ in info], [u for _, u in info], labels)
        # E006: unclosed spans capped at 50 words; per-label capped word lists for pooled reference medians
        wc = capped_words([w for w, _ in info], [u for _, u in info])
        stats["answer_capped"] = answer_stats(wc, [u for _, u in info], labels)
        labs = sorted({str(l["e004_label"]) for l in labels if l.get("e004_label") is not None})
        stats["answer_words_capped"] = {lab: [w for w, l in zip(wc, labels) if str(l.get("e004_label")) == lab and w is not None]
                                        for lab in labs}
        stats["components"] = component_stats([r["outcome"] for r in results], [r["arm_term"] for r in results],
                                              [r["score"] for r in results], [r["n_exec_search"] for r in results],
                                              keys, labels)
        stats["data_sources"] = {str(d): sum(1 for x in data_sources if x == d) for d in sorted(set(map(str, data_sources)))}
        if garble_dir and split == "train":
            try:
                from agent.garble import by_server, read_new_events
                stats["garble"] = by_server(read_new_events(garble_dir), float(garble_warn))
            except Exception as exc:  # diagnostics must never stop training
                print(f"compute_score_batch: garble stats failed: {exc!r}")
        rec = log_call(log_path, split, int(step_offset), stats)
        g = stats.get("garble")
        if g and g["warn_servers"]:
            msg = (f"GARBLE WARNING step {rec['step']}: vLLM server(s) {g['warn_servers']} garbled share > {g['warn_share']}: "
                   + ", ".join(f"server {r} {c['garbled']}/{c['n']}" for r, c in g["by_server"].items()))
            print(msg, flush=True)
            with open(os.path.join(garble_dir, "warnings.log"), "a") as f:
                f.write(msg + "\n")
    except Exception as exc:  # statistics must never stop training
        print(f"compute_score_batch: step stats failed: {exc!r}")
        rec = None
    if sample_path:
        try:
            step = rec["step"] if rec else -1
            write_step_samples(sample_path, step, split, data_sources, solution_strs, extras, results, n=int(sample_n))
        except Exception as exc:  # sample dumping must never stop training
            print(f"compute_score_batch: sample dump failed: {exc!r}")
    return results
