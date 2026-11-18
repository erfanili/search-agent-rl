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
Shaped reward arms (em_hit, em_multi) are on the archive/depth-study branch.

Batch entry point. compute_score_batch (verl's "batch" reward manager, the launcher's default) scores a
whole step with compute_score, row by row, so rewards are identical; it also writes one line of depth
statistics per training step and per validation pass to log_path (agent/step_stats.py), which needs
the whole GRPO group at once.

verl loads this file by path (importlib), not as part of a package, so it puts the repo root on
sys.path itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qa_metrics import answer_in_information, distinct_searches, score_output, strip_environment_text  # noqa: E402
from agent.step_stats import log_call, step_stats  # noqa: E402

REWARDS = ("em", "f1")


def golds_from(ground_truth):
    """Rows store reward_model.ground_truth = {"target": [answers...]}; parquet may hand back a numpy array."""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, str):
        return [ground_truth]
    return [str(g) for g in ground_truth]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, metric="em"):
    if metric not in REWARDS:
        raise ValueError(f"metric={metric!r}; expected one of {REWARDS}")
    golds = golds_from(ground_truth)
    s = score_output(solution_str, golds)
    hit = answer_in_information(solution_str, golds)
    n_distinct = distinct_searches(strip_environment_text(solution_str))
    return {
        "score": s[metric],
        "em": s["em"],
        "f1": s["f1"],
        "acc": s["acc"],
        "answered": float(s["answered"]),
        "n_search": float(s["n_search"]),
        "n_distinct_search": float(n_distinct),
        "evidence_hit": hit,
    }


LABEL_KEYS = ("closed_book_em", "d1_hit")   # training-data labels in extra_info (agent/label_train_data.py)


def _extra(e):
    return e if isinstance(e, dict) else {}


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None, log_path=None, step_offset=0,
                        **kwargs):
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
        stats["data_sources"] = {str(d): sum(1 for x in data_sources if x == d) for d in sorted(set(map(str, data_sources)))}
        log_call(log_path, split, int(step_offset), stats)
    except Exception as exc:  # statistics must never stop training
        print(f"compute_score_batch: step stats failed: {exc!r}")
    return results
