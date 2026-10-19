"""
GRPO reward for the search agent (P1.3), loaded by verl as a custom reward function:

  custom_reward_function.path=agent/reward.py
  custom_reward_function.name=compute_score
  +custom_reward_function.reward_kwargs.metric=em      # em (Search-R1 default), f1 or acc

verl's NaiveRewardManager decodes each trajectory's response (model AND env tokens, special
tokens skipped) and calls compute_score once per trajectory. The scalar goes on the last response
token. The other keys are logged per trajectory (verl's reward_extra_info), which gives search
counts and answer rates during training for free.

Scoring is agent/qa_metrics.score_output, the same function tests/score_predictions.py uses for
evaluation: environment text is stripped, the last <answer> block is the answer, no answer = 0.
Phase 3 adds the other reward shapes here (F1, cost-aware) as further `metric` options.

verl loads this file by path (importlib), not as part of a package, so it puts the repo root on
sys.path itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qa_metrics import score_output  # noqa: E402

METRICS = ("em", "f1", "acc")


def golds_from(ground_truth):
    """Rows store reward_model.ground_truth = {"target": [answers...]}; parquet may hand back a numpy array."""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, str):
        return [ground_truth]
    return [str(g) for g in ground_truth]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, metric="em"):
    if metric not in METRICS:
        raise ValueError(f"metric={metric!r}; expected one of {METRICS}")
    s = score_output(solution_str, golds_from(ground_truth))
    return {
        "score": s[metric],
        "em": s["em"],
        "f1": s["f1"],
        "acc": s["acc"],
        "answered": float(s["answered"]),
        "n_search": float(s["n_search"]),
    }