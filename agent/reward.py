"""
GRPO reward for the search agent (P1.3; reward arms added for the multi-hop study), loaded by verl as
a custom reward function:

  custom_reward_function.path=agent/reward.py
  custom_reward_function.name=compute_score
  +custom_reward_function.reward_kwargs.metric=em      # one of REWARDS below

verl's NaiveRewardManager decodes each trajectory's response (model AND env tokens, special
tokens skipped) and calls compute_score once per trajectory. The scalar goes on the last response
token. The other keys are logged per trajectory (verl's reward_extra_info), which gives search
counts and answer rates during training for free.

Scoring is agent/qa_metrics.score_output, the same function tests/score_predictions.py uses for
evaluation: environment text is stripped, the last <answer> block is the answer, no answer = 0.

Reward arms. The 60-step EM run converged to "search once, answer": with every rollout in a group
searching once, GRPO never sees a one-vs-two-search contrast. Each arm below adds a different
signal for a second search; evaluation is EM on every arm.
  em          EM (Search-R1; the 60-step run)
  f1          token F1: partial credit for getting part of a multi-hop answer
  acc         cover match (lenient; kept for comparison, hackable by long answers)
  em_hit      EM + hit_bonus * [a gold answer appears in some retrieved passage]: rewards the search
              that finds the evidence, so a second query that succeeds where the first failed is paid
              even when the final answer is wrong
  em_multi    EM + multi_bonus * [EM = 1 and >= 2 distinct queries]: pays only correct answers reached
              with more than one query, so spamming searches earns nothing on its own

verl loads this file by path (importlib), not as part of a package, so it puts the repo root on
sys.path itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qa_metrics import answer_in_information, distinct_searches, score_output, strip_environment_text  # noqa: E402

REWARDS = ("em", "f1", "acc", "em_hit", "em_multi")


def golds_from(ground_truth):
    """Rows store reward_model.ground_truth = {"target": [answers...]}; parquet may hand back a numpy array."""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, str):
        return [ground_truth]
    return [str(g) for g in ground_truth]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, metric="em", hit_bonus=0.2, multi_bonus=0.2):
    if metric not in REWARDS:
        raise ValueError(f"metric={metric!r}; expected one of {REWARDS}")
    golds = golds_from(ground_truth)
    s = score_output(solution_str, golds)
    hit = answer_in_information(solution_str, golds)
    n_distinct = distinct_searches(strip_environment_text(solution_str))
    multi = float(s["em"] == 1 and n_distinct >= 2)
    score = {
        "em": s["em"],
        "f1": s["f1"],
        "acc": s["acc"],
        "em_hit": s["em"] + hit_bonus * hit,
        "em_multi": s["em"] + multi_bonus * multi,
    }[metric]
    return {
        "score": score,
        "em": s["em"],
        "f1": s["f1"],
        "acc": s["acc"],
        "answered": float(s["answered"]),
        "n_search": float(s["n_search"]),
        "n_distinct_search": float(n_distinct),
        "evidence_hit": hit,
        "correct_multi": multi,
    }
