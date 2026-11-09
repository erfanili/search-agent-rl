"""Reward arms in agent/reward.py, including the adversarial inputs from roadmap P3.3.
Run: .venv/bin/python -m pytest tests/ -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from agent.qa_metrics import SEARCH_R1_INVALID, answer_in_information  # noqa: E402
from agent.reward import REWARDS, compute_score  # noqa: E402

GOLD = {"target": ["Paris"]}


def info(text):
    return f"\n\n<information>Doc 1(Title: T) {text}\n</information>\n\n"


def traj(*parts):
    return "".join(parts)


ONE_SEARCH_RIGHT = traj("<search> capital of France </search>", info("Paris is the capital."), "<answer> Paris </answer>")
TWO_SEARCH_RIGHT = traj("<search> q1 </search>", info("nothing useful"), "<search> q2 </search>", info("Paris."),
                        "<answer> Paris </answer>")
TWO_SAME_QUERY = traj("<search> q1 </search>", info("x"), "<search> Q1. </search>", info("y"), "<answer> Paris </answer>")
EVIDENCE_WRONG_ANSWER = traj("<search> q </search>", info("Paris and Lyon"), "<answer> Lyon </answer>")
NO_SEARCH_RIGHT = "<answer> Paris </answer>"


def score(t, metric, **kw):
    return compute_score("hotpotqa", t, GOLD, metric=metric, **kw)["score"]


@pytest.mark.parametrize("metric", REWARDS)
def test_no_answer_scores_zero_on_every_arm(metric):
    for t in ("", "<search> Paris </search>" + info("Paris"), "<answer> Paris", SEARCH_R1_INVALID):
        if metric == "em_hit" and "information" in t:
            continue  # evidence alone is paid by design
        assert score(t, metric) == 0.0


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        compute_score("x", "", GOLD, metric="bogus")


def test_em_and_f1_ignore_search_count():
    for m in ("em", "f1"):
        assert score(ONE_SEARCH_RIGHT, m) == score(TWO_SEARCH_RIGHT, m) == score(NO_SEARCH_RIGHT, m) == 1.0


def test_em_hit_pays_evidence_even_with_wrong_answer():
    assert score(EVIDENCE_WRONG_ANSWER, "em_hit") == pytest.approx(0.2)
    assert score(ONE_SEARCH_RIGHT, "em_hit") == pytest.approx(1.2)
    assert score(NO_SEARCH_RIGHT, "em_hit") == 1.0          # no retrieval, no bonus
    assert score(EVIDENCE_WRONG_ANSWER, "em_hit", hit_bonus=0.5) == pytest.approx(0.5)


def test_evidence_hit_is_whole_word_and_reads_only_env_text():
    assert answer_in_information(info("Parisian cafe"), ["Paris"]) == 0.0
    assert answer_in_information(info("in paris, france"), ["Paris"]) == 1.0
    # the gold string in the model's own query or answer is not evidence
    assert answer_in_information("<search> Paris </search><answer> Paris </answer>", ["Paris"]) == 0.0


def test_em_multi_pays_only_correct_answers_with_two_distinct_queries():
    assert score(TWO_SEARCH_RIGHT, "em_multi") == pytest.approx(1.2)
    assert score(ONE_SEARCH_RIGHT, "em_multi") == 1.0
    assert score(TWO_SAME_QUERY, "em_multi") == 1.0          # repeating a query (after normalization) doesn't count
    wrong = TWO_SEARCH_RIGHT.replace("<answer> Paris </answer>", "<answer> Lyon </answer>")
    assert score(wrong, "em_multi") == 0.0                    # search spam alone earns nothing


def test_long_answer_containing_gold_is_not_paid_by_em_arms():
    long = "<answer> it is probably Paris or maybe somewhere else in Europe </answer>"
    for m in ("em", "em_hit", "em_multi"):
        assert score(long, m) == 0.0
    assert score(long, "acc") == 1.0                          # why acc is not a training arm


def test_logged_keys():
    out = compute_score("hotpotqa", TWO_SEARCH_RIGHT, GOLD)
    assert out["n_search"] == 2 and out["n_distinct_search"] == 2
    assert out["evidence_hit"] == 1.0 and out["correct_multi"] == 1.0
