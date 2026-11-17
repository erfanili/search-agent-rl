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


# ---- batch entry point and per-step depth statistics (agent/step_stats.py) ------------------

def test_batch_scores_equal_row_scores_and_log_one_line_per_step(tmp_path):
    import json

    from agent import step_stats
    from agent.reward import compute_score_batch

    step_stats.STATE["train_calls"] = 0
    # two groups of 3: group 0 mixes depth and score, group 1 is all-correct at one search (zero variance)
    sols = [ONE_SEARCH_RIGHT, TWO_SEARCH_RIGHT, EVIDENCE_WRONG_ANSWER] + [ONE_SEARCH_RIGHT] * 3
    extras = [{"index": 0, "d1_hit": 1}] * 3 + [{"index": 1, "d1_hit": 0}] * 3
    log = tmp_path / "reward_steps.jsonl"
    kw = dict(metric="em", log_path=str(log), step_offset=40)
    out = compute_score_batch(["nq"] * 6, sols, [GOLD] * 6, extras, **kw)
    assert out == [compute_score("nq", s, GOLD, e, metric="em") for s, e in zip(sols, extras)]
    compute_score_batch(["hotpotqa"] * 2, sols[:2], [GOLD] * 2, [{"index": 7}, {"index": 8}], **kw)  # n=1: val

    train, val = [json.loads(line) for line in open(log)]
    assert (train["split"], train["step"], val["split"], val["step"]) == ("train", 41, "val", 41)
    assert train["n"] == 6 and train["n_groups"] == 2 and train["group_size"] == 3
    assert train["mean_search"] == pytest.approx(7 / 6)
    assert train["share_gt"]["0"] == 1 and train["share_gt"]["1"] == pytest.approx(1 / 6)
    assert train["score_by_depth"] == {"1": pytest.approx(4 / 5), "2": 1.0}
    assert train["zero_var_groups"] == 0.5 and train["all_one_groups"] == 0.5 and train["all_zero_groups"] == 0
    assert train["mixed_depth_groups"] == 0.5 and train["depth_contrast_groups"] == 0.5
    assert train["by_label"]["d1_hit"]["0"]["mean_score"] == 1.0
    assert train["by_label"]["d1_hit"]["1"]["share_gt1"] == pytest.approx(1 / 3)


def test_batch_stats_failure_never_breaks_rewards():
    from agent.reward import compute_score_batch

    out = compute_score_batch(["nq"], [ONE_SEARCH_RIGHT], [GOLD], [{"index": 0}], metric="em",
                              log_path="/proc/forbidden/x.jsonl")
    assert out[0]["score"] == 1.0
