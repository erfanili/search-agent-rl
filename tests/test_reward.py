"""Rewards in agent/reward.py, including the adversarial inputs from roadmap P3.3.
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
def test_no_answer_scores_zero(metric):
    for t in ("", "<search> Paris </search>" + info("Paris"), "<answer> Paris", SEARCH_R1_INVALID):
        if metric == "alias_match" and t == "<answer> Paris":
            continue  # E001 rule: an unclosed <answer> is the text to the end (test_alias_match_e001_rule)
        assert score(t, metric) == 0.0


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        compute_score("x", "", GOLD, metric="bogus")


def test_em_and_f1_ignore_search_count():
    for m in ("em", "f1"):
        assert score(ONE_SEARCH_RIGHT, m) == score(TWO_SEARCH_RIGHT, m) == score(NO_SEARCH_RIGHT, m) == 1.0


def test_alias_match_e001_rule():
    m = "alias_match"
    assert score(NO_SEARCH_RIGHT, m) == score(ONE_SEARCH_RIGHT, m) == 1.0
    assert score("<answer> it is probably Paris, France </answer>", m) == 1.0   # alias substring of the span
    assert score("<answer> the city of paris. </answer>", m) == 1.0             # normalized
    assert score("<answer> Lyon </answer><answer> Paris </answer>", m) == 0.0   # FIRST answer only
    assert score("<think> Paris? </think><answer> Lyon </answer>", m) == 0.0    # never the full output
    assert score("<answer> Paris", m) == 1.0                                    # unclosed: text to the end
    assert score(EVIDENCE_WRONG_ANSWER, m) == 0.0                               # evidence alone pays nothing
    assert score(traj("<search> q </search>", info("<answer> Paris </answer>"), "<answer> Lyon </answer>"), m) == 0.0
    assert score("<answer>  </answer>", m) == 0.0
    assert score("<answer> Par </answer>", m) == 0.0                            # span inside the alias does not count
    assert compute_score("x", NO_SEARCH_RIGHT, {"target": ["France", "Paris"]}, metric=m)["score"] == 1.0
    assert compute_score("x", NO_SEARCH_RIGHT, GOLD)["alias_match"] == 1.0      # logged under every metric


def test_evidence_hit_is_whole_word_and_reads_only_env_text():
    assert answer_in_information(info("Parisian cafe"), ["Paris"]) == 0.0
    assert answer_in_information(info("in paris, france"), ["Paris"]) == 1.0
    # the gold string in the model's own query or answer is not evidence
    assert answer_in_information("<search> Paris </search><answer> Paris </answer>", ["Paris"]) == 0.0


def test_long_answer_containing_gold_is_not_paid_by_em():
    long = "<answer> it is probably Paris or maybe somewhere else in Europe </answer>"
    out = compute_score("hotpotqa", long, GOLD)
    assert out["score"] == 0.0 and out["acc"] == 1.0          # why lenient cover match is not a reward


def test_logged_keys():
    out = compute_score("hotpotqa", TWO_SEARCH_RIGHT, GOLD)
    assert out["n_search"] == 2 and out["n_distinct_search"] == 2
    assert out["evidence_hit"] == 1.0
    assert compute_score("hotpotqa", TWO_SAME_QUERY, GOLD)["n_distinct_search"] == 1   # repeats count once


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


def test_step_stats_per_label_search_and_group_counts():
    from agent.step_stats import step_stats
    # group a (knowable): 2 search, 1 not; scores differ -> mixed in search and score
    # group b (knowable): all search -> not mixed
    # group c (unknowable): 1 search, 2 not; all score 0 -> mixed in search only
    scores = [1, 0, 1, 1, 1, 1, 0, 0, 0]
    n_search = [1, 2, 0, 1, 1, 1, 1, 0, 0]
    keys = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    labels = [{"e004_label": "knowable"}] * 6 + [{"e004_label": "unknowable"}] * 3
    by = step_stats(scores, scores, n_search, keys, labels)["by_label"]["e004_label"]
    k, u = by["knowable"], by["unknowable"]
    assert k["n"] == 6 and k["share_search"] == pytest.approx(5 / 6) and k["share_gt1"] == pytest.approx(1 / 6)
    assert k["mean_score_0search"] == 1.0 and k["mean_score_search"] == pytest.approx(4 / 5)
    assert (k["n_groups"], k["n_groups_mixed_search"], k["n_groups_mixed_search_and_score"]) == (2, 1, 1)
    assert u["share_search"] == pytest.approx(1 / 3) and u["mean_score_0search"] == 0.0 and u["mean_score_search"] == 0.0
    assert (u["n_groups"], u["n_groups_mixed_search"], u["n_groups_mixed_search_and_score"]) == (1, 1, 0)


def test_batch_logs_e004_label(tmp_path):
    from agent import step_stats
    from agent.reward import compute_score_batch
    step_stats.STATE["train_calls"] = 0
    log = tmp_path / "steps.jsonl"
    sols = [NO_SEARCH_RIGHT, ONE_SEARCH_RIGHT, "<answer> Lyon </answer>", ONE_SEARCH_RIGHT]
    extras = [{"index": 0, "e004_label": "knowable"}] * 2 + [{"index": 1, "e004_label": "unknowable"}] * 2
    out = compute_score_batch(["x"] * 4, sols, [GOLD] * 4, extras, log_path=str(log), metric="alias_match")
    assert [r["score"] for r in out] == [1.0, 1.0, 0.0, 1.0]
    import json
    rec = json.loads(log.read_text().splitlines()[0])
    u = rec["by_label"]["e004_label"]["unknowable"]
    assert u["n_groups_mixed_search_and_score"] == 1 and u["share_search"] == 0.5


def test_batch_logs_answer_span_stats(tmp_path):
    import json
    from agent import step_stats
    from agent.reward import answer_span_info, compute_score_batch
    assert answer_span_info("no tags") == (None, None)
    assert answer_span_info("<answer> the city of Paris </answer> <answer> x") == (4, False)
    assert answer_span_info("<think> t </think> <answer> Paris is the") == (3, True)
    assert step_stats._p95([1, 2, 3, 4, 5]) == 4.8
    step_stats.STATE["train_calls"] = 0
    log = tmp_path / "steps.jsonl"
    sols = ["<answer> Paris </answer>", "<answer> Paris France", "no answer", "<answer> a b c d </answer>"]
    extras = [{"index": 0, "e004_label": "knowable"}] * 2 + [{"index": 1, "e004_label": "unknowable"}] * 2
    compute_score_batch(["x"] * 4, sols, [GOLD] * 4, extras, log_path=str(log), metric="alias_match")
    a = json.loads(log.read_text().splitlines()[0])["answer"]
    assert a["all"]["n_answer"] == 3 and a["all"]["n_no_answer"] == 1
    k, u = a["by_label"]["e004_label"]["knowable"], a["by_label"]["e004_label"]["unknowable"]
    assert k["mean_words"] == 1.5 and k["share_unclosed"] == 0.5
    assert u["n_answer"] == 1 and u["mean_words"] == 4 and u["share_unclosed"] == 0.0


# ---- E005: alias_match_strict, decision term (arms S, O), cost term (arm C), component logging ----------

def test_alias_match_strict_rule():
    from agent.reward import alias_match_strict as ams
    g = ["Paris", "New York City"]
    assert ams("<answer> Paris </answer>", g) == 1.0                       # equal
    assert ams("<answer> The Paris. </answer>", g) == 1.0                  # equal after normalize_answer
    assert ams("<answer> Paris </answer>", ["paris!"]) == 1.0
    assert ams("<answer> Paris </answer>", g) == ams("<think> x </think><answer> Paris </answer>", g)
    assert ams("<answer> Paris France capital city </answer>", g) == 1.0   # alias 1 word + 3 = 4 words: allowed
    assert ams("<answer> Paris is the capital city </answer>", g) == 1.0   # "the" removed -> 4 words
    assert ams("<answer> it is Paris in France now </answer>", g) == 0.0   # 6 words > 4
    assert ams("<answer> in New York City USA now </answer>", g) == 1.0   # 3 + 3 = 6 words
    assert ams("<answer> Paris", g) == 0.0                                 # unclosed
    assert ams("no answer at all Paris", g) == 0.0
    assert ams("<answer> Lyon </answer> <answer> Paris </answer>", g) == 0.0   # first span only
    assert ams("<search> q </search>" + info("Paris") + "<answer> Lyon </answer>", g) == 0.0   # env text ignored
    assert ams("<answer> Parisian </answer>", g) == 1.0                    # substring (character), 1 word
    assert ams("<answer>  </answer>", g) == 0.0
    assert ams("<answer> Paris </answer>", [""]) == 0.0


def test_strict_outcome_and_no_arm_term_by_default():
    out = compute_score("x", ONE_SEARCH_RIGHT, GOLD, metric="alias_match_strict")
    assert out["score"] == out["outcome"] == 1.0 and out["arm_term"] == 0.0 and out["n_exec_search"] == 1.0
    out = compute_score("x", "<answer> Paris", GOLD, metric="alias_match_strict")
    assert out["score"] == 0.0 and out["alias_match"] == 1.0


def test_executed_searches_count_information_blocks():
    from agent.reward import executed_searches
    assert executed_searches(TWO_SEARCH_RIGHT) == 2
    assert executed_searches(NO_SEARCH_RIGHT) == 0
    assert executed_searches("<search> q" + SEARCH_R1_INVALID + "<answer> Paris </answer>") == 0   # malformed: not executed
    assert executed_searches("<search> a </search>" + info("x") + "<search> b </search>" + info("y")
                             + "<search> c </search>" + info("z") + "<answer> Paris </answer>") == 3


def test_cost_term():
    kw = dict(metric="alias_match_strict", arm="C", cost=0.15)
    assert compute_score("x", NO_SEARCH_RIGHT, GOLD, **kw)["score"] == 1.0
    assert compute_score("x", ONE_SEARCH_RIGHT, GOLD, **kw)["score"] == pytest.approx(0.85)
    out = compute_score("x", TWO_SEARCH_RIGHT, GOLD, **kw)
    assert out["arm_term"] == pytest.approx(-0.30) and out["score"] == pytest.approx(0.70)
    assert compute_score("x", EVIDENCE_WRONG_ANSWER, GOLD, **kw)["score"] == pytest.approx(-0.15)


@pytest.mark.parametrize("a", [0.2, 0.6, 1.0])
def test_self_consistency_decision_term(a):
    kw = dict(metric="alias_match_strict", arm="S", lam=0.3, extra_info={"sc_agreement": a})
    s = compute_score("x", ONE_SEARCH_RIGHT, GOLD, **kw)
    n = compute_score("x", NO_SEARCH_RIGHT, GOLD, **kw)
    assert s["arm_term"] == pytest.approx(0.3 * (1 - a)) and s["score"] == pytest.approx(1 + 0.3 * (1 - a))
    assert n["arm_term"] == pytest.approx(0.3 * a)
    two = compute_score("x", TWO_SEARCH_RIGHT, GOLD, **kw)   # d is binary: the number of searches does not matter
    assert two["arm_term"] == pytest.approx(s["arm_term"])
    wrong = compute_score("x", "<answer> Lyon </answer>", GOLD, **kw)
    assert wrong["score"] == pytest.approx(0.3 * a)          # the term is paid whatever the outcome


def test_oracle_decision_term():
    kw = dict(metric="alias_match_strict", arm="O", lam=0.3)
    K, U = {"e004_label": "knowable"}, {"e004_label": "unknowable"}
    assert compute_score("x", NO_SEARCH_RIGHT, GOLD, extra_info=K, **kw)["arm_term"] == pytest.approx(0.3)
    assert compute_score("x", ONE_SEARCH_RIGHT, GOLD, extra_info=K, **kw)["arm_term"] == pytest.approx(0.0)
    assert compute_score("x", NO_SEARCH_RIGHT, GOLD, extra_info=U, **kw)["arm_term"] == pytest.approx(0.0)
    assert compute_score("x", ONE_SEARCH_RIGHT, GOLD, extra_info=U, **kw)["arm_term"] == pytest.approx(0.3)
    with pytest.raises(KeyError):
        compute_score("x", NO_SEARCH_RIGHT, GOLD, extra_info={}, **kw)
    with pytest.raises(ValueError):
        compute_score("x", NO_SEARCH_RIGHT, GOLD, metric="alias_match_strict", arm="Z")


def test_batch_logs_reward_components(tmp_path):
    import json
    from agent import step_stats
    from agent.reward import compute_score_batch
    step_stats.STATE["train_calls"] = 0
    log = tmp_path / "steps.jsonl"
    sols = [NO_SEARCH_RIGHT, ONE_SEARCH_RIGHT, NO_SEARCH_RIGHT, ONE_SEARCH_RIGHT, "<answer> Lyon </answer>",
            EVIDENCE_WRONG_ANSWER]
    extras = ([{"index": 0, "e004_label": "knowable"}] * 2 + [{"index": 1, "e004_label": "knowable"}] * 2
              + [{"index": 2, "e004_label": "unknowable"}] * 2)
    out = compute_score_batch(["x"] * 6, sols, [GOLD] * 6, extras, log_path=str(log), metric="alias_match_strict",
                              arm="O", lam=0.3)
    assert [r["score"] for r in out] == pytest.approx([1.3, 1.0, 1.3, 1.0, 0.0, 0.3])
    c = json.loads(log.read_text().splitlines()[0])["components"]
    k, u = c["by_label"]["e004_label"]["knowable"], c["by_label"]["e004_label"]["unknowable"]
    assert k["n"] == 4 and k["share_search"] == 0.5 and k["n_groups"] == 2
    assert k["mean_outcome_0search"] == 1.0 and k["mean_arm_0search"] == pytest.approx(0.3)
    assert k["mean_total_search"] == 1.0 and k["mean_arm_search"] == 0.0
    assert k["n_groups_mixed_search"] == 2 and k["n_groups_mixed_search_and_total"] == 2
    assert u["mean_outcome"] == 0.0 and u["mean_arm_search"] == pytest.approx(0.3) and u["mean_total_0search"] == 0.0
    assert c["all"]["n"] == 6 and c["all"]["share_search"] == 0.5


# --- per-step rollout samples (E005 diagnosis, 2026-11-22) ---
def test_compute_score_batch_writes_step_samples(tmp_path):
    import json
    from agent import step_stats
    from agent.reward import compute_score_batch
    step_stats.STATE["train_calls"] = 0
    n = 320
    texts = ["<search>q</search><information>doc</information><answer>Paris</answer>"] * (n // 2) + ["<answer>no idea"] * (n // 2)
    extras = [{"index": i // 5, "id": f"q{i // 5}", "e004_label": "knowable" if i % 2 else "unknowable"} for i in range(n)]
    sp = tmp_path / "samples.jsonl"
    for _ in range(2):
        compute_score_batch(["popqa"] * n, texts, [{"target": ["Paris"]}] * n, extras, log_path=str(tmp_path / "steps.jsonl"),
                            sample_path=str(sp), sample_n=50, metric="alias_match_strict", arm="C", cost=0.2)
    rows = [json.loads(l) for l in open(sp)]
    assert [sum(r["step"] == s for r in rows) for s in (1, 2)] == [50, 50]
    assert len({r["row"] for r in rows if r["step"] == 1}) == 50
    r = next(r for r in rows if r["text"].endswith("no idea"))
    assert r["answer_unclosed"] is True and r["outcome"] == 0.0
    r = next(r for r in rows if r["n_exec_search"] == 1)
    assert abs(r["score"] - 0.8) < 1e-9 and r["label"]["e004_label"] in ("knowable", "unknowable")


def test_compute_score_batch_small_batch_and_no_path(tmp_path):
    import json
    from agent.reward import compute_score_batch
    sp = tmp_path / "s.jsonl"
    compute_score_batch(["popqa"] * 3, ["<answer>x</answer>"] * 3, [{"target": ["x"]}] * 3, [{"index": 0}] * 3,
                        sample_path=str(sp), sample_n=50, metric="alias_match_strict")
    assert len(open(sp).readlines()) == 3
    out = compute_score_batch(["popqa"], ["<answer>x</answer>"], [{"target": ["x"]}], [{"index": 0}], metric="alias_match_strict")
    assert out[0]["score"] == 1.0


# --- per-server garble metric (E005 diagnosis, 2026-11-22) ---
def test_garble_events_by_server_and_warning(tmp_path, capsys):
    import json
    from agent import step_stats
    from agent.garble import by_server, is_garbled, read_new_events, write_event
    from agent.reward import compute_score_batch
    assert is_garbled(["ok </.  German arc"]) and is_garbled(["<think> x </think>", "</ answer >"])
    assert not is_garbled(["<think> fine </think>\n<answer> Paris </answer>", "<search> q </search>"])
    gd = str(tmp_path / "garble")
    for r in range(4):
        for k in range(80):
            write_event(gd, r, garbled=(r == 1 and k < 20), garbled_first=False)
    g = by_server(read_new_events(gd))
    assert g["by_server"]["1"] == {"n": 80, "garbled": 20, "share": 0.25, "garbled_first": 0} and g["warn_servers"] == [1]
    assert read_new_events(gd) == []  # cursor advanced
    # one training step through compute_score_batch: stats logged, warning printed and written
    step_stats.STATE["train_calls"] = 0
    for k in range(80):
        write_event(gd, 3, garbled=k < 9, garbled_first=False)   # 0.1125 > 0.10
        write_event(gd, 0, garbled=k < 8, garbled_first=False)   # 0.10, not above
    n = 10
    compute_score_batch(["popqa"] * n, ["<answer>x</answer>"] * n, [{"target": ["x"]}] * n,
                        [{"index": i // 5} for i in range(n)], log_path=str(tmp_path / "steps.jsonl"),
                        garble_dir=gd, metric="alias_match_strict")
    rec = json.loads(open(tmp_path / "steps.jsonl").readline())
    assert rec["garble"]["warn_servers"] == [3] and rec["garble"]["all"]["n"] == 160
    assert "GARBLE WARNING step 1" in capsys.readouterr().out
    assert "server 3 9/80" in open(f"{gd}/warnings.log").read()


# --- E006 capped length statistics ---
def test_capped_answer_stats(tmp_path):
    import json
    from agent import step_stats
    from agent.reward import compute_score_batch
    step_stats.STATE["train_calls"] = 0
    long_unclosed = "<answer> " + "w " * 300
    texts = ["<answer> Paris </answer>"] * 4 + [long_unclosed] + ["<answer> a b c </answer>"] * 5
    extras = [{"index": i // 5, "e004_label": "knowable" if i < 5 else "unknowable"} for i in range(10)]
    compute_score_batch(["popqa"] * 10, texts, [{"target": ["Paris"]}] * 10, extras, log_path=str(tmp_path / "s.jsonl"),
                        metric="alias_match_strict")
    r = json.loads(open(tmp_path / "s.jsonl").readline())
    kn_raw = r["answer"]["by_label"]["e004_label"]["knowable"]
    kn_cap = r["answer_capped"]["by_label"]["e004_label"]["knowable"]
    assert kn_raw["max_words"] == 300 and kn_cap["max_words"] == 50
    assert kn_cap["median_words"] == 1.0 and abs(kn_cap["mean_words"] - (4 + 50) / 5) < 1e-9
    assert r["answer_words_capped"] == {"knowable": [1, 1, 1, 1, 50], "unknowable": [3, 3, 3, 3, 3]}
    assert r["answer_capped"]["by_label"]["e004_label"]["unknowable"]["share_over_20"] == 0.0
