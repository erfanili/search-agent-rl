"""Hand-written cases for qa_metrics.py and an end-to-end check of score_predictions.py.
Run: .venv/bin/python -m pytest tests/ -q"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent   # score_predictions.py lives here
sys.path.insert(0, str(HERE.parent))     # repo root, for agent/qa_metrics.py

from agent.qa_metrics import (  # noqa: E402
    SEARCH_R1_INVALID, aggregate, count_searches, cover_match, exact_match, extract_answer,
    normalize_answer, score_output, score_prediction, strip_environment_text, token_f1,
)


# ---- normalization and metrics -------------------------------------------------------

def test_normalize_drops_case_punct_articles_spaces():
    assert normalize_answer("  The  Eiffel-Tower! ") == "eiffeltower"
    assert normalize_answer("A man, a plan") == "man plan"


def test_em_uses_best_of_gold_list():
    assert exact_match("USA", ["United States", "usa"]) == 1.0
    assert exact_match("the Beatles", "Beatles") == 1.0
    assert exact_match("Beatles band", ["Beatles"]) == 0.0


def test_f1_partial_credit_and_best_gold():
    # pred tokens {barack, obama}, gold {obama}: precision 1/2, recall 1 -> F1 2/3
    assert token_f1("Barack Obama", ["Obama"]) == pytest.approx(2 / 3)
    assert token_f1("Barack Obama", ["Obama", "Barack Obama"]) == 1.0
    assert token_f1("Paris", ["London"]) == 0.0


def test_f1_empty_after_normalization():
    assert token_f1("the", ["a"]) == 1.0   # both empty -> treated as a match
    assert token_f1("the", ["Paris"]) == 0.0


def test_cover_match_substring_and_empty_gold():
    assert cover_match("It was the year 1969 I think", ["1969"]) == 1.0
    assert cover_match("anything", ["!!!"]) == 0.0   # gold normalizes to "", never matches


def test_cover_rewards_long_answers_but_em_does_not():
    # the reward-hacking case for Phase 3: a long answer containing the gold
    long = "Paris London Berlin Rome Madrid"
    assert cover_match(long, ["Paris"]) == 1.0 and exact_match(long, ["Paris"]) == 0.0
    assert token_f1(long, ["Paris"]) == pytest.approx(2 * (1 / 5) / (1 / 5 + 1))


# ---- trajectory parsing --------------------------------------------------------------

def test_extract_last_answer_and_strip():
    out = "<answer> draft </answer> more thinking <answer>\n final \n</answer>"
    assert extract_answer(out) == "final"


def test_unclosed_or_missing_answer_is_none():
    assert extract_answer("<answer> truncated at the token cap") is None
    assert extract_answer("") is None and extract_answer(None) is None


def test_count_searches():
    out = "<search>a</search><information>x</information><search>b\nc</search><search>unclosed"
    assert count_searches(out) == 2


def test_score_output_abort_scores_zero():
    s = score_output("<search>q</search> ran out of tokens", ["Paris"])
    assert s == {"answer": None, "answered": False, "n_search": 1, "em": 0.0, "f1": 0.0, "acc": 0.0}


def test_score_output_and_prediction_agree():
    a = score_output("<search>q</search><answer>Paris</answer>", ["paris"])
    b = score_prediction("Paris", ["paris"])
    assert (a["em"], a["f1"], a["acc"]) == (b["em"], b["f1"], b["acc"]) == (1.0, 1.0, 1.0)
    assert score_prediction(None, ["x"])["answered"] is False


def test_aggregate_macro_is_unweighted():
    rows = [{"dataset": "big", "em": 1.0, "f1": 1.0, "acc": 1.0, "answered": True, "n_search": 1}] * 9
    rows += [{"dataset": "small", "em": 0.0, "f1": 0.0, "acc": 0.0, "answered": False, "n_search": 3}]
    m = aggregate(rows)
    assert m["big"]["em"] == 1.0 and m["small"]["abort_rate"] == 1.0
    assert m["macro_avg"]["em"] == 0.5          # micro average would be 0.9
    assert m["macro_avg"]["mean_searches"] == 2.0


# ---- end to end: dummy predictions through score_predictions.py ----------------------

def _gold_dir(tmp_path, n=10):
    d = tmp_path / "eval"
    (d / "toy").mkdir(parents=True)
    with open(d / "toy" / "sub300.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"q{i}", "question": f"Q{i}?", "golden_answers": [f"answer {i}", f"alt {i}"]}) + "\n")
    (d / "MANIFEST.json").write_text(json.dumps({"sets": {"toy": {}}}))
    return d


def _run(*args):
    return subprocess.run([sys.executable, str(HERE / "score_predictions.py"), *args], capture_output=True, text=True, cwd=HERE)


def test_dummy_end_to_end(tmp_path):
    gold = _gold_dir(tmp_path)
    preds, out = tmp_path / "dummy.jsonl", tmp_path / "scored"
    assert _run("--gold-dir", str(gold), "--make-dummy", str(preds)).returncode == 0
    r = _run("--gold-dir", str(gold), "--preds", str(preds), "--out", str(out))
    assert r.returncode == 0, r.stderr
    m = json.loads((out / "metrics.json").read_text())["metrics"]["toy"]
    # rows 0..9: abort {0}; correct {2,4,6,8}; wrong {1,3,5,7,9}
    assert m["em"] == pytest.approx(0.4) and m["acc"] == pytest.approx(0.4)
    assert m["abort_rate"] == pytest.approx(0.1)
    assert m["mean_searches"] == pytest.approx((1 + 4 * 1 + 5 * 2) / 10)
    assert len((out / "per_example.jsonl").read_text().splitlines()) == 10


def test_scorer_refuses_bad_files(tmp_path):
    gold = _gold_dir(tmp_path, n=2)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"dataset": "toy", "id": "q0", "prediction": "x"}) + "\n"
        + json.dumps({"dataset": "toy", "id": "q0", "prediction": "x"}) + "\n"      # duplicate
        + json.dumps({"dataset": "toy", "id": "nope", "prediction": "x"}) + "\n"    # unknown id
    )
    r = _run("--gold-dir", str(gold), "--preds", str(bad))
    assert r.returncode != 0
    assert "duplicate" in r.stderr and "not in toy" in r.stderr and "no prediction" in r.stderr
    ok = tmp_path / "partial.jsonl"
    ok.write_text(json.dumps({"dataset": "toy", "id": "q0", "prediction": "answer 0"}) + "\n")
    r = _run("--gold-dir", str(gold), "--preds", str(ok), "--allow-missing", "--out", str(tmp_path / "o"))
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "o" / "metrics.json").read_text())["metrics"]["toy"]
    assert m["em"] == 0.5 and m["abort_rate"] == 0.5   # the missing one counts as an abort


# ---- environment text is not the model's ---------------------------------------------

def test_strip_removes_information_blocks():
    t = "<search>q</search>\n\n<information>Doc 1 says <answer> X </answer></information>\n\n<answer> Y </answer>"
    assert strip_environment_text(t) == "<search>q</search>\n\n\n\n<answer> Y </answer>"


def test_invalid_action_message_is_not_an_answer():
    t = "I think it is Babbage." + SEARCH_R1_INVALID + "still thinking"
    s = score_output(t, ["and"])
    assert s["answered"] is False and s["em"] == 0.0


def test_answer_inside_passage_is_ignored():
    t = "<search>q</search><information>Doc 1(Title: T) <answer> Ada </answer> <search> x </search></information>"
    s = score_output(t, ["Ada"])
    assert s["answered"] is False and s["n_search"] == 1 and s["em"] == 0.0
