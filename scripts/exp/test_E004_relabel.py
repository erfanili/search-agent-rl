"""Tests for the E004/E005 relabel sample collection (E004_eval.collect_closed_samples) and the relabel output check.

  .venv-verl/bin/python -m pytest -q scripts/exp/test_E004_relabel.py

The fake engine streams cumulative outputs the way vLLM 0.8.5's async engine did for the old single n=5 request:
completions appear one by one, and the last streamed RequestOutput can hold fewer than n of them.
"""
import asyncio, json, os, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from E004_eval import N_RELABEL, collect_closed_samples, relabel_seed  # noqa: E402
from E004_check_relabel import check  # noqa: E402


def fake_gen_factory(calls):
    async def gen(ids, sp):
        calls.append(sp)
        n = sp["n"]
        # old failure mode: an n>1 request returns only its last completion
        return SimpleNamespace(outputs=[SimpleNamespace(text=f"<answer>x{sp['seed']}</answer>")] * (1 if n > 1 else n))
    return gen


def test_collect_exactly_five_distinct_seeds():
    calls = []
    for qid in ["popqa_1", "tqa_2", "q3"]:
        cs = asyncio.run(collect_closed_samples(fake_gen_factory(calls), [1, 2, 3], qid, lambda s: {"n": 1, "seed": s}))
        assert len(cs) == N_RELABEL == 5
    assert all(c["n"] == 1 for c in calls) and len(calls) == 15
    seeds = [c["seed"] for c in calls]
    assert len(set(seeds)) == 15  # distinct per (question, sample)
    assert relabel_seed("q3", 0) == relabel_seed("q3", 0)


def test_old_single_request_would_fail_the_count():
    calls = []
    o = asyncio.run(fake_gen_factory(calls)([1], {"n": 5, "seed": 0}))
    assert len(o.outputs) != 5


def _write(d, counts):
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/samples.jsonl", "w") as fs, open(f"{d}/labels.jsonl", "w") as fl:
        for i, n in enumerate(counts):
            fs.write(json.dumps({"id": f"q{i}", "samples": [{"correct": 0}] * n}) + "\n")
            fl.write(json.dumps({"id": f"q{i}", "n_correct": 0, "label": "unknowable"}) + "\n")


def test_check_relabel_dir(tmp_path):
    good, bad = str(tmp_path / "good"), str(tmp_path / "bad")
    _write(good, [5] * 1000)
    _write(bad, [5] * 999 + [1])
    assert check(good, n_questions=1000)["ok"]
    r = check(bad, n_questions=1000)
    assert not r["ok"] and r["count_hist"] == {1: 1, 5: 999}
