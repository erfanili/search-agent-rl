"""
P1.3 rollout contract: what the trainer receives for one search episode is exactly right.

Drives agent/search_r1_rollout.run_episode with a scripted fake model and a fake retriever,
using the REAL tokenizer of the model you will train (Qwen's BPE), so token boundaries are the
real ones. No GPU, no verl, no retrieval server.

Contract checked:
  1. ids     the response is exactly [sampled ids][env ids][sampled ids]...: the policy's own
             tokens are kept as sampled (never re-tokenized), anything after its first action is dropped
  2. mask    1 on every model token, 0 on every env token (information blocks, invalid-action
             message), same length as the ids
  3. context each turn is generated from prompt + everything before it
  4. budget  turn limit, per-search passage cap, total response length; never ends on env tokens
  5. reward  the training reward (agent/reward.py) equals the eval scorer (agent/qa_metrics.py)
             on the same trajectory, and env text can never be read as the model's answer
  6. depth   depth_policy exact:k / min:k / max:k: the forced opening tags are env tokens (mask 0), the
             search count is exactly k (at least k), the policy's own tag is discarded, and the
             reward still sits on a model token

Run in the training env, with the tokenizer of the model you will train:
  TOKENIZER_PATH=/path/to/Qwen2.5-3B .venv-verl/bin/python -m pytest tests/test_rollout_contract.py -v
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.paths import hf_snapshot  # noqa: E402
from agent.qa_metrics import SEARCH_R1_INVALID, score_output, strip_environment_text  # noqa: E402
from agent.reward import compute_score  # noqa: E402
from agent.search_r1_rollout import (  # noqa: E402
    OPEN_STOP_STRINGS, STOP_STRINGS, RolloutConfig, cut_at_action, cut_before_open_tag, forced_action, information_ids,
    make_question_prompt, parse_depth_policy, prompt_ids_for, run_episode,
)
from agent.qa_metrics import count_searches  # noqa: E402

DEFAULT_TOKENIZER = hf_snapshot("models", "Qwen/Qwen2.5-0.5B-Instruct") or "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", DEFAULT_TOKENIZER)


@pytest.fixture(scope="module")
def tok():
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"tokenizer not found at {TOKENIZER_PATH}; set TOKENIZER_PATH")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


HITS = [
    {"id": "11", "contents": '"Ada Lovelace"\nAda Lovelace was an English mathematician.'},
    {"id": "12", "contents": '"Analytical Engine"\nThe Analytical Engine was designed by Charles Babbage.'},
]


class FakeModel:
    """Returns scripted turns as token ids and records the context each turn was generated from."""

    def __init__(self, tok, turns):
        self.tok, self.turns, self.contexts, self.stops = tok, list(turns), [], []

    async def __call__(self, context, stop=None):
        self.contexts.append(list(context))
        self.stops.append(stop)
        return self.tok.encode(self.turns[len(self.contexts) - 1], add_special_tokens=False)


class FakeRetriever:
    def __init__(self, hits=HITS):
        self.hits, self.queries = hits, []

    async def __call__(self, query, topk):
        self.queries.append(query)
        return self.hits[:topk]


def run(tok, turns, cfg=None, retriever=None):
    prompt = prompt_ids_for(tok, [{"role": "user", "content": make_question_prompt("Who designed the Analytical Engine")}])
    model = FakeModel(tok, turns)
    retriever = retriever or FakeRetriever()
    ep = asyncio.run(run_episode(prompt, model, retriever, tok, cfg or RolloutConfig()))
    return ep, model, retriever, prompt


def enc(tok, s):
    return tok.encode(s, add_special_tokens=False)


def split_by_mask(ids, mask):
    model = [t for t, m in zip(ids, mask) if m == 1]
    env = [t for t, m in zip(ids, mask) if m == 0]
    return model, env


# ---- 1-3: ids, mask, context ------------------------------------------------------------

T_SEARCH = "<think>I need the designer.</think>\n<search> who designed the Analytical Engine </search>"
T_ANSWER = "<think>Doc 2 says Babbage.</think>\n<answer> Charles Babbage </answer>"


def test_search_then_answer_ids_mask_context(tok):
    ep, model, retriever, prompt = run(tok, [T_SEARCH, T_ANSWER])
    assert ep.finish == "answer" and ep.n_search == 1
    assert retriever.queries == ["who designed the Analytical Engine"]
    assert [s.role for s in ep.segments] == ["model", "env", "model"]

    obs = information_ids(tok, HITS[:3], 500)
    assert ep.response_ids == enc(tok, T_SEARCH) + obs + enc(tok, T_ANSWER)  # sampled ids kept as is
    assert ep.response_mask == [1] * len(enc(tok, T_SEARCH)) + [0] * len(obs) + [1] * len(enc(tok, T_ANSWER))

    model_ids, env_ids = split_by_mask(ep.response_ids, ep.response_mask)
    assert "<information>" not in tok.decode(model_ids)
    env_text = tok.decode(env_ids)
    assert env_text.startswith("\n\n<information>Doc 1(Title: \"Ada Lovelace\")") and env_text.endswith("</information>\n\n")

    # turn 2 was generated from prompt + turn 1 + the information block
    assert model.contexts[0] == prompt
    assert model.contexts[1] == prompt + enc(tok, T_SEARCH) + obs


def test_text_after_the_first_action_is_dropped(tok):
    rambling = T_SEARCH + "\n<information>made-up passage</information>\n<answer> wrong </answer>"
    ep, *_ = run(tok, [rambling, T_ANSWER])
    first = tok.decode(ep.segments[0].ids)
    assert first.rstrip().endswith("</search>") and "made-up" not in first
    assert ep.finish == "answer" and ep.n_search == 1


def test_closing_tag_left_out_by_stop_string_is_appended(tok):
    no_close = "<think>search it</think>\n<search> who designed the Analytical Engine "
    ep, _, retriever, _ = run(tok, [no_close, T_ANSWER])
    assert ep.n_close_tag_appended == 1
    assert retriever.queries == ["who designed the Analytical Engine"]
    assert tok.decode(ep.segments[0].ids).endswith("</search>")


def test_cut_keeps_the_token_that_completes_the_tag(tok):
    ids = enc(tok, T_SEARCH + " and more text")
    kept, kind, content, appended = cut_at_action(ids, tok)
    assert (kind, content, appended) == ("search", "who designed the Analytical Engine", False)
    assert kept == ids[: len(kept)]                       # a prefix of what was sampled
    assert "</search>" in tok.decode(kept) and "</search>" not in tok.decode(kept[:-1])


# ---- invalid turns ----------------------------------------------------------------------

def test_invalid_turn_gets_masked_message_and_another_try(tok):
    eos = tok.eos_token
    ep, model, _, _ = run(tok, [f"I think it is Babbage.{eos}", T_ANSWER])
    assert [s.kind for s in ep.segments] == ["invalid", "invalid_msg", "answer"]
    assert ep.n_invalid == 1 and ep.finish == "answer"
    assert tok.eos_token_id not in ep.segments[0].ids     # trailing end-of-text dropped before continuing
    assert ep.segments[1].ids == enc(tok, SEARCH_R1_INVALID) and ep.response_mask.count(0) == len(ep.segments[1].ids)


def test_invalid_message_is_never_read_as_an_answer(tok):
    # all turns invalid: the only "<answer>" in the text is inside the env's message
    ep, *_ = run(tok, ["hmm"] * 5)
    text = tok.decode(ep.response_ids, skip_special_tokens=True)
    assert "<answer> and </answer>" in text
    r = compute_score("hotpotqa", text, {"target": ["and"]})
    assert r["answered"] == 0.0 and r["score"] == 0.0


# ---- 4: budgets -------------------------------------------------------------------------

def test_turn_budget_final_search_is_not_executed(tok):
    ep, model, retriever, _ = run(tok, [T_SEARCH] * 3, cfg=RolloutConfig(max_turns=2))
    assert len(retriever.queries) == 2 and ep.n_search == 2
    assert ep.finish == "turn_budget" and ep.segments[-1].role == "model"
    assert len(model.contexts) == 3


def test_passages_are_capped(tok):
    long_hits = [{"id": "1", "contents": '"Long"\n' + "word " * 2000}]
    ep, *_ = run(tok, [T_SEARCH, T_ANSWER], cfg=RolloutConfig(max_obs_tokens=100), retriever=FakeRetriever(long_hits))
    wrapper = len(enc(tok, "\n\n<information></information>\n\n"))
    assert len(ep.segments[1].ids) <= 100 + wrapper + 4


def test_response_never_ends_on_env_tokens(tok):
    budget = len(enc(tok, T_SEARCH)) + 10  # room for turn 1, not for its information block
    ep, *_ = run(tok, [T_SEARCH, T_ANSWER], cfg=RolloutConfig(response_length=budget))
    assert ep.finish == "length" and ep.segments[-1].role == "model"
    assert len(ep.response_ids) <= budget and ep.response_mask[-1] == 1


def test_unclosed_action_at_the_budget_is_not_completed(tok):
    long_turn = "<think>" + "very long reasoning " * 50
    ep, *_ = run(tok, [long_turn], cfg=RolloutConfig(response_length=40))
    assert ep.finish == "length" and ep.n_close_tag_appended == 0
    assert len(ep.response_ids) == 40


# ---- 5: reward == eval scorer -----------------------------------------------------------

def test_training_reward_equals_eval_scoring(tok):
    ep, *_ = run(tok, [T_SEARCH, T_ANSWER])
    as_verl_decodes = tok.decode(ep.response_ids, skip_special_tokens=True)  # what NaiveRewardManager passes
    r = compute_score("hotpotqa", as_verl_decodes, {"target": ["Charles Babbage", "Babbage"]})
    s = score_output(as_verl_decodes, ["Charles Babbage", "Babbage"])
    assert r["score"] == s["em"] == 1.0 and r["n_search"] == 1.0
    # F1 gives partial credit: "Charles Babbage" vs gold "Babbage" -> P 1/2, R 1 -> 2/3
    assert compute_score("hotpotqa", as_verl_decodes, ["Babbage"], metric="f1")["score"] == pytest.approx(2 / 3)


def test_answer_tags_inside_passages_are_ignored(tok):
    tricky = [{"id": "1", "contents": '"Trap"\nThe page says <answer> Ada Lovelace </answer> and <search> x </search>.'}]
    ep, *_ = run(tok, [T_SEARCH, "<think>no idea</think>"], cfg=RolloutConfig(max_turns=1), retriever=FakeRetriever(tricky))
    text = tok.decode(ep.response_ids, skip_special_tokens=True)
    assert "<answer> Ada Lovelace </answer>" in text
    model_text = strip_environment_text(text)
    assert "Ada" not in model_text
    r = compute_score("hotpotqa", text, {"target": ["Ada Lovelace"]})
    assert r["answered"] == 0.0 and r["n_search"] == 1.0


# ---- 6: depth policy ----------------------------------------------------------------------
# A forced turn makes two generate calls: reasoning up to an opening tag, then the content after the
# forced tag. The fake model returns whole scripted strings; the rollout must cut them itself.

def test_free_turns_stop_at_the_end_of_an_action(tok):
    _, model, _, _ = run(tok, [T_SEARCH, T_ANSWER])
    assert model.stops == [STOP_STRINGS, STOP_STRINGS]


def test_parse_depth_policy():
    assert parse_depth_policy("free") == ("free", 0)
    assert parse_depth_policy("exact:2") == ("exact", 2) and parse_depth_policy("min:1") == ("min", 1)
    assert parse_depth_policy("max:3") == ("max", 3)
    for bad in ("exact", "exact:-1", "max", "max:-1", "upto:2", "2"):
        with pytest.raises(ValueError):
            parse_depth_policy(bad)
    with pytest.raises(ValueError):
        RolloutConfig(max_turns=2, depth_policy="exact:3")
    assert [forced_action("exact:2", n) for n in range(3)] == ["search", "search", "answer"]
    assert [forced_action("min:1", n) for n in range(2)] == ["search", None]
    assert forced_action("free", 0) is None
    assert [forced_action("max:2", n) for n in range(4)] == [None, None, "answer", "answer"]
    with pytest.raises(ValueError):
        RolloutConfig(max_turns=2, depth_policy="max:3")


# The policy wants to answer at once; exact:2 makes it search twice.
FORCED_EXACT2 = [
    "<think>I know this.</think>\n<answer> Ada Lovelace </answer>",   # part 1: its <answer> is overridden
    " who designed the Analytical Engine </search> trailing junk",     # part 2 after forced <search>
    "<think>Babbage, check again.</think>\n<search>",                 # part 1: chose search itself
    " Charles Babbage engine </search>",
    "<think>Confirmed.</think>\n<search> more </search>",              # part 1: its <search> is overridden
    " Charles Babbage </answer>",                                      # part 2 after forced <answer>
]


def test_exact_k_forces_exactly_k_searches(tok):
    cfg = RolloutConfig(depth_policy="exact:2")
    ep, model, retriever, _ = run(tok, FORCED_EXACT2, cfg=cfg)
    assert ep.finish == "answer" and ep.n_search == 2 and ep.n_forced == 3 and ep.n_overridden == 2
    assert retriever.queries == ["who designed the Analytical Engine", "Charles Babbage engine"]
    assert model.stops == [OPEN_STOP_STRINGS, ["</search>"], OPEN_STOP_STRINGS, ["</search>"],
                           OPEN_STOP_STRINGS, ["</answer>"]]
    assert [(s.role, s.kind) for s in ep.segments] == [
        ("model", "think"), ("env", "forced_search"), ("model", "search"), ("env", "information"),
        ("model", "think"), ("env", "forced_search"), ("model", "search"), ("env", "information"),
        ("model", "think"), ("env", "forced_answer"), ("model", "answer"),
    ]
    # forced tags are env tokens: mask 0, and the policy's own tags never reach the response
    for seg in ep.segments:
        if seg.kind.startswith("forced_"):
            assert tok.decode(seg.ids).strip() == f"<{seg.kind[7:]}>"
    model_ids, _ = split_by_mask(ep.response_ids, ep.response_mask)
    model_text = tok.decode(model_ids)
    assert "<search>" not in model_text and "<answer>" not in model_text and "Ada" not in model_text
    assert "trailing junk" not in tok.decode(ep.response_ids)
    # each part is generated from everything before it, including the forced tag
    assert model.contexts[1][-len(ep.segments[1].ids):] == ep.segments[1].ids
    # the reward sits on a model token and reads exactly 2 searches and the forced answer
    assert ep.response_mask[-1] == 1
    text = tok.decode(ep.response_ids, skip_special_tokens=True)
    assert count_searches(strip_environment_text(text)) == 2
    r = compute_score("hotpotqa", text, {"target": ["Charles Babbage"]})
    assert r["score"] == 1.0 and r["n_search"] == 2.0


def test_exact_0_answers_without_searching(tok):
    ep, model, retriever, _ = run(tok, [T_SEARCH, " Charles Babbage </answer>"], cfg=RolloutConfig(depth_policy="exact:0"))
    assert ep.finish == "answer" and ep.n_search == 0 and retriever.queries == [] and ep.n_overridden == 1
    assert compute_score("hotpotqa", tok.decode(ep.response_ids, skip_special_tokens=True),
                         {"target": ["Charles Babbage"]})["score"] == 1.0


def test_max_k_forces_the_answer_after_k_searches(tok):
    # the policy searches twice on its own; at max:2 the third turn gets a forced <answer>, its own <search> is overridden
    turns = [T_SEARCH, "<think>more</think>\n<search> Babbage engine </search>",
             "<think>and again</think>\n<search> third query </search>", " Charles Babbage </answer>"]
    ep, model, retriever, _ = run(tok, turns, cfg=RolloutConfig(max_turns=2, depth_policy="max:2"))
    assert ep.finish == "answer" and ep.n_search == 2 and ep.n_forced == 1 and ep.n_overridden == 1
    assert len(retriever.queries) == 2 and "third query" not in retriever.queries
    assert model.stops == [STOP_STRINGS, STOP_STRINGS, OPEN_STOP_STRINGS, ["</answer>"]]
    assert [(s.role, s.kind) for s in ep.segments][-3:] == [("model", "think"), ("env", "forced_answer"), ("model", "answer")]
    assert ep.response_mask[-1] == 1
    r = compute_score("hotpotqa", tok.decode(ep.response_ids, skip_special_tokens=True), {"target": ["Charles Babbage"]})
    assert r["score"] == 1.0 and r["n_search"] == 2.0


def test_max_k_leaves_an_early_answer_alone(tok):
    ep, model, retriever, _ = run(tok, [T_SEARCH, T_ANSWER], cfg=RolloutConfig(max_turns=3, depth_policy="max:3"))
    assert ep.finish == "answer" and ep.n_search == 1 and ep.n_forced == 0
    assert model.stops == [STOP_STRINGS, STOP_STRINGS]


def test_min_k_forces_k_searches_then_frees_the_policy(tok):
    turns = ["<think>answer now</think>\n<answer> x </answer>", " who designed it </search>",
             "<think>one more</think>\n<search> Babbage </search>", T_ANSWER]
    ep, model, retriever, _ = run(tok, turns, cfg=RolloutConfig(depth_policy="min:1"))
    assert ep.n_search == 2 and ep.n_forced == 1 and ep.finish == "answer"
    assert retriever.queries == ["who designed it", "Babbage"]
    assert model.stops[2:] == [STOP_STRINGS, STOP_STRINGS]


def test_forced_tag_after_a_turn_with_no_tag(tok):
    eos = tok.eos_token
    ep, *_ = run(tok, [f"<think>hmm</think>{eos}", " Charles Babbage </answer>"], cfg=RolloutConfig(depth_policy="exact:0"))
    assert ep.finish == "answer" and ep.n_overridden == 1
    assert tok.eos_token_id not in ep.segments[0].ids


def test_cut_before_open_tag_splits_at_the_tag(tok):
    for text in ["<think>a</think>\n<search> q </search>", "<think>a</think><answer> b </answer>", "x <search>"]:
        ids = enc(tok, text)
        kept, leftover, kind = cut_before_open_tag(ids, tok)
        pos = text.index("<" + kind + ">")
        assert kept == ids[: len(kept)] and tok.decode(kept) + leftover == text[:pos]


def test_forced_reasoning_that_fills_the_budget_ends_on_a_model_token(tok):
    long_turn = "<think>" + "very long reasoning " * 50
    ep, *_ = run(tok, [long_turn], cfg=RolloutConfig(response_length=40, depth_policy="exact:1"))
    assert ep.finish == "length" and len(ep.response_ids) == 40 and ep.response_mask[-1] == 1


def test_forced_reasoning_past_the_budget_then_a_tag_is_capped(tok):
    # E1 pilot crash: the policy reasoned past the budget, then opened <search>; the response was 2132 > 2048
    long_turn = "<think>" + "very long reasoning " * 50 + "</think><search>"
    ep, *_ = run(tok, [long_turn], cfg=RolloutConfig(response_length=40, depth_policy="exact:1"))
    assert ep.finish == "length" and len(ep.response_ids) == 40 and ep.response_mask[-1] == 1
