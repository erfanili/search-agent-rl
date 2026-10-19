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

from agent.qa_metrics import SEARCH_R1_INVALID, score_output, strip_environment_text  # noqa: E402
from agent.reward import compute_score  # noqa: E402
from agent.search_r1_rollout import (  # noqa: E402
    RolloutConfig, cut_at_action, information_ids, make_question_prompt, prompt_ids_for, run_episode,
)

DEFAULT_TOKENIZER = (
    "/data/erfan/search-rl-agent-data/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
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
        self.tok, self.turns, self.contexts = tok, list(turns), []

    async def __call__(self, context):
        self.contexts.append(list(context))
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