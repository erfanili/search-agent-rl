"""
Search-R1 rollout logic, independent of verl (P1.3).

One episode = one question answered by the policy, interleaved with retrieval:

  prompt | model turn | env block | model turn | env block | ... | model turn
  mask:  (not in response) |  1...1  |  0...0  |  1...1  | ... |  1...1

Everything here works on TOKEN IDS, one segment at a time:
  - model segments are the ids the policy actually sampled (cut at the end of its first action);
  - env segments are tokenized on their own and appended.
The response mask is built alongside the ids, so it cannot be off by one at a boundary. The
alternative (join the strings, tokenize once, map character offsets to tokens) breaks whenever
a BPE token straddles the boundary between model text and environment text.

The episode follows Search-R1 (search_r1/llm_agent/generation.py) and our baseline rollout
(baselines/searchr1_infer.py):
  - a turn ends at the first complete <search>...</search> or <answer>...</answer>;
  - <search> -> retrieval -> "\\n\\n<information>Doc 1(Title: T) text\\n...</information>\\n\\n",
    passages cut to max_obs_tokens;
  - a turn with neither tag -> Search-R1's invalid-action message, then another turn;
  - after max_turns turns, one final turn in which a search is not executed;
  - the response never ends on environment tokens (the reward sits on the last token, which
    must be a model token).

Known difference from Search-R1: verl v0.5's vLLM server sets max_tokens itself, so there is no
per-turn cap of 500 tokens here. Each turn is bounded by the remaining response budget instead.

Nothing here imports verl, vLLM, torch or the retrieval code, so tests/test_rollout_contract.py
can drive it with a fake generator and a fake retriever. agent/verl_agent_loop.py plugs in the
real ones.
"""

import re
from dataclasses import dataclass, field

from agent.qa_metrics import SEARCH_R1_INVALID

INSTRUCTION = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
)

ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
OPEN_RE = re.compile(r"<(search|answer)>(?:(?!</(?:search|answer)>).)*$", re.DOTALL)  # opened, never closed
STOP_STRINGS = ["</search>", "</answer>"]


def make_question_prompt(question):
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    return INSTRUCTION.format(question=q)


def prompt_ids_for(tokenizer, messages, chat_template=True):
    """Prompt token ids. chat_template=False: the raw instruction text (base-model prompting).
    (transformers 5 returns a dict from apply_chat_template(tokenize=True); 4.x returns a list.)"""
    if not chat_template:
        return tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def passages_to_string(hits):
    """Search-R1's format. hits: documents ({"id", "contents"}) or {"document": ..., "score": ...}."""
    out = ""
    for i, h in enumerate(hits):
        doc = h.get("document", h)
        lines = doc["contents"].split("\n")
        out += f"Doc {i + 1}(Title: {lines[0]}) {chr(10).join(lines[1:])}\n"
    return out


@dataclass
class RolloutConfig:
    max_turns: int = 4            # turns that may search; then one final turn without retrieval
    topk: int = 3
    max_obs_tokens: int = 500     # passages text is cut to this many tokens (Search-R1 max_obs_length)
    response_length: int = 4096   # total response budget: model + env tokens (verl rollout.response_length)
    invalid_action: str = "retry"  # "retry": Search-R1's message and another turn; "stop": end the episode


@dataclass
class Segment:
    role: str                     # "model" or "env"
    ids: list
    kind: str = ""                # model: "search" / "answer" / "invalid"; env: "information" / "invalid_msg"
    query: str = ""


@dataclass
class Episode:
    prompt_ids: list
    segments: list = field(default_factory=list)
    finish: str = ""              # answer | turn_budget | length | invalid
    n_search: int = 0
    n_invalid: int = 0
    n_close_tag_appended: int = 0  # turns where the closing tag was missing from the sampled ids
    queries: list = field(default_factory=list)

    @property
    def response_ids(self):
        return [t for s in self.segments for t in s.ids]

    @property
    def response_mask(self):
        return [1 if s.role == "model" else 0 for s in self.segments for _ in s.ids]


def _decode(tokenizer, ids):
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def _end_token_ids(tokenizer):
    ends = set()
    if tokenizer.eos_token_id is not None:
        ends.add(tokenizer.eos_token_id)
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tokenizer.unk_token_id:
            ends.add(i)
    return ends


def cut_at_action(ids, tokenizer, hit_budget=False):
    """Split one turn's sampled ids at the end of its first action.

    Returns (kept_ids, kind, content, close_tag_appended):
      kind "search" / "answer": kept_ids end with the token that completes the closing tag; anything
        sampled after it (e.g. text past a stop string) is dropped.
      kind "invalid": no complete action; kept_ids are the sampled ids minus trailing end-of-text
        tokens, so the episode can continue after an invalid-action message.
    If an action was opened but not closed and the turn did not hit the token budget, generation
    stopped on a stop string that vLLM left out of the token ids: the closing tag is appended
    (tokenized on its own) and close_tag_appended is True. The P1.3 rollout check counts this.
    """
    ids = list(ids)
    text = _decode(tokenizer, ids)
    m = ACTION_RE.search(text)
    if m:
        lo, hi = 1, len(ids)  # smallest prefix whose text already contains the first complete action
        while lo < hi:
            mid = (lo + hi) // 2
            if ACTION_RE.search(_decode(tokenizer, ids[:mid])):
                hi = mid
            else:
                lo = mid + 1
        kept = ids[:lo]
        m = ACTION_RE.search(_decode(tokenizer, kept))
        return kept, m.group(1), m.group(2).strip(), False

    o = OPEN_RE.search(text)
    if o and not hit_budget:
        kind = o.group(1)
        close = tokenizer.encode(f"</{kind}>", add_special_tokens=False)
        kept = ids + close
        m = ACTION_RE.search(_decode(tokenizer, kept))
        if m:
            return kept, m.group(1), m.group(2).strip(), True

    ends = _end_token_ids(tokenizer)
    while ids and ids[-1] in ends:
        ids.pop()
    return ids, "invalid", "", False


def information_ids(tokenizer, hits, max_obs_tokens):
    """The env block after a search, tokenized on its own."""
    text = passages_to_string(hits).strip()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_obs_tokens:
        text = tokenizer.decode(ids[:max_obs_tokens])
    return tokenizer.encode(f"\n\n<information>{text}</information>\n\n", add_special_tokens=False)


async def run_episode(prompt_ids, generate, retrieve, tokenizer, cfg):
    """generate(context_ids) -> sampled ids for one turn (async).
    retrieve(query, topk) -> list of hits (async).
    Returns an Episode; its response ids/mask are what verl trains on."""
    ep = Episode(prompt_ids=list(prompt_ids))
    context = list(prompt_ids)
    used = 0
    invalid_msg = tokenizer.encode(SEARCH_R1_INVALID, add_special_tokens=False)

    for turn in range(cfg.max_turns + 1):
        final = turn == cfg.max_turns
        budget = cfg.response_length - used
        if budget <= 0:
            ep.finish = "length"
            break

        sampled = list(await generate(context))
        hit_budget = len(sampled) >= budget
        kept, kind, content, appended = cut_at_action(sampled, tokenizer, hit_budget)
        truncated = len(kept) > budget
        kept = kept[:budget]
        ep.n_close_tag_appended += appended
        ep.segments.append(Segment("model", kept, kind, content if kind == "search" else ""))
        used += len(kept)
        context += kept

        if truncated:
            ep.finish = "length"
            break
        if kind == "answer":
            ep.finish = "answer"
            break
        if kind == "search":
            if final:
                ep.finish = "turn_budget"
                break
            hits = await retrieve(content, cfg.topk)
            obs, obs_kind = information_ids(tokenizer, hits, cfg.max_obs_tokens), "information"
            ep.n_search += 1
            ep.queries.append(content)
        else:
            ep.n_invalid += 1
            if final or cfg.invalid_action == "stop":
                ep.finish = "turn_budget" if final else "invalid"
                break
            obs, obs_kind = invalid_msg, "invalid_msg"

        if used + len(obs) >= cfg.response_length:  # never end on env tokens
            ep.finish = "length"
            break
        ep.segments.append(Segment("env", obs, obs_kind))
        used += len(obs)
        context += obs

    if not ep.finish:
        ep.finish = "turn_budget"
    return ep