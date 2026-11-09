#!/usr/bin/env python
"""
Build questions that need information from two passages and that neither closed-book recall nor a single
retrieval can answer. Questions are grounded in our corpus (the retrieval server's passage store), so the
evidence is reachable with the agent's own retriever.

Per candidate:
  1. sample  a random passage A from the store (random passages are mostly about obscure subjects)
  2. bridge  Haiku names an entity mentioned in A that has its own article; passage B is that article's
             first passage, looked up by exact title
  3. write   Haiku writes a question needing a fact from A and a fact from B, either
               bridge      A identifies the entity only by its relation; the answer is a fact in B
               comparison  the answer compares a fact in A with a fact in B
  4. filter  drop the question if any check fails, and record which one:
               format       answer must appear in A or B (or be yes/no); question must not name the
                            bridge entity or contain the answer
               single_rag   the answer appears in the top-10 passages retrieved for the question itself
               closed_book  Haiku answers it correctly with no passages
               rag_reader   Haiku answers it correctly from the top-5 passages retrieved for the question
               oracle       Haiku fails to answer it from A and B (the question is broken)
Stops once --n questions pass. Closed-book recall by the 3B policy is a separate check on GPU.

Output (in --out):
  questions.jsonl   kept questions, same columns as the frozen eval sets (id, question, golden_answers)
                    plus type, passage ids, and the facts used
  candidates.jsonl  every candidate with the check that removed it, for auditing the filters
  stats.json        counts per check and API token usage

Runs in .venv. Needs the retrieval server and Anthropic credentials (ANTHROPIC_API_KEY).
  .venv/bin/python datagen/make_multihop.py --n 100 --out $DATA_ROOT/eval/hard_multihop
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal

import anthropic
import numpy as np
import requests
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from agent.paths import DATA  # noqa: E402
from agent.qa_metrics import answer_in_text, cover_match, normalize_answer, token_f1  # noqa: E402

MODEL = "claude-haiku-5-5"
URL = "http://127.0.0.1:8000"


# ---- structured outputs ---------------------------------------------------------------

class Bridge(BaseModel):
    entity: str      # exact name as it would title its own Wikipedia article
    relation: str    # how passage A's subject relates to it, e.g. "the label that released the album"


class Bridges(BaseModel):
    usable: bool     # False for lists, stubs, disambiguation or garbled passages
    bridges: List[Bridge]


class Question(BaseModel):
    ok: bool                                   # False if no good question exists for this pair
    type: Literal["bridge", "comparison"]
    question: str
    answer: str                                # short span copied from a passage, or yes/no
    answer_aliases: List[str]
    fact_from_a: str
    fact_from_b: str


class Answer(BaseModel):
    answer: str


BRIDGE_PROMPT = """Here is a Wikipedia passage.

<passage>
{a}
</passage>

List up to two specific named entities mentioned in the passage (not the passage's own subject) that would
have their own Wikipedia article: people, places, organizations, works, events. Prefer entities that are
not famous. For each, give its name exactly as its article would be titled, and describe in a few words how
the passage's subject relates to it. Set usable to false if the passage is a list, a stub, or garbled."""

QUESTION_PROMPT = """Write one question that can only be answered by combining a fact from passage A with a
fact from passage B.

<passage_a>
{a}
</passage_a>

<passage_b>
{b}
</passage_b>

The bridge entity is "{entity}" ({relation}).

Choose one type:
- bridge: describe the bridge entity only through its relation to passage A's subject, never by name, and
  ask for a fact that passage B gives about it. Someone must first learn from A which entity is meant, then
  look it up.
- comparison: ask something whose answer needs one fact from A and one from B (which came first, which is
  larger, do both share a property). The answer may be yes, no, or one of the two names.

Rules:
- The answer is short (a name, date, number, place, or yes/no) and copied exactly from a passage.
- The question must not contain the bridge entity's name or the answer.
- The question must be answerable from the two passages alone, with exactly one correct answer.
- Avoid facts a well-read person would know without looking anything up.
- answer_aliases lists other correct surface forms of the answer (may be empty).
Set ok to false if no question meeting these rules exists for this pair."""

CLOSED_PROMPT = """Answer the question with a short answer only (a name, date, number, place, or yes/no).
If you are unsure, give your best guess.

Question: {q}"""

READER_PROMPT = """Answer the question with a short answer only (a name, date, number, place, or yes/no),
using the passages below. If they don't contain the answer, give your best guess.

{passages}

Question: {q}"""


# ---- corpus and retrieval ------------------------------------------------------------------

class PassageStore:
    """Same files as retrieval_server.PassageStore: row i is '"Title"\\ntext'."""

    def __init__(self, index_dir):
        self.offsets = np.load(os.path.join(index_dir, "docs_offsets.npy"), mmap_mode="r")
        self.data = np.memmap(os.path.join(index_dir, "docs.bin"), dtype=np.uint8, mode="r")
        self.n = len(self.offsets) - 1

    def get(self, row):
        a, b = int(self.offsets[row]), int(self.offsets[row + 1])
        return bytes(self.data[a:b]).decode("utf-8")


def title_index(corpus_dir, cache):
    """normalized article title -> row of the article's first passage (its intro). Rows are the corpus
    order (row = id - 1) and an article's passages are contiguous. Built once, cached in DATA."""
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
        import pyarrow.parquet as pq

    import pyarrow.parquet as pq

    index, row, prev = {}, 0, None
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.parquet"))):
        for t in pq.read_table(path, columns=["title"]).column("title").to_pylist():
            if t != prev:
                index.setdefault(normalize_answer(t.strip('"')), row)
                prev = t
            row += 1
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(index, f)
    return index


def title_of(contents):
    return contents.split("\n", 1)[0].strip().strip('"')


def as_text(contents):
    title, _, body = contents.partition("\n")
    return f"Title: {title.strip().strip(chr(34))}\n{body.strip()}"


def retrieve(query, topk):
    r = requests.post(f"{URL}/retrieve", json={"queries": [query], "topk": topk, "return_scores": False}, timeout=120)
    r.raise_for_status()
    return r.json()["result"][0]


# ---- Haiku calls ---------------------------------------------------------------------------

class Haiku:
    def __init__(self):
        self.client = anthropic.Anthropic(max_retries=5)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "refusals": 0}
        self.lock = threading.Lock()

    def parse(self, prompt, schema, effort):
        r = self.client.messages.parse(
            model=MODEL, max_tokens=4000, output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}], output_format=schema,
        )
        with self.lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += r.usage.input_tokens
            self.usage["output_tokens"] += r.usage.output_tokens
            self.usage["refusals"] += r.stop_reason == "refusal"
        if r.stop_reason in ("refusal", "max_tokens"):
            return None
        return r.parsed_output

    def answer(self, prompt):
        out = self.parse(prompt, Answer, effort="low")
        return out.answer if out else ""


def correct(pred, golds):
    """Lenient on purpose: a filter that drops a question should err toward 'the model knew it'."""
    return bool(pred) and (cover_match(pred, golds) or token_f1(pred, golds) >= 0.5)


# ---- one candidate ---------------------------------------------------------------------------

def make_candidate(row, store, titles, haiku):
    rec = {"row_a": row}
    a = store.get(row)
    rec["title_a"] = title_of(a)
    if len(a.split()) < 60:
        return {**rec, "dropped": "short_passage"}
    br = haiku.parse(BRIDGE_PROMPT.format(a=as_text(a)), Bridges, effort="low")
    if not br or not br.usable or not br.bridges:
        return {**rec, "dropped": "no_bridge"}

    for bridge in br.bridges:
        row_b = titles.get(normalize_answer(bridge.entity))
        if row_b is not None and normalize_answer(bridge.entity) != normalize_answer(rec["title_a"]):
            break
    else:
        return {**rec, "bridges": [x.entity for x in br.bridges], "dropped": "bridge_not_in_corpus"}
    b = store.get(row_b)
    rec.update(entity=bridge.entity, relation=bridge.relation, row_b=row_b, title_b=title_of(b))

    q = haiku.parse(QUESTION_PROMPT.format(a=as_text(a), b=as_text(b), entity=bridge.entity,
                                           relation=bridge.relation), Question, effort="medium")
    if not q or not q.ok:
        return {**rec, "dropped": "no_question"}
    golds = [q.answer] + [x for x in q.answer_aliases if x.strip()]
    rec.update(question=q.question, golden_answers=golds, type=q.type, fact_from_a=q.fact_from_a,
               fact_from_b=q.fact_from_b)

    yes_no = normalize_answer(q.answer) in ("yes", "no")
    in_passages = answer_in_text(a + " " + b, golds)
    named = answer_in_text(q.question, [bridge.entity]) if q.type == "bridge" else 0.0
    if (not yes_no and not in_passages) or named or (not yes_no and answer_in_text(q.question, golds)):
        return {**rec, "dropped": "format"}

    top10 = retrieve(q.question, 10)
    if not yes_no and answer_in_text(" ".join(h["contents"] for h in top10), golds):
        return {**rec, "dropped": "single_rag"}

    rec["closed_book_pred"] = haiku.answer(CLOSED_PROMPT.format(q=q.question))
    if correct(rec["closed_book_pred"], golds):
        return {**rec, "dropped": "closed_book"}

    passages = "\n\n".join(f"<passage>\n{as_text(h['contents'])}\n</passage>" for h in top10[:5])
    rec["rag_pred"] = haiku.answer(READER_PROMPT.format(passages=passages, q=q.question))
    if correct(rec["rag_pred"], golds):
        return {**rec, "dropped": "rag_reader"}

    oracle = f"<passage>\n{as_text(a)}\n</passage>\n\n<passage>\n{as_text(b)}\n</passage>"
    rec["oracle_pred"] = haiku.answer(READER_PROMPT.format(passages=oracle, q=q.question))
    if not correct(rec["oracle_pred"], golds):
        return {**rec, "dropped": "oracle"}
    return {**rec, "dropped": None}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="questions to keep")
    p.add_argument("--max-candidates", type=int, default=2000, help="stop after this many even if short of --n")
    p.add_argument("--index-dir", default=f"{DATA}/index_e5_IVF4096_SQ8")
    p.add_argument("--corpus-dir", default=f"{DATA}/corpus/data/psgs_w100/no_embeddings")
    p.add_argument("--title-cache", default=f"{DATA}/datagen/title_index.pkl")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    requests.get(f"{URL}/health", timeout=10).raise_for_status()
    store, haiku = PassageStore(a.index_dir), Haiku()
    titles = title_index(a.corpus_dir, a.title_cache)
    print(f"{len(titles):,} article titles", flush=True)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    kept, counts, lock = [], {}, threading.Lock()

    with open(os.path.join(a.out, "candidates.jsonl"), "w") as cand_f, ThreadPoolExecutor(a.workers) as pool:
        def work(row):
            try:
                rec = make_candidate(row, store, titles, haiku)
            except (anthropic.APIError, requests.RequestException) as e:
                rec = {"row_a": row, "dropped": "error", "error": repr(e)[:300]}
            with lock:
                counts[rec["dropped"] or "kept"] = counts.get(rec["dropped"] or "kept", 0) + 1
                cand_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cand_f.flush()
                if rec["dropped"] is None and len(kept) < a.n:
                    kept.append(rec)
                    print(f"[{len(kept)}/{a.n}] {rec['type']}: {rec['question']}  ->  {rec['golden_answers'][0]}", flush=True)

        submitted = 0
        while submitted < a.max_candidates and len(kept) < a.n:
            batch = [rng.randrange(store.n) for _ in range(a.workers * 4)]
            submitted += len(batch)
            list(pool.map(work, batch))

    with open(os.path.join(a.out, "questions.jsonl"), "w") as f:
        for i, r in enumerate(kept):
            f.write(json.dumps({
                "id": f"hmh_{i}", "question": r["question"], "golden_answers": r["golden_answers"], "type": r["type"],
                "passage_a": {"row": r["row_a"], "title": r["title_a"]}, "passage_b": {"row": r["row_b"], "title": r["title_b"]},
                "fact_from_a": r["fact_from_a"], "fact_from_b": r["fact_from_b"],
            }, ensure_ascii=False) + "\n")
    stats = {"model": MODEL, "seed": a.seed, "kept": len(kept), "candidates": sum(counts.values()),
             "by_outcome": dict(sorted(counts.items(), key=lambda kv: -kv[1])), "usage": haiku.usage}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
