#!/usr/bin/env python
"""
Run a Search-R1 checkpoint on the frozen eval sets and write predictions for tests/score_predictions.py.

Runs in .venv-verl (vLLM). Talks to the retrieval server over HTTP, so it imports nothing from retrieval/.

The rollout copies Search-R1 (search_r1/llm_agent/generation.py, infer.py) so the numbers are comparable:
  prompt      Search-R1's instruction + "Question: ...?", through the tokenizer's chat template
  one turn    greedy generation until </search> or </answer> (or --max-turn-tokens)
  search      first <search>...</search> in the turn -> POST /retrieve, top-k docs appended as
              "\\n\\n<information>Doc 1(Title: T) text\\n...</information>\\n\\n"
  invalid     a turn with neither tag gets Search-R1's "My previous action is invalid..." message
  budget      after --max-turns search turns, one last turn whose searches are not executed

Output (one line per question): {"dataset", "id", "output", "n_turns", "finish"} where "output" is
everything after the prompt (model text + information blocks); score it with score_predictions.py.

Usage: bash scripts/run_baseline_searchr1.sh
"""

import argparse
import json
import os
import re
import time

import requests

INSTRUCTION = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
)
INVALID = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
)
ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def load_questions(gold_dir, split, datasets, limit):
    rows = []
    for name in datasets:
        with open(os.path.join(gold_dir, name, f"{split}.jsonl")) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                rows.append({"dataset": name, "id": r["id"], "question": r["question"]})
    return rows


def make_prompt(tokenizer, question):
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    text = INSTRUCTION.format(question=q)
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return text


def passages_to_string(hits):
    out = ""
    for i, h in enumerate(hits):
        lines = h["document"]["contents"].split("\n")
        out += f"Doc {i + 1}(Title: {lines[0]}) {chr(10).join(lines[1:])}\n"
    return out


def retrieve(url, queries, topk, batch=256):
    results = []
    for s in range(0, len(queries), batch):
        r = requests.post(f"{url}/retrieve", json={"queries": queries[s : s + batch], "topk": topk, "return_scores": True}, timeout=300)
        r.raise_for_status()
        results.extend(r.json()["result"])
    return results


def truncate_tokens(tokenizer, text, n):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= n else tokenizer.decode(ids[:n])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--gold-dir", default="/data/erfan/search-rl-agent-data/eval")
    p.add_argument("--split", default="sub300")
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"])
    p.add_argument("--limit", type=int, default=0, help="first N questions per dataset (0 = all); for quick debugging")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-turn-tokens", type=int, default=500, help="Search-R1 max_response_length")
    p.add_argument("--max-obs-tokens", type=int, default=500, help="Search-R1 max_obs_length")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem", type=float, default=0.5)
    p.add_argument("--out", required=True, help="predictions JSONL")
    a = p.parse_args()

    from vllm import LLM, SamplingParams

    health = requests.get(f"{a.url}/health", timeout=10).json()
    print("retrieval server:", health, flush=True)

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_turn_tokens, stop=["</search>", "</answer>"],
                        include_stop_str_in_output=True)

    rows = load_questions(a.gold_dir, a.split, a.datasets, a.limit)
    for r in rows:
        r.update(prompt=make_prompt(tok, r["question"]), output="", n_turns=0, n_search=0, finish=None)
    print(f"{len(rows)} questions from {a.datasets} ({a.split})", flush=True)

    t0 = time.time()
    active = list(range(len(rows)))
    for turn in range(a.max_turns + 1):
        if not active:
            break
        last = turn == a.max_turns  # final turn: searches are not executed
        too_long = [i for i in active if len(tok.encode(rows[i]["prompt"] + rows[i]["output"])) + a.max_turn_tokens > a.max_model_len]
        for i in too_long:
            rows[i]["finish"] = "context_full"
        active = [i for i in active if i not in set(too_long)]
        outs = llm.generate([rows[i]["prompt"] + rows[i]["output"] for i in active], sp, use_tqdm=False)

        searches, still = [], []
        for i, o in zip(active, outs):
            text = o.outputs[0].text
            r = rows[i]
            r["n_turns"] += 1
            m = ACTION_RE.search(text)
            if m and m.group(1) == "answer":
                r["output"] += text[: m.end()]
                r["finish"] = "answer"
            elif m and m.group(1) == "search" and not last:
                r["output"] += text[: m.end()]
                searches.append((i, m.group(2).strip()))
            elif last:
                r["output"] += text
                r["finish"] = "turn_budget"
            else:
                r["output"] += text + INVALID
                still.append(i)

        if searches:
            hits = retrieve(a.url, [q for _, q in searches], a.topk)
            for (i, _), h in zip(searches, hits):
                info = truncate_tokens(tok, passages_to_string(h).strip(), a.max_obs_tokens)
                rows[i]["output"] += f"\n\n<information>{info}</information>\n\n"
                rows[i]["n_search"] += 1
                still.append(i)
        active = sorted(still)
        done = sum(r["finish"] is not None for r in rows)
        print(f"turn {turn + 1}: {len(searches)} searches, {done}/{len(rows)} finished, {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps({"dataset": r["dataset"], "id": r["id"], "output": r["output"],
                                "n_turns": r["n_turns"], "finish": r["finish"]}, ensure_ascii=False) + "\n")
    finish = {}
    for r in rows:
        finish[r["finish"]] = finish.get(r["finish"], 0) + 1
    print(f"wrote {a.out} in {time.time() - t0:.0f}s; finish reasons: {finish}", flush=True)


if __name__ == "__main__":
    main()
