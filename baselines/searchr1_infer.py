#!/usr/bin/env python
"""
Run a Search-R1 checkpoint on the frozen eval sets and write predictions for tests/score_predictions.py.

Runs in .venv-verl (vLLM). Talks to the retrieval server over HTTP, so it imports nothing from retrieval/.

The rollout copies Search-R1 and shares its prompt, tags and passage format with agent/search_r1_rollout.py,
so the numbers are comparable with Search-R1 and with our trained agents. Search-R1's search_r1/llm_agent/generation.py and infer.py:
  prompt      Search-R1's instruction + "Question: ...?", through the tokenizer's chat template
  one turn    greedy generation until </search> or </answer> (or --max-turn-tokens)
  search      first <search>...</search> in the turn -> POST /retrieve, top-k docs appended as
              "\\n\\n<information>Doc 1(Title: T) text\\n...</information>\\n\\n"
  invalid     a turn with neither tag gets Search-R1's "My previous action is invalid..." message
  budget      after --max-turns search turns, one last turn whose searches are not executed
  depth       --depth-policy exact:k / min:k forces the action tags, as agent/search_r1_rollout.py does in
              training: a forced turn generates until the policy opens a tag (or --max-turn-tokens), the
              script writes <search> while fewer than k searches are done (exact:k: <answer> after), and
              the policy writes the query or answer up to its closing tag. exact:k = exactly k searches.
              max:k: free until k searches are done, then the script writes <answer>.

Output (one line per question): {"dataset", "id", "output", "n_turns", "n_search", "n_forced", "n_overridden",
"n_close_appended", "finish", "depth_policy"} where n_overridden counts forced turns in which the policy had
written the other tag or none, and "output" is
everything after the prompt (model text + information blocks); score it with score_predictions.py.

Usage: bash scripts/run_baseline_searchr1.sh
"""

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # repo root, for agent/
from agent.paths import DATA  # noqa: E402
from agent.qa_metrics import SEARCH_R1_INVALID  # noqa: E402
from agent.search_r1_rollout import (  # noqa: E402
    ACTION_RE, OPEN_STOP_STRINGS, OPEN_TAG_RE, STOP_STRINGS, forced_action, make_question_prompt, parse_depth_policy,
    passages_to_string,
)


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


def load_parquet_questions(path, limit):
    """Questions from a verl parquet (agent/make_train_data.py): id = extra_info.id, dataset = data_source."""
    import pandas as pd

    df = pd.read_parquet(path, columns=["data_source", "extra_info"])
    if limit:
        df = df.iloc[:limit]
    return [{"dataset": str(ds), "id": e["id"], "question": e["question"]} for ds, e in zip(df.data_source, df.extra_info)]


def make_prompt(tokenizer, question):
    text = make_question_prompt(question)
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return text


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


def forced_turns(llm, rows, idx, forced, sampling, searches):
    """One forced turn for rows idx (forced[i] = "search" / "answer"), in two batched generate calls.
    Part 1 runs until the policy opens a tag; its tag is dropped and the forced one written. Part 2 runs
    to the forced closing tag; if --max-turn-tokens cuts it off, the closing tag is appended (counted)
    so the episode keeps exactly k searches. Forced searches are queued in `searches`."""
    if not idx:
        return
    outs = llm.generate([rows[i]["prompt"] + rows[i]["output"] for i in idx], sampling(OPEN_STOP_STRINGS), use_tqdm=False)
    for i, o in zip(idx, outs):
        text = o.outputs[0].text
        m = OPEN_TAG_RE.search(text)
        rows[i]["output"] += (text[: m.start()] if m else text) + f"<{forced[i]}>"
        rows[i]["n_turns"] += 1
        rows[i]["n_forced"] += 1
        rows[i]["n_overridden"] += (m.group(1) if m else None) != forced[i]
    for kind in ("search", "answer"):
        sub = [i for i in idx if forced[i] == kind]
        if not sub:
            continue
        close = f"</{kind}>"
        outs = llm.generate([rows[i]["prompt"] + rows[i]["output"] for i in sub], sampling([close]), use_tqdm=False)
        for i, o in zip(sub, outs):
            text = o.outputs[0].text
            if close in text:
                text = text[: text.index(close) + len(close)]
            else:
                text += close
                rows[i]["n_close_appended"] += 1
            rows[i]["output"] += text
            if kind == "answer":
                rows[i]["finish"] = "answer"
            else:
                searches.append((i, text[: -len(close)].strip()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--gold-dir", default=f"{DATA}/eval")
    p.add_argument("--split", default="sub300")
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"])
    p.add_argument("--limit", type=int, default=0, help="first N questions per dataset (0 = all); for quick debugging")
    p.add_argument("--questions-parquet", default=None,
                   help="read questions from a verl parquet instead of --gold-dir (training-data labels; --limit = first N rows)")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--depth-policy", default="free", help="free | exact:k | min:k | max:k (k <= --max-turns)")
    p.add_argument("--max-turn-tokens", type=int, default=500, help="Search-R1 max_response_length")
    p.add_argument("--max-obs-tokens", type=int, default=500, help="Search-R1 max_obs_length")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem", type=float, default=0.5)
    p.add_argument("--out", required=True, help="predictions JSONL")
    a = p.parse_args()
    mode, k = parse_depth_policy(a.depth_policy)
    if mode != "free" and k > a.max_turns:
        p.error(f"--depth-policy {a.depth_policy} needs --max-turns >= {k}")

    from vllm import LLM, SamplingParams

    health = requests.get(f"{a.url}/health", timeout=10).json()
    print("retrieval server:", health, flush=True)

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    def sampling(stop):
        return SamplingParams(temperature=0.0, max_tokens=a.max_turn_tokens, stop=stop, include_stop_str_in_output=True)

    sp = sampling(STOP_STRINGS)

    if a.questions_parquet:
        rows = load_parquet_questions(a.questions_parquet, a.limit)
        a.datasets, a.split = ["parquet"], os.path.basename(a.questions_parquet)
    else:
        rows = load_questions(a.gold_dir, a.split, a.datasets, a.limit)
    for r in rows:
        r.update(prompt=make_prompt(tok, r["question"]), output="", n_turns=0, n_search=0, n_forced=0, n_overridden=0,
                 n_close_appended=0, finish=None)
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
        forced = {i: forced_action(a.depth_policy, rows[i]["n_search"]) for i in active}
        free = [i for i in active if forced[i] is None]
        outs = llm.generate([rows[i]["prompt"] + rows[i]["output"] for i in free], sp, use_tqdm=False) if free else []

        searches, still = [], []
        forced_turns(llm, rows, [i for i in active if forced[i]], forced, sampling, searches)
        for i, o in zip(free, outs):
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
                r["output"] += text + SEARCH_R1_INVALID
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
            f.write(json.dumps({"dataset": r["dataset"], "id": r["id"], "output": r["output"], "n_turns": r["n_turns"],
                                "n_search": r["n_search"], "n_forced": r["n_forced"], "n_overridden": r["n_overridden"],
                                "n_close_appended": r["n_close_appended"], "finish": r["finish"],
                                "depth_policy": a.depth_policy}, ensure_ascii=False) + "\n")
    finish = {}
    for r in rows:
        finish[r["finish"]] = finish.get(r["finish"], 0) + 1
    print(f"wrote {a.out} in {time.time() - t0:.0f}s; finish reasons: {finish}", flush=True)
    n = len(rows) or 1
    print(f"depth_policy {a.depth_policy}: searches/q {sum(r['n_search'] for r in rows) / n:.3f}, "
          f"forced turns overridden {sum(r['n_overridden'] for r in rows)}/{sum(r['n_forced'] for r in rows)}, "
          f"closing tags appended {sum(r['n_close_appended'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
