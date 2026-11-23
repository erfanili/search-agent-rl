#!/usr/bin/env python
"""E003 generation stage (vLLM) for one model: S2 samples, S3 greedy answer log-prob, S5 greedy P2 first turn.

  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E003_gen.py --model qwen3b-base [--pilot]

S2  E001 prompt and decoding (n=5, temperature 0.7, top_p 1.0, max 512 new tokens, stop "</answer>" kept),
    with seed 1 (vLLM engine seed and SamplingParams seed) instead of 0. Parsed with the E001 rule.
S3  E001 prompt, greedy (temperature 0), max 512, stop "</answer>" kept, logprobs of the sampled tokens.
    Answer span = characters between the first "<answer>" and the next "</answer>" (or the end of the text if
    it never closes), whitespace-trimmed. Tokens whose character span overlaps it are averaged
    (token character spans from cumulative decoding). No "<answer>" or an empty span -> S3 missing.
S5  generation part: the E002 P2 prompt (scripts/exp/E002_prompts.py, chat template for instruct models),
    greedy, max 512, stop "</search>" / "</answer>" (the rollout's free-turn stops). Only the first turn is
    needed: the margin is read at the first tag, which E003_hidden.py computes with a forward pass.
Outputs: $DATA_ROOT/logs/E003[/pilot]/<model>/{s2_samples.jsonl, s3_greedy.jsonl, s5_turn1.jsonl, gen_run.json}
"""
import argparse, json, os, random, subprocess, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA, hf_snapshot  # noqa: E402
from E001_prompts import MODELS, make_prompt as e001_prompt  # noqa: E402
from E001_run import parse  # noqa: E402
from E002_prompts import make_prompt as e002_prompt  # noqa: E402

SET = os.path.join(DATA, "eval", "e001", "e001_easy_1k.jsonl")


def pilot_rows(rows):
    """25 PopQA from bin 4 and 25 from bin 5 (random.Random(0).sample, file order kept), first 50 TriviaQA."""
    rng = random.Random(0)
    out = []
    for b in (4, 5):
        pool = [r for r in rows if r["source"] == "popqa" and r["bin"] == b]
        keep = set(r["id"] for r in rng.sample(pool, 25))
        out += [r for r in pool if r["id"] in keep]
    return out + [r for r in rows if r["source"] == "triviaqa"][:50]


def token_spans(tok, ids):
    """Character [start, end) of each token in tok.decode(ids), by cumulative decoding."""
    spans, prev = [], 0
    for i in range(len(ids)):
        n = len(tok.decode(ids[: i + 1]))
        spans.append((prev, n))
        prev = n
    return spans


def answer_logprob(tok, ids, lps):
    text = tok.decode(ids)
    i = text.find("<answer>")
    if i < 0:
        return None, 0, "untagged"
    a = i + len("<answer>")
    j = text.find("</answer>", a)
    b = j if j >= 0 else len(text)
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    if a >= b:
        return None, 0, "empty"
    sel = [lp for (s, e), lp in zip(token_spans(tok, ids), lps) if s < b and e > a]
    return sum(sel) / len(sel), len(sel), "closed" if j >= 0 else "open"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    rows = [json.loads(l) for l in open(SET)]
    if a.pilot:
        rows = pilot_rows(rows)
    out = os.path.join(DATA, "logs", "E003", "pilot" if a.pilot else "", a.model)
    os.makedirs(out, exist_ok=True)
    src, kind = MODELS[a.model]
    path = src if os.path.isdir(src) else hf_snapshot("models", src)
    assert path and os.path.isdir(path), src

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=path, seed=1, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    p1 = [e001_prompt(kind, tok, r["question"]) for r in rows]

    # S2
    outs = llm.generate(p1, SamplingParams(n=5, temperature=0.7, top_p=1.0, max_tokens=512, seed=1,
                                           stop=["</answer>"], include_stop_str_in_output=True))
    with open(f"{out}/s2_samples.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            S = []
            for c in o.outputs:
                ans, st = parse(c.text)
                S.append({"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids),
                          "answer": ans, "parse": st})
            f.write(json.dumps({"id": r["id"], "source": r["source"], "bin": r["bin"], "question": r["question"],
                                "samples": S}) + "\n")

    # S3
    outs = llm.generate(p1, SamplingParams(temperature=0.0, max_tokens=512, stop=["</answer>"],
                                           include_stop_str_in_output=True, logprobs=0))
    with open(f"{out}/s3_greedy.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            lps = [d[t].logprob for t, d in zip(ids, c.logprobs)]
            s3, n_tok, st = answer_logprob(tok, ids, lps)
            ans, _ = parse(c.text)
            f.write(json.dumps({"id": r["id"], "text": c.text, "finish_reason": c.finish_reason, "answer": ans,
                                "span": st, "n_span_tokens": n_tok, "s3": s3, "token_logprobs": lps}) + "\n")

    # S5 generation (first turn of a greedy P2 rollout)
    p2 = [e002_prompt("P2", kind, tok, r["question"]) for r in rows]
    outs = llm.generate(p2, SamplingParams(temperature=0.0, max_tokens=512, stop=["</search>", "</answer>"],
                                           include_stop_str_in_output=True))
    with open(f"{out}/s5_turn1.jsonl", "w") as f:
        for r, pr, o in zip(rows, p2, outs):
            c = o.outputs[0]
            f.write(json.dumps({"id": r["id"], "prompt": pr, "text": c.text, "token_ids": list(c.token_ids),
                                "finish_reason": c.finish_reason}) + "\n")

    git = lambda *c: subprocess.run(["git", "-C", ROOT, *c], capture_output=True, text=True).stdout.strip()
    meta = {"model_tag": a.model, "model_path": path, "prompt_type": kind, "n_questions": len(rows), "pilot": a.pilot,
            "set": SET, "wall_s": round(time.time() - t0, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty_files": len(git("status", "--porcelain").splitlines())}
    json.dump(meta, open(f"{out}/gen_run.json", "w"), indent=2)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
