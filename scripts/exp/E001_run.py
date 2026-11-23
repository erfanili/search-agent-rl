#!/usr/bin/env python
"""E001 (revision 2): closed-book 3-shot sampling on the frozen easy set for one model, then per-question labels.

  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E001_run.py --model qwen3b-base [--pilot]

Decoding (card): temperature 0.7, top_p 1.0, n=5, max 512 new tokens, stop "</answer>" kept in the output,
vLLM seed 0. Parse (card): text inside <answer>...</answer>; <answer> opened but not closed -> text from <answer>
to the end; no <answer> tag -> untagged (wrong). Never matched against the full output.
Correct = cover_match (a normalized alias is a substring of the normalized parsed answer); EM = exact_match.
Labels: knowable n_correct >= 3, unknowable 0, uncertain 1-2.
--pilot: first 13 PopQA bin-4, first 12 PopQA bin-5 and first 25 TriviaQA rows in file order (50), written
under $DATA_ROOT/logs/E001/pilot2/.
Outputs: <out>/E001-<model>-s0/{samples.jsonl, labels.jsonl, run.json}
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA, hf_snapshot  # noqa: E402
from agent.qa_metrics import cover_match, exact_match  # noqa: E402
from E001_prompts import MODELS, make_prompt  # noqa: E402

SET = os.path.join(DATA, "eval", "e001", "e001_easy_1k.jsonl")


def parse(text):
    """-> (answer or None, status in {closed, open, untagged})"""
    i = text.find("<answer>")
    if i < 0:
        return None, "untagged"
    rest = text[i + len("<answer>"):]
    j = rest.find("</answer>")
    return (rest[:j].strip(), "closed") if j >= 0 else (rest.strip(), "open")


def label(n):
    return "knowable" if n >= 3 else "unknowable" if n == 0 else "uncertain"


def pilot_rows(rows):
    pq4 = [r for r in rows if r["source"] == "popqa" and r["bin"] == 4][:13]
    pq5 = [r for r in rows if r["source"] == "popqa" and r["bin"] == 5][:12]
    tq = [r for r in rows if r["source"] == "triviaqa"][:25]
    return pq4 + pq5 + tq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()

    rows = [json.loads(l) for l in open(SET)]
    if a.pilot:
        rows = pilot_rows(rows)
    out = os.path.join(DATA, "logs", "E001", "pilot2" if a.pilot else "", f"E001-{a.model}-s0")
    os.makedirs(out, exist_ok=True)

    src, kind = MODELS[a.model]
    path = src if os.path.isdir(src) else hf_snapshot("models", src)
    assert path and os.path.isdir(path), f"model not found locally: {src}"
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=path, seed=0, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    prompts = [make_prompt(kind, tok, r["question"]) for r in rows]
    sp = SamplingParams(n=5, temperature=0.7, top_p=1.0, max_tokens=512, seed=0,
                        stop=["</answer>"], include_stop_str_in_output=True)
    outs = llm.generate(prompts, sp)

    with open(f"{out}/samples.jsonl", "w") as fs, open(f"{out}/labels.jsonl", "w") as fl:
        for r, pr, o in zip(rows, prompts, outs):
            samples = []
            for c in o.outputs:
                ans, status = parse(c.text)
                samples.append({"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids),
                                "answer": ans, "parse": status,
                                "correct": int(ans is not None and cover_match(ans, r["answers"]) == 1.0),
                                "em": int(ans is not None and exact_match(ans, r["answers"]) == 1.0)})
            n_c = sum(s["correct"] for s in samples)
            fs.write(json.dumps({"id": r["id"], "source": r["source"], "bin": r["bin"], "s_pop": r["s_pop"],
                                 "question": r["question"], "answers": r["answers"], "prompt": pr,
                                 "samples": samples}) + "\n")
            fl.write(json.dumps({"id": r["id"], "n_correct": n_c, "n_em": sum(s["em"] for s in samples),
                                 "n_untagged": sum(s["parse"] == "untagged" for s in samples),
                                 "label": label(n_c)}) + "\n")

    git = lambda *c: subprocess.run(["git", "-C", ROOT, *c], capture_output=True, text=True).stdout.strip()
    meta = {"model_tag": a.model, "model_path": path, "prompt_type": kind, "n_questions": len(rows), "n_samples": 5,
            "pilot": a.pilot, "set": SET, "wall_s": round(time.time() - t0, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty_files": len(git("status", "--porcelain").splitlines())}
    json.dump(meta, open(f"{out}/run.json", "w"), indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
