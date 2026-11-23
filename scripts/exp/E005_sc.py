#!/usr/bin/env python
"""E005 self-consistency targets for the 3,200 MIX-50 training questions, and the arm-S/O training data.

  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E005_sc.py gen     # vLLM sampling
  .venv-verl/bin/python scripts/exp/E005_sc.py data                          # parquet + report

gen   Qwen2.5-3B-Instruct, the E001 revision-2 closed-book prompt (E001_prompts.make_prompt, chat template) and
      decoding (n=5, temperature 0.7, top_p 1.0, max 512 new tokens, stop "</answer>" kept), seed 1 (vLLM engine
      and SamplingParams), as E003 S2 (E003_gen.py). Parsed with the E001 rule (E001_run.parse).
      agreement a = share of the 5 samples whose normalized answer equals the most common one; untagged samples
      never agree (E003_tables.agreement).
data  $DATA_ROOT/verl_data/e004/mix50/train.parquet with extra_info.sc_agreement = a added (rows, order and all other
      fields unchanged) -> $DATA_ROOT/verl_data/e005/mix50sc/{train,val}.parquet (val = first 64 rows, as E004).
      Report: AUROC of (1 - a) for unknowable (positive) vs knowable, bootstrap 95% CI (E002_tables.auroc_ci), and
      the histogram of a by label.
Outputs: $DATA_ROOT/verl_data/e005/{sc_samples.jsonl, sc_agreement.jsonl, sc_run.json, sc_report.md}, mix50sc/
"""
import argparse, collections, json, os, subprocess, sys, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA, hf_snapshot  # noqa: E402
from agent.qa_metrics import normalize_answer  # noqa: E402

OUT = os.path.join(DATA, "verl_data", "e005")
SRC = os.path.join(DATA, "verl_data", "e004", "mix50")


def agreement(samples):  # = E003_tables.agreement
    keys = [normalize_answer(s["answer"]) if s["parse"] != "untagged" else f"__untagged_{k}" for k, s in enumerate(samples)]
    return collections.Counter(keys).most_common(1)[0][1] / len(keys)


def gen():
    from E001_prompts import MODELS, make_prompt
    from E001_run import parse
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(os.path.join(SRC, "e004_train_mix50.jsonl"))]
    os.makedirs(OUT, exist_ok=True)
    src, kind = MODELS["qwen3b-instruct"]
    path = src if os.path.isdir(src) else hf_snapshot("models", src)
    t0 = time.time()
    llm = LLM(model=path, seed=1, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    prompts = [make_prompt(kind, tok, r["question"]) for r in rows]
    outs = llm.generate(prompts, SamplingParams(n=5, temperature=0.7, top_p=1.0, max_tokens=512, seed=1,
                                                stop=["</answer>"], include_stop_str_in_output=True))
    with open(f"{OUT}/sc_samples.jsonl", "w") as fs, open(f"{OUT}/sc_agreement.jsonl", "w") as fa:
        for r, o in zip(rows, outs):
            S = []
            for c in o.outputs:
                ans, st = parse(c.text)
                S.append({"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids),
                          "answer": ans, "parse": st})
            fs.write(json.dumps({"id": r["id"], "question": r["question"], "label": r["label"], "samples": S}) + "\n")
            fa.write(json.dumps({"id": r["id"], "label": r["label"], "sc_agreement": agreement(S),
                                 "n_untagged": sum(s["parse"] == "untagged" for s in S)}) + "\n")
    git = lambda *c: subprocess.run(["git", "-C", ROOT, *c], capture_output=True, text=True).stdout.strip()
    meta = {"model_path": path, "prompt_type": kind, "n_questions": len(rows), "n_samples": 5, "seed": 1,
            "wall_s": round(time.time() - t0, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty_files": len(git("status", "--porcelain").splitlines())}
    json.dump(meta, open(f"{OUT}/sc_run.json", "w"), indent=2)
    print(json.dumps(meta), flush=True)


def data():
    import numpy as np
    import pandas as pd
    from E002_tables import auroc, auroc_ci
    A = {r["id"]: r for r in map(json.loads, open(f"{OUT}/sc_agreement.jsonl"))}
    d = pd.read_parquet(os.path.join(SRC, "train.parquet"))
    assert len(d) == len(A) == 3200
    ei = []
    for e in d["extra_info"]:
        e = dict(e)
        assert A[e["id"]]["label"] == e["e004_label"]
        e["sc_agreement"] = float(A[e["id"]]["sc_agreement"])
        ei.append(e)
    d["extra_info"] = ei
    os.makedirs(f"{OUT}/mix50sc", exist_ok=True)
    d.to_parquet(f"{OUT}/mix50sc/train.parquet")
    d.iloc[:64].to_parquet(f"{OUT}/mix50sc/val.parquet")
    y = np.array([A[e["id"]]["label"] == "unknowable" for e in ei], dtype=int)
    a = np.array([e["sc_agreement"] for e in ei])
    au = auroc((1 - a)[y == 1], (1 - a)[y == 0])
    ci = auroc_ci((1 - a)[y == 1], (1 - a)[y == 0])
    L = ["# E005 self-consistency targets (training set, n = 3,200)", "",
         f"AUROC of (1 - a), unknowable vs knowable: {au:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] "
         f"(n knowable {int((y == 0).sum())}, unknowable {int(y.sum())})", "",
         "| a | knowable n | knowable share | unknowable n | unknowable share |", "|---|---|---|---|---|"]
    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = int(((np.isclose(a, v)) & (y == 0)).sum()); u = int(((np.isclose(a, v)) & (y == 1)).sum())
        L.append(f"| {v:.1f} | {k} | {k / (y == 0).sum():.3f} | {u} | {u / y.sum():.3f} |")
    L += ["", f"mean a: knowable {a[y == 0].mean():.3f}, unknowable {a[y == 1].mean():.3f}; "
              f"untagged samples: {sum(r['n_untagged'] for r in A.values())} of {5 * len(A)}"]
    open(f"{OUT}/sc_report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "data"])
    {"gen": gen, "data": data}[ap.parse_args().stage]()
