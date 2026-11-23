#!/usr/bin/env python
"""E004 training data: candidate pool -> E001 labels on the start model -> MIX-50 / MIX-20 (jsonl + verl parquet).

  .venv-verl/bin/python scripts/exp/E004_build_data.py pool
  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E004_build_data.py label
  .venv-verl/bin/python scripts/exp/E004_build_data.py mix

pool   4,000 PopQA test questions from quintiles 4-5 (E001's log10(s_pop) edges, same dataset revision),
       not in the E001 set, drawn with numpy default_rng(0); 4,000 TriviaQA rc.nocontext TRAIN questions, first
       row per question_id (the split repeats questions per evidence document), drawn with default_rng(0).
       Answers: PopQA possible_answers, TriviaQA answer.aliases (as E001).
label  E001 revision 2 procedure on Qwen2.5-3B-Instruct: E001 3-shot closed-book prompt in the chat template,
       n=5, temperature 0.7, top_p 1.0, max 512 tokens, stop "</answer>", vLLM seed 0; E001 parse; cover match.
       knowable n_correct >= 3, unknowable 0, uncertain 1-2. clarification = some untagged sample matches E001's
       CLARIFY_RE (E001_tables.uclass).
mix    Keeps knowable / unknowable questions with no clarification sample.
       MIX-50: per source equal knowable and unknowable counts, PopQA up to 800 + 800, TriviaQA fills the rest up to
       1,600 + 1,600 in total (largest size the pool supports, up to 3,200). MIX-20: the same per-source sizes,
       20% knowable / 80% unknowable within each source. Rows drawn with default_rng(0), then shuffled.
       verl rows: the E002 P2 prompt text (E002_prompts.plain_prompt) as one user message (the agent loop applies
       the chat template), ground truth = answers, extra_info {index, id, question, split: train, e004_label, source}.
       val.parquet = the first 64 rows of train (required by the launcher's data paths; validation is off).
Outputs: $DATA_ROOT/verl_data/e004/{pool.jsonl, pool_labels.jsonl, pool_samples.jsonl, label_run.json}
         $DATA_ROOT/verl_data/e004/mix50/{e004_train_mix50.jsonl, train.parquet, val.parquet}  (card: data/e004_train_mix50.jsonl)
         $DATA_ROOT/verl_data/e004/mix20/{e004_train_mix20.jsonl, train.parquet, val.parquet}
         $DATA_ROOT/verl_data/e004/mix_meta.json
"""
import hashlib, json, os, subprocess, sys, time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA, hf_snapshot  # noqa: E402

OUT = os.path.join(DATA, "verl_data", "e004")
POPQA_REV = "098765c79ea10a2cb19c828324e33281b8336ec0"
TQA_REV = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
E001_SET = os.path.join(DATA, "eval", "e001", "e001_easy_1k.jsonl")
N_POOL = 4000
TARGET = 3200
AGENT_NAME = "search_r1_agent"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def pool():
    from datasets import load_dataset
    e001 = {json.loads(l)["id"] for l in open(E001_SET)}
    edges = json.load(open(E001_SET.replace(".jsonl", ".meta.json")))["popqa"]["log10_spop_edges"]
    p = load_dataset("akariasai/PopQA", split="test", revision=POPQA_REV)
    x = np.log10(np.array(p["s_pop"], dtype=float))
    assert np.allclose(np.quantile(x, [0, .2, .4, .6, .8, 1]), edges)
    bins = np.clip(np.searchsorted(edges, x, side="right"), 1, 5)
    cand = [i for i in np.flatnonzero(bins >= 4) if f"popqa_{p[int(i)]['id']}" not in e001]
    rows = []
    for i in sorted(np.random.default_rng(0).choice(cand, size=N_POOL, replace=False)):
        r = p[int(i)]
        rows.append({"id": f"popqa_{r['id']}", "source": "popqa", "question": r["question"],
                     "answers": json.loads(r["possible_answers"]), "s_pop": int(r["s_pop"]), "bin": int(bins[i])})
    t = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", revision=TQA_REV)
    first = {}
    for i, q in enumerate(t["question_id"]):
        first.setdefault(q, i)
    uniq = sorted(first.values())
    for i in sorted(np.random.default_rng(0).choice(uniq, size=N_POOL, replace=False)):
        r = t[int(i)]
        rows.append({"id": f"tqa_{r['question_id']}", "source": "triviaqa", "question": r["question"],
                     "answers": list(r["answer"]["aliases"]), "s_pop": None, "bin": None})
    assert len({r["id"] for r in rows}) == 2 * N_POOL and not ({r["id"] for r in rows} & e001)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/pool.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"popqa": {"revision": POPQA_REV, "n_candidates_q45_not_e001": len(cand)},
            "triviaqa": {"revision": TQA_REV, "split": "train", "n_rows": len(t), "n_unique_questions": len(uniq)},
            "seed": 0, "sha256": sha(f"{OUT}/pool.jsonl")}
    json.dump(meta, open(f"{OUT}/pool_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


def label():
    from vllm import LLM, SamplingParams
    from agent.qa_metrics import cover_match
    from E001_prompts import make_prompt
    from E001_run import label as lab, parse
    from E001_tables import uclass
    rows = [json.loads(l) for l in open(f"{OUT}/pool.jsonl")]
    path = hf_snapshot("models", "Qwen/Qwen2.5-3B-Instruct")
    t0 = time.time()
    llm = LLM(model=path, seed=0, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    outs = llm.generate([make_prompt("chat", tok, r["question"]) for r in rows],
                        SamplingParams(n=5, temperature=0.7, top_p=1.0, max_tokens=512, seed=0, stop=["</answer>"],
                                       include_stop_str_in_output=True))
    with open(f"{OUT}/pool_samples.jsonl", "w") as fs, open(f"{OUT}/pool_labels.jsonl", "w") as fl:
        for r, o in zip(rows, outs):
            S = []
            for c in o.outputs:
                ans, st = parse(c.text)
                s = {"text": c.text, "answer": ans, "parse": st,
                     "correct": int(ans is not None and cover_match(ans, r["answers"]) == 1.0)}
                s["untagged_class"] = uclass(s)
                S.append(s)
            n = sum(s["correct"] for s in S)
            fs.write(json.dumps({"id": r["id"], "samples": S}) + "\n")
            fl.write(json.dumps({"id": r["id"], "source": r["source"], "n_correct": n, "label": lab(n),
                                 "clarification": any(s["untagged_class"] == "clarification" for s in S)}) + "\n")
    json.dump({"model_path": path, "n": len(rows), "wall_s": round(time.time() - t0, 1),
               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}, open(f"{OUT}/label_run.json", "w"), indent=2)


def mix():
    import pandas as pd
    from E002_prompts import plain_prompt
    rows = {r["id"]: r for r in map(json.loads, open(f"{OUT}/pool.jsonl"))}
    labs = [json.loads(l) for l in open(f"{OUT}/pool_labels.jsonl")]
    keep = [l for l in labs if l["label"] in ("knowable", "unknowable") and not l["clarification"]]
    by = {(s, L): sorted(l["id"] for l in keep if l["source"] == s and l["label"] == L)
          for s in ("popqa", "triviaqa") for L in ("knowable", "unknowable")}
    half = TARGET // 2
    cap = {s: min(len(by[(s, "knowable")]), len(by[(s, "unknowable")])) for s in ("popqa", "triviaqa")}
    n_pq = min(cap["popqa"], half // 2)
    n_tq = min(cap["triviaqa"], half - n_pq)
    n_pq = min(cap["popqa"], half - n_tq)
    size = {"popqa": 2 * n_pq, "triviaqa": 2 * n_tq}
    rng = np.random.default_rng(0)
    meta = {"pool_counts": {f"{s}/{L}": len(v) for (s, L), v in by.items()},
            "dropped": {"uncertain": sum(l["label"] == "uncertain" for l in labs),
                        "clarification_knowable_or_unknowable": sum(l["clarification"] and l["label"] != "uncertain" for l in labs)},
            "size_per_source": size}
    for name, kfrac in (("mix50", 0.5), ("mix20", 0.2)):
        sel = []
        for s in ("popqa", "triviaqa"):
            nk = round(kfrac * size[s])
            nu = size[s] - nk
            assert nk <= len(by[(s, "knowable")]) and nu <= len(by[(s, "unknowable")]), (name, s, nk, nu)
            sel += [(i, "knowable") for i in rng.choice(by[(s, "knowable")], nk, replace=False)]
            sel += [(i, "unknowable") for i in rng.choice(by[(s, "unknowable")], nu, replace=False)]
        sel = [sel[k] for k in rng.permutation(len(sel))]
        d = f"{OUT}/{name}"
        os.makedirs(d, exist_ok=True)
        jl = f"{d}/e004_train_{name}.jsonl"
        with open(jl, "w") as f:
            for i, L in sel:
                f.write(json.dumps({**rows[i], "label": L}) + "\n")
        vr = [{"data_source": rows[i]["source"], "agent_name": AGENT_NAME,
               "prompt": [{"role": "user", "content": plain_prompt("P2", rows[i]["question"])}],
               "ability": "fact-reasoning",
               "reward_model": {"style": "rule", "ground_truth": {"target": [str(a) for a in rows[i]["answers"]]}},
               "extra_info": {"index": k, "id": i, "question": rows[i]["question"], "split": "train",
                              "e004_label": L, "source": rows[i]["source"]}} for k, (i, L) in enumerate(sel)]
        pd.DataFrame(vr).to_parquet(f"{d}/train.parquet")
        pd.DataFrame(vr[:64]).to_parquet(f"{d}/val.parquet")
        cnt = {f"{s}/{L}": sum(rows[i]["source"] == s and l == L for i, l in sel) for s in ("popqa", "triviaqa") for L in ("knowable", "unknowable")}
        meta[name] = {"n": len(sel), "counts": cnt, "jsonl": jl, "sha256": sha(jl)}
    json.dump(meta, open(f"{OUT}/mix_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    {"pool": pool, "label": label, "mix": mix}[sys.argv[1]]()
