#!/usr/bin/env python
"""E001 frozen set, revision 2: $DATA_ROOT/eval/e001/e001_easy_1k.jsonl (card path data/e001_easy_1k.jsonl).

- 500 PopQA (akariasai/PopQA test): top 2 quintiles of log10(s_pop) over the whole test split (bin 4, bin 5),
  250 each, drawn without replacement with numpy default_rng(0) (bin 4 first, then bin 5).
- 500 TriviaQA (mandarjoshi/trivia_qa rc.nocontext validation), drawn with a separate default_rng(0);
  answers = answer["aliases"]. bin = null, s_pop = null. Sampled from the 9,960 unique question_ids
  (first row of each), since rc.nocontext repeats questions once per evidence document.
Fields: id, source, question, answers, s_pop, bin. Refuses to overwrite. Writes a .meta.json alongside.
(Revision 1, $DATA_ROOT/eval/popqa/popqa_e001_1k.jsonl, used by the first pilot, is left in place.)
"""
import hashlib, json, os, sys
import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from agent.paths import DATA  # noqa: E402

OUT = os.path.join(DATA, "eval", "e001", "e001_easy_1k.jsonl")
POPQA_REV = "098765c79ea10a2cb19c828324e33281b8336ec0"
TQA_REV = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"

if os.path.exists(OUT):
    sys.exit(f"{OUT} exists; frozen set, not overwriting")
p = load_dataset("akariasai/PopQA", split="test", revision=POPQA_REV)
x = np.log10(np.array(p["s_pop"], dtype=float))
edges = np.quantile(x, [0, .2, .4, .6, .8, 1])
bins = np.clip(np.searchsorted(edges, x, side="right"), 1, 5)
rng = np.random.default_rng(0)
rows = []
for b in (4, 5):
    for i in rng.choice(np.flatnonzero(bins == b), size=250, replace=False):
        r = p[int(i)]
        rows.append({"id": f"popqa_{r['id']}", "source": "popqa", "question": r["question"],
                     "answers": json.loads(r["possible_answers"]), "s_pop": int(r["s_pop"]), "bin": b})
t = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", revision=TQA_REV)
# rc.nocontext validation repeats each question once per evidence document (17,944 rows, 9,960 question_ids):
# keep the first row of each question_id, then sample.
first = {}
for i, q in enumerate(t["question_id"]):
    first.setdefault(q, i)
uniq = sorted(first.values())
for i in np.random.default_rng(0).choice(uniq, size=500, replace=False):
    r = t[int(i)]
    rows.append({"id": f"tqa_{r['question_id']}", "source": "triviaqa", "question": r["question"],
                 "answers": list(r["answer"]["aliases"]), "s_pop": None, "bin": None})
assert len({r["id"] for r in rows}) == 1000
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
meta = {"popqa": {"revision": POPQA_REV, "n_source": len(p), "log10_spop_edges": edges.tolist(),
                  "spop_range_sampled": {b: [min(r["s_pop"] for r in rows if r["bin"] == b),
                                             max(r["s_pop"] for r in rows if r["bin"] == b)] for b in (4, 5)}},
        "triviaqa": {"revision": TQA_REV, "config": "rc.nocontext", "split": "validation", "n_source_rows": len(t), "n_unique_questions": len(uniq)},
        "seed": 0, "sha256": hashlib.sha256(open(OUT, "rb").read()).hexdigest()}
json.dump(meta, open(OUT.replace(".jsonl", ".meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
