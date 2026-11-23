#!/usr/bin/env python
"""E007 Sshuf training data: E005 mix50sc with extra_info.sc_agreement permuted across the 3,200 training rows.

  .venv-verl/bin/python scripts/exp/E007_shuf_data.py

Permutation: numpy default_rng(0).permutation(3200); row i gets the agreement of row perm[i] (rows, order and all other
fields unchanged). val = first 64 rows (as E004/E005). Saved: permutation.jsonl (row, id, label, a_orig, src_row, src_id,
a_new) and report.md (AUROC of 1 - a for unknowable vs knowable, before and after; mean a by label).
Outputs: $DATA_ROOT/verl_data/e007/mix50sc_shuf/{train,val}.parquet, permutation.jsonl, report.md
"""
import json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA  # noqa: E402
from E002_tables import auroc  # noqa: E402

SRC = os.path.join(DATA, "verl_data", "e005", "mix50sc", "train.parquet")
OUT = os.path.join(DATA, "verl_data", "e007", "mix50sc_shuf")
os.makedirs(OUT, exist_ok=True)
d = pd.read_parquet(SRC)
assert len(d) == 3200
E = [dict(e) for e in d["extra_info"]]
a = np.array([e["sc_agreement"] for e in E])
perm = np.random.default_rng(0).permutation(len(E))
new = a[perm]
with open(f"{OUT}/permutation.jsonl", "w") as f:
    for i, e in enumerate(E):
        f.write(json.dumps({"row": i, "id": e["id"], "label": e["e004_label"], "a_orig": float(a[i]), "src_row": int(perm[i]),
                            "src_id": E[perm[i]]["id"], "a_new": float(new[i])}) + "\n")
        e["sc_agreement"] = float(new[i])
d["extra_info"] = E
d.to_parquet(f"{OUT}/train.parquet")
d.iloc[:64].to_parquet(f"{OUT}/val.parquet")
y = np.array([e["e004_label"] == "unknowable" for e in E])
L = ["# E007 Sshuf data (n = 3,200 training rows)", "",
     f"permutation: numpy default_rng(0).permutation(3200); fixed points {int((perm == np.arange(len(perm))).sum())}; "
     f"rows whose value changed {int((new != a).sum())}",
     f"AUROC of (1 - a), unknowable vs knowable: original {auroc((1 - a)[y], (1 - a)[~y]):.3f}, permuted {auroc((1 - new)[y], (1 - new)[~y]):.3f}",
     f"mean a knowable / unknowable: original {a[~y].mean():.3f} / {a[y].mean():.3f}, permuted {new[~y].mean():.3f} / {new[y].mean():.3f}",
     f"sorted values identical: {bool(np.array_equal(np.sort(a), np.sort(new)))}"]
open(f"{OUT}/report.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
