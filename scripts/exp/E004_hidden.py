#!/usr/bin/env python
"""E004 forward-pass stage for one merged checkpoint: S4 hidden states and S5 margin (E003_hidden.py method).

  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E004_hidden.py --run E004-A-s0 --step 20

Model $DATA_ROOT/logs/E004/<run>/hf/step_<k>; prompt = full E002 P2 prompt in the chat template (last token = the one
right before the reply). Hidden states of the last prompt token at layers 4, 8, ..., 36 (all saved; E004 uses 24).
S5 = log p("search") - log p("answer") at the token right after the "<" of the first tag in the greedy P2 first
turn (E004_eval.py's s5_turn1.jsonl). Step 0 is not run here: E003's qwen3b-instruct outputs are reused.
Outputs: $DATA_ROOT/logs/E004/eval/<run>/step<k>/{hidden.npz, s5_margin.jsonl, hidden_run.json}
"""
import argparse, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA  # noqa: E402
from E002_prompts import make_prompt  # noqa: E402
from E003_gen import SET, token_spans  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--exp", default="E004")  # E005 (2026-11-22): same evaluation, runs under $DATA_ROOT/logs/<exp>/
    a = p.parse_args()
    rows = [json.loads(l) for l in open(SET)]
    path = os.path.join(DATA, "logs", a.exp, a.run, "hf", f"step_{a.step}")
    out = os.path.join(DATA, "logs", a.exp, "eval", a.run, f"step{a.step}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    L = model.config.num_hidden_layers
    layers = list(range(4, L + 1, 4))
    H = np.zeros((len(rows), len(layers), model.config.hidden_size), np.float32)
    with torch.no_grad():
        for k, r in enumerate(rows):
            ids = tok.encode(make_prompt("P2", "chat", tok, r["question"]), add_special_tokens=False)
            hs = model(torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states
            assert len(hs) == L + 1
            H[k] = torch.stack([hs[l][0, -1] for l in layers]).float().cpu().numpy()
    np.savez(f"{out}/hidden.npz", ids=np.array([r["id"] for r in rows]), layers=np.array(layers), H=H)

    sid, aid = tok.encode("<search>", add_special_tokens=False)[1], tok.encode("<answer>", add_special_tokens=False)[1]
    gen = {g["id"]: g for g in map(json.loads, open(f"{out}/s5_turn1.jsonl"))}
    reasons = {}
    with open(f"{out}/s5_margin.jsonl", "w") as f, torch.no_grad():
        for r in rows:
            g = gen[r["id"]]
            text, gids = tok.decode(g["token_ids"]), g["token_ids"]
            cand = [(text.find(t), t) for t in ("<search>", "<answer>") if text.find(t) >= 0]
            rec = {"id": r["id"], "first_tag": None, "s5": None, "reason": None}
            if not cand:
                rec["reason"] = "no_tag_turn1"
            else:
                pos, tag = min(cand)
                rec["first_tag"] = tag[1:-1]
                starts = [s for s, _ in token_spans(tok, gids)]
                if pos + 1 not in starts:
                    rec["reason"] = "no_boundary_after_lt"
                else:
                    j = starts.index(pos + 1)
                    ids = tok.encode(g["prompt"], add_special_tokens=False) + gids[:j]
                    lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1)
                    rec.update(lp_search=lp[sid].item(), lp_answer=lp[aid].item(), s5=(lp[sid] - lp[aid]).item(), reason="ok")
            reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1
            f.write(json.dumps(rec) + "\n")
    meta = {"run": a.run, "step": a.step, "model_path": path, "layers": layers, "H_shape": list(H.shape), "s5_reasons": reasons,
            "wall_s": round(time.time() - t0, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    json.dump(meta, open(f"{out}/hidden_run.json", "w"), indent=2)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
