#!/usr/bin/env python
"""E003 forward-pass stage (transformers, no generation) for one model: S4 hidden states and S5 margin.

  CUDA_VISIBLE_DEVICES=4 .venv-verl/bin/python scripts/exp/E003_hidden.py --model qwen3b-base [--pilot]

S4  Input = the full E002 P2 prompt (scripts/exp/E002_prompts.py; instruct models: chat template with the
    generation prompt, so the last token is the one right before the reply; base models: the "\\n" after
    "Question: <q>"). Hidden state of the last prompt token at layers 4, 8, ..., L (L = 36 for 3B, 28 for 7B;
    hidden_states[i] = output of block i, hidden_states[L] is after the final norm). bf16 model, states saved
    as float32.
S5  From E003_gen.py's greedy P2 first turn: the first "<search>" or "<answer>" in the generated text; the token
    starting right after its "<" must be the tag-name token. Forward pass on prompt + generated tokens before
    that token; S5 = log p(id("search")) - log p(id("answer")) at the last position (ids = the second token of
    tok.encode("<search>") / tok.encode("<answer>")). No tag in turn 1, or no token boundary right after the
    "<" -> S5 missing (counted by reason).
Outputs: $DATA_ROOT/logs/E003[/pilot]/<model>/{hidden.npz (ids, layers, H[n, n_layers, d]), s5_margin.jsonl,
         hidden_run.json}
"""
import argparse, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.paths import DATA, hf_snapshot  # noqa: E402
from E001_prompts import MODELS  # noqa: E402
from E002_prompts import make_prompt  # noqa: E402
from E003_gen import SET, pilot_rows, token_spans  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    rows = [json.loads(l) for l in open(SET)]
    if a.pilot:
        rows = pilot_rows(rows)
    out = os.path.join(DATA, "logs", "E003", "pilot" if a.pilot else "", a.model)
    src, kind = MODELS[a.model]
    path = src if os.path.isdir(src) else hf_snapshot("models", src)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    L = model.config.num_hidden_layers
    layers = list(range(4, L + 1, 4))
    if layers[-1] != L:
        layers.append(L)

    H = np.zeros((len(rows), len(layers), model.config.hidden_size), np.float32)
    n_prompt = []
    with torch.no_grad():
        for k, r in enumerate(rows):
            ids = tok.encode(make_prompt("P2", kind, tok, r["question"]), add_special_tokens=False)
            n_prompt.append(len(ids))
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
                    rec["tag_token"] = tok.decode([gids[j]])
                    ids = tok.encode(g["prompt"], add_special_tokens=False) + gids[:j]
                    lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1)
                    rec.update(lp_search=lp[sid].item(), lp_answer=lp[aid].item(), s5=(lp[sid] - lp[aid]).item(),
                               reason="ok")
            reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1
            f.write(json.dumps(rec) + "\n")

    meta = {"model_tag": a.model, "model_path": path, "n_questions": len(rows), "layers": layers, "H_shape": list(H.shape),
            "prompt_tokens_min_max": [min(n_prompt), max(n_prompt)], "tag_token_ids": {"search": sid, "answer": aid},
            "s5_reasons": reasons, "wall_s": round(time.time() - t0, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    json.dump(meta, open(f"{out}/hidden_run.json", "w"), indent=2)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
