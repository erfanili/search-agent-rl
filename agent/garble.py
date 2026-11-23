"""Garbled-rollout detection and per-vLLM-server aggregation (E005 diagnosis, 2026-11-22).

Garbled = GARBLE_RE matches any model segment of an episode (broken tags such as "</." or "< /" that the model emits in
degenerate text). The verl agent loop appends one event per episode to <garble_dir>/events_<pid>.jsonl (server dp rank,
garbled flag); compute_score_batch (agent/reward.py) reads the events written since its previous call
(per-file byte offsets in <garble_dir>/_cursor.json), so each training step's rollouts are aggregated per server.
"""
import glob
import json
import os
import re

GARBLE_RE = re.compile(r"</\s*\.|<\s+/|</ |< answer|</\.")
WARN_SHARE = 0.10


def is_garbled(model_texts):
    return any(GARBLE_RE.search(t) for t in model_texts)


def write_event(garble_dir, server_rank, garbled, garbled_first):
    os.makedirs(garble_dir, exist_ok=True)
    with open(os.path.join(garble_dir, f"events_{os.getpid()}.jsonl"), "a") as f:
        f.write(json.dumps({"server": int(server_rank), "garbled": bool(garbled), "garbled_first": bool(garbled_first)}) + "\n")


def read_new_events(garble_dir):
    """Events appended since the previous call; advances the cursor."""
    cur_path = os.path.join(garble_dir, "_cursor.json")
    cur = json.load(open(cur_path)) if os.path.exists(cur_path) else {}
    events = []
    for p in sorted(glob.glob(os.path.join(garble_dir, "events_*.jsonl"))):
        name = os.path.basename(p)
        with open(p) as f:
            f.seek(cur.get(name, 0))
            data = f.read()
        end = data.rfind("\n") + 1  # only complete lines
        for line in data[:end].splitlines():
            if line.strip():
                events.append(json.loads(line))
        cur[name] = cur.get(name, 0) + len(data[:end].encode())
    tmp = cur_path + ".tmp"
    json.dump(cur, open(tmp, "w"))
    os.replace(tmp, cur_path)
    return events


def by_server(events, warn_share=WARN_SHARE):
    """{"all": cell, "by_server": {rank: cell}, "warn_servers": [ranks with share > warn_share]}; cell = n, garbled, share."""
    def cell(ev):
        n = len(ev)
        g = sum(e["garbled"] for e in ev)
        return {"n": n, "garbled": g, "share": g / n if n else None,
                "garbled_first": sum(e.get("garbled_first", False) for e in ev)}
    ranks = sorted({e["server"] for e in events})
    bs = {str(r): cell([e for e in events if e["server"] == r]) for r in ranks}
    warn = [int(r) for r, c in bs.items() if c["share"] is not None and c["share"] > warn_share]
    return {"all": cell(events), "by_server": bs, "warn_servers": warn, "warn_share": warn_share}
