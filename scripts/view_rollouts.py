#!/usr/bin/env python
"""
Check and display episodes dumped by the search agent loop (P1.3).

Input: a directory of rollouts_<pid>.jsonl files written by agent/verl_agent_loop.py (dump_dir).

Checks every episode, exit code 1 if any fails:
  mask      response_mask has one entry per token, 1 exactly on model segments, 0 on env segments
  env text  information blocks are "\\n\\n<information>...</information>\\n\\n"; invalid-action
            messages are exactly Search-R1's text
  ending    the last segment is the model's (the reward sits on the last token)
  actions   a model segment of kind search/answer ends with its closing tag

Reports:
  finish reasons, searches per episode, invalid turns, response length, share of env tokens
  closing tags appended   turns where vLLM's ids lacked the stop string (tells you how vLLM
                          returns stop strings; nonzero is fine, it is handled)
  with --tokenizer, two tokenization facts that show why the rollout works on ids per segment:
    re-tokenizing the decoded response gives different ids   (a mask built from text offsets
                                                              would be misaligned there)
    sampled model ids differ from tokenizing their own text  (re-tokenizing model text would
                                                              train on tokens it never sampled)

With --html, writes one page with every episode: env tokens shaded, token boundaries marked
(with --tokenizer), queries and finish reason per episode. Read a dozen by eye.

Usage (training env, for the tokenizer):
  .venv-verl/bin/python scripts/view_rollouts.py /data/erfan/search-agent-rl-data/logs/p1_3_.../rollouts \\
      --tokenizer /path/to/model --html report.html
"""

import argparse
import collections
import glob
import html
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.qa_metrics import SEARCH_R1_INVALID  # noqa: E402


def load(d):
    eps = []
    for path in sorted(glob.glob(os.path.join(d, "rollouts_*.jsonl"))):
        with open(path) as f:
            eps += [json.loads(line) for line in f if line.strip()]
    return eps


def check(ep):
    errs = []
    segs = ep["segments"]
    ids = [t for s in segs for t in s["ids"]]
    mask = ep["response_mask"]
    expect = [1 if s["role"] == "model" else 0 for s in segs for _ in s["ids"]]
    if len(mask) != len(ids):
        errs.append(f"mask length {len(mask)} != {len(ids)} ids")
    elif mask != expect:
        errs.append("mask does not match segment roles")
    for i, s in enumerate(segs):
        if s["role"] == "env":
            if s["kind"] == "information":
                if not (s["text"].startswith("\n\n<information>") and s["text"].endswith("</information>\n\n")):
                    errs.append(f"segment {i}: information block malformed: {s['text'][:60]!r}...")
            elif s["kind"] == "invalid_msg":
                if s["text"] != SEARCH_R1_INVALID:
                    errs.append(f"segment {i}: invalid-action message differs from Search-R1's")
            else:
                errs.append(f"segment {i}: unknown env kind {s['kind']!r}")
        elif s["kind"] in ("search", "answer") and f"</{s['kind']}>" not in s["text"][-24:]:
            errs.append(f"segment {i}: {s['kind']} turn does not end with </{s['kind']}>")
    if segs and segs[-1]["role"] != "model":
        errs.append("episode ends on env tokens")
    return errs


def token_facts(eps, tok):
    retok_diff = model_noncanonical = n_model = 0
    for ep in eps:
        ids = [t for s in ep["segments"] for t in s["ids"]]
        if tok.encode(tok.decode(ids), add_special_tokens=False) != ids:
            retok_diff += 1
        for s in ep["segments"]:
            if s["role"] == "model" and s["ids"]:
                n_model += 1
                if tok.encode(tok.decode(s["ids"]), add_special_tokens=False) != s["ids"]:
                    model_noncanonical += 1
    return retok_diff, model_noncanonical, n_model


def render(eps, tok, path):
    css = (
        "body{font:13px/1.5 ui-monospace,monospace;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}"
        ".ep{border:1px solid #ccc;border-radius:6px;padding:12px;margin:16px 0}"
        ".meta{color:#555;margin-bottom:8px}.env{background:#ffe9a8}.tk{border-right:1px solid #ddd}"
        ".prompt{color:#888;white-space:pre-wrap}pre{white-space:pre-wrap;margin:0}"
        ".bad{color:#b00;font-weight:bold}"
        "@media(prefers-color-scheme:dark){body{background:#151515;color:#ddd}.env{background:#5a4a12}.tk{border-color:#444}}"
    )
    out = [f"<!doctype html><meta charset=utf-8><title>P1.3 rollouts</title><style>{css}</style>",
           "<h1>P1.3 rollouts</h1><p>Shaded = environment tokens (mask 0, not trained on). "
           "Unshaded = model tokens (mask 1). With a tokenizer, thin lines mark token boundaries.</p>"]
    for n, ep in enumerate(eps):
        errs = check(ep)
        out.append("<div class=ep>")
        out.append(f"<div class=meta>#{n} · finish <b>{ep['finish']}</b> · searches {ep['n_search']} · "
                   f"invalid {ep['n_invalid']} · close tags appended {ep['n_close_tag_appended']} · "
                   f"queries {html.escape(json.dumps(ep['queries']))}</div>")
        if errs:
            out.append("<div class=bad>" + "<br>".join(html.escape(e) for e in errs) + "</div>")
        out.append(f"<details><summary>prompt</summary><div class=prompt>{html.escape(ep['prompt_text'])}</div></details><pre>")
        for s in ep["segments"]:
            cls = "env" if s["role"] == "env" else ""
            if tok is not None:
                pieces = "".join(f"<span class=tk title={t}>{html.escape(tok.decode([t]))}</span>" for t in s["ids"])
            else:
                pieces = html.escape(s["text"])
            out.append(f"<span class='{cls}'>{pieces}</span>")
        out.append("</pre></div>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump_dir")
    p.add_argument("--tokenizer", help="model path, for token boundaries and tokenization facts")
    p.add_argument("--html")
    a = p.parse_args()

    eps = load(a.dump_dir)
    if not eps:
        sys.exit(f"no episodes in {a.dump_dir}: did the agent loop run with dump_dir set?")
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)

    failures = [(i, e) for i, ep in enumerate(eps) for e in check(ep)]
    lens = [len(ep["response_mask"]) for ep in eps]
    env_tokens = sum(ep["response_mask"].count(0) for ep in eps)
    finish = collections.Counter(ep["finish"] for ep in eps)
    searches = collections.Counter(ep["n_search"] for ep in eps)

    print(f"episodes: {len(eps)}")
    print(f"finish: {dict(finish)}")
    print(f"searches per episode: {dict(sorted(searches.items()))} (mean {statistics.mean(ep['n_search'] for ep in eps):.2f})")
    print(f"invalid turns: {sum(ep['n_invalid'] for ep in eps)}")
    print(f"closing tags appended: {sum(ep['n_close_tag_appended'] for ep in eps)}")
    print(f"response tokens: mean {statistics.mean(lens):.0f}, max {max(lens)}; env share {env_tokens / sum(lens):.1%}")
    if tok is not None:
        retok, noncanon, n_model = token_facts(eps, tok)
        print(f"re-tokenizing the decoded response changes the ids: {retok}/{len(eps)} episodes")
        print(f"sampled model ids differ from tokenizing their text: {noncanon}/{n_model} model turns")
    if a.html:
        render(eps, tok, a.html)
        print(f"wrote {a.html}")
    if failures:
        print(f"\nFAILED checks: {len(failures)}")
        for i, e in failures[:30]:
            print(f"  episode {i}: {e}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()