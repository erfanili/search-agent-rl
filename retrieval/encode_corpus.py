#!/usr/bin/env python
"""
Encode the wiki_dpr passage corpus with E5-base-v2 into resumable shards.

Step P0.5 / P0.6 of the search-agent RL project.

Three modes:
  stats   - token-length percentiles on a random sample, to choose --max-length (p95, not max)
  encode  - write embeddings in fixed row-range shards; safe to kill and rerun
  verify  - check every shard exists, row counts add up, ids match the corpus row for row,
            and vectors are unit-norm

Output layout (--out):
  meta.json          - every setting that affects the vectors; a rerun with different
                       settings refuses to mix into the same directory
  ids_00000.npy      - int64 passage ids for shard 0, in corpus row order
  emb_00000.npy      - float16 [rows, 768] L2-normalized embeddings for shard 0
  ...
Row r of the concatenated emb_* files is passage ids[r]. The FAISS index stores vectors by
insertion position, so build it by concatenating shards in shard order and keep the
concatenated ids as the position -> passage id map.

Shard k always covers global rows [k*shard_size, (k+1)*shard_size), independent of the
number of GPUs, so a run can be resumed with a different --world-size.

Example:
  # 0. choose max_length
  python retrieval/encode_corpus.py stats --input /data/erfan/search-rl-agent-data/corpus/data/psgs_w100/no_embeddings --sample 20000

  # 1. end-to-end test on the 10k-row dummy corpus
  python retrieval/encode_corpus.py encode --input /data/erfan/search-rl-agent-data/corpus/data/psgs_w100/dummy.no_embeddings \
      --out /data/erfan/search-rl-agent-data/emb_test --max-length 256 --shard-size 2000
  python retrieval/encode_corpus.py verify --input /data/erfan/search-rl-agent-data/corpus/data/psgs_w100/dummy.no_embeddings --out /data/erfan/search-rl-agent-data/emb_test

  # 2. full corpus, one process per GPU listed in configs/gpus.env
  bash scripts/encode_corpus.sh
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

MODEL = "intfloat/e5-base-v2"
PASSAGE_PREFIX = "passage: "  # E5 requires this; "query: " goes on queries at search time
DIM = 768
GPU_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "gpus.env")


# ----------------------------------------------------------------------------- corpus


class Corpus:
    """Random access to global row ranges across sorted parquet files."""

    def __init__(self, input_dir):
        files = sorted(glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True))
        if not files:
            sys.exit(f"no parquet files under {input_dir}")
        self.files = files
        counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
        self.offsets = [0]
        for c in counts:
            self.offsets.append(self.offsets[-1] + c)
        self.n = self.offsets[-1]
        self._cached_index = None
        self._cached = None

    def _load(self, i, columns):
        key = (i, tuple(columns))
        if self._cached_index != key:
            self._cached = pq.read_table(self.files[i], columns=list(columns)).to_pydict()
            self._cached_index = key
        return self._cached

    def rows(self, start, end, columns=("id", "title", "text")):
        out = {c: [] for c in columns}
        # first file whose range contains `start`
        fi = max(i for i in range(len(self.files)) if self.offsets[i] <= start)
        pos = start
        while pos < end:
            d = self._load(fi, columns)
            lo = pos - self.offsets[fi]
            hi = min(end, self.offsets[fi + 1]) - self.offsets[fi]
            for c in columns:
                out[c].extend(d[c][lo:hi])
            pos = self.offsets[fi] + hi
            fi += 1
        return out


def format_passage(title, text, title_format):
    """D-04b: how the title is attached. Record the choice in meta.json and the decision log.

    searchr1: '"Title"\\ntext'. Any quotes already on the DPR title are stripped first so they
              are not doubled. Check one line of Search-R1's wiki-18.jsonl before committing
              to this; the goal is byte-identical passages to theirs.
    raw:      'Title\\ntext' with the title exactly as stored (stray quotes kept).
    none:     text only. Note the first chunk of each article already starts with the title.
    """
    if title_format == "searchr1":
        return f'"{title.strip().strip(chr(34))}"\n{text}'
    if title_format == "raw":
        return f"{title}\n{text}"
    if title_format == "none":
        return text
    raise ValueError(title_format)


# ----------------------------------------------------------------------------- model


def load_model(name, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
    return tok, model


def encode_texts(texts, tok, model, max_length, batch_size, device):
    """Mean pooling over non-pad tokens, then L2 normalization (E5's recipe)."""
    import torch
    import torch.nn.functional as F

    out = np.empty((len(texts), DIM), dtype=np.float16)
    # longest first: similar lengths share a batch (less padding), and any OOM shows up at once
    order = np.argsort([-len(t) for t in texts], kind="stable")
    with torch.inference_mode():
        for b in range(0, len(texts), batch_size):
            idx = order[b : b + batch_size]
            enc = tok(
                [PASSAGE_PREFIX + texts[i] for i in idx],
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
            e = F.normalize(e.float(), dim=-1)
            out[idx] = e.cpu().numpy().astype(np.float16)
    return out


# ----------------------------------------------------------------------------- modes


def cmd_stats(args):
    from transformers import AutoTokenizer

    corpus = Corpus(args.input)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    rng = np.random.default_rng(0)
    picks = np.sort(rng.choice(corpus.n, size=min(args.sample, corpus.n), replace=False))
    lens = []
    for r in picks:
        row = corpus.rows(int(r), int(r) + 1)
        text = PASSAGE_PREFIX + format_passage(row["title"][0], row["text"][0], args.title_format)
        lens.append(len(tok(text, add_special_tokens=True)["input_ids"]))
    lens = np.array(lens)
    print(f"rows in corpus: {corpus.n:,}   sampled: {len(lens):,}   title_format: {args.title_format}")
    for p in (50, 90, 95, 99, 99.9):
        print(f"  p{p:<5} {np.percentile(lens, p):7.0f} tokens")
    print(f"  max    {lens.max():7d} tokens")
    for L in (128, 192, 256, 320, 512):
        print(f"  truncated at {L}: {np.mean(lens > L) * 100:5.2f}% of passages")


def meta_for(args, corpus):
    return {
        "model": args.model,
        "passage_prefix": PASSAGE_PREFIX,
        "pooling": "mean",
        "normalized": True,
        "dtype": "float16",
        "dim": DIM,
        "title_format": args.title_format,
        "max_length": args.max_length,
        "shard_size": args.shard_size,
        "n_rows": corpus.n,
        "files": [os.path.basename(f) for f in corpus.files],
    }


def write_or_check_meta(path, meta):
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != meta:
            diff = {k: (old.get(k), meta.get(k)) for k in set(old) | set(meta) if old.get(k) != meta.get(k)}
            sys.exit(f"settings differ from existing {path}: {diff}\nuse a new --out directory")
        return
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)


def save_atomic(path, array):
    tmp = path + f".tmp{os.getpid()}.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def shard_paths(out, k):
    return os.path.join(out, f"ids_{k:05d}.npy"), os.path.join(out, f"emb_{k:05d}.npy")


def allowed_gpus():
    """GPUS from configs/gpus.env, the single source of truth shared with scripts/gpus.sh."""
    with open(GPU_CONFIG) as f:
        for line in f:
            key, sep, value = line.split("#", 1)[0].strip().partition("=")
            if sep and key.strip() == "GPUS":
                return {int(g) for g in value.strip().split(",")}
    sys.exit(f"no GPUS= line in {GPU_CONFIG}")


def check_gpus():
    # without CUDA_VISIBLE_DEVICES every rank would land on GPU 0, or on a GPU that isn't ours
    allowed = sorted(allowed_gpus())
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis:
        sys.exit(f"set CUDA_VISIBLE_DEVICES to one of {allowed} (configs/gpus.env)")
    bad = [g for g in vis.split(",") if not (g.strip().isdigit() and int(g) in allowed)]
    if bad:
        sys.exit(f"CUDA_VISIBLE_DEVICES={vis}: {bad} not allowed, use only {allowed} (configs/gpus.env)")


def cmd_encode(args):
    import torch

    os.makedirs(args.out, exist_ok=True)
    corpus = Corpus(args.input)
    write_or_check_meta(os.path.join(args.out, "meta.json"), meta_for(args, corpus))

    n_shards = (corpus.n + args.shard_size - 1) // args.shard_size
    mine = [k for k in range(n_shards) if k % args.world_size == args.rank]
    todo = [k for k in mine if not os.path.exists(shard_paths(args.out, k)[1])]
    print(f"rank {args.rank}/{args.world_size}: {len(mine)} shards assigned, {len(todo)} left", flush=True)
    if not todo:
        return

    if not torch.cuda.is_available():
        sys.exit(f"CUDA unavailable (torch {torch.__version__}); refusing to encode on CPU")
    check_gpus()
    device = "cuda"
    tok, model = load_model(args.model, device)

    for k in todo:
        start, end = k * args.shard_size, min((k + 1) * args.shard_size, corpus.n)
        t0 = time.time()
        rows = corpus.rows(start, end)
        texts = [format_passage(t, x, args.title_format) for t, x in zip(rows["title"], rows["text"])]
        ids = np.array([int(i) for i in rows["id"]], dtype=np.int64)
        emb = encode_texts(texts, tok, model, args.max_length, args.batch_size, device)
        ids_path, emb_path = shard_paths(args.out, k)
        save_atomic(ids_path, ids)
        save_atomic(emb_path, emb)  # written last: its presence marks the shard as complete
        dt = time.time() - t0
        print(f"shard {k:05d} rows {start:,}-{end:,}  {dt:6.0f}s  {(end - start) / dt:7.0f} passages/s", flush=True)


def cmd_verify(args):
    corpus = Corpus(args.input)
    with open(os.path.join(args.out, "meta.json")) as f:
        meta = json.load(f)
    shard_size = meta["shard_size"]
    n_shards = (corpus.n + shard_size - 1) // shard_size
    problems = []

    missing = [k for k in range(n_shards) if not os.path.exists(shard_paths(args.out, k)[1])]
    if missing:
        sys.exit(f"{len(missing)} shards missing, e.g. {missing[:10]}")

    total, prev_last, rng = 0, None, np.random.default_rng(0)
    increasing, positional = True, True
    for k in range(n_shards):
        ids_path, emb_path = shard_paths(args.out, k)
        ids = np.load(ids_path)
        emb = np.load(emb_path, mmap_mode="r")
        start, end = k * shard_size, min((k + 1) * shard_size, corpus.n)
        if len(ids) != end - start or emb.shape != (end - start, meta["dim"]):
            problems.append(f"shard {k}: shape ids={ids.shape} emb={emb.shape}, expected {end - start} rows")
            continue
        # exact row -> passage id mapping against the corpus itself
        ref = np.array([int(i) for i in corpus.rows(start, end, columns=("id",))["id"]], dtype=np.int64)
        if not np.array_equal(ids, ref):
            problems.append(f"shard {k}: ids do not match corpus rows {start}-{end}")
        if np.any(np.diff(ids) <= 0) or (prev_last is not None and ids[0] <= prev_last):
            increasing = False
        if not np.array_equal(ids, np.arange(start + 1, end + 1)):
            positional = False
        prev_last = ids[-1]
        sample = np.asarray(emb[rng.choice(len(ids), size=min(256, len(ids)), replace=False)], dtype=np.float32)
        norms = np.linalg.norm(sample, axis=1)
        if np.any(np.abs(norms - 1) > 1e-2) or not np.all(np.isfinite(sample)):
            problems.append(f"shard {k}: norms out of range [{norms.min():.4f}, {norms.max():.4f}] or non-finite")
        total += len(ids)

    print(f"shards: {n_shards}   rows: {total:,} / {corpus.n:,}")
    print(f"ids strictly increasing: {increasing}   id == row + 1 everywhere: {positional}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("OK")


# ----------------------------------------------------------------------------- cli


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["stats", "encode", "verify"])
    p.add_argument("--input", required=True, help="directory of wiki_dpr no_embeddings parquet files")
    p.add_argument("--out", help="output directory for shards (encode, verify)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--title-format", default="searchr1", choices=["searchr1", "raw", "none"])
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--shard-size", type=int, default=500_000)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--sample", type=int, default=20_000, help="rows sampled in stats mode")
    args = p.parse_args()

    if args.mode != "stats" and not args.out:
        p.error("--out is required for encode and verify")
    {"stats": cmd_stats, "encode": cmd_encode, "verify": cmd_verify}[args.mode](args)


if __name__ == "__main__":
    main()