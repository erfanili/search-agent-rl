#!/usr/bin/env python
"""
Build the FAISS index and the passage store from encode_corpus.py shards.

Step P0.6 of the search-agent RL project.

Inputs:
  --emb-dir   output of `encode_corpus.py encode` (meta.json, ids_*.npy, emb_*.npy)
  --corpus    the same no_embeddings/ parquet directory that was encoded

Outputs (--out):
  index.faiss         FAISS index, inner-product metric (E5 vectors are unit-norm, so IP = cosine)
  ids.npy             int64, row r of the index -> wiki_dpr passage id
  docs.bin            UTF-8 passage contents, concatenated, in index row order
  docs_offsets.npy    int64 [n+1]; contents of row r = docs.bin[off[r]:off[r+1]]
  index_meta.json     factory string, train size, counts, and the encoding meta.json it was built from

Passage contents use Search-R1's corpus format, '"Title"\\ntext', whatever --title-format was
used for encoding. That is what the agent sees, and verl's search tool splits it on the first
newline into title and text.

Index choice (D-05):
  --factory "IVF4096,SQ8"  ~16 GB RAM; 4096 k-means lists, 8-bit scalar-quantized vectors.
                           Needs training and an nprobe setting at query time; approximate.
  --factory "Flat"         ~65 GB RAM (fits: the node has 503 GB); exact search; what Search-R1
                           used. Brute force, so CPU latency under many concurrent rollouts is the
                           thing to measure.

Example: scripts/build_index.sh
"""

import argparse
import json
import os
import sys
import time

import faiss
import numpy as np
import pyarrow.parquet as pq

from encode_corpus import Corpus, format_passage, shard_paths


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_emb_meta(emb_dir):
    with open(os.path.join(emb_dir, "meta.json")) as f:
        meta = json.load(f)
    n_shards = (meta["n_rows"] + meta["shard_size"] - 1) // meta["shard_size"]
    missing = [k for k in range(n_shards) if not os.path.exists(shard_paths(emb_dir, k)[1])]
    if missing:
        sys.exit(f"{len(missing)} shards missing (e.g. {missing[:5]}); finish encoding and run verify first")
    return meta, n_shards


def sample_training_vectors(emb_dir, meta, n_shards, train_size, seed):
    """Uniform sample across the whole corpus, drawn proportionally from each shard."""
    rng = np.random.default_rng(seed)
    n, size = meta["n_rows"], meta["shard_size"]
    parts = []
    for k in range(n_shards):
        rows_k = min(size, n - k * size)
        take = min(rows_k, int(round(train_size * rows_k / n)))
        emb = np.load(shard_paths(emb_dir, k)[1], mmap_mode="r")
        idx = np.sort(rng.choice(rows_k, size=take, replace=False))
        parts.append(np.asarray(emb[idx], dtype=np.float32))
    return np.ascontiguousarray(np.concatenate(parts))


def build_docstore(corpus, ids, out):
    """Write contents in corpus row order and check row ids against the index's ids.npy."""
    off_path, bin_path = os.path.join(out, "docs_offsets.npy"), os.path.join(out, "docs.bin")
    offsets = np.zeros(corpus.n + 1, dtype=np.int64)
    pos = 0
    tmp = bin_path + ".tmp"
    with open(tmp, "wb") as f:
        for fi, path in enumerate(corpus.files):
            d = pq.read_table(path, columns=["id", "title", "text"]).to_pydict()
            file_ids = np.array([int(i) for i in d["id"]], dtype=np.int64)
            if not np.array_equal(file_ids, ids[pos : pos + len(file_ids)]):
                sys.exit(f"corpus ids in {os.path.basename(path)} do not match ids.npy at rows {pos}+")
            blobs = [format_passage(t, x, "searchr1").encode("utf-8") for t, x in zip(d["title"], d["text"])]
            lens = np.fromiter((len(b) for b in blobs), dtype=np.int64, count=len(blobs))
            offsets[pos + 1 : pos + 1 + len(blobs)] = offsets[pos] + np.cumsum(lens)
            f.write(b"".join(blobs))
            pos += len(blobs)
            log(f"docstore: {pos:,} / {corpus.n:,} rows")
    if pos != len(ids):
        sys.exit(f"docstore rows {pos} != index rows {len(ids)}")
    os.replace(tmp, bin_path)
    np.save(off_path, offsets)
    log(f"docstore: {pos:,} rows, {offsets[-1] / 1e9:.1f} GB")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--factory", default="IVF4096,SQ8")
    p.add_argument("--train-size", type=int, default=1_000_000, help="vectors used to train IVF/SQ (~250 per list)")
    p.add_argument("--gpu-train", action="store_true", help="run IVF k-means on GPU(s); much faster")
    p.add_argument("--add-chunk", type=int, default=200_000)
    p.add_argument("--threads", type=int, default=0, help="0 = faiss default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-docstore", action="store_true")
    a = p.parse_args()

    if a.threads:
        faiss.omp_set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    meta, n_shards = load_emb_meta(a.emb_dir)
    dim, n = meta["dim"], meta["n_rows"]

    ids = np.concatenate([np.load(shard_paths(a.emb_dir, k)[0]) for k in range(n_shards)])
    if len(ids) != n:
        sys.exit(f"ids total {len(ids)} != meta n_rows {n}")
    log(f"{n:,} vectors, dim {dim}, {n_shards} shards, factory '{a.factory}'")

    index = faiss.index_factory(dim, a.factory, faiss.METRIC_INNER_PRODUCT)
    train_s = 0.0
    if not index.is_trained:
        x = sample_training_vectors(a.emb_dir, meta, n_shards, a.train_size, a.seed)
        log(f"training on {len(x):,} vectors")
        if a.gpu_train and faiss.get_num_gpus() > 0:
            ivf = faiss.extract_index_ivf(index)
            ivf.clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(dim))
            log(f"k-means on {faiss.get_num_gpus()} GPU(s)")
        t0 = time.time()
        index.train(x)
        train_s = time.time() - t0
        log(f"trained in {train_s:.0f}s")
        del x

    t0 = time.time()
    for k in range(n_shards):  # shard order = corpus row order = index row order
        emb = np.load(shard_paths(a.emb_dir, k)[1], mmap_mode="r")
        for s in range(0, len(emb), a.add_chunk):
            index.add(np.ascontiguousarray(emb[s : s + a.add_chunk], dtype=np.float32))
        log(f"added shard {k + 1}/{n_shards}, ntotal {index.ntotal:,}")
    if index.ntotal != n:
        sys.exit(f"index ntotal {index.ntotal} != {n}")
    add_s = time.time() - t0

    idx_path = os.path.join(a.out, "index.faiss")
    faiss.write_index(index, idx_path + ".tmp")
    os.replace(idx_path + ".tmp", idx_path)
    np.save(os.path.join(a.out, "ids.npy"), ids)
    log(f"index written: {os.path.getsize(idx_path) / 1e9:.1f} GB")

    if not a.skip_docstore:
        build_docstore(Corpus(a.corpus), ids, a.out)

    with open(os.path.join(a.out, "index_meta.json"), "w") as f:
        json.dump(
            {
                "factory": a.factory,
                "metric": "inner_product",
                "train_size": a.train_size if train_s else 0,
                "train_seconds": round(train_s),
                "add_seconds": round(add_s),
                "ntotal": int(index.ntotal),
                "contents_format": "searchr1",
                "encoding": meta,
            },
            f,
            indent=2,
        )
    log("done")


if __name__ == "__main__":
    main()