#!/usr/bin/env python
"""
HTTP retrieval server over the index built by build_index.py.

Step P0.6 of the search-agent RL project.

API matches Search-R1's search_r1/search/retrieval_server.py, so verl's SearchTool can point at it:

  POST /retrieve
    {"queries": ["who wrote ..."], "topk": 3, "return_scores": true}
  -> {"result": [[{"document": {"id": "123", "contents": "\\"Title\\"\\ntext"}, "score": 0.83}, ...], ...]}
     (with return_scores false, each inner item is just the document dict)

  GET /health -> index size, nprobe, process memory

Queries get E5's "query: " prefix, mean pooling, L2 normalization: the same recipe as the
passages in encode_corpus.py, which is what makes inner product meaningful.

Example: scripts/serve_retrieval.sh
"""

import argparse
import json
import os
import threading
import time
from typing import List, Optional

import faiss
import numpy as np

QUERY_PREFIX = "query: "


class E5QueryEncoder:
    def __init__(self, model_name, device, max_length=128):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.lock = threading.Lock()  # one GPU forward at a time; FAISS search runs outside it

    def encode(self, queries):
        torch = self.torch
        with self.lock, torch.inference_mode():
            enc = self.tok(
                [QUERY_PREFIX + q for q in queries],
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            return np.ascontiguousarray(e.cpu().numpy(), dtype=np.float32)


class PassageStore:
    """Memory-mapped contents; only the pages for returned passages are touched."""

    def __init__(self, index_dir):
        self.offsets = np.load(os.path.join(index_dir, "docs_offsets.npy"), mmap_mode="r")
        self.data = np.memmap(os.path.join(index_dir, "docs.bin"), dtype=np.uint8, mode="r")

    def get(self, row):
        a, b = int(self.offsets[row]), int(self.offsets[row + 1])
        return bytes(self.data[a:b]).decode("utf-8")


class Retriever:
    def __init__(self, index_dir, model_name, device, nprobe, max_query_length):
        with open(os.path.join(index_dir, "index_meta.json")) as f:
            self.meta = json.load(f)
        if self.meta["encoding"]["model"] != model_name:
            raise SystemExit(f"index was encoded with {self.meta['encoding']['model']}, not {model_name}")
        t0 = time.time()
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        self.nprobe = None
        try:
            ivf = faiss.extract_index_ivf(self.index)
            ivf.nprobe = nprobe
            self.nprobe = nprobe
        except RuntimeError:
            pass  # flat index: no nprobe
        self.ids = np.load(os.path.join(index_dir, "ids.npy"))
        self.store = PassageStore(index_dir)
        self.encoder = E5QueryEncoder(model_name, device, max_query_length)
        self.load_seconds = time.time() - t0

    def search(self, queries, k):
        q = self.encoder.encode(queries)
        scores, rows = self.index.search(q, k)
        out = []
        for srow, rrow in zip(scores, rows):
            hits = []
            for s, r in zip(srow, rrow):
                if r < 0:  # IVF can return fewer than k when nprobe is small
                    continue
                hits.append(({"id": str(int(self.ids[r])), "contents": self.store.get(int(r))}, float(s)))
            out.append(hits)
        return out


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def make_app(retriever, default_topk):
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class QueryRequest(BaseModel):
        queries: List[str]
        topk: Optional[int] = None
        return_scores: bool = False

    @app.post("/retrieve")
    def retrieve(req: QueryRequest):  # sync def: FastAPI runs it in a thread pool
        k = req.topk or default_topk
        results = retriever.search(req.queries, k)
        if req.return_scores:
            return {"result": [[{"document": d, "score": s} for d, s in hits] for hits in results]}
        return {"result": [[d for d, _ in hits] for hits in results]}

    @app.get("/health")
    def health():
        return {
            "ntotal": int(retriever.index.ntotal),
            "factory": retriever.meta["factory"],
            "nprobe": retriever.nprobe,
            "default_topk": default_topk,
            "load_seconds": round(retriever.load_seconds, 1),
            "rss_gb": rss_gb(),
        }

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--topk", type=int, default=3, help="default when a request omits topk (Search-R1: 3)")
    p.add_argument("--max-query-length", type=int, default=128)
    p.add_argument("--threads", type=int, default=0, help="FAISS OpenMP threads; 0 = default")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    import uvicorn

    if a.device.startswith("cuda"):
        from encode_corpus import check_gpus  # same GPU allow-list as encoding

        check_gpus()
    if a.threads:
        faiss.omp_set_num_threads(a.threads)
    retriever = Retriever(a.index_dir, a.model, a.device, a.nprobe, a.max_query_length)
    print(f"loaded {retriever.index.ntotal:,} vectors in {retriever.load_seconds:.0f}s, rss {rss_gb()} GB", flush=True)
    uvicorn.run(make_app(retriever, a.topk), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()