"""Inspect the wiki_dpr `no_embeddings` corpus and write a JSON report.

Usage:
    python inspect_corpus.py \
        --root /data/erfan/corpus/data/psgs_w100/no_embeddings \
        --out notes/corpus_inspection.json

Reads parquet metadata, the `id` column of every file (about 170 MB as int64),
and a uniform random sample of rows. Never loads the whole text column.
"""
import argparse
import glob
import json
import os
import random
import statistics
from collections import defaultdict

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def summarize(values):
    """Min / mean / percentiles / max of a list of numbers."""
    v = sorted(values)
    pct = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return {
        "min": v[0],
        "p05": pct(0.05),
        "median": statistics.median(v),
        "mean": round(statistics.mean(v), 2),
        "p95": pct(0.95),
        "max": v[-1],
    }


def truncate(row, n_chars):
    return {k: (v if not isinstance(v, str) or len(v) <= n_chars else v[:n_chars] + "...") for k, v in row.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/erfan/corpus/data/psgs_w100/no_embeddings")
    ap.add_argument("--out", default="corpus_inspection.json")
    ap.add_argument("--sample", type=int, default=1000, help="rows drawn uniformly from the whole corpus")
    ap.add_argument("--n_examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = sorted(glob.glob(os.path.join(args.root, "*.parquet")))
    assert files, f"no parquet files in {args.root}"
    report = {"corpus_dir": args.root, "seed": args.seed}

    # 1. Schema, taken from the first file and checked against the rest.
    schema = pq.read_schema(files[0])
    report["schema"] = {f.name: str(f.type) for f in schema}
    mismatched = [os.path.basename(f) for f in files if pq.read_schema(f).names != schema.names]
    report["schema_mismatched_files"] = mismatched

    # 2. Per-file metadata and the id column of every file.
    file_info, id_chunks = [], []
    offsets = []  # (global_start, file_idx, row_group_idx, n_rows) for uniform sampling
    total = 0
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        md = pf.metadata
        for rg in range(md.num_row_groups):
            n = md.row_group(rg).num_rows
            offsets.append((total, fi, rg, n))
            total += n
        ids = pc.cast(pq.read_table(f, columns=["id"]).column("id"), "int64").to_numpy()
        id_chunks.append(ids)
        file_info.append({
            "file": os.path.basename(f),
            "bytes": os.path.getsize(f),
            "rows": md.num_rows,
            "row_groups": md.num_row_groups,
            "first_id": int(ids[0]),
            "last_id": int(ids[-1]),
            "ids_sorted_within_file": bool(np.all(np.diff(ids) > 0)),
        })

    all_ids = np.concatenate(id_chunks)
    sorted_ids = np.sort(all_ids)
    diffs = np.diff(sorted_ids)
    report["files"] = {
        "count": len(files),
        "total_bytes": sum(x["bytes"] for x in file_info),
        "total_rows": total,
        "rows_per_file": summarize([x["rows"] for x in file_info]),
        "first_3": file_info[:3],
        "last_1": file_info[-1:],
    }
    report["ids"] = {
        "dtype_on_disk": report["schema"].get("id"),
        "min": int(sorted_ids[0]),
        "max": int(sorted_ids[-1]),
        "unique": int(len(sorted_ids) - np.sum(diffs == 0)),
        "duplicates": int(np.sum(diffs == 0)),
        "gaps_in_range": int(np.sum(diffs > 1)),
        "sorted_across_files": bool(np.all(np.diff(all_ids) > 0)),
        "all_files_sorted_internally": all(x["ids_sorted_within_file"] for x in file_info),
    }
    del all_ids, sorted_ids, diffs, id_chunks

    # 3. The first rows on disk, i.e. the start of the first article.
    head = pq.ParquetFile(files[0]).read_row_group(0).slice(0, args.n_examples).to_pylist()
    report["head_rows"] = [truncate(r, 300) for r in head]

    # 4. Uniform random sample over the whole corpus (not clustered by file),
    #    grouped by row group so each row group is read once.
    picks = sorted(rng.sample(range(total), min(args.sample, total)))
    by_rg = defaultdict(list)
    j = 0
    for start, fi, rg, n in offsets:
        while j < len(picks) and picks[j] < start + n:
            by_rg[(fi, rg)].append(picks[j] - start)
            j += 1
    rows = []
    for (fi, rg), local in by_rg.items():
        t = pq.ParquetFile(files[fi]).read_row_group(rg)
        rows.extend(t.take(local).to_pylist())

    words = [len(r["text"].split()) for r in rows]
    chars = [len(r["text"]) for r in rows]
    titles = [r["title"] for r in rows]
    quoted = [t for t in titles if len(t) >= 2 and t.startswith('"') and t.endswith('"')]
    report["random_sample"] = {
        "n": len(rows),
        "method": "uniform over global row index",
        "words_per_passage": summarize(words),
        "passages_not_100_words": sum(w != 100 for w in words),
        "chars_per_passage": summarize(chars),
        "empty_text": sum(1 for r in rows if not r["text"].strip()),
        "distinct_titles": len(set(titles)),
        "titles_wrapped_in_quotes_fraction": round(len(quoted) / len(titles), 3),
        "text_starts_with_its_title_fraction": round(
            sum(r["text"].startswith(r["title"].strip('"')) for r in rows) / len(rows), 3),
        "example_titles": rng.sample(titles, min(10, len(titles))),
        "example_rows": [truncate(r, 400) for r in rng.sample(rows, min(args.n_examples, len(rows)))],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()