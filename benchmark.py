"""Benchmark FAISS indexes on a locally downloaded BEIR dataset."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL = "BAAI/bge-small-en-v1.5"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_data(root, split):
    corpus = read_jsonl(root / "corpus.jsonl")
    queries = {str(q["_id"]): q["text"] for q in read_jsonl(root / "queries.jsonl")}
    qrels = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if float(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = float(row["score"])
    ids = [str(d["_id"]) for d in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate corpus IDs")
    corpus_ids = set(ids)
    missing = set(qrels) - set(queries)
    missing_docs = {doc for rels in qrels.values() for doc in rels if doc not in corpus_ids}
    if missing or missing_docs:
        raise ValueError(f"Missing {len(missing)} queries and {len(missing_docs)} judged documents")
    qids = sorted(qrels)
    if not qids:
        raise ValueError("No positively judged queries")
    texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in corpus]
    return ids, texts, qids, [queries[q] for q in qids], qrels


def metrics(found, exact, relevant, k):
    found = [x for x in found if x is not None]
    ann = len(set(found) & set(exact)) / k
    recall = len(set(found) & set(relevant)) / len(relevant)
    dcg = sum((2 ** relevant.get(doc, 0) - 1) / math.log2(rank + 2) for rank, doc in enumerate(found))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return ann, recall, dcg / idcg if idcg else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="BEIR directory containing corpus.jsonl, queries.jsonl and qrels/")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("results/run"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face model revision/commit")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.k < 1 or args.nlist < 1 or args.repeats < 1 or args.batch_size < 1:
        parser.error("k, nlist, repeats and batch-size must be positive")
    import numpy as np
    import faiss
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    faiss.omp_set_num_threads(1)
    ids, docs, qids, queries, qrels = load_data(args.dataset, args.split)
    if args.k > len(ids) or args.nlist > len(ids):
        parser.error("k and nlist cannot exceed the corpus size")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    # Content-based cache prevents silently reusing vectors for different inputs.
    key = hashlib.sha256(json.dumps([MODEL, args.revision, ids, docs, qids, queries, QUERY_PREFIX], ensure_ascii=False).encode()).hexdigest()
    cache = args.cache / (key + ".npz")
    started = time.perf_counter()
    if cache.exists():
        with np.load(cache) as saved:
            vectors, qvectors = saved["vectors"], saved["queries"]
        embedding_seconds = None
    else:
        model = SentenceTransformer(MODEL, revision=args.revision, device="cpu")
        vectors = model.encode(docs, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype("float32")
        qvectors = model.encode([QUERY_PREFIX + q for q in queries], batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype("float32")
        embedding_seconds = time.perf_counter() - started
        np.savez(cache, vectors=vectors, queries=qvectors)
    vectors, qvectors = np.ascontiguousarray(vectors), np.ascontiguousarray(qvectors)
    dimension = vectors.shape[1]
    flat = faiss.IndexFlatIP(dimension)
    started = time.perf_counter(); flat.add(vectors); flat_build = time.perf_counter() - started
    _, truth = flat.search(qvectors, args.k)
    configs = [("flat", flat, 0.0, flat_build, None)]
    hnsw = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    started = time.perf_counter(); hnsw.add(vectors); build = time.perf_counter() - started
    for ef in (16, 32, 64, 128):
        configs.append(("hnsw", hnsw, 0.0, build, ef))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, args.nlist, faiss.METRIC_INNER_PRODUCT)
    started = time.perf_counter(); ivf.train(vectors); train = time.perf_counter() - started
    started = time.perf_counter(); ivf.add(vectors); build = time.perf_counter() - started
    for nprobe in sorted({min(x, args.nlist) for x in (1, 4, 16, 32)}):
        configs.append(("ivf_flat", ivf, train, build, nprobe))
    rows, summary = [], []
    for name, index, train, build, parameter in configs:
        if name == "hnsw": index.hnsw.efSearch = parameter
        if name == "ivf_flat": index.nprobe = parameter
        size = len(faiss.serialize_index(index))
        index.search(qvectors[:min(10, len(qvectors))], args.k)
        times, scores = [], []
        for qi, vector in enumerate(qvectors):
            found = None
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                _, result = index.search(vector.reshape(1, -1), args.k)
                times.append((time.perf_counter_ns() - started) / 1e6)
                found = result[0]
            found_ids = [ids[i] if i >= 0 else None for i in found]
            exact_ids = [ids[i] for i in truth[qi] if i >= 0]
            values = metrics(found_ids, exact_ids, qrels[qids[qi]], args.k)
            scores.append(values)
            rows.append({"method": name, "parameter": parameter, "query_id": qids[qi], "ann_recall": values[0], "evidence_recall": values[1], "ndcg": values[2], "result_ids": found_ids, "exact_ids": exact_ids})
        means = np.mean(scores, axis=0)
        summary.append({"method": name, "parameter": parameter, "train_seconds": train, "add_seconds": build, "serialized_bytes": size, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "ann_recall": float(means[0]), "evidence_recall": float(means[1]), "ndcg": float(means[2])})
        print(name, parameter, summary[-1], flush=True)
    metadata = {"dataset": str(args.dataset), "split": args.split, "model": MODEL, "revision": args.revision, "cache_key": key, "cache_hit": embedding_seconds is None, "embedding_seconds": embedding_seconds, "corpus_size": len(ids), "query_count": len(qids), "k": args.k, "nlist": args.nlist, "repeats": args.repeats, "threads": 1, "python": sys.version, "platform": platform.platform(), "processor": platform.processor(), "faiss": faiss.__version__, "numpy": np.__version__, "sentence_transformers": sentence_transformers.__version__}
    (args.output / "results.json").write_text(json.dumps({"metadata": metadata, "summary": summary, "queries": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
