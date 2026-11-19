"""Benchmark FAISS indexes on a locally downloaded BEIR dataset."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL = "BAAI/bge-small-en-v1.5"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def encode_checkpointed(model, texts, cache_root, revision, batch_size, chunk_size=512):
    """Reuse completed chunks, including corpus chunks across evaluation splits."""
    import numpy as np
    key = hashlib.sha256(json.dumps([MODEL, revision, texts, chunk_size], ensure_ascii=False).encode()).hexdigest()
    directory = cache_root / 'chunks' / key
    directory.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(texts), chunk_size):
        stop = min(start + chunk_size, len(texts))
        path = directory / f'{start:08d}.npy'
        if path.exists():
            vectors = np.load(path, allow_pickle=False)
            if vectors.ndim != 2 or len(vectors) != stop - start or not np.isfinite(vectors).all():
                raise ValueError(f'Invalid embedding checkpoint: {path}')
            action = 'Reused'
        else:
            vectors = model.encode(texts[start:stop], batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True).astype('float32')
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, vectors, allow_pickle=False)
            temporary.replace(path)
            action = 'Saved'
        chunks.append(vectors)
        print(f'{action} embedding checkpoint {stop}/{len(texts)} ({key[:12]})', flush=True)
    return np.concatenate(chunks)


def select_queries(qids, queries, partition, fraction, seed):
    """Freeze a disjoint development/held-out partition before parameter selection."""
    if partition == "all":
        return qids, queries
    if len(qids) < 2:
        raise ValueError("A development holdout requires at least two queries")
    shuffled = sorted(qids)
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(qids) - 1, int(len(qids) * fraction)))
    development = set(shuffled[:count])
    selected = [(qid, query) for qid, query in zip(qids, queries)
                if (qid in development) == (partition == "dev")]
    return [item[0] for item in selected], [item[1] for item in selected]


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
    parser.add_argument("--partition", choices=("all", "dev", "heldout"), default="all")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 4, 16, 32])
    args = parser.parse_args()
    if args.k < 1 or args.nlist < 1 or args.repeats < 1 or args.batch_size < 1:
        parser.error("k, nlist, repeats and batch-size must be positive")
    if not 0 < args.dev_fraction < 1:
        parser.error("dev-fraction must be between zero and one")
    if min(args.ef_search + args.nprobe) < 1:
        parser.error("search parameters must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("revision must be a full 40-character model commit hash")
    if (args.output / "results.json").exists():
        parser.error("output already contains results; choose a new output directory")
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
        vectors = encode_checkpointed(model, docs, args.cache, args.revision, args.batch_size)
        qvectors = encode_checkpointed(model, [QUERY_PREFIX + q for q in queries], args.cache, args.revision, args.batch_size)
        embedding_seconds = time.perf_counter() - started
        temporary = cache.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, vectors=vectors, queries=qvectors)
        temporary.replace(cache)
    vectors, qvectors = np.ascontiguousarray(vectors), np.ascontiguousarray(qvectors)
    all_qids = qids
    qids, queries = select_queries(qids, queries, args.partition, args.dev_fraction, args.seed)
    positions = {qid: i for i, qid in enumerate(all_qids)}
    qvectors = np.ascontiguousarray(qvectors[[positions[qid] for qid in qids]])
    dimension = vectors.shape[1]
    flat = faiss.IndexFlatIP(dimension)
    started = time.perf_counter(); flat.add(vectors); flat_build = time.perf_counter() - started
    _, truth = flat.search(qvectors, args.k)
    configs = [("flat", flat, 0.0, flat_build, None)]
    hnsw = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    started = time.perf_counter(); hnsw.add(vectors); build = time.perf_counter() - started
    for ef in sorted(set(args.ef_search)):
        configs.append(("hnsw", hnsw, 0.0, build, ef))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, args.nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.cp.seed = args.seed
    started = time.perf_counter(); ivf.train(vectors); train = time.perf_counter() - started
    started = time.perf_counter(); ivf.add(vectors); build = time.perf_counter() - started
    for nprobe in sorted({min(x, args.nlist) for x in args.nprobe}):
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
            query_times = []
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                _, result = index.search(vector.reshape(1, -1), args.k)
                query_times.append((time.perf_counter_ns() - started) / 1e6)
                found = result[0]
            times.extend(query_times)
            found_ids = [ids[i] if i >= 0 else None for i in found]
            exact_ids = [ids[i] for i in truth[qi] if i >= 0]
            values = metrics(found_ids, exact_ids, qrels[qids[qi]], args.k)
            scores.append(values)
            rows.append({"method": name, "parameter": parameter, "query_id": qids[qi], "ann_recall": values[0], "evidence_recall": values[1], "ndcg": values[2], "result_ids": found_ids, "exact_ids": exact_ids})
            rows[-1]["latency_ms"] = query_times
        means = np.mean(scores, axis=0)
        summary.append({"method": name, "parameter": parameter, "train_seconds": train, "add_seconds": build, "serialized_bytes": size, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "ann_recall": float(means[0]), "evidence_recall": float(means[1]), "ndcg": float(means[2])})
        print(name, parameter, summary[-1], flush=True)
        summary[-1]["build_seconds"] = train + build
        atomic_json(args.output / 'partial-results.json', {'summary': summary, 'queries': rows})
    metadata = {"dataset": str(args.dataset), "split": args.split, "model": MODEL, "revision": args.revision, "cache_key": key, "cache_hit": embedding_seconds is None, "embedding_seconds": embedding_seconds, "corpus_size": len(ids), "query_count": len(qids), "k": args.k, "nlist": args.nlist, "repeats": args.repeats, "threads": 1, "python": sys.version, "platform": platform.platform(), "processor": platform.processor(), "faiss": faiss.__version__, "numpy": np.__version__, "sentence_transformers": sentence_transformers.__version__}
    metadata.update(partition=args.partition, dev_fraction=args.dev_fraction, seed=args.seed,
                    query_ids=qids, ef_search=args.ef_search, nprobe=args.nprobe,
                    hnsw_m=16, ef_construction=200)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    atomic_json(args.output / 'results.json', {'metadata': metadata, 'summary': summary, 'queries': rows})


if __name__ == "__main__":
    main()
