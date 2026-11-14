# RAG retrieval benchmark

This first implementation compares exact inner-product scan, HNSW, and IVF-Flat on the same normalized BGE embeddings and BEIR document-level relevance judgments. It measures ANN recall, evidence recall, NDCG, build time, serialized index size and single-query search latency. It does not generate answers.

## Setup

Use Python 3.10–3.12 in a virtual environment; FAISS and PyTorch availability depends on the platform. For Windows, WSL with a Linux Python environment may be needed. Install `pip install -r requirements.txt`. Download and extract [SciFact or FiQA from BEIR](https://github.com/beir-cellar/beir#datasets) into `data/`, preserving `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`.

Choose an immutable commit from the [BGE model repository](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/main) and run:

```bash
python benchmark.py data/scifact --revision MODEL_COMMIT --output results/scifact-test
python benchmark.py data/fiqa --revision MODEL_COMMIT --output results/fiqa-test
python -m unittest
```

The first run downloads the model, encodes on CPU, and caches embeddings keyed by input content and model revision. Results contain configuration, library and hardware metadata, index summaries, and per-query IDs and metrics. Use `--split dev` when available to select settings; do not tune using the test split. Dataset downloads and model downloads require network access.

The code indexes each whole BEIR document, joining title and body; the model truncates inputs to its token limit. Check truncation before drawing conclusions. All positive qrels count as relevant; unjudged documents are treated as nonrelevant for the reported metrics. Latency repeats are pooled across queries in fixed order after warm-up, and serialized size is not resident memory. Ties follow FAISS's returned ordering. FAISS may warn about insufficient IVF training data on small corpora; adjust `--nlist` and record it. This initial version does not implement scaling subsets, confidence intervals, repeated builds, truncation reporting, or downstream generation.
