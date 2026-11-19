# Project plan: Search algorithms in retrieval-augmented generation

## Project focus

**Research question:** How do exact and approximate nearest-neighbor search methods affect retrieval speed, memory use, and retrieval quality in a RAG system, and when do those differences affect the final response?

The paper should study the retrieval algorithm as one component of a RAG pipeline. Its contribution is a controlled comparison of algorithms, supported by a theoretical analysis, rather than an attempt to find the best combination of every RAG component. The comparison should explain both *why* methods have different costs and *whether* those differences matter for retrieving useful material.

The [project abstract](abstract.md) identifies speed, precision, memory, and tokens as possible tradeoffs. A useful refinement is to separate three concepts: approximation to the exact vector-search result, relevance of retrieved documents to the query, and correctness of an answer generated from those documents. They are related but are not interchangeable.

## How the system fits together

```text
Corpus -> fixed document preparation -> fixed embedding model -> normalized vectors
                                                                  |
                                    +-----------------------------+----------------------+
                                    |              |              |                      |
                                 exact scan       HNSW         IVF-Flat               IVF-PQ / LSH
                                    |              |              |                      |
Query -> same query encoder --------+--------------+--------------+----------------------+
                                    |              |              |                      |
                                    +---------- ranked document IDs ---------------------+
                                                       |
                                      common retrieval evaluation and optional generator
```

Every method receives the **same vectors, query vectors, distance objective, document IDs, and k**. Only the index and its search parameters change. Use a shared interface such as `build(vectors)`, `search(query_vectors, k)`, and `index_size()`; keep a mapping from returned vector IDs to the same original documents. This makes a wide-ranging algorithm comparison possible without turning the experiment into a comparison of unrelated pipelines.

## Recommended models

### Embedding model: BAAI/bge-small-en-v1.5

Use [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) as the primary embedding model. It is a relatively small English retrieval model that can be run locally, and the top RAG paper includes bge-en-small among its embedding models. A single locally run model avoids API costs and model changes during experiments. Record the exact model revision and software versions.

Use the model's recommended short-query instruction consistently for queries, with no instruction for corpus documents. Apply the same preprocessing and L2 normalization to all vectors. With normalized vectors, cosine ranking and inner-product ranking are equivalent; exact squared L2 ranking is also equivalent, subject to floating-point ties. Configure each index accordingly. Some methods approximate that objective through compressed representations: describe their actual scoring explicitly rather than assuming all returned scores are cosine similarities.

Embedding time is measured separately from search time, and corpus and query embeddings are computed once and reused. A second embedding model could be a later sensitivity experiment, but changing the embedding model is not part of the primary algorithm comparison.

### Generation model: Qwen2.5-3B-Instruct, optional

If hardware permits, use [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) as a fixed, locally run generator for a small downstream experiment. Check its model card, license, and hardware needs before use; a quantized copy may be useful on limited hardware, but record its revision and quantization. If it does not fit, use one accessible model consistently and record its settings. Do not require generation to complete the core search study.

Keep the prompt, context order, maximum context budget, decoding settings, and output limit fixed. Ask the model to answer only from supplied passages, cite document IDs, and indicate when evidence is insufficient. Record input and output tokens separately. Identical k does not guarantee identical token counts because retrieved documents can have different lengths; index choice does not inherently change token use. Keep generation latency separate from index search latency.

## Datasets

Use [BEIR](https://github.com/beir-cellar/beir) as the common retrieval framework because it provides corpora, queries, and relevance judgments in a consistent format. The initial choices offer different domains and scales without requiring several million vectors.

| Dataset | Role | Approximate size | Why use it? |
| --- | --- | --- | --- |
| SciFact | Pilot and scientific-domain evaluation | 5,183 documents; 300 judged test queries | Small enough to validate the full pipeline and manually inspect results. Queries are scientific claims, so describe this as evidence retrieval, not ordinary question answering. |
| FiQA-2018 | Main cross-domain and scaling study | About 57,000 documents; 648 judged test queries | Financial questions and community answers give a different domain and a larger corpus. |
| NFCorpus | Optional third-domain check | About 3,600 documents; 323 judged test queries | Nutrition and medical retrieval gives another domain, though its small corpus will not establish large-scale performance. |

Counts refer to the BEIR benchmark's corpus and judged test queries, not necessarily all available queries; verify the downloaded version and record exclusions. BEIR's [dataset table](https://github.com/beir-cellar/beir/blob/main/README.md) provides dataset links and statistics. Review each dataset's terms before downloading or redistributing it.

**Document unit:** Start by indexing each BEIR document as one unit, combining title and body in a documented order. Preserve document IDs and relevance judgments. Record truncation caused by the embedding model's token limit. If substantial truncation makes fixed-size chunks necessary, map every chunk back to its parent document and aggregate results consistently before evaluating document-level judgments; use an identical chunking and aggregation policy across methods. Chunk-level evidence cannot be inferred automatically from document-level labels.

**Scaling:** For FiQA, run reproducible nested corpus sizes such as 5,000, 20,000, and the full corpus. Select queries and construct subsets ahead of time; retain each selected query's judged relevant documents in every subset, and report how subsets were constructed and which queries qualify. Full-corpus results are the primary relevance comparison, since changing the corpus changes the search problem. Do not claim internet-scale behavior from 57,000 documents. A larger BEIR corpus is an optional extension if hardware and time allow.

**Generation data:** BEIR relevance labels alone are not necessarily gold answers. For an optional answer-quality study, use a separately identified question-answer dataset with reference answers and an associated searchable corpus, or create and manually verify a small set of question–answer–evidence triples from the corpus. Do not treat FiQA relevance judgments as answer keys. Select and freeze the questions before comparing algorithms.

## Algorithms: broad coverage with a common objective

| Method | Mechanism and role | Starting configuration and tuning |
| --- | --- | --- |
| Exact flat scan | Computes similarity against every stored vector; provides the exact top-k reference. | FAISS `IndexFlatIP`, normalized vectors. No approximation parameter. |
| HNSW | Traverses a graph of neighboring vectors; offers a search effort versus recall tradeoff, with graph memory overhead. | FAISS `IndexHNSWFlat`, inner product where supported. Start with M=16, construction effort 100–200, and test `efSearch` 16, 32, 64, 128. |
| IVF-Flat | Clusters vectors into inverted lists and searches selected lists, retaining full vectors inside lists. | FAISS `IndexIVFFlat`; for FiQA pilot `nlist=128`, test `nprobe` 1, 4, 16, 32. Train on a representative sample of corpus vectors only. |
| IVF-PQ | Combines inverted lists with product-quantized vector codes, reducing stored vector size while introducing distance approximation. | FAISS `IndexIVFPQ`; start with `nlist=128`, 16 subquantizers and 8 bits per code, then use the same `nprobe` sweep. Confirm dimensions and training sample are sufficient; reduce complexity or omit on small datasets if training is unreliable. |
| LSH | Uses locality-sensitive binary codes to approximate similarity. | Optional breadth extension: FAISS `IndexLSH` with a small bit-count sweep. FAISS scans its binary codes exhaustively, so label it **binary-code flat LSH**, not hash-table candidate lookup. Its Hamming scores are approximate proxies, not cosine scores. |

The settings are starting points, not guaranteed optimal values. Run a feasibility pilot, ensure every requested k is supported, and document failures rather than silently changing methods. Use a development query set for choosing settings and held-out test queries for reporting; where a dataset lacks a suitable development split, reserve and disclose a reproducible subset of labeled queries. Index training may use corpus vectors but not test relevance labels. Fix seeds where available.

The [FAISS index reference](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) documents these index families and shows that `IndexLSH` is exhaustive over binary codes. A shared library reduces implementation differences but does not eliminate them: index methods have different training requirements, data structures, optimized code paths, and score representations. Results describe the tested implementations and hardware, not universal algorithm performance.

Theoretical discussion should cover exact scan's O(nd) distance work per query, plus top-k selection cost; space for stored vectors is O(nd). For HNSW, IVF, PQ, and LSH, explain parameters, construction/training costs, and assumptions behind any cited bounds. Avoid promising a universal logarithmic or sublinear bound for real high-dimensional datasets. BK-trees are better treated as a possible extension for metric spaces such as edit distance; including them in the core dense-vector test would require a compatible metric and a defensible implementation.

## What to measure

### Retrieval quality

- **ANN recall@k:** for each query, the number of returned IDs also in the exact top-k divided by k; average over queries. This quantifies approximation to the chosen vector objective. Handle ties deterministically or report the tie policy.
- **Evidence recall@k:** for each query, the fraction of judged relevant documents retrieved in the top k; average across queries. Report the number of queries with relevant judgments. Because judgments may be incomplete, unjudged documents are not necessarily irrelevant.
- **NDCG@k:** use the dataset's graded relevance when available; apply its documented relevance convention. It accounts for ranking, unlike set overlap. State how unjudged documents are handled.
- Use k=10 for the primary search comparison. If the downstream prompt uses a different k, report it separately and run retrieval metrics at that k too.

### Efficiency

- **Search latency:** median and 95th percentile per-query latency, using the same CPU, thread count, batch size, warm-up procedure, and query order. Start with single-query, single-thread CPU search; record hardware, library versions, and timing boundaries. Repeat timed runs and report variability. Query embedding and generation are excluded and measured separately if needed.
- **Index build:** time training plus insertion, reporting each separately and their total. Repeat builds where stochastic construction matters.
- **Memory:** report serialized index bytes and, if practical, process resident-memory increase or peak during construction. Describe measurement method and distinguish stored index, peak build memory, and any extra raw vectors retained for reranking; serialized size is not automatically resident memory.
- **Generation:** if conducted, compare answer correctness against verified references and passage support with a documented rubric or blinded manual review. Report input/output tokens and generation time separately. A correct answer can be unsupported, and a supported answer can still be incorrect.

Report averages alongside per-query paired differences and uncertainty, such as bootstrap confidence intervals over queries. Repeated timings reflect runtime variability; repeated index builds capture construction variability. Avoid claiming statistical significance from a handful of examples.

## Fair comparison and interpretation

1. Encode the same corpus and queries once per dataset; record preprocessing, truncation, IDs, and vector normalization.
2. Build all indexes from exactly the same corpus vectors. Keep k and evaluation queries identical within a dataset and corpus size.
3. Sweep each method's own relevant parameters. Compare **quality–latency and quality–memory curves**, including a comparison near a shared target such as 95% ANN recall@10 if attainable; also show the actual achieved recall. Equal parameter values across different algorithms are not meaningful.
4. If reranking is used, apply and report the same reranking protocol and candidate budget, including its time and raw-vector memory. Keep the main comparison without reranking unless it is explicitly an experimental condition.
5. Explain where exact-search agreement and relevance diverge. An approximate result can sometimes have higher judged relevance than exact vector search, because the embedding model's exact ranking is not a perfect relevance oracle.
6. For downstream testing, choose representative settings on development data, then freeze them. Run the same question set through all selected indexes and the same generator. Note that identical retrieved passages can still yield different outputs if generation is nondeterministic.

Plot ANN recall against median and p95 latency, and index size against recall. Include build time and relevance metrics in tables. Show results by dataset and corpus size rather than averaging away differences. State that speed, quality, and memory tradeoffs depend on implementation, data, and hardware.

## How the supplied resources connect

- [Maximizing RAG efficiency](https://www.cambridge.org/core/journals/natural-language-processing/article/maximizing-rag-efficiency-a-comparative-analysis-of-rag-methods/D7B259BCD35586E04358DF06006E0A85) motivates the broader efficiency tradeoff. It varies RAG methods, vector stores, models, and filters; this study narrows the question to search indexes under controlled conditions. Its 23,625-run grid is not a required project scale.
- [Fast LSH frameworks](https://arxiv.org/abs/1708.07586) can support the theory section; match its assumptions to the specific LSH implementation before applying its complexity bounds.
- The IEEE link is [Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202), relevant to IVF-PQ and compression. The IEEE page blocked direct access during planning; consult the paper itself for its precise claims.
- [Semantic chunking](https://link.springer.com/article/10.1007/s10791-025-09638-7) and the [RAG survey](https://arxiv.org/abs/2410.12837) provide pipeline context and explain why preprocessing is controlled.
- [RGB](https://arxiv.org/abs/2309.01431), [RAGTruth](https://arxiv.org/abs/2401.00396), [WiCE](https://arxiv.org/abs/2303.01432), and [long-form QA attribution](https://arxiv.org/abs/2310.12150) inform optional downstream evaluation, not the definition of ANN recall.
- [Needle in a Haystack](https://github.com/gkamradt/needle-in-a-haystack) and [LongGenBench](https://arxiv.org/abs/2410.04199) mainly test long-context model behavior and are lower priority for index comparison. Duplicate PDF/abstract links in the resource list refer to the same papers.

## Milestones and deliverables

1. **Pilot:** download SciFact, verify IDs and judgments, encode documents and queries, and validate exact top-k and metrics on a few manually inspected examples.
2. **Core comparison:** add HNSW and IVF-Flat, establish timing and memory procedures, and run parameter sweeps on SciFact and FiQA.
3. **Breadth:** add IVF-PQ if the training sample and available resources are sufficient; add binary-code LSH if time allows. Run the predetermined scaling experiment.
4. **Optional RAG check:** select settings on development data and run a small verified QA study with the fixed generator. Report it as a limited downstream study.
5. **Paper:** explain algorithms and assumptions, document experimental controls, show tradeoff plots and limitations, and connect findings to the research question. Provide configuration files, seeds, dataset versions, environment details, and scripts so results can be reproduced.

The minimum complete project is exact scan, HNSW, and IVF-Flat on SciFact and FiQA, with both retrieval quality and efficiency. IVF-PQ, LSH, a third dataset, and generated-answer evaluation add breadth as resources permit.
