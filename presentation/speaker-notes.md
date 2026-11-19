# Project overview — speaker notes

Snapshot: November 18, 2026 • 08:47 PM ET

## 1. Search algorithms in retrieval-augmented generation

Present this as a controlled empirical algorithms project. The study follows retrieval speed and quality into a limited scientific-claim generation task. The final held-out generation evaluation is still running at this snapshot; no partial accuracy is presented.

Source: `Project snapshot • November 18, 2026 • 08:47 PM ET`

## 2. The research question

Separate the retrieval question from the downstream question. Better means a measured tradeoff, not an assumption that approximate methods improve answers. End-to-end generation claims must wait for the frozen held-out evaluation.

Source: `docs/project-plan.md; docs/GENERATION_FINAL_PROTOCOL.md`

## 3. Where the search algorithm fits

Corpus documents are embedded once with normalized BGE vectors. The query uses the same encoder. Search produces document IDs, then a common packing policy constructs the prompt. Query encoding, search, packing, and generation are distinct costs. The final held-out runner measures these online.

Source: `benchmark.py; generation_pilot.py; generation_heldout.py`

## 4. Three search methods, one comparison

Exact inner-product search is the reference for ANN recall, not a perfect relevance oracle. HNSW searches a graph. IVF-Flat clusters vectors and searches selected lists while retaining full-precision vectors. Graph and cluster construction add cost. This project does not implement LSH, BK-trees, or product quantization.

Source: `benchmark.py; FAISS index documentation`

## 5. The project progressed in four stages

First validate the small dataset, then expand to a larger domain. Next assess uncertainty and scale within a single dataset. Finally investigate generated claim verification, including failures. Caching and resumable background workers allowed longer experiments to run reliably.

Source: `Repository experiment artifacts and protocol documents`

## 6. A controlled experimental design

SciFact uses a reproducible subdivision of the BEIR test queries: 60 for development and 240 held out. FiQA uses its official 500-query development and 648-query test splits. All methods share cached BGE-small-en-v1.5 embeddings and a pinned revision. Settings are selected on development performance and then fixed for test evaluation.

Source: `SciFact/FiQA results metadata; docs/FIQA_RUN.md`

## 7. What each metric tells us

ANN recall measures overlap with exact neighbors. Evidence recall is recall against positive dataset relevance judgments, which are not proof of answer support. NDCG accounts for rank. Serialized size is not peak resident memory. Generation correctness and citation support require additional evaluation.

Source: `benchmark.py; results/analysis/report.md`

## 8. Initial retrieval results: fast search, small quality losses

These are the original held-out/test measurements, not averages from the later scaling study. HNSW settings are 32 for SciFact and 64 for FiQA; IVF nprobe is 16 for both. HNSW is about 10x and 50x faster than exact search respectively. Absolute latency savings are far smaller than these ratios suggest.

Source: `results/scifact-heldout/results.json; results/fiqa-run/test/results.json`

## 9. We tested stability, not just one favorable run

Paired percentile bootstrap intervals use 5,000 resamples of queries from the original test runs. They condition on the built index. Five separate construction seeds and timing orders assess build/run variation separately. The HNSW evidence differences include zero; this is not proof of equivalence. IVF intervals here are below zero and describe a small observed loss.

Source: `results/analysis/confidence-intervals.json; results/analysis/report.md`

## 10. Controlled scaling isolates corpus growth

Use nested subsets at 5,000, 20,000 and all 57,638 FiQA documents. Keep 500 development and 648 test queries fixed. Retain all 2,944 documents judged relevant to either split and add seeded distractors. Test judgments determine membership only; search tuning uses development queries. Smaller subsets are relevance-enriched; one subset sequence cannot estimate sampling variability.

Source: `docs/SCALING_RUN.md; results/fiqa-scaling/subsets.json`

## 11. The search advantage grows with corpus size

The y axis is logarithmic. The chart uses means of per-run median latencies; error bars show observed min-max ranges, not confidence intervals. HNSW speedups are approximately 11.4x, 17.8x and 50.7x. Search effort is tuned separately by size, so these are practical configurations near a recall target, not a pure asymptotic-complexity measurement.

Source: `results/fiqa-scaling/summary.csv; means and min–max ranges across five builds`

## 12. Index choice depends on the workload

Illustrative break-even uses mean build times and mean per-run median latencies at full FiQA size: (19.5430-.0429)/((4.5045-.0889)/1000), approximately 4,400 queries. This is a simplified query-work model, not measured total throughput. Shared embeddings, update costs, concurrency and generation are excluded. Serialized storage is distinct from RAM.

Source: `results/fiqa-scaling/report.md; derived illustration using mean build times and latencies`

## 13. Generation connects retrieval to the final response

This is generated scientific claim verification, not open-ended general question answering. Original SciFact labels provide support, refutation and insufficient-evidence references. The pilot uses 24 balanced development claims. Begin with a small 0.5B generator for feasibility, then use pinned Qwen2.5-3B-Instruct. No-context is an evidence-only abstention baseline; gold evidence is privileged and can still be truncated.

Source: `docs/GENERATION_PILOT.md; docs/GENERATION_ALGORITHM_COMPARISON.md`

## 14. A pilot exposed an output-budget problem

The three-condition 3B experiment replayed identical prompts at 96 and 256 output tokens. Seventeen exact-context outputs reached 96 tokens, but one was already valid; 16 invalid outputs were recovered by increasing the cap. No 256-token response hit the cap. The parser and prompts were kept unchanged to isolate the budget effect. Higher accuracy does not imply improved model reasoning: many previously incomplete outputs simply became scoreable.

Source: `results/generation-comparison-3b/summary.json; generation-comparison-3b-256/summary.json`

## 15. Development generation results: same verdicts

Strict accuracy is shown. All three retrieval methods produce identical verdicts on every development claim. HNSW packed prompts match exact in 21/24 cases, IVF in 20/24. All outputs stop naturally under the 256-token cap. Median timings are generation-only saved-prompt replays; roughly one-second differences are not evidence of algorithmic speedup. Normalized citation formatting retains strict scores separately.

Source: `results/generation-algorithms-3b-256/results.json; 24 claims per condition`

## 16. Failure analysis revealed a context bottleneck

The review covered incorrect development responses and all gold-evidence responses. It is explicitly an unblinded assistant audit, not completed human faithfulness scoring. Under the 256-token-per-document policy, only 7/16 evidence-bearing gold contexts retained one complete annotated rationale set by conservative exact-sentence matching. Partial text can still be informative. Distinguish this generator-context issue from embedding truncation, measured at 8.8% SciFact and 4.2% FiQA documents over the BGE limit.

Source: `docs/GENERATION_FINAL_REVIEW.md; unblinded assistant audit`

## 17. Final evaluation: frozen and still running

Status is a snapshot, not a live indicator. Do not present partial held-out accuracy or retune from partial results. The final protocol evaluates all 240 reserved claims with their natural class distribution. It measures online query embedding, search, context packing, generation and direct end-to-end timing. The held-out partition was already used for retrieval evaluation but not for generation selection. Human faithfulness review remains unfinished.

Source: `docs/GENERATION_FINAL_PROTOCOL.md; results/generation-heldout-3b-256/status.json`

## 18. What the project can—and cannot—claim

This is a focused controlled course project, not a universal RAG leaderboard. Scope is one embedding model, one final generator, one machine and scientific claim verification for generation. Complete the frozen run, inspect errors and conduct a documented human review. Peak resident-memory measurement and broad generalization are not established. The project already has sufficient breadth; prioritize interpretation and reproducibility over adding unrelated algorithms.

Source: `Completed retrieval/scaling results; development generation study; frozen final protocol`

## 19. Sources and reproducible artifacts

All numeric findings come from this project’s saved artifacts. Paths are relative to the repository root. The PowerPoint includes these speaker notes, and the companion Markdown file contains the full note text. Charts are regenerated from numeric result files by build_presentation.py. The status snapshot is saved separately.

Source: `Local repository artifacts; primary model/dataset documentation`