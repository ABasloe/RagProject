FiQA completed successfully, including automatic parameter selection and evaluation on all 648 test queries. Development and test queries do not overlap.
Method	ANN recall@10	Evidence recall@10	Median search time
Exact	100%	46.39%	4.314 ms
HNSW (efSearch=64)	96.28%	45.68%	0.086 ms
IVF-Flat (nprobe=16)	95.88%	45.38%	0.465 ms


HNSW achieved 50× faster search, with evidence recall falling by 0.71 percentage points versus exact search. IVF-Flat was 9.3× faster. Both exceeded the 95% ANN-recall target.
Results: [FiQA test summary](/home/apbasloe/analysis/RagProject/results/fiqa-run/test/summary.csv).
You now have initial results for both core datasets. Next: create comparison plots and add uncertainty estimates and repeated-build measurements before drawing final conclusions. These timings currently represent one index build per method.


2:07 PM







is there any other benchmarks/metrics I should run? Can you go over what so far has been done to what for what







2:08 PM






You’ve completed the initial retrieval benchmark on both core datasets. You’re measuring how the search algorithm affects speed, storage, and retrieval quality while keeping the embedding model and documents fixed.
You have not yet tested generated answers—these experiments evaluate the retrieval part of RAG.
What you’ve run and why
Component	What you used	Purpose
Embeddings	BGE-small-en-v1.5, fixed revision	Convert documents and queries into comparable vectors
SciFact	5,183 documents; 60 development and 240 held-out queries	Validate the pipeline on a small scientific corpus
FiQA	57,638 documents; 500 development and 648 test queries	Test another domain and a substantially larger corpus
Exact search	FAISS Flat	Establish the exact vector-search reference
HNSW	Graph-based approximate search	Test faster retrieval with additional index storage/build cost
IVF-Flat	Cluster-based approximate search	Test the tradeoff between searching fewer clusters and missing neighbors


For each dataset, you selected settings on development queries, then evaluated fixed settings on separate test queries. FiQA also ran unattended with embedding checkpoints and automatic development-to-test progression.
What your metrics tell you
Metric already recorded	What it answers
ANN recall@10	How closely does approximate search reproduce exact search’s top 10?
Evidence recall@10	How many of the judged relevant documents are retrieved?
NDCG@10	Are relevant documents ranked near the top?
Median and p95 search latency	How fast is a typical search, and how slow is the upper tail?
Training, insertion, and total build time	How expensive is creating the index?
Serialized index bytes	How much storage does the saved index require?
Per-query timing repeats and results	Which queries behave differently, and how variable are timings?


ANN recall is not answer accuracy. A method can closely reproduce exact search while both methods miss relevant documents. Your FiQA results illustrate this: approximate search achieved over 95% ANN recall, but evidence recall was around 45–46%.
What you’ve learned so far
Dataset	HNSW speedup over exact	HNSW ANN recall	Evidence-recall decrease
SciFact	About 10×	97.21%	0.42 percentage points
FiQA	About 50×	96.28%	0.71 percentage points


These are promising pilot results. They describe the tested configurations and machine; they aren’t yet evidence that HNSW will always perform this way.
What I’d run next, in priority order
1. Confidence intervals and paired query comparisons.
   Bootstrap the saved per-query results to estimate uncertainty around recall and NDCG, especially their differences from exact search. This can use existing results without re-embedding.
2. Repeated index builds and independent timing runs.
   You currently have three timing repeats per query within each configuration, but one measured build per method per stage. Run several independent builds, record construction seeds where supported, and vary timing order to assess stability.
3. A truncation audit.
   Measure how many documents exceed BGE’s input limit and how much text is omitted. This helps explain retrieval quality. Changing chunking afterward should be a separate experiment.
4. Controlled scaling within FiQA.
   Compare nested corpus sizes—such as 5,000, 20,000, and all 57,638 documents—with the same queries and their relevant documents retained. Comparing SciFact with FiQA changes both domain and size, so it does not isolate scaling.
5. Peak memory usage.
   You already measure serialized index size. Add peak process memory during construction and memory during search, ideally with each method measured in a separate process.
After those, useful extensions are:
Extension	Why add it?	Priority
IVF-PQ	Study compression and the memory–quality tradeoff	Strong extension
Recall/NDCG at k=5 and k=20	Check sensitivity to the number of retrieved documents	Useful
NFCorpus	Add a third retrieval domain	Optional
Generated-answer evaluation	Test whether retrieval differences change supported answer quality	Separate downstream study


My recommendation is to strengthen the two existing benchmarks before adding more datasets: create tradeoff plots, calculate confidence intervals, audit truncation, and repeat builds. That turns the initial runs into a much more defensible comparison.