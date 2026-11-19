# Controlled FiQA scaling

Launch or resume from the project root:

```bash
.venv/bin/python run_scaling.py --background
```

Monitor with `tail -f results/fiqa-scaling/run.log` and `cat results/fiqa-scaling/status.json`. The detached process survives ordinary terminal disconnects, but not a machine reboot or server policy that kills users' processes. Run the same command after an interruption. A lock prevents duplicate runners.

## Frozen protocol

- Nested corpora of 5,000, 20,000, and 57,638 documents. All 500 official development and 648 official test queries are retained at every size, with no overlap.
- All 2,944 documents judged relevant to any of these queries appear in every corpus. Remaining documents are drawn from one seeded permutation (subset seed 2026). Preserve original document insertion order in each subset. Save all membership IDs to `subsets.json`.
- Reuse identical BGE embeddings. Verify dataset-content keys, cache file hashes, and equality of corpus vectors across the development/test caches. Recompute exact top-10 neighbors for each corpus and split.
- Fix k=10, HNSW M=16/efConstruction=200, and IVF nlist=128 across sizes. nlist=128 is supported by the 5,000-document training set. Vary only search effort: efSearch 16/32/64/128/256/512 and nprobe 1/4/16/32/64/128.
- Select the lowest measured development median latency attaining ANN recall >=0.95 per approximate method and size, using construction seed 42. Freeze all three size selections before any test evaluation. Stop if no configuration qualifies.
- Evaluate frozen settings at each size over five new construction seeds: 101/202/303/404/505. Each seed randomizes corpus-size execution order, method order, and query order. Query order is shared across methods within each seed. Development parameter timing order is also randomized.
- Single-threaded CPU FAISS, three timings per query, warm-up before each setting. Measure median/p95 search latency, ANN/evidence recall@10, NDCG@10, training/insertion time, and serialized index bytes. No embedding or generation time is included.

This is a relevance-preserving scaling study. Test judgments influence only predetermined corpus membership; they never select index parameters. The smaller corpora are enriched with relevant documents and have fewer distractors, so they are not representative random FiQA samples. Full-corpus results remain the primary relevance comparison. One subset seed does not estimate subset-sampling uncertainty. Since search effort is tuned per size, curves describe the tested configurations near a development recall target, not pure asymptotic complexity or universally optimal index configurations.

## Checkpoints and outputs

Each completed method/parameter result is written atomically, with query results and timing repeats. Resuming reuses those files. If interrupted during construction/search, rebuild the incomplete method using the same seed; no embeddings are recomputed. Do not change code, data, packages, or protocol midway. The manifest checks code/data/cache hashes and core software versions and rejects incompatible restarts.

`results/fiqa-scaling/` contains `manifest.json`, `subsets.json`, `selected-settings.json`, per-size/per-split/per-seed JSON measurements, `summary.csv`, `report.md`, and `scaling.png`/`scaling.pdf`. Final graphs show means and observed min–max ranges across five builds, not confidence intervals. Original SciFact, FiQA, and analysis results are unchanged.
