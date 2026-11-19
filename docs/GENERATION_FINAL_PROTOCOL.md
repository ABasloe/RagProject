# Frozen held-out generation protocol

Freeze date: 2026-11-18, after the development review and before held-out generation. The machine-readable `generation-final-protocol.json` binds this document, review, implementation, development artifacts, partition files, data, revision files and embedding cache by SHA-256. The runner rejects changed inputs. Previous retrieval evaluation already used this held-out partition; it is held out for generation selection, not a new untouched dataset.

Evaluate all 240 unique reserved SciFact claim IDs, disjoint from the original 60 development IDs. No class balancing or subset selection. Shuffle claim order with seed 2026; shuffle five conditions per claim using `2026:<claim ID>`. Keep no context, Flat, HNSW, IVF-Flat, and privileged gold evidence. Gold documents use annotation IDs only in that diagnostic condition; no annotated sentences or labels enter retrieval prompts.

Use Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, float32 CPU, one thread, greedy decoding, seed 2026, maximum 256 new tokens. Keep the exact system prompt in the runner (identical to the pilot). Use pinned BGE embeddings, normalized inner product, top 10, development-selected HNSW efSearch=32 and IVF nprobe=16; HNSW M=16/efConstruction=200, IVF nlist=64, construction seed 42. Pack in retrieval order, at most 256 generator tokens per document, 1536 passage tokens including headers, stop at the first non-fitting document, maximum 2048 input tokens. No response repair, retry, or post-result tuning.

Primary metrics: strict verdict accuracy and three-class macro-F1, with invalid schemas counted incorrect. Report secondary normalized verdict/schema and citation metrics using existing v1 normalization: trim strings, remove a complete Document prefix from digit IDs, convert nonnegative integer IDs. Preserve strict raw results. Report schema validity, output-cap counts, class distribution/confusion, citation IDs outside supplied context and overlap with annotated documents. These citation proxies do not measure entailment. No-context measures evidence-only abstention; gold evidence is privileged and may be truncated.

Measure online query embedding, search, packing/tokenization, generation, and directly observed end-to-end time; exclude startup, index building, warm-up, scoring, and checkpoint writes. One warm-model timing observation per claim/condition; report median/p95 without asserting precise speedups. Preserve raw prompts, outputs, retrieved/packed/truncated IDs, token counts, stop reasons, and shuffled human-review worksheet. Human faithfulness fields remain unscored until actually reviewed.

The final report uses claim-level paired bootstrap resampling (seed 2026, 2000 draws) for 95% percentile accuracy intervals and HNSW/IVF-minus-Flat accuracy differences. These are descriptive uncertainty estimates for this fixed partition; no significance-based selection. Report all conditions and failures. Operational interruptions may resume exact checkpoints; any scientific protocol change invalidates the frozen run and requires a separately labeled experiment, never retuning on held-out responses.

Run/resume:

```bash
.venv/bin/python generation_heldout.py --background
```

Outputs: `results/generation-heldout-3b-256/`. Monitor `status.json` and `run.log`; final `results.json`, `summary.csv`, `report.json`, and human-review files are written at completion. Approximately 1200 outputs, likely 30–40 hours based on development timings; actual runtime can vary. The host must stay running and the temporary model cache must remain available.
