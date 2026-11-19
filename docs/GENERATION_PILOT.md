# SciFact generation feasibility pilot

This is a development-only, three-class evidence-grounded claim-verification experiment, not a held-out answer-quality result. The generator is Qwen/Qwen2.5-0.5B-Instruct, a small CPU feasibility baseline. BGE remains the embedding model. Both revisions are pinned; the generator revision is saved in `data/generator-revision.txt`.

## Start and resume

```bash
.venv/bin/python generation_pilot.py --background
tail -f results/generation-pilot/run.log
cat results/generation-pilot/status.json
```

Rerun the launch command to resume. Each claim/condition is saved atomically. Completed outputs are reused; an interrupted generation restarts only that output. A lock prevents duplicate runs. The detached process survives ordinary terminal disconnection, but the host must stay running. The manifest rejects changes to code, inputs, model revision, prompt, or recorded library versions; use a new output directory for a changed protocol.

## Data and isolation

Original annotations are from https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz (schema: https://github.com/allenai/scifact/blob/master/doc/data.md). `data/scifact-original/claims_dev.jsonl` is mapped by claim ID and exact claim text to the existing BEIR data. Original SciFact's public development split corresponds to the BEIR queries used here; our earlier 60/240 development/held-out subdivision is retained.

Select 24 claims (eight per class) reproducibly from the 60-query development partition using seed 2026. No held-out claims are generated or inspected. Gold labels come from `evidence`: SUPPORT maps to SUPPORTED; CONTRADICT maps to REFUTED; no annotated evidence maps to INSUFFICIENT_EVIDENCE. Mixed labels fail explicitly. A cited source document alone is not proof of supporting or refuting evidence. Balanced pilot scores do not estimate accuracy under the natural class distribution.

## Five conditions

- `no_context`: no passages and the same evidence-only instruction. This measures abstention/instruction following, not closed-book world knowledge; a compliant model should choose insufficient evidence.
- `flat`, `hnsw`, `ivf_flat`: actual online query embedding and search over the same SciFact corpus. Settings come only from the original development metrics. HNSW uses M=16/efConstruction=200 and construction seed 42; IVF uses nlist=64 and training seed 42.
- `gold_evidence`: annotated evidence documents, with privileged label-based document selection. This is a diagnostic comparator, not a deployable method or guaranteed quality upper bound. For insufficient-evidence claims, this condition has no documents.

Use one fixed prompt, greedy decoding, maximum 96 output tokens, and single-thread CPU inference in float32. The generator must output JSON with verdict, one-sentence explanation, and document-ID citations. Invalid outputs are scored as incorrect and reported, not silently converted into abstentions or retried. Raw outputs are retained.

Retrieve top 10 documents. Pack them in retrieval order with at most 256 generator tokens per document and a 1,536-token passage budget including headers. A document that does not fit stops packing. Some retrieved documents will be excluded. The full chat prompt must fit 2,048 tokens; otherwise the run fails. Record retrieved IDs, included IDs, truncated IDs, prompt text, and token counts. Gold-document presence does not guarantee that its rationale survived this truncation policy.

## Measurements

Each output records query embedding, search, context packing/tokenization, generation, and directly measured end-to-end latency. End-to-end measurement begins before query encoding and ends after output decoding; it excludes startup, index construction, warm-up, scoring, and disk checkpoint writes. No synthetic addition of prior timing results is used. Models and indexes remain warm. Condition order is deterministically shuffled per claim, and identical prompts are generated again for real timing measurements. One timing observation per claim/condition is a pilot limitation; final timing comparisons need repetition.

`summary.csv` includes accuracy, macro-F1 over three labels, JSON-format success, insufficient-evidence rate, median/p95 end-to-end latency, generation latency, and input/output token counts. `results.json` includes all details. Citation fields check whether IDs exist in supplied context and overlap annotated evidence documents; these are proxies, not checks that an explanation is entailed.

`manual-review.csv` includes responses and supplied passages with shuffled IDs and blank human-review columns. Method names and gold verdicts are withheld from that sheet; `review-key.json` maps IDs back to conditions for analysis after review. Do not treat empty human-review fields as automatic faithfulness scores.

## Before a final experiment

Review format failures, output-limit hits, runtime, evidence retention, and performance with gold evidence. If the small generator performs poorly even with evidence, evaluate a stronger generator on development examples before freezing a final protocol. The held-out generation experiment should be implemented and launched only after the development prompt/model/context settings are frozen. Generation inference is scientifically distinct from the previous retrieval benchmark; large retrieval speedups need not imply large end-to-end speedups.

The cached generator model card is https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct. Data, models, and outputs are Git-ignored. Preserve the manifest, annotations, raw outputs, and model revision when backing up the experiment.

## Completed pilot review

See [GENERATION_PILOT_REVIEW.md](GENERATION_PILOT_REVIEW.md) for the failure audit and the resumable 3B development comparison.
