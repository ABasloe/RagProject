# Development pilot review and larger-model comparison

Reviewed 2026-11-17. Source: `results/generation-pilot/results.json` (120 unique claim/condition pairs, 24 claims, eight per class). No held-out generation results were inspected.

## Baseline findings

| Condition | Correct / 24 | Schema-valid / 24 | Output-limit hits | Annotated document in packed context |
|---|---:|---:|---:|---:|
| No context | 2 | 5 | 0 | Not applicable |
| Flat | 7 | 22 | 2 | 13 / 16 evidence-bearing claims |
| HNSW | 7 | 22 | 2 | 13 / 16 |
| IVF-Flat | 7 | 22 | 2 | 12 / 16 |
| Gold evidence | 9 | 18 | 0 | 16 / 16 |

The existing `valid_json_rate` metric actually requires both parseable JSON and a valid response schema, including a nonempty explanation. All 19 invalid no-context responses and all six invalid gold-evidence responses are parseable JSON with empty explanations. They are schema failures, not JSON syntax failures. Each retrieval condition has two malformed responses, both at the 96-token cap. Preserve the original scoring rule; do not retroactively turn these failures into correct abstentions.

All 22 valid responses in each retrieval condition predict REFUTED, and none of the valid responses across any condition supply citations. Several explanations explicitly describe insufficient evidence while the verdict is REFUTED. For example, flat claim 42 says the passage does not directly address the claim but returns REFUTED. Gold-evidence outputs comprise 17 REFUTED, one INSUFFICIENT_EVIDENCE, and six invalid outputs. No valid response predicts SUPPORTED.

Document presence measures above do not establish that annotated rationale sentences survived the per-document truncation. For claim 674 under flat retrieval, annotated document 2095573 was retrieved at rank 10 but excluded from the packed context. Thus model behavior, output length, and context retention all deserve attention. Human faithfulness fields remain blank; this inspection is not a completed blinded human evaluation.

Retrieval-condition median end-to-end times are 14.39–14.57 seconds, with generation accounting for almost all of that latency. One observation per claim/condition is insufficient for a precise timing comparison.

## Next experiment

Use Qwen/Qwen2.5-3B-Instruct, pinned to `aa8e72537993ba99e69dfaafa59ed015b17504d1`, as a larger candidate in the same model family. Model card: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct . Improved performance on this task is a hypothesis, not an assumption.

`generation_comparison.py` replays the exact saved prompts for all 24 development claims under no context, flat retrieval, and gold evidence (72 outputs). It checks that the claim IDs belong to the original development partition, that pairs are complete and unique, and that token counts match the baseline. Keep greedy decoding, float32, one CPU thread, 96 output tokens, and the original parser. This isolates the generator change before prompt or context tuning. ANN conditions are omitted because this step diagnoses generator feasibility. Timing covers generation only; it is not a new online end-to-end RAG benchmark.

The manifest records model revision, source-result and implementation hashes, selected IDs, decoding settings, and library versions. Each output is checkpointed atomically. Completed outputs are reused, and a lock prevents duplicate workers. A changed manifest requires a new output directory. Baseline artifacts and the original runner are unchanged.

```bash
.venv/bin/python generation_comparison.py --background
cat results/generation-comparison-3b/status.json
tail -f results/generation-comparison-3b/run.log
```

The weights are cached in `/tmp/ragproject-model-cache` because the project filesystem had only 1.9 GB free. Outputs remain in the project. Temporary weights may disappear after reboot or cleanup; re-download the pinned revision to that cache before resuming if needed:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct',
    revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
    cache_dir='/tmp/ragproject-model-cache',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'README.md'],
    max_workers=2)
PY
```

On completion, inspect `summary.csv`, `results.json` (including comparable baseline summaries), and the separate `manual-review.csv`. Assess class collapse, schema failures, output caps, citation presence/validity, and correctness with gold evidence before deciding whether to revise the prompt or context policy. A larger model passing this diagnostic does not by itself establish sufficient quality for the final experiment. Freeze development choices before implementing or launching held-out generation.
