# Five-condition 3B generation comparison

Run/resume from the project root:

```bash
.venv/bin/python generation_comparison.py \
  --max-new-tokens 256 \
  --conditions no_context flat hnsw ivf_flat gold_evidence \
  --output results/generation-algorithms-3b-256 \
  --background
```

Monitor `results/generation-algorithms-3b-256/run.log` and `status.json`. The job is detached, with one atomic checkpoint per output. Repeat the same command to resume; duplicate workers are rejected. The host must remain running. The 3B model is cached in `/tmp/ragproject-model-cache`, which may not survive reboot.

This reruns all five conditions on the same 24 development claims (120 outputs) with the pinned Qwen2.5-3B-Instruct model, original saved condition-specific prompts, greedy decoding, and a 256-token cap. The exact retrieved passages, packing policy, and development-selected retrieval settings come from the original pilot. Conditions retain their original per-claim randomized order. The final 240-query held-out partition remains untouched. Original 96-token and 256-token three-condition runs are preserved.

## Citation scoring

Raw responses, strict parsing, strict verdict accuracy/F1, and strict citation-ID checks are retained. Separate `normalized_*` fields apply the same policy to every condition:

- Trim whitespace around string citations.
- Convert `Document 123` to `123` when the entire citation matches that pattern.
- Convert nonnegative integer IDs to strings. Reject booleans, floats, nulls, and structured values.
- Require the original valid verdict, nonempty explanation, and citation list. No invented fields, completion of truncated JSON, fuzzy ID matching, or reasoning corrections.

Unknown string IDs remain unknown and are flagged against the supplied context. Citation presence in context and overlap with gold evidence documents are proxies, not entailment/faithfulness judgments. Review explanation support separately using the shuffled manual-review sheet. Normalized scoring is a documented development-time evaluation change; strict scores remain available for comparison.

## Timing and interpretation

This experiment replays saved prompts and measures generation time only. It does not perform online retrieval or claim end-to-end speedups. It produces a fresh `summary.csv`, `results.json`, and human-review worksheet. No-context is an evidence-only abstention baseline; gold evidence is a privileged diagnostic condition. Balanced development results do not establish final held-out accuracy. HNSW/IVF may supply identical packed prompts to exact retrieval, in which case output differences should be checked carefully before attributing effects to the search method.
