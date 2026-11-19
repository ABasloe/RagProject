# 3B generator output-budget review

The completed 96-token comparison was inspected before changing the budget. Its 72 responses cover 24 development claims under three conditions. Original results remain in `results/generation-comparison-3b/`.

| Condition | Valid responses | Invalid responses at token cap | Other invalid responses |
|---|---:|---:|---:|
| No context | 24/24 | 0 | 0 |
| Exact-search context | 7/24 | 16 | 1 missing explanation |
| Gold evidence | 22/24 | 0 | 2 with numeric instead of string citations |

Seventeen exact-context outputs reached 96 tokens, but one already contained valid, complete JSON. Thus reaching the token cap is not synonymous with an invalid response. Many of the other 16 outputs stopped mid-citation list or mid-string. The remaining schema errors and citations prefixed with `Document ` will not automatically be fixed by a longer output allowance. The strict parser and exact document-ID checks are preserved for this comparison.

## Controlled rerun

Rerun all 72 responses, not just failures, with a 256-token maximum. Preserve the exact saved prompts, model revision, float32 CPU execution, greedy decoding, conditions, and scoring rules. Token and latency measurements will change as outputs are allowed to finish. This remains a development-only comparison, not a final evaluation.

```bash
.venv/bin/python generation_comparison.py \
  --max-new-tokens 256 \
  --output results/generation-comparison-3b-256 \
  --background
```

Monitor `results/generation-comparison-3b-256/run.log` and `status.json`. Rerun the same command after interruption; completed checkpoints are reused. The manifest includes the token budget and rejects incompatible resumes. The model cache is `/tmp/ragproject-model-cache`; it must still exist after a reboot.

New outputs include `max_new_tokens`, `finish_reason` (`eos`, `length`, or `other`), and `reached_token_budget`. `hit_output_limit` now indicates length stopping without a terminal EOS token. An EOS at the final allowed position is recorded as natural stopping. Historical limit flags used output length alone and are not rewritten.

Compare format validity, output-limit hits, accuracy/macro-F1, citation errors, and generation latency after completion. A higher budget may reduce truncation without improving scientific reasoning or citation fidelity. Any later prompt/schema changes should use a separate development run.
