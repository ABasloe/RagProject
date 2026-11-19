# Final development review — 2026-11-18

Source: `results/generation-algorithms-3b-256/results.json`, 24 balanced development claims, 120 outputs. Reviewed all 56 normalized-incorrect responses (including 16 expected no-context abstentions) and all 24 gold-evidence responses: 70 distinct rows in `generation-development-audit.csv`. This is an unblinded assistant audit, not completed human faithfulness scoring. Existing human review fields remain untouched.

| Condition | Strict correct | Normalized correct | Strict schema valid | Normalized schema valid |
|---|---:|---:|---:|---:|
| No context | 8/24 | 8/24 | 24/24 | 24/24 |
| Flat | 14/24 | 14/24 | 23/24 | 23/24 |
| HNSW | 14/24 | 14/24 | 23/24 | 23/24 |
| IVF-Flat | 14/24 | 14/24 | 23/24 | 23/24 |
| Gold evidence | 14/24 | 14/24 | 22/24 | 24/24 |

No response hit the output cap. Citation normalization removes all outside-context ID flags, but does not improve verdict accuracy. Gold numeric citations on claims 533 and 1339 become schema-valid without making their verdicts correct. All retrieval methods fail on the same ten claims: 42, 212, 338, 533, 742, 911, 1144, 1221, 1339, 1359. Claim 338 lacks an explanation in each retrieval condition. Similar scores do not demonstrate equivalence.

Gold failures separate several mechanisms. Claims 42 and 212 reverse the direction of visible protective/age-reducing evidence. Claim 1221 overlooks visible mutation differences. Claims 742, 338, 1144, 1359, 533, and 1339 lose relevant results to the 256-token document limit. Claim 911 has partial evidence and spelling/isoform ambiguity. Claims 1144 and 1359 also expose interpretation issues with modeling versus observed intervention effects and combination-therapy terminology. The row-level audit retains raw responses and passages for checking these judgments.

A whitespace-normalized exact-sentence audit of the original development annotations finds at least one fully retained annotated rationale set for only 7/16 evidence-bearing gold contexts. This conservative diagnostic is not semantic entailment: partial rationale text and titles can still be informative (notably claim 42). Correct-label claims 163 and 1259 lack complete annotated rationales and their explanations infer outcomes from aims/mechanisms. Claim 756 does not explicitly substantiate the breadth of the claim. Correct labels therefore cannot be treated as supported explanations.

Decision: freeze the tested 3B/256-token configuration as a limited baseline, with no further development tuning. Increasing context or changing the prompt could help, but would create an untested configuration. Evaluate generalization of this configuration on all 240 reserved claims; do not claim adequate scientific reasoning, a gold upper bound, or a held-out estimate from these 24 balanced examples.
