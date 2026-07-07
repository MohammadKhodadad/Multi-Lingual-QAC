# QA-generation + verification model comparison — cost & token usage

Same 30 documents, each with a fixed (mode, strategy -> query-language) shared across all generators. Identical generation + verifier prompts from the `qac_chempatents` pipeline. Each generation produces 3 Q&A pairs, verified by faithfulness + quality graders; the best pair (faith+quality) is kept.

Verifier = `anthropic/claude-sonnet-4.6` by default; `gpt-5.5` when the generator is sonnet-4.6 (no self-grading). Generators use medium reasoning; verifiers use low.

| Generator | verifier | ok | avg gen tok (in/out) | avg verify tok (in/out) | gen $/batch | verify $/batch | total $/batch | est. 1000 (keep best 1) | est. 1000 (use all 3) |
|---|---|---|---|---|---|---|---|---|---|
| gpt-5-mini | `anthropic/claude-sonnet-4.6` | 30/30 | 2748 / 2306 | 5409 / 538 | $0.00266 | $0.02430 | $0.02696 | $26.96 | $8.99 |
| gpt-5.4-mini | `anthropic/claude-sonnet-4.6` | 30/30 | 2748 / 1398 | 5324 / 532 | $0.00835 | $0.02395 | $0.03231 | $32.31 | $10.77 |
| sonnet-4.6 | `gpt-5.5` | 30/30 | 3501 / 503 | 4685 / 1136 | $0.01805 | $0.05620 | $0.07425 | $74.25 | $24.75 |
| grok-4.3 | `anthropic/claude-sonnet-4.6` | 30/30 | 2810 / 2558 | 5405 / 550 | $0.00978 | $0.02446 | $0.03424 | $34.23 | $11.41 |
| gemini-3.5-flash | `anthropic/claude-sonnet-4.6` | 30/30 | 2730 / 2900 | 5381 / 544 | $0.03019 | $0.02430 | $0.05449 | $54.49 | $18.16 |
| qwen3.6-35b-a3b | `anthropic/claude-sonnet-4.6` | 30/30 | 2777 / 4542 | 5385 / 527 | $0.00527 | $0.02407 | $0.02933 | $29.33 | $9.78 |

- A **batch** = 1 generation call + 2 verifier calls (faithfulness + quality), producing 3 graded queries.
- **est. 1000 (keep best 1)** = total $/batch × 1000 — 1 kept query per batch (matches qac_chempatents_best: generate 3, keep the best).
- **est. 1000 (use all 3)** = total $/batch × 1000/3 — all 3 queries kept (matches qac_chempatents, all candidates).
- Generation cost: OpenRouter is provider-measured; gpt-* computed from list price. Verification cost is attributed to the generator it grades.
- The `cost_summary.csv` also has `est_*_gen_only` / `est_*_verify_only` splits.
