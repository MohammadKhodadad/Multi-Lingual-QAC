# QAC Quality Notes

Living notes for reviewing the generated Question-Answer-Context output over time.

## Current Snapshot

- Source file reviewed: `data/google_patents/qac/qac.csv`
- Current sample size: 225 rows
- Unique source documents: 15
- Languages present: `en`, `de`, `fr`, `es`, `ja`, `ko`, `zh`, `ru`, `pt`, `it`, `nl`, `ar`, `tr`, `pl`, `hi`
- Current design: English-first generation, then translation to the other languages
- Current validation: English language check + faithfulness check + retrieval-quality check before translation
- Current corpus gate: preprocessing now skips documents whose cleaned abstract is shorter than `50` words, so title-only and ultra-short records are excluded before Q&A generation

## What Looks Good

- The `en` rows are genuinely English in the current sample.
- The answers appear grounded in the source patent abstract or context.
- The English answers are generally concise and readable.
- The structure is consistent across languages: one approved English QA gets translated into the target languages.
- The current validators seem to remove obvious language failures and some unsupported outputs.
- The latest prompt revision produces better questions when it focuses on method, ingredients, component role, technical purpose, or process effect.
- The new preprocessing gate appears to have fixed the previous title-only failure mode in the reviewed sample.
- Several of the latest questions are more procedural and discriminative than earlier runs, especially around production steps, mixing logic, sealing, and operating constraints.

## What Still Looks Weak

- Question style is still somewhat repetitive. Several current questions still use broad patterns like:
  - `What is the advantage ...?`
  - `What is the main technical advantage ...?`
  - `What is the purpose ...?`
- Some questions are faithful but still too broad for strong retrieval benchmarking. They summarize a benefit or use case instead of isolating one narrower technical fact.
- Some translated outputs are accurate but feel literal rather than natural.
- Technical-name normalization is better than before, but should still be monitored on chemistry-heavy examples.
- The current sample is still modest, so quality judgments are still preliminary.
- We now validate language, faithfulness, and retrieval usefulness, but the prompt and checker still need tuning to push questions away from `advantage` / `purpose` fallback wording.

## Example Strengths

### Good: grounded and understandable

- `CH-720331-B1_it`
  - Q: `How does the system ensure the separate storage and mixing of the whitening gel components before application to discolored teeth?`
  - A: `The system uses two syringes, one containing a liquid and the other a powder, which are kept separate until they are mixed via a connector just before use.`
  - Why it is good: The question targets a concrete mechanism and the answer is specific, procedural, and clearly grounded.

- `US-2025327009-A1_en`
  - Q: `How does the double-sided structure of the cell production devices contribute to safety during robotic operation?`
  - A: `The double-sided structure separates the dangerous region, where the robot operates, from the safe region on the opposite side, enhancing operator safety.`
  - Why it is good: The question is technical, natural enough, and focused on a specific safety mechanism rather than a generic invention summary.

- `TR-2021006663-A2_tr`
  - Q: `Why is cold rolling performed after hot rolling or forging in the production of this steel for valves?`
  - A: `Cold rolling is used to achieve the desired dimensions and surface quality after the steel has been hot rolled or forged.`
  - Why it is good: The question asks about process rationale instead of just restating a material or product name, and the answer is narrow and factual.

## Example Weaknesses

### Weak: broad benefit / purpose framing

- `CN-117945379-B_zh`
  - Q: `What is the main technical advantage of using an electrochemical sodium insertion step in water for preparing Na3Ti2(PO4)3 from NaTi2(PO4)3?`
  - Why it is weak: It is grounded, but still asks for a broad technical advantage. A harder retrieval query would target one sharper property such as mild preparation conditions, scalability, or low-risk mass production.

- `US-2025325674-A1_en`
  - Q: `What is the advantage of using the specified DNA oligonucleotide as an autoimmune disease treatment compared to protein-based agents?`
  - Why it is weak: The answer is supported, but the question still uses generic `advantage` framing instead of asking about the more concrete differences: room-temperature storage or lack of biological contamination.

### Weak: technical normalization still worth watching

- Chemical and material names are generally better normalized than in earlier runs, but this should still be monitored in future chemistry-heavy samples where exact English rendering matters.

## Current Quality Summary

- English correctness: much improved
- Faithfulness to source: clearly improved after removing title-only and ultra-short records
- Translation quality: acceptable, sometimes literal
- Retrieval usefulness: clearly better than the earlier runs, with more mechanism/process questions and fewer obvious title-lift failures
- Overall status: best run so far; usable pilot output with better trustworthiness, though still not final-quality due to remaining broad `advantage/purpose` question patterns

## Recommended Next Improvements

1. Tune the question-quality validator so it rejects broad `advantage` / `purpose` prompts without over-filtering.
2. Push the prompt further toward sharper search intent instead of asking for whole-document benefits or general technical advantages.
3. Keep improving prompts to encourage more diverse question types.
4. Continue monitoring English normalization of technical terms.
5. Review a larger sample before trusting the pipeline broadly.
6. Keep periodic human review notes in this file after each generation update.

## Review Log

### Review 1

- Result: Better than the previous run.
- Main improvement: `en` outputs are now genuinely English.
- Main remaining issue: questions are still somewhat generic and repetitive.

### Review 2

- Result: Better than Review 1.
- Main improvement: questions are more retrieval-oriented and less stuck on the old `main feature / main components` pattern.
- Main remaining issue: some questions still sound too document-centered or slightly generic for strong retrieval benchmarking.

### Review 3

- Result: Still usable and generally grounded, with no obvious English-language failures in the current 6-question source set.
- Main improvement: the best questions now target method steps, ingredients, or component roles more directly.
- Main remaining issue: several questions are still broad summary prompts rather than highly discriminative retrieval queries.

### Review 4

- Result: Best run so far.
- Main improvement: title-only and ultra-short records no longer appear in the corpus, which materially improved trustworthiness of the generated QAs.
- Main remaining issue: wording is better, but some questions still fall back to broad `advantage` or `purpose` framing instead of narrower technical facts.
