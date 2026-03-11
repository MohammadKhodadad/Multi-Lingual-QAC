# QAC Quality Notes

Living notes for reviewing the generated Question-Answer-Context output over time.

## Current Snapshot

- Source file reviewed: `data/google_patents/qac/qac.csv`
- Current sample size: 75 rows
- Unique source documents: 5
- Languages present: `en`, `de`, `fr`, `es`, `ja`, `ko`, `zh`, `ru`, `pt`, `it`, `nl`, `ar`, `tr`, `pl`, `hi`
- Current design: English-first generation, then translation to the other languages
- Current validation: English language check + faithfulness check before translation

## What Looks Good

- The `en` rows are now actually English. This is a major improvement over the previous version.
- Most answers appear grounded in the source patent abstract or context.
- The English answers are generally concise and readable.
- The structure is consistent across languages: one approved English QA gets translated into the target languages.
- The current validators seem to remove obvious language failures and some unsupported outputs.

## What Still Looks Weak

- Question style is still repetitive. Many questions follow patterns like:
  - `What is the main feature ...?`
  - `What are the main components ...?`
  - `What type of products ...?`
- Some English outputs still preserve non-English technical spellings when cleaner English normalization would be better.
- Some translated outputs are accurate but feel literal rather than natural.
- The current sample is still very small, so quality judgments are preliminary.
- We currently validate language and faithfulness, but not yet question usefulness or retrieval quality.

## Example Strengths

### Good: grounded and understandable

- `WO-2025215522-A1_ko`
  - Q: `What is the main feature of the artificial nail according to the invention?`
  - A: `The artificial nail includes a shape deformation layer that fills the gap between the bottom of the nail body and the top surface of the natural nail, adapting to their respective shapes when pressed together.`
  - Why it is good: English is clear, answer is specific, and it matches the source context.

- `EP-4633790-A1_fr`
  - Q: `What type of products can include the polyamide-based microcapsules mentioned in the invention?`
  - A: `The polyamide-based microcapsules can be included in perfumed consumer products, particularly in household or personal care products.`
  - Why it is good: The question targets a concrete detail from the source and is useful for retrieval.

## Example Weaknesses

### Weak: untranslated or partially normalized terminology

- `PL-448242-A1_pl`
  - Q: `What is the application of the compound 8-(4-trifluorometoksy)benzyloamino-2'-deoksyadenozyny?`
  - Why it is weak: The English is mostly correct, but some chemical terms remain in Polish-style spelling instead of cleaner English transliteration or normalization.

### Weak: question pattern too generic

- Several questions still rely on broad templates such as:
  - `What is the main feature ...?`
  - `What are the main components ...?`
  - `What type of products ...?`
  - Why it is weak: These are often faithful, but not always the best retrieval-style questions.

## Current Quality Summary

- English correctness: much improved
- Faithfulness to source: good, but should keep being spot-checked
- Translation quality: acceptable, sometimes literal
- Retrieval usefulness: moderate
- Overall status: usable pilot output, but not yet strong enough to treat as final dataset quality

## Recommended Next Improvements

1. Add a third validator for question usefulness / retrieval quality.
2. Improve prompts to encourage more diverse question types.
3. Normalize technical terms in English more carefully.
4. Review a larger sample before trusting the pipeline broadly.
5. Keep periodic human review notes in this file after each generation update.

## Review Log

### Review 1

- Result: Better than the previous run.
- Main improvement: `en` outputs are now genuinely English.
- Main remaining issue: questions are still somewhat generic and repetitive.
