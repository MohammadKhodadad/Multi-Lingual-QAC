# QAC Quality Notes

Living notes for reviewing the generated Question-Answer-Context output over time.

## Current Snapshot

- Source file reviewed: `data/google_patents/qac/qac.csv`
- Current sample size: 75 rows
- Unique source documents: 5
- Languages present: `en`, `de`, `fr`, `es`, `ja`, `ko`, `zh`, `ru`, `pt`, `it`, `nl`, `ar`, `tr`, `pl`, `hi`
- Current design: English-first generation, then translation to the other languages
- Current validation: English language check + faithfulness check + retrieval-quality check before translation

## What Looks Good

- The `en` rows are now actually English. This is a major improvement over the previous version.
- Most answers appear grounded in the source patent abstract or context.
- The English answers are generally concise and readable.
- The structure is consistent across languages: one approved English QA gets translated into the target languages.
- The current validators seem to remove obvious language failures and some unsupported outputs.
- The latest prompt revision produces more retrieval-style questions that ask about purpose, application, method, or composition rather than falling back as often to generic invention-summary templates.

## What Still Looks Weak

- Question style is still repetitive. Many questions follow patterns like:
  - `What is the main feature ...?`
  - `What are the main components ...?`
  - `What type of products ...?`
- Some English outputs still preserve non-English technical spellings when cleaner English normalization would be better.
- Some translated outputs are accurate but feel literal rather than natural.
- The current sample is still very small, so quality judgments are preliminary.
- Some English questions still sound a bit document-centered, for example using phrases like `described in the invention` or `mentioned in the invention`, instead of sounding like natural user queries.
- We now validate language, faithfulness, and retrieval usefulness, but the prompt and checker may still need tuning over time.

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

- `WO-2025201614-A1_ru`
  - Q: `What are the recommended application methods for the hair strengthening and growth stimulating preparation?`
  - A: `The preparation should be applied in a sequential strand-by-strand manner, followed by exposure to infrared rays for at least 10-20 minutes at a temperature of 30-70 degrees, or under a thermal cap for at least 20-30 minutes at 45-80 degrees.`
  - Why it is good: The question is more retrieval-like than a generic component question and targets a concrete usage detail from the source.

## Example Weaknesses

### Weak: untranslated or partially normalized terminology

- `PL-448242-A1_pl`
  - Previous weak form: `What is the application of the compound 8-(4-trifluorometoksy)benzyloamino-2'-deoksyadenozyny?`
  - Latest status: improved to `8-(4-trifluoromethoxy)benzyloamino-2'-deoxyadenosine`
  - Why it matters: This shows the revised prompting is improving English normalization of technical terms, though similar cases should still be monitored.

### Weak: question pattern too generic

- Several questions still rely on broad templates such as:
  - `What are the applications ...?`
  - `What is the composition ...?`
  - `What type of products ...?`
  - Why it is weak: These are usually faithful, but some still sound slightly generic or summary-like rather than like natural user retrieval queries.

## Current Quality Summary

- English correctness: much improved
- Faithfulness to source: good, but should keep being spot-checked
- Translation quality: acceptable, sometimes literal
- Retrieval usefulness: improving and now clearly better than the earlier run
- Overall status: usable pilot output with meaningful prompt improvements, but not yet strong enough to treat as final dataset quality

## Recommended Next Improvements

1. Tune the new question-quality validator so it rejects generic questions without over-filtering.
2. Push the prompt further toward natural search intent and away from document-centered phrasing like `described in the invention`.
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
