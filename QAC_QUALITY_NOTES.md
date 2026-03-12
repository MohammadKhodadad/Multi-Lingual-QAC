# QAC Quality Notes

Living notes for reviewing the generated Question-Answer-Context output over time.

## Current Snapshot

- Source file reviewed: `data/google_patents/qac/qac.csv`
- Current sample size: 90 rows
- Unique source documents: 6
- Languages present: `en`, `de`, `fr`, `es`, `ja`, `ko`, `zh`, `ru`, `pt`, `it`, `nl`, `ar`, `tr`, `pl`, `hi`
- Current design: English-first generation, then translation to the other languages
- Current validation: English language check + faithfulness check + retrieval-quality check before translation

## What Looks Good

- The `en` rows are genuinely English in the current sample.
- The answers appear grounded in the source patent abstract or context.
- The English answers are generally concise and readable.
- The structure is consistent across languages: one approved English QA gets translated into the target languages.
- The current validators seem to remove obvious language failures and some unsupported outputs.
- The latest prompt revision produces better questions when it focuses on method, ingredients, component role, or technical purpose.

## What Still Looks Weak

- Question style is still somewhat repetitive. Several current questions use broad patterns like:
  - `What types of products ...?`
  - `What is the application ...?`
  - `What is the main advantage ...?`
- Some questions are faithful but still too broad for strong retrieval benchmarking. They summarize a whole benefit or use case instead of isolating one sharp fact.
- Some translated outputs are accurate but feel literal rather than natural.
- Technical-name normalization is better than before, but should still be monitored on chemistry-heavy examples.
- The current sample is still very small, so quality judgments are preliminary.
- We now validate language, faithfulness, and retrieval usefulness, but the prompt and checker still need tuning to push questions toward more discriminative search intent.

## Example Strengths

### Good: grounded and understandable

- `DE-102020108236-B4_de`
  - Q: `What is the role of the open-pore vent textile in the bonding process of the two parts?`
  - A: `The open-pore vent textile is placed between the parts to ensure that it touches each part's surface, facilitating the bonding process under vacuum conditions.`
  - Why it is good: The question is specific, technical, and directly tied to a concrete mechanism in the source.

- `EP-4633790-A1_fr`
  - Q: `What types of products can incorporate the polyamide-based microcapsules?`
  - A: `The polyamide-based microcapsules can be used in perfumed consumer products, particularly in household or personal care items.`
  - Why it is good: The question targets a concrete detail from the source and remains useful for retrieval, even if it is still somewhat broad.

- `WO-2025201614-A1_ru`
  - Q: `What is the method for applying the hair-strengthening preparation?`
  - A: `The preparation should be applied in sections to the hair base and left under infrared rays for at least 10-20 minutes at a temperature of 30-70 degrees or under a thermal cap for at least 20-30 minutes at a temperature of 45-80 degrees.`
  - Why it is good: The question asks for a practical procedure, and the answer is concrete and well grounded in the source.

## Example Weaknesses

### Weak: technical normalization still worth watching

- `PL-448242-A1_pl`
  - Previous weak form: `What is the application of the compound 8-(4-trifluorometoksy)benzyloamino-2'-deoksyadenozyny?`
  - Latest status: improved to `8-(4-trifluoromethoxy)benzyloamino-2'-deoxyadenosine`
  - Why it matters: This shows the revised prompting is improving English normalization of technical terms, though similar cases should still be monitored.

### Weak: question pattern too generic

- `PL-448242-A1_pl`
  - Q: `What is the application of 8-(4-trifluoromethoxy)benzyloamino-2'-deoxyadenosine?`
  - Why it is weak: It is faithful, but still sounds like a template question. A more natural retrieval query would focus on radiosensitizing DNA damage or making cancer cells more sensitive to ionizing radiation.

- `WO-2025202696-A1_es`
  - Q: `What is the main advantage of the machine for fine defibrillation of pineapple leaves and other plant stems?`
  - Why it is weak: It asks for a broad summary benefit, so the answer bundles multiple ideas. A better question would isolate one stronger retrieval target such as corrosion reduction, safety structure, or the 2,000 RPM operating detail.

## Current Quality Summary

- English correctness: much improved
- Faithfulness to source: good, but should keep being spot-checked
- Translation quality: acceptable, sometimes literal
- Retrieval usefulness: improving and clearly better than the earlier run, but still limited by broad question templates in part of the sample
- Overall status: usable pilot output with meaningful prompt improvements, but not yet strong enough to treat as final dataset quality

## Recommended Next Improvements

1. Tune the question-quality validator so it rejects broad summary prompts without over-filtering.
2. Push the prompt further toward sharper search intent instead of asking for whole-document advantages or applications.
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
