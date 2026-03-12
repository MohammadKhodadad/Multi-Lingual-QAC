# QAC Quality Notes

Living notes for reviewing the generated Question-Answer-Context output over time.

## Current Snapshot

- Source file reviewed: `data/google_patents/qac/qac.csv`
- Current sample size: 240 rows
- Unique source documents: 16
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
- The newest generation-prompt update improved question-shape diversity. In the current 16-question English sample, the set now mixes `Why`, `Which`, `What property`, `What function`, and `How does` instead of clustering mostly around one opening pattern.

## What Still Looks Weak

- Question style is more varied than before, but some mild repetition still remains around causal/rationale framing.
- A small number of questions still use broad purpose or advantage framing.
- Some questions are faithful but still a bit broad for strong retrieval benchmarking. They summarize a benefit or use case instead of isolating one narrower technical fact.
- Some translated outputs are accurate but feel literal rather than natural.
- Technical-name normalization is better than before, but should still be monitored on chemistry-heavy examples.
- The current sample is still modest, so quality judgments are still preliminary.
- We now validate language, faithfulness, and retrieval usefulness, but the prompt and checker may still need tuning to improve question-form diversity further and reduce the remaining broad purpose/advantage cases.

## Example Strengths

### Good: grounded and understandable

- `WO-2025211942-A1_ko`
  - Q: `What property of this glass substrate makes it suitable for forming fine patterns in semiconductor core substrates?`
  - A: `Its surface roughness (Ra) is at most 10 nm, which supports the formation of fine patterns.`
  - Why it is good: This is a good example of the newer prompt producing a property-focused question instead of defaulting to a generic mechanism or benefit question.

- `WO-2025211985-A1_ru`
  - Q: `Why is the mixed solution evaporated to a density of 1400-1800 kg/m3 in the process of preparing the heavy well-killing fluid?`
  - A: `The solution is evaporated to that density so the resulting filtrate has the required characteristics as a heavy well-killing fluid.`
  - Why it is good: The question points to a specific process step and operating target instead of asking for a vague process summary.

- `EP-4634668-A1_fr`
  - Q: `Which biomarker pairs are quantified in small extracellular vesicles to assess early-onset preeclampsia risk?`
  - A: `The biomarker pairs chosen from CD10, CD63, and placental alkaline phosphatase (PLAP) are quantified.`
  - Why it is good: This is a clear improvement in question-form diversity and asks for a concrete identifying detail rather than a broad summary.

- `CH-720331-B1_it`
  - Q: `How are the liquid and powder components combined in this dental whitening system before application?`
  - A: `The liquid from the first syringe and the powder from the second syringe are mixed together using a connector between the two syringes to prepare the whitening gel.`
  - Why it is good: The question targets a concrete mechanism and the answer is specific, procedural, and clearly grounded.

- `TR-2021006663-A2_tr`
  - Q: `Why is cold rolling performed after hot rolling or forging in the production of this steel for valves?`
  - A: `Cold rolling is used to achieve the desired dimensions and surface quality after the steel has been hot rolled or forged.`
  - Why it is good: The question asks about process rationale instead of just restating a material or product name, and the answer is narrow and factual.

## Example Weaknesses

### Weak: broad benefit / purpose framing

- `EP-4634126-A1_fr`
  - Q: `What is the purpose of using at least two highly hermetic sealing elements spaced apart and formed simultaneously during the cold welding of the metallic bridge element to the substrate?`
  - Why it is weak: It is grounded, but still framed as a broad `purpose` question. A stronger alternative would ask directly how the spaced simultaneous seals affect vacuum tightness or seal reliability.

### Weak: question-shape diversity still limited

- The current sample is much more varied than the previous run, but there is still room to add more `At what ...`, `Under what conditions ...`, and `Which component ...` questions.
  - Why it is weak: Diversity is improved, but it can still widen further so the benchmark covers more natural query forms and not just a few recurring patterns.

### Weak: technical normalization still worth watching

- Chemical and material names are generally better normalized than in earlier runs, but this should still be monitored in future chemistry-heavy samples where exact English rendering matters.

## Current Quality Summary

- English correctness: much improved
- Faithfulness to source: clearly improved after removing title-only and ultra-short records
- Translation quality: acceptable, sometimes literal
- Retrieval usefulness: clearly better than the earlier runs, with stronger step-specific queries and noticeably better diversity of question forms
- Overall status: best run so far; usable pilot output with better trustworthiness, better question specificity, and improved question-form diversity, though a few broad purpose-style questions still remain

## Recommended Next Improvements

1. Keep reducing the remaining broad `purpose` / `advantage` cases without over-filtering.
2. Encourage even more diversity in question forms, especially `At what ...`, `Under what conditions ...`, and `Which component ...` patterns.
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

### Review 5

- Result: Better than Review 4.
- Main improvement: the latest prompt/judge update further reduced broad `advantage` / `purpose` questions and produced more specific step- and mechanism-focused queries.
- Main remaining issue: the set now leans heavily toward `How does ...` framing, so diversity of high-quality question forms is the next area to improve.

### Review 6

- Result: Better than Review 5.
- Main improvement: the generation-only prompt update noticeably improved question-form diversity, with strong `Why`, `Which`, `What property`, and `What function` questions now appearing in the sample.
- Main remaining issue: a few broad `purpose`-style questions still remain and should be pushed toward more direct technical formulations.
