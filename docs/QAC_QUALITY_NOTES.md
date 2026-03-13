# QAC Quality Notes

Living notes for reviewing the generated Question-Answer-Context output over time.

## Current Snapshot

- Source file reviewed: `data/google_patents/qac/qac.csv`
- Current sample size: 300 rows
- Unique source documents: 20
- Languages present: `en`, `de`, `fr`, `es`, `ja`, `ko`, `zh`, `ru`, `pt`, `it`, `nl`, `ar`, `tr`, `pl`, `hi`
- Current design: English-first generation, then translation to the other languages
- Current validation: English language check + faithfulness check + retrieval-quality check before translation
- Current translation behavior: each target language is translated separately, checked for translation quality, retried on failure, and skipped if it still fails after the retry budget
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
- The newest generation-prompt update still shows good question-shape diversity. In the current 20-question English sample, the set mixes `Why`, `Which`, `What function`, `What effect`, `How does`, `At what`, and other targeted question forms rather than clustering around one opening pattern.
- The refined translation checker improved the quality/coverage balance compared with the previous stricter run. In the current sample, all kept documents retain full multilingual coverage across the target languages.
- The latest `gpt-5-mini` prompt revision plus higher reasoning effort for English generation reduced the worst `purpose` / `advantages` fallback questions and improved semantic phrasing compared with the earlier `gpt-5-mini` runs.

## What Still Looks Weak

- Question style is more varied than before, but some mild repetition still remains around causal/rationale framing.
- A small number of questions still use broad function/purpose framing.
- Some questions are faithful but still a bit broad for strong retrieval benchmarking. They summarize a benefit or use case instead of isolating one narrower technical fact.
- Some translated outputs are accurate but still feel literal rather than fully native.
- A few target-language rows still show grammar or phrasing issues even when the meaning is preserved.
- Coverage is better than the previous strict-check run, but some sampled documents are still skipped at the English validation stage.
- Technical-name normalization is better than before, but should still be monitored on chemistry-heavy examples.
- The current sample is still modest, so quality judgments are still preliminary.
- We now validate language, faithfulness, retrieval usefulness, and translation quality, but the prompt and checker may still need tuning to improve question-form diversity further, reduce the remaining broad function/purpose cases, and catch more subtle multilingual fluency problems.
- A few questions are still somewhat extractive or lookup-oriented, especially around exact named compounds, percentage targets, or allowed classes.
- The latest `gpt-5-mini` run is the best `gpt-5-mini` result so far, but it still looks slightly weaker than the strongest `gpt-4.1` run for pure query quality.

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

### Weak: translation fluency still uneven

- `WO-2025211942-A1_ko`
  - Example issue: the Russian question still contains a grammar error (`этого стеклянного подложки`), even though the meaning is understandable.
  - Why it is weak: This is good enough for retrieval experiments, but not good enough to call the multilingual text polished or native-quality.

- `WO-2025208192-A1_pt`
  - Example issue: the Japanese question is understandable, but the phrasing is awkward and mixes a property question with unnatural `reason` wording.
  - Why it is weak: The information is preserved, but the sentence does not read like clean native technical Japanese.

### Weak: stricter validation reduced yield in some earlier runs

- In the latest run, all `20` sampled documents reached the final QAC file, but some earlier stricter runs dropped sampled English QAs before translation.
  - Why it is weak: quality control remains useful, but coverage can still drop if the checks become stricter than the generator can consistently satisfy.

### Weak: technical normalization still worth watching

- Chemical and material names are generally better normalized than in earlier runs, but this should still be monitored in future chemistry-heavy samples where exact English rendering matters.

## Current Quality Summary

- English correctness: much improved
- Faithfulness to source: clearly improved after removing title-only and ultra-short records
- Translation quality: improved and more controlled, usable for multilingual retrieval, but still only moderate in fluency
- Retrieval usefulness: clearly better than the earlier runs, with stronger step-specific queries and noticeably better diversity of question forms
- Overall status: best `gpt-5-mini` run so far and clearly better than the earlier `gpt-5-mini` outputs; usable pilot output with full multilingual coverage and better question shaping, though still slightly behind the best `gpt-4.1` run and still showing a few extractive questions and some uneven multilingual phrasing

## Recommended Next Improvements

1. Keep reducing the remaining extractive `lookup` questions without pushing the model back into broad summaries.
2. Keep improving prompt wording for dense-retrieval style questions before adding more pipeline complexity.
3. Tighten the translation checker so it catches grammar issues like the current Russian example without over-rejecting good rows.
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

### Review 7

- Result: Better than Review 6.
- Main improvement: the latest run keeps the stronger English question diversity and shows better translation faithfulness and somewhat more natural multilingual phrasing.
- Main remaining issue: translation fluency is still uneven across languages, with some literal wording and occasional grammar problems even when the meaning is preserved.

### Review 8

- Result: Mixed compared with Review 7.
- Main improvement: the new per-language translation validation increased quality control and prevents some weak multilingual outputs from being kept automatically.
- Main remaining issue: coverage dropped, one language row was lost, and some translated text is still not fluent enough despite passing the checker.

### Review 9

- Result: Better than Review 8.
- Main improvement: the refined translation checker recovered full language coverage for the kept documents while preserving the stronger quality-control setup.
- Main remaining issue: translation fluency is still only moderate, and the checker still misses some grammar problems and some broad English `function` wording.

### Review 10

- Result: Mixed relative to the best `gpt-4.1` run.
- Main improvement: the first `gpt-5-mini` switch produced cleaner question hygiene and kept full multilingual coverage.
- Main remaining issue: the questions became too literal and too lookup-oriented, making them less suitable for the desired dense-retrieval setting than the best `gpt-4.1` output.

### Review 11

- Result: Better than Review 10 and the best `gpt-5-mini` run so far.
- Main improvement: prompt tightening plus `medium` reasoning for English generation reduced the worst broad fallback wording and improved semantic question framing.
- Main remaining issue: a few English questions are still extractive or list/percentage oriented, so the best `gpt-4.1` run still looks slightly stronger overall.
