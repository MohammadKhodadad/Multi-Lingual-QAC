# QAC Quality Notes

Living notes for reviewing the generated Question-Answer-Context output over time.

## Current Snapshot

- Source file reviewed: `data/google_patents/qac/qac.csv`
- Current sample size: 449 rows
- Unique source documents: 30
- Languages present: `en`, `de`, `fr`, `es`, `ja`, `ko`, `zh`, `ru`, `pt`, `it`, `nl`, `ar`, `tr`, `pl`, `hi`
- Current design: English-first generation, then translation to the other languages
- Current validation: English language check + faithfulness check + retrieval-quality check before translation
- Current translation behavior: each target language is translated separately with `medium` reasoning, checked for generic artifact/fluency/meaning quality, retried with failure-type feedback plus the previous failed translation, and skipped if it still fails after the retry budget
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
- The newest stricter retry-feedback update appears to help again: the latest reviewed sample keeps full multilingual coverage while shifting more English questions toward `Why`, `What property`, and `What function` framing instead of broad summary wording.
- The latest generic translation prompt/checker update removed the need for handcrafted per-language translation hints while still improving artifact control.
- The latest run no longer shows the earlier kept Hindi row with Korean-script leakage; the new generic artifact rules appear to be catching that class of failure.
- The latest multilingual sample is the best overall `gpt-5-mini` run so far for end-to-end balance across English quality, translation cleanliness, and pipeline robustness.

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
- The remaining weak cases are now more concentrated in literal list/value questions than in vague `purpose` / `advantage` questions, which is better than before but still not ideal for a dense-retrieval benchmark.
- Coverage is no longer always full at the row level in the latest stricter multilingual runs. The current `30`-document run dropped one `hi` translation, finishing with `449` rows instead of `450`.
- The stricter translation checker may now be slightly too strict on some low-severity Hindi phrasing or terminology choices, even when meaning is preserved.

## Example Strengths

### Good: grounded and understandable

- `US-2025325674-A1_en`
  - Q: `What function does the DNA oligonucleotide (SEQ ID NO:1–3) perform in the autoimmune disease therapeutic agent?`
  - A: `The DNA oligonucleotide selectively binds to IFN-γ and thereby selectively inhibits IFN-γ, producing a therapeutic effect against autoimmune diseases.`
  - Why it is good: In the latest run, the kept multilingual rows for this example are cleaner than before, and the earlier Hindi foreign-script contamination no longer appears.

- `US-2025327402-A1_en`
  - Q: `Why is the evaporation area selected for its existing evaporite deposits?`
  - A: `Because evaporite-rich areas show that natural evaporation there concentrates and deposits dissolved solids; sending metal-rich produced water to such an area and letting it evaporate causes the dissolved metals to precipitate and be recoverable.`
  - Why it is good: This is a strong causal retrieval question. It asks why a specific site feature matters and pulls out a concrete mechanism rather than a broad process summary.

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

### Weak: still too extractive in places

- `WO-2025211579-A1_ko`
  - Q: `Which pH adjuster is specified to set the enteric coating composition to pH 2.5–7.0?`
  - Why it is weak: This is faithful and precise, but it is still mostly a direct lookup for a named ingredient rather than a more semantic query about the role or effect of that ingredient.

- `DE-112023005440-T5_de`
  - Q: `What are the required area-fraction ranges for ferrite, retained austenite and the M-A constituent in the microstructure of the high-strength cold-rolled steel sheet?`
  - Why it is weak: This is exact and answerable, but it is mainly a range-list extraction question and still feels easier for literal matching than for dense semantic retrieval.

### Weak: question-shape diversity still limited

- The current sample is much more varied than the previous run, but some clustering still remains around `Why ...` rationale framing.
  - Why it is weak: Diversity is improved, but it can still widen further so the benchmark covers more natural query forms and not just a few recurring patterns.

### Weak: translation fluency still uneven

- `WO-2025211983-A1_ru`
  - Example issue: the `hi` row was dropped in the latest run because the checker flagged low-severity wording around an English-derived term for `oncological`.
  - Why it is weak: This is preferable to keeping a clearly bad row, but it shows the stricter checker can still trade away coverage for relatively mild style issues.

- `DE-102024111126-A1_de`
  - Example issue: the latest Russian row is much cleaner than before, but still reads somewhat formal and procedural rather than fully native.
  - Why it is weak: The artifact problems are reduced, but some target-language outputs still sound translated rather than naturally authored.

### Weak: stricter validation still sometimes reduces yield

- In the latest `30`-document run, `29/30` documents kept full `15-language` coverage, while one `hi` row was dropped after translation retries.
  - Why it is weak: quality control is working, but coverage can still drop if the checker rejects low-severity target-language wording that might otherwise be acceptable for retrieval use.

### Weak: technical normalization still worth watching

- Chemical and material names are generally better normalized than in earlier runs, but this should still be monitored in future chemistry-heavy samples where exact English rendering matters.

## Current Quality Summary

- English correctness: much improved
- Faithfulness to source: clearly improved after removing title-only and ultra-short records
- Translation quality: clearly improved and better controlled, with stronger artifact filtering and cleaner kept rows; now solid for multilingual retrieval, though still not fully native-quality in every language
- Retrieval usefulness: clearly better than the earlier runs, with stronger step-specific queries and noticeably better diversity of question forms
- Overall status: best overall `gpt-5-mini` run so far and the strongest multilingual run yet; better than the earlier `gpt-5-mini` outputs in translation cleanliness and robustness, and probably the best end-to-end balance so far, though the strongest `gpt-4.1` run may still retain a slight edge on pure English query quality

## Recommended Next Improvements

1. Keep reducing the remaining extractive `lookup` questions without pushing the model back into broad summaries.
2. Relax or refine the translation checker slightly so it does not drop acceptable rows for low-severity Hindi wording while still rejecting real artifacts.
3. Continue monitoring foreign-script leakage, code-mixing, and English glosses to confirm the new generic artifact rules stay effective on larger samples.
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

### Review 12

- Result: Better than Review 11 and still the best `gpt-5-mini` run so far.
- Main improvement: the stricter retry feedback appears to have improved English question shaping again without hurting multilingual coverage; the sample now leans more toward causal, property, and function questions and away from broad summary wording.
- Main remaining issue: some questions are still literal lookup prompts for named compounds, permitted classes, or numeric ranges, so the strongest `gpt-4.1` run still retains a small edge in pure retrieval quality.

### Review 13

- Result: Better than Review 12 and the best translation-focused run so far.
- Main improvement: the new generic translation artifact checks plus feedback-aware retries reduced obvious multilingual cleanup problems, and the earlier kept Hindi foreign-script leakage no longer appears in the reviewed sample.
- Main remaining issue: one Hindi row was dropped in the latest `30`-document run, so the stricter checker still looks slightly over-sensitive on some low-severity target-language wording.
