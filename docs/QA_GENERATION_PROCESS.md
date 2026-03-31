# QA Generation Process

This note explains how `src/multi_lingual_qac/qac_generation/openai_qa.py` is used today across the project.

## Common Flow

All QA generation runs through `run_qa_pipeline()`.

Shared steps:

1. Load the input corpus CSV.
2. Sample rows, usually stratified by language.
3. Generate one question-answer pair from each sampled row.
4. Run validation checks before keeping the pair.
5. Retry failed generations with feedback from the validation step.
6. Write accepted rows to `qac/qac.csv`.

The validation stack is:

- `check_english_language()` or `check_language_match()`
- `check_faithfulness()`
- `check_question_quality()`
- `check_translation_quality()` for translated outputs

The retry loop passes back:

- the failure reason
- the previous failed question/answer
- a short "better direction" hint from the quality checker when available

So generation is not one-shot. It is generate -> validate -> retry with feedback.

## Output Format

`qac.csv` always contains:

- `corpus_id`
- `language`
- `question`
- `answer`

Each row is one final kept query-answer pair for one corpus document in one language.

## Source-Specific Modes

### EPO

EPO uses the default English-first flow.

Process:

1. Sample patent corpus rows.
2. Generate the canonical question/answer in English.
3. Validate English language, faithfulness, and retrieval quality.
4. Translate the approved English pair into target languages.
5. Validate each translation for meaning, fluency, specificity, and artifacts.

Domain behavior:

- Prompt style is patent/technical.
- Questions prefer function, mechanism, role, effect, operating condition, or other semantic technical facts over raw lookup.

### Wikidata

Wikidata also uses the English-first flow.

Process:

1. Sample multilingual Wikipedia-derived chunk rows.
2. Generate the canonical question/answer in English.
3. Validate the English pair.
4. Translate into target languages.
5. Write one `qac.csv` row per `(corpus_id, language)`.

Domain behavior:

- Prompt style is closer to encyclopedia/explanatory retrieval than patents.
- The resulting `qac.csv` is later used by `label_wikidata_qrels.py` to build per-language `queries.csv` and duplicated qrels.

### JRC-Acquis

JRC now uses a legal pair-level generation flow.

Current process:

1. `prepare_jrc_qa_inputs()` samples directional language pairs from `document_pairs_all.csv`.
2. It selects source QA candidates by language.
3. For each selected source document, it chooses one sampled pair.
4. It writes `qac/qa_generation_sources.csv`, where each row represents one pair and uses the translated/target document as the generation side.
5. `run_qa_pipeline()` reads those rows in same-language mode.
6. The question/answer is generated directly from the target-side document text, in the target-side language.
7. The generated pair is validated with the legal-domain checks.
8. The final query is linked to both documents in the pair:
   - the source/main-side document
   - the translated/target-side document

So JRC is no longer CELEX-wide and no longer expands one query to all aligned languages.

Domain behavior:

- Prompt style is legal/regulatory, not chemistry-like.
- Questions should ask about operative meaning, condition, threshold, exception, evidence, consequence, scope, or legal effect.
- Questions should not mention article numbers, paragraph numbers, recital numbers, annex labels, or phrases like `this article` unless essential.
- The generated query is attached to both documents in the selected pair, not to all CELEX-aligned languages.

## Domain Hint System

`run_qa_pipeline()` now accepts `domain_hint`.

Current use:

- `patent` for EPO
- `encyclopedia` for Wikidata
- `legal` for JRC-Acquis

This domain hint changes:

- generation prompts
- faithfulness checking
- question-quality checking
- translation prompts
- translation-quality checking

This is important because JRC legal material behaves very differently from chemistry patents or Wikipedia pages.

## Key JRC Helper Files

JRC QA generation currently depends on:

- `src/multi_lingual_qac/qac_generation/jrc_acquis.py`
- `src/multi_lingual_qac/qac_generation/openai_qa.py`
- `src/multi_lingual_qac/pipeline.py`

Main intermediate files:

- `data/JRC-ACQUIS/preprocessed/document_pairs_all.csv`
- `data/JRC-ACQUIS/preprocessed/corpus_qa_candidates.csv`
- `data/JRC-ACQUIS/qac/qa_generation_sources.csv`
- `data/JRC-ACQUIS/qac/sampled_pairs.csv`
- `data/JRC-ACQUIS/qac/qac.csv`

## Practical Summary

- EPO: English generation -> translate
- Wikidata: English generation -> translate
- JRC: pick one pair -> generate legal query from the translated side -> attach that query to both pair documents

The JRC setup is designed so the corpus stays natural in each language, while the query intent stays aligned across languages.
