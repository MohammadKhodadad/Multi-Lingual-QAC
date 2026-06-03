# Branch Pipeline Architecture

This document explains how the `chempatents` and `JRC-Acquis` branches build documents and generate queries. The focus is the flow from raw source data to retrieval corpus and QAC/query generation, so the two implementations can later be integrated under one shared architecture.

## Shared Concepts

Both branches ultimately build the same kind of benchmark objects:

- A retrieval `corpus`: documents with ids, titles, text, language metadata, and source-specific identifiers.
- A query/QAC table: generated questions and grounded answers linked to corpus documents.
- Relevance links: either implicit by shared document identifiers or explicit linked corpus ids.
- Optional export/evaluation artifacts for Hugging Face and MTEB.

The branches differ mainly in the source data model:

- `chempatents` starts from patent publications, mostly Google Patents / EPO chemistry data.
- `JRC-Acquis` starts from aligned EU legal documents grouped by CELEX id.

## Chempatents Branch

### Purpose

The `chempatents` branch builds a multilingual chemistry-patent retrieval benchmark. It supports two related QAC generation paths:

1. Generate English Q&A from patent abstracts/claims, then translate the approved Q&A into multiple languages.
2. Use multilingual patent documents directly and generate target-language questions without a translation step.

### Main Entry Points

- `main.py`
- `src/multi_lingual_qac/cli.py`
- `src/multi_lingual_qac/pipeline.py`
- `src/multi_lingual_qac/dataloaders/google_patents.py`
- `src/multi_lingual_qac/qac_generation/openai_qa.py`
- `src/multi_lingual_qac/qac_generation/multilingual_qa.py`
- `src/multi_lingual_qac/qac_generation/balanced_multilingual_qa.py`

### Flow 1: Google Patents Extraction

The main chemical branch starts by querying Google BigQuery patent data.

Relevant module:

- `src/multi_lingual_qac/dataloaders/google_patents.py`

Key functions:

- `build_query()`
- `build_query_per_language_top_n()`
- `run_query()`
- `extract_chemistry_patents()`
- `extract_chemistry_patents_per_language()`

The loader queries patent publication data and filters for chemistry-like patents using CPC/IPC classes and optional SureChEMBL signals. The output is an offline NDJSON cache:

```text
data/google_patents/chemistry_patents.ndjson
```

This raw file is the branch's main reusable extraction artifact. Later stages can run without re-querying BigQuery.

### Flow 2: Preprocessing And Corpus Building

Raw patent JSON is converted into per-language CSV files and then merged into a retrieval corpus.

Relevant module:

- `src/multi_lingual_qac/dataloaders/google_patents.py`

Key functions:

- `clean_text()`
- `preprocess_ndjson_to_csv()`
- `merge_corpus_csv()`

Typical artifacts:

```text
data/google_patents/preprocessed/{lang}.csv
data/google_patents/corpus.csv
```

Each row represents one patent-language document. The corpus includes fields such as:

- `id`
- `language`
- `title`
- `abstract`
- `first_claim`
- `context`
- `publication_number`
- `publication_date`
- `source`
- `ipc_codes`

The `context` is built primarily from title, abstract, and first claim. Very short abstracts are filtered out so title-only or low-substance records are less likely to enter QAC generation.

### Flow 3: Optional Multilingual Patent Subset

The branch can also build a subset of patent documents that exist in multiple languages.

Relevant module:

- `src/multi_lingual_qac/preprocess/filter_multilingual.py`

Key function:

- `find_multilingual_documents()`

Typical artifact:

```text
data/google_patents/multilingual_corpus.csv
```

This grouped multilingual corpus supports direct target-language query generation, where the model receives multiple language realizations of the same publication.

### Flow 4A: English-First QAC Generation

The original QAC pipeline samples the corpus, generates one English Q&A pair per selected document, verifies it, and translates it into target languages.

Relevant module:

- `src/multi_lingual_qac/qac_generation/openai_qa.py`

Key functions:

- `run_qa_pipeline()`
- `sample_corpus()`
- `generate_qa_english()`
- `check_english_language()`
- `check_faithfulness()`
- `check_question_quality()`
- `translate_qa()`
- `check_translation_quality()`

Typical artifact:

```text
data/google_patents/qac/qac.csv
```

Per sampled document, the flow is:

1. Sample a corpus row.
2. Generate one English question and answer from `context`.
3. Check that the output is English.
4. Check faithfulness to the source text.
5. Check retrieval/question quality.
6. Retry generation with feedback if validation fails.
7. Translate the approved English Q&A to the target languages.
8. Validate translations for language, meaning preservation, terminology, artifacts, and fluency.
9. Write English plus approved translated QAC rows.

This path is translation-centered. The source passage may be multilingual, but the primary generated Q&A is English.

### Flow 4B: Direct Multilingual QAC Generation

The later multilingual path generates questions directly in the target language from multilingual patent groups. It does not use translation for query generation.

Relevant module:

- `src/multi_lingual_qac/qac_generation/multilingual_qa.py`

Key functions:

- `load_multilingual_corpus()`
- `pick_target_languages()`
- `_build_all_passages_text()`
- `generate_qa_batch()`
- `grade_faithfulness()`
- `grade_quality()`
- `_process_document()`
- `run_multilingual_qa_pipeline()`

This path supports two generation modes:

- `technical`: single concrete fact questions.
- `semantic`: concept, problem, approach, or application questions.

For each publication and target language:

1. Group all rows by `publication_number`.
2. Select target language or languages using a strategy.
3. Build an all-passages context containing available language versions.
4. Generate three Q&A candidates in the target language.
5. Grade faithfulness for all three candidates.
6. Grade question quality for all three candidates.
7. Compute `total_score = faithfulness + quality`.
8. Sort candidates best-first.
9. Write all candidates and usually consume the top one downstream.

The balanced variant is implemented in:

- `src/multi_lingual_qac/qac_generation/balanced_multilingual_qa.py`

Key functions:

- `_allocate_question_quotas()`
- `_build_generation_plan()`
- `_select_best_rows()`
- `run_balanced_multilingual_qa()`

Typical artifacts include branch-local files such as:

```text
balanced_100_qac_regraded.csv
data/google_patents/qac/balanced_*_qac.csv
data/google_patents/qac/balanced_*_qac_all_generated.csv
```

### Flow 5: Optional EPO Ingestion

The `chempatents` branch also contains EPO ingestion support. This is a side path rather than the original Google Patents pipeline.

Relevant modules:

- `src/multi_lingual_qac/dataloaders/epo_bdds.py`
- `src/multi_lingual_qac/dataloaders/epo_xml.py`

Key functions:

- `ingest_n_batches()`
- `parse_epo_patent_bytes()`
- `build_row_for_language()`
- `analyze_epo_chemistry()`

Typical artifacts:

```text
data/EPO/multilingual_corpus.csv
data/EPO/manifest.json
```

EPO ingestion streams BDDS XML, extracts multilingual patent rows, filters for chemistry, and appends eligible multilingual documents.

### Chempatents Summary

The chemical branch has a mature QAC-generation stack with multiple modes:

- Patent extraction and corpus building are source-specific.
- English-first QAC generation handles broad multilingual output through translation.
- Direct multilingual generation supports target-language questions and scored candidate selection.
- Verifier prompts are central to quality control.
- Export and MTEB evaluation operate after QAC generation.

## JRC-Acquis Branch

### Purpose

The `JRC-Acquis` branch builds a multilingual legal retrieval benchmark from aligned EU legal documents. Documents are grouped by CELEX id, making cross-language relevance more explicit than in the patent branch.

The current branch can optionally restrict the corpus to chemistry-related legal documents through EuroVoc-based filtering.

### Main Entry Points

- `main.py`
- `src/multi_lingual_qac/cli.py`
- `src/multi_lingual_qac/pipeline.py`
- `src/multi_lingual_qac/preprocess/corpus.py`
- `src/multi_lingual_qac/dataloaders/jrc_acquis.py`
- `src/multi_lingual_qac/qac_generation/jrc_acquis.py`
- `src/multi_lingual_qac/qac_generation/openai_qa.py`

### Flow 1: Download And Raw Loading

The branch downloads or reads official JRC-Acquis language archives.

Relevant module:

- `src/multi_lingual_qac/dataloaders/jrc_acquis.py`

Key constants:

- `JRC_ACQUIS_LANGS`
- `JRC_ACQUIS_CORPUS_INDEX_URL`
- `CHEMICAL_EUROVOC_IDS`

Key functions:

- `download_jrc_acquis_archives()`
- `iter_jrc_acquis_raw_records()`
- `_iter_archive_records()`
- `_iter_xml_records()`
- `_parse_xml_record()`
- `load_jrc_acquis_raw()`

Typical input and output:

```text
data/JRC-ACQUIS/input/jrc-<lang>.tgz
data/JRC-ACQUIS/prepared/raw_documents.jsonl
data/JRC-ACQUIS/prepared/raw_load_stats.json
```

The parser extracts CELEX id, language, title, EuroVoc metadata, and paragraph-level document text. If chemical filtering is active, only legal documents matching chemistry-related EuroVoc ids are retained.

### Flow 2: JRC Text Cleaning

The JRC branch performs source-specific cleanup because the source documents contain legal headers, reference lines, article labels, annexes, signatures, and formatting artifacts.

Relevant functions:

- `_normalize_jrc_text()`
- `_clean_jrc_paragraphs()`
- `_trim_jrc_to_operative_body()`
- `_looks_like_article_heading()`
- `_looks_like_reference_line()`
- `_looks_like_institution_heading()`

The goal is to keep substantive legal text while removing helper/header pseudo-documents and boilerplate that would pollute retrieval.

### Flow 3: Corpus Building

Raw JSONL documents are converted into full corpus rows, MTEB-style corpus rows, multilingual subsets, CELEX pair tables, and QA-candidate subsets.

Relevant module:

- `src/multi_lingual_qac/dataloaders/jrc_acquis.py`

Key functions:

- `build_jrc_acquis_document_corpus()`
- `_build_jrc_document_entry()`
- `_build_jrc_document_batch()`
- `_build_jrc_pair_batch()`
- `_assess_jrc_qa_candidate()`
- `_is_jrc_qa_candidate()`

Typical artifacts:

```text
data/JRC-ACQUIS/corpus.csv
data/JRC-ACQUIS/preprocessed/corpus_full.csv
data/JRC-ACQUIS/preprocessed/corpus_multilingual.csv
data/JRC-ACQUIS/preprocessed/corpus_multilingual_full.csv
data/JRC-ACQUIS/preprocessed/document_pairs_all.csv
data/JRC-ACQUIS/preprocessed/corpus_qa_candidates.csv
data/JRC-ACQUIS/preprocessed/inspection_sample.csv
data/JRC-ACQUIS/preprocessed/document_corpus_stats.json
```

Important fields in full corpus rows include:

- `id`
- `language`
- `title`
- `header_notes`
- `abstract`
- `context`
- `generation_context`
- `operative_context`
- `celex`
- metadata about body, annex, and signature zones

The JRC branch differs from chempatents by explicitly separating:

- `context`: retrieval text.
- `generation_context`: text used for Q&A generation.
- `operative_context`: legal body text used for QA suitability.

### Flow 4: QA Candidate Filtering

JRC uses a source-specific QA filter before generation.

Relevant constants/functions:

- `JRC_QA_FILTER_PROFILES`
- `_jrc_qa_filter_config()`
- `_assess_jrc_qa_candidate()`
- `_is_jrc_qa_candidate()`

The filter checks features such as:

- minimum text length
- number of body paragraphs
- amount of operative text
- number of medium-length operative paragraphs
- ratio of short paragraphs

The output is:

```text
data/JRC-ACQUIS/preprocessed/corpus_qa_candidates.csv
```

This file is the pool from which generation sources are selected.

### Flow 5: QA Source Selection

Unlike chempatents, JRC has a dedicated source-selection stage before QAC generation. This stage builds a smaller retrieval corpus and generation list from multilingual CELEX groups.

Relevant module:

- `src/multi_lingual_qac/qac_generation/jrc_acquis.py`

Key functions:

- `prepare_jrc_qa_inputs()`
- `_weighted_sample_without_replacement()`
- `_assign_synthetic_targets()`

Typical artifacts:

```text
data/JRC-ACQUIS/qac/sampled_sources.csv
data/JRC-ACQUIS/qac/sampled_pairs.csv
data/JRC-ACQUIS/qac/qa_generation_sources.csv
data/JRC-ACQUIS/qac/corpus_full.csv
data/JRC-ACQUIS/qac/corpus.csv
data/JRC-ACQUIS/qac/qa_selection_stats.json
```

The source-selection flow is:

1. Load multilingual full corpus rows.
2. Load QA candidate rows.
3. Group documents by CELEX id.
4. For each allowed source language, sample QA-eligible documents.
5. Oversample generation candidates so failed rows can be skipped.
6. Build a QA-scoped retrieval corpus containing selected documents and their same-CELEX positives.
7. Write generation rows with linked corpus ids and linked language metadata.
8. Optionally assign synthetic translation targets such as Chinese.

This stage makes relevance links explicit before QAC generation. Each generated query can later point to multiple relevant corpus documents in other languages.

### Flow 6: JRC Query Generation

JRC query generation reuses the generic `openai_qa.py` pipeline, but with legal-specific settings.

Relevant module:

- `src/multi_lingual_qac/qac_generation/openai_qa.py`

Key functions:

- `run_qa_pipeline()`
- `generate_qa_in_language()`
- `check_language_match()`
- `check_faithfulness()`
- `check_question_quality()`
- `check_legal_question_shape()`
- `check_cross_language_answer_support()`
- `translate_qa()`
- `check_translation_quality()`

The JRC call uses settings like:

```text
same_language=True
domain_hint="legal"
require_cross_language_support=True
synthetic_translation_targets=["zh"]
```

Per generation row, the flow is:

1. Generate Q&A in the same language as the source document.
2. Check that question and answer match the expected language.
3. Check faithfulness to the source legal text.
4. Check retrieval/question quality.
5. Check legal question shape, rejecting weak checklist, condition-list, menu, or broad procedural questions.
6. Check that linked same-CELEX translations can support the same answer.
7. Retry generation with feedback when validation fails.
8. Optionally translate accepted Q&A into synthetic target languages such as Chinese.
9. Validate synthetic translations.
10. Write base and synthetic QAC rows.

Typical artifact:

```text
data/JRC-ACQUIS/qac/qac.csv
data/JRC-ACQUIS/qac/qac_generation_stats.json
```

The JRC `qac.csv` contains legal-specific metadata such as:

- `celex`
- `source_language`
- `source_corpus_id`
- `target_language`
- `target_corpus_id`
- `linked_corpus_ids_json`
- `linked_languages_json`
- `cross_language_support_checked`
- `cross_language_supported_corpus_ids_json`
- `cross_language_support_reasons_json`

### JRC Summary

The JRC branch is more structured around document alignment than the chemical branch:

- CELEX id is the central grouping key.
- Corpus building separates retrieval text from generation text.
- QA source selection is a first-class stage.
- Query generation requires cross-language answer support.
- Legal-specific validation rejects weak legal question shapes.

## Side-by-Side Comparison

| Area | Chempatents | JRC-Acquis |
| --- | --- | --- |
| Source domain | Chemistry patents | EU legal documents, optionally chemistry-filtered |
| Main source id | `publication_number` | `celex` |
| Raw loading | BigQuery / EPO XML | JRC-Acquis TGZ/XML |
| Raw artifact | `chemistry_patents.ndjson` | `prepared/raw_documents.jsonl` |
| Main corpus artifact | `data/google_patents/corpus.csv` | `data/JRC-ACQUIS/corpus.csv` |
| Full metadata corpus | Per-language preprocessed CSVs | `preprocessed/corpus_full.csv` |
| Multilingual grouping | Optional by publication number | Core by CELEX id |
| Generation context | Patent title, abstract, first claim | Legal `generation_context` and operative text |
| Candidate filtering | Mostly text length and sampling | Dedicated QA candidate filter |
| Query generation path | English-first translation and direct multilingual generation | Same-language legal generation with cross-language support |
| Verifier emphasis | Language, faithfulness, question quality, translation quality | Language, faithfulness, question quality, legal shape, cross-language support |
| Translation | Main path translates English Q&A to many languages | Optional synthetic translations, currently Chinese |
| Relevance links | Usually publication-number based | Explicit linked corpus ids by CELEX |

## Integration Opportunities

The two branches can be unified if the source-specific parts are isolated behind common interfaces.

### 1. Common Source Builder Interface

Each source should expose the same high-level operations:

```text
prepare_raw_source()
build_full_corpus()
build_retrieval_corpus()
build_qa_candidates()
```

Chempatents would implement these for Google Patents and EPO. JRC would implement them for JRC-Acquis.

### 2. Common Document Schema

A shared full-corpus schema should separate these concepts:

- stable document id
- source dataset name
- source grouping id (`publication_number`, `celex`, etc.)
- language
- title
- retrieval text
- generation text
- metadata JSON

This would allow both branches to export one canonical internal format even if source-specific metadata differs.

### 3. Common QA Source Selection Stage

JRC already has a strong explicit source-selection stage. Chempatents could benefit from the same abstraction:

```text
full corpus + qa candidates -> generation sources + retrieval subset + linked positives
```

For patents, linked positives would be documents sharing `publication_number` or patent family id. For JRC, linked positives remain same-CELEX documents.

### 4. Common QAC Generation Engine

Both branches already depend heavily on `openai_qa.py`. The shared engine should accept a domain profile:

```text
domain = chemistry | legal
generation_mode = same_language | english_first | direct_multilingual
require_cross_language_support = true | false
synthetic_translation_targets = [...]
```

Domain profiles would choose:

- generation prompt
- faithfulness prompt
- query-quality prompt
- domain-specific shape checker
- retry feedback style

### 5. Common Validation Stack

The validation stack can be layered:

```text
language check
faithfulness check
query quality check
domain shape check
cross-language support check
translation quality check
```

Chempatents would use a chemistry/scientific query-shape checker. JRC would use the legal-shape checker.

### 6. Common Artifact Layout

Both branches should write a consistent layout:

```text
data/{SOURCE}/prepared/
data/{SOURCE}/preprocessed/
data/{SOURCE}/qac/
reports/{source}/
```

This is already mostly true for JRC and partly true for chempatents.

## Proposed Unified Flow

The integrated architecture could look like this:

```text
1. prepare source
   raw provider -> raw source artifact

2. build corpus
   raw artifact -> full corpus -> retrieval corpus -> multilingual groups

3. select QA sources
   full corpus + QA candidates -> generation source rows + retrieval subset + linked positives

4. generate QAC
   generation rows -> candidate Q&A -> validators -> accepted QAC rows

5. export/evaluate
   retrieval corpus + QAC + relevance links -> HF/MTEB artifacts
```

Source-specific code should stop after producing canonical corpus and generation-source rows. Query generation and validation should then be shared.

## Merge Effort And GPT-5.5 Token Budget

The safest way to budget this merge is to assume it is a real refactor, not a simple branch merge. The goal is not only to put both code paths on one branch, but to create shared base methods/classes and make `chempatents` and `JRC-Acquis` source-specific children or adapters of those shared abstractions.

### Conservative Token Estimate

| Scope | Estimated Tokens | What It Covers |
| --- | ---: | --- |
| Architecture/design only | `80k-180k` | Read both branches, design base interfaces/classes, define migration steps, no major code edits |
| Minimal implementation | `400k-800k` | Add shared base classes, adapt both branches lightly, preserve current behavior, add limited tests |
| Solid production refactor | `1.2M-2.5M` | Shared source/corpus/QA/QAC abstractions, adapters for both branches, artifact compatibility, focused tests, docs |
| Very careful merge | `2.5M-5M+` | Full branch reconciliation, CLI cleanup, regression tests, generated-data checks, edge-case handling, multiple fix passes |

For planning, use the upper hand:

```text
Expected careful implementation budget: 2M-3.5M total tokens
No-surprises reserve budget:           3.5M-5M total tokens
Likely output/code tokens:             100k-400k tokens
Most tokens:                           input/tool context from reading and comparing code
```

The exact dollar cost depends on the GPT-5.5 pricing available in the environment:

```text
cost = input_tokens / 1M * input_price_per_1M
     + output_tokens / 1M * output_price_per_1M
```

Because the work requires repeatedly reading branch-specific code, generated artifacts, prompts, and tests, input tokens dominate the budget. A realistic first pass should be budgeted as a solid production refactor, not as a minimal implementation.

### Cursor Pricing Interpretation

When using Cursor pricing, the dollar estimate should be read as **Cursor API usage consumed**, not necessarily as an extra out-of-pocket bill. If the plan still has included usage remaining, the task consumes that allowance. If included usage is exhausted and on-demand usage is enabled, it can become a real pay-as-you-go charge.

For Cursor GPT-5.5 planning, use these approximate rates:

```text
Input tokens:        $5 / 1M
Cached input tokens: $0.50 / 1M
Output tokens:       $30 / 1M
```

The rough calculation assumes the merge/refactor is input-heavy because the agent must repeatedly read code, prompts, generated artifacts, and tests across both branches.

| Scenario | Token Assumption | Calculation | Rough Usage |
| --- | --- | --- | ---: |
| Adapter-first, no major redo | `2.0M input + 200k output`, with about half of input effectively cached/discounted | `(1.0M * $5) + (1.0M * $0.50) + (0.20M * $30)` | `~$12` |
| Adapter-first plus review/fix buffer | additional targeted review, smoke tests, and small fixes | previous row plus rounded buffer for missed files and small regressions | `~$18-$22` |
| Expected careful pass with some shared-code redo | `3.0M input + 350k output`, with about half of input effectively cached/discounted | `(1.5M * $5) + (1.5M * $0.50) + (0.35M * $30)` | `~$19` |
| Expected pass plus review/fix buffer | additional long-context/code-review iterations | previous row plus rounded buffer for branch conflicts, missed files, and test fixes | `~$26` |
| Redo shared pipeline parts more deeply | `4.0M input + 400k output`, partly cached | `(2.5M * $5) + (1.5M * $0.50) + (0.40M * $30)` plus larger redo/test buffer | `~$35` |
| Conservative ceiling if repeated redo happens | `4.5M input + 350k output`, mostly uncached | `(4.5M * $5) + (0.35M * $30)` plus small safety buffer | `~$40` |

Max Mode, long-context behavior, Teams token-rate rules, model routing, and on-demand settings can change the final bill. If using `Auto`, raw GPT-5.5 token pricing may not apply directly.

Final planning numbers:

| Estimate | Rough Cursor GPT-5.5 API Usage |
| --- | ---: |
| Average expected merge/refactor | `~$26` |
| Conservative max budget | `~$40` |

### Cost-Reduction Instructions

To keep the merge closer to the average estimate instead of the max estimate:

1. Do a design-only pass first and freeze the base interfaces before major edits.
2. Prefer an adapter-first merge: wrap existing `chempatents` and `JRC-Acquis` behavior behind shared interfaces before rewriting internals.
3. Avoid full-repo rereads. Use this document as the working map and read only the files needed for the current phase.
4. Split work into small phases: schemas and paths, source adapters, QA source selection, QAC generation, CLI, tests.
5. Use GPT-5.5 for architecture decisions and cheaper/Auto routing for mechanical edits, formatting, simple renames, and documentation cleanup.
6. Use small fixtures and smoke tests instead of regenerating full datasets during the refactor.
7. Avoid Max Mode or long-context mode unless a phase is blocked by missing context.
8. Keep compatibility first. Do not redesign prompts, generation quality, or dataset semantics while also merging architecture.
9. Checkpoint after each phase so the agent does not re-explore or rework already-settled decisions.
10. Defer cleanup of duplicated legacy functions until both source adapters pass their smoke tests.

The cheapest safe path is: adapter-first unification, minimal behavior changes, targeted tests, then a second cleanup pass only after both branches still run.

### Main Cost Drivers

- Reconciling branch differences in `cli.py`, `pipeline.py`, `config.py`, and path handling.
- Designing a shared source/corpus/QA-selection abstraction without breaking existing artifacts.
- Splitting source-specific logic from shared QAC generation logic in `openai_qa.py`.
- Preserving both generation modes: chemistry English-first/direct multilingual and JRC same-language legal generation.
- Adding domain profiles for chemistry and legal validators.
- Testing that old commands still write compatible outputs.
- Checking generated data after the refactor so query/corpus links still resolve.

### Practical Recommendation

Plan the implementation in phases:

1. Define shared schemas and interfaces.
2. Move each branch behind adapters with minimal behavior changes.
3. Unify CLI/config/path dispatch.
4. Unify QA source selection and QAC generation interfaces.
5. Add compatibility tests and smoke-run both sources.
6. Only then clean up duplicated legacy functions.

This reduces the chance of a surprise rewrite halfway through the merge.

