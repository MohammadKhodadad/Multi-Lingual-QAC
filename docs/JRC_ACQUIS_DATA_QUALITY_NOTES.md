# JRC-Acquis Data Quality Notes

This note summarizes the current findings from the document-level JRC-Acquis pipeline after the latest `--build-corpus JRC-ACQUIS` run and the follow-up QA review.

## Current Outputs

Main generated files:

- `data/JRC-ACQUIS/corpus.csv`
- `data/JRC-ACQUIS/preprocessed/corpus_full.csv`
- `data/JRC-ACQUIS/preprocessed/document_pairs_all.csv`
- `data/JRC-ACQUIS/preprocessed/corpus_multilingual.csv`
- `data/JRC-ACQUIS/preprocessed/corpus_multilingual_full.csv`
- `data/JRC-ACQUIS/preprocessed/corpus_qa_candidates.csv`
- `data/JRC-ACQUIS/preprocessed/inspection_sample.csv`
- `data/JRC-ACQUIS/preprocessed/document_corpus_stats.json`
- `docs/JRC_ACQUIS_LANGUAGE_PAIR_COUNTS.md`

## Current Corpus Scale

From the latest `document_corpus_stats.json`:

- Total document-language rows: `462,532`
- Total CELEX ids: `116,120`
- Multilingual CELEX ids: `18,033`
- Multilingual document rows: `364,445`
- All undirected CELEX document pairs: `3,520,894`
- QA candidates after the stricter substance filter: `159,480`
- Languages covered: `22`

These numbers still support a strong document-level cross-lingual benchmark even after the stricter filtering pass.

## Pairing Structure

Pairing is still based on shared `celex`, so the benchmark remains document-aligned across languages rather than paragraph-aligned.

- The base pairing table is stored in `data/JRC-ACQUIS/preprocessed/document_pairs_all.csv`.
- `docs/JRC_ACQUIS_LANGUAGE_PAIR_COUNTS.md` now tracks the **QA-based directional matrix** used for dataset building, not the older symmetric raw pair matrix.

This is still a coarse structure, but it is stable and easy to audit.

## Current Preprocessing Behavior

The current JRC preprocessing now does the following:

- normalizes whitespace
- removes formatting artifacts such as `++++`
- removes artifact/header lines such as `[pic] ...` and `*****`
- removes standalone reference-note lines like `(1) OJ ...`
- derives cleaner document titles from early body paragraphs
- extracts and classifies text into body / annex / signature zones
- drops `jrcHeader-*` style helper/header pseudo-documents
- keeps `header_notes` only as metadata instead of prepending them into retrieval text
- builds longer retrieval text with a structured cap:
  - body budget: `8000` chars
  - annex budget: `2000` chars
  - signature budget: `800` chars
  - total retrieval cap: `12000` chars
- builds a stricter QA-candidate subset using multilingual-safe paragraph/substance heuristics

Important design note:

- Earlier versions were too aggressive about trimming to `Article 1`.
- The current version keeps richer retrieval text while using a separate `generation_context` for QA.
- The current QA-candidate gate is now driven more by paragraph/substance signals than by shallow document length alone.

## What Improved

The latest build is materially cleaner than the earlier JRC corpus state.

### Clean retrieval text

- `jrcHeader-*` pseudo-documents are no longer present in `corpus_full.csv`
- the repeated authenticity / translation disclaimer text is no longer present in retrieval `corpus.csv`
- retrieval text now starts from real document content instead of note boilerplate

### Cleaner structure

- Documents with formatting cleaned: `149,975`
- Documents trimmed to operative body: `273,731`
- Documents over `30,000` chars: `0`
- Documents under `1,500` chars: `38,139`

### Better QA source pool

- The QA pool is now more selective: `159,480` candidates instead of the earlier broader pool
- The stricter filter is removing many obviously weak operative structures before QA sampling

## Corpus Quality Scorecard

### Compared to the earlier corpus state

Before the recent cleanup pass:

- Header/helper pseudo-docs still leaked into the corpus
- note/disclaimer boilerplate still appeared at the start of many retrieval documents
- retrieval text selection was less disciplined
- QA candidate filtering was looser and more likely to admit low-substance documents

Current state:

- header/helper leakage fixed
- retrieval boilerplate leakage fixed
- retrieval text is richer but capped
- QA candidate filtering is stricter and more substance-aware

### Scores

Current corpus quality score by perspective:

- Overall corpus quality now: `8.0 / 10`
- Previous corpus quality before the recent cleanup: `6.5 / 10`
- Retrieval text cleanliness now: `8.5 / 10`
- Retrieval text cleanliness before: `5.5 / 10`
- Structural alignment quality now: `8.5 / 10`
- Structural alignment quality before: `8.5 / 10`
- QA-source suitability now: `7.0 / 10`
- QA-source suitability before: `6.0 / 10`
- Cross-lingual benchmark usefulness now: `8.5 / 10`
- Cross-lingual benchmark usefulness before: `7.5 / 10`
- Language-uniformity of filtering now: `5.5 / 10`
- Language-uniformity of filtering before: `6.5 / 10`

Interpretation:

- The corpus is clearly better than before.
- The biggest gain is in retrieval cleanliness and removal of non-substantive junk.
- The main new downside is that the stricter QA filter is uneven across languages.

## Current Quality Assessment

Overall assessment:

- Alignment quality: strong
- Corpus cleanliness: clearly improved
- Retrieval corpus quality: good
- QA readiness: usable, but now visibly more selective and uneven by language
- Cross-lingual benchmark potential: high

Practical interpretation:

- The current JRC corpus is good enough to continue benchmark construction.
- It is meaningfully better than the earlier version for retrieval.
- The main remaining weakness is not raw corpus dirt anymore; it is QA-pool balance and downstream question sharpness.

## Current Main Risks

### 1. QA-candidate retention is now uneven by language

Retention rate = `qa_candidates_by_language / multilingual_docs_by_language`

Lowest retention:

- `cs`: `31.8%`
- `et`: `33.3%`
- `sl`: `34.7%`
- `lt`: `36.0%`
- `da`: `37.5%`

Highest retention:

- `mt`: `82.5%`
- `sk`: `65.3%`
- `lv`: `61.9%`
- `hu`: `59.0%`
- `pt`: `50.8%`

This indicates that the current substance thresholds are still interacting with language formatting differences, not just document quality.

### 2. The new QA gate is stricter, but probably too harsh for some languages

Current QA rejection reasons:

- `too_few_medium_operative_paragraphs`: `179,056`
- `too_few_operative_chars`: `159,860`
- `too_many_short_operative_paragraphs`: `117,753`
- `too_short`: `6,121`
- `too_few_body_paragraphs`: `821`

This means the new heuristic is mostly acting on paragraph-shape and operative-density, not on trivial short-document rejection.

### 3. Corpus quality improved more than QA quality

The corpus-side cleanup worked well, but the follow-up QA run still produced some dual-part, list-shaped, or value/procedure-heavy questions. So the current preprocessing/filtering improvement should be judged mainly as a **corpus-quality win**, not as a complete fix for QA quality.

## Recommended Next Improvements

The most useful next steps are now:

1. Tune the multilingual substance thresholds so QA retention is less skewed across languages.
2. Keep the current retrieval cleanup in place; it appears to be the right direction.
3. Tighten question-generation prompts and validation against:
   - dual-part questions
   - checklist / inventory questions
   - timing-only or value-only lookup questions
4. Recheck per-language QA retention after each threshold change.

## Recommendation For Benchmark Construction

The current best path is:

1. Keep the document-level benchmark structure based on shared `celex`.
2. Build the dataset from the stricter QA candidate pool.
3. Use the QA-based directional pair matrix in `docs/JRC_ACQUIS_LANGUAGE_PAIR_COUNTS.md` as the relevant planning table.
4. Continue using the sampled QA subset for JRC benchmark construction, while treating the current substance filter as a tunable component rather than final policy.

## Bottom Line

The JRC-Acquis corpus is now in a clearly better state than before:

- still large
- still aligned
- cleaner for retrieval
- less polluted by helper/header artifacts
- more selective for QA generation

Compared with the earlier corpus state, the current build is a real improvement and deserves a higher corpus-quality score.

The main remaining issue is now **not corpus dirt**, but **how evenly the stricter QA-source filter behaves across the 22 languages** and whether the improved pool yields consistently sharper legal questions.
