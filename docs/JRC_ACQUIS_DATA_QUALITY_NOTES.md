# JRC-Acquis Data Quality Notes

This note summarizes the current findings from the document-level JRC-Acquis pipeline after the latest `--build-corpus JRC-ACQUIS` run.

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

## Corpus Scale

From the latest `document_corpus_stats.json`:

- Total document-language rows: `462,554`
- Total CELEX ids: `116,142`
- Multilingual CELEX ids: `18,033`
- Multilingual document rows: `364,445`
- All cross-language document pairs: `3,520,894`
- QA candidates: `169,169`
- Languages covered: `22`

These numbers indicate that the corpus is large enough for a strong document-level cross-lingual retrieval benchmark.

## Pairing Structure

Pairing is based on shared `celex`, so the current benchmark view is document-aligned across languages rather than paragraph-aligned.

- The full language-pair matrix is documented in `docs/JRC_ACQUIS_LANGUAGE_PAIR_COUNTS.md`.
- The pairing table is stored in `data/JRC-ACQUIS/preprocessed/document_pairs_all.csv`.

This is a good first benchmark structure because it is robust and simple, even though it is coarser than paragraph-level alignment.

## Current Preprocessing Behavior

The current JRC preprocessing does the following:

- normalizes whitespace
- removes formatting artifacts such as `++++`
- removes artifact/header lines such as `[pic] ...` and `*****`
- removes standalone reference-note lines like `(1) OJ ...`
- derives cleaner document titles from early body paragraphs
- attempts to trim long legal preambles and start from the operative section, usually at `Article 1` or the local equivalent
- builds multilingual-only and QA-candidate subsets

Important design note:

- Earlier versions used a large hand-written set of language-specific preamble phrases.
- That was replaced with more generic structural heuristics.
- The remaining language-aware part is mainly article-heading detection across legal-writing variants such as `Article`, `Artikel`, `Člen`, `Член`, `Artigo`, `Articolo`, `Artykuł`, `1 straipsnis`, `1 artikla`, etc.

This remaining language-aware logic is still generic legal-structure matching, not document-specific matching.

## Positive Quality Findings

The latest build looks materially better than earlier versions.

- Documents with formatting cleaned: `149,975`
- Documents trimmed to operative body: `273,731`
- Generic fallback titles: `0`
- Leading `++++` artifacts in QA candidates: `0`
- Leading `[pic]` / `*****` artifact lines in QA candidates: `0`
- Leading reference-note lines in QA candidates: `0`

QA-candidate length summary:

- Median characters: `4,238`
- 90th percentile characters: `18,754`
- Median body paragraphs: `32`
- 90th percentile body paragraphs: `105`

These figures are in a usable range for LLM-based question generation.

## Sample-Level Observations

Many cleaned samples now start directly with operative content such as:

- `Article 1`
- `Artikel 1`
- `Člen 1`
- `Článek 1`
- `Член 1`
- `1 straipsnis`
- `1 artikla`

This is a major improvement over earlier versions that often started with long institutional boilerplate.

Examples that now trim well include Dutch, Slovene, Swedish, Portuguese, Danish, German, French, Czech, and Bulgarian documents.

## Remaining Quality Issues

The corpus is usable, but not fully uniform across languages.

### 1. Many QA candidates still do not start at the operative section

From the latest QA-candidate scan:

- QA candidates starting with an article heading: `93,650 / 169,169`

So only about `55%` clearly start from the operative article section. The remaining roughly `45%` still begin with title-style or preamble-style content.

### 2. Residual preamble/title starts remain language-skewed

Languages with especially high counts of non-article starts include:

- `lv`
- `sk`
- `hu`
- `mt`
- `pt`

Observed non-article starts include things like:

- regulation titles
- decision titles
- common positions
- committee or council decisions
- opinions and interinstitutional texts

This means the current trimming logic still misses some legal-act formats.

### 3. QA-candidate retention is uneven by language

Retention rate = `qa_candidates_by_language / multilingual_docs_by_language`

Lowest retention:

- `cs`: `34.0%`
- `sl`: `36.3%`
- `lt`: `37.6%`
- `da`: `37.9%`
- `sv`: `38.3%`

Highest retention:

- `mt`: `87.1%`
- `lv`: `82.6%`
- `sk`: `76.0%`
- `hu`: `73.9%`

This indicates that the current filters do not behave evenly across languages.

### 4. Some titles are still weak

- Short titles remaining in QA candidates: `880`

This is not a major issue, but it suggests some rows still have imperfect title extraction.

## Current Quality Assessment

Overall assessment:

- Alignment quality: strong
- Corpus cleanliness: good
- Generic preprocessing quality: clearly improved
- QA readiness: usable, but uneven
- Cross-lingual benchmark potential: high

Practical interpretation:

- The current JRC data is good enough to proceed with a first document-level benchmark.
- It is not yet ideal if the goal is maximally content-focused question generation across all languages.

## Recommended Next Improvements

Before large-scale JRC question generation, the most useful remaining preprocessing work is:

1. Improve generic detection of operative starts for additional legal-act formats beyond standard article-based openings.
2. Reduce residual title/preamble starts in the languages with the highest miss rates.
3. Recheck the QA-candidate balance by language after each preprocessing refinement.

## Recommendation For Benchmark Construction

The current best path is:

1. Keep the document-level benchmark structure based on shared `celex`.
2. Generate questions only from `corpus_qa_candidates.csv`.
3. Use documents with the same `celex` in the other languages as relevant documents.
4. Decide later whether same-language documents should also remain in qrels or whether evaluation should be strictly cross-lingual.

## Bottom Line

The JRC-Acquis preprocessing is now in a good intermediate state:

- large
- aligned
- mostly clean
- much better than the initial raw-document build

The benchmark can already move forward from here, but one more generic preprocessing refinement would likely improve the eventual question quality and make language behavior more consistent.
