# Pipeline Steps Done So Far

Short recap of every step implemented in the Multi-Lingual Chemical QAC pipeline.

## Current JRC Flow

```text
+---------------------------+
| Raw JRC-Acquis XML        |
| archives                  |
+---------------------------+
              |
              v
+---------------------------+
| Prepare source            |
| download / parse /        |
| normalize                 |
+---------------------------+
              |
              v
+---------------------------+
| Build document corpus     |
| corpus_full.csv /         |
| corpus.csv                |
+---------------------------+
              |
              v
+---------------------------+
| Build document pairs      |
| document_pairs_all.csv    |
| grouped by CELEX          |
+---------------------------+
              |
              v
+---------------------------+
| Filter QA candidates      |
| multilingual /            |
| quality gates             |
+---------------------------+
              |
              v
+---------------------------+
| Sample directional        |
| language pairs            |
+---------------------------+
              |
              v
+---------------------------+
| Select one pair per       |
| source document           |
+---------------------------+
              |
              v
+---------------------------+
| Use translated side as    |
| generation text           |
+---------------------------+
              |
              v
+---------------------------+
| Generate question /       |
| answer in same-language   |
| legal mode                |
+---------------------------+
              |
              v
+---------------------------+
| Validate                  |
| language / faithfulness / |
| legal quality /           |
| legal shape               |
+---------------------------+
       | yes                      ^ no
       v                          |
+---------------------------+     |
| Write qac.csv            |     |
+---------------------------+     |
              |                   |
              v                   |
+---------------------------+     |
| Link one query to both   |     |
| documents in the pair    |     |
+---------------------------+     |
              |                   |
              v                   |
+---------------------------+     |
| Export corpus / queries /|     |
| qrels / HF dataset       |     |
+---------------------------+     |
                                  |
                     +---------------------------+
                     | Retry with feedback       |
                     +---------------------------+
                                  |
                                  +--------------+
```

---

## 1. Plan
Created `docs/MULTILINGUAL_CHEMICAL_QAC_PLAN.md`: data gathering -> QAC generation -> multilingual setup -> cleaning -> export. Targets WIPO, Lens, Google Patents. Defines MTEB format (corpus, queries, qrels).

---

## 2. Project Layout
Set up `data/`, `src/multi_lingual_qac/`, and a thin `main.py` entrypoint. Structured code into `cli`, `config`, `pipeline`, `dataloaders`, `qac_generation`, `export`, and `preprocess`.

---

## 3. BigQuery Extraction
Query `patents-public-data.patents.publications` for chemistry patents using CPC/IPC prefixes and SureChEMBL. Writes NDJSON with multilingual title/abstract.

---

## 4. Per-Language Extraction
`extract_chemistry_patents_per_language()` pulls up to N patents per language into one NDJSON. Each language gets its own BigQuery query with `primary_lang` filter.

---

## 5. Preprocessing (NDJSON -> CSV)
`preprocess_ndjson_to_csv()`: extract title/abstract per language, dedupe by publication number, apply `clean_text` (HTML decode + whitespace collapse), skip records whose cleaned abstract is under 50 words, then write `data/google_patents/preprocessed/{lang}.csv`.

---

## 6. Text Cleaning
`clean_text()` decodes HTML entities (`html.unescape`) and normalizes whitespace. Applied to title, abstract, and context in preprocessing and merge.

---

## 7. Corpus Merge
`merge_corpus_csv()` merges all per-language CSVs into one `data/google_patents/corpus.csv`. Corpus = documents for retrieval; queries and answers come from later QAC generation.

---

## 8. Main Flow
`main.py` now calls the structured pipeline. Interactive runs can prompt for `limit`, `qa_sample`, whether to batch-create QAs using available CPUs, redo decisions, and an optional Hugging Face push after corpus and QAC generation complete.

## 9. Q&A Generation (Option A)
Sample corpus (stratified by language). Generate English retrieval-style Q&A via OpenAI either sequentially or in CPU-based batches, require language, faithfulness, and question-quality checks, then translate each approved pair with generic artifact/fluency/meaning validation, retry using failure feedback plus the previous failed translation, and use `medium` reasoning for translation generation.

## 10. Push to Hugging Face
`push_to_hub()` uploads corpus, queries, qrels, qac to HF. Requires HF_TOKEN. Works with `--push-hf --hf-repo username/dataset`, or interactive end-of-run prompting for push confirmation and repo ID.

---

## 11. Wikidata / Wikipedia (chemistry corpus)
- **`prepare_corpus_source` + `wikidata.py`**: SPARQL entity selection, sitelinks, fetch of multilingual Wikipedia extracts; artifacts under `data/WIKIDATA/prepared/`.
- **`build_wikidata_corpus`**: `wikipedia_clean.clean_wikipedia_text` (per-lang noise + section heuristics) → `chunk_plain_text_multilingual` → `data/WIKIDATA/preprocessed/corpus_full.csv` and `data/WIKIDATA/corpus.csv` (MTEB retrieval format).

## 12. Wikidata Q&A and qrels (MTEB-style)
- **Q&A** (`--source wikidata --qa-sample N`): same flow as EPO — English generation, checks, translation — via `openai_qa.run_qa_pipeline` with `same_language=False`. Output: `data/WIKIDATA/qac/qac.csv`.
- **Qrels + queries** (`--label-qrels WIKIDATA`): `label_wikidata_qrels.run_wikidata_qrels_labeling` groups `qac` by `corpus_id`, judges sibling chunks (same Wikidata `qid`) with **`gpt-5-nano`**, writes `qac/queries.csv` (one query id per language: `{corpus_id}_q_{lang}`) and `qac/qrels.csv` (shared relevant `corpus-id` set duplicated per query-id). Stats in `qrels_label_stats.json`.

## 13. JRC-Acquis Raw Loading and Document Corpus
- Added `dataloaders/jrc_acquis.py` for JRC-Acquis archive download, XML parsing, raw loading, and document-level corpus build.
- Built `data/JRC-ACQUIS/preprocessed/corpus_full.csv`, `corpus.csv`, multilingual subsets, QA-candidate subsets, inspection sample, and `document_pairs_all.csv`.
- Pairing is document-level and based on shared `celex`, so each pair is the same EU legal act in two languages.
- Added multi-CPU support for both `--prepare-source JRC-ACQUIS` and `--build-corpus JRC-ACQUIS`.

## 14. JRC-Acquis Preprocessing and Inspection
- Replaced brittle case-specific cleanup rules with more generic legal-structure cleanup and operative-section trimming.
- Added `docs/JRC_ACQUIS_LANGUAGE_PAIR_COUNTS.md` with the language-pair count matrix.
- Added `docs/JRC_ACQUIS_DATA_QUALITY_NOTES.md` with corpus-scale and preprocessing-quality review notes.

## 15. JRC Pair-Level Legal QA Generation
- Added `qac_generation/jrc_acquis.py` to prepare pair-level QA inputs from `document_pairs_all.csv`.
- Current JRC QA flow:
  1. sample directional language pairs
  2. select source QA candidates
  3. choose one sampled pair per selected source document
  4. generate the question from the translated/target side of the pair
  5. attach the resulting query to all retained sampled translations for the same `celex`
- JRC runs through `openai_qa.run_qa_pipeline(..., same_language=True, domain_hint="legal")`.
- The active JRC prompt/check path is legal/regulatory, not chemistry/patent.

## 16. JRC Legal Prompt and Quality Iteration
- Tightened in-language legal generation prompts to avoid article numbers, clause lookup, and copied phrasing.
- Tightened legal quality checking to reject:
  - article/provision label questions
  - broad summary questions
  - multi-condition checklist questions
  - deadline-only / list-only / amount-only lookup questions when a better semantic legal question exists
- Added stronger retry feedback so rejected questions are regenerated toward one narrower legal information need.
- Added visible per-attempt QA logging so failed language / faithfulness / quality checks can be seen in the terminal during JRC runs.
- Added a dedicated legal-shape checker after the general quality checker to reject broad legal-shape, condition-list, menu-of-measures, definition-inventory, date-value-lookup, and multi-branch question forms.
- Added Hugging Face export columns for `corpus_language` and `question_language` in the pushed dataset configs.
- Cleaned `openai_qa.py` so the same-language path is now explicitly JRC/legal-only in this branch and dead prompt branches were removed.

## 17. JRC QAC Review Notes
- Replaced the old chemistry-focused `docs/QAC_QUALITY_NOTES.md` with JRC-specific QA review notes.
- Current JRC review focuses on:
  - language correctness
  - faithfulness
  - article-label avoidance
  - broad/checklist-shaped legal questions
  - literal deadline/list/numeric lookup questions
- Latest review snapshot:
  - latest sample wrote `43/44` rows
  - latest sample covered `20` languages
  - average question length was about `139` chars
  - average answer length was about `186` chars
  - `17/44` generation units were visibly rejected and then accepted on retry
  - the new legal-shape checker visibly rejected several weak rows before acceptance
- Current conclusion: the JRC questions are better than the previous reviewed run, but the main remaining weakness is still accepted condition-list / inventory / procedural-limit legal questions rather than article-label phrasing.

## 18. MTEB Evaluation and Comparison Reports
- Added `src/multi_lingual_qac/mteb/` with a custom Hugging Face-backed MTEB retrieval task.
- Added `--evaluate-mteb` to `main.py` / `cli.py` so saved benchmark data can be evaluated directly from the command line.
- The default MTEB dataset is `MohammadKhodadad/multi-lingual-qac`.
- Running `uv run main.py --evaluate-mteb` with no model arguments now uses the built-in multilingual default model list:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - `intfloat/multilingual-e5-large`
  - `BAAI/bge-m3`
- Evaluation outputs are written to `reports/mteb/` with `summary.json`, `summary.csv`, and `summary.md`.
- MTEB model downloads are cached locally in `.cache/huggingface/` so repeated runs do not depend on the user-profile cache.
- Added a second reporting-only command, `uv run main.py --generate-mteb-tables`, which reads saved MTEB summaries and writes standalone comparison tables to `reports/mteb_tables/`.
- Table generation now also works when `summary.json` is missing by falling back to the raw per-model `*_retrieval.json` files already written inside `reports/mteb/`.
- After local table generation, the CLI can ask whether the generated benchmark report artifacts should also be uploaded to a dataset repo under `benchmark_outputs/mteb_tables/`.
- That upload now also refreshes a `## Leaderboard` section in the dataset `README.md`, replacing the previous generated leaderboard block instead of appending duplicates.
- Comparison outputs now include:
  - `model_comparison.json`
  - `model_comparison.csv`
  - `model_comparison.md`
  - `model_comparison.tex`
- Latest partial CPU comparison snapshot:
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`: `ndcg_at_10 = 0.2062`
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: `ndcg_at_10 = 0.1340`
- Next MTEB work package:
  - finish the default model sweep
  - rerun the strongest models on GPU hardware
  - record hardware metadata with benchmark outputs
  - extend the comparison set with larger GPU-oriented multilingual models once the baseline sweep is complete
- This separates expensive benchmark execution from lightweight report regeneration, so model tables can be rebuilt without rerunning embeddings.
