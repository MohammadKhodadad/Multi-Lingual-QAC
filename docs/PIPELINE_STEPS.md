# Pipeline Steps Done So Far

Short recap of every step implemented in the Multi-Lingual Chemical QAC pipeline.

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
