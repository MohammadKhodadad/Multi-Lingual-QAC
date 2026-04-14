# Multi-Lingual Chemical QAC

Build multi-lingual Question–Answer–Context (QAC) data from chemistry patents (EPO) and chemistry Wikipedia pages (Wikidata) for Hugging Face and MTEB.

## Setup

```bash
uv sync
```

Add `.env`:
```
OPENAI_API_KEY=your-openai-api-key
HF_TOKEN=your-huggingface-token
```

- Put EPO source package folders under `data/EPO/input`
- For Q&A generation: `OPENAI_API_KEY` (OpenAI API)
- For Hugging Face push: `HF_TOKEN` (create at huggingface.co/settings/tokens)

## Usage

```bash
uv run main.py --prepare-source EPO         # extract XML files from EPO zips into data/EPO/xmls
uv run main.py --build-corpus EPO           # parse EPO XMLs, filter chemistry-related documents, create corpus files
uv run main.py --build-corpus EPO --build-corpus-batch   # same, but with half the available CPUs, capped at 5
uv run main.py --source epo --yes --qa-sample 0
uv run main.py --source epo --qa-sample 50
uv run main.py --source epo --qa-sample 50 --qa-batch
uv run main.py --source epo --push-hf --hf-repo username/multi-lingual-chemical-qac
```

### Wikidata / Wikipedia (chemistry pages)

```bash
uv run main.py --prepare-source WIKIDATA              # Wikidata + multilingual Wikipedia fetch → data/WIKIDATA/prepared/
uv run main.py --build-corpus WIKIDATA                # chunk page extracts → corpus_full.csv + corpus.csv
uv run main.py --source wikidata --qa-sample 50       # English Q&A + translate (one row per language per sampled chunk)
uv run main.py --label-qrels WIKIDATA                 # LLM judge: qrels + queries (gpt-5-nano)
# Optional: uv run main.py --label-qrels WIKIDATA --label-qrels-batch-size 8
```

### JRC-Acquis (legal / regulatory corpus)

```bash
uv run main.py --prepare-source JRC-ACQUIS      # download / extract / parse raw JRC-Acquis XML archives
uv run main.py --build-corpus JRC-ACQUIS        # build document corpus, multilingual subsets, and QA candidates
uv run main.py --source JRC-ACQUIS              # interactive CELEX-group-based legal QA generation from sampled target-side documents
```

The JRC retrieval corpus keeps body text plus a capped annex slice and excludes signature tail text from the main retrieval `context`.

Typical interactive JRC QA review run:

```bash
uv run main.py --source JRC-ACQUIS
# Example answers:
# directional language pairs per source language: 100
# sampled source documents per language: 2
# batch create QAs: y
# regenerate QAC: y
# push to Hugging Face: n
```

JRC uses a CELEX-group-based legal QA flow:

- group multilingual documents by `celex`
- sample source-side documents from multilingual `celex` groups
- choose one target-language document realization as the generation text
- generate the query in that target-side language
- connect the final query to all retained sampled documents for the same `celex`

### MTEB benchmark and report generation

Run the benchmark with the built-in default multilingual model set:

```bash
uv run main.py --evaluate-mteb
```

Default benchmark settings:

- dataset: `MohammadKhodadad/multi-lingual-qac`
- models:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - `intfloat/multilingual-e5-large`
  - `BAAI/bge-m3`
- model cache folder: `.cache/huggingface`
- output folder: `reports/mteb`

Run the benchmark with specific models:

```bash
uv run main.py --evaluate-mteb sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-m3
```

Override the dataset repo or output folder:

```bash
uv run main.py --evaluate-mteb --mteb-dataset-repo MohammadKhodadad/multi-lingual-qac --mteb-output-dir reports/my_mteb_run
```

After the benchmark finishes, generate comparison tables from the saved results without rerunning evaluation:

```bash
uv run main.py --generate-mteb-tables
```

That reads from `reports/mteb` and writes comparison tables to `reports/mteb_tables`.

After local generation, the CLI can also ask whether you want to upload the generated files to a Hugging Face dataset repo under `benchmark_outputs/mteb_tables/`. The prompt accepts either a dataset repo ID such as `MohammadKhodadad/multi-lingual-qac` or a full dataset URL.

If uploaded, the dataset `README.md` on Hugging Face is also refreshed with a `## Leaderboard` section built from the latest `model_comparison.md`. This section is replaced in place on later uploads rather than appended repeatedly.

If a long benchmark run is still in progress or stopped early, table generation can also fall back to the raw per-model MTEB result JSON files already written under `reports/mteb/**`.

Point comparison-table generation at a different benchmark run if needed:

```bash
uv run main.py --generate-mteb-tables --mteb-results-dir reports/my_mteb_run --mteb-tables-dir reports/my_mteb_tables
```

Benchmark outputs:

- `reports/mteb/summary.json`
- `reports/mteb/summary.csv`
- `reports/mteb/summary.md`

Comparison-table outputs:

- `reports/mteb_tables/model_comparison.json`
- `reports/mteb_tables/model_comparison.csv`
- `reports/mteb_tables/model_comparison.md`
- `reports/mteb_tables/model_comparison.tex`
- Hugging Face dataset artifact path: `benchmark_outputs/mteb_tables/`

Current partial CPU results from the latest run:

- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`: `ndcg_at_10 = 0.2062`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: `ndcg_at_10 = 0.1340`

Suggested next steps:

- finish the full default model sweep so the comparison table includes `intfloat/multilingual-e5-large` and `BAAI/bge-m3`
- rerun the strongest models on GPU hardware for faster turnaround and cleaner large-model coverage
- record hardware details for each benchmark batch (CPU vs GPU, GPU type, batch size) alongside the reports
- add a second comparison run focused on larger GPU-oriented multilingual encoders after the default baseline sweep is stable
- optionally add incremental summary writing per completed model so long runs always leave a top-level `summary.json`

Recommended flow:

```bash
uv run main.py --evaluate-mteb
uv run main.py --generate-mteb-tables
```

Use `--source wikidata` for the main run so paths point at `data/WIKIDATA/`. Q&A generation matches the EPO flow: **English** question/answer from the sampled corpus context, then **translation** to the configured target languages—so `qac.csv` has **one row per language** per sampled source chunk (`corpus_id` ties the translations together).

**`--label-qrels WIKIDATA`** (see `qac_generation/label_wikidata_qrels.py`):

- **Inputs:** `preprocessed/corpus_full.csv` (chunk-level rows with `id`, `qid`, `context`, `language`) and `qac/qac.csv`.
- **Grouping:** one logical Q&A per `corpus_id`; all languages in `qac` share one **merged** relevance set over corpus rows with the same Wikidata **`qid`** (other chunks of the same page, other languages, etc.). The **source** chunk is always counted relevant; the LLM judges only **siblings** (batched by passage language).
- **Judge model:** **`gpt-5-nano`** (independent of Q&A models in `openai_qa.py`). Optional: `--label-qrels-batch-size` (passages per API call, default 5).
- **Outputs:**
  - **`qac/queries.csv`** — one row per `(corpus_id, language)` in `qac`: `_id` = `{corpus_id}_q_{lang}`, `text` = that language’s question.
  - **`qac/qrels.csv`** — TREC-style rows: `query-id`, `corpus-id`, `score` (always `1.0`); the **same** relevant document ids are repeated for **every** query-id of that chunk (per-language queries, shared gold pool).
  - **`qac/qrels_label_stats.json`** — row counts and `judge_api_calls`.

`--prepare-source EPO` is separate from the main pipeline on purpose. Normal `main.py` runs do not unpack source zip files by default, which makes it easier to support multiple patent sources such as EPO and USPTO later.

`--build-corpus EPO` is also explicit. It reads `data/EPO/xmls`, extracts useful bibliographic metadata, scores chemistry relevance from CPC/IPC codes plus title keywords, writes all parsed rows to `data/EPO/preprocessed/all_epo_records.csv`, and writes chemistry-focused rows to `data/EPO/corpus.csv`.

At the end of an interactive run, the CLI can ask whether to batch-create QAs using available CPUs. If the corpus and QAC files are ready, it can also ask whether you want to push to Hugging Face and then ask for the repo ID.

By default, Q&A generation uses **`gpt-5-mini`** (and configured reasoning effort) in `openai_qa.py` for English generation, checks, and translation. **Wikidata qrels labeling** uses **`gpt-5-nano`** only for the relevance judge (`label_wikidata_qrels.py`).

## Code Structure

The project now uses a structured package under `src/multi_lingual_qac/`.

```text
src/
├── multi_lingual_qac/
│   ├── cli.py                  # argparse + env loading
│   ├── config.py               # shared paths and pipeline config
│   ├── pipeline.py             # main orchestration flow
│   ├── dataloaders/
│   │   ├── epo.py              # EPO zip scanning + XML extraction + corpus build
│   │   ├── wikidata.py         # Wikidata/Wikipedia fetch + chunk corpus build
│   │   └── wikipedia_clean.py  # Multilingual extract cleanup + sentence-aware chunking
│   ├── qac_generation/
│   │   ├── openai_qa.py              # Q&A generation (EN + translate; optional same-language mode)
│   │   └── label_wikidata_qrels.py   # Wikidata: multilingual qrels + queries (gpt-5-nano judge)
│   ├── export/
│   │   └── hf_upload.py        # Hugging Face / MTEB upload
│   ├── mteb/
│   │   ├── __init__.py         # MTEB entrypoints / exports
│   │   └── evaluation.py       # benchmark execution + comparison table generation
│   └── preprocess/
│       └── corpus.py           # source-specific corpus preparation helpers
```

`main.py` is now a thin entrypoint that calls `src.multi_lingual_qac.cli:main`.

## Data flow

1. **Source prep** → `uv run main.py --prepare-source EPO`
2. **EPO zip extraction** → `data/EPO/input/...` to `data/EPO/xmls/*.xml`
3. **Corpus build** → `uv run main.py --build-corpus EPO`
4. **Parsed records** → `data/EPO/preprocessed/all_epo_records.csv`
5. **Chemistry corpus (full)** → `data/EPO/preprocessed/corpus_full.csv`
6. **Chemistry corpus (MTEB retrieval format)** → `data/EPO/corpus.csv`
7. **Q&A generation** → `data/EPO/qac/qac.csv`
8. **Push to Hugging Face** → dataset with configs/subsets: corpus, queries, qrels, qac, each with a `train` split (MTEB retrieval format)

The extracted XML directory is intended to be the source-local raw cache. It keeps the unpacked patent XMLs so later corpus experiments can be done without repeatedly opening the original zip files.

### Wikidata data flow

1. **`--prepare-source WIKIDATA`** → `data/WIKIDATA/prepared/` (entities, page JSONL, coverage).
2. **`--build-corpus WIKIDATA`** → `preprocessed/corpus_full.csv` (full chunks + metadata) and **`corpus.csv`** (MTEB: `_id`, `title`, `text`). Extracts are cleaned per language (`dataloaders/wikipedia_clean.py`) then sentence-aware chunked (`chunk_plain_text_multilingual`).
3. **`--source wikidata --qa-sample N`** → `qac/qac.csv` (English + translations; one row per language per sampled chunk).
4. **`--label-qrels WIKIDATA`** → `qac/queries.csv`, `qac/qrels.csv`, `qrels_label_stats.json`.

Re-run **step 4** alone if you change labeling logic only; re-run **3** (and **4**) if `qac.csv` changes; re-run **2** onward if chunking or corpus content changes.

### JRC-Acquis (legal / regulatory corpus)

JRC now uses a **CELEX-group-based** QA flow:

1. group multilingual legal documents by `celex`
2. sample a broader source-side pool per language from multilingual `celex` groups
3. retain a smaller question-generation set per language from that sampled pool
4. build the final retrieval corpus from the sampled source pool plus all same-`celex` document realizations for the selected question-generation set
5. choose one target-language document realization from each selected generation group
6. generate one same-language legal question/answer from that target-side text
7. validate it with language, faithfulness, retrieval-quality, and legal-shape checks
8. connect the resulting query to all documents in the final retrieval corpus for the same `celex`

The generation/checking prompts for JRC are domain-specific: legal/regulatory rather than chemistry/patent. The current prompt stack is intentionally separated into generation, faithfulness, retrieval-quality, and legal-shape stages so that provision-led wording, status/label questions, content-list questions, and multi-part legal questions can be filtered with targeted feedback rather than one monolithic blacklist.

See `docs/QA_GENERATION_PROCESS.md` for the full current QA-generation flow across EPO, Wikidata, and JRC.

### Q&A generation (Option A)

- Samples corpus (stratified by language)
- Generates (question, answer) in **English** via OpenAI per document
- Runs an English language check and a faithfulness check before keeping each QA pair
- Runs a retrieval-quality check to reject overly generic or weak questions
- Translates Q&A to all target languages (de, fr, es, ja, ko, zh, ru, pt, it, nl, ar, fa, tr, pl, hi)
- Output: `corpus_id`, `language`, `question`, `answer` (one row per document per language)

### Current EPO status

- EPO source preparation is implemented.
- The XML extraction step currently pulls all `.xml` members from each patent zip and stores them with collision-safe names.
- The corpus-build step parses non-TOC patent XMLs, keeps useful metadata, writes a rich chemistry corpus to `data/EPO/preprocessed/corpus_full.csv`, and writes an MTEB-style retrieval corpus to `data/EPO/corpus.csv`.
- Chemistry detection currently uses broad CPC/IPC prefix matching plus title keyword matching.

## Plan

See [docs/MULTILINGUAL_CHEMICAL_QAC_PLAN.md](docs/MULTILINGUAL_CHEMICAL_QAC_PLAN.md) for the full pipeline (WIPO, Lens, Google Patents -> QAC generation -> HF/MTEB).
