# Multi-Lingual Chemical QAC

Build multi-lingual Question–Answer–Context (QAC) data from chemistry patents for Hugging Face and MTEB.

## Data source and license

- **Source dataset:** Patent text (titles, abstracts) in this project is derived from **Google Patents Public Data** on BigQuery (`patents-public-data.patents.publications`), provided by IFI CLAIMS Patent Services and Google. See [Marketplace](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data) and [announcement](https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data).
- **License:** That source data is made available under [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0).
- **This project’s data:** The corpus, questions, and answers produced by this pipeline (including all Q&A pairs and translations) form a **derived/adapted dataset** based on that source.
- **No endorsement:** This dataset is not affiliated with, endorsed by, or officially connected with Google or IFI CLAIMS. Only the underlying patent publication text is from that source; the pipeline, Q&A generation, and benchmark design are independent.
- **Scope:** Attribution and license refer only to the patent dataset content used here (bibliographic and abstract text from the public BigQuery tables). They do not cover other Google services, products, or UI content.

## Setup

```bash
uv sync
```

Add `.env`:
```
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
OPENAI_API_KEY=your-openai-api-key
HF_TOKEN=your-huggingface-token
```

- For Google Patents extraction: `gcloud auth application-default login`
- For Q&A generation: `OPENAI_API_KEY` (OpenAI API)
- For Hugging Face push: `HF_TOKEN` (create at huggingface.co/settings/tokens)

## Usage

```bash
uv run main.py                  # interactive: asks for limit, Q&A sample size, batch mode, and optional HF push at the end
uv run main.py --yes            # no prompts; redo all, uses fallback defaults
uv run main.py --no-extraction  # skip extraction; only preprocess and corpus merge
uv run main.py --limit 100      # 100 per language (en, de, fr, ...) into one NDJSON
uv run main.py --qa-sample 50   # generate Q&A for 50 sampled corpus documents; use 0 to skip
uv run main.py --qa-sample 50 --qa-batch      # batch Q&A generation using available CPUs
uv run main.py --qa-sample 50 --qa-no-batch   # force single-threaded Q&A generation
uv run main.py --push-hf --hf-repo username/multi-lingual-chemical-qac   # push to Hugging Face
```

At the end of an interactive run, the CLI can ask whether to batch-create QAs using available CPUs. If the corpus and QAC files are ready, it can also ask whether you want to push to Hugging Face and then ask for the repo ID.

By default, Q&A generation now uses a stronger OpenAI model for English question creation and quality judging, while keeping translation and lighter validation checks on a cheaper model.

## Code Structure

The project now uses a structured package under `src/multi_lingual_qac/`.

```text
src/
├── multi_lingual_qac/
│   ├── cli.py                  # argparse + env loading
│   ├── config.py               # shared paths and pipeline config
│   ├── pipeline.py             # main orchestration flow
│   ├── dataloaders/
│   │   └── google_patents.py   # extraction + preprocessing + corpus merge
│   ├── qac_generation/
│   │   └── openai_qa.py        # Q&A generation + translation
│   ├── export/
│   │   └── hf_upload.py        # Hugging Face / MTEB upload
│   └── preprocess/
│       └── corpus.py           # corpus-related helpers
```

`main.py` is now a thin entrypoint that calls `src.multi_lingual_qac.cli:main`.

## Data flow

1. **Extraction** → `data/google_patents/chemistry_patents.ndjson` (BigQuery)
2. **Preprocessing** → `data/google_patents/preprocessed/{en,de,fr,...}.csv`
3. **Corpus merge** → `data/google_patents/corpus.csv` (all languages combined)
4. **Q&A generation** → `data/google_patents/qac/qac.csv`
5. **Push to Hugging Face** → dataset with configs/subsets: corpus, queries, qrels, qac, each with a `train` split (MTEB retrieval format)

### Q&A generation (Option A)

- Samples corpus (stratified by language)
- Generates (question, answer) in **English** via OpenAI per document
- Runs an English language check and a faithfulness check before keeping each QA pair
- Runs a retrieval-quality check to reject overly generic or weak questions
- Translates Q&A to all target languages (de, fr, es, ja, ko, zh, ru, pt, it, nl, ar, tr, pl, hi)
- Output: `corpus_id`, `language`, `question`, `answer` (one row per document per language)

### Corpus / preprocessed columns

- `id`, `language`, `title`, `abstract`, `context`, `publication_number`, `country_code`, `publication_date`, `source`

Preprocessing filters out low-information records before they enter the corpus: documents whose cleaned abstract is shorter than `50` words are skipped, which also removes title-only entries from downstream Q&A generation.

## Plan

See [docs/MULTILINGUAL_CHEMICAL_QAC_PLAN.md](docs/MULTILINGUAL_CHEMICAL_QAC_PLAN.md) for the full pipeline (WIPO, Lens, Google Patents -> QAC generation -> HF/MTEB).
