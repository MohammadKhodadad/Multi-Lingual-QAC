# Multi-Lingual Chemical QAC

Build multi-lingual Question–Answer–Context (QAC) data from chemistry patents for Hugging Face and MTEB.

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
uv run main.py                  # interactive: also asks for limit and Q&A sample size
uv run main.py --yes            # no prompts; redo all, uses fallback defaults
uv run main.py --no-extraction  # skip extraction; only preprocess and corpus merge
uv run main.py --limit 100      # 100 per language (en, de, fr, ...) into one NDJSON
uv run main.py --qa-sample 50   # generate Q&A for 50 sampled corpus documents; use 0 to skip
uv run main.py --push-hf --hf-repo username/multi-lingual-chemical-qac   # push to Hugging Face
```

If you use `--push-hf` in interactive mode without `--hf-repo`, the CLI will ask for the repo ID.

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
5. **Push to Hugging Face** → dataset with splits: corpus, queries, qrels, qac (MTEB retrieval format)

### Q&A generation (Option A)

- Samples corpus (stratified by language)
- Generates (question, answer) in **English** via OpenAI per document
- Translates Q&A to all target languages (de, fr, es, ja, ko, zh, ru, pt, it, nl, ar, tr, pl, hi)
- Output: `corpus_id`, `language`, `question`, `answer` (one row per document per language)

### Corpus / preprocessed columns

- `id`, `language`, `title`, `abstract`, `context`, `publication_number`, `country_code`, `publication_date`, `source`

## Plan

See [MULTILINGUAL_CHEMICAL_QAC_PLAN.md](MULTILINGUAL_CHEMICAL_QAC_PLAN.md) for the full pipeline (WIPO, Lens, Google Patents → QAC generation → HF/MTEB).
