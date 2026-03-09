# Multi-Lingual Chemical QAC

Build multi-lingual Question–Answer–Context (QAC) data from chemistry patents for Hugging Face and MTEB.

## Setup

```bash
uv sync
```

Add `.env`:
```
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
```

For Google Patents extraction: `gcloud auth application-default login`

## Usage

```bash
uv run main.py              # interactive: asks redo for extraction and each language
uv run main.py --yes        # no prompts; redo all
uv run main.py --no-extraction   # skip extraction; only preprocess
uv run main.py --limit 100  # 100 per language (en, de, fr, ...) into one NDJSON
```

## Data flow

1. **Extraction** → `data/google_patents/chemistry_patents.ndjson` (BigQuery)
2. **Preprocessing** → `data/google_patents/preprocessed/{en,de,fr,...}.csv`
3. Each CSV has: `id`, `language`, `title`, `abstract`, `context`, `publication_number`, `country_code`, `publication_date`, `source`

## Plan

See [MULTILINGUAL_CHEMICAL_QAC_PLAN.md](MULTILINGUAL_CHEMICAL_QAC_PLAN.md) for the full pipeline (WIPO, Lens, Google Patents → QAC generation → HF/MTEB).
