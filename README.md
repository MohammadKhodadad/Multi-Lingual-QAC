# Multi-Lingual Chemical QAC

Build multi-lingual Question–Answer–Context (QAC) data from chemistry patents for Hugging Face and MTEB.

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

`--prepare-source EPO` is separate from the main pipeline on purpose. Normal `main.py` runs do not unpack source zip files by default, which makes it easier to support multiple patent sources such as EPO and USPTO later.

`--build-corpus EPO` is also explicit. It reads `data/EPO/xmls`, extracts useful bibliographic metadata, scores chemistry relevance from CPC/IPC codes plus title keywords, writes all parsed rows to `data/EPO/preprocessed/all_epo_records.csv`, and writes chemistry-focused rows to `data/EPO/corpus.csv`.

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
│   │   └── epo.py              # EPO zip scanning + XML extraction
│   ├── qac_generation/
│   │   └── openai_qa.py        # Q&A generation + translation
│   ├── export/
│   │   └── hf_upload.py        # Hugging Face / MTEB upload
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
