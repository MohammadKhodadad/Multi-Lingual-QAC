# Multi-Lingual Chemical QAC Data Pipeline — Plan

A step-by-step plan for building chemical Question–Answer–Context (QAC) data that is multi-lingual (questions and answers can be in different languages) and compatible with **Hugging Face Datasets** and **MTEB**.

---

## Overview

| Step | Goal | Key Output |
|------|------|------------|
| 1 | Gather chemical text | Raw corpora (multi-language) |
| 2 | Extract or generate QAC triplets | (context, question, answer) pairs |
| 3 | Enforce multi-lingual setup | Cross-language QAC |
| 4 | Clean & validate | Gold QAC dataset |
| 5 | Export for HF & MTEB | Parquet / HF dataset format |

---

## Step 1: Data Gathering — Chemical Text Corpora

### Goals
- Collect chemical text in multiple languages
- Include full-text and abstracts
- Ensure sufficient volume for QAC generation

### Target datasets: WIPO PatentScope, Lens, and Google Patents

Three primary target datasets for patent-based chemical QAC:

| Dataset | Type | Languages | Notes |
|---------|------|-----------|-------|
| **WIPO PatentScope** | Patents | 60+ patent collections, many languages | ~100M documents; PCT + national/regional |
| **Lens** | Patents (and scholarly) | EN, FR, DE, CN, JP, ES, RU, KR, and more | REST API; chemistry via IPC class `C` |
| **Google Patents** | Patents | Many; search by language/country | No official API; third-party APIs or scraping tools |

---

### How to get patents: WIPO PatentScope

**Important:** The PatentScope search URL (`patentscope.wipo.int/search/...`) is a **web UI**, not a REST API. Simple HTTP GET requests will return permission errors.

#### Option A: Web-based download (recommended for beginners)

1. **Register** for a free account: [PatentScope Advanced Search](https://patentscope.wipo.int/search/en/advancedSearch.jsf)
2. **Search** for chemistry patents (e.g. IPC class prefix `C` for chemistry/metallurgy, or keywords)
3. **Export** results: PatentScope allows downloads of up to **10,000 records** per search
4. **Format:** Typically CSV/XML export from the web interface

#### Option B: Bulk data and data products

- **PCT Data Products:** WIPO offers PCT data via [Bulk Data Services](https://www.wipo.int/patentscope/en/data)
- Formats: DVDs, hard drives, FTP, web services
- **Terms:** Check [WIPO Terms for PatentScope data](https://www.wipo.int/en/web/patentscope/data/terms_patentscope)

#### Option C: API Catalog

- WIPO maintains an [API Catalog for IP](https://apicatalog.wipo.int/en/)
- Check for official PatentScope-related APIs; use their documentation for proper authentication and endpoints
- Do **not** scrape the search URL; use only documented APIs

#### Filters for chemistry

- **IPC (International Patent Classification):** Use prefix `C` for chemistry and metallurgy (e.g. C07, C08)
- **Field combination search:** Combine IPC with language, date, jurisdiction

---

### How to get patents: Lens

Lens provides a **REST API** for patent search. Programmatic access is straightforward once you have a token.

#### Step 1: Get API access

1. **Register** at [lens.org](https://www.lens.org/)
2. Go to your **profile** → API section
3. **Request trial access** (free 14-day trial for non-commercial/academic use) or subscribe for ongoing access
4. Generate an **API token**
5. Set the token: `export LENS_API_TOKEN='your_token_here'`

#### Step 2: API usage

- **Endpoint:** `POST https://api.lens.org/patent/search`
- **Auth:** `Authorization: Bearer <LENS_API_TOKEN>`
- **Content-Type:** `application/json`
- **Body:** JSON query (see [Lens Patent API docs](https://docs.api.lens.org/request-patent.html))

#### Step 3: Chemistry query example

Filter by IPC class prefix `C` for chemistry-related patents, and optionally by language:

```json
{
  "query": {
    "bool": {
      "must": [
        { "prefix": { "class_ipcr.symbol": "C" } },
        { "terms": { "lang": ["EN", "FR", "DE", "CN", "JP", "ES", "RU", "KR"] } }
      ]
    }
  },
  "include": ["lens_id", "biblio", "abstract", "description", "claim", "family"],
  "size": 100,
  "scroll": "1m"
}
```

#### Step 4: Pagination

- Use the `scroll_id` from the response to fetch the next page: `POST` same endpoint with `{"scroll_id": "...", "scroll": "1m"}`
- Add a small delay (e.g. 0.3–0.5 s) between requests to respect rate limits

#### Step 5: Output

- Extract `abstract`, `description`, `claim`, `title` (from `biblio.invention_title`), `lang`, and identifiers
- Save as JSONL for downstream QAC generation

---

### How to get patents: Google Patents

**Important:** Google does not offer an official public API for Google Patents. The original Google Patent Search API was deprecated in 2011.

#### Option A: Web search + manual export

1. Go to [Google Patents](https://patents.google.com/)
2. Search for chemistry patents (keywords, IPC codes, or CPC)
3. Filter by language, country, date
4. Browse results; export options are limited in the web UI
5. Save results to `data/google_patents/` as you collect them

#### Option B: Third-party APIs (paid)

- **SerpApi** — [Google Patents API](https://serpapi.com/google-patents-api): search by keywords, patent numbers, CPC; paginated results
- **SearchApi** — [Google Patents API](https://searchapi.io/docs/google-patents): similar capabilities
- Both require API keys and have usage-based pricing
- Filter by country, language, patent status, and type

#### Option C: Python scraping tools

- **google-patent-scraper** — [PyPI](https://pypi.org/project/google-patent-scraper/): extract patent data (inventors, assignees, dates, citations)
- **GooglePatentsPdfDownloader** — [GitHub](https://github.com/lorenzbr/GooglePatentsPdfDownloader): download patents as PDF
- Check terms of service; use responsibly and respect rate limits

#### Filters for chemistry

- **CPC (Cooperative Patent Classification):** Chemistry classes (e.g. C07, C08) mirror IPC
- **Keywords:** e.g. "catalyst", "synthesis", "polymer"
- Use `src/dataloader/google_patents.py` to load and process data from `data/google_patents/`

---

### Other sources (supplementary)

| Source | Type | Use case |
|--------|------|----------|
| **PubMed / PMC** | Abstracts, full text | Biomedical/chemistry literature |
| **ChemPile** | Curated corpus | Large-scale chemistry text |
| **EPO OPS** | Patents | Alternative multilingual patent API (free 4 GB/week) |
| **Wikipedia (Chemistry)** | Articles | Cross-lingual alignment via interlanguage links |

---

## Step 2: Question Extraction / Generation

### Goals
- Obtain (context, question, answer) triplets
- Support both extractive and abstractive QA
- Keep answers grounded in context

### Approaches (where to start)

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **LLM-based generation** | Use GPT-4, Claude, LLaMA, etc. to generate Q&A from context | High throughput, controllable formats | Cost, hallucinations, needs validation |
| **Existing chemistry QA datasets** | Adapt ChemLit-QA, ChemRxivQuest, ChemIQ | Human/expert validated | Mostly English; limited size |
| **Template-based** | Templates over entities (e.g., “What is the melting point of X?”) | Deterministic, cheap | Less diverse, needs entity extraction |
| **Synthetic augmentation** | Paraphrase questions, translate | Can add languages | May drift from original semantics |
| **Human annotation** | Crowdsource or expert labeling | High quality | Expensive and slow |

### Existing chemistry QA resources
- **[ChemLit-QA](https://github.com/geemi725/chemlit-qa)** — 1,000+ expert-validated QAC triplets (EN)
- **[ChemRxivQuest](https://arxiv.org/html/2505.05232)** — 970 QA pairs from ChemRxiv (EN)
- **[ChemIQ](https://github.com/oxpig/ChemIQ)** — Complex chemistry QA
- **[Microsoft ChemistryQA](https://github.com/microsoft/chemistry-qa)** — Chemistry QA tasks

### Recommended starting points
1. **LLM-based generation** (e.g., GPT-4) with prompting similar to ChemLit-QA / ChemRxivQuest.
2. **ChemLit-QA / ChemRxivQuest** — use as seed data, evaluation benchmarks, and prompt templates.
3. **Entity-based templates** — use CHEMDNER / NLM-Chem annotations to generate entity-centric questions.

---

## Step 3: Multi-Lingual Setup

### Goals
- Allow questions in one language and answers in another
- Support cross-lingual retrieval and QA
- Store explicit language metadata

### Strategies

| Strategy | Description | Where to start |
|----------|-------------|----------------|
| **Source-language text** | Use original-language patents, abstracts, articles | WIPO, Lens, Google Patents, EPO, Wikipedia dumps, PubMed language filters |
| **Translation** | Use NLLB, M2M, or commercial APIs to translate Q/A/context | [Hugging Face NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M), [OPUS-MT](https://huggingface.co/Helsinki-NLP) |
| **Paraphrase in target language** | Paraphrase in same language, then translate | LLM APIs (Claude, GPT) with explicit language instructions |
| **Parallel corpora** | Align same content across languages (e.g., EPO, Wikipedia) | [Europarl](https://opus.nlpl.eu/Europarl.php), Wikipedia dumps with interwiki links |
| **Cross-lingual QA pairs** | Generate Q in L1, keep A in L2 (or vice versa) | Custom pipeline: generate in EN, translate only Q or only A |

### Recommended starting points
1. **WIPO / Lens patents** — same patent or family in multiple languages; extract QAC in each language and link by patent ID or family.
2. **Wikipedia** — use [interlanguage links](https://www.mediawiki.org/wiki/Manual:Interwiki) to build (EN, DE, FR, …) aligned chemistry articles; generate QAC per language.
3. **Translation pipeline** — start from EN QAC (ChemLit-QA, ChemRxivQuest), translate Q or A to create cross-language variants.

---

## Step 4: Cleaning, Validation & Quality Control

### Goals
- Remove unanswerable or low-quality QAC pairs
- Ensure answer faithfulness to context
- Deduplicate and normalize

### Checks (where to start)

| Check | Implementation | Tools / Ideas |
|-------|----------------|---------------|
| **Answer in context** | Verify answer span appears in context (exact or fuzzy) | `str.find`, fuzzy matching (e.g., `rapidfuzz`) |
| **Relevancy** | Ensure question and answer are semantically related | NLI model (e.g., `microsoft/deberta-v3-base-mnli`), similarity scores |
| **Faithfulness** | No invented content in answers | Rule-based + LLM-as-judge |
| **Language tagging** | Detect language of Q, A, context | `langdetect`, `fasttext`, `pycld3` |
| **Deduplication** | Remove near-duplicate QAC | MinHash, embedding similarity + threshold |
| **Length & format** | Reasonable lengths, valid characters | Heuristics, schema validation |

### Recommended starting points
1. **ChemLit-QA metrics** — reuse their answer relevancy and faithfulness evaluation.
2. **NLI-based filtering** — entailment score between (context, answer) and (question).
3. **Language detection** — attach `lang_question`, `lang_answer`, `lang_context` for downstream analysis.

---

## Step 5: Format & Export for Hugging Face and MTEB

### Goals
- Produce a standard Hugging Face dataset
- Optionally support MTEB retrieval tasks (query, corpus, qrels)

### Hugging Face QAC Format

Standard fields (e.g., SQuAD-style):

```python
{
    "id": str,           # Unique identifier
    "context": str,      # Source passage
    "question": str,     # Question (any language)
    "answers": [         # Can have multiple valid answers
        {"text": str, "answer_start": int}
    ],
    "language_question": str,   # e.g., "en", "de"
    "language_answer": str,     # e.g., "en", "fr"
    "language_context": str,
    "source": str,       # e.g., "wipo", "lens", "google_patents", "pubmed"
    "domain": str,       # e.g., "organic", "biochemistry"
}
```

### MTEB Retrieval Format (Retrieval task)

MTEB uses **three parquet datasets** (separate configs/splits). Load with:
`load_dataset(name, "corpus")`, `load_dataset(name, "queries")`, `load_dataset(name, "qrels")`.

#### 1. Corpus (`corpus/*.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `_id` | str | Unique document ID |
| `title` | str | Document title (can be empty string) |
| `text` | str | Document content |

#### 2. Queries (`queries/*.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `_id` | str | Unique query ID |
| `text` | str | Query text (the question) |

#### 3. Qrels (`qrels/*.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `query-id` | str | References query `_id` |
| `corpus-id` | str | References corpus `_id` |
| `score` | float | Relevance (1 = relevant, 0 = not) |

#### Directory structure (Hugging Face)

```
dataset_name/
├── corpus/
│   └── corpus-00000-of-00001.parquet
├── queries/
│   └── queries-00000-of-00001.parquet
└── qrels/
    └── qrels-00000-of-00001.parquet
```

#### Mapping from QAC to MTEB

| QAC field | MTEB field |
|-----------|------------|
| `context` | corpus `text` (and optionally `title`) |
| `question` | queries `text` |
| context `id` | corpus `_id` |
| question `id` | queries `_id` |
| — | qrels: `query-id` = question id, `corpus-id` = context id, `score` = 1 |

Each QAC triplet becomes: one corpus row (context), one query row (question), one qrels row linking them with score 1.

### Where to start

| Resource | Purpose |
|----------|---------|
| [Hugging Face Datasets](https://huggingface.co/docs/datasets/) | Create and push datasets |
| [MTEB GitHub](https://github.com/embeddings-benchmark/mteb) | Add new tasks, see `CONTRIBUTING.md` |
| [MTEB datasets on HF](https://huggingface.co/datasets?search=mteb) | Inspect structure of existing retrieval tasks |
| [Parquet format](https://parquet.apache.org/) | Recommended storage for corpus, queries, qrels |

### Suggested export flow
1. **QAC dataset** → Hugging Face via `datasets.Dataset.from_*` and `push_to_hub`.
2. **MTEB retrieval** → Derive `corpus`, `queries`, `qrels` from QAC and load via MTEB’s `RetrievalTask` with parquet or custom loader.
3. **Multilingual subsets** — create splits like `en_en`, `en_de`, `de_en` for different (question_lang, answer_lang) combinations.

---

## Step 6: Optional — MTEB Task Contribution

### Goals
- Contribute a new retrieval (or other) task to MTEB
- Enable standardized evaluation of embedding models on chemical QAC

### Where to start
- **MTEB repository**: [embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)
- **Contributing guide**: `CONTRIBUTING.md` in the repo
- **Task types**: retrieval (`RetrievalTask`), classification, clustering, etc.
- **Dataset loading**: MTEB expects corpus, queries, qrels; parquet is supported.

---

## Suggested Implementation Order

1. **Data gathering**: WIPO PatentScope, Lens (API), and/or Google Patents (web, third-party API, or scraper). Target chemistry patents (IPC/CPC class `C`) in multiple languages.
2. **Question generation**: LLM pipeline inspired by ChemLit-QA/ChemRxivQuest; start with English.
3. **Multi-lingual expansion**: Use source-language text from WIPO/Lens, or translation of Q/A.
4. **Validation**: Answer-in-context checks + NLI-based relevancy.
5. **Export**: Hugging Face dataset with `source: "wipo"`, `source: "lens"`, or `source: "google_patents"` and language metadata.
6. **MTEB**: Derive retrieval task (corpus, queries, qrels) and submit if desired.

---

## Quick Reference — URLs

| Resource | URL |
|----------|-----|
| **WIPO PatentScope** | https://patentscope.wipo.int/search/en/advancedSearch.jsf |
| **WIPO Bulk Data** | https://www.wipo.int/patentscope/en/data |
| **WIPO API Catalog** | https://apicatalog.wipo.int/en/ |
| **Lens** | https://www.lens.org/ |
| **Lens Patent API** | https://docs.api.lens.org/request-patent.html |
| **Google Patents** | https://patents.google.com/ |
| **SerpApi Google Patents** | https://serpapi.com/google-patents-api |
| **google-patent-scraper** | https://pypi.org/project/google-patent-scraper/ |
| PubMed E-utilities | https://www.ncbi.nlm.nih.gov/books/NBK25501/ |
| ChemPile | https://chempile.lamalab.org/ |
| EPO OPS API | https://developers.epo.org/ |
| ChemLit-QA | https://github.com/geemi725/chemlit-qa |
| ChemRxivQuest paper | https://arxiv.org/html/2505.05232 |
| MTEB GitHub | https://github.com/embeddings-benchmark/mteb |
| Hugging Face Datasets | https://huggingface.co/docs/datasets |
| NLLB (translation) | https://huggingface.co/facebook/nllb-200-distilled-600M |
