"""
Push corpus and QAC data to Hugging Face Hub.

Creates a dataset with splits: corpus, queries, qrels (MTEB retrieval format),
plus qac (full question-answer-context triplets).
Uploads a dataset card (README.md) that includes source attribution and license.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional

from datasets import Dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

# Dataset card text: attribution and license (CC BY 4.0, derived dataset, no endorsement, scope).
# Used for the Hugging Face dataset README so the same terms appear on the Hub.
DATASET_CARD_ATTRIBUTION = """
## Data source and license

- **Source dataset:** Patent text (titles, abstracts) in this dataset is derived from **Google Patents Public Data** on BigQuery (`patents-public-data.patents.publications`), provided by IFI CLAIMS Patent Services and Google. See [Marketplace](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data) and [announcement](https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data).
- **License:** That source data is made available under [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0).
- **This dataset:** The corpus, questions, and answers (including all Q&A pairs and translations) form a **derived/adapted dataset** based on that source.
- **No endorsement:** This dataset is not affiliated with, endorsed by, or officially connected with Google or IFI CLAIMS. Only the underlying patent publication text is from that source; the Q&A generation and benchmark design are independent.
- **Scope:** Attribution and license refer only to the patent dataset content (bibliographic and abstract text from the public BigQuery tables). They do not cover other Google services, products, or UI content.
"""

README_YAML = """---
configs:
- config_name: corpus
  data_files:
  - split: train
    path: data/corpus/*.parquet
- config_name: queries
  data_files:
  - split: train
    path: data/queries/*.parquet
- config_name: qrels
  data_files:
  - split: train
    path: data/qrels/*.parquet
- config_name: qac
  data_files:
  - split: train
    path: data/qac/*.parquet
---
"""

GENERIC_DATASET_INTRO = (
    "# Multi-lingual QAC retrieval dataset\n\n"
    "Question–Answer–Context (QAC) data for multilingual retrieval benchmarking. "
    "Configs/subsets: `corpus`, `queries`, `qrels`, `qac`. "
    "Each config currently contains a `train` split.\n"
)

PATENT_DATASET_INTRO = (
    "# Multi-lingual chemical QAC (retrieval benchmark)\n\n"
    "Question–Answer–Context (QAC) data for chemistry patent retrieval, multiple languages. "
    "Configs/subsets: `corpus`, `queries`, `qrels`, `qac` (MTEB-style). "
    "Each config currently contains a `train` split.\n"
)

JRC_DATASET_INTRO = (
    "# Multi-lingual JRC-Acquis QAC\n\n"
    "Question–Answer–Context (QAC) data derived from the JRC-Acquis multilingual legal corpus. "
    "Configs/subsets: `corpus`, `queries`, `qrels`, `qac`. "
    "Each config currently contains a `train` split.\n"
)

JRC_DATASET_CARD_ATTRIBUTION = """
## Data source

- **Source dataset:** JRC-Acquis, a multilingual aligned corpus of European Union legal texts.
- **This dataset:** The corpus subset, questions, and answers are derived benchmark artifacts built from JRC-Acquis language pairs, where one query is generated from the translated side of a selected pair and linked to both paired documents.
- **Note:** Verify the latest upstream distribution terms and citation guidance from the official JRC-Acquis source before public redistribution.
"""

WIKIDATA_DATASET_CARD_ATTRIBUTION = """
## Data source

- **Source datasets:** Wikidata entity metadata and multilingual Wikipedia text extracts.
- **This dataset:** The corpus, questions, and answers are derived benchmark artifacts built from those sources.
- **Note:** Verify the latest upstream attribution and redistribution requirements for Wikidata and Wikipedia before public redistribution.
"""


def _build_readme(source_name: str) -> str:
    source_name = (source_name or "").strip().lower()
    if source_name == "epo":
        intro = PATENT_DATASET_INTRO
        attribution = DATASET_CARD_ATTRIBUTION
    elif source_name == "jrc-acquis":
        intro = JRC_DATASET_INTRO
        attribution = JRC_DATASET_CARD_ATTRIBUTION
    elif source_name == "wikidata":
        intro = GENERIC_DATASET_INTRO
        attribution = WIKIDATA_DATASET_CARD_ATTRIBUTION
    else:
        intro = GENERIC_DATASET_INTRO
        attribution = ""
    return README_YAML + intro + attribution


def _set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _is_retryable_hf_error(exc: Exception) -> bool:
    if isinstance(exc, HfHubHTTPError):
        message = str(exc)
        return any(code in message for code in ("500", "502", "503", "504"))
    message = str(exc)
    return "Gateway Time-out" in message or "Server error" in message


def _with_hf_retries(action_name: str, func, *, max_attempts: int = 4):
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not _is_retryable_hf_error(exc):
                raise
            wait_s = min(20, 2 ** (attempt - 1) * 3)
            print(
                f"{action_name} failed with a transient Hub error "
                f"(attempt {attempt}/{max_attempts}). Retrying in {wait_s}s..."
            )
            time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc


def load_corpus(corpus_path: Path) -> list[dict]:
    rows = []
    _set_csv_field_size_limit()
    with corpus_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            if "_id" in parsed and "id" not in parsed:
                parsed["id"] = parsed["_id"]
            if "text" in parsed and "context" not in parsed:
                parsed["context"] = parsed["text"]
            rows.append(parsed)
    return rows


def load_qac(qac_path: Path) -> list[dict]:
    rows = []
    _set_csv_field_size_limit()
    with qac_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _linked_corpus_ids(row: dict, fallback_corpus_id: str) -> list[str]:
    raw = str(row.get("linked_corpus_ids_json", "")).strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            linked = []
            seen = set()
            for item in parsed:
                cid = str(item).strip()
                if cid and cid not in seen:
                    linked.append(cid)
                    seen.add(cid)
            if linked:
                return linked
    return [fallback_corpus_id] if fallback_corpus_id else []


def push_to_hub(
    corpus_path: Path,
    qac_path: Path,
    repo_id: str,
    *,
    source_name: str = "",
    token: Optional[str] = None,
    private: bool = False,
) -> str:
    """
    Push corpus and QAC to Hugging Face as a dataset.
    Creates splits: corpus, queries, qrels, qac (full triplets).
    Returns the dataset URL.
    """
    corpus_path = Path(corpus_path)
    qac_path = Path(qac_path)
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env for Hugging Face upload.")

    corpus_rows = load_corpus(corpus_path)
    qac_rows = load_qac(qac_path)

    # Corpus: MTEB format (_id, title, text)
    corpus_ids = {r["id"] for r in corpus_rows}
    corpus_data = [
        {"_id": r["id"], "title": r.get("title", ""), "text": r.get("context", r.get("abstract", ""))}
        for r in corpus_rows
    ]

    # Queries: _id, text (one per qac row)
    # Qrels: query-id, corpus-id, score (links each query to its corpus doc)
    queries_data = []
    qrels_data = []
    qac_full = []  # full triplets with answer

    seen_query_ids = set()
    for i, r in enumerate(qac_rows):
        cid = r.get("corpus_id", "")
        lang = r.get("language", "")
        q = r.get("question", "")
        a = r.get("answer", "")
        query_id_hint = str(r.get("query_id_hint", "")).strip()
        if query_id_hint:
            query_id = f"{query_id_hint}_q_{lang}"
        else:
            query_id = f"{cid}_q_{lang}" if cid in corpus_ids else f"q_{i}_{lang}"
        if query_id in seen_query_ids:
            query_id = f"{cid}_q_{lang}_{i}"
        seen_query_ids.add(query_id)

        queries_data.append({"_id": query_id, "text": q})
        for linked_corpus_id in _linked_corpus_ids(r, cid):
            qrels_data.append({"query-id": query_id, "corpus-id": linked_corpus_id, "score": 1.0})
        qac_full.append({
            "query_id": query_id,
            "corpus_id": cid,
            "language": lang,
            "question": q,
            "answer": a,
        })

    corpus_ds = Dataset.from_list(corpus_data)
    queries_ds = Dataset.from_list(queries_data)
    qrels_ds = Dataset.from_list(qrels_data)
    qac_ds = Dataset.from_list(qac_full)
    api = HfApi(token=token)

    _with_hf_retries(
        "Create/update dataset repo",
        lambda: api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        ),
    )

    # Push each subset/config separately, each with its own train split.
    _with_hf_retries(
        "Push corpus split",
        lambda: corpus_ds.push_to_hub(
            repo_id,
            config_name="corpus",
            split="train",
            data_dir="data/corpus",
            token=token,
            private=private,
        ),
    )
    _with_hf_retries(
        "Push queries split",
        lambda: queries_ds.push_to_hub(
            repo_id,
            config_name="queries",
            split="train",
            data_dir="data/queries",
            token=token,
            private=private,
        ),
    )
    _with_hf_retries(
        "Push qrels split",
        lambda: qrels_ds.push_to_hub(
            repo_id,
            config_name="qrels",
            split="train",
            data_dir="data/qrels",
            token=token,
            private=private,
        ),
    )
    _with_hf_retries(
        "Push qac split",
        lambda: qac_ds.push_to_hub(
            repo_id,
            config_name="qac",
            split="train",
            data_dir="data/qac",
            token=token,
            private=private,
        ),
    )

    # Upload dataset card (README.md) with attribution and license
    readme_body = _build_readme(source_name)
    _with_hf_retries(
        "Upload dataset README",
        lambda: api.upload_file(
            path_or_fileobj=io.BytesIO(readme_body.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        ),
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Pushed to {url}")
    return url
