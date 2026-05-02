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
from typing import Any, Optional

from datasets import Dataset
from huggingface_hub import HfApi

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
- config_name: cross_language-corpus
  data_files:
  - split: train
    path: data/cross_language-corpus/*.parquet
- config_name: cross_language-queries
  data_files:
  - split: train
    path: data/cross_language-queries/*.parquet
- config_name: cross_language-qrels
  data_files:
  - split: train
    path: data/cross_language-qrels/*.parquet
- config_name: cross_language-qac
  data_files:
  - split: train
    path: data/cross_language-qac/*.parquet
---
"""


def load_corpus(corpus_path: Path) -> list[dict]:
    rows = []
    with corpus_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def load_qac(qac_path: Path) -> list[dict]:
    rows = []
    with qac_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _get_question_language(row: dict[str, Any]) -> str:
    return str(
        row.get("question_language")
        or row.get("language")
        or ""
    ).strip()


def _query_id(corpus_id: str, question_language: str, index: int, seen: set[str]) -> str:
    query_id = f"{corpus_id}_q_{question_language}" if corpus_id else f"q_{index}_{question_language}"
    if query_id in seen:
        query_id = f"{query_id}_{index}"
    seen.add(query_id)
    return query_id


def push_to_hub(
    corpus_path: Path,
    qac_path: Path,
    repo_id: str,
    *,
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

    # Corpus: MTEB format plus language metadata.
    corpus_ids = {r["id"] for r in corpus_rows}
    corpus_ids_by_publication: dict[str, list[str]] = {}
    corpus_language_by_id: dict[str, str] = {}
    for row in corpus_rows:
        pub_num = str(row.get("publication_number", "")).strip()
        corpus_id = str(row.get("id", "")).strip()
        if pub_num and corpus_id:
            corpus_ids_by_publication.setdefault(pub_num, []).append(corpus_id)
        if corpus_id:
            corpus_language_by_id[corpus_id] = str(row.get("language", "")).strip()
    corpus_data = [
        {
            "_id": r["id"],
            "corpus_id": r["id"],
            "title": r.get("title", ""),
            "text": r.get("context", r.get("abstract", "")),
            "corpus_language": str(r.get("language", "")).strip(),
        }
        for r in corpus_rows
    ]

    # Queries: _id, text (one per qac row)
    # Qrels: query-id, corpus-id, score (links each query to its corpus doc)
    queries_data = []
    qrels_data = []
    cross_language_qrels_data = []
    qac_full = []  # full triplets with answer
    cross_language_qac_full = []

    seen_query_ids = set()
    for i, r in enumerate(qac_rows):
        cid = str(r.get("corpus_id", "")).strip()
        lang = _get_question_language(r)
        corpus_lang = corpus_language_by_id.get(cid, "")
        q = r.get("question", "")
        a = r.get("answer", "")
        query_id = _query_id(cid if cid in corpus_ids else "", lang, i, seen_query_ids)
        is_synthetic_translation = bool(lang and corpus_lang and lang != corpus_lang)
        publication_number = str(r.get("publication_number", "")).strip()
        relevant_corpus_ids = corpus_ids_by_publication.get(publication_number, [])
        if not relevant_corpus_ids and cid:
            relevant_corpus_ids = [cid]
        cross_language_corpus_ids = [
            relevant_corpus_id
            for relevant_corpus_id in relevant_corpus_ids
            if corpus_language_by_id.get(relevant_corpus_id, "") != lang
        ]
        if not cross_language_corpus_ids:
            cross_language_corpus_ids = list(relevant_corpus_ids)

        query_row = {
            "_id": query_id,
            "query_id": query_id,
            "text": q,
            "query_language": lang,
            "corpus_id": cid,
            "corpus_language": corpus_lang,
            "is_synthetic_translation": is_synthetic_translation,
        }
        queries_data.append(query_row)
        for relevant_corpus_id in relevant_corpus_ids:
            qrels_data.append(
                {"query-id": query_id, "corpus-id": relevant_corpus_id, "score": 1.0}
            )
        for relevant_corpus_id in cross_language_corpus_ids:
            cross_language_qrels_data.append(
                {"query-id": query_id, "corpus-id": relevant_corpus_id, "score": 1.0}
            )
        qac_full.append({
            "query_id": query_id,
            "corpus_id": cid,
            "query_language": lang,
            "corpus_language": corpus_lang,
            "question": q,
            "answer": a,
            "is_synthetic_translation": is_synthetic_translation,
            "linked_corpus_ids_json": json.dumps(relevant_corpus_ids, ensure_ascii=False),
        })
        cross_language_qac_full.append({
            "query_id": query_id,
            "corpus_id": cid,
            "query_language": lang,
            "corpus_language": corpus_lang,
            "question": q,
            "answer": a,
            "is_synthetic_translation": is_synthetic_translation,
            "linked_corpus_ids_json": json.dumps(cross_language_corpus_ids, ensure_ascii=False),
        })

    corpus_ds = Dataset.from_list(corpus_data)
    queries_ds = Dataset.from_list(queries_data)
    qrels_ds = Dataset.from_list(qrels_data)
    qac_ds = Dataset.from_list(qac_full)
    cross_language_corpus_ds = Dataset.from_list(corpus_data)
    cross_language_queries_ds = Dataset.from_list(queries_data)
    cross_language_qrels_ds = Dataset.from_list(cross_language_qrels_data)
    cross_language_qac_ds = Dataset.from_list(cross_language_qac_full)

    # Push each subset/config separately, each with its own train split.
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    corpus_ds.push_to_hub(
        repo_id,
        config_name="corpus",
        split="train",
        data_dir="data/corpus",
        token=token,
        private=private,
    )
    queries_ds.push_to_hub(
        repo_id,
        config_name="queries",
        split="train",
        data_dir="data/queries",
        token=token,
        private=private,
    )
    qrels_ds.push_to_hub(
        repo_id,
        config_name="qrels",
        split="train",
        data_dir="data/qrels",
        token=token,
        private=private,
    )
    qac_ds.push_to_hub(
        repo_id,
        config_name="qac",
        split="train",
        data_dir="data/qac",
        token=token,
        private=private,
    )
    cross_language_corpus_ds.push_to_hub(
        repo_id,
        config_name="cross_language-corpus",
        split="train",
        data_dir="data/cross_language-corpus",
        token=token,
        private=private,
    )
    cross_language_queries_ds.push_to_hub(
        repo_id,
        config_name="cross_language-queries",
        split="train",
        data_dir="data/cross_language-queries",
        token=token,
        private=private,
    )
    cross_language_qrels_ds.push_to_hub(
        repo_id,
        config_name="cross_language-qrels",
        split="train",
        data_dir="data/cross_language-qrels",
        token=token,
        private=private,
    )
    cross_language_qac_ds.push_to_hub(
        repo_id,
        config_name="cross_language-qac",
        split="train",
        data_dir="data/cross_language-qac",
        token=token,
        private=private,
    )

    # Upload dataset card (README.md) with attribution and license
    readme_body = (
        README_YAML
        + "# Multi-lingual chemical QAC (retrieval benchmark)\n\n"
        "Question–Answer–Context (QAC) data for chemistry patent retrieval, multiple languages. "
        "Configs/subsets: `corpus`, `queries`, `qrels`, `qac`, plus "
        "`cross_language-corpus`, `cross_language-queries`, `cross_language-qrels`, "
        "`cross_language-qac` (MTEB-style). "
        "Each config currently contains a `train` split.\n"
        + DATASET_CARD_ATTRIBUTION
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_body.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Pushed to {url}")
    return url


def upload_benchmark_outputs(
    local_dir: Path,
    repo_id: str,
    *,
    path_in_repo: str,
    token: Optional[str] = None,
    private: bool = False,
) -> str:
    """Upload generated benchmark artifacts to a Hugging Face dataset repo."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise ValueError(f"Benchmark output directory does not exist: {local_dir}")

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env for Hugging Face upload.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update {path_in_repo}",
    )
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}"
