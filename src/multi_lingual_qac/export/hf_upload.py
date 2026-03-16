"""
Push corpus and QAC data to Hugging Face Hub.

Creates a dataset with splits: corpus, queries, qrels (MTEB retrieval format),
plus qac (full question-answer-context triplets).
Uploads a dataset card (README.md) that includes source attribution and license.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Optional

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
        query_id = f"{cid}_q_{lang}" if cid in corpus_ids else f"q_{i}_{lang}"
        if query_id in seen_query_ids:
            query_id = f"{cid}_q_{lang}_{i}"
        seen_query_ids.add(query_id)

        queries_data.append({"_id": query_id, "text": q})
        qrels_data.append({"query-id": query_id, "corpus-id": cid, "score": 1.0})
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

    # Push each subset/config separately, each with its own train split.
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

    # Upload dataset card (README.md) with attribution and license
    readme_body = (
        README_YAML
        + "# Multi-lingual chemical QAC (retrieval benchmark)\n\n"
        "Question–Answer–Context (QAC) data for chemistry patent retrieval, multiple languages. "
        "Configs/subsets: `corpus`, `queries`, `qrels`, `qac` (MTEB-style). "
        "Each config currently contains a `train` split.\n"
        + DATASET_CARD_ATTRIBUTION
    )
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_body.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Pushed to {url}")
    return url
