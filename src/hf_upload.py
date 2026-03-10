"""
Push corpus and QAC data to Hugging Face Hub.

Creates a dataset with splits: corpus, queries, qrels (MTEB retrieval format),
plus qac (full question-answer-context triplets).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset


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

    # Push each config separately (different schemas: corpus has title, queries does not, etc.)
    corpus_ds.push_to_hub(repo_id, config_name="corpus", token=token, private=private)
    queries_ds.push_to_hub(repo_id, config_name="queries", token=token, private=private)
    qrels_ds.push_to_hub(repo_id, config_name="qrels", token=token, private=private)
    qac_ds.push_to_hub(repo_id, config_name="qac", token=token, private=private)
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Pushed to {url}")
    return url
