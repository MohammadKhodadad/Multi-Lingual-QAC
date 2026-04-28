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
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

# Dataset card text: attribution and license (CC BY 4.0, derived dataset, no endorsement, scope).
# Used for the Hugging Face dataset README so the same terms appear on the Hub.
DATASET_CARD_ATTRIBUTION = """
## Data Source And License

- **Source dataset:** Patent text (titles, abstracts) in this dataset is derived from **Google Patents Public Data** on BigQuery (`patents-public-data.patents.publications`), provided by IFI CLAIMS Patent Services and Google. See [Marketplace](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data) and [announcement](https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data).
- **License:** That source data is made available under [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0).
- **This dataset:** The corpus, questions, and answers (including all Q&A pairs and translations) form a **derived/adapted dataset** based on that source.
- **No endorsement:** This dataset is not affiliated with, endorsed by, or officially connected with Google or IFI CLAIMS. Only the underlying patent publication text is from that source; the Q&A generation and benchmark design are independent.
- **Scope:** Attribution and license refer only to the patent dataset content (bibliographic and abstract text from the public BigQuery tables). They do not cover other Google services, products, or UI content.
"""

COMMON_DATASET_STRUCTURE = """
## Dataset Structure

- `corpus`: retrieval documents
- `queries`: benchmark queries
- `qrels`: relevance judgments
- `qac`: full question-answer-context rows for inspection and analysis

When variant-specific configs are present, the unprefixed `corpus` / `queries` / `qrels` / `qac` configs remain the default multilingual benchmark. The `cross_language-corpus` / `cross_language-queries` / `cross_language-qrels` configs keep the cross-language retrieval setup, and `cross_language-qac` keeps one row per query while filtering `linked_corpus_ids_json` down to cross-language relevant corpus rows.

Each config currently contains a `train` split.
"""

GENERIC_DATASET_INTRO = (
    "# Multi-lingual QAC Retrieval Dataset\n\n"
    "## Overview\n\n"
    "Question–Answer–Context (QAC) data for multilingual retrieval benchmarking.\n\n"
)

PATENT_DATASET_INTRO = (
    "# Multi-lingual Chemical QAC\n\n"
    "## Overview\n\n"
    "Question–Answer–Context (QAC) data for chemistry patent retrieval across multiple languages.\n\n"
)

JRC_DATASET_INTRO = (
    "# Multi-lingual JRC-Acquis QAC\n\n"
    "## Overview\n\n"
    "Question–Answer–Context (QAC) data derived from the JRC-Acquis multilingual legal corpus.\n\n"
)

JRC_DATASET_CARD_ATTRIBUTION = """
## Data Source

- **Source dataset:** JRC-Acquis, a multilingual aligned corpus of European Union legal texts.
- **This dataset:** The corpus subset, questions, and answers are derived benchmark artifacts built from JRC-Acquis language pairs, where one query is generated from the translated side of a selected pair and linked to both paired documents.
- **Note:** Verify the latest upstream distribution terms and citation guidance from the official JRC-Acquis source before public redistribution.
"""

WIKIDATA_DATASET_CARD_ATTRIBUTION = """
## Data Source

- **Source datasets:** Wikidata entity metadata and multilingual Wikipedia text extracts.
- **This dataset:** The corpus, questions, and answers are derived benchmark artifacts built from those sources.
- **Note:** Verify the latest upstream attribution and redistribution requirements for Wikidata and Wikipedia before public redistribution.
"""

LEADERBOARD_START = "<!-- BEGIN MTEB LEADERBOARD -->"
LEADERBOARD_END = "<!-- END MTEB LEADERBOARD -->"


def _build_readme_yaml(*, include_variant_configs: bool) -> str:
    configs = [
        ("corpus", "data/corpus/*.parquet"),
        ("queries", "data/queries/*.parquet"),
        ("qrels", "data/qrels/*.parquet"),
        ("qac", "data/qac/*.parquet"),
    ]
    if include_variant_configs:
        configs.extend(
            [
                ("cross_language-corpus", "data/cross_language-corpus/*.parquet"),
                ("cross_language-queries", "data/cross_language-queries/*.parquet"),
                ("cross_language-qrels", "data/cross_language-qrels/*.parquet"),
                ("cross_language-qac", "data/cross_language-qac/*.parquet"),
            ]
        )
    lines = ["---", "configs:"]
    for config_name, path in configs:
        lines.extend(
            [
                f"- config_name: {config_name}",
                "  data_files:",
                "  - split: train",
                f"    path: {path}",
            ]
        )
    lines.append("---")
    return "\n".join(lines) + "\n"


def _build_readme(source_name: str, *, include_variant_configs: bool) -> str:
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
    return _build_readme_yaml(include_variant_configs=include_variant_configs) + intro + COMMON_DATASET_STRUCTURE + "\n" + attribution


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


def _download_repo_readme(repo_id: str, token: str) -> str:
    try:
        readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
            token=token,
        )
        return Path(readme_path).read_text(encoding="utf-8")
    except EntryNotFoundError:
        return ""


def _leaderboard_section_body(leaderboard_md: str, artifact_path: str) -> str:
    lines = leaderboard_md.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    if lines and lines[0].strip() == "## Leaderboard":
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    body = "\n".join(lines).strip()
    section_lines = [
        "## Leaderboard",
        "",
        f"Latest generated benchmark comparison tables are also available under `{artifact_path}`.",
    ]
    if body:
        section_lines.extend(["", body])
    return "\n".join(section_lines).strip() + "\n"


def _replace_or_append_leaderboard(readme_body: str, leaderboard_body: str) -> str:
    block = f"{LEADERBOARD_START}\n{leaderboard_body}{LEADERBOARD_END}\n"
    if LEADERBOARD_START in readme_body and LEADERBOARD_END in readme_body:
        prefix, remainder = readme_body.split(LEADERBOARD_START, 1)
        _, suffix = remainder.split(LEADERBOARD_END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix.lstrip()
        return updated.rstrip() + "\n"
    if readme_body.strip():
        return readme_body.rstrip() + "\n\n" + block
    return block


def _update_dataset_readme_leaderboard(
    *,
    api: HfApi,
    repo_id: str,
    token: str,
    leaderboard_md_path: Path,
    artifact_path: str,
) -> None:
    if not leaderboard_md_path.is_file():
        raise ValueError(f"Missing leaderboard markdown file: {leaderboard_md_path}")

    current_readme = _download_repo_readme(repo_id, token)
    leaderboard_body = _leaderboard_section_body(
        leaderboard_md_path.read_text(encoding="utf-8"),
        artifact_path,
    )
    updated_readme = _replace_or_append_leaderboard(current_readme, leaderboard_body)
    _with_hf_retries(
        "Upload dataset README with leaderboard",
        lambda: api.upload_file(
            path_or_fileobj=io.BytesIO(updated_readme.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        ),
    )


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


def _infer_language(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    parts = [part for part in raw.replace("-", "_").split("_") if part]
    if not parts:
        return ""
    candidate = parts[-1]
    if 2 <= len(candidate) <= 5 and candidate.isalpha():
        return candidate
    return ""


def _corpus_language(row: dict) -> str:
    explicit = str(row.get("language", "") or row.get("lang", "")).strip().lower()
    if explicit:
        return explicit
    return _infer_language(row.get("id", "") or row.get("_id", ""))


def _parse_boolish(value: object) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "y"}


def _cross_language_linked_corpus_ids(
    row: dict,
    fallback_corpus_id: str,
    *,
    query_language: str,
    corpus_language_by_id: dict[str, str],
) -> list[str]:
    linked_ids = _linked_corpus_ids(row, fallback_corpus_id)
    query_language = (query_language or "").strip().lower()
    return [
        corpus_id
        for corpus_id in linked_ids
        if corpus_language_by_id.get(corpus_id, _infer_language(corpus_id)) != query_language
    ]


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
    corpus_language_by_id = {
        r["id"]: _corpus_language(r)
        for r in corpus_rows
    }
    corpus_data = [
        {
            "_id": r["id"],
            "corpus_id": r["id"],
            "title": r.get("title", ""),
            "text": r.get("context", r.get("abstract", "")),
            "corpus_language": corpus_language_by_id.get(r["id"], ""),
        }
        for r in corpus_rows
    ]

    # Queries: keep `_id` for retrieval-tool compatibility, plus explicit
    # readability fields for Hugging Face browsing.
    # Qrels: query-id, corpus-id, score (links each query to its corpus doc)
    queries_data = []
    qrels_data = []
    cross_language_qrels_data = []
    qac_full = []  # full triplets with answer
    cross_language_qac_full = []

    seen_query_ids = set()
    for i, r in enumerate(qac_rows):
        cid = r.get("corpus_id", "")
        lang = str(r.get("language", "")).strip().lower()
        q = r.get("question", "")
        a = r.get("answer", "")
        corpus_lang = corpus_language_by_id.get(cid, _infer_language(cid))
        is_synthetic_translation = _parse_boolish(r.get("is_synthetic_translation", ""))
        query_id_hint = str(r.get("query_id_hint", "")).strip()
        if query_id_hint:
            query_id = f"{query_id_hint}_q_{lang}"
        else:
            query_id = f"{cid}_q_{lang}" if cid in corpus_ids else f"q_{i}_{lang}"
        if query_id in seen_query_ids:
            query_id = f"{cid}_q_{lang}_{i}"
        seen_query_ids.add(query_id)

        queries_data.append({
            "_id": query_id,
            "query_id": query_id,
            "text": q,
            "query_language": lang,
            "corpus_id": cid,
            "corpus_language": corpus_lang,
            "is_synthetic_translation": is_synthetic_translation,
        })
        linked_ids = _linked_corpus_ids(r, cid)
        for linked_corpus_id in linked_ids:
            qrel_row = {"query-id": query_id, "corpus-id": linked_corpus_id, "score": 1.0}
            qrels_data.append(qrel_row)
        cross_language_linked_ids = _cross_language_linked_corpus_ids(
            r,
            cid,
            query_language=lang,
            corpus_language_by_id=corpus_language_by_id,
        )
        for linked_corpus_id in cross_language_linked_ids:
            cross_language_qrels_data.append(
                {"query-id": query_id, "corpus-id": linked_corpus_id, "score": 1.0}
            )
        qac_full.append({
            "query_id": query_id,
            "corpus_id": cid,
            "query_language": lang,
            "corpus_language": corpus_lang,
            "question": q,
            "answer": a,
            "is_synthetic_translation": is_synthetic_translation,
            "linked_corpus_ids_json": json.dumps(linked_ids, ensure_ascii=False),
        })
        if cross_language_linked_ids:
            cross_language_qac_full.append({
                "query_id": query_id,
                "corpus_id": cid,
                "query_language": lang,
                "corpus_language": corpus_lang,
                "question": q,
                "answer": a,
                "is_synthetic_translation": is_synthetic_translation,
                "linked_corpus_ids_json": json.dumps(
                    cross_language_linked_ids,
                    ensure_ascii=False,
                ),
            })

    corpus_ds = Dataset.from_list(corpus_data)
    queries_ds = Dataset.from_list(queries_data)
    qrels_ds = Dataset.from_list(qrels_data)
    qac_ds = Dataset.from_list(qac_full)
    cross_language_corpus_ds = Dataset.from_list(corpus_data) if cross_language_qrels_data else None
    cross_language_queries_ds = Dataset.from_list(queries_data) if cross_language_qrels_data else None
    cross_language_qrels_ds = (
        Dataset.from_list(cross_language_qrels_data) if cross_language_qrels_data else None
    )
    cross_language_qac_ds = (
        Dataset.from_list(cross_language_qac_full) if cross_language_qac_full else None
    )
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
    if cross_language_qrels_data:
        _with_hf_retries(
            "Push cross-language corpus split",
            lambda: cross_language_corpus_ds.push_to_hub(
                repo_id,
                config_name="cross_language-corpus",
                split="train",
                data_dir="data/cross_language-corpus",
                token=token,
                private=private,
            ),
        )
        _with_hf_retries(
            "Push cross-language queries split",
            lambda: cross_language_queries_ds.push_to_hub(
                repo_id,
                config_name="cross_language-queries",
                split="train",
                data_dir="data/cross_language-queries",
                token=token,
                private=private,
            ),
        )
        _with_hf_retries(
            "Push cross-language qrels split",
            lambda: cross_language_qrels_ds.push_to_hub(
                repo_id,
                config_name="cross_language-qrels",
                split="train",
                data_dir="data/cross_language-qrels",
                token=token,
                private=private,
            ),
        )
        if cross_language_qac_ds is not None:
            _with_hf_retries(
                "Push cross-language qac split",
                lambda: cross_language_qac_ds.push_to_hub(
                    repo_id,
                    config_name="cross_language-qac",
                    split="train",
                    data_dir="data/cross_language-qac",
                    token=token,
                    private=private,
                ),
            )

    # Upload dataset card (README.md) with attribution and license
    readme_body = _build_readme(
        source_name,
        include_variant_configs=bool(cross_language_qrels_data),
    )
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
    _with_hf_retries(
        "Create/update dataset repo",
        lambda: api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        ),
    )
    _with_hf_retries(
        "Upload benchmark outputs",
        lambda: api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update {path_in_repo}",
        ),
    )
    _update_dataset_readme_leaderboard(
        api=api,
        repo_id=repo_id,
        token=token,
        leaderboard_md_path=local_dir / "model_comparison.md",
        artifact_path=path_in_repo,
    )
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}"
