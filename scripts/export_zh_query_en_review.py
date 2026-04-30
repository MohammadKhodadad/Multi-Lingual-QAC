from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_SOURCE = "jrc-acquis"


def infer_language(value: str) -> str:
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


def csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def default_paths(project_root: Path, source: str) -> tuple[Path, Path, Path]:
    data_dir = project_root / "data" / source.strip().upper()
    qac_path = data_dir / "qac" / "qac.csv"
    corpus_path = data_dir / "corpus.csv"
    output_path = project_root / "reports" / source.strip().lower() / "zh_query_en_corpus_review.csv"
    return qac_path, corpus_path, output_path


def load_corpus(corpus_path: Path) -> dict[str, dict[str, str]]:
    csv_field_size_limit()
    corpus_by_id: dict[str, dict[str, str]] = {}
    with corpus_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            corpus_id = (row.get("id") or row.get("_id") or "").strip()
            if not corpus_id:
                continue
            corpus_by_id[corpus_id] = {
                "title": (row.get("title") or "").strip(),
                "text": (row.get("context") or row.get("text") or row.get("abstract") or "").strip(),
                "language": (
                    (row.get("language") or row.get("lang") or "").strip().lower()
                    or infer_language(corpus_id)
                ),
            }
    return corpus_by_id


def load_qac_rows(qac_path: Path) -> list[dict[str, str]]:
    csv_field_size_limit()
    with qac_path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def linked_corpus_ids(row: dict[str, str]) -> list[str]:
    raw = (row.get("linked_corpus_ids_json") or "").strip()
    if not raw:
        fallback = (row.get("corpus_id") or "").strip()
        return [fallback] if fallback else []
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if not isinstance(parsed, list):
        fallback = (row.get("corpus_id") or "").strip()
        return [fallback] if fallback else []
    linked: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        corpus_id = str(item).strip()
        if corpus_id and corpus_id not in seen:
            linked.append(corpus_id)
            seen.add(corpus_id)
    return linked


def build_query_id(row: dict[str, str], row_idx: int) -> str:
    lang = (row.get("language") or "").strip().lower()
    corpus_id = (row.get("corpus_id") or "").strip()
    query_id_hint = (row.get("query_id_hint") or "").strip()
    if query_id_hint:
        return f"{query_id_hint}_q_{lang}"
    if corpus_id:
        return f"{corpus_id}_q_{lang}_{row_idx}"
    return f"q_{row_idx}_{lang}"


def pick_english_corpus_id(row: dict[str, str], corpus_by_id: dict[str, dict[str, str]]) -> str:
    for corpus_id in linked_corpus_ids(row):
        corpus_meta = corpus_by_id.get(corpus_id)
        corpus_language = (corpus_meta or {}).get("language") or infer_language(corpus_id)
        if corpus_language == "en":
            return corpus_id
    return ""


def export_zh_query_en_corpus_review(
    *,
    qac_path: Path,
    corpus_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    qac_rows = load_qac_rows(qac_path)
    corpus_by_id = load_corpus(corpus_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "source_corpus_id",
        "english_corpus_id",
        "query_language",
        "english_corpus_language",
        "question",
        "answer",
        "english_corpus_title",
        "english_corpus_text",
        "is_synthetic_translation",
        "faithfullness",
        "quality",
        "note",
    ]

    written = 0
    missing_english = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, row in enumerate(qac_rows):
            query_language = (row.get("language") or "").strip().lower()
            if query_language != "zh":
                continue

            english_corpus_id = pick_english_corpus_id(row, corpus_by_id)
            if not english_corpus_id:
                missing_english += 1
                continue

            english_corpus = corpus_by_id.get(english_corpus_id, {})
            writer.writerow(
                {
                    "query_id": build_query_id(row, row_idx),
                    "source_corpus_id": (row.get("corpus_id") or "").strip(),
                    "english_corpus_id": english_corpus_id,
                    "query_language": query_language,
                    "english_corpus_language": (english_corpus.get("language") or "en").strip(),
                    "question": (row.get("question") or "").strip(),
                    "answer": (row.get("answer") or "").strip(),
                    "english_corpus_title": (english_corpus.get("title") or "").strip(),
                    "english_corpus_text": (english_corpus.get("text") or "").strip(),
                    "is_synthetic_translation": (row.get("is_synthetic_translation") or "").strip(),
                    "faithfullness": "",
                    "quality": "",
                    "note": "",
                }
            )
            written += 1

    return written, missing_english


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a review CSV for Chinese queries paired with an English relevant corpus "
            "from the linked qrels-style corpus ids."
        )
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Dataset source name under data/ (default: jrc-acquis)",
    )
    parser.add_argument("--qac", type=Path, help="Path to qac.csv")
    parser.add_argument("--corpus", type=Path, help="Path to corpus.csv")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    default_qac, default_corpus, default_output = default_paths(project_root, args.source)
    qac_path = args.qac or default_qac
    corpus_path = args.corpus or default_corpus
    output_path = args.output or default_output

    if not qac_path.is_file():
        raise FileNotFoundError(f"Missing QAC file: {qac_path}")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    written, missing_english = export_zh_query_en_corpus_review(
        qac_path=qac_path,
        corpus_path=corpus_path,
        output_path=output_path,
    )
    print(f"Wrote {written} Chinese query review rows to {output_path}")
    if missing_english:
        print(f"Skipped {missing_english} Chinese queries with no English linked corpus.")


if __name__ == "__main__":
    main()
