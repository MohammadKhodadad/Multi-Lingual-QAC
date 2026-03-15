from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_QAC_PATH = Path("/home/mohammad/Multi-Lingual-QAC/data/google_patents/qac/qac.csv")
DEFAULT_CORPUS_PATH = Path("/home/mohammad/Multi-Lingual-QAC/data/google_patents/corpus.csv")
DEFAULT_OUTPUT_PATH = Path("/home/mohammad/Multi-Lingual-QAC/data/google_patents/qac/qac_for_excel.csv")


def load_corpus_metadata(corpus_path: Path) -> dict[str, tuple[str, str]]:
    metadata: dict[str, tuple[str, str]] = {}
    with corpus_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            corpus_id = (row.get("id") or "").strip()
            if corpus_id:
                metadata[corpus_id] = (
                    (row.get("context") or "").strip(),
                    (row.get("language") or "").strip(),
                )
    return metadata


def export_qac_for_excel(qac_path: Path, corpus_path: Path, output_path: Path) -> tuple[int, int]:
    corpus_metadata = load_corpus_metadata(corpus_path)
    written = 0
    missing_contexts = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with qac_path.open(encoding="utf-8", newline="") as qac_handle, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(qac_handle)
        writer = csv.DictWriter(
            output_handle,
            fieldnames=["question", "answer", "corpus", "question_lang", "source_lang"],
        )
        writer.writeheader()

        for row in reader:
            corpus_id = (row.get("corpus_id") or "").strip()
            context, source_lang = corpus_metadata.get(corpus_id, ("", ""))
            if not context:
                missing_contexts += 1

            writer.writerow(
                {
                    "question": (row.get("question") or "").strip(),
                    "answer": (row.get("answer") or "").strip(),
                    "corpus": context,
                    "question_lang": (row.get("language") or "").strip(),
                    "source_lang": source_lang,
                }
            )
            written += 1

    return written, missing_contexts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export QAC data to a simple q,a,c CSV for Excel."
    )
    parser.add_argument("--qac", type=Path, default=DEFAULT_QAC_PATH, help="Path to qac.csv")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH, help="Path to corpus.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path for Excel-friendly q,a,c data",
    )
    args = parser.parse_args()

    written, missing_contexts = export_qac_for_excel(args.qac, args.corpus, args.output)
    print(f"Wrote {written} rows to {args.output}")
    if missing_contexts:
        print(f"Warning: {missing_contexts} rows had no matching context in the corpus.")


if __name__ == "__main__":
    main()
