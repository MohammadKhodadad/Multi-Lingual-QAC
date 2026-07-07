"""
Combine the per-model question files in reports/qa_generation_model_comparison into a
single side-by-side CSV.

Each generations__<model>.csv holds one row per document (the same 30 documents, same
order) with that model's best question plus scores/tokens/cost. This script gathers them
into one file:

  - metadata (from selected_documents.csv)
  - two columns per model: `<label>` (best_question) and `<label>_answer` (best_answer)
  - a `passages` column: the document's per-language passages as a JSON list of
    language-tagged strings (re-derived from the corpus, since they are not stored in the
    per-model CSVs)

Scores, tokens and cost are intentionally dropped.

Usage:
    .venv/bin/python scripts/combine_qa_questions.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

csv.field_size_limit(sys.maxsize)

# Reuse the corpus loader so passages are grouped exactly like the generation pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.multi_lingual_qac.qac_generation.multilingual_qa import (  # noqa: E402
    load_multilingual_corpus,
)

# Model labels, in the same order as the comparison script (one output column each).
MODEL_LABELS = ["gpt-5-mini", "gpt-5.4-mini", "sonnet-4.6", "grok-4.3", "gemini-3.5-flash", "qwen3.6-35b-a3b"]

# Metadata columns carried over from selected_documents.csv.
META_FIELDS = [
    "publication_number", "mode", "strategy_name",
    "question_language", "context_language", "n_passage_chars",
]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _passages_list(rows: List[Dict[str, Any]]) -> List[str]:
    """Per-language passages as language-tagged strings, mirroring _build_all_passages_text
    (one entry per language with non-empty context)."""
    parts: List[str] = []
    for r in rows:
        lang = (r.get("language") or "?").upper()
        ctx = r.get("context") or r.get("abstract") or r.get("title", "")
        if ctx.strip():
            parts.append(f"[{lang}] {ctx.strip()}")
    return parts


def combine(*, input_dir: Path, corpus_path: Path, output_path: Path) -> None:
    docs = _read_rows(input_dir / "selected_documents.csv")

    # {publication_number: best_question} and {publication_number: best_answer} per model.
    questions: Dict[str, Dict[str, str]] = {}
    answers: Dict[str, Dict[str, str]] = {}
    for label in MODEL_LABELS:
        rows = _read_rows(input_dir / f"generations__{label}.csv")
        questions[label] = {r["publication_number"]: r.get("best_question", "") for r in rows}
        answers[label] = {r["publication_number"]: r.get("best_answer", "") for r in rows}

    print(f"Loading corpus from {corpus_path} ...")
    corpus_groups = load_multilingual_corpus(corpus_path)

    # Per model: question column (bare label) followed immediately by its answer column.
    model_fields = [c for label in MODEL_LABELS for c in (label, f"{label}_answer")]
    fieldnames = META_FIELDS + model_fields + ["passages"]
    n_missing_corpus = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for doc in docs:
            pub = doc["publication_number"]
            record: Dict[str, Any] = {k: doc.get(k, "") for k in META_FIELDS}
            for label in MODEL_LABELS:
                record[label] = questions[label].get(pub, "")
                record[f"{label}_answer"] = answers[label].get(pub, "")
            corpus_rows = corpus_groups.get(pub)
            if not corpus_rows:
                n_missing_corpus += 1
            record["passages"] = json.dumps(_passages_list(corpus_rows or []), ensure_ascii=False)
            writer.writerow(record)

    print(f"Wrote {len(docs)} rows -> {output_path}")
    if n_missing_corpus:
        print(f"  warning: {n_missing_corpus} document(s) not found in corpus (empty passages)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("reports/qa_generation_model_comparison"))
    parser.add_argument("--corpus", type=Path, default=Path("data/google_patents/multilingual_corpus.csv"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default: <input-dir>/all_questions_by_model.csv).")
    args = parser.parse_args()

    output_path = args.output or (args.input_dir / "all_questions_by_model.csv")
    combine(input_dir=args.input_dir, corpus_path=args.corpus, output_path=output_path)


if __name__ == "__main__":
    main()
