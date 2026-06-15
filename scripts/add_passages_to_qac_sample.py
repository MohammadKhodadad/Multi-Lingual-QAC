"""Add a `passages` column to a QAC sample CSV and blank the grading values.

For every row, look up its `publication_number` in the EPO multilingual corpus
and attach the list of source passages (one per available language) as a
JSON-encoded list of ``{"language", "passage"}`` dicts. The passage text mirrors
``_build_all_passages_text`` in the QAC generator: ``context or abstract or title``.

Then blank out the LLM-judge grading cells (faith_*, qual_*, total_score) while
keeping the column headers, so the sample becomes a clean, ungraded artifact.

The transform overwrites the sample CSV in place.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO / "data" / "EPO" / "qac" / "qac_sample_20_balanced.csv"
CORPUS_CSV = REPO / "data" / "EPO" / "multilingual_corpus.csv"

# Order passages by this language priority; unknown langs keep corpus order after.
LANG_PRIORITY = ["en", "de", "fr"]


def build_passages_by_pub(corpus: pd.DataFrame) -> dict[str, str]:
    """Map publication_number -> JSON list of {"language", "passage"} dicts."""
    out: dict[str, str] = {}
    for pub, group in corpus.groupby("publication_number", sort=False):
        entries = []
        for row in group.itertuples():
            passage = (row.context or row.abstract or row.title or "").strip()
            if not passage:
                continue
            entries.append({"language": row.language, "passage": passage})
        entries.sort(
            key=lambda e: LANG_PRIORITY.index(e["language"])
            if e["language"] in LANG_PRIORITY
            else len(LANG_PRIORITY)
        )
        out[pub] = json.dumps(entries, ensure_ascii=False)
    return out


def main() -> None:
    sample = pd.read_csv(SAMPLE_CSV, dtype=str, keep_default_na=False)
    corpus = pd.read_csv(CORPUS_CSV, dtype=str, keep_default_na=False)

    passages_by_pub = build_passages_by_pub(corpus)

    missing = sorted(set(sample["publication_number"]) - passages_by_pub.keys())
    if missing:
        raise SystemExit(f"No corpus passages for publication numbers: {missing}")

    # Add the passages column (appended last).
    sample["passages"] = sample["publication_number"].map(passages_by_pub)

    # Blank the grading values, keeping the headers.
    grade_cols = [
        c
        for c in sample.columns
        if c.startswith(("faith_", "qual_")) or c == "total_score"
    ]
    sample[grade_cols] = ""

    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"Wrote {SAMPLE_CSV} ({len(sample)} rows)")
    print(f"Added `passages` column; blanked {len(grade_cols)} grade columns: {grade_cols}")


if __name__ == "__main__":
    main()
