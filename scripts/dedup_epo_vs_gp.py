"""Remove EPO corpus rows whose patent already exists in the Google Patents corpus.

Dedup key: country_code + bare doc-number (e.g. EP_4634118).
When a duplicate is found, the Google Patents version is kept and the EPO rows
are dropped. The EPO CSV is modified in-place.

Usage:
    uv run python scripts/dedup_epo_vs_gp.py \
        --epo-csv data/EPO/multilingual_corpus.csv \
        --gp-csv data/google_patents/multilingual_corpus.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def _dedup_key(publication_number: str, country_code: str) -> str:
    m = re.search(r"(\d{5,})", publication_number)
    bare = m.group(1) if m else publication_number
    return f"{country_code}_{bare}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove EPO rows that duplicate Google Patents entries (in-place).",
    )
    parser.add_argument("--epo-csv", type=Path, required=True, help="Path to EPO multilingual corpus CSV")
    parser.add_argument("--gp-csv", type=Path, required=True, help="Path to Google Patents multilingual corpus CSV")
    args = parser.parse_args()

    if not args.gp_csv.exists():
        sys.exit(f"Google Patents CSV not found: {args.gp_csv}")
    if not args.epo_csv.exists():
        sys.exit(f"EPO CSV not found: {args.epo_csv}")

    gp_keys: set[str] = set()
    with args.gp_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gp_keys.add(_dedup_key(row["publication_number"], row["country_code"]))

    with args.epo_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        epo_rows = list(reader)

    kept: list[dict[str, str]] = []
    removed_pubs: set[str] = set()
    removed_rows = 0
    for row in epo_rows:
        key = _dedup_key(row["publication_number"], row["country_code"])
        if key in gp_keys:
            removed_pubs.add(key)
            removed_rows += 1
        else:
            kept.append(row)

    with args.epo_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    kept_pubs = len({_dedup_key(r["publication_number"], r["country_code"]) for r in kept})
    print(
        f"Removed {len(removed_pubs)} duplicate pub(s) ({removed_rows} rows) from {args.epo_csv}. "
        f"{kept_pubs} pubs ({len(kept)} rows) remain."
    )
    if removed_pubs:
        for key in sorted(removed_pubs):
            print(f"  - {key}")


if __name__ == "__main__":
    main()
