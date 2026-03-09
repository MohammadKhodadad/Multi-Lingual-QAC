#!/usr/bin/env python3
"""
Multi-Lingual Chemical QAC pipeline.

For each part (raw extraction, then each language's preprocessed CSV),
asks: redo? (y/n). For languages: s = skip remaining.

Usage:
    uv run main.py              # interactive
    uv run main.py --yes        # no prompts, redo all
    uv run main.py --no-extraction   # skip extraction, only preprocess
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Add project root so imports work when run from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env from project root
from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")
import sys

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataloader.google_patents import (
    DEFAULT_LANGS,
    extract_chemistry_patents,
    extract_chemistry_patents_per_language,
    preprocess_ndjson_to_csv,
)

# Paths
RAW_NDJSON = _PROJECT_ROOT / "data" / "google_patents" / "chemistry_patents.ndjson"
PREPROCESSED_DIR = _PROJECT_ROOT / "data" / "google_patents" / "preprocessed"


def ask_interactive(prompt: str, default: str = "n") -> str:
    """Ask user, return lowercase first char (y/n/s)."""
    choice = input(prompt).strip().lower() or default
    return choice[0] if choice else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Lingual Chemical QAC: extract patents, preprocess to CSV."
    )
    parser.add_argument("--yes", "-y", action="store_true", help="No prompts; redo all")
    parser.add_argument("--no-extraction", action="store_true", help="Skip extraction; only preprocess")
    parser.add_argument("--limit", type=int, default=None, help="Max patents per language (pull 100 en, 100 de, ... into one NDJSON)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ----- Step 1: Raw extraction -----
    run_extraction = not args.no_extraction

    if run_extraction:
        if RAW_NDJSON.exists() and not args.yes:
            line_count = sum(1 for _ in RAW_NDJSON.open()) if RAW_NDJSON.stat().st_size > 0 else 0
            r = ask_interactive(
                f"Raw data exists ({line_count} records). Redo extraction? (y/n): ",
                "n",
            )
            if r != "y":
                run_extraction = False

        if run_extraction:
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                print("Error: Set GOOGLE_CLOUD_PROJECT in .env for extraction.")
                raise SystemExit(1)
            print("Running extraction...")
            if args.limit:
                extract_chemistry_patents_per_language(
                    project_id=project_id,
                    output_path=RAW_NDJSON,
                    limit_per_lang=args.limit,
                )
            else:
                extract_chemistry_patents(
                    project_id=project_id,
                    output_path=RAW_NDJSON,
                )

    if not RAW_NDJSON.exists():
        print(f"Error: Raw data not found at {RAW_NDJSON}. Run extraction first.")
        raise SystemExit(1)

    # ----- Step 2: Preprocess per language -----
    print("\nPreprocessing to CSV per language...")
    skip_remaining = False

    for lang in DEFAULT_LANGS:
        if skip_remaining:
            print(f"  Skipping {lang} (user chose skip remaining).")
            continue

        out_csv = PREPROCESSED_DIR / f"{lang}.csv"
        if out_csv.exists() and not args.yes:
            n = sum(1 for _ in out_csv.open()) - 1  # subtract header
            r = ask_interactive(
                f"  {lang}: preprocessed exists ({n} rows). Redo? (y/n/s=skip remaining): ",
                "n",
            )
            if r == "s":
                skip_remaining = True
                print(f"  Skipping {lang} and remaining.")
                continue
            if r != "y":
                print(f"  {lang}: skipped.")
                continue

        # Preprocess this language (per_lang_limit caps each CSV when extraction used --limit)
        preprocess_ndjson_to_csv(
            ndjson_path=RAW_NDJSON,
            output_dir=PREPROCESSED_DIR,
            languages=[lang],
            per_lang_limit=args.limit,
        )

    print("\nDone. Preprocessed CSVs in:", PREPROCESSED_DIR)


if __name__ == "__main__":
    main()
