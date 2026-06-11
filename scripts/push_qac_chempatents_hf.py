#!/usr/bin/env python3
"""Push a multilingual QAC retrieval benchmark to the Hugging Face Hub.

Builds the MTEB-style configs (``corpus``, ``queries``, ``qrels``, ``qac``, plus
the ``cross_language-*`` variants) from a best-only QAC file and a multilingual
corpus, and publishes them to a Hugging Face dataset repo (default:
``MehdiAstaraki/multi-lingual-qac-chem-patents``).

The best-only file already contains the full benchmark — previously published
questions plus any newly generated ones — so each push REPLACES the existing
configs on the Hub (one row per query becomes a query + qrel; passing the
all-candidates file instead would create duplicate queries, so always push the
best-only file).

Examples:
  # Google Patents benchmark
  python scripts/push_qac_chempatents_hf.py

  # EPO benchmark
  python scripts/push_qac_chempatents_hf.py \
      --repo MehdiAstaraki/multi-lingual-qac-epo \
      --qac data/EPO/qac/qac_epo_best.csv \
      --corpus data/EPO/multilingual_corpus.csv

Requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in environment / .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_lingual_qac.export.hf_upload import push_to_hub


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    p = argparse.ArgumentParser(
        description="Publish a multilingual QAC benchmark to Hugging Face."
    )
    p.add_argument(
        "--repo",
        default="MehdiAstaraki/multi-lingual-qac-chem-patents",
        help="Target HF dataset repo (default: MehdiAstaraki/multi-lingual-qac-chem-patents).",
    )
    p.add_argument(
        "--qac",
        type=Path,
        default=PROJECT_ROOT / "data/google_patents/qac/qac_chempatents_best.csv",
        help="Best-only QAC CSV (one row per query) -> queries/qrels/qac configs.",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data/google_patents/multilingual_corpus.csv",
        help="Multilingual corpus CSV (retrieval haystack / gold docs) -> corpus config.",
    )
    p.add_argument("--private", action="store_true", help="Create/keep the dataset repo private.")
    args = p.parse_args()

    if not args.qac.exists():
        raise SystemExit(f"QAC file not found: {args.qac}")
    if not args.corpus.exists():
        raise SystemExit(f"Corpus file not found: {args.corpus}")

    print(f"Publishing QAC benchmark -> {args.repo}")
    print(f"  queries/qrels/qac from: {args.qac}")
    print(f"  corpus from:            {args.corpus}")
    print("  (this REPLACES the existing configs on the Hub)")

    url = push_to_hub(
        corpus_path=args.corpus,
        qac_path=args.qac,
        repo_id=args.repo,
        private=args.private,
    )
    print(f"Done. {url}")


if __name__ == "__main__":
    main()
