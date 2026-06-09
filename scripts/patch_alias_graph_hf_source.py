#!/usr/bin/env python3
"""Patch the already-published Alias-Graph benchmark with the exact
(query -> source publication) mapping.

The original export only stored concept-level `qrels` (every document about the
concept). This adds the missing piece: the single publication each query was
actually generated from, plus that publication's translations (the gold document
in every language). It re-pushes only the configs that change — `queries` (now
carrying `source_publication`), `qac` (same), and the new `source_qrels` — so the
large, unchanged `corpus`/`qrels`/`hard_negatives`/`concepts` are left in place.

Examples:
  python scripts/patch_alias_graph_hf_source.py --dry-run   # write updated parquet locally
  python scripts/patch_alias_graph_hf_source.py             # patch the HF repo (needs HF_TOKEN)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alias_graph.hf_export import push_alias_graph_to_hub

# Configs that gain/contain the source-publication information.
PATCH_CONFIGS = ["queries", "qac", "source_qrels"]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    p = argparse.ArgumentParser(description="Add (query -> source publication) to the Alias-Graph HF dataset.")
    p.add_argument("--repo", default="MehdiAstaraki/multi-lingual-qac-alias-graph")
    p.add_argument("--alias-json", type=Path, default=PROJECT_ROOT / "data/alias_graph/alias_graph.json")
    p.add_argument("--qac", type=Path, default=PROJECT_ROOT / "data/alias_graph/qac/concept_qa.csv")
    p.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/google_patents/multilingual_corpus.csv")
    p.add_argument("--dry-run", action="store_true", help="Write parquet locally; do not upload.")
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--all-configs", action="store_true",
        help="Rebuild and push every config instead of only the source-publication ones.",
    )
    args = p.parse_args()

    push_alias_graph_to_hub(
        alias_json=args.alias_json, qac_csv=args.qac, corpus_csv=args.corpus,
        repo_id=args.repo, private=args.private, dry_run=args.dry_run,
        chebi_cache_dir=PROJECT_ROOT / "data/chebi",
        only_configs=None if args.all_configs else PATCH_CONFIGS,
    )


if __name__ == "__main__":
    main()
