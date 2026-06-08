#!/usr/bin/env python3
"""Push the Alias-Graph Retrieval benchmark to the Hugging Face Hub.

Examples:
  python scripts/push_alias_graph_hf.py --dry-run            # build parquet locally
  python scripts/push_alias_graph_hf.py                      # publish (needs HF_TOKEN)
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


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    p = argparse.ArgumentParser(description="Publish the Alias-Graph benchmark to Hugging Face.")
    p.add_argument("--repo", default="MehdiAstaraki/multi-lingual-qac-alias-graph")
    p.add_argument("--alias-json", type=Path, default=PROJECT_ROOT / "data/alias_graph/alias_graph.json")
    p.add_argument("--qac", type=Path, default=PROJECT_ROOT / "data/alias_graph/qac/concept_qa.csv")
    p.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/google_patents/multilingual_corpus.csv")
    p.add_argument("--dry-run", action="store_true", help="Write parquet locally; do not upload.")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    push_alias_graph_to_hub(
        alias_json=args.alias_json, qac_csv=args.qac, corpus_csv=args.corpus,
        repo_id=args.repo, private=args.private, dry_run=args.dry_run,
        chebi_cache_dir=PROJECT_ROOT / "data/chebi",
    )


if __name__ == "__main__":
    main()
