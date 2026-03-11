from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.pipeline import run_pipeline


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Multi-Lingual Chemical QAC: extract patents, preprocess to CSV."
    )
    parser.add_argument("--yes", "-y", action="store_true", help="No prompts; redo all")
    parser.add_argument("--no-extraction", action="store_true", help="Skip extraction; only preprocess")
    parser.add_argument("--limit", type=int, default=None, help="Max patents per language (if omitted in interactive mode, you will be prompted)")
    parser.add_argument("--qa-sample", type=int, default=None, help="Sample size for Q&A generation (if omitted in interactive mode, you will be prompted; 0 = skip Q&A)")
    parser.add_argument("--push-hf", action="store_true", help="Push corpus + QAC to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID (e.g. username/multi-lingual-chemical-qac); required if --push-hf")
    args = parser.parse_args()
    return PipelineConfig(
        yes=args.yes,
        no_extraction=args.no_extraction,
        limit=args.limit,
        qa_sample=args.qa_sample,
        push_hf=args.push_hf,
        hf_repo=args.hf_repo,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config = parse_args()
    paths = PipelinePaths.from_project_root(project_root)
    run_pipeline(config, paths)
