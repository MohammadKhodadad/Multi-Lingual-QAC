from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.preprocess.corpus import (
    build_corpus_from_source,
    prepare_corpus_source,
)
from src.multi_lingual_qac.pipeline import run_pipeline


def _normalize_source_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"epo", "wikidata"}:
        raise argparse.ArgumentTypeError(f"Unsupported source: {value}")
    return normalized


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Multi-Lingual Chemical QAC: prepare source data and build QAC."
    )
    parser.add_argument(
        "--source",
        type=_normalize_source_name,
        default="epo",
        help="Source pipeline to run for the main workflow",
    )
    parser.add_argument(
        "--prepare-source",
        type=_normalize_source_name,
        default=None,
        metavar="SOURCE",
        help="Prepare raw source artifacts only, e.g. `--prepare-source EPO`",
    )
    parser.add_argument(
        "--build-corpus",
        type=_normalize_source_name,
        default=None,
        metavar="SOURCE",
        help="Build corpus files only, e.g. `--build-corpus EPO`",
    )
    parser.add_argument(
        "--build-corpus-batch",
        action="store_true",
        help="Build corpus using multiple CPU workers (default: single CPU)",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="No prompts; redo all")
    parser.add_argument(
        "--no-extraction",
        action="store_true",
        help="Deprecated: source preparation is now a separate command",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional source-specific limit (for Wikidata, max selected entities)")
    parser.add_argument("--qa-sample", type=int, default=None, help="Sample size for Q&A generation (if omitted in interactive mode, you will be prompted; 0 = skip Q&A)")
    parser.add_argument("--qa-batch", action="store_true", help="Batch QA generation using worker threads based on available CPUs")
    parser.add_argument("--qa-no-batch", action="store_true", help="Disable batch QA generation")
    parser.add_argument("--push-hf", action="store_true", help="Push corpus + QAC to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID (e.g. username/multi-lingual-chemical-qac); required if --push-hf")
    args = parser.parse_args()
    qa_batch = None
    if args.qa_batch and args.qa_no_batch:
        parser.error("Use only one of --qa-batch or --qa-no-batch")
    if args.qa_batch:
        qa_batch = True
    elif args.qa_no_batch:
        qa_batch = False
    return PipelineConfig(
        source=args.source,
        prepare_source=args.prepare_source,
        build_corpus=args.build_corpus,
        build_corpus_batch=args.build_corpus_batch,
        yes=args.yes,
        no_extraction=args.no_extraction,
        limit=args.limit,
        qa_sample=args.qa_sample,
        qa_batch=qa_batch,
        push_hf=args.push_hf,
        hf_repo=args.hf_repo,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config = parse_args()
    active_source = config.prepare_source or config.build_corpus or config.source
    paths = PipelinePaths.from_project_root(project_root, source=active_source)

    if config.prepare_source:
        prepare_config = PipelineConfig(
            source=config.prepare_source,
            prepare_source=config.prepare_source,
            yes=config.yes,
            no_extraction=config.no_extraction,
            limit=config.limit,
            qa_sample=config.qa_sample,
            qa_batch=config.qa_batch,
            push_hf=config.push_hf,
            hf_repo=config.hf_repo,
            languages=config.languages,
        )
        stats = prepare_corpus_source(prepare_config, paths, overwrite=True)
        source_label = config.prepare_source.upper()
        if config.prepare_source == "epo":
            print(
                f"Prepared {source_label} source files:"
                f" {stats['xml_files']} XMLs from {stats['zip_files']} zip files"
                f" ({stats['skipped_existing']} skipped existing, {stats['bad_zips']} bad zips)."
            )
            print("  Input:", paths.input_dir)
            print("  XMLs:", paths.xml_dir)
        else:
            print(
                f"Prepared {source_label} source files:"
                f" {stats['selected_entities']} entities"
                f" -> {stats['pages_fetched']} multilingual Wikipedia pages"
                f" across {stats['languages_with_pages']} languages."
            )
            print("  Data:", paths.data_dir)
            print("  Prepared:", paths.prepared_dir)
            print("  Entities:", paths.prepared_dir / "entities.csv")
            print("  Pages:", paths.raw_pages_dir)
            print("  Coverage:", paths.prepared_dir / "coverage_report.json")
        return

    if config.build_corpus:
        build_config = PipelineConfig(
            source=config.build_corpus,
            prepare_source=config.prepare_source,
            build_corpus=config.build_corpus,
            build_corpus_batch=config.build_corpus_batch,
            yes=config.yes,
            no_extraction=config.no_extraction,
            limit=config.limit,
            qa_sample=config.qa_sample,
            qa_batch=config.qa_batch,
            push_hf=config.push_hf,
            hf_repo=config.hf_repo,
            languages=config.languages,
        )
        stats = build_corpus_from_source(build_config, paths)
        source_label = config.build_corpus.upper()
        if config.build_corpus == "wikidata":
            print(
                f"Built {source_label} corpus:"
                f" read {stats['documents_parsed']} Wikipedia pages"
                f" across {stats['xml_files']} language files"
                f" ({stats['parse_errors']} pages skipped empty or unchunkable)."
            )
            print(f"  Chunks: {stats['corpus_rows']} corpus rows")
        else:
            print(
                f"Built {source_label} corpus:"
                f" parsed {stats['documents_parsed']} documents"
                f" from {stats['xml_files']} XML files"
                f" ({stats['parse_errors']} parse errors)."
            )
            if stats.get("workers"):
                print(f"  Workers used: {stats['workers']}")
            print(
                f"  Chemistry-kept documents: {stats['documents_kept']}"
                f" -> {stats['corpus_rows']} corpus rows"
            )
            print(f"  All parsed rows: {stats['all_rows']}")
            print("  Parsed records:", paths.preprocessed_dir / "all_epo_records.csv")
        print("  Corpus (full):", paths.corpus_full_csv)
        print("  Corpus (MTEB):", paths.corpus_csv)
        return

    run_pipeline(config, paths)
