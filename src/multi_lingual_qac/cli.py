from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.dataloaders.jrc_acquis import (
    JRC_ACQUIS_LANGS,
    count_jrc_acquis_input_files,
    download_jrc_acquis_archives,
)
from src.multi_lingual_qac.export.hf_upload import upload_benchmark_outputs
from src.multi_lingual_qac.mteb import (
    DEFAULT_MTEB_DATASET_REPO,
    DEFAULT_MTEB_MODELS,
    DEFAULT_MTEB_TABLES_DIR,
    generate_mteb_comparison_tables,
    run_mteb_evaluation,
)
from src.multi_lingual_qac.preprocess.corpus import (
    build_corpus_from_source,
    count_source_records,
    prepare_corpus_source,
)
from src.multi_lingual_qac.pipeline import ask_interactive, ask_text, run_pipeline
from src.multi_lingual_qac.qac_generation.label_wikidata_qrels import run_wikidata_qrels_labeling


def _normalize_source_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized not in {"epo", "wikidata", "jrc-acquis"}:
        raise argparse.ArgumentTypeError(f"Unsupported source: {value}")
    return normalized


def _normalize_hf_dataset_repo(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    marker = "huggingface.co/datasets/"
    if marker in raw:
        raw = raw.split(marker, 1)[1]
    return raw.strip().strip("/")


def _normalize_mteb_variant(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"multilingual", "cross_language"}:
        raise argparse.ArgumentTypeError(
            "Unsupported MTEB variant. Use `multilingual` or `cross_language`."
        )
    return normalized


def _normalize_jrc_qa_languages(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        parts = [part.strip().lower() for part in value.split(",")]
        for part in parts:
            if not part:
                continue
            if part not in JRC_ACQUIS_LANGS:
                raise argparse.ArgumentTypeError(
                    f"Unsupported JRC QA language: {part}. Supported: {', '.join(JRC_ACQUIS_LANGS)}"
                )
            if part not in seen:
                normalized.append(part)
                seen.add(part)
    return tuple(normalized) or None


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
    parser.add_argument(
        "--label-qrels",
        type=_normalize_source_name,
        default=None,
        metavar="SOURCE",
        help="Label multilingual retrieval qrels (WIKIDATA only; needs corpus_full + qac)",
    )
    parser.add_argument(
        "--label-qrels-batch-size",
        type=int,
        default=5,
        metavar="N",
        help="Passages per LLM judge call for --label-qrels (default: 5)",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="No prompts; redo all")
    parser.add_argument(
        "--no-extraction",
        action="store_true",
        help="Deprecated: source preparation is now a separate command",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional source-specific limit (for Wikidata, max selected entities)")
    parser.add_argument("--qa-sample", type=int, default=None, help="Sample size for Q&A generation (if omitted in interactive mode, you will be prompted; 0 = skip Q&A)")
    parser.add_argument("--qa-pairs-per-language", type=int, default=None, help="JRC-Acquis only: sampled source-document candidates per source language from multilingual CELEX groups before final QA selection")
    parser.add_argument("--qa-docs-per-language", type=int, default=None, help="JRC-Acquis only: retained source documents per language used for QA generation")
    parser.add_argument(
        "--jrc-qa-languages",
        nargs="+",
        metavar="LANG",
        help=(
            "JRC-Acquis only: optional language subset for QA sampling/generation "
            f"(supported: {', '.join(JRC_ACQUIS_LANGS)}; accepts space- or comma-separated values)"
        ),
    )
    parser.add_argument(
        "--jrc-synthetic-chinese",
        action="store_true",
        help="JRC-Acquis only: add synthetic Chinese translations for generated QA pairs",
    )
    parser.add_argument(
        "--no-jrc-synthetic-chinese",
        action="store_true",
        help="JRC-Acquis only: disable synthetic Chinese translations for generated QA pairs",
    )
    parser.add_argument("--qa-batch", action="store_true", help="Batch QA generation using worker threads based on available CPUs")
    parser.add_argument("--qa-no-batch", action="store_true", help="Disable batch QA generation")
    parser.add_argument("--push-hf", action="store_true", help="Push corpus + QAC to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID (e.g. username/multi-lingual-chemical-qac); required if --push-hf")
    parser.add_argument(
        "--evaluate-mteb",
        nargs="*",
        metavar="MODEL",
        help=(
            "Evaluate embedding models against the pushed HF retrieval dataset via MTEB. "
            "If no models are provided, uses the built-in multilingual default set."
        ),
    )
    parser.add_argument(
        "--mteb-dataset-repo",
        type=str,
        default=DEFAULT_MTEB_DATASET_REPO,
        help=f"Hugging Face dataset repo to evaluate with MTEB (default: {DEFAULT_MTEB_DATASET_REPO})",
    )
    parser.add_argument(
        "--mteb-output-dir",
        type=str,
        default=None,
        help="Directory for MTEB results and summary reports (default: reports/mteb)",
    )
    parser.add_argument(
        "--mteb-variant",
        type=_normalize_mteb_variant,
        default="multilingual",
        help="Benchmark variant to load from HF for MTEB evaluation (default: multilingual)",
    )
    parser.add_argument(
        "--mteb-batch-size",
        type=int,
        default=32,
        help="Batch size passed to sentence-transformers encoding during MTEB evaluation",
    )
    parser.add_argument(
        "--generate-mteb-tables",
        action="store_true",
        help="Generate model comparison tables from saved MTEB results without rerunning evaluation",
    )
    parser.add_argument(
        "--mteb-results-dir",
        type=str,
        default=None,
        help="Directory containing saved MTEB results and summary.json (default: reports/mteb)",
    )
    parser.add_argument(
        "--mteb-tables-dir",
        type=str,
        default=None,
        help=f"Directory for generated comparison tables (default: {DEFAULT_MTEB_TABLES_DIR})",
    )
    args = parser.parse_args()
    qa_batch = None
    jrc_synthetic_chinese = None
    if args.qa_batch and args.qa_no_batch:
        parser.error("Use only one of --qa-batch or --qa-no-batch")
    if args.jrc_synthetic_chinese and args.no_jrc_synthetic_chinese:
        parser.error("Use only one of --jrc-synthetic-chinese or --no-jrc-synthetic-chinese")
    if args.qa_batch:
        qa_batch = True
    elif args.qa_no_batch:
        qa_batch = False
    if args.jrc_synthetic_chinese:
        jrc_synthetic_chinese = True
    elif args.no_jrc_synthetic_chinese:
        jrc_synthetic_chinese = False
    return PipelineConfig(
        source=args.source,
        prepare_source=args.prepare_source,
        build_corpus=args.build_corpus,
        build_corpus_batch=args.build_corpus_batch,
        label_qrels=args.label_qrels,
        label_qrels_batch_size=max(1, args.label_qrels_batch_size),
        yes=args.yes,
        no_extraction=args.no_extraction,
        limit=args.limit,
        qa_sample=args.qa_sample,
        qa_pairs_per_language=args.qa_pairs_per_language,
        qa_docs_per_language=args.qa_docs_per_language,
        jrc_qa_languages=_normalize_jrc_qa_languages(args.jrc_qa_languages),
        jrc_synthetic_chinese=jrc_synthetic_chinese,
        qa_batch=qa_batch,
        push_hf=args.push_hf,
        hf_repo=args.hf_repo,
        evaluate_mteb_models=tuple(
            args.evaluate_mteb
            if args.evaluate_mteb is not None and len(args.evaluate_mteb) > 0
            else DEFAULT_MTEB_MODELS
            if args.evaluate_mteb is not None
            else ()
        ),
        mteb_dataset_repo=args.mteb_dataset_repo,
        mteb_variant=args.mteb_variant,
        mteb_output_dir=args.mteb_output_dir,
        mteb_batch_size=max(1, args.mteb_batch_size),
        generate_mteb_tables=args.generate_mteb_tables,
        mteb_results_dir=args.mteb_results_dir,
        mteb_tables_dir=args.mteb_tables_dir,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config = parse_args()
    if config.evaluate_mteb_models:
        output_dir = config.mteb_output_dir or (project_root / "reports" / "mteb")
        summaries = run_mteb_evaluation(
            list(config.evaluate_mteb_models),
            dataset_repo=config.mteb_dataset_repo,
            dataset_variant=config.mteb_variant,
            output_dir=output_dir,
            batch_size=config.mteb_batch_size,
        )
        print("MTEB evaluation finished.")
        print(f"  Dataset: {config.mteb_dataset_repo}")
        print(f"  Variant: {config.mteb_variant}")
        print(f"  Output: {output_dir}")
        for item in summaries:
            print(f"  {item.model_name}: {item.main_score:.4f}")
        return

    if config.generate_mteb_tables:
        results_dir = config.mteb_results_dir or (project_root / "reports" / "mteb")
        tables_dir = config.mteb_tables_dir or (project_root / DEFAULT_MTEB_TABLES_DIR)
        generated_dir = generate_mteb_comparison_tables(
            results_dir=results_dir,
            output_dir=tables_dir,
        )
        print("MTEB comparison tables generated.")
        print(f"  Source results: {results_dir}")
        print(f"  Output: {generated_dir}")
        repo_id = _normalize_hf_dataset_repo(config.mteb_dataset_repo)
        should_upload = config.yes
        if not config.yes:
            should_upload = (
                ask_interactive(
                    f"Do you want to upload the benchmark outputs to Hugging Face? (default: {repo_id}) (y/n): ",
                    "n",
                )
                == "y"
            )
        if should_upload:
            if not config.yes:
                entered_repo = input(
                    f"Hugging Face dataset repo ID or URL for benchmark outputs [{repo_id}]: "
                ).strip()
                if entered_repo:
                    repo_id = _normalize_hf_dataset_repo(entered_repo)
                elif not repo_id:
                    repo_id = _normalize_hf_dataset_repo(
                        ask_text(
                            "Hugging Face dataset repo ID or URL for benchmark outputs: "
                        )
                    )
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not hf_token:
                print("  Hugging Face benchmark outputs: skipped (HF_TOKEN not set)")
            else:
                repo_tree_url = upload_benchmark_outputs(
                    generated_dir,
                    repo_id,
                    path_in_repo="benchmark_outputs/mteb_tables",
                    token=hf_token,
                )
                print(f"  Hugging Face benchmark outputs: {repo_tree_url}")
        else:
            print("  Hugging Face benchmark outputs: skipped")
        return

    active_source = (
        config.prepare_source
        or config.build_corpus
        or config.label_qrels
        or config.source
    )
    paths = PipelinePaths.from_project_root(project_root, source=active_source)

    if config.label_qrels:
        if config.label_qrels != "wikidata":
            print("Error: --label-qrels currently supports only WIKIDATA.")
            raise SystemExit(1)
        qac_file = paths.qac_dir / "qac.csv"
        if not paths.corpus_full_csv.is_file():
            print(f"Error: Missing corpus full file: {paths.corpus_full_csv}")
            print("Run `uv run main.py --build-corpus WIKIDATA` first.")
            raise SystemExit(1)
        if not qac_file.is_file():
            print(f"Error: Missing QAC file: {qac_file}")
            print("Run `uv run main.py --source wikidata --qa-sample N` first.")
            raise SystemExit(1)
        run_wikidata_qrels_labeling(
            corpus_full_path=paths.corpus_full_csv,
            qac_path=qac_file,
            output_dir=paths.qac_dir,
            batch_size=config.label_qrels_batch_size,
        )
        print("  Queries:", paths.qac_dir / "queries.csv")
        print("  Qrels:", paths.qac_dir / "qrels.csv")
        return

    if config.prepare_source:
        prepare_workers = 1
        prepare_config = PipelineConfig(
            source=config.prepare_source,
            prepare_source=config.prepare_source,
            prepare_workers=prepare_workers,
            yes=config.yes,
            no_extraction=config.no_extraction,
            limit=config.limit,
            qa_sample=config.qa_sample,
            qa_batch=config.qa_batch,
            push_hf=config.push_hf,
            hf_repo=config.hf_repo,
            languages=config.languages,
        )
        paths.input_dir.mkdir(parents=True, exist_ok=True)
        if config.prepare_source == "jrc-acquis":
            raw_input_count = count_jrc_acquis_input_files(paths.input_dir)
            if raw_input_count == 0:
                should_download = config.yes or (
                    ask_interactive(
                        "No local JRC-Acquis raw files found. Download the official JRC-Acquis archives now? (y/n): ",
                        "y",
                    )
                    == "y"
                )
                if should_download:
                    try:
                        dl_stats = download_jrc_acquis_archives(paths.input_dir)
                    except ValueError as exc:
                        print(f"Error: {exc}")
                        raise SystemExit(1)
                    print(
                        f"Downloaded JRC-ACQUIS archives: {dl_stats['downloaded']} downloaded, "
                        f"{dl_stats['skipped_existing']} already present."
                    )
                else:
                    print("Created JRC-ACQUIS input folder.")
                    print("  Input:", paths.input_dir)
                    print("Place raw `jrc-<lang>.tgz` archives or extracted `.xml` files there, then rerun.")
                    raise SystemExit(0)
        existing_prepared = count_source_records(prepare_config, paths)
        run_prepare = True
        if existing_prepared and not config.yes:
            redo = ask_interactive(
                f"Prepared {config.prepare_source.upper()} data already exists ({existing_prepared} records). Redo and overwrite it? (y/n): ",
                "n",
            )
            run_prepare = redo == "y"
        if not run_prepare:
            print("Prepare skipped.")
            return
        if config.prepare_source == "jrc-acquis":
            use_multi_cpu = False
            if not config.yes:
                use_multi_cpu = (
                    ask_interactive(
                        "Do you want multi CPU for JRC-Acquis raw loading? (y/n): ",
                        "y",
                    )
                    == "y"
                )
            if use_multi_cpu:
                cpu_count = os.cpu_count() or 1
                prepare_workers = max(1, min(5, cpu_count // 2))
                print(f"Using {prepare_workers} worker(s) for JRC-Acquis prepare.")
            prepare_config = PipelineConfig(
                source=config.prepare_source,
                prepare_source=config.prepare_source,
                prepare_workers=prepare_workers,
                yes=config.yes,
                no_extraction=config.no_extraction,
                limit=config.limit,
                qa_sample=config.qa_sample,
                qa_batch=config.qa_batch,
                push_hf=config.push_hf,
                hf_repo=config.hf_repo,
                languages=config.languages,
            )
        try:
            stats = prepare_corpus_source(prepare_config, paths, overwrite=True)
        except ValueError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1)
        source_label = config.prepare_source.upper()
        if config.prepare_source == "epo":
            print(
                f"Prepared {source_label} source files:"
                f" {stats['xml_files']} XMLs from {stats['zip_files']} zip files"
                f" ({stats['skipped_existing']} skipped existing, {stats['bad_zips']} bad zips)."
            )
            print("  Input:", paths.input_dir)
            print("  XMLs:", paths.xml_dir)
        elif config.prepare_source == "jrc-acquis":
            print(
                f"Prepared {source_label} source files:"
                f" loaded {stats['documents_loaded']} XML documents"
                f" across {len(stats['languages'])} languages."
            )
            print("  Workers:", stats.get("workers", prepare_workers))
            print("  Input:", paths.input_dir)
            print("  Prepared:", paths.prepared_dir)
            print("  Raw JSONL:", paths.prepared_dir / "raw_documents.jsonl")
            print("  Stats:", paths.prepared_dir / "raw_load_stats.json")
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
        build_workers = 1
        if config.build_corpus == "jrc-acquis":
            use_multi_cpu = config.build_corpus_batch
            if not config.build_corpus_batch and not config.yes:
                use_multi_cpu = (
                    ask_interactive(
                        "Do you want multi CPU for JRC-Acquis document build? (y/n): ",
                        "y",
                    )
                    == "y"
                )
            if use_multi_cpu:
                cpu_count = os.cpu_count() or 1
                build_workers = max(1, min(5, cpu_count // 2))
                print(f"Using {build_workers} worker(s) for JRC-Acquis build.")
        build_config = PipelineConfig(
            source=config.build_corpus,
            prepare_source=config.prepare_source,
            build_corpus=config.build_corpus,
            build_corpus_batch=config.build_corpus_batch,
            build_workers=build_workers,
            yes=config.yes,
            no_extraction=config.no_extraction,
            limit=config.limit,
            qa_sample=config.qa_sample,
            qa_batch=config.qa_batch,
            push_hf=config.push_hf,
            hf_repo=config.hf_repo,
            languages=config.languages,
        )
        try:
            stats = build_corpus_from_source(build_config, paths)
        except ValueError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1)
        source_label = config.build_corpus.upper()
        if config.build_corpus == "wikidata":
            print(
                f"Built {source_label} corpus:"
                f" read {stats['documents_parsed']} Wikipedia pages"
                f" across {stats['xml_files']} language files"
                f" ({stats['parse_errors']} pages skipped empty or unchunkable)."
            )
            print(f"  Chunks: {stats['corpus_rows']} corpus rows")
        elif config.build_corpus == "jrc-acquis":
            print(
                f"Built {source_label} document corpus:"
                f" {stats['documents_written']} multilingual documents"
                f" from {stats['celex_total']} CELEX ids."
            )
            print(f"  Workers: {stats.get('workers', build_workers)}")
            print(f"  Multilingual CELEX ids: {stats['celex_multilingual']}")
            print(f"  All language pairs: {stats['pairs_written']}")
            print(f"  Multilingual docs: {stats['multilingual_docs_written']}")
            print(f"  QA candidates: {stats['qa_candidates_written']}")
            print(f"  Inspection rows: {stats['inspection_rows_written']}")
            print("  Pairs:", paths.preprocessed_dir / "document_pairs_all.csv")
            print("  Stats:", paths.preprocessed_dir / "document_corpus_stats.json")
            print("  Multilingual corpus:", paths.preprocessed_dir / "corpus_multilingual.csv")
            print("  Multilingual full:", paths.preprocessed_dir / "corpus_multilingual_full.csv")
            print("  QA candidates:", paths.preprocessed_dir / "corpus_qa_candidates.csv")
            print("  Inspection sample:", paths.preprocessed_dir / "inspection_sample.csv")
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
