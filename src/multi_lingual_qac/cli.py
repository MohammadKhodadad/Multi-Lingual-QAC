from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.pipeline import run_pipeline


def _normalize_hf_dataset_repo(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    marker = "huggingface.co/datasets/"
    if marker in raw:
        raw = raw.split(marker, 1)[1]
    return raw.strip().strip("/")


def parse_args() -> PipelineConfig:
    default_mteb_dataset_repo = ""
    default_mteb_local_corpus = "data/google_patents/corpus.csv"
    default_mteb_local_qac = "data/google_patents/qac/balanced_100_qac.csv"
    parser = argparse.ArgumentParser(
        description="Multi-Lingual Chemical QAC: extract patents, preprocess to CSV."
    )
    parser.add_argument("--yes", "-y", action="store_true", help="No prompts; redo all")
    parser.add_argument("--no-extraction", action="store_true", help="Skip extraction; only preprocess")
    parser.add_argument("--limit", type=int, default=None, help="Max patents per language (if omitted in interactive mode, you will be prompted)")
    parser.add_argument("--qa-sample", type=int, default=None, help="Sample size for Q&A generation (if omitted in interactive mode, you will be prompted; 0 = skip Q&A)")
    parser.add_argument("--qa-batch", action="store_true", help="Batch QA generation using worker threads based on available CPUs")
    parser.add_argument("--qa-no-batch", action="store_true", help="Disable batch QA generation")
    parser.add_argument("--push-hf", action="store_true", help="Push corpus + QAC to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID (e.g. username/multi-lingual-chemical-qac); required if --push-hf")
    parser.add_argument(
        "--evaluate-mteb",
        nargs="*",
        metavar="MODEL",
        help=(
            "Evaluate embedding models against the HF retrieval dataset via MTEB. "
            "If no models are provided, uses the built-in multilingual default set."
        ),
    )
    parser.add_argument(
        "--mteb-dataset-repo",
        type=str,
        default=default_mteb_dataset_repo,
        help=(
            "Optional Hugging Face dataset repo to evaluate with MTEB. "
            "If omitted, benchmarking uses local CSVs."
        ),
    )
    parser.add_argument(
        "--mteb-local-corpus-path",
        type=str,
        default=default_mteb_local_corpus,
        help=(
            "Local corpus CSV for MTEB benchmarking when no dataset repo is given "
            f"(default: {default_mteb_local_corpus})"
        ),
    )
    parser.add_argument(
        "--mteb-local-qac-path",
        type=str,
        default=default_mteb_local_qac,
        help=(
            "Local QAC CSV for MTEB benchmarking when no dataset repo is given "
            f"(default: {default_mteb_local_qac})"
        ),
    )
    parser.add_argument(
        "--mteb-output-dir",
        type=str,
        default=None,
        help="Directory for MTEB results and summary reports (default: reports/mteb)",
    )
    parser.add_argument(
        "--mteb-batch-size",
        type=int,
        default=32,
        help="Batch size passed to sentence-transformers encoding during MTEB evaluation",
    )
    parser.add_argument(
        "--mteb-include-mode-strategy",
        action="store_true",
        help="Also evaluate mode/strategy combination slices in addition to overall, mode, and strategy slices",
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
        help="Directory for generated comparison tables (default: reports/mteb_tables)",
    )
    parser.add_argument(
        "--upload-mteb-results",
        action="store_true",
        help="Upload generated MTEB comparison tables to a Hugging Face dataset repo",
    )
    parser.add_argument(
        "--mteb-upload-repo",
        type=str,
        default=None,
        help="Hugging Face dataset repo ID or URL for uploaded MTEB comparison tables",
    )
    args = parser.parse_args()
    qa_batch = None
    if args.qa_batch and args.qa_no_batch:
        parser.error("Use only one of --qa-batch or --qa-no-batch")
    if args.qa_batch:
        qa_batch = True
    elif args.qa_no_batch:
        qa_batch = False
    return PipelineConfig(
        yes=args.yes,
        no_extraction=args.no_extraction,
        limit=args.limit,
        qa_sample=args.qa_sample,
        qa_batch=qa_batch,
        push_hf=args.push_hf,
        hf_repo=args.hf_repo,
        evaluate_mteb_models=tuple(
            args.evaluate_mteb
            if args.evaluate_mteb is not None and len(args.evaluate_mteb) > 0
            else (
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "intfloat/multilingual-e5-large",
                "BAAI/bge-m3",
            )
            if args.evaluate_mteb is not None
            else ()
        ),
        mteb_dataset_repo=args.mteb_dataset_repo,
        mteb_local_corpus_path=args.mteb_local_corpus_path,
        mteb_local_qac_path=args.mteb_local_qac_path,
        mteb_output_dir=args.mteb_output_dir,
        mteb_batch_size=max(1, args.mteb_batch_size),
        generate_mteb_tables=args.generate_mteb_tables,
        mteb_results_dir=args.mteb_results_dir,
        mteb_tables_dir=args.mteb_tables_dir,
        mteb_include_mode_strategy=args.mteb_include_mode_strategy,
        upload_mteb_results=args.upload_mteb_results,
        mteb_upload_repo=args.mteb_upload_repo,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config = parse_args()
    if config.evaluate_mteb_models:
        from src.multi_lingual_qac.mteb import run_mteb_evaluation

        output_dir = config.mteb_output_dir or (project_root / "reports" / "mteb")
        summaries = run_mteb_evaluation(
            list(config.evaluate_mteb_models),
            dataset_repo=config.mteb_dataset_repo,
            local_corpus_path=config.mteb_local_corpus_path,
            local_qac_path=config.mteb_local_qac_path,
            output_dir=output_dir,
            batch_size=config.mteb_batch_size,
            include_mode_strategy=config.mteb_include_mode_strategy,
        )
        print("MTEB evaluation finished.")
        if config.mteb_dataset_repo:
            print(f"  Dataset: {config.mteb_dataset_repo}")
        else:
            print(f"  Local corpus: {config.mteb_local_corpus_path}")
            print(f"  Local QAC: {config.mteb_local_qac_path}")
        print(f"  Output: {output_dir}")
        for item in summaries:
            print(f"  [{item.slice_name}] {item.model_name}: {item.main_score:.4f}")
        return

    if config.generate_mteb_tables:
        from src.multi_lingual_qac.export.hf_upload import upload_benchmark_outputs
        from src.multi_lingual_qac.mteb import (
            DEFAULT_MTEB_TABLES_DIR,
            generate_mteb_comparison_tables,
        )

        results_dir = config.mteb_results_dir or (project_root / "reports" / "mteb")
        tables_dir = config.mteb_tables_dir or (project_root / DEFAULT_MTEB_TABLES_DIR)
        generated_dir = generate_mteb_comparison_tables(
            results_dir=results_dir,
            output_dir=tables_dir,
        )
        print("MTEB comparison tables generated.")
        print(f"  Source results: {results_dir}")
        print(f"  Output: {generated_dir}")

        if config.upload_mteb_results:
            repo_id = _normalize_hf_dataset_repo(
                config.mteb_upload_repo or config.mteb_dataset_repo
            )
            repo_tree_url = upload_benchmark_outputs(
                generated_dir,
                repo_id,
                path_in_repo="benchmark_outputs/mteb_tables",
            )
            print(f"  Hugging Face benchmark outputs: {repo_tree_url}")
        return

    paths = PipelinePaths.from_project_root(project_root)
    run_pipeline(config, paths)
