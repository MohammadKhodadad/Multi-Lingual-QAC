from __future__ import annotations

import os
from pathlib import Path

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.dataloaders.google_patents import (
    extract_chemistry_patents,
    extract_chemistry_patents_per_language,
    merge_corpus_csv,
    preprocess_ndjson_to_csv,
)
from src.multi_lingual_qac.export.hf_upload import push_to_hub
from src.multi_lingual_qac.qac_generation.openai_qa import run_qa_pipeline


def ask_interactive(prompt: str, default: str = "n") -> str:
    choice = input(prompt).strip().lower() or default
    return choice[0] if choice else default


def ask_text(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Please enter a non-empty value.")


def ask_int(prompt: str, *, allow_zero: bool = True) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < 0 or (value == 0 and not allow_zero):
            print("Please enter a valid non-negative integer.")
            continue
        return value


def _count_rows(path: Path) -> int:
    return sum(1 for _ in path.open()) - 1


def run_pipeline(config: PipelineConfig, paths: PipelinePaths) -> None:
    limit = config.limit
    qa_sample = config.qa_sample

    if not config.yes:
        if limit is None:
            entered_limit = ask_int(
                "Limit per language for extraction/preprocessing? Enter 0 for no limit: "
            )
            limit = None if entered_limit == 0 else entered_limit
        if qa_sample is None:
            qa_sample = ask_int(
                "How many corpus documents should be sampled for Q&A generation? Enter 0 to skip: "
            )
    else:
        if qa_sample is None:
            qa_sample = 50

    run_extraction = not config.no_extraction

    if run_extraction:
        if paths.raw_ndjson.exists() and not config.yes:
            line_count = sum(1 for _ in paths.raw_ndjson.open()) if paths.raw_ndjson.stat().st_size > 0 else 0
            redo = ask_interactive(f"Raw data exists ({line_count} records). Redo extraction? (y/n): ", "n")
            run_extraction = redo == "y"

        if run_extraction:
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                print("Error: Set GOOGLE_CLOUD_PROJECT in .env for extraction.")
                raise SystemExit(1)
            print("Running extraction...")
            if limit:
                extract_chemistry_patents_per_language(
                    project_id=project_id,
                    output_path=paths.raw_ndjson,
                    limit_per_lang=limit,
                )
            else:
                extract_chemistry_patents(
                    project_id=project_id,
                    output_path=paths.raw_ndjson,
                )

    if not paths.raw_ndjson.exists():
        print(f"Error: Raw data not found at {paths.raw_ndjson}. Run extraction first.")
        raise SystemExit(1)

    print("\nPreprocessing to CSV per language...")
    skip_remaining = False
    for lang in config.languages:
        if skip_remaining:
            print(f"  Skipping {lang} (user chose skip remaining).")
            continue

        out_csv = paths.preprocessed_dir / f"{lang}.csv"
        if out_csv.exists() and not config.yes:
            redo = ask_interactive(
                f"  {lang}: preprocessed exists ({_count_rows(out_csv)} rows). Redo? (y/n/s=skip remaining): ",
                "n",
            )
            if redo == "s":
                skip_remaining = True
                print(f"  Skipping {lang} and remaining.")
                continue
            if redo != "y":
                print(f"  {lang}: skipped.")
                continue

        preprocess_ndjson_to_csv(
            ndjson_path=paths.raw_ndjson,
            output_dir=paths.preprocessed_dir,
            languages=[lang],
            per_lang_limit=limit,
        )

    run_merge = True
    if paths.corpus_csv.exists() and not config.yes:
        redo = ask_interactive(f"Corpus exists ({_count_rows(paths.corpus_csv)} rows). Redo merge? (y/n): ", "n")
        run_merge = redo == "y"

    if run_merge:
        merge_corpus_csv(
            preprocessed_dir=paths.preprocessed_dir,
            output_path=paths.corpus_csv,
        )

    if qa_sample > 0:
        qac_csv = paths.qac_dir / "qac.csv"
        run_qa = True
        if qac_csv.exists() and not config.yes:
            redo = ask_interactive(f"QAC exists ({_count_rows(qac_csv)} rows). Redo Q&A generation? (y/n): ", "n")
            run_qa = redo == "y"
        if run_qa:
            try:
                run_qa_pipeline(
                    corpus_path=paths.corpus_csv,
                    output_dir=paths.qac_dir,
                    sample_size=qa_sample,
                )
            except ValueError as exc:
                print(f"Q&A generation skipped: {exc}")

    if config.push_hf:
        hf_repo = config.hf_repo
        if not hf_repo and not config.yes:
            hf_repo = ask_text(
                "Hugging Face repo ID for upload (e.g. username/multi-lingual-chemical-qac): "
            )
        if not hf_repo:
            print("Error: --hf-repo required when using --push-hf (e.g. --hf-repo username/multi-lingual-chemical-qac)")
            raise SystemExit(1)
        qac_csv = paths.qac_dir / "qac.csv"
        if not paths.corpus_csv.exists():
            print("Error: Corpus not found. Run pipeline first.")
            raise SystemExit(1)
        if not qac_csv.exists():
            print("Error: QAC not found. Run with --qa-sample > 0 first.")
            raise SystemExit(1)

        do_push = config.yes
        if not config.yes:
            do_push = ask_interactive(f"Push to {hf_repo}? (y/n): ", "n") == "y"
            if not do_push:
                print("Push skipped.")
        if do_push:
            push_to_hub(
                corpus_path=paths.corpus_csv,
                qac_path=qac_csv,
                repo_id=hf_repo,
            )

    print("\nDone.")
    print("  Preprocessed CSVs:", paths.preprocessed_dir)
    print("  Corpus:", paths.corpus_csv)
    if qa_sample > 0:
        print("  QAC:", paths.qac_dir / "qac.csv")
    if config.push_hf:
        hf_repo = config.hf_repo
        if hf_repo:
            print("  Hugging Face: https://huggingface.co/datasets/" + hf_repo)
