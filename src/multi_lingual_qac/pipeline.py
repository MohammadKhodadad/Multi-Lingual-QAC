from __future__ import annotations

from pathlib import Path

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.export.hf_upload import push_to_hub
from src.multi_lingual_qac.preprocess.corpus import count_source_records
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
    qa_sample = config.qa_sample
    qa_batch = config.qa_batch
    prepared_label = "XMLs" if config.source == "epo" else "Prepared source"
    prepared_path = paths.xml_dir if config.source == "epo" else paths.prepared_dir

    if not config.yes:
        if qa_sample is None:
            qa_sample = ask_int(
                "How many corpus documents should be sampled for Q&A generation? Enter 0 to skip: "
            )
        if qa_sample > 0 and qa_batch is None:
            qa_batch = (
                ask_interactive(
                    "Do you want to batch create QAs using available CPUs? (y/n): ",
                    "y",
                )
                == "y"
            )
    else:
        if qa_sample is None:
            qa_sample = 50
        if qa_batch is None:
            qa_batch = False

    xml_count = count_source_records(config, paths)
    if not xml_count:
        print(f"Error: No prepared source files found at {prepared_path}.")
        print(f"Run `uv run main.py --prepare-source {config.source.upper()}` first.")
        raise SystemExit(1)

    if not paths.corpus_csv.exists() or not paths.corpus_full_csv.exists():
        print("\nPrepared source artifacts are available.")
        print(f"  {prepared_label}:", prepared_path)
        print(f"Run `uv run main.py --build-corpus {config.source.upper()}` to create the corpus.")
        return

    if qa_sample > 0:
        qac_csv = paths.qac_dir / "qac.csv"
        run_qa = True
        if qac_csv.exists() and not config.yes:
            redo = ask_interactive(
                f"QAC already exists ({_count_rows(qac_csv)} rows). Regenerate Q&A and overwrite it? (y/n): ",
                "n",
            )
            run_qa = redo == "y"
        if run_qa:
            try:
                run_qa_pipeline(
                    corpus_path=paths.corpus_full_csv,
                    output_dir=paths.qac_dir,
                    sample_size=qa_sample,
                    batch_mode=bool(qa_batch),
                )
            except ValueError as exc:
                print(f"Q&A generation skipped: {exc}")

    qac_csv = paths.qac_dir / "qac.csv"
    hf_repo = config.hf_repo
    should_push = config.push_hf

    if (
        not config.yes
        and not should_push
        and paths.corpus_csv.exists()
        and qac_csv.exists()
    ):
        should_push = ask_interactive(
            "Data is ready. Do you want to push it to Hugging Face? (y/n): ",
            "n",
        ) == "y"

    if should_push:
        if not paths.corpus_csv.exists():
            print("Error: Corpus not found. Run pipeline first.")
            raise SystemExit(1)
        if not qac_csv.exists():
            print("Error: QAC not found. Run with --qa-sample > 0 first.")
            raise SystemExit(1)

        if not hf_repo and not config.yes:
            hf_repo = ask_text(
                "Hugging Face repo ID for upload (e.g. username/multi-lingual-chemical-qac): "
            )
        if not hf_repo:
            print("Error: --hf-repo required when using --push-hf (e.g. --hf-repo username/multi-lingual-chemical-qac)")
            raise SystemExit(1)

        if config.push_hf and not config.yes:
            confirmed = ask_interactive(f"Push to {hf_repo}? (y/n): ", "n") == "y"
            if not confirmed:
                print("Push skipped.")
                should_push = False
        if should_push:
            push_to_hub(
                corpus_path=paths.corpus_csv,
                qac_path=qac_csv,
                repo_id=hf_repo,
            )

    print("\nDone.")
    print(f"  {prepared_label}:", prepared_path)
    print("  Corpus (MTEB):", paths.corpus_csv)
    print("  Corpus (full):", paths.corpus_full_csv)
    if qa_sample > 0:
        print("  QAC:", paths.qac_dir / "qac.csv")
    if should_push and hf_repo:
        print("  Hugging Face: https://huggingface.co/datasets/" + hf_repo)
