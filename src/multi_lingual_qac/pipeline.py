from __future__ import annotations

from pathlib import Path

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.export.hf_upload import push_to_hub
from src.multi_lingual_qac.preprocess.corpus import count_source_records
from src.multi_lingual_qac.qac_generation.jrc_acquis import prepare_jrc_qa_inputs
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
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh) - 1


def _qa_corpus_for_push(config: PipelineConfig, paths: PipelinePaths) -> Path:
    if config.source == "jrc-acquis":
        subset_corpus = paths.qac_dir / "corpus.csv"
        if subset_corpus.exists():
            return subset_corpus
    return paths.corpus_csv


def run_pipeline(config: PipelineConfig, paths: PipelinePaths) -> None:
    qa_sample = config.qa_sample
    qa_pairs_per_language = config.qa_pairs_per_language
    qa_docs_per_language = config.qa_docs_per_language
    qa_batch = config.qa_batch
    prepared_label = "XMLs" if config.source == "epo" else "Prepared source"
    prepared_path = paths.xml_dir if config.source == "epo" else paths.prepared_dir

    if not config.yes:
        if config.source == "jrc-acquis":
            if qa_pairs_per_language is None:
                qa_pairs_per_language = ask_int(
                    "How many multilingual CELEX-group source documents should be sampled per source language for JRC QA prep? Enter 0 to skip: "
                )
            if qa_pairs_per_language > 0 and qa_docs_per_language is None:
                qa_docs_per_language = ask_int(
                    "How many sampled source documents per language should be retained for JRC question generation? Enter 0 to skip: "
                )
            qa_sample = qa_docs_per_language or 0
        else:
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
        if config.source == "jrc-acquis":
            if qa_pairs_per_language is None:
                qa_pairs_per_language = 2000
            if qa_docs_per_language is None:
                qa_docs_per_language = 200
            qa_sample = qa_docs_per_language
        elif qa_sample is None:
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
                if config.source == "jrc-acquis":
                    selection_stats = prepare_jrc_qa_inputs(
                        corpus_full_path=paths.preprocessed_dir / "corpus_multilingual_full.csv",
                        qa_candidates_path=paths.preprocessed_dir / "corpus_qa_candidates.csv",
                        output_dir=paths.qac_dir,
                        pairs_per_language=qa_pairs_per_language or 0,
                        generation_docs_per_language=qa_docs_per_language or 0,
                    )
                    selected_sources_path = paths.qac_dir / "qa_generation_sources.csv"
                    generation_units_total = int(selection_stats.get("generation_units_total", 0))
                    selected_source_docs_total = int(selection_stats.get("selected_source_docs_total", 0))
                    if generation_units_total <= 0:
                        raise ValueError("JRC QA preparation selected zero generation units.")
                    print(
                        "Prepared JRC QA subset:"
                        f" {selection_stats['sampled_source_docs_total']} sampled source docs,"
                        f" {selection_stats['subset_corpus_docs_total']} corpus docs,"
                        f" {selected_source_docs_total} selected source docs,"
                        f" {generation_units_total} generation units."
                    )
                    run_qa_pipeline(
                        corpus_path=selected_sources_path,
                        output_dir=paths.qac_dir,
                        sample_size=generation_units_total,
                        batch_mode=bool(qa_batch),
                        target_languages=[],
                        same_language=True,
                        domain_hint="legal",
                    )
                else:
                    run_qa_pipeline(
                        corpus_path=paths.corpus_full_csv,
                        output_dir=paths.qac_dir,
                        sample_size=qa_sample,
                        batch_mode=bool(qa_batch),
                        same_language=False,
                        domain_hint="encyclopedia" if config.source == "wikidata" else "patent",
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
                corpus_path=_qa_corpus_for_push(config, paths),
                qac_path=qac_csv,
                repo_id=hf_repo,
                source_name=config.source,
            )

    print("\nDone.")
    print(f"  {prepared_label}:", prepared_path)
    print("  Corpus (MTEB):", paths.corpus_csv)
    print("  Corpus (full):", paths.corpus_full_csv)
    if qa_sample > 0:
        print("  QAC:", paths.qac_dir / "qac.csv")
        if config.source == "jrc-acquis":
            print("  QA corpus:", paths.qac_dir / "corpus.csv")
            print("  Sampled pairs:", paths.qac_dir / "sampled_pairs.csv")
            print("  QA sources:", paths.qac_dir / "qa_generation_sources.csv")
    if should_push and hf_repo:
        print("  Hugging Face: https://huggingface.co/datasets/" + hf_repo)
