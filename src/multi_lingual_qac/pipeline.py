from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.export.hf_upload import push_to_hub
from src.multi_lingual_qac.preprocess.corpus import build_corpus_from_source, count_source_records
from src.multi_lingual_qac.qac_generation.jrc_acquis import prepare_jrc_qa_inputs
from src.multi_lingual_qac.qac_generation.openai_qa import run_qa_pipeline
from src.multi_lingual_qac.reporting import update_jrc_pipeline_report

DEFAULT_JRC_QA_LANGUAGES = ("en", "es", "de", "fr", "pt")


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
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _existing_qac_source_ids(qac_path: Path) -> set[str]:
    if not qac_path.is_file():
        return set()
    source_ids: set[str] = set()
    for row in _read_csv_rows(qac_path):
        source_id = (
            row.get("source_corpus_id")
            or row.get("target_corpus_id")
            or row.get("corpus_id")
            or ""
        ).strip()
        if source_id:
            source_ids.add(source_id)
    return source_ids


def _append_qac_rows(existing_qac_path: Path, new_qac_path: Path) -> int:
    existing_rows = _read_csv_rows(existing_qac_path) if existing_qac_path.is_file() else []
    new_rows = _read_csv_rows(new_qac_path) if new_qac_path.is_file() else []
    seen = {
        (
            row.get("corpus_id", ""),
            row.get("language", ""),
            row.get("question", ""),
            row.get("answer", ""),
            row.get("is_synthetic_translation", ""),
        )
        for row in existing_rows
    }
    appended_rows: list[dict[str, str]] = []
    for row in new_rows:
        key = (
            row.get("corpus_id", ""),
            row.get("language", ""),
            row.get("question", ""),
            row.get("answer", ""),
            row.get("is_synthetic_translation", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        appended_rows.append(row)
    _write_csv_rows(existing_qac_path, existing_rows + appended_rows)
    return len(appended_rows)


def _build_corpus(config: PipelineConfig, paths: PipelinePaths) -> dict[str, int]:
    stats = build_corpus_from_source(config, paths)
    if config.source == "jrc-acquis":
        print(
            "Rebuilt JRC-ACQUIS source document pool:"
            f" {stats.get('documents_written', 0)} multilingual documents"
            f" from {stats.get('celex_total', 0)} CELEX ids."
        )
        report_path = update_jrc_pipeline_report(paths, stage="corpus build")
        print("  Report:", report_path)
    else:
        print(f"Rebuilt {config.source.upper()} corpus: {stats.get('corpus_rows', 0)} rows.")
    return stats


def migrate_legacy_jrc_source_pool_paths(paths: PipelinePaths) -> None:
    if paths.source != "jrc-acquis":
        return
    legacy_paths = [
        (paths.data_dir / "corpus.csv", paths.corpus_csv),
        (paths.preprocessed_dir / "corpus_full.csv", paths.corpus_full_csv),
    ]
    for legacy_path, new_path in legacy_paths:
        if legacy_path == new_path or not legacy_path.is_file() or new_path.exists():
            continue
        new_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.replace(new_path)
        print(f"Renamed legacy JRC source document pool artifact: {legacy_path} -> {new_path}")


def _qa_corpus_for_push(config: PipelineConfig, paths: PipelinePaths) -> Path:
    if config.source == "jrc-acquis":
        subset_corpus = paths.qac_dir / "corpus.csv"
        if subset_corpus.exists():
            return subset_corpus
    return paths.corpus_csv


def run_pipeline(config: PipelineConfig, paths: PipelinePaths) -> None:
    migrate_legacy_jrc_source_pool_paths(paths)

    qa_sample = config.qa_sample
    qa_pairs_per_language = config.qa_pairs_per_language
    qa_docs_per_language = config.qa_docs_per_language
    jrc_qa_languages = config.jrc_qa_languages
    jrc_synthetic_chinese = config.jrc_synthetic_chinese
    qa_batch = config.qa_batch
    prepared_label = "XMLs" if config.source == "epo" else "Prepared source"
    prepared_path = paths.xml_dir if config.source == "epo" else paths.prepared_dir

    xml_count = count_source_records(config, paths)
    if not xml_count:
        print(f"Error: No prepared source files found at {prepared_path}.")
        print(f"Run `uv run main.py --prepare-source {config.source.upper()}` first.")
        raise SystemExit(1)

    corpus_exists = paths.corpus_csv.exists() and paths.corpus_full_csv.exists()
    if not corpus_exists:
        print("\nPrepared source artifacts are available.")
        print(f"  {prepared_label}:", prepared_path)
        corpus_label = "source document pool" if config.source == "jrc-acquis" else "corpus"
        print(f"Run `uv run main.py --build-corpus {config.source.upper()}` to create the {corpus_label}.")
        return

    corpus_rebuilt = False
    jrc_hf_corpus_rebuild = False
    jrc_hf_corpus_path = paths.qac_dir / "corpus.csv"
    if config.source == "jrc-acquis":
        if config.force_rebuild_corpus:
            jrc_hf_corpus_rebuild = True
        elif jrc_hf_corpus_path.is_file() and not config.yes:
            rebuild_corpus = ask_interactive(
                f"Benchmark/HF corpus is already built at {jrc_hf_corpus_path} ({_count_rows(jrc_hf_corpus_path)} rows). "
                "Do you want to rebuild it? (y/N): ",
                "n",
            )
            if rebuild_corpus == "y":
                jrc_hf_corpus_rebuild = True
            else:
                print("Reusing existing benchmark/HF corpus.")
        elif not jrc_hf_corpus_path.is_file():
            print("No JRC benchmark/HF corpus exists yet; it will be created during QA source selection.")
    elif config.force_rebuild_corpus:
        _build_corpus(config, paths)
        corpus_rebuilt = True
    elif not config.yes:
        rebuild_corpus = ask_interactive(
            f"Corpus is already built at {paths.corpus_csv} ({_count_rows(paths.corpus_csv)} rows). "
            "Do you want to rebuild it? (y/N): ",
            "n",
        )
        if rebuild_corpus == "y":
            _build_corpus(config, paths)
            corpus_rebuilt = True
        else:
            print("Reusing existing corpus.")

    if not config.yes:
        if config.source == "jrc-acquis":
            if jrc_qa_languages is None:
                use_default_subset = (
                    ask_interactive(
                        "Do you want JRC QA generation to be only from these 5 languages: en, es, de, fr, pt? (y/n): ",
                        "n",
                    )
                    == "y"
                )
                if use_default_subset:
                    jrc_qa_languages = DEFAULT_JRC_QA_LANGUAGES
            if jrc_synthetic_chinese is None:
                jrc_synthetic_chinese = (
                    ask_interactive(
                        "Do you want synthetic translations (Chinese)? (y/n): ",
                        "n",
                    )
                    == "y"
                )
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

    if qa_sample > 0:
        qac_csv = paths.qac_dir / "qac.csv"
        run_qa = True
        append_qac = False
        if qac_csv.exists() and (corpus_rebuilt or jrc_hf_corpus_rebuild):
            print("Benchmark corpus was rebuilt; regenerating QAC instead of appending to the old queries.")
        elif qac_csv.exists() and not config.yes:
            append = ask_interactive(
                f"QAC already exists ({_count_rows(qac_csv)} rows). Append new queries to it? (Y/n): ",
                "y",
            )
            if append == "y":
                append_qac = True
            else:
                redo = ask_interactive(
                    "Regenerate Q&A and overwrite the existing QAC instead? (y/N): ",
                    "n",
                )
                run_qa = redo == "y"
        if run_qa:
            try:
                if config.source == "jrc-acquis":
                    qac_output_dir = paths.qac_dir
                    selection_output_dir = paths.qac_dir
                    linked_corpus_path = paths.qac_dir / "corpus_full.csv"
                    existing_generation_sources_path = paths.qac_dir / "qa_generation_sources.csv"
                    source_corpus_full_path = paths.preprocessed_dir / "corpus_multilingual_full.csv"
                    exclude_source_ids: set[str] | None = None
                    temp_dir_context = None
                    reuse_fixed_hf_corpus = (
                        jrc_hf_corpus_path.is_file()
                        and linked_corpus_path.is_file()
                        and existing_generation_sources_path.is_file()
                        and not jrc_hf_corpus_rebuild
                    )
                    if append_qac:
                        temp_dir_context = tempfile.TemporaryDirectory(
                            prefix="qac_append_",
                            dir=paths.qac_dir,
                        )
                        selection_output_dir = Path(temp_dir_context.name)
                        qac_output_dir = selection_output_dir
                        exclude_source_ids = _existing_qac_source_ids(qac_csv)
                        if linked_corpus_path.is_file():
                            source_corpus_full_path = linked_corpus_path
                    elif reuse_fixed_hf_corpus:
                        source_corpus_full_path = linked_corpus_path

                    if reuse_fixed_hf_corpus and not append_qac:
                        selected_sources_path = existing_generation_sources_path
                        generation_units_total = _count_rows(selected_sources_path)
                        sampled_source_pool_total = _count_rows(jrc_hf_corpus_path)
                        selected_generation_docs_total = generation_units_total
                        final_retrieval_corpus_total = sampled_source_pool_total
                        print(
                            "Using fixed JRC benchmark/HF corpus:"
                            f" {sampled_source_pool_total} retrieval-corpus docs,"
                            f" {generation_units_total} generation units."
                        )
                    else:
                        selection_stats = prepare_jrc_qa_inputs(
                            corpus_full_path=source_corpus_full_path,
                            qa_candidates_path=paths.preprocessed_dir / "corpus_qa_candidates.csv",
                            output_dir=selection_output_dir,
                            pairs_per_language=qa_pairs_per_language or 0,
                            generation_docs_per_language=qa_docs_per_language or 0,
                            allowed_languages=jrc_qa_languages,
                            synthetic_target_languages=("zh",) if jrc_synthetic_chinese else (),
                            exclude_source_ids=exclude_source_ids,
                        )
                        selected_sources_path = selection_output_dir / "qa_generation_sources.csv"
                        generation_units_total = int(selection_stats.get("generation_units_total", 0))
                        sampled_source_pool_total = int(selection_stats.get("sampled_source_pool_docs_total", 0))
                        selected_generation_docs_total = int(selection_stats.get("selected_generation_source_docs_total", 0))
                        final_retrieval_corpus_total = int(selection_stats.get("final_retrieval_corpus_docs_total", 0))
                    if generation_units_total <= 0:
                        raise ValueError("JRC QA preparation selected zero generation units.")
                    if not (reuse_fixed_hf_corpus and not append_qac):
                        print(
                            "Prepared JRC QA benchmark inputs:"
                            f" {sampled_source_pool_total} sampled source-pool docs,"
                            f" {selected_generation_docs_total} generation source candidates,"
                            f" {final_retrieval_corpus_total} final retrieval-corpus docs,"
                            f" {generation_units_total} generation units."
                        )
                    if not append_qac and not reuse_fixed_hf_corpus:
                        report_path = update_jrc_pipeline_report(paths, stage="qa source selection")
                        print("  Report:", report_path)
                    run_qa_pipeline(
                        corpus_path=selected_sources_path,
                        output_dir=qac_output_dir,
                        sample_size=generation_units_total,
                        batch_mode=bool(qa_batch),
                        target_languages=[],
                        same_language=True,
                        domain_hint="legal",
                        synthetic_translation_targets=["zh"] if jrc_synthetic_chinese else [],
                        linked_corpus_path=source_corpus_full_path if append_qac else linked_corpus_path,
                        require_cross_language_support=True,
                        accepted_per_language=qa_docs_per_language,
                    )
                    if append_qac:
                        appended_count = _append_qac_rows(qac_csv, qac_output_dir / "qac.csv")
                        print(f"Appended {appended_count} new QAC rows -> {qac_csv}")
                        if temp_dir_context is not None:
                            temp_dir_context.cleanup()
                    report_path = update_jrc_pipeline_report(paths, stage="qa generation")
                    print("  Report:", report_path)
                else:
                    qac_output_dir = paths.qac_dir
                    temp_dir_context = None
                    if append_qac:
                        temp_dir_context = tempfile.TemporaryDirectory(
                            prefix="qac_append_",
                            dir=paths.qac_dir,
                        )
                        qac_output_dir = Path(temp_dir_context.name)
                    run_qa_pipeline(
                        corpus_path=paths.corpus_full_csv,
                        output_dir=qac_output_dir,
                        sample_size=qa_sample,
                        batch_mode=bool(qa_batch),
                        same_language=False,
                        domain_hint="encyclopedia" if config.source == "wikidata" else "patent",
                    )
                    if append_qac:
                        appended_count = _append_qac_rows(qac_csv, qac_output_dir / "qac.csv")
                        print(f"Appended {appended_count} new QAC rows -> {qac_csv}")
                        if temp_dir_context is not None:
                            temp_dir_context.cleanup()
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
            print("Error: Source document pool not found. Run pipeline first.")
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
            if config.source == "jrc-acquis":
                report_path = update_jrc_pipeline_report(paths, stage="hugging face upload")
                print("  Report:", report_path)

    print("\nDone.")
    print(f"  {prepared_label}:", prepared_path)
    if config.source == "jrc-acquis":
        print("  Source document pool (MTEB-style):", paths.corpus_csv)
        print("  Source document pool (full):", paths.corpus_full_csv)
    else:
        print("  Corpus (MTEB):", paths.corpus_csv)
        print("  Corpus (full):", paths.corpus_full_csv)
    if qa_sample > 0:
        print("  QAC:", paths.qac_dir / "qac.csv")
        if config.source == "jrc-acquis":
            print("  Benchmark/HF corpus:", paths.qac_dir / "corpus.csv")
            print("  Sampled pairs:", paths.qac_dir / "sampled_pairs.csv")
            print("  QA sources:", paths.qac_dir / "qa_generation_sources.csv")
    if should_push and hf_repo:
        print("  Hugging Face: https://huggingface.co/datasets/" + hf_repo)
