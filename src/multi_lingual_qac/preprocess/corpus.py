from __future__ import annotations

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.constants import DEFAULT_WIKIDATA_ENTITY_TARGET
from src.multi_lingual_qac.dataloaders.epo import (
    build_epo_corpus,
    clean_text,
    count_epo_xml_files,
    extract_epo_xml_files,
)
from src.multi_lingual_qac.dataloaders.wikidata import (
    build_wikidata_corpus,
    count_wikidata_prepared_records,
    prepare_wikidata_source,
)


def prepare_corpus_source(
    config: PipelineConfig,
    paths: PipelinePaths,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Run the source-specific corpus preparation stage."""
    if config.source == "epo":
        return extract_epo_xml_files(
            input_dir=paths.input_dir,
            output_dir=paths.xml_dir,
            overwrite=overwrite,
        )
    if config.source == "wikidata":
        return prepare_wikidata_source(
            prepared_dir=paths.prepared_dir,
            raw_pages_dir=paths.raw_pages_dir,
            languages=config.languages,
            target_entities=config.limit or DEFAULT_WIKIDATA_ENTITY_TARGET,
            overwrite=overwrite,
        )
    raise ValueError(f"Unsupported corpus source: {config.source}")


def count_source_records(config: PipelineConfig, paths: PipelinePaths) -> int:
    """Count prepared source artifacts for the configured source."""
    if config.source == "epo":
        return count_epo_xml_files(paths.xml_dir)
    if config.source == "wikidata":
        return count_wikidata_prepared_records(paths.prepared_dir)
    raise ValueError(f"Unsupported corpus source: {config.source}")


def build_corpus_from_source(
    config: PipelineConfig,
    paths: PipelinePaths,
) -> dict[str, int]:
    """Build a source-specific corpus from prepared source artifacts."""
    if config.source == "epo":
        return build_epo_corpus(
            xml_dir=paths.xml_dir,
            preprocessed_dir=paths.preprocessed_dir,
            full_output_path=paths.corpus_full_csv,
            output_path=paths.corpus_csv,
            batch_mode=config.build_corpus_batch,
        )
    if config.source == "wikidata":
        stats = build_wikidata_corpus(
            raw_pages_dir=paths.raw_pages_dir,
            preprocessed_dir=paths.preprocessed_dir,
            full_output_path=paths.corpus_full_csv,
            output_path=paths.corpus_csv,
        )
        return {
            "xml_files": stats["languages"],
            "documents_parsed": stats["pages_read"],
            "documents_kept": stats["chunks_written"],
            "all_rows": stats["pages_read"],
            "corpus_rows": stats["chunks_written"],
            "parse_errors": stats["pages_skipped_empty"],
            "skipped_auxiliary": 0,
            "workers": 1,
        }
    raise ValueError(f"Unsupported corpus source: {config.source}")


__all__ = [
    "clean_text",
    "prepare_corpus_source",
    "count_source_records",
    "build_corpus_from_source",
]
