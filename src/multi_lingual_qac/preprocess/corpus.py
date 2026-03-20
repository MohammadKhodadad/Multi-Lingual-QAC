from __future__ import annotations

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.dataloaders.epo import (
    build_epo_corpus,
    clean_text,
    count_epo_xml_files,
    extract_epo_xml_files,
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
    raise ValueError(f"Unsupported corpus source: {config.source}")


def count_source_records(config: PipelineConfig, paths: PipelinePaths) -> int:
    """Count prepared source artifacts for the configured patent source."""
    if config.source == "epo":
        return count_epo_xml_files(paths.xml_dir)
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
    raise ValueError(f"Unsupported corpus source: {config.source}")


__all__ = [
    "clean_text",
    "prepare_corpus_source",
    "count_source_records",
    "build_corpus_from_source",
]
