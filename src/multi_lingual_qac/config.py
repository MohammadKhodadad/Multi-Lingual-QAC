from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.multi_lingual_qac.constants import DEFAULT_LANGS


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    source: str
    data_dir: Path
    input_dir: Path
    prepared_dir: Path
    raw_pages_dir: Path
    xml_dir: Path
    preprocessed_dir: Path
    corpus_full_csv: Path
    corpus_csv: Path
    qac_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path, source: str = "epo") -> "PipelinePaths":
        normalized_source = source.strip().lower()
        data_dir = project_root / "data" / normalized_source.upper()
        prepared_dir = data_dir / ("xmls" if normalized_source == "epo" else "prepared")
        return cls(
            project_root=project_root,
            source=normalized_source,
            data_dir=data_dir,
            input_dir=data_dir / "input",
            prepared_dir=prepared_dir,
            raw_pages_dir=prepared_dir / "pages",
            xml_dir=prepared_dir,
            preprocessed_dir=data_dir / "preprocessed",
            corpus_full_csv=data_dir / "preprocessed" / "corpus_full.csv",
            corpus_csv=data_dir / "corpus.csv",
            qac_dir=data_dir / "qac",
        )


@dataclass(frozen=True)
class PipelineConfig:
    source: str = "epo"
    prepare_source: Optional[str] = None
    prepare_workers: int = 1
    build_corpus: Optional[str] = None
    build_corpus_batch: bool = False
    build_workers: int = 1
    label_qrels: Optional[str] = None
    label_qrels_batch_size: int = 5
    yes: bool = False
    no_extraction: bool = False
    limit: Optional[int] = None
    qa_sample: Optional[int] = None
    qa_batch: Optional[bool] = None
    push_hf: bool = False
    hf_repo: Optional[str] = None
    languages: tuple[str, ...] = tuple(DEFAULT_LANGS)
