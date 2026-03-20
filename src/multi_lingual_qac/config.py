from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.multi_lingual_qac.dataloaders.epo import DEFAULT_LANGS


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    input_dir: Path
    xml_dir: Path
    preprocessed_dir: Path
    corpus_full_csv: Path
    corpus_csv: Path
    qac_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PipelinePaths":
        data_dir = project_root / "data" / "EPO"
        return cls(
            project_root=project_root,
            input_dir=data_dir / "input",
            xml_dir=data_dir / "xmls",
            preprocessed_dir=data_dir / "preprocessed",
            corpus_full_csv=data_dir / "preprocessed" / "corpus_full.csv",
            corpus_csv=data_dir / "corpus.csv",
            qac_dir=data_dir / "qac",
        )


@dataclass(frozen=True)
class PipelineConfig:
    source: str = "epo"
    prepare_source: Optional[str] = None
    build_corpus: Optional[str] = None
    build_corpus_batch: bool = False
    yes: bool = False
    no_extraction: bool = False
    limit: Optional[int] = None
    qa_sample: Optional[int] = None
    qa_batch: Optional[bool] = None
    push_hf: bool = False
    hf_repo: Optional[str] = None
    languages: tuple[str, ...] = tuple(DEFAULT_LANGS)
