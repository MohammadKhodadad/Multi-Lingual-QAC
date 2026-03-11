from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.multi_lingual_qac.dataloaders.google_patents import DEFAULT_LANGS


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    raw_ndjson: Path
    preprocessed_dir: Path
    corpus_csv: Path
    qac_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PipelinePaths":
        data_dir = project_root / "data" / "google_patents"
        return cls(
            project_root=project_root,
            raw_ndjson=data_dir / "chemistry_patents.ndjson",
            preprocessed_dir=data_dir / "preprocessed",
            corpus_csv=data_dir / "corpus.csv",
            qac_dir=data_dir / "qac",
        )


@dataclass(frozen=True)
class PipelineConfig:
    yes: bool = False
    no_extraction: bool = False
    limit: Optional[int] = None
    qa_sample: Optional[int] = None
    push_hf: bool = False
    hf_repo: Optional[str] = None
    languages: tuple[str, ...] = tuple(DEFAULT_LANGS)
