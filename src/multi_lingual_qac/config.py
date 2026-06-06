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
    epo_data_dir: Path
    epo_manifest_path: Path
    epo_corpus_path: Path
    multilingual_corpus_csv: Path
    chebi_dir: Path
    alias_graph_dir: Path
    wiki_quality_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PipelinePaths":
        data_dir = project_root / "data" / "google_patents"
        epo_dir = project_root / "data" / "EPO"
        return cls(
            project_root=project_root,
            raw_ndjson=data_dir / "chemistry_patents.ndjson",
            preprocessed_dir=data_dir / "preprocessed",
            corpus_csv=data_dir / "corpus.csv",
            qac_dir=data_dir / "qac",
            epo_data_dir=epo_dir,
            epo_manifest_path=epo_dir / "manifest.json",
            epo_corpus_path=epo_dir / "multilingual_corpus.csv",
            multilingual_corpus_csv=data_dir / "multilingual_corpus.csv",
            chebi_dir=project_root / "data" / "chebi",
            alias_graph_dir=project_root / "data" / "alias_graph",
            wiki_quality_dir=project_root / "reports" / "wiki_name_quality",
        )


@dataclass(frozen=True)
class PipelineConfig:
    yes: bool = False
    no_extraction: bool = False
    limit: Optional[int] = None
    qa_sample: Optional[int] = None
    qa_batch: Optional[bool] = None
    push_hf: bool = False
    hf_repo: Optional[str] = None
    evaluate_mteb_models: tuple[str, ...] = ()
    mteb_dataset_repo: str = ""
    mteb_dataset_variant: str = "multilingual"
    mteb_output_dir: Optional[str] = None
    mteb_batch_size: int = 32
    mteb_save_predictions: bool = False
    analyze_questions: bool = False
    mteb_analysis_dir: Optional[str] = None
    mteb_no_plots: bool = False
    mteb_query_metadata: Optional[str] = None
    run_id_label: Optional[str] = None
    generate_mteb_tables: bool = False
    mteb_results_dir: Optional[str] = None
    mteb_tables_dir: Optional[str] = None
    upload_mteb_results: bool = False
    mteb_upload_repo: Optional[str] = None
    languages: tuple[str, ...] = tuple(DEFAULT_LANGS)
    epo_ingest: bool = False
    epo_num_batches: int = 1
    epo_chemistry_strict: bool = False
    build_alias_graph: bool = False
    alias_corpus: Optional[str] = None
    alias_output_dir: Optional[str] = None
    chebi_variant: str = "full"
    alias_langs: tuple[str, ...] = ("zh", "en", "de", "fr", "es")
    alias_use_wikipedia: bool = True
    alias_min_gold: int = 2
    alias_min_neg: int = 3
    alias_max_concepts: Optional[int] = None
    alias_max_df: float = 0.02
    alias_molecular_only: bool = True
    alias_leaf_only: bool = True
    check_wiki_names: bool = False
