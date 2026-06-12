"""One-off: regenerate question_analysis / confusion / mteb_tables for the three runs
with gte-multilingual-base EXCLUDED (model-loading artifact). Offline, reads existing
saved predictions + cached dataset metadata; runs no models. Raw predictions/summary.json
are left untouched — only the derived analysis outputs are rewritten.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.*` imports resolve

from src.multi_lingual_qac.mteb.question_analysis import run_question_analysis
from src.multi_lingual_qac.mteb.evaluation import generate_mteb_comparison_tables
from src.alias_graph.confusion_analysis import run_confusion_from_predictions

GTE = "Alibaba-NLP/gte-multilingual-base"
ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "alias_graph": dict(repo="MehdiAstaraki/multi-lingual-qac-alias-graph", confusion=True),
    "chem_patents": dict(repo="MehdiAstaraki/multi-lingual-qac-chem-patents", confusion=False),
    "epo": dict(repo="MehdiAstaraki/multi-lingual-qac-epo", confusion=False),
}


def keep_models(run_dir: Path) -> list[str]:
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    return [m for m in meta.get("models", []) if m != GTE]


def filtered_summary_dir(run_dir: Path, tmp: Path) -> Path:
    """Write a copy of summary.json with gte dropped, into tmp; return tmp."""
    payload = json.loads((run_dir / "summary.json").read_text())
    payload["models"] = [m for m in payload.get("models", []) if m.get("model_name") != GTE]
    (tmp / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return tmp


def main() -> None:
    selected = [a for a in sys.argv[1:] if not a.startswith("-")]
    for name, cfg in RUNS.items():
        if selected and name not in selected:
            continue
        run_dir = ROOT / "reports" / "runs" / name
        if not run_dir.is_dir():
            print(f"[skip] {name}: no run dir")
            continue
        keep = keep_models(run_dir)
        print(f"\n==== {name}: keeping {len(keep)} models (gte dropped) ====")

        # 1) question_analysis (per-model plots) from saved predictions
        run_question_analysis(
            predictions_dir=run_dir / "predictions",
            output_dir=run_dir / "question_analysis",
            dataset_repo=cfg["repo"],
            dataset_variant="multilingual",
            revision="main",
            model_names=keep,
            make_plots=True,
        )
        print(f"[{name}] question_analysis regenerated")

        # 2) confusion (alias_graph only — needs hard negatives)
        if cfg["confusion"]:
            run_confusion_from_predictions(
                predictions_dir=run_dir / "predictions",
                output_dir=run_dir / "confusion",
                dataset_repo=cfg["repo"],
                dataset_variant="multilingual",
                revision="main",
                model_names=keep,
                make_plots=True,
            )
            print(f"[{name}] confusion regenerated")

        # 3) mteb_tables (leaderboard) from a gte-filtered summary copy
        with tempfile.TemporaryDirectory() as td:
            src = filtered_summary_dir(run_dir, Path(td))
            generate_mteb_comparison_tables(results_dir=src, output_dir=run_dir / "mteb_tables")
        print(f"[{name}] mteb_tables regenerated")


if __name__ == "__main__":
    main()
