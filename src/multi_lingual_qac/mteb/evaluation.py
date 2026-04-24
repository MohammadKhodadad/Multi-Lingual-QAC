from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from mteb import MTEB
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.results import TaskResult
from sentence_transformers import SentenceTransformer

DEFAULT_MTEB_DATASET_REPO = "MohammadKhodadad/multi-lingual-qac"
DEFAULT_MTEB_OUTPUT_DIR = "reports/mteb"
DEFAULT_MTEB_TABLES_DIR = "reports/mteb_tables"
DEFAULT_MTEB_CACHE_DIR = ".cache/huggingface"
DEFAULT_MTEB_MAIN_SCORE = "ndcg_at_10"
DEFAULT_MTEB_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
]

LANGUAGE_TO_MTEB = {
    "ar": "arb-Arab",
    "bg": "bul-Cyrl",
    "cs": "ces-Latn",
    "da": "dan-Latn",
    "de": "deu-Latn",
    "el": "ell-Grek",
    "en": "eng-Latn",
    "es": "spa-Latn",
    "et": "est-Latn",
    "fa": "pes-Arab",
    "fi": "fin-Latn",
    "fr": "fra-Latn",
    "hi": "hin-Deva",
    "hu": "hun-Latn",
    "it": "ita-Latn",
    "ja": "jpn-Jpan",
    "ko": "kor-Hang",
    "lt": "lit-Latn",
    "lv": "lav-Latn",
    "mt": "mlt-Latn",
    "nl": "nld-Latn",
    "pl": "pol-Latn",
    "pt": "por-Latn",
    "ro": "ron-Latn",
    "ru": "rus-Cyrl",
    "sk": "slk-Latn",
    "sl": "slv-Latn",
    "sv": "swe-Latn",
    "tr": "tur-Latn",
    "zh": "zho-Hans",
}


@dataclass(frozen=True)
class ModelEvaluationSummary:
    model_name: str
    model_slug: str
    main_score: float
    metrics: dict[str, float]
    output_dir: str
    eval_languages: list[str]
    evaluation_time_seconds: float | None


class HubDatasetRetrievalTask(AbsTaskRetrieval):
    def __init__(self, metadata: TaskMetadata):
        self.metadata = metadata
        super().__init__()


COMPARISON_METRICS = [
    "main_score",
    "ndcg_at_10",
    "map_at_10",
    "mrr_at_10",
    "hit_rate_at_10",
    "recall_at_10",
    "ndcg_at_100",
    "hit_rate_at_100",
]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "default"


def _dataset_task_name(dataset_repo: str) -> str:
    owner, _, name = dataset_repo.partition("/")
    owner_slug = _slugify(owner or "hf")
    name_slug = _slugify(name or dataset_repo)
    return f"{owner_slug}_{name_slug}_retrieval"


def _default_model_cache_dir() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_MTEB_CACHE_DIR


def _configure_local_model_cache() -> Path:
    cache_dir = _default_model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    sentence_transformers_dir = cache_dir / "sentence_transformers"
    hub_dir = cache_dir / "hub"
    transformers_dir = cache_dir / "transformers"
    sentence_transformers_dir.mkdir(parents=True, exist_ok=True)
    hub_dir.mkdir(parents=True, exist_ok=True)
    transformers_dir.mkdir(parents=True, exist_ok=True)

    # Keep MTEB model downloads inside the repository instead of the user profile cache.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_dir)
    return sentence_transformers_dir


def _detect_query_languages(dataset_repo: str, revision: str) -> list[str]:
    queries = load_dataset(dataset_repo, "queries", split="train", revision=revision)
    lang_column = None
    if "query_language" in queries.column_names:
        lang_column = "query_language"
    elif "question_language" in queries.column_names:
        lang_column = "question_language"
    elif "language" in queries.column_names:
        lang_column = "language"
    if lang_column is None:
        return [LANGUAGE_TO_MTEB["en"]]

    langs = sorted(
        {
            str(value).strip().lower()
            for value in queries[lang_column]
            if str(value).strip()
        }
    )
    mapped = [LANGUAGE_TO_MTEB[lang] for lang in langs if lang in LANGUAGE_TO_MTEB]
    return mapped or [LANGUAGE_TO_MTEB["en"]]


def build_mteb_task(
    dataset_repo: str = DEFAULT_MTEB_DATASET_REPO,
    *,
    revision: str = "main",
) -> HubDatasetRetrievalTask:
    eval_langs = _detect_query_languages(dataset_repo, revision)
    metadata = TaskMetadata(
        name=_dataset_task_name(dataset_repo),
        dataset={"path": dataset_repo, "revision": revision},
        description=f"Custom retrieval evaluation over the Hugging Face dataset `{dataset_repo}`.",
        reference=f"https://huggingface.co/datasets/{dataset_repo}",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"default": eval_langs},
        main_score=DEFAULT_MTEB_MAIN_SCORE,
        domains=["Legal"],
        task_subtypes=["Question Answering Retrieval"],
        license="not specified",
        annotations_creators="LM-generated and reviewed",
        sample_creation="LM-generated and verified",
        is_public=True,
        contributed_by="multi-lingual-qac",
    )
    return HubDatasetRetrievalTask(metadata)


def _extract_numeric_metrics(result: TaskResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_rows in result.scores.values():
        for row in split_rows:
            for key, value in row.items():
                if key in {"hf_subset", "languages"}:
                    continue
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            if metrics:
                return metrics
    return metrics


def _write_summary_reports(
    output_dir: Path,
    dataset_repo: str,
    summaries: list[ModelEvaluationSummary],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"

    payload = {
        "dataset_repo": dataset_repo,
        "models": [
            {
                "model_name": item.model_name,
                "model_slug": item.model_slug,
                "main_score": item.main_score,
                "metrics": item.metrics,
                "output_dir": item.output_dir,
                "eval_languages": item.eval_languages,
                "evaluation_time_seconds": item.evaluation_time_seconds,
            }
            for item in summaries
        ],
    }
    summary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metric_keys = sorted({key for item in summaries for key in item.metrics})
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_name",
                "main_score",
                "evaluation_time_seconds",
                "eval_languages",
                "output_dir",
                *metric_keys,
            ],
        )
        writer.writeheader()
        for item in summaries:
            row: dict[str, Any] = {
                "model_name": item.model_name,
                "main_score": item.main_score,
                "evaluation_time_seconds": item.evaluation_time_seconds,
                "eval_languages": ", ".join(item.eval_languages),
                "output_dir": item.output_dir,
            }
            row.update(item.metrics)
            writer.writerow(row)

    lines = [
        "# MTEB Evaluation Summary",
        "",
        f"- Dataset: `{dataset_repo}`",
        f"- Main score: `{DEFAULT_MTEB_MAIN_SCORE}`",
        "",
        "| Model | Main score | Eval time (s) |",
        "| --- | ---: | ---: |",
    ]
    for item in summaries:
        eval_time = (
            f"{item.evaluation_time_seconds:.1f}"
            if item.evaluation_time_seconds is not None
            else ""
        )
        lines.append(f"| `{item.model_name}` | {item.main_score:.4f} | {eval_time} |")
        if item.metrics:
            metric_parts = [
                f"`{key}`={value:.4f}" for key, value in sorted(item.metrics.items())
            ]
            lines.append(f"| `{item.model_name}` metrics | {'; '.join(metric_parts)} | |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_summary_models(results_dir: Path) -> tuple[str, list[ModelEvaluationSummary]]:
    summary_json = results_dir / "summary.json"
    if not summary_json.exists():
        return _load_raw_result_models(results_dir)

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    dataset_repo = str(payload.get("dataset_repo", DEFAULT_MTEB_DATASET_REPO))
    models_payload = payload.get("models", [])
    summaries: list[ModelEvaluationSummary] = []
    for item in models_payload:
        summaries.append(
            ModelEvaluationSummary(
                model_name=str(item.get("model_name", "")).strip(),
                model_slug=str(item.get("model_slug", "")).strip() or _slugify(str(item.get("model_name", ""))),
                main_score=float(item.get("main_score", 0.0)),
                metrics={
                    str(key): float(value)
                    for key, value in dict(item.get("metrics", {})).items()
                    if isinstance(value, (int, float))
                },
                output_dir=str(item.get("output_dir", "")).strip(),
                eval_languages=[str(x) for x in item.get("eval_languages", [])],
                evaluation_time_seconds=(
                    float(item["evaluation_time_seconds"])
                    if item.get("evaluation_time_seconds") is not None
                    else None
                ),
            )
        )
    return dataset_repo, summaries


def _load_raw_result_models(results_dir: Path) -> tuple[str, list[ModelEvaluationSummary]]:
    result_files = sorted(
        path
        for path in results_dir.rglob("*.json")
        if path.name not in {"summary.json", "model_meta.json", "model_comparison.json"}
    )
    summaries: list[ModelEvaluationSummary] = []
    dataset_repo = DEFAULT_MTEB_DATASET_REPO

    for result_file in result_files:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        if "scores" not in payload:
            continue

        train_rows = payload.get("scores", {}).get("train", [])
        if not train_rows:
            continue
        first_row = train_rows[0]
        metrics = {
            str(key): float(value)
            for key, value in first_row.items()
            if key not in {"hf_subset", "languages"} and isinstance(value, (int, float))
        }
        main_score = float(
            first_row.get("main_score", metrics.get(DEFAULT_MTEB_MAIN_SCORE, 0.0))
        )
        eval_languages = [str(x) for x in first_row.get("languages", [])]

        model_meta_path = result_file.with_name("model_meta.json")
        model_name = result_file.parent.parent.name.replace("__", "/", 1)
        model_slug = _slugify(model_name)
        if model_meta_path.exists():
            model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
            model_name = str(model_meta.get("name", model_name))
            model_slug = _slugify(model_name)

        summaries.append(
            ModelEvaluationSummary(
                model_name=model_name,
                model_slug=model_slug,
                main_score=main_score,
                metrics=metrics,
                output_dir=str(result_file.parent),
                eval_languages=eval_languages,
                evaluation_time_seconds=(
                    float(payload["evaluation_time"])
                    if payload.get("evaluation_time") is not None
                    else None
                ),
            )
        )

    if not summaries:
        raise ValueError(
            f"Could not find `summary.json` or any raw MTEB result json files under `{results_dir}`."
        )
    return dataset_repo, summaries


def _metric_value(item: ModelEvaluationSummary, metric: str) -> float | None:
    if metric == "main_score":
        return item.main_score
    return item.metrics.get(metric)


def _best_metric_values(
    summaries: list[ModelEvaluationSummary],
    metrics: list[str],
) -> dict[str, float]:
    best: dict[str, float] = {}
    for metric in metrics:
        values = [_metric_value(item, metric) for item in summaries]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            best[metric] = max(numeric_values)
    return best


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _build_markdown_comparison(
    dataset_repo: str,
    ranked: list[ModelEvaluationSummary],
) -> str:
    best = _best_metric_values(ranked, COMPARISON_METRICS)
    top = ranked[0]
    lines = [
        "# MTEB Model Comparison",
        "",
        "## Leaderboard",
        "",
        "### Overview",
        "",
        f"- Dataset: `{dataset_repo}`",
        f"- Models compared: `{len(ranked)}`",
        f"- Best model by `{DEFAULT_MTEB_MAIN_SCORE}`: `{top.model_name}` ({top.main_score:.4f})",
        "",
        "### Ranking",
        "",
        "| Rank | Model | Main score | nDCG@10 | MAP@10 | MRR@10 | Hit@10 | Recall@10 | Time (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, item in enumerate(ranked, start=1):
        cells = [
            str(idx),
            f"`{item.model_name}`",
            f"**{item.main_score:.4f}**"
            if item.main_score == best.get("main_score")
            else f"{item.main_score:.4f}",
            f"**{_format_metric(item.metrics.get('ndcg_at_10', item.main_score))}**"
            if _metric_value(item, "ndcg_at_10") == best.get("ndcg_at_10")
            else _format_metric(item.metrics.get("ndcg_at_10", item.main_score)),
            f"**{_format_metric(item.metrics.get('map_at_10'))}**"
            if _metric_value(item, "map_at_10") == best.get("map_at_10")
            else _format_metric(item.metrics.get("map_at_10")),
            f"**{_format_metric(item.metrics.get('mrr_at_10'))}**"
            if _metric_value(item, "mrr_at_10") == best.get("mrr_at_10")
            else _format_metric(item.metrics.get("mrr_at_10")),
            f"**{_format_metric(item.metrics.get('hit_rate_at_10'))}**"
            if _metric_value(item, "hit_rate_at_10") == best.get("hit_rate_at_10")
            else _format_metric(item.metrics.get("hit_rate_at_10")),
            f"**{_format_metric(item.metrics.get('recall_at_10'))}**"
            if _metric_value(item, "recall_at_10") == best.get("recall_at_10")
            else _format_metric(item.metrics.get("recall_at_10")),
            (
                f"{item.evaluation_time_seconds:.1f}"
                if item.evaluation_time_seconds is not None
                else ""
            ),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### Metric Winners",
            "",
            "| Metric | Best model | Score |",
            "| --- | --- | ---: |",
        ]
    )
    for metric in COMPARISON_METRICS:
        winner = max(
            ranked,
            key=lambda item: _metric_value(item, metric)
            if _metric_value(item, metric) is not None
            else float("-inf"),
        )
        winner_value = _metric_value(winner, metric)
        if winner_value is None:
            continue
        lines.append(
            f"| `{metric}` | `{winner.model_name}` | {winner_value:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _build_latex_comparison(
    dataset_repo: str,
    ranked: list[ModelEvaluationSummary],
) -> str:
    best = _best_metric_values(ranked, COMPARISON_METRICS)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{r l r r r r r r}",
        r"\hline",
        r"Rank & Model & Main & MAP@10 & MRR@10 & Hit@10 & Recall@10 & Time (s) \\",
        r"\hline",
    ]
    for idx, item in enumerate(ranked, start=1):
        main_score = _format_metric(item.main_score)
        map_at_10 = _format_metric(item.metrics.get("map_at_10"))
        mrr_at_10 = _format_metric(item.metrics.get("mrr_at_10"))
        hit_rate_at_10 = _format_metric(item.metrics.get("hit_rate_at_10"))
        recall_at_10 = _format_metric(item.metrics.get("recall_at_10"))
        eval_time = (
            f"{item.evaluation_time_seconds:.1f}"
            if item.evaluation_time_seconds is not None
            else "--"
        )
        if item.main_score == best.get("main_score"):
            main_score = rf"\textbf{{{main_score}}}"
        if _metric_value(item, "map_at_10") == best.get("map_at_10"):
            map_at_10 = rf"\textbf{{{map_at_10}}}"
        if _metric_value(item, "mrr_at_10") == best.get("mrr_at_10"):
            mrr_at_10 = rf"\textbf{{{mrr_at_10}}}"
        if _metric_value(item, "hit_rate_at_10") == best.get("hit_rate_at_10"):
            hit_rate_at_10 = rf"\textbf{{{hit_rate_at_10}}}"
        if _metric_value(item, "recall_at_10") == best.get("recall_at_10"):
            recall_at_10 = rf"\textbf{{{recall_at_10}}}"
        lines.append(
            " & ".join(
                [
                    str(idx),
                    r"\texttt{" + _latex_escape(item.model_name) + "}",
                    main_score,
                    map_at_10,
                    mrr_at_10,
                    hit_rate_at_10,
                    recall_at_10,
                    eval_time,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{MTEB retrieval comparison on \texttt{{{_latex_escape(dataset_repo)}}}. Bold marks the best score per metric.}}",
            r"\label{tab:mteb-model-comparison}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_mteb_comparison_tables(
    *,
    results_dir: str | Path = DEFAULT_MTEB_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_MTEB_TABLES_DIR,
) -> Path:
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    dataset_repo, summaries = _load_summary_models(results_path)
    if not summaries:
        raise ValueError(f"No model summaries found in `{results_path}`.")

    ranked = sorted(summaries, key=lambda item: item.main_score, reverse=True)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_json = output_path / "model_comparison.json"
    comparison_csv = output_path / "model_comparison.csv"
    comparison_md = output_path / "model_comparison.md"
    comparison_tex = output_path / "model_comparison.tex"

    payload = {
        "dataset_repo": dataset_repo,
        "results_dir": str(results_path),
        "metrics": COMPARISON_METRICS,
        "models": [
            {
                "rank": idx,
                "model_name": item.model_name,
                "main_score": item.main_score,
                "evaluation_time_seconds": item.evaluation_time_seconds,
                "output_dir": item.output_dir,
                "metrics": {metric: item.metrics.get(metric) for metric in COMPARISON_METRICS},
            }
            for idx, item in enumerate(ranked, start=1)
        ],
    }
    comparison_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with comparison_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "model_name",
                "evaluation_time_seconds",
                "output_dir",
                *COMPARISON_METRICS,
            ],
        )
        writer.writeheader()
        for idx, item in enumerate(ranked, start=1):
            row: dict[str, Any] = {
                "rank": idx,
                "model_name": item.model_name,
                "main_score": item.main_score,
                "evaluation_time_seconds": item.evaluation_time_seconds,
                "output_dir": item.output_dir,
            }
            for metric in COMPARISON_METRICS:
                row[metric] = item.metrics.get(metric, item.main_score if metric == "main_score" else "")
            writer.writerow(row)

    comparison_md.write_text(
        _build_markdown_comparison(dataset_repo, ranked),
        encoding="utf-8",
    )
    comparison_tex.write_text(
        _build_latex_comparison(dataset_repo, ranked),
        encoding="utf-8",
    )
    return output_path


def run_mteb_evaluation(
    models: list[str],
    *,
    dataset_repo: str = DEFAULT_MTEB_DATASET_REPO,
    output_dir: str | Path = DEFAULT_MTEB_OUTPUT_DIR,
    revision: str = "main",
    batch_size: int = 32,
) -> list[ModelEvaluationSummary]:
    if not models:
        raise ValueError("Provide at least one model name for MTEB evaluation.")

    task = build_mteb_task(dataset_repo, revision=revision)
    evaluator = MTEB(tasks=[task])
    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = _configure_local_model_cache()
    summaries: list[ModelEvaluationSummary] = []
    eval_languages = _detect_query_languages(dataset_repo, revision)

    for model_name in models:
        model_slug = _slugify(model_name)
        print(f"Evaluating `{model_name}` on `{dataset_repo}`...")
        model = SentenceTransformer(model_name, cache_folder=str(model_cache_dir))
        model_meta = evaluator.create_model_meta(model)
        model_output_dir = base_output_dir / model_meta.model_name_as_path() / (
            model_meta.revision or "no_revision_available"
        )
        results = evaluator.run(
            model,
            verbosity=2,
            output_folder=str(base_output_dir),
            eval_splits=["train"],
            overwrite_results=True,
            encode_kwargs={"batch_size": batch_size},
        )
        if not results:
            raise ValueError(f"MTEB returned no results for model `{model_name}`.")

        result = results[0]
        metrics = _extract_numeric_metrics(result)
        summary = ModelEvaluationSummary(
            model_name=model_name,
            model_slug=model_slug,
            main_score=float(result.main_score),
            metrics=metrics,
            output_dir=str(model_output_dir),
            eval_languages=eval_languages,
            evaluation_time_seconds=result.evaluation_time,
        )
        summaries.append(summary)

    _write_summary_reports(base_output_dir, dataset_repo, summaries)
    return summaries
