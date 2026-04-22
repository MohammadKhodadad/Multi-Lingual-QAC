from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset

DEFAULT_MTEB_DATASET_REPO = ""
DEFAULT_MTEB_OUTPUT_DIR = "reports/mteb"
DEFAULT_MTEB_TABLES_DIR = "reports/mteb_tables"
DEFAULT_MTEB_CACHE_DIR = ".cache/huggingface"
DEFAULT_MTEB_MAIN_SCORE = "ndcg_at_10"
DEFAULT_MTEB_LOCAL_CORPUS_PATH = "data/google_patents/corpus.csv"
DEFAULT_MTEB_LOCAL_QAC_PATH = "data/google_patents/qac/balanced_100_qac.csv"
PROJECT_REFERENCE_URL = "https://github.com/MohammadKhodadad/Multi-Lingual-QAC"
DEFAULT_MTEB_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
]

_LOCAL_BENCHMARK_CACHE: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}

STRATEGY_ID_TO_NAME = {
    "1": "random_any",
    "2": "random_missing",
    "3": "random_existing",
    "4": "all",
}
STRATEGY_SORT_ORDER = {
    "random_any": 0,
    "random_missing": 1,
    "random_existing": 2,
    "all": 3,
}
MODE_SORT_ORDER = {
    "technical": 0,
    "semantic": 1,
}

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


@dataclass(frozen=True)
class BenchmarkSlice:
    name: str
    label: str
    mode: Optional[str] = None
    strategy_name: Optional[str] = None

    def matches(self, row: dict[str, Any]) -> bool:
        row_mode = str(row.get("mode", "")).strip().lower()
        row_strategy = _row_strategy_name(row)
        if self.mode is not None and row_mode != self.mode:
            return False
        if self.strategy_name is not None and row_strategy != self.strategy_name:
            return False
        return True


@dataclass(frozen=True)
class ModelEvaluationSummary:
    model_name: str
    model_slug: str
    slice_name: str
    slice_label: str
    filter_mode: str
    filter_strategy_name: str
    task_name: str
    main_score: float
    metrics: dict[str, float]
    output_dir: str
    eval_languages: list[str]
    evaluation_time_seconds: float | None


@dataclass(frozen=True)
class BenchmarkSource:
    dataset_repo: str = ""
    local_corpus_path: str = DEFAULT_MTEB_LOCAL_CORPUS_PATH
    local_qac_path: str = DEFAULT_MTEB_LOCAL_QAC_PATH
    revision: str = "main"

    @property
    def uses_hf(self) -> bool:
        return bool(self.dataset_repo.strip())

    @property
    def label(self) -> str:
        if self.uses_hf:
            return self.dataset_repo
        return f"local:{self.local_qac_path}"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "default"


def _dataset_task_name(dataset_repo: str) -> str:
    owner, _, name = dataset_repo.partition("/")
    owner_slug = _slugify(owner or "hf")
    name_slug = _slugify(name or dataset_repo)
    return f"{owner_slug}_{name_slug}_retrieval"


def _row_strategy_name(row: dict[str, Any]) -> str:
    strategy_name = str(row.get("strategy_name", "")).strip().lower()
    if strategy_name:
        return strategy_name
    strategy = str(row.get("strategy", "")).strip()
    return STRATEGY_ID_TO_NAME.get(strategy, strategy.lower())


def _slice_sort_key(slice_filter: BenchmarkSlice) -> tuple[int, int, int, str]:
    if slice_filter.name == "overall":
        return (0, 0, 0, slice_filter.name)
    if slice_filter.mode and slice_filter.strategy_name:
        return (
            3,
            MODE_SORT_ORDER.get(slice_filter.mode, 99),
            STRATEGY_SORT_ORDER.get(slice_filter.strategy_name, 99),
            slice_filter.name,
        )
    if slice_filter.mode:
        return (1, MODE_SORT_ORDER.get(slice_filter.mode, 99), 0, slice_filter.name)
    if slice_filter.strategy_name:
        return (2, 0, STRATEGY_SORT_ORDER.get(slice_filter.strategy_name, 99), slice_filter.name)
    return (9, 99, 99, slice_filter.name)


def _slice_name_from_task_name(task_name: str) -> str:
    if "__" not in task_name:
        return "overall"
    return task_name.rsplit("__", 1)[-1]


def _slice_label_from_name(slice_name: str) -> str:
    if slice_name == "overall":
        return "Overall"
    if slice_name.startswith("mode-"):
        mode = slice_name.removeprefix("mode-").replace("-", " ")
        return f"Mode: {mode}"
    if slice_name.startswith("strategy-"):
        strategy = slice_name.removeprefix("strategy-").replace("-", " ")
        return f"Strategy: {strategy}"
    if slice_name.startswith("mode-strategy-"):
        _, _, rest = slice_name.partition("mode-strategy-")
        mode, _, strategy = rest.partition("__")
        return f"Mode/Strategy: {mode.replace('-', ' ')} / {strategy.replace('-', ' ')}"
    return slice_name.replace("-", " ")


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

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_dir)
    return sentence_transformers_dir


def _model_cache_path(cache_dir: Path, model_name: str) -> Path:
    return cache_dir / f"models--{model_name.replace('/', '--')}"


def _has_cached_model(cache_dir: Path, model_name: str) -> bool:
    snapshots_dir = _model_cache_path(cache_dir, model_name) / "snapshots"
    return snapshots_dir.exists() and any(snapshots_dir.iterdir())


def _load_hf_split(dataset_repo: str, config_name: str, revision: str) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_repo, config_name, split="train", revision=revision)
    return [dict(row) for row in dataset]


def _load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    return rows


def _get_question_language(row: dict[str, Any]) -> str:
    return str(row.get("question_language") or row.get("language") or "").strip()


def _build_local_benchmark_rows(
    corpus_path: str | Path,
    qac_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    cache_key = (str(Path(corpus_path)), str(Path(qac_path)))
    cached = _LOCAL_BENCHMARK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    corpus_rows = _load_csv_rows(corpus_path)
    qac_rows = _load_csv_rows(qac_path)

    corpus_rows_out = [
        {
            "_id": str(row.get("id", "")).strip(),
            "title": str(row.get("title", "")).strip(),
            "text": str(row.get("context", row.get("abstract", ""))).strip(),
            "publication_number": str(row.get("publication_number", "")).strip(),
            "language": str(row.get("language", "")).strip(),
        }
        for row in corpus_rows
        if str(row.get("id", "")).strip()
    ]

    corpus_ids_by_publication: dict[str, list[str]] = defaultdict(list)
    corpus_id_set = set()
    for row in corpus_rows_out:
        corpus_id_set.add(row["_id"])
        if row["publication_number"]:
            corpus_ids_by_publication[row["publication_number"]].append(row["_id"])

    queries_rows: list[dict[str, Any]] = []
    qrels_rows: list[dict[str, Any]] = []
    qac_rows_out: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()

    for i, row in enumerate(qac_rows):
        corpus_id = str(row.get("corpus_id", "")).strip()
        question_language = _get_question_language(row)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        publication_number = str(row.get("publication_number", "")).strip()
        query_id = (
            f"{corpus_id}_q_{question_language}"
            if corpus_id in corpus_id_set
            else f"q_{i}_{question_language}"
        )
        if query_id in seen_query_ids:
            query_id = f"{query_id}_{i}"
        seen_query_ids.add(query_id)

        query_row = {
            "_id": query_id,
            "text": question,
            "language": question_language,
            "question_language": question_language,
            "mode": str(row.get("mode", "")).strip(),
            "strategy": str(row.get("strategy", "")).strip(),
            "strategy_name": str(row.get("strategy_name", "")).strip(),
            "publication_number": publication_number,
            "corpus_id": corpus_id,
        }
        queries_rows.append(query_row)

        relevant_corpus_ids = corpus_ids_by_publication.get(publication_number, [])
        if not relevant_corpus_ids and corpus_id:
            relevant_corpus_ids = [corpus_id]
        for relevant_corpus_id in relevant_corpus_ids:
            qrels_rows.append(
                {"query-id": query_id, "corpus-id": relevant_corpus_id, "score": 1.0}
            )

        qac_rows_out.append(
            {
                "query_id": query_id,
                "corpus_id": corpus_id,
                "language": question_language,
                "question_language": question_language,
                "mode": str(row.get("mode", "")).strip(),
                "strategy": str(row.get("strategy", "")).strip(),
                "strategy_name": str(row.get("strategy_name", "")).strip(),
                "publication_number": publication_number,
                "question": question,
                "answer": answer,
            }
        )

    built = {
        "corpus": corpus_rows_out,
        "queries": queries_rows,
        "qrels": qrels_rows,
        "qac": qac_rows_out,
    }
    _LOCAL_BENCHMARK_CACHE[cache_key] = built
    return built


def _load_benchmark_split(
    source: BenchmarkSource,
    config_name: str,
) -> list[dict[str, Any]]:
    if source.uses_hf:
        return _load_hf_split(source.dataset_repo, config_name, source.revision)
    return _build_local_benchmark_rows(
        source.local_corpus_path,
        source.local_qac_path,
    )[config_name]


def _detect_query_languages(
    source: BenchmarkSource,
    *,
    slice_filter: Optional[BenchmarkSlice] = None,
) -> list[str]:
    query_rows = _load_benchmark_split(source, "queries")
    if slice_filter is not None:
        query_rows = [row for row in query_rows if slice_filter.matches(row)]

    lang_column = None
    if query_rows and "question_language" in query_rows[0]:
        lang_column = "question_language"
    elif query_rows and "language" in query_rows[0]:
        lang_column = "language"
    if lang_column is None:
        return [LANGUAGE_TO_MTEB["en"]]

    langs = sorted(
        {
            str(row.get(lang_column, "")).strip().lower()
            for row in query_rows
            if str(row.get(lang_column, "")).strip()
        }
    )
    mapped = [LANGUAGE_TO_MTEB[lang] for lang in langs if lang in LANGUAGE_TO_MTEB]
    return mapped or [LANGUAGE_TO_MTEB["en"]]


def _detect_benchmark_slices(
    source: BenchmarkSource,
    *,
    include_mode_strategy: bool = False,
) -> list[BenchmarkSlice]:
    query_rows = _load_benchmark_split(source, "queries")
    slices = [BenchmarkSlice(name="overall", label="Overall")]

    modes = sorted(
        {
            str(row.get("mode", "")).strip().lower()
            for row in query_rows
            if str(row.get("mode", "")).strip()
        },
        key=lambda value: (MODE_SORT_ORDER.get(value, 99), value),
    )
    for mode in modes:
        slices.append(
            BenchmarkSlice(
                name=f"mode-{_slugify(mode)}",
                label=f"Mode: {mode}",
                mode=mode,
            )
        )

    strategies = sorted(
        {
            _row_strategy_name(row)
            for row in query_rows
            if _row_strategy_name(row)
        },
        key=lambda value: (STRATEGY_SORT_ORDER.get(value, 99), value),
    )
    for strategy_name in strategies:
        slices.append(
            BenchmarkSlice(
                name=f"strategy-{_slugify(strategy_name)}",
                label=f"Strategy: {strategy_name}",
                strategy_name=strategy_name,
            )
        )

    if include_mode_strategy and modes and strategies:
        for mode in modes:
            for strategy_name in strategies:
                if any(
                    str(row.get("mode", "")).strip().lower() == mode
                    and _row_strategy_name(row) == strategy_name
                    for row in query_rows
                ):
                    slices.append(
                        BenchmarkSlice(
                            name=f"mode-strategy-{_slugify(mode)}__{_slugify(strategy_name)}",
                            label=f"Mode/Strategy: {mode} / {strategy_name}",
                            mode=mode,
                            strategy_name=strategy_name,
                        )
                    )

    return sorted(slices, key=_slice_sort_key)


def _build_task_class() -> type:
    try:
        from mteb.abstasks import AbsTaskRetrieval
    except ModuleNotFoundError as exc:
        raise ValueError(
            "MTEB benchmarking requires the `mteb` package. Install project dependencies first."
        ) from exc

    class HubDatasetRetrievalTask(AbsTaskRetrieval):
        def __init__(
            self,
            metadata: Any,
            *,
            dataset_repo: str,
            revision: str,
            slice_filter: BenchmarkSlice,
            source: BenchmarkSource,
        ):
            self.metadata = metadata
            self.dataset_repo = dataset_repo
            self.revision = revision
            self.slice_filter = slice_filter
            self.source = source
            super().__init__()

        def load_data(self, **_: Any) -> None:
            corpus_rows = _load_benchmark_split(self.source, "corpus")
            query_rows = [
                row
                for row in _load_benchmark_split(self.source, "queries")
                if self.slice_filter.matches(row)
            ]
            qrel_rows = _load_benchmark_split(self.source, "qrels")

            query_texts = {
                str(row.get("_id", "")).strip(): str(row.get("text", "")).strip()
                for row in query_rows
                if str(row.get("_id", "")).strip()
            }
            if not query_texts:
                raise ValueError(
                    f"No queries matched slice `{self.slice_filter.name}` in `{self.dataset_repo}`."
                )

            relevant_docs: dict[str, dict[str, float]] = defaultdict(dict)
            for row in qrel_rows:
                query_id = str(row.get("query-id", "")).strip()
                corpus_id = str(row.get("corpus-id", "")).strip()
                if query_id not in query_texts or not corpus_id:
                    continue
                relevant_docs[query_id][corpus_id] = float(row.get("score", 1.0))

            query_texts = {
                query_id: text
                for query_id, text in query_texts.items()
                if query_id in relevant_docs
            }
            if not query_texts:
                raise ValueError(
                    f"Slice `{self.slice_filter.name}` matched queries but no qrels in `{self.dataset_repo}`."
                )

            corpus = {}
            for row in corpus_rows:
                corpus_id = str(row.get("_id", row.get("id", ""))).strip()
                if not corpus_id:
                    continue
                corpus[corpus_id] = {
                    "title": str(row.get("title", "")).strip(),
                    "text": str(row.get("text", row.get("context", ""))).strip(),
                }

            self.corpus = {"train": corpus}
            self.queries = {"train": query_texts}
            self.relevant_docs = {"train": dict(relevant_docs)}
            self.data_loaded = True

    return HubDatasetRetrievalTask


def build_mteb_tasks(
    dataset_repo: str = DEFAULT_MTEB_DATASET_REPO,
    *,
    local_corpus_path: str = DEFAULT_MTEB_LOCAL_CORPUS_PATH,
    local_qac_path: str = DEFAULT_MTEB_LOCAL_QAC_PATH,
    revision: str = "main",
    include_mode_strategy: bool = False,
) -> list[Any]:
    try:
        from mteb.abstasks.task_metadata import TaskMetadata
    except ModuleNotFoundError as exc:
        raise ValueError(
            "MTEB benchmarking requires the `mteb` package. Install project dependencies first."
        ) from exc

    task_class = _build_task_class()
    tasks = []
    source = BenchmarkSource(
        dataset_repo=dataset_repo,
        local_corpus_path=local_corpus_path,
        local_qac_path=local_qac_path,
        revision=revision,
    )
    base_name = _dataset_task_name(source.label)
    for slice_filter in _detect_benchmark_slices(
        source,
        include_mode_strategy=include_mode_strategy,
    ):
        eval_langs = _detect_query_languages(
            source,
            slice_filter=slice_filter,
        )
        metadata = TaskMetadata(
            name=f"{base_name}__{slice_filter.name}",
            dataset={"path": source.label, "revision": revision},
            description=(
                f"Custom retrieval evaluation over `{source.label}` "
                f"for slice `{slice_filter.label}`."
            ),
            reference=(
                f"https://huggingface.co/datasets/{dataset_repo}"
                if source.uses_hf
                else PROJECT_REFERENCE_URL
            ),
            type="Retrieval",
            category="t2t",
            modalities=["text"],
            eval_splits=["train"],
            eval_langs=eval_langs,
            main_score=DEFAULT_MTEB_MAIN_SCORE,
            domains=["Chemistry", "Engineering"],
            task_subtypes=["Question Answering Retrieval"],
            license="not specified",
            annotations_creators="LM-generated and reviewed",
            sample_creation="LM-generated and verified",
            is_public=True,
            contributed_by="multi-lingual-qac",
        )
        tasks.append(
            task_class(
                metadata,
                dataset_repo=source.dataset_repo,
                revision=source.revision,
                slice_filter=slice_filter,
                source=source,
            )
        )
    return tasks


def _extract_numeric_metrics(result: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_rows in result.scores.values():
        for row in split_rows:
            for key, value in row.items():
                if key in {"hf_subset", "languages", "main_score"}:
                    continue
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            if metrics:
                return metrics
    return metrics


def _summary_sort_key(item: ModelEvaluationSummary) -> tuple[tuple[int, int, int, str], float, str]:
    slice_filter = BenchmarkSlice(
        name=item.slice_name,
        label=item.slice_label,
        mode=item.filter_mode or None,
        strategy_name=item.filter_strategy_name or None,
    )
    return (_slice_sort_key(slice_filter), -item.main_score, item.model_name.lower())


def _group_summaries_by_slice(
    summaries: list[ModelEvaluationSummary],
) -> list[tuple[str, str, list[ModelEvaluationSummary]]]:
    grouped: dict[str, list[ModelEvaluationSummary]] = defaultdict(list)
    labels: dict[str, str] = {}
    for item in summaries:
        grouped[item.slice_name].append(item)
        labels[item.slice_name] = item.slice_label

    ordered_slices = sorted(
        grouped.keys(),
        key=lambda name: _slice_sort_key(
            BenchmarkSlice(
                name=name,
                label=labels[name],
                mode=grouped[name][0].filter_mode or None,
                strategy_name=grouped[name][0].filter_strategy_name or None,
            )
        ),
    )
    return [
        (
            slice_name,
            labels[slice_name],
            sorted(grouped[slice_name], key=lambda item: item.main_score, reverse=True),
        )
        for slice_name in ordered_slices
    ]


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
        "slices": [
            {
                "slice_name": slice_name,
                "slice_label": slice_label,
                "models": [
                    {
                        "model_name": item.model_name,
                        "model_slug": item.model_slug,
                        "filter_mode": item.filter_mode,
                        "filter_strategy_name": item.filter_strategy_name,
                        "task_name": item.task_name,
                        "main_score": item.main_score,
                        "metrics": item.metrics,
                        "output_dir": item.output_dir,
                        "eval_languages": item.eval_languages,
                        "evaluation_time_seconds": item.evaluation_time_seconds,
                    }
                    for item in items
                ],
            }
            for slice_name, slice_label, items in _group_summaries_by_slice(summaries)
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
                "slice_name",
                "slice_label",
                "filter_mode",
                "filter_strategy_name",
                "model_name",
                "main_score",
                "evaluation_time_seconds",
                "eval_languages",
                "output_dir",
                *metric_keys,
            ],
        )
        writer.writeheader()
        for item in sorted(summaries, key=_summary_sort_key):
            row: dict[str, Any] = {
                "slice_name": item.slice_name,
                "slice_label": item.slice_label,
                "filter_mode": item.filter_mode,
                "filter_strategy_name": item.filter_strategy_name,
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
    ]
    for slice_name, slice_label, items in _group_summaries_by_slice(summaries):
        lines.extend(
            [
                f"## {slice_label}",
                "",
                "| Model | Main score | Eval time (s) |",
                "| --- | ---: | ---: |",
            ]
        )
        for item in items:
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
        lines.append("")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_summary_models(results_dir: Path) -> tuple[str, list[ModelEvaluationSummary]]:
    summary_json = results_dir / "summary.json"
    if not summary_json.exists():
        return _load_raw_result_models(results_dir)

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    dataset_repo = str(payload.get("dataset_repo", DEFAULT_MTEB_DATASET_REPO))
    summaries: list[ModelEvaluationSummary] = []
    for slice_payload in payload.get("slices", []):
        slice_name = str(slice_payload.get("slice_name", "overall"))
        slice_label = str(slice_payload.get("slice_label", _slice_label_from_name(slice_name)))
        for item in slice_payload.get("models", []):
            summaries.append(
                ModelEvaluationSummary(
                    model_name=str(item.get("model_name", "")).strip(),
                    model_slug=str(item.get("model_slug", "")).strip()
                    or _slugify(str(item.get("model_name", ""))),
                    slice_name=slice_name,
                    slice_label=slice_label,
                    filter_mode=str(item.get("filter_mode", "")).strip(),
                    filter_strategy_name=str(item.get("filter_strategy_name", "")).strip(),
                    task_name=str(item.get("task_name", "")).strip(),
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
            if key not in {"hf_subset", "languages", "main_score"} and isinstance(value, (int, float))
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

        task_name = result_file.stem
        slice_name = _slice_name_from_task_name(task_name)
        summaries.append(
            ModelEvaluationSummary(
                model_name=model_name,
                model_slug=model_slug,
                slice_name=slice_name,
                slice_label=_slice_label_from_name(slice_name),
                filter_mode="",
                filter_strategy_name="",
                task_name=task_name,
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
    grouped: list[tuple[str, str, list[ModelEvaluationSummary]]],
) -> str:
    lines = [
        "# MTEB Model Comparison",
        "",
        f"- Dataset: `{dataset_repo}`",
        f"- Main score: `{DEFAULT_MTEB_MAIN_SCORE}`",
        "",
    ]
    for _, slice_label, ranked in grouped:
        best = _best_metric_values(ranked, COMPARISON_METRICS)
        top = ranked[0]
        lines.extend(
            [
                f"## {slice_label}",
                "",
                f"- Best model by `{DEFAULT_MTEB_MAIN_SCORE}`: `{top.model_name}` ({top.main_score:.4f})",
                "",
                "| Rank | Model | Main score | nDCG@10 | MAP@10 | MRR@10 | Hit@10 | Recall@10 | Time (s) |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
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
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_latex_comparison(
    dataset_repo: str,
    grouped: list[tuple[str, str, list[ModelEvaluationSummary]]],
) -> str:
    lines = []
    for _, slice_label, ranked in grouped:
        best = _best_metric_values(ranked, COMPARISON_METRICS)
        lines.extend(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                rf"\caption{{MTEB retrieval comparison on \texttt{{{_latex_escape(dataset_repo)}}} for {_latex_escape(slice_label)}.}}",
                r"\begin{tabular}{r l r r r r r r}",
                r"\hline",
                r"Rank & Model & Main & MAP@10 & MRR@10 & Hit@10 & Recall@10 & Time (s) \\",
                r"\hline",
            ]
        )
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

    grouped = _group_summaries_by_slice(summaries)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_json = output_path / "model_comparison.json"
    comparison_csv = output_path / "model_comparison.csv"
    comparison_md = output_path / "model_comparison.md"
    comparison_tex = output_path / "model_comparison.tex"

    payload = {
        "dataset_repo": dataset_repo,
        "results_dir": str(results_path),
        "metrics": COMPARISON_METRICS,
        "slices": [
            {
                "slice_name": slice_name,
                "slice_label": slice_label,
                "models": [
                    {
                        "rank": idx,
                        "model_name": item.model_name,
                        "main_score": item.main_score,
                        "evaluation_time_seconds": item.evaluation_time_seconds,
                        "output_dir": item.output_dir,
                        "metrics": {
                            metric: item.metrics.get(metric)
                            for metric in COMPARISON_METRICS
                        },
                    }
                    for idx, item in enumerate(ranked, start=1)
                ],
            }
            for slice_name, slice_label, ranked in grouped
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
                "slice_name",
                "slice_label",
                "rank",
                "model_name",
                "evaluation_time_seconds",
                "output_dir",
                *COMPARISON_METRICS,
            ],
        )
        writer.writeheader()
        for slice_name, slice_label, ranked in grouped:
            for idx, item in enumerate(ranked, start=1):
                row: dict[str, Any] = {
                    "slice_name": slice_name,
                    "slice_label": slice_label,
                    "rank": idx,
                    "model_name": item.model_name,
                    "main_score": item.main_score,
                    "evaluation_time_seconds": item.evaluation_time_seconds,
                    "output_dir": item.output_dir,
                }
                for metric in COMPARISON_METRICS:
                    row[metric] = item.metrics.get(
                        metric,
                        item.main_score if metric == "main_score" else "",
                    )
                writer.writerow(row)

    comparison_md.write_text(
        _build_markdown_comparison(dataset_repo, grouped),
        encoding="utf-8",
    )
    comparison_tex.write_text(
        _build_latex_comparison(dataset_repo, grouped),
        encoding="utf-8",
    )
    return output_path


def run_mteb_evaluation(
    models: list[str],
    *,
    dataset_repo: str = DEFAULT_MTEB_DATASET_REPO,
    local_corpus_path: str = DEFAULT_MTEB_LOCAL_CORPUS_PATH,
    local_qac_path: str = DEFAULT_MTEB_LOCAL_QAC_PATH,
    output_dir: str | Path = DEFAULT_MTEB_OUTPUT_DIR,
    revision: str = "main",
    batch_size: int = 32,
    include_mode_strategy: bool = False,
) -> list[ModelEvaluationSummary]:
    if not models:
        raise ValueError("Provide at least one model name for MTEB evaluation.")

    try:
        from mteb import MTEB
    except ModuleNotFoundError as exc:
        raise ValueError(
            "MTEB benchmarking requires the `mteb` package. Install project dependencies first."
        ) from exc
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ValueError(
            "MTEB benchmarking requires the `sentence-transformers` package. Install project dependencies first."
        ) from exc

    source = BenchmarkSource(
        dataset_repo=dataset_repo,
        local_corpus_path=local_corpus_path,
        local_qac_path=local_qac_path,
        revision=revision,
    )

    tasks = build_mteb_tasks(
        dataset_repo,
        local_corpus_path=local_corpus_path,
        local_qac_path=local_qac_path,
        revision=revision,
        include_mode_strategy=include_mode_strategy,
    )
    evaluator = MTEB(tasks=tasks)
    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = _configure_local_model_cache()
    summaries: list[ModelEvaluationSummary] = []

    for model_name in models:
        model_slug = _slugify(model_name)
        print(f"Evaluating `{model_name}` on `{source.label}`...")
        has_cached_model = _has_cached_model(model_cache_dir, model_name)
        try:
            model = SentenceTransformer(
                model_name,
                cache_folder=str(model_cache_dir),
                local_files_only=has_cached_model,
            )
        except Exception as exc:
            if not has_cached_model:
                raise
            print(
                f"  Cached model load failed ({exc}). Retrying with standard resolution..."
            )
            model = SentenceTransformer(
                model_name,
                cache_folder=str(model_cache_dir),
                local_files_only=False,
            )
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

        if len(results) != len(tasks):
            raise ValueError(
                f"MTEB returned {len(results)} results for {len(tasks)} tasks on `{model_name}`."
            )

        for task, result in zip(tasks, results):
            slice_filter = task.slice_filter
            metrics = _extract_numeric_metrics(result)
            summaries.append(
                ModelEvaluationSummary(
                    model_name=model_name,
                    model_slug=model_slug,
                    slice_name=slice_filter.name,
                    slice_label=slice_filter.label,
                    filter_mode=slice_filter.mode or "",
                    filter_strategy_name=slice_filter.strategy_name or "",
                    task_name=task.metadata.name,
                    main_score=float(result.main_score),
                    metrics=metrics,
                    output_dir=str(model_output_dir / task.metadata.name),
                    eval_languages=_detect_query_languages(
                        source,
                        slice_filter=slice_filter,
                    ),
                    evaluation_time_seconds=result.evaluation_time,
                )
            )

    _write_summary_reports(base_output_dir, source.label, summaries)
    return summaries
