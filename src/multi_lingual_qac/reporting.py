from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from src.multi_lingual_qac.config import PipelinePaths

JRC_REPORT_RELATIVE_PATH = Path("reports") / "jrc-acquis" / "jrc_pipeline_generation_report.md"
JRC_REPORT_JSON_RELATIVE_PATH = Path("reports") / "jrc-acquis" / "jrc_pipeline_generation_report.json"


def update_jrc_pipeline_report(paths: PipelinePaths, *, stage: str = "") -> Path:
    """Write a consolidated numerical report for the current JRC pipeline state."""
    if paths.source != "jrc-acquis":
        return paths.project_root / JRC_REPORT_RELATIVE_PATH

    report_path = paths.project_root / JRC_REPORT_RELATIVE_PATH
    payload_path = paths.project_root / JRC_REPORT_JSON_RELATIVE_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)

    raw_stats = _read_json(paths.prepared_dir / "raw_load_stats.json")
    corpus_stats = _read_json(paths.preprocessed_dir / "document_corpus_stats.json")
    selection_stats = _read_json(paths.qac_dir / "qa_selection_stats.json")
    generation_stats = _read_json(paths.qac_dir / "qac_generation_stats.json")
    qac_summary = _summarize_qac(paths.qac_dir / "qac.csv")

    payload = {
        "stage": stage,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_load": raw_stats,
        "corpus_build": corpus_stats,
        "qa_selection": selection_stats,
        "qa_generation": generation_stats,
        "qac_summary": qac_summary,
        "artifacts": _artifact_status(paths),
    }
    payload_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(_render_jrc_report(payload), encoding="utf-8")
    return report_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_error": f"Could not read {path}"}
    return data if isinstance(data, dict) else {"value": data}


def _csv_row_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    _set_csv_field_size_limit()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        return max(0, sum(1 for _ in reader) - 1)


def _set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _parse_json_list(value: str) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_json_dict(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _summarize_qac(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}

    rows = 0
    base_rows = 0
    synthetic_rows = 0
    by_language: Counter[str] = Counter()
    synthetic_by_language: Counter[str] = Counter()
    support_checked = 0
    supported_translation_links = 0
    supported_translation_links_by_language: Counter[str] = Counter()
    linked_positive_counts: Counter[int] = Counter()
    support_reason_counts: Counter[int] = Counter()

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            language = str(row.get("language", "")).strip().lower() or "unknown"
            by_language[language] += 1
            is_synthetic = str(row.get("is_synthetic_translation", "")).strip().lower() == "true"
            if is_synthetic:
                synthetic_rows += 1
                synthetic_by_language[language] += 1
            else:
                base_rows += 1

            linked_ids = _parse_json_list(str(row.get("linked_corpus_ids_json", "")))
            if linked_ids:
                linked_positive_counts[len(linked_ids)] += 1

            if str(row.get("cross_language_support_checked", "")).strip().lower() == "true":
                support_checked += 1
                supported_ids = _parse_json_list(
                    str(row.get("cross_language_supported_corpus_ids_json", ""))
                )
                supported_translation_links += len(supported_ids)
                supported_languages = _parse_json_list(str(row.get("linked_languages_json", "")))
                source_lang = str(row.get("source_language", "")).strip().lower()
                for lang in supported_languages:
                    normalized = str(lang).strip().lower()
                    if normalized and normalized != source_lang:
                        supported_translation_links_by_language[normalized] += 1
                support_reasons = _parse_json_dict(
                    str(row.get("cross_language_support_reasons_json", ""))
                )
                support_reason_counts[len(support_reasons)] += 1

    return {
        "qac_rows": rows,
        "base_rows": base_rows,
        "synthetic_translation_rows": synthetic_rows,
        "rows_by_language": dict(sorted(by_language.items())),
        "synthetic_translation_rows_by_language": dict(sorted(synthetic_by_language.items())),
        "cross_language_support_checked_rows": support_checked,
        "supported_translation_links": supported_translation_links,
        "supported_translation_links_by_language": dict(
            sorted(supported_translation_links_by_language.items())
        ),
        "linked_positive_count_distribution": {
            str(count): freq for count, freq in sorted(linked_positive_counts.items())
        },
        "checked_translation_count_distribution": {
            str(count): freq for count, freq in sorted(support_reason_counts.items())
        },
    }


def _artifact_status(paths: PipelinePaths) -> dict[str, Any]:
    artifacts = {
        "raw_documents_jsonl": paths.prepared_dir / "raw_documents.jsonl",
        "raw_load_stats_json": paths.prepared_dir / "raw_load_stats.json",
        "source_document_pool_full_csv": paths.corpus_full_csv,
        "source_document_pool_csv": paths.corpus_csv,
        "document_corpus_stats_json": paths.preprocessed_dir / "document_corpus_stats.json",
        "corpus_multilingual_full_csv": paths.preprocessed_dir / "corpus_multilingual_full.csv",
        "corpus_qa_candidates_csv": paths.preprocessed_dir / "corpus_qa_candidates.csv",
        "qa_selection_stats_json": paths.qac_dir / "qa_selection_stats.json",
        "qa_generation_sources_csv": paths.qac_dir / "qa_generation_sources.csv",
        "benchmark_hf_corpus_csv": paths.qac_dir / "corpus.csv",
        "qac_csv": paths.qac_dir / "qac.csv",
        "qac_generation_stats_json": paths.qac_dir / "qac_generation_stats.json",
    }
    return {
        name: {
            "exists": path.is_file(),
            "path": str(path),
            "rows": _csv_row_count(path) if path.suffix.lower() == ".csv" else None,
        }
        for name, path in artifacts.items()
    }


def _render_jrc_report(payload: dict[str, Any]) -> str:
    lines = [
        "# JRC-Acquis Pipeline Generation Report",
        "",
        f"- Last updated: `{payload.get('updated_at', '')}`",
        f"- Last stage: `{payload.get('stage') or 'unknown'}`",
        "",
        "This report is regenerated after each JRC pipeline stage and consolidates the numerical audit trail from the current local artifacts.",
        "",
    ]

    raw_stats = payload.get("raw_load", {})
    corpus_stats = payload.get("corpus_build", {})
    selection_stats = payload.get("qa_selection", {})
    generation_stats = payload.get("qa_generation", {})
    qac_summary = payload.get("qac_summary", {})

    lines.extend(_section("1. Raw Loading", _raw_metrics(raw_stats)))
    lines.extend(_dict_table("Raw documents by language", raw_stats.get("languages")))
    lines.extend(
        _section(
            "2. Corpus Cleaning And Build",
            _corpus_metrics(raw_stats, corpus_stats),
        )
    )
    lines.extend(_dict_table("Built documents by language", corpus_stats.get("languages")))
    lines.extend(
        _dict_table(
            "QA candidate rejection reasons",
            corpus_stats.get("qa_rejection_reasons"),
        )
    )
    lines.extend(_section("3. QA Source Selection", _selection_metrics(selection_stats)))
    lines.extend(
        _nested_language_table(
            "QA source selection by language",
            selection_stats.get("languages"),
        )
    )
    lines.extend(_section("4. Q&A Generation", _generation_metrics(generation_stats)))
    lines.extend(
        _dict_table(
            "Generation outcomes by source language",
            generation_stats.get("results_by_language"),
        )
    )
    lines.extend(
        _dict_table(
            "Rejected source documents by reason",
            generation_stats.get("skipped_by_reason"),
        )
    )
    lines.extend(
        _dict_table(
            "Failed synthetic translations by language",
            generation_stats.get("failed_translation_languages"),
        )
    )
    lines.extend(_section("5. Final QAC", _qac_metrics(qac_summary)))
    lines.extend(_dict_table("QAC rows by language", qac_summary.get("rows_by_language")))
    lines.extend(
        _dict_table(
            "Supported translation links by language",
            qac_summary.get("supported_translation_links_by_language"),
        )
    )
    lines.extend(_artifact_table(payload.get("artifacts", {})))
    return "\n".join(lines).rstrip() + "\n"


def _section(title: str, rows: list[tuple[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["No data yet.", ""])
        return lines
    lines.extend(["| Metric | Value |", "| --- | ---: |"])
    for key, value in rows:
        lines.append(f"| {key} | {_format_value(value)} |")
    lines.append("")
    return lines


def _dict_table(title: str, values: Any) -> list[str]:
    if not isinstance(values, dict) or not values:
        return [f"### {title}", "", "No data yet.", ""]
    lines = [f"### {title}", "", "| Key | Value |", "| --- | ---: |"]
    for key, value in sorted(values.items(), key=lambda item: str(item[0])):
        lines.append(f"| `{key}` | {_format_value(value)} |")
    lines.append("")
    return lines


def _nested_language_table(title: str, values: Any) -> list[str]:
    if not isinstance(values, dict) or not values:
        return [f"### {title}", "", "No data yet.", ""]
    metric_names = sorted(
        {
            metric
            for metrics in values.values()
            if isinstance(metrics, dict)
            for metric in metrics
        }
    )
    if not metric_names:
        return [f"### {title}", "", "No data yet.", ""]
    lines = [
        f"### {title}",
        "",
        "| Language | " + " | ".join(metric_names) + " |",
        "| --- | " + " | ".join(["---:"] * len(metric_names)) + " |",
    ]
    for lang, metrics in sorted(values.items()):
        metric_values = [
            _format_value(metrics.get(metric, "")) if isinstance(metrics, dict) else ""
            for metric in metric_names
        ]
        lines.append(f"| `{lang}` | " + " | ".join(metric_values) + " |")
    lines.append("")
    return lines


def _artifact_table(artifacts: dict[str, Any]) -> list[str]:
    lines = ["## 6. Artifact Status", ""]
    if not artifacts:
        lines.extend(["No data yet.", ""])
        return lines
    lines.extend(["| Artifact | Exists | Rows | Path |", "| --- | ---: | ---: | --- |"])
    for name, info in sorted(artifacts.items()):
        if not isinstance(info, dict):
            continue
        lines.append(
            f"| `{name}` | {_format_value(info.get('exists'))} | "
            f"{_format_value(info.get('rows'))} | `{info.get('path', '')}` |"
        )
    lines.append("")
    return lines


def _raw_metrics(stats: dict[str, Any]) -> list[tuple[str, Any]]:
    if not stats:
        return []
    return [
        ("Chemical-only filter active", stats.get("chemical_only")),
        ("Documents loaded", stats.get("documents_loaded")),
        ("Documents filtered as non-chemical", stats.get("documents_filtered_non_chemical")),
        ("Languages loaded", len(stats.get("languages", {}) or {})),
        ("Workers", stats.get("workers")),
    ]


def _corpus_metrics(raw_stats: dict[str, Any], stats: dict[str, Any]) -> list[tuple[str, Any]]:
    if not stats:
        return []
    raw_loaded = raw_stats.get("documents_loaded")
    written = stats.get("documents_written")
    return [
        ("Documents entering build", raw_loaded),
        ("Documents written after cleaning", written),
        ("Documents lost during build", _difference(raw_loaded, written)),
        ("Documents filtered as non-chemical during build", stats.get("documents_filtered_non_chemical")),
        ("Documents under 1500 chars", stats.get("docs_under_1500_chars")),
        ("Documents over 30000 chars", stats.get("docs_over_30000_chars")),
        ("Documents with formatting cleaned", stats.get("docs_with_formatting_cleaned")),
        ("Documents trimmed to operative body", stats.get("docs_trimmed_to_operative_body")),
        ("CELEX ids total", stats.get("celex_total")),
        ("Multilingual CELEX ids", stats.get("celex_multilingual")),
        ("All language pairs written", stats.get("pairs_written")),
        ("Multilingual documents", stats.get("multilingual_docs_written")),
        ("QA candidates", stats.get("qa_candidates_written")),
        ("QA candidate filter profile", (stats.get("qa_filter") or {}).get("profile")),
    ]


def _selection_metrics(stats: dict[str, Any]) -> list[tuple[str, Any]]:
    if not stats:
        return []
    return [
        ("Source pool documents sampled", stats.get("sampled_source_pool_docs_total")),
        ("Generation source candidates", stats.get("selected_generation_source_docs_total")),
        ("Requested accepted docs per language", stats.get("generation_docs_per_language_requested")),
        ("Oversample factor", stats.get("generation_source_oversample_factor")),
        ("Selected CELEX groups", stats.get("selected_generation_celex_groups_total")),
        ("Final retrieval corpus documents", stats.get("final_retrieval_corpus_docs_total")),
        ("Generation units", stats.get("generation_units_total")),
        ("Average linked positives per generation unit", stats.get("avg_relevant_docs_per_generation_unit")),
        ("Synthetic target languages", ", ".join(stats.get("synthetic_target_languages", []) or [])),
    ]


def _generation_metrics(stats: dict[str, Any]) -> list[tuple[str, Any]]:
    if not stats:
        return []
    return [
        ("Source rows loaded", stats.get("source_rows_loaded")),
        ("Source rows sampled", stats.get("source_rows_sampled")),
        ("Source rows processed", stats.get("source_rows_processed")),
        ("Source rows accepted", stats.get("source_rows_accepted")),
        ("Source rows skipped", stats.get("source_rows_skipped")),
        ("Source row errors", stats.get("source_rows_error")),
        ("QAC rows written", stats.get("qac_rows_written")),
        ("Base QAC rows written", stats.get("base_qac_rows_written")),
        ("Synthetic translation rows written", stats.get("synthetic_translation_rows_written")),
        ("Failed synthetic translations", stats.get("failed_translation_count")),
        ("Cross-language support failures", stats.get("cross_language_support_failures")),
        ("Total validation rejection attempts", stats.get("validation_rejection_attempts")),
        ("Average approved attempt", stats.get("average_approved_attempt")),
        ("Accepted per language target", stats.get("accepted_per_language")),
    ]


def _qac_metrics(summary: dict[str, Any]) -> list[tuple[str, Any]]:
    if not summary:
        return []
    return [
        ("QAC rows", summary.get("qac_rows")),
        ("Base rows", summary.get("base_rows")),
        ("Synthetic translation rows", summary.get("synthetic_translation_rows")),
        ("Rows with cross-language support checked", summary.get("cross_language_support_checked_rows")),
        ("Supported translation links", summary.get("supported_translation_links")),
    ]


def _difference(left: Any, right: Any) -> int | str:
    try:
        return int(left) - int(right)
    except Exception:
        return ""


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "`" + json.dumps(value, ensure_ascii=False, sort_keys=True) + "`"
    return str(value)
