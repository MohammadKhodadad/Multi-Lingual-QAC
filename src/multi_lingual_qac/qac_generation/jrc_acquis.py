from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any


def _set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    _set_csv_field_size_limit()
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _iter_csv_rows(path: Path):
    _set_csv_field_size_limit()
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            yield dict(row)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_jrc_qa_inputs(
    *,
    corpus_full_path: Path,
    qa_candidates_path: Path,
    output_dir: Path,
    pairs_per_language: int,
    generation_docs_per_language: int,
    allowed_languages: tuple[str, ...] | None = None,
    synthetic_target_languages: tuple[str, ...] = (),
    seed: int = 42,
) -> dict[str, Any]:
    """
    Prepare a JRC-Acquis QA subset.

    Workflow:
    - group multilingual documents by CELEX
    - sample source-side QA candidates per language from multilingual CELEX groups
    - retain a bounded number of source documents per language
    - choose one target-language realization per selected source document
    - build a retrieval corpus from the sampled source pool plus all retained
      CELEX-linked positives for the selected generation units
    - write QA generation rows using the chosen target-side document
    - map each generated query to all retained sampled documents for the same CELEX
    - write helper CSVs into `output_dir`
    """
    corpus_full_path = Path(corpus_full_path)
    qa_candidates_path = Path(qa_candidates_path)
    output_dir = Path(output_dir)

    if not corpus_full_path.is_file():
        raise ValueError(f"Missing JRC corpus full file: {corpus_full_path}")
    if not qa_candidates_path.is_file():
        raise ValueError(f"Missing JRC QA candidates file: {qa_candidates_path}")
    if pairs_per_language <= 0:
        raise ValueError("pairs_per_language must be > 0")
    if generation_docs_per_language <= 0:
        raise ValueError("generation_docs_per_language must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    allowed_language_set = {
        lang.strip().lower()
        for lang in (allowed_languages or ())
        if lang and lang.strip()
    }
    synthetic_target_language_list = [
        lang.strip().lower()
        for lang in synthetic_target_languages
        if lang and lang.strip()
    ]

    corpus_fieldnames: list[str] = []
    docs_by_id: dict[str, dict[str, str]] = {}
    docs_by_celex: dict[str, list[dict[str, str]]] = defaultdict(list)
    allowed_corpus_rows: list[dict[str, str]] = []
    for row in _iter_csv_rows(corpus_full_path):
        if not corpus_fieldnames:
            corpus_fieldnames = list(row.keys())
        row_lang = row.get("language", "").strip().lower()
        if allowed_language_set and row_lang not in allowed_language_set:
            continue
        allowed_corpus_rows.append(row)
        row_id = row.get("id", "")
        celex = row.get("celex", "")
        if row_id:
            docs_by_id[row_id] = row
        if celex:
            docs_by_celex[celex].append(row)

    qa_candidate_rows: list[dict[str, str]] = []
    qa_candidate_fieldnames: list[str] = []
    reservoir_by_lang: dict[str, list[dict[str, str]]] = defaultdict(list)
    available_source_docs_by_lang: Counter[str] = Counter()
    for row in _iter_csv_rows(qa_candidates_path):
        qa_candidate_rows.append(row)
        if not qa_candidate_fieldnames:
            qa_candidate_fieldnames = list(row.keys())
        row_id = row.get("id", "")
        celex = row.get("celex", "")
        lang = row.get("language", "").strip().lower()
        if not row_id or not celex or not lang:
            continue
        if allowed_language_set and lang not in allowed_language_set:
            continue
        eligible_targets = [
            doc
            for doc in docs_by_celex.get(celex, [])
            if doc.get("id", "") != row_id
            and (
                not allowed_language_set
                or doc.get("language", "").strip().lower() in allowed_language_set
            )
        ]
        if not eligible_targets:
            continue
        available_source_docs_by_lang[lang] += 1
        reservoir = reservoir_by_lang[lang]
        seen = available_source_docs_by_lang[lang]
        if len(reservoir) < pairs_per_language:
            reservoir.append(row)
        else:
            replace_idx = rng.randint(1, seen)
            if replace_idx <= pairs_per_language:
                reservoir[replace_idx - 1] = row

    sampled_source_rows: list[dict[str, str]] = []
    selected_source_rows: list[dict[str, str]] = []
    selected_sources_stats: dict[str, dict[str, int]] = {}
    for lang in sorted(reservoir_by_lang):
        sampled_lang_rows = list(reservoir_by_lang[lang])
        if not sampled_lang_rows:
            continue
        sampled_source_rows.extend(sampled_lang_rows)
        if len(sampled_lang_rows) > generation_docs_per_language:
            chosen_rows = rng.sample(sampled_lang_rows, generation_docs_per_language)
        else:
            chosen_rows = list(sampled_lang_rows)
        chosen_rows.sort(key=lambda row: row.get("id", ""))
        selected_source_rows.extend(chosen_rows)
        selected_sources_stats[lang] = {
            "available_source_docs": int(available_source_docs_by_lang[lang]),
            "sampled_source_docs": len(sampled_lang_rows),
            "selected_source_docs": len(chosen_rows),
        }

    sampled_source_rows.sort(key=lambda row: (row.get("language", ""), row.get("id", "")))
    selected_source_rows.sort(key=lambda row: (row.get("language", ""), row.get("id", "")))

    sampled_source_pool_ids = {
        row.get("id", "")
        for row in sampled_source_rows
        if row.get("id", "")
    }
    selected_celexes = {
        row.get("celex", "")
        for row in selected_source_rows
        if row.get("celex", "")
    }
    final_corpus_rows_by_id: dict[str, dict[str, str]] = {}
    for sampled_source_id in sampled_source_pool_ids:
        row = docs_by_id.get(sampled_source_id)
        if row is not None:
            final_corpus_rows_by_id[sampled_source_id] = row

    for row in allowed_corpus_rows:
        row_id = row.get("id", "")
        if not row_id or row_id in final_corpus_rows_by_id:
            continue
        if row.get("celex", "") in selected_celexes:
            final_corpus_rows_by_id[row_id] = row

    corpus_subset_rows = list(final_corpus_rows_by_id.values())
    corpus_subset_rows.sort(key=lambda row: (row.get("celex", ""), row.get("language", ""), row.get("id", "")))
    subset_docs_by_celex: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in corpus_subset_rows:
        celex = row.get("celex", "")
        if celex:
            subset_docs_by_celex[celex].append(row)
    corpus_subset_mteb_rows = [
        {"_id": row["id"], "title": row.get("title", ""), "text": row.get("context", "")}
        for row in corpus_subset_rows
    ]

    generation_rows: list[dict[str, str]] = []
    generation_target_language_counts: Counter[str] = Counter()
    relevant_docs_per_generation_unit: Counter[int] = Counter()
    generation_rows_by_source_language: Counter[str] = Counter()
    for source_row in selected_source_rows:
        source_id = source_row.get("id", "")
        source_lang = source_row.get("language", "")
        celex = source_row.get("celex", "")
        if not source_id or not source_lang or not celex:
            continue
        linked_docs = subset_docs_by_celex.get(celex, [])
        if len([row for row in linked_docs if row.get("id", "")]) < 2:
            continue
        linked_corpus_ids = [row["id"] for row in linked_docs if row.get("id", "")]
        linked_languages = [row.get("language", "") for row in linked_docs if row.get("language", "")]
        if not linked_corpus_ids:
            linked_corpus_ids = [source_id]
            linked_languages = [source_lang]
        generation_row = dict(source_row)
        generation_row["celex"] = celex
        generation_row["source_language"] = source_lang
        generation_row["source_corpus_id"] = source_id
        generation_row["target_language"] = source_lang
        generation_row["target_corpus_id"] = source_id
        generation_row["query_corpus_id"] = source_id
        generation_row["query_id_hint"] = (
            f"{celex}__{source_lang}__{source_lang}"
        )
        generation_row["synthetic_target_languages_json"] = json.dumps(
            synthetic_target_language_list,
            ensure_ascii=False,
        )
        generation_row["linked_corpus_ids_json"] = json.dumps(linked_corpus_ids, ensure_ascii=False)
        generation_row["linked_languages_json"] = json.dumps(linked_languages, ensure_ascii=False)
        generation_row["linked_corpus_count"] = str(len(linked_corpus_ids))
        generation_rows.append(generation_row)
        generation_target_language_counts[source_lang] += 1
        generation_rows_by_source_language[source_lang] += 1
        relevant_docs_per_generation_unit[len(linked_corpus_ids)] += 1

    generation_rows.sort(
        key=lambda row: (
            row.get("source_language", ""),
            row.get("source_corpus_id", ""),
            row.get("target_language", ""),
            row.get("target_corpus_id", ""),
        )
    )

    sampled_sources_path = output_dir / "sampled_sources.csv"
    selected_sources_path = output_dir / "qa_generation_sources.csv"
    corpus_subset_full_path = output_dir / "corpus_full.csv"
    corpus_subset_path = output_dir / "corpus.csv"
    selection_stats_path = output_dir / "qa_selection_stats.json"

    _write_csv(
        sampled_sources_path,
        qa_candidate_fieldnames,
        sampled_source_rows,
    )
    generation_row_fieldnames = (
        list(generation_rows[0].keys())
        if generation_rows
        else qa_candidate_fieldnames + [
            "celex",
            "source_language",
            "source_corpus_id",
            "target_language",
            "target_corpus_id",
            "query_corpus_id",
            "query_id_hint",
            "synthetic_target_languages_json",
            "linked_corpus_ids_json",
            "linked_languages_json",
            "linked_corpus_count",
        ]
    )

    if generation_rows:
        _write_csv(
            selected_sources_path,
            generation_row_fieldnames,
            generation_rows,
        )
    else:
        _write_csv(selected_sources_path, generation_row_fieldnames, [])

    _write_csv(corpus_subset_full_path, corpus_fieldnames, corpus_subset_rows)
    _write_csv(corpus_subset_path, ["_id", "title", "text"], corpus_subset_mteb_rows)

    avg_relevant_docs_per_generation_unit = (
        sum(count * freq for count, freq in relevant_docs_per_generation_unit.items()) / len(generation_rows)
        if generation_rows
        else 0.0
    )

    stats = {
        "pairs_per_language_requested": pairs_per_language,
        "generation_docs_per_language_requested": generation_docs_per_language,
        "allowed_languages": sorted(allowed_language_set),
        "synthetic_target_languages": synthetic_target_language_list,
        "sampled_source_pool_docs_total": len(sampled_source_rows),
        "sampled_source_docs_total": len(sampled_source_rows),
        "selected_generation_source_docs_total": len(selected_source_rows),
        "selected_source_docs_total": len(selected_source_rows),
        "selected_generation_celex_groups_total": len(selected_celexes),
        "final_retrieval_corpus_docs_total": len(corpus_subset_rows),
        "subset_corpus_docs_total": len(corpus_subset_rows),
        "generation_units_total": len(generation_rows),
        "avg_relevant_docs_per_generation_unit": avg_relevant_docs_per_generation_unit,
        "generation_units_by_source_language": dict(sorted(generation_rows_by_source_language.items())),
        "generation_target_languages": dict(sorted(generation_target_language_counts.items())),
        "relevant_docs_per_generation_unit": {
            str(count): freq for count, freq in sorted(relevant_docs_per_generation_unit.items())
        },
        "languages": selected_sources_stats,
        "sampled_sources_csv": str(sampled_sources_path),
        "qa_generation_sources_csv": str(selected_sources_path),
        "subset_corpus_full_csv": str(corpus_subset_full_path),
        "subset_corpus_csv": str(corpus_subset_path),
    }
    selection_stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats

