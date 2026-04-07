from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict


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
    pair_path: Path,
    output_dir: Path,
    pairs_per_language: int,
    generation_docs_per_language: int,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Prepare a JRC-Acquis QA subset.

    Workflow:
    - sample directional cross-language pairs per source language
    - select unique source documents per language
    - choose one sampled pair per selected source document
    - build a subset corpus containing all retained sampled translations
    - write QA generation rows using one chosen translated/target side
    - map each generated query to all sampled retained translations for the same CELEX
    - write helper CSVs into `output_dir`
    """
    corpus_full_path = Path(corpus_full_path)
    qa_candidates_path = Path(qa_candidates_path)
    pair_path = Path(pair_path)
    output_dir = Path(output_dir)

    if not corpus_full_path.is_file():
        raise ValueError(f"Missing JRC corpus full file: {corpus_full_path}")
    if not qa_candidates_path.is_file():
        raise ValueError(f"Missing JRC QA candidates file: {qa_candidates_path}")
    if not pair_path.is_file():
        raise ValueError(f"Missing JRC pair file: {pair_path}")
    if pairs_per_language <= 0:
        raise ValueError("pairs_per_language must be > 0")
    if generation_docs_per_language <= 0:
        raise ValueError("generation_docs_per_language must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    qa_candidate_ids: set[str] = set()
    qa_candidate_fieldnames: list[str] = []
    for row in _iter_csv_rows(qa_candidates_path):
        qa_candidate_ids.add(row["id"])
        if not qa_candidate_fieldnames:
            qa_candidate_fieldnames = list(row.keys())

    reservoir_by_lang: dict[str, list[dict[str, str]]] = defaultdict(list)
    available_pairs_by_lang: Counter[str] = Counter()
    for row in _iter_csv_rows(pair_path):
        directions: list[dict[str, str]] = []
        if row["corpus_id_a"] in qa_candidate_ids:
            directions.append(
                {
                    "pair_id": row["pair_id"],
                    "celex": row["celex"],
                    "source_language": row["lang_a"],
                    "source_corpus_id": row["corpus_id_a"],
                    "target_language": row["lang_b"],
                    "target_corpus_id": row["corpus_id_b"],
                }
            )
        if row["corpus_id_b"] in qa_candidate_ids:
            directions.append(
                {
                    "pair_id": row["pair_id"],
                    "celex": row["celex"],
                    "source_language": row["lang_b"],
                    "source_corpus_id": row["corpus_id_b"],
                    "target_language": row["lang_a"],
                    "target_corpus_id": row["corpus_id_a"],
                }
            )
        for direction in directions:
            lang = direction["source_language"]
            available_pairs_by_lang[lang] += 1
            reservoir = reservoir_by_lang[lang]
            seen = available_pairs_by_lang[lang]
            if len(reservoir) < pairs_per_language:
                reservoir.append(direction)
            else:
                replace_idx = rng.randint(1, seen)
                if replace_idx <= pairs_per_language:
                    reservoir[replace_idx - 1] = direction

    sampled_pairs: list[dict[str, str]] = []
    selected_source_ids: set[str] = set()
    selected_source_rows: list[dict[str, str]] = []
    selected_sources_stats: dict[str, dict[str, int]] = {}

    for lang in sorted(reservoir_by_lang):
        sampled_lang_pairs = list(reservoir_by_lang[lang])
        if not sampled_lang_pairs:
            continue
        sampled_pairs.extend(sampled_lang_pairs)

        unique_source_ids = sorted({row["source_corpus_id"] for row in sampled_lang_pairs})
        if len(unique_source_ids) > generation_docs_per_language:
            chosen_ids = set(rng.sample(unique_source_ids, generation_docs_per_language))
        else:
            chosen_ids = set(unique_source_ids)

        selected_source_ids.update(chosen_ids)
        selected_sources_stats[lang] = {
            "available_pairs": int(available_pairs_by_lang[lang]),
            "sampled_pairs": len(sampled_lang_pairs),
            "available_source_docs": len(unique_source_ids),
            "selected_source_docs": len(chosen_ids),
        }

    sampled_pairs.sort(
        key=lambda row: (
            row["source_language"],
            row["source_corpus_id"],
            row["target_language"],
            row["target_corpus_id"],
        )
    )

    selected_source_rows = []
    selected_source_rows_by_id: dict[str, dict[str, str]] = {}
    for row in _iter_csv_rows(qa_candidates_path):
        if row["id"] in selected_source_ids:
            selected_source_rows.append(row)
            selected_source_rows_by_id[row["id"]] = row
    selected_source_rows.sort(key=lambda row: (row.get("language", ""), row.get("id", "")))

    generation_pairs: list[dict[str, str]] = []
    generation_pairs_stats: dict[str, int] = {}
    for lang in sorted(reservoir_by_lang):
        sampled_lang_pairs = list(reservoir_by_lang[lang])
        if not sampled_lang_pairs:
            continue
        chosen_source_ids = {
            row["id"] for row in selected_source_rows if row.get("language", "") == lang
        }
        chosen_pairs_for_lang: list[dict[str, str]] = []
        for source_id in sorted(chosen_source_ids):
            candidates = [pair for pair in sampled_lang_pairs if pair["source_corpus_id"] == source_id]
            if not candidates:
                continue
            chosen_pairs_for_lang.append(rng.choice(candidates))
        generation_pairs.extend(chosen_pairs_for_lang)
        generation_pairs_stats[lang] = len(chosen_pairs_for_lang)

    generation_pairs.sort(
        key=lambda row: (
            row["source_language"],
            row["source_corpus_id"],
            row["target_language"],
            row["target_corpus_id"],
        )
    )

    corpus_subset_ids = {
        pair["source_corpus_id"] for pair in sampled_pairs
    } | {
        pair["target_corpus_id"] for pair in sampled_pairs
    }

    corpus_fieldnames: list[str] = []
    corpus_subset_rows: list[dict[str, str]] = []
    docs_by_id: dict[str, dict[str, str]] = {}
    for row in _iter_csv_rows(corpus_full_path):
        if not corpus_fieldnames:
            corpus_fieldnames = list(row.keys())
        row_id = row.get("id", "")
        if row_id in corpus_subset_ids:
            corpus_subset_rows.append(row)
            docs_by_id[row_id] = row

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
    for pair in generation_pairs:
        target_row = docs_by_id.get(pair["target_corpus_id"])
        source_row = selected_source_rows_by_id.get(pair["source_corpus_id"]) or docs_by_id.get(pair["source_corpus_id"])
        if not target_row or not source_row:
            continue
        linked_docs = subset_docs_by_celex.get(pair["celex"], [])
        linked_corpus_ids = [row["id"] for row in linked_docs if row.get("id", "")]
        linked_languages = [row.get("language", "") for row in linked_docs if row.get("language", "")]
        if not linked_corpus_ids:
            linked_corpus_ids = [pair["source_corpus_id"], pair["target_corpus_id"]]
            linked_languages = [pair["source_language"], pair["target_language"]]
        generation_row = dict(target_row)
        generation_row["pair_id"] = pair["pair_id"]
        generation_row["celex"] = pair["celex"]
        generation_row["source_language"] = pair["source_language"]
        generation_row["source_corpus_id"] = pair["source_corpus_id"]
        generation_row["target_language"] = pair["target_language"]
        generation_row["target_corpus_id"] = pair["target_corpus_id"]
        generation_row["query_corpus_id"] = pair["target_corpus_id"]
        generation_row["query_id_hint"] = pair["pair_id"]
        generation_row["linked_corpus_ids_json"] = json.dumps(linked_corpus_ids, ensure_ascii=False)
        generation_row["linked_languages_json"] = json.dumps(linked_languages, ensure_ascii=False)
        generation_row["linked_corpus_count"] = str(len(linked_corpus_ids))
        generation_rows.append(generation_row)
        generation_target_language_counts[pair["target_language"]] += 1
        relevant_docs_per_generation_unit[len(linked_corpus_ids)] += 1

    generation_rows.sort(
        key=lambda row: (
            row.get("source_language", ""),
            row.get("source_corpus_id", ""),
            row.get("target_language", ""),
            row.get("target_corpus_id", ""),
        )
    )

    sampled_pairs_path = output_dir / "sampled_pairs.csv"
    selected_sources_path = output_dir / "qa_generation_sources.csv"
    corpus_subset_full_path = output_dir / "corpus_full.csv"
    corpus_subset_path = output_dir / "corpus.csv"
    selection_stats_path = output_dir / "qa_selection_stats.json"

    _write_csv(
        sampled_pairs_path,
        [
            "pair_id",
            "celex",
            "source_language",
            "source_corpus_id",
            "target_language",
            "target_corpus_id",
        ],
        sampled_pairs,
    )
    generation_row_fieldnames = (
        list(generation_rows[0].keys())
        if generation_rows
        else qa_candidate_fieldnames + [
            "pair_id",
            "source_language",
            "source_corpus_id",
            "target_language",
            "target_corpus_id",
            "query_corpus_id",
            "query_id_hint",
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

    if corpus_subset_rows:
        _write_csv(corpus_subset_full_path, list(corpus_subset_rows[0].keys()), corpus_subset_rows)
    else:
        _write_csv(corpus_subset_full_path, corpus_fieldnames, [])
    _write_csv(corpus_subset_path, ["_id", "title", "text"], corpus_subset_mteb_rows)

    stats = {
        "pairs_per_language_requested": pairs_per_language,
        "generation_docs_per_language_requested": generation_docs_per_language,
        "sampled_pairs_total": len(sampled_pairs),
        "subset_corpus_docs_total": len(corpus_subset_rows),
        "selected_source_docs_total": len(selected_source_rows),
        "generation_units_total": len(generation_rows),
        "generation_pairs_by_source_language": dict(sorted(generation_pairs_stats.items())),
        "generation_target_languages": dict(sorted(generation_target_language_counts.items())),
        "relevant_docs_per_generation_unit": {
            str(count): freq for count, freq in sorted(relevant_docs_per_generation_unit.items())
        },
        "languages": selected_sources_stats,
        "sampled_pairs_csv": str(sampled_pairs_path),
        "qa_generation_sources_csv": str(selected_sources_path),
        "subset_corpus_full_csv": str(corpus_subset_full_path),
        "subset_corpus_csv": str(corpus_subset_path),
    }
    selection_stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats

