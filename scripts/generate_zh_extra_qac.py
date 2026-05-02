"""
Generate 40 additional QAC rows that exercise the new Chinese support:

  Phase A — 20 questions whose query language is forced to Chinese.
            Source publications are drawn at random from the corpus,
            EXCLUDING any publication already covered by
            data/google_patents/qac/balanced_100_qac_all_generated_regraded.csv.
            Source passages are NOT required to be Chinese — the prompt is
            instructed to write the query/answer in Chinese regardless of the
            source language. 10 technical + 10 semantic.

  Phase B — 20 questions drawn from publications that already have a Chinese
            row in the corpus (typically created by
            scripts/translate_to_chinese.py). Query language for each row is
            chosen by the existing strategy logic (random_any / random_missing
            / random_existing / all). 10 technical + 10 semantic, split across
            strategies using the same quota allocation as
            balanced_multilingual_qa.

The existing question-generation logic is reused unchanged: this script wires
up `generate_qa_batch`, `grade_faithfulness`, `grade_quality`, and the row
builder from `multi_lingual_qac.qac_generation.multilingual_qa`. Output
schema matches `balanced_multilingual_qa`'s output (best-only + all-generated
sibling file).

Usage:
    python scripts/generate_zh_extra_qac.py \\
        --corpus data/google_patents/multilingual_corpus.csv \\
        --exclude-from data/google_patents/qac/balanced_100_qac_all_generated_regraded.csv \\
        --output data/google_patents/qac/extra_40_zh_qac.csv

Requires OPENAI_API_KEY in environment (or .env file).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_lingual_qac.qac_generation.multilingual_qa import (  # noqa: E402
    ALL_LANGS,
    FAITHFULNESS_FIELDS,
    MODE_SEMANTIC,
    MODE_TECHNICAL,
    SEMANTIC_QUALITY_FIELDS,
    STRATEGY_ALL,
    STRATEGY_NAMES,
    STRATEGY_RANDOM_ANY,
    STRATEGY_RANDOM_EXISTING,
    STRATEGY_RANDOM_MISSING,
    TECHNICAL_QUALITY_FIELDS,
    _build_all_passages_text,
    _build_output_row,
    _get_client,
    _pick_context,
    _serialize_context_languages,
    generate_qa_batch,
    grade_faithfulness,
    grade_quality,
    load_multilingual_corpus,
    pick_target_languages,
)
from multi_lingual_qac.qac_generation.balanced_multilingual_qa import (  # noqa: E402
    _output_fieldnames,
    _select_best_rows,
)


DEFAULT_MODEL = "gpt-5-mini"
PHASE_A_FORCED_LANG = "zh"
PHASE_A_STRATEGY_LABEL = "forced_zh"
# Sentinel integer for strategy column when query language is forced.
PHASE_A_STRATEGY_CODE = 0

STRATEGIES = [
    STRATEGY_RANDOM_ANY,
    STRATEGY_RANDOM_MISSING,
    STRATEGY_RANDOM_EXISTING,
    STRATEGY_ALL,
]


def _generate_for_target_langs(
    pub_num: str,
    rows: List[Dict[str, Any]],
    target_langs: List[str],
    *,
    mode: str,
    model: str,
    client,
) -> List[Dict[str, Any]]:
    """Replica of `_process_document` orchestration that takes pre-resolved
    target_langs instead of computing them from a strategy.

    The underlying generation/grading logic is reused unchanged via the
    exported helpers."""
    all_passages = _build_all_passages_text(rows)
    context_languages = _serialize_context_languages(rows)
    results: List[Dict[str, Any]] = []

    for target_lang in target_langs:
        context_row, context_text = _pick_context(rows, target_lang)
        if not context_text.strip():
            tqdm.write(f"  {pub_num} [{target_lang}]: skipped (empty context)")
            continue

        try:
            qa_pairs = generate_qa_batch(client, all_passages, target_lang, mode, model=model)
        except Exception as exc:
            tqdm.write(f"  {pub_num} [{target_lang}]: generation error: {exc}")
            continue

        if len(qa_pairs) < 3:
            tqdm.write(
                f"  {pub_num} [{target_lang}]: only {len(qa_pairs)} questions generated, skipping"
            )
            continue

        try:
            faith_grades = grade_faithfulness(client, all_passages, qa_pairs, model=model)
        except Exception as exc:
            tqdm.write(f"  {pub_num} [{target_lang}]: faithfulness grading error: {exc}")
            continue

        try:
            qual_grades = grade_quality(client, all_passages, qa_pairs, mode, model=model)
        except Exception as exc:
            tqdm.write(f"  {pub_num} [{target_lang}]: quality grading error: {exc}")
            continue

        doc_rows: List[Dict[str, Any]] = []
        for i in range(3):
            row = _build_output_row(
                qa_pairs[i],
                faith_grades[i],
                qual_grades[i],
                mode,
                corpus_id=context_row.get("id", ""),
                publication_number=pub_num,
                question_language=target_lang,
                context_language=context_languages,
            )
            doc_rows.append(row)
        doc_rows.sort(key=lambda r: r["total_score"], reverse=True)
        results.extend(doc_rows)

        best_score = doc_rows[0]["total_score"]
        cat_field = "question_type" if mode == MODE_TECHNICAL else "framing"
        tqdm.write(
            f"  {pub_num} [{target_lang}]: ok (best={best_score}, "
            f"{cat_field}={doc_rows[0].get(cat_field, '?')})"
        )

    return results


def _normalize_row(row: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    normalized = {f: "" for f in fieldnames}
    normalized.update(row)
    return normalized


def _load_excluded_pubs(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["publication_number"] for row in reader if row.get("publication_number")}


def _build_phase_a_plan(
    pub_nums_pool: List[str],
    *,
    questions_per_mode: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Phase A: each picked publication produces one Chinese question (best-of-3)."""
    rng = random.Random(seed)
    needed = questions_per_mode * 2
    if needed > len(pub_nums_pool):
        raise ValueError(
            f"Phase A needs {needed} unique publications outside the exclusion set, "
            f"but only {len(pub_nums_pool)} are available."
        )
    picked = rng.sample(sorted(pub_nums_pool), needed)

    plan: List[Dict[str, Any]] = []
    for pub_num in picked[:questions_per_mode]:
        plan.append(
            {
                "publication_number": pub_num,
                "mode": MODE_TECHNICAL,
                "strategy": PHASE_A_STRATEGY_CODE,
                "strategy_name": PHASE_A_STRATEGY_LABEL,
                "target_langs": [PHASE_A_FORCED_LANG],
                "expected_question_count": 1,
            }
        )
    for pub_num in picked[questions_per_mode:]:
        plan.append(
            {
                "publication_number": pub_num,
                "mode": MODE_SEMANTIC,
                "strategy": PHASE_A_STRATEGY_CODE,
                "strategy_name": PHASE_A_STRATEGY_LABEL,
                "target_langs": [PHASE_A_FORCED_LANG],
                "expected_question_count": 1,
            }
        )
    return plan


def _allocate_phase_b_quotas(questions_per_mode: int) -> Dict[int, int]:
    """Quota allocator that is aware of the current ``len(ALL_LANGS)`` so the
    STRATEGY_ALL bucket produces exactly the right number of questions.

    A single STRATEGY_ALL document generates one question per language, which
    is now ``len(ALL_LANGS)`` (5 with Chinese added). The remaining
    questions are distributed across strategies 1-3 as evenly as possible.

    For the default ``questions_per_mode=10`` and ``len(ALL_LANGS)=5`` this
    yields ``{RANDOM_ANY: 2, RANDOM_MISSING: 2, RANDOM_EXISTING: 1,
    ALL: 5}`` (1 ALL document, 5 single-language documents -> 6 docs and
    10 questions per mode).
    """
    n_langs = len(ALL_LANGS)
    if questions_per_mode < n_langs + 3:
        raise ValueError(
            f"questions_per_mode must be at least {n_langs + 3} so each strategy "
            f"can be exercised at least once."
        )

    best_counts: Dict[int, int] | None = None
    best_score: tuple[int, int] | None = None

    for all_docs in range(1, questions_per_mode // n_langs + 1):
        all_questions = all_docs * n_langs
        remaining = questions_per_mode - all_questions
        if remaining < 3:
            continue
        base, rem = divmod(remaining, 3)
        counts = {
            STRATEGY_RANDOM_ANY: base + (1 if rem >= 1 else 0),
            STRATEGY_RANDOM_MISSING: base + (1 if rem >= 2 else 0),
            STRATEGY_RANDOM_EXISTING: base,
            STRATEGY_ALL: all_questions,
        }
        spread = max(counts.values()) - min(counts.values())
        target_gap = abs(all_questions - (questions_per_mode / 4))
        score = (spread, int(target_gap * 1000))
        if best_score is None or score < best_score:
            best_score = score
            best_counts = counts

    if best_counts is None:
        raise ValueError(
            f"Unable to allocate balanced phase-B quotas for {questions_per_mode} questions "
            f"per mode with {n_langs} languages."
        )
    return best_counts


def _build_phase_b_plan(
    translated_pubs: List[str],
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    questions_per_mode: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Phase B: pubs that have a zh row, sampled across all strategies/modes."""
    n_langs = len(ALL_LANGS)
    quotas = _allocate_phase_b_quotas(questions_per_mode)

    docs_per_mode = (
        quotas[STRATEGY_RANDOM_ANY]
        + quotas[STRATEGY_RANDOM_MISSING]
        + quotas[STRATEGY_RANDOM_EXISTING]
        + (quotas[STRATEGY_ALL] // n_langs)
    )
    needed = docs_per_mode * 2
    if needed > len(translated_pubs):
        raise ValueError(
            f"Phase B needs {needed} translated publications (with a zh row), "
            f"but only {len(translated_pubs)} are available."
        )

    rng = random.Random(seed + 1)
    picked = rng.sample(sorted(translated_pubs), needed)
    technical_pool = picked[:docs_per_mode]
    semantic_pool = picked[docs_per_mode:]

    plan: List[Dict[str, Any]] = []

    def add_mode(mode: str, candidates: List[str]) -> None:
        cursor = 0
        for strategy in STRATEGIES:
            doc_count = (
                quotas[strategy] if strategy != STRATEGY_ALL else quotas[strategy] // n_langs
            )
            expected = 1 if strategy != STRATEGY_ALL else n_langs
            for _ in range(doc_count):
                pub_num = candidates[cursor]
                cursor += 1
                rows = groups[pub_num]
                available_langs = [r["language"] for r in rows]
                target_langs = pick_target_languages(strategy, available_langs)
                plan.append(
                    {
                        "publication_number": pub_num,
                        "mode": mode,
                        "strategy": strategy,
                        "strategy_name": STRATEGY_NAMES[strategy],
                        "target_langs": target_langs,
                        "expected_question_count": expected,
                    }
                )

    add_mode(MODE_TECHNICAL, technical_pool)
    add_mode(MODE_SEMANTIC, semantic_pool)
    return plan


def run(
    corpus_path: Path,
    output_path: Path,
    *,
    exclude_from: Optional[Path],
    questions_per_phase_per_mode: int,
    model: str,
    seed: int,
    dry_run: bool,
) -> None:
    groups = load_multilingual_corpus(corpus_path)

    excluded = _load_excluded_pubs(exclude_from)

    translated_pubs = sorted(
        pub for pub, rows in groups.items() if any(r["language"] == "zh" for r in rows)
    )

    phase_a_pool = sorted(
        pub
        for pub, rows in groups.items()
        if pub not in excluded
        and any(
            (r.get("context") or r.get("abstract") or r.get("title") or "").strip() for r in rows
        )
    )

    phase_a_plan = _build_phase_a_plan(
        phase_a_pool,
        questions_per_mode=questions_per_phase_per_mode,
        seed=seed,
    )
    phase_b_plan = _build_phase_b_plan(
        translated_pubs,
        groups,
        questions_per_mode=questions_per_phase_per_mode,
        seed=seed,
    )

    print(
        f"Phase A: {len(phase_a_plan)} forced-Chinese questions "
        f"(pool size {len(phase_a_pool)}, excluded {len(excluded)})"
    )
    print(
        f"Phase B: {sum(item['expected_question_count'] for item in phase_b_plan)} "
        f"questions across {len(phase_b_plan)} translated documents "
        f"(translated pool size {len(translated_pubs)})"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_name(output_path.stem + "_manifest" + output_path.suffix)
    manifest_fields = [
        "phase",
        "publication_number",
        "mode",
        "strategy",
        "strategy_name",
        "target_langs",
        "expected_question_count",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for item in phase_a_plan:
            writer.writerow(
                {
                    "phase": "A",
                    "publication_number": item["publication_number"],
                    "mode": item["mode"],
                    "strategy": item["strategy"],
                    "strategy_name": item["strategy_name"],
                    "target_langs": ",".join(item["target_langs"]),
                    "expected_question_count": item["expected_question_count"],
                }
            )
        for item in phase_b_plan:
            writer.writerow(
                {
                    "phase": "B",
                    "publication_number": item["publication_number"],
                    "mode": item["mode"],
                    "strategy": item["strategy"],
                    "strategy_name": item["strategy_name"],
                    "target_langs": ",".join(item["target_langs"]),
                    "expected_question_count": item["expected_question_count"],
                }
            )
    print(f"Wrote manifest -> {manifest_path}")

    if dry_run:
        print("Dry run only: manifest written, generation skipped.")
        return

    fieldnames = _output_fieldnames()
    all_output_path = output_path.with_name(output_path.stem + "_all_generated" + output_path.suffix)

    client = _get_client()

    best_count = 0
    all_count = 0
    full_plan = [("A", item) for item in phase_a_plan] + [("B", item) for item in phase_b_plan]
    progress = tqdm(full_plan, desc="Generate zh extra Q&A", unit="doc")
    with output_path.open("w", encoding="utf-8", newline="") as best_f, all_output_path.open(
        "w", encoding="utf-8", newline=""
    ) as all_f:
        best_writer = csv.DictWriter(best_f, fieldnames=fieldnames)
        all_writer = csv.DictWriter(all_f, fieldnames=fieldnames)
        best_writer.writeheader()
        all_writer.writeheader()
        best_f.flush()
        all_f.flush()

        for phase, item in progress:
            pub_num = item["publication_number"]
            rows = groups[pub_num]
            results = _generate_for_target_langs(
                pub_num,
                rows,
                item["target_langs"],
                mode=item["mode"],
                model=model,
                client=client,
            )

            stamped: List[Dict[str, Any]] = []
            for row in results:
                row["mode"] = item["mode"]
                row["strategy"] = item["strategy"]
                row["strategy_name"] = item["strategy_name"]
                stamped.append(_normalize_row(row, fieldnames))

            best_rows = _select_best_rows(results)
            best_normalized = [_normalize_row(r, fieldnames) for r in best_rows]

            if stamped:
                all_writer.writerows(stamped)
                all_f.flush()
                all_count += len(stamped)
            if best_normalized:
                best_writer.writerows(best_normalized)
                best_f.flush()
                best_count += len(best_normalized)

            if len(best_rows) != item["expected_question_count"]:
                tqdm.write(
                    f"  {pub_num} [{phase}/{item['mode']}/{item['strategy_name']}]: "
                    f"expected {item['expected_question_count']} questions, got {len(best_rows)}"
                )

    print(f"\nWrote {best_count} best-only QAC rows -> {output_path}")
    print(f"Wrote {all_count} all-generated QAC rows -> {all_output_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Generate 40 additional QAC rows: 20 forced-Chinese queries from "
            "non-overlapping publications, plus 20 questions over translated "
            "(Chinese-augmented) publications using the standard strategy mix."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/google_patents/multilingual_corpus.csv"),
        help="Path to multilingual corpus CSV",
    )
    parser.add_argument(
        "--exclude-from",
        type=Path,
        default=Path("data/google_patents/qac/balanced_100_qac_all_generated_regraded.csv"),
        help="CSV whose publication_number column lists pubs that Phase A must skip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/google_patents/qac/extra_40_zh_qac.csv"),
        help="Output CSV path for the best-only sample.",
    )
    parser.add_argument(
        "--questions-per-phase-per-mode",
        type=int,
        default=10,
        help="Questions per (phase, mode) pair (default: 10 -> 40 total).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the manifest and skip API calls.",
    )
    args = parser.parse_args()

    run(
        corpus_path=args.corpus,
        output_path=args.output,
        exclude_from=args.exclude_from,
        questions_per_phase_per_mode=args.questions_per_phase_per_mode,
        model=args.model,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
