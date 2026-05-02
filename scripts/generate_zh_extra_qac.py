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

Question generation runs on OpenAI (default ``gpt-5-mini``). The
faithfulness and quality verifiers run on OpenRouter using Claude Sonnet 4.6
with extended thinking enabled — the same model and configuration used by
``scripts/regrade_with_openrouter.py``. Because the verifier matches the
regraded balanced files, this script APPENDS its results directly into
those existing files instead of writing standalone outputs:

  - all-generated rows (3 candidates per group) ->
        data/google_patents/qac/balanced_100_qac_all_generated_regraded.csv
  - best-only rows (top candidate per group) ->
        data/google_patents/qac/balanced_100_qac_regraded.csv

A small manifest is still written separately so each run is traceable.

Append targets are validated before any API call: the file must exist and
its header must match the standard QAC schema. Override with --all-generated
and --best if you need different targets.

Usage:
    python scripts/generate_zh_extra_qac.py

Requires both ``OPENAI_API_KEY`` (for generation) and ``OPENROUTER_API_KEY``
(for the Claude verifier) in environment / .env file.
"""

from __future__ import annotations

import argparse
import csv
import os
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
    _compute_faith_overall,
    _compute_quality_overall,
    _get_client,
    _load_faithfulness_prompt,
    _load_quality_prompt,
    _parse_json_response,
    _pick_context,
    _serialize_context_languages,
    generate_qa_batch,
    load_multilingual_corpus,
    pick_target_languages,
)
from multi_lingual_qac.qac_generation.balanced_multilingual_qa import (  # noqa: E402
    _output_fieldnames,
    _select_best_rows,
)

from openai import OpenAI  # noqa: E402

DEFAULT_GENERATION_MODEL = "gpt-5-mini"
DEFAULT_VERIFIER_MODEL = "anthropic/claude-sonnet-4.6"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
THINKING_BUDGET_TOKENS = 8000
MAX_TOKENS = 12000
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


# ---------------------------------------------------------------------------
# OpenRouter verifier (Claude Sonnet 4.6 thinking by default).
# Mirrors scripts/regrade_with_openrouter.py so the verifier model used here
# matches the one used to regrade balanced_100_qac_all_generated.csv.
# ---------------------------------------------------------------------------


def _get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENROUTER_API_KEY in environment or .env to run the Claude verifier "
            "(default verifier is anthropic/claude-sonnet-4.6)."
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _call_with_thinking(client: OpenAI, model: str, messages: list) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        extra_body={
            "thinking": {
                "type": "enabled",
                "budget_tokens": THINKING_BUDGET_TOKENS,
            }
        },
    )
    return response.choices[0].message.content or ""


def _verify_faithfulness(
    client: OpenAI,
    model: str,
    all_passages: str,
    qa_pairs: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    prompt = _load_faithfulness_prompt()
    candidates = "\n\n".join(
        f"Candidate {i}:\n  Question: {qa['question']}\n  Answer: {qa['answer']}"
        for i, qa in enumerate(qa_pairs)
    )
    content = _call_with_thinking(
        client,
        model,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{all_passages}\n\n{candidates}"},
        ],
    )
    data = _parse_json_response(content)
    if isinstance(data, dict):
        data = [data]

    results = sorted(data[:3], key=lambda x: x.get("index", 0))
    normalised: List[Dict[str, Any]] = []
    for item in results:
        row: Dict[str, Any] = {
            "grounding": int(item.get("grounding", 1)),
            "precision": int(item.get("precision", 1)),
            "numerical_fidelity": int(item.get("numerical_fidelity", 1)),
            "reason": str(item.get("reason", "")).strip(),
        }
        row["overall"] = _compute_faith_overall(row)
        normalised.append(row)

    while len(normalised) < 3:
        fallback: Dict[str, Any] = {
            "grounding": 1,
            "precision": 1,
            "numerical_fidelity": 1,
            "reason": "missing",
        }
        fallback["overall"] = _compute_faith_overall(fallback)
        normalised.append(fallback)
    return normalised


def _verify_quality(
    client: OpenAI,
    model: str,
    all_passages: str,
    qa_pairs: List[Dict[str, str]],
    mode: str,
) -> List[Dict[str, Any]]:
    prompt = _load_quality_prompt(mode)
    candidates = "\n\n".join(
        f"Candidate {i}:\n  Question: {qa['question']}\n  Answer: {qa['answer']}"
        for i, qa in enumerate(qa_pairs)
    )
    content = _call_with_thinking(
        client,
        model,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{all_passages}\n\n{candidates}"},
        ],
    )
    data = _parse_json_response(content)
    if isinstance(data, dict):
        data = [data]

    results = sorted(data[:3], key=lambda x: x.get("index", 0))
    normalised: List[Dict[str, Any]] = []

    if mode == MODE_TECHNICAL:
        for item in results:
            row: Dict[str, Any] = {
                "search_bar_realism": int(item.get("search_bar_realism", 1)),
                "specificity": int(item.get("specificity", 1)),
                "phrasing_economy": int(item.get("phrasing_economy", 1)),
                "focus": int(item.get("focus", 1)),
                "linguistic_quality": int(item.get("linguistic_quality", 1)),
                "failure_type": str(item.get("failure_type", "none")).strip(),
                "reason": str(item.get("reason", "")).strip(),
            }
            row["overall"] = _compute_quality_overall(row, mode)
            normalised.append(row)
        default: Dict[str, Any] = {
            "search_bar_realism": 1,
            "specificity": 1,
            "phrasing_economy": 1,
            "focus": 1,
            "linguistic_quality": 1,
            "failure_type": "missing",
            "reason": "missing",
        }
    else:
        for item in results:
            row = {
                "search_realism": int(item.get("search_realism", 1)),
                "lexical_distance": int(item.get("lexical_distance", 1)),
                "conceptual_framing": int(item.get("conceptual_framing", 1)),
                "retrievability": int(item.get("retrievability", 1)),
                "linguistic_quality": int(item.get("linguistic_quality", 1)),
                "failure_type": str(item.get("failure_type", "none")).strip(),
                "reason": str(item.get("reason", "")).strip(),
            }
            row["overall"] = _compute_quality_overall(row, mode)
            normalised.append(row)
        default = {
            "search_realism": 1,
            "lexical_distance": 1,
            "conceptual_framing": 1,
            "retrievability": 1,
            "linguistic_quality": 1,
            "failure_type": "missing",
            "reason": "missing",
        }

    while len(normalised) < 3:
        fallback = dict(default)
        fallback["overall"] = _compute_quality_overall(fallback, mode)
        normalised.append(fallback)
    return normalised


def _generate_for_target_langs(
    pub_num: str,
    rows: List[Dict[str, Any]],
    target_langs: List[str],
    *,
    mode: str,
    generation_client: OpenAI,
    generation_model: str,
    verifier_client: OpenAI,
    verifier_model: str,
) -> List[Dict[str, Any]]:
    """Generation orchestration with split clients/models: question generation
    runs against ``generation_client`` (default OpenAI gpt-5-mini), while
    faithfulness and quality verification run against ``verifier_client``
    (default OpenRouter Claude Sonnet 4.6 with thinking)."""
    all_passages = _build_all_passages_text(rows)
    context_languages = _serialize_context_languages(rows)
    results: List[Dict[str, Any]] = []

    for target_lang in target_langs:
        context_row, context_text = _pick_context(rows, target_lang)
        if not context_text.strip():
            tqdm.write(f"  {pub_num} [{target_lang}]: skipped (empty context)")
            continue

        try:
            qa_pairs = generate_qa_batch(
                generation_client, all_passages, target_lang, mode, model=generation_model,
            )
        except Exception as exc:
            tqdm.write(f"  {pub_num} [{target_lang}]: generation error: {exc}")
            continue

        if len(qa_pairs) < 3:
            tqdm.write(
                f"  {pub_num} [{target_lang}]: only {len(qa_pairs)} questions generated, skipping"
            )
            continue

        try:
            faith_grades = _verify_faithfulness(
                verifier_client, verifier_model, all_passages, qa_pairs,
            )
        except Exception as exc:
            tqdm.write(f"  {pub_num} [{target_lang}]: faithfulness grading error: {exc}")
            continue

        try:
            qual_grades = _verify_quality(
                verifier_client, verifier_model, all_passages, qa_pairs, mode,
            )
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


def _check_appendable(path: Path, expected_fieldnames: List[str]) -> None:
    """Validate that *path* exists, is non-empty, and its header matches the
    schema we are about to append. Refuses to silently create the file or
    write into a file with a different header."""
    if not path.exists():
        raise FileNotFoundError(
            f"Append target {path} does not exist. Create it first (e.g. by "
            f"running the balanced pipeline + regrade) or pass a different path."
        )
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header != expected_fieldnames:
        raise ValueError(
            f"Header mismatch in {path}.\n  expected: {expected_fieldnames}\n  found:    {header}"
        )


def run(
    corpus_path: Path,
    *,
    all_generated_path: Path,
    best_path: Path,
    manifest_path: Path,
    exclude_from: Optional[Path],
    questions_per_phase_per_mode: int,
    generation_model: str,
    verifier_model: str,
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

    all_generated_path = Path(all_generated_path)
    best_path = Path(best_path)
    manifest_path = Path(manifest_path)

    fieldnames = _output_fieldnames()
    if not dry_run:
        _check_appendable(all_generated_path, fieldnames)
        _check_appendable(best_path, fieldnames)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
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

    generation_client = _get_client()
    verifier_client = _get_openrouter_client()
    print(f"Generator:        {generation_model} (OpenAI)")
    print(f"Verifier:         {verifier_model} (OpenRouter, thinking)")
    print(f"Append target:    {all_generated_path}")
    print(f"Best-only target: {best_path}")

    best_count = 0
    all_count = 0
    full_plan = [("A", item) for item in phase_a_plan] + [("B", item) for item in phase_b_plan]
    progress = tqdm(full_plan, desc="Generate zh extra Q&A", unit="doc")
    # Open for append; assume header is already present (validated above).
    with best_path.open("a", encoding="utf-8", newline="") as best_f, all_generated_path.open(
        "a", encoding="utf-8", newline=""
    ) as all_f:
        best_writer = csv.DictWriter(best_f, fieldnames=fieldnames)
        all_writer = csv.DictWriter(all_f, fieldnames=fieldnames)

        for phase, item in progress:
            pub_num = item["publication_number"]
            rows = groups[pub_num]
            results = _generate_for_target_langs(
                pub_num,
                rows,
                item["target_langs"],
                mode=item["mode"],
                generation_client=generation_client,
                generation_model=generation_model,
                verifier_client=verifier_client,
                verifier_model=verifier_model,
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

    print(f"\nAppended {all_count} all-generated rows -> {all_generated_path}")
    print(f"Appended {best_count} best-only rows    -> {best_path}")


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
        "--all-generated",
        type=Path,
        default=Path("data/google_patents/qac/balanced_100_qac_all_generated_regraded.csv"),
        help=(
            "Existing CSV that the new all-generated rows (3 candidates per group) "
            "are APPENDED to. Must already exist with the standard QAC header."
        ),
    )
    parser.add_argument(
        "--best",
        type=Path,
        default=Path("data/google_patents/qac/balanced_100_qac_regraded.csv"),
        help=(
            "Existing CSV that the new best-only rows (top candidate per group) "
            "are APPENDED to. Must already exist with the standard QAC header."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/google_patents/qac/extra_40_zh_qac_manifest.csv"),
        help="Path for the run plan / manifest (written fresh each run).",
    )
    parser.add_argument(
        "--questions-per-phase-per-mode",
        type=int,
        default=10,
        help="Questions per (phase, mode) pair (default: 10 -> 40 total).",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default=DEFAULT_GENERATION_MODEL,
        help=f"OpenAI model used for question generation (default: {DEFAULT_GENERATION_MODEL}).",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default=DEFAULT_VERIFIER_MODEL,
        help=(
            "OpenRouter model used for the faithfulness and quality verifiers, "
            f"called with extended thinking (default: {DEFAULT_VERIFIER_MODEL}). "
            "Requires OPENROUTER_API_KEY."
        ),
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
        all_generated_path=args.all_generated,
        best_path=args.best,
        manifest_path=args.manifest,
        exclude_from=args.exclude_from,
        questions_per_phase_per_mode=args.questions_per_phase_per_mode,
        generation_model=args.generation_model,
        verifier_model=args.verifier_model,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
