"""
Generate 400 additional QAC rows (200 technical + 200 semantic) from documents
that are NOT yet covered by the existing benchmark, and APPEND them to the two
final regraded files:

  - all-generated rows (3 candidates per group) ->
        data/google_patents/qac/qac_chempatents.csv
  - best-only rows (top candidate per group) ->
        data/google_patents/qac/qac_chempatents_best.csv

This is a generalization of ``scripts/generate_zh_extra_qac.py``. It reuses the
same generation + grading machinery (OpenAI ``gpt-5-mini`` for question
generation; OpenRouter Claude Sonnet 4.6 with extended thinking for the
faithfulness/quality verifiers), so the new rows are directly comparable to the
existing regraded rows and can be appended in place.

Selection rules
---------------
Per mode (technical, semantic) we generate exactly ``--questions-per-mode``
questions (default 200), distributed EQUALLY across the four strategies
(random_any / random_missing / random_existing / all). With 5 languages this is
50 questions per strategy per mode:

    random_any      -> 50 docs (1 question each)
    random_missing  -> 50 docs (1 question each)
    random_existing -> 50 docs (1 question each)
    all             -> 10 docs (5 questions each = 50)
    -------------------------------------------------
    160 docs / 200 questions per mode

Documents are drawn ONLY from publications in the corpus that are not already
covered by ``--exclude-from`` (the all-candidates file, which lists every
covered publication). Each mode's 160 documents are stratified:

    60 docs that have at least one Chinese (zh) row in the corpus
    30 docs that have at least one Spanish (es) row in the corpus
    70 docs drawn at random from the remaining uncovered pool

"zh/es-available" means the publication has a version in that language; it does
NOT mean the publication exists only in that language. The query LANGUAGE of
each generated question is still decided by the strategy via
``pick_target_languages`` — the zh/es enforcement only governs which source
documents are chosen.

All 320 documents (160 per mode x 2 modes) are distinct, so no publication is
re-used across modes or strategies.

Usage:
    python scripts/generate_extra_qac.py --dry-run   # plan/manifest only, no API
    python scripts/generate_extra_qac.py             # full generation + append

Requires both ``OPENAI_API_KEY`` (generation) and ``OPENROUTER_API_KEY`` (the
Claude verifier) in environment / .env file.
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

# Make the repo root, the package (src/), and this scripts/ directory importable
# so we can reuse the existing generation/grading helpers without duplicating
# them. The repo root is needed because multi_lingual_qac/__init__.py uses
# absolute ``from src.multi_lingual_qac ...`` imports.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_SCRIPTS_DIR))

from multi_lingual_qac.qac_generation.multilingual_qa import (  # noqa: E402
    ALL_LANGS,
    MODE_SEMANTIC,
    MODE_TECHNICAL,
    STRATEGY_ALL,
    STRATEGY_NAMES,
    STRATEGY_RANDOM_ANY,
    STRATEGY_RANDOM_EXISTING,
    STRATEGY_RANDOM_MISSING,
    _get_client,
    load_multilingual_corpus,
    pick_target_languages,
)
from multi_lingual_qac.qac_generation.balanced_multilingual_qa import (  # noqa: E402
    _output_fieldnames,
    _select_best_rows,
)

# Reuse the verifier + generation orchestration from the Chinese-extra script.
# Importing the module is side-effect free (its load_dotenv() runs inside main()).
from generate_zh_extra_qac import (  # noqa: E402
    DEFAULT_GENERATION_MODEL,
    DEFAULT_VERIFIER_MODEL,
    _allocate_phase_b_quotas,
    _check_appendable,
    _generate_for_target_langs,
    _get_openrouter_client,
    _load_excluded_pubs,
    _normalize_row,
)

STRATEGIES = [
    STRATEGY_RANDOM_ANY,
    STRATEGY_RANDOM_MISSING,
    STRATEGY_RANDOM_EXISTING,
    STRATEGY_ALL,
]

MODES = [MODE_TECHNICAL, MODE_SEMANTIC]


def _has_content(rows: List[Dict[str, Any]]) -> bool:
    """True if at least one row has non-empty generatable text."""
    return any(
        (r.get("context") or r.get("abstract") or r.get("title") or "").strip() for r in rows
    )


def _doc_languages(rows: List[Dict[str, Any]]) -> set[str]:
    return {r.get("language", "") for r in rows}


def build_plan(
    groups: Dict[str, List[Dict[str, Any]]],
    excluded: set[str],
    *,
    questions_per_mode: int,
    zh_per_mode: int,
    es_per_mode: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Build the per-document generation plan.

    Returns a flat list of plan items, each:
        {publication_number, mode, strategy, strategy_name, doc_class,
         target_langs, expected_question_count}
    """
    n_langs = len(ALL_LANGS)
    quotas = _allocate_phase_b_quotas(questions_per_mode)
    docs_per_mode = (
        quotas[STRATEGY_RANDOM_ANY]
        + quotas[STRATEGY_RANDOM_MISSING]
        + quotas[STRATEGY_RANDOM_EXISTING]
        + (quotas[STRATEGY_ALL] // n_langs)
    )
    random_per_mode = docs_per_mode - zh_per_mode - es_per_mode
    if random_per_mode < 0:
        raise ValueError(
            f"zh_per_mode ({zh_per_mode}) + es_per_mode ({es_per_mode}) exceeds "
            f"docs_per_mode ({docs_per_mode}); reduce the enforced quotas."
        )

    # Eligible (uncovered, has content) publications, plus zh/es sub-pools.
    uncovered = sorted(
        pub
        for pub, rows in groups.items()
        if pub not in excluded and _has_content(rows)
    )
    zh_pool = [p for p in uncovered if "zh" in _doc_languages(groups[p])]
    es_pool = [p for p in uncovered if "es" in _doc_languages(groups[p])]

    # Seed both the dedicated sampler and the module-level RNG used by
    # pick_target_languages, so the whole plan (docs + target languages) is
    # reproducible for a given seed.
    rng = random.Random(seed)
    random.seed(seed)

    need_zh = zh_per_mode * len(MODES)
    need_es = es_per_mode * len(MODES)
    need_rand = random_per_mode * len(MODES)

    if need_zh > len(zh_pool):
        raise ValueError(
            f"Need {need_zh} distinct zh-available uncovered docs but only "
            f"{len(zh_pool)} are available."
        )
    zh_pick = rng.sample(zh_pool, need_zh)
    used: set[str] = set(zh_pick)

    es_avail = [p for p in es_pool if p not in used]
    if need_es > len(es_avail):
        raise ValueError(
            f"Need {need_es} distinct es-available uncovered docs (disjoint from "
            f"the zh picks) but only {len(es_avail)} are available."
        )
    es_pick = rng.sample(es_avail, need_es)
    used |= set(es_pick)

    rand_avail = [p for p in uncovered if p not in used]
    if need_rand > len(rand_avail):
        raise ValueError(
            f"Need {need_rand} distinct random uncovered docs (disjoint from the "
            f"enforced picks) but only {len(rand_avail)} are available."
        )
    rand_pick = rng.sample(rand_avail, need_rand)

    doc_class: Dict[str, str] = {}
    for p in zh_pick:
        doc_class[p] = "zh"
    for p in es_pick:
        doc_class[p] = "es"
    for p in rand_pick:
        doc_class[p] = "random"

    # Split each sub-pool evenly across the modes, then build each mode's doc list.
    mode_docs: Dict[str, List[str]] = {}
    for i, mode in enumerate(MODES):
        zh_slice = zh_pick[i * zh_per_mode : (i + 1) * zh_per_mode]
        es_slice = es_pick[i * es_per_mode : (i + 1) * es_per_mode]
        rand_slice = rand_pick[i * random_per_mode : (i + 1) * random_per_mode]
        docs = zh_slice + es_slice + rand_slice
        rng.shuffle(docs)
        mode_docs[mode] = docs

    plan: List[Dict[str, Any]] = []
    for mode in MODES:
        docs = mode_docs[mode]
        cursor = 0
        for strategy in STRATEGIES:
            doc_count = (
                quotas[strategy] if strategy != STRATEGY_ALL else quotas[STRATEGY_ALL] // n_langs
            )
            expected = 1 if strategy != STRATEGY_ALL else n_langs
            for _ in range(doc_count):
                pub = docs[cursor]
                cursor += 1
                available_langs = [r["language"] for r in groups[pub]]
                target_langs = pick_target_languages(strategy, available_langs)
                plan.append(
                    {
                        "publication_number": pub,
                        "mode": mode,
                        "strategy": strategy,
                        "strategy_name": STRATEGY_NAMES[strategy],
                        "doc_class": doc_class[pub],
                        "target_langs": target_langs,
                        "expected_question_count": expected,
                    }
                )
    return plan


def _write_manifest(manifest_path: Path, plan: List[Dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "publication_number",
        "mode",
        "strategy",
        "strategy_name",
        "doc_class",
        "target_langs",
        "expected_question_count",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in plan:
            writer.writerow(
                {
                    "publication_number": item["publication_number"],
                    "mode": item["mode"],
                    "strategy": item["strategy"],
                    "strategy_name": item["strategy_name"],
                    "doc_class": item["doc_class"],
                    "target_langs": ",".join(item["target_langs"]),
                    "expected_question_count": item["expected_question_count"],
                }
            )


def run(
    corpus_path: Path,
    *,
    all_generated_path: Path,
    best_path: Path,
    manifest_path: Path,
    exclude_from: Optional[Path],
    questions_per_mode: int,
    zh_per_mode: int,
    es_per_mode: int,
    generation_model: str,
    verifier_model: str,
    seed: int,
    dry_run: bool,
) -> None:
    groups = load_multilingual_corpus(corpus_path)
    excluded = _load_excluded_pubs(exclude_from)

    plan = build_plan(
        groups,
        excluded,
        questions_per_mode=questions_per_mode,
        zh_per_mode=zh_per_mode,
        es_per_mode=es_per_mode,
        seed=seed,
    )

    total_questions = sum(item["expected_question_count"] for item in plan)
    print(
        f"Planned {len(plan)} documents -> {total_questions} questions "
        f"({questions_per_mode} per mode x {len(MODES)} modes), excluding "
        f"{len(excluded)} covered publications."
    )
    for mode in MODES:
        items = [it for it in plan if it["mode"] == mode]
        by_class: Dict[str, int] = {}
        by_strategy: Dict[str, int] = {}
        for it in items:
            by_class[it["doc_class"]] = by_class.get(it["doc_class"], 0) + 1
            by_strategy[it["strategy_name"]] = by_strategy.get(it["strategy_name"], 0) + 1
        print(f"  [{mode}] {len(items)} docs  class={by_class}  strategy={by_strategy}")

    all_generated_path = Path(all_generated_path)
    best_path = Path(best_path)
    manifest_path = Path(manifest_path)

    fieldnames = _output_fieldnames()
    if not dry_run:
        _check_appendable(all_generated_path, fieldnames)
        _check_appendable(best_path, fieldnames)

    _write_manifest(manifest_path, plan)
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
    progress = tqdm(plan, desc="Generate extra Q&A", unit="doc")
    # Open for append; header is assumed present (validated above).
    with best_path.open("a", encoding="utf-8", newline="") as best_f, all_generated_path.open(
        "a", encoding="utf-8", newline=""
    ) as all_f:
        best_writer = csv.DictWriter(best_f, fieldnames=fieldnames)
        all_writer = csv.DictWriter(all_f, fieldnames=fieldnames)

        for item in progress:
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
                    f"  {pub_num} [{item['mode']}/{item['strategy_name']}]: "
                    f"expected {item['expected_question_count']} questions, got {len(best_rows)}"
                )

    print(f"\nAppended {all_count} all-generated rows -> {all_generated_path}")
    print(f"Appended {best_count} best-only rows    -> {best_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Generate 400 extra QAC rows (200 technical + 200 semantic) from "
            "uncovered publications, distributed equally across the four "
            "strategies, with a per-mode quota of 60 zh-available + 30 "
            "es-available documents. Appends to the two final regraded files."
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
        default=Path("data/google_patents/qac/qac_chempatents.csv"),
        help=(
            "CSV whose publication_number column lists pubs already covered; these "
            "are excluded from selection (default: the all-candidates file)."
        ),
    )
    parser.add_argument(
        "--all-generated",
        type=Path,
        default=Path("data/google_patents/qac/qac_chempatents.csv"),
        help=(
            "Existing CSV that the new all-generated rows (3 candidates per group) "
            "are APPENDED to. Must already exist with the standard QAC header."
        ),
    )
    parser.add_argument(
        "--best",
        type=Path,
        default=Path("data/google_patents/qac/qac_chempatents_best.csv"),
        help=(
            "Existing CSV that the new best-only rows (top candidate per group) "
            "are APPENDED to. Must already exist with the standard QAC header."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/google_patents/qac/extra_400_qac_manifest.csv"),
        help="Path for the run plan / manifest (written fresh each run).",
    )
    parser.add_argument(
        "--questions-per-mode",
        type=int,
        default=200,
        help="Questions per mode (default: 200 -> 400 total across both modes).",
    )
    parser.add_argument(
        "--zh-per-mode",
        type=int,
        default=60,
        help="Enforced number of zh-available source docs per mode (default: 60).",
    )
    parser.add_argument(
        "--es-per-mode",
        type=int,
        default=30,
        help="Enforced number of es-available source docs per mode (default: 30).",
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
        questions_per_mode=args.questions_per_mode,
        zh_per_mode=args.zh_per_mode,
        es_per_mode=args.es_per_mode,
        generation_model=args.generation_model,
        verifier_model=args.verifier_model,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
