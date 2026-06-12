"""
Generate a balanced multilingual QAC sample for the EPO corpus
(``data/EPO/multilingual_corpus.csv``), the same way the Google Patents
benchmark was built, and write the rows to ``data/EPO/qac/``.

EPO documents exist in three languages only (en, de, fr), so generation is
restricted to that language set. As with Google Patents, the QUERY LANGUAGE of
each question is decided by the strategy (random_any / random_missing /
random_existing / all), not forced.

Default target: 200 questions = 100 technical + 100 semantic, distributed as
evenly as possible across the four strategies. With three languages the
STRATEGY_ALL bucket emits 3 questions per document, so a perfectly equal 25/25/
25/25 split is impossible; the allocator picks the closest balanced split
(26/25/25/24 questions per mode). Source documents are drawn at random from the
uncovered pool (no language enforcement — EPO has no zh/es).

Generation runs on OpenAI (default ``gpt-5-mini``); the faithfulness and quality
verifiers run on OpenRouter Claude Sonnet 4.6 with extended thinking — the same
models used for the Google Patents benchmark, so the rows are directly
comparable. Output:

  - all-generated rows (3 candidates per group) -> data/EPO/qac/qac_epo.csv
  - best-only rows (top candidate per group)    -> data/EPO/qac/qac_epo_best.csv

Both files are created with the standard QAC header on first run and appended to
(with exclusion of already-covered publications) on subsequent runs. Publish to
Hugging Face afterwards with scripts/push_qac_chempatents_hf.py (pass --repo,
--qac data/EPO/qac/qac_epo_best.csv, --corpus data/EPO/multilingual_corpus.csv).

Usage:
    python scripts/generate_epo_qac.py --dry-run    # plan/manifest only, no API
    python scripts/generate_epo_qac.py --workers 8  # full generation

Requires OPENAI_API_KEY (generation) and OPENROUTER_API_KEY (verifier).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_SCRIPTS_DIR))

from multi_lingual_qac.qac_generation.multilingual_qa import (  # noqa: E402
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

from generate_zh_extra_qac import (  # noqa: E402
    DEFAULT_GENERATION_MODEL,
    DEFAULT_RETRIES,
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

DEFAULT_EPO_LANGS = ["en", "de", "fr"]


def _has_content(rows: List[Dict[str, Any]]) -> bool:
    return any(
        (r.get("context") or r.get("abstract") or r.get("title") or "").strip() for r in rows
    )


def build_plan(
    groups: Dict[str, List[Dict[str, Any]]],
    excluded: set[str],
    *,
    questions_per_mode: int,
    langs: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    """Build the per-document plan: ``questions_per_mode`` questions per mode,
    distributed as evenly as possible across the four strategies, over random
    uncovered documents (no language enforcement)."""
    n_langs = len(langs)
    quotas = _allocate_phase_b_quotas(questions_per_mode, n_langs=n_langs)

    def doc_target(strategy: int) -> int:
        return quotas[strategy] if strategy != STRATEGY_ALL else quotas[STRATEGY_ALL] // n_langs

    docs_per_mode = sum(doc_target(s) for s in STRATEGIES)

    uncovered = sorted(
        pub for pub, rows in groups.items() if pub not in excluded and _has_content(rows)
    )
    need = docs_per_mode * len(MODES)
    if need > len(uncovered):
        raise ValueError(
            f"Need {need} distinct uncovered documents but only {len(uncovered)} are available."
        )

    rng = random.Random(seed)
    random.seed(seed)
    picked = rng.sample(uncovered, need)

    plan: List[Dict[str, Any]] = []
    for i, mode in enumerate(MODES):
        docs = picked[i * docs_per_mode : (i + 1) * docs_per_mode]
        rng.shuffle(docs)
        cursor = 0
        for strategy in STRATEGIES:
            expected = 1 if strategy != STRATEGY_ALL else n_langs
            for _ in range(doc_target(strategy)):
                pub = docs[cursor]
                cursor += 1
                available_langs = [r["language"] for r in groups[pub]]
                target_langs = pick_target_languages(strategy, available_langs, langs=langs)
                plan.append(
                    {
                        "publication_number": pub,
                        "mode": mode,
                        "strategy": strategy,
                        "strategy_name": STRATEGY_NAMES[strategy],
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
                    "target_langs": ",".join(item["target_langs"]),
                    "expected_question_count": item["expected_question_count"],
                }
            )


def _ensure_output_file(path: Path, fieldnames: List[str]) -> None:
    """Create *path* with the QAC header if missing; otherwise validate its
    header so appends stay schema-consistent."""
    path = Path(path)
    if path.exists():
        _check_appendable(path, fieldnames)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def run(
    corpus_path: Path,
    *,
    all_generated_path: Path,
    best_path: Path,
    manifest_path: Path,
    exclude_from: Optional[Path],
    questions_per_mode: int,
    langs: List[str],
    generation_model: str,
    verifier_model: str,
    seed: int,
    workers: int,
    retries: int,
    dry_run: bool,
) -> None:
    groups = load_multilingual_corpus(corpus_path)

    excluded: set[str] = set()
    if exclude_from is not None and Path(exclude_from).exists():
        excluded = _load_excluded_pubs(Path(exclude_from))

    plan = build_plan(
        groups,
        excluded,
        questions_per_mode=questions_per_mode,
        langs=langs,
        seed=seed,
    )

    total_questions = sum(item["expected_question_count"] for item in plan)
    print(
        f"Planned {len(plan)} documents -> {total_questions} questions "
        f"({questions_per_mode} per mode x {len(MODES)} modes), langs={langs}, "
        f"excluding {len(excluded)} covered publications."
    )
    for mode in MODES:
        items = [it for it in plan if it["mode"] == mode]
        by_strategy: Dict[str, int] = {}
        for it in items:
            by_strategy[it["strategy_name"]] = by_strategy.get(it["strategy_name"], 0) + 1
        print(f"  [{mode}] {len(items)} docs  strategy={by_strategy}")

    all_generated_path = Path(all_generated_path)
    best_path = Path(best_path)
    manifest_path = Path(manifest_path)

    fieldnames = _output_fieldnames()
    if not dry_run:
        _ensure_output_file(all_generated_path, fieldnames)
        _ensure_output_file(best_path, fieldnames)

    _write_manifest(manifest_path, plan)
    print(f"Wrote manifest -> {manifest_path}")

    if dry_run:
        print("Dry run only: manifest written, generation skipped.")
        return

    generation_client = _get_client()
    verifier_client = _get_openrouter_client()
    print(f"Generator:        {generation_model} (OpenAI)")
    print(f"Verifier:         {verifier_model} (OpenRouter, thinking)")
    print(f"Workers:          {workers}")
    print(f"Retries/call:     {retries}")
    print(f"Append target:    {all_generated_path}")
    print(f"Best-only target: {best_path}")

    def _generate(item: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        results = _generate_for_target_langs(
            item["publication_number"],
            groups[item["publication_number"]],
            item["target_langs"],
            mode=item["mode"],
            generation_client=generation_client,
            generation_model=generation_model,
            verifier_client=verifier_client,
            verifier_model=verifier_model,
            retries=retries,
        )
        return item, results

    best_count = 0
    all_count = 0
    with best_path.open("a", encoding="utf-8", newline="") as best_f, all_generated_path.open(
        "a", encoding="utf-8", newline=""
    ) as all_f:
        best_writer = csv.DictWriter(best_f, fieldnames=fieldnames)
        all_writer = csv.DictWriter(all_f, fieldnames=fieldnames)

        def _persist(item: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
            nonlocal best_count, all_count
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
                    f"  {item['publication_number']} [{item['mode']}/{item['strategy_name']}]: "
                    f"expected {item['expected_question_count']} questions, got {len(best_rows)}"
                )

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_generate, item) for item in plan]
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Generate EPO Q&A", unit="doc"
                ):
                    item, results = future.result()
                    _persist(item, results)
        else:
            for item in tqdm(plan, desc="Generate EPO Q&A", unit="doc"):
                _persist(*_generate(item))

    print(f"\nAppended {all_count} all-generated rows -> {all_generated_path}")
    print(f"Appended {best_count} best-only rows    -> {best_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced multilingual QAC sample for the EPO corpus "
            "(200 questions = 100 technical + 100 semantic, equal across the four "
            "strategies, languages en/de/fr)."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/EPO/multilingual_corpus.csv"),
        help="Path to the EPO multilingual corpus CSV.",
    )
    parser.add_argument(
        "--all-generated",
        type=Path,
        default=Path("data/EPO/qac/qac_epo.csv"),
        help="Output CSV for all-generated rows (3 candidates per group). Created if missing.",
    )
    parser.add_argument(
        "--best",
        type=Path,
        default=Path("data/EPO/qac/qac_epo_best.csv"),
        help="Output CSV for best-only rows (top candidate per group). Created if missing.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/EPO/qac/qac_epo_manifest.csv"),
        help="Path for the run plan / manifest (written fresh each run).",
    )
    parser.add_argument(
        "--exclude-from",
        type=Path,
        default=Path("data/EPO/qac/qac_epo.csv"),
        help=(
            "CSV whose publication_number column lists pubs to skip (default: the "
            "all-generated file, so re-runs add new documents). Ignored if missing."
        ),
    )
    parser.add_argument(
        "--questions-per-mode",
        type=int,
        default=100,
        help="Questions per mode (default: 100 -> 200 total across both modes).",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default=",".join(DEFAULT_EPO_LANGS),
        help="Comma-separated language set to generate in (default: en,de,fr).",
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
            "OpenRouter model used for the faithfulness and quality verifiers, called "
            f"with extended thinking (default: {DEFAULT_VERIFIER_MODEL}). Requires OPENROUTER_API_KEY."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Concurrent worker threads for generation+grading (default: 1 = serial). API "
            "calls run in parallel; file writes stay single-threaded. 4-8 is reasonable."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=(
            "Retry attempts per failed API call with exponential backoff "
            f"(default: {DEFAULT_RETRIES})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the manifest and skip API calls.",
    )
    args = parser.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    run(
        corpus_path=args.corpus,
        all_generated_path=args.all_generated,
        best_path=args.best,
        manifest_path=args.manifest,
        exclude_from=args.exclude_from,
        questions_per_mode=args.questions_per_mode,
        langs=langs,
        generation_model=args.generation_model,
        verifier_model=args.verifier_model,
        seed=args.seed,
        workers=args.workers,
        retries=args.retries,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
