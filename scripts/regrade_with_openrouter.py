"""
Re-grade existing QAC candidates using OpenRouter with Claude Sonnet 4.6 thinking.

Reads questions from an existing _all_generated CSV, re-runs the faithfulness
and quality verifiers using a new model (no question generation), and writes a
new CSV in exactly the same format.

Usage:
    python scripts/regrade_with_openrouter.py \
        --input  data/google_patents/qac/balanced_100_qac_all_generated.csv \
        --corpus data/google_patents/multilingual_corpus.csv \
        --output data/google_patents/qac/balanced_100_qac_all_generated_regraded.csv

Requires OPENROUTER_API_KEY in environment (or .env file).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_lingual_qac.qac_generation.multilingual_qa import (
    FAITHFULNESS_FIELDS,
    MODE_SEMANTIC,
    MODE_TECHNICAL,
    SEMANTIC_QUALITY_FIELDS,
    TECHNICAL_QUALITY_FIELDS,
    _build_all_passages_text,
    _compute_faith_overall,
    _compute_quality_overall,
    _compute_total_score,
    _load_faithfulness_prompt,
    _load_quality_prompt,
    _parse_json_response,
    load_multilingual_corpus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
THINKING_BUDGET_TOKENS = 8000
MAX_TOKENS = 12000


# ---------------------------------------------------------------------------
# Output schema  (mirrors balanced_multilingual_qa._output_fieldnames)
# ---------------------------------------------------------------------------

def _output_fieldnames() -> List[str]:
    return [
        "mode", "strategy", "strategy_name",
        "corpus_id", "publication_number", "question_language",
        "context_language", "question", "answer",
        "question_type", "framing",
        *FAITHFULNESS_FIELDS,
        *TECHNICAL_QUALITY_FIELDS,
        *SEMANTIC_QUALITY_FIELDS,
        "qual_failure_type",
        "total_score",
    ]


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields with empty strings so DictWriter is happy."""
    normalized = {field: "" for field in _output_fieldnames()}
    normalized.update(row)
    return normalized


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

def _get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in environment or .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _call_with_thinking(client: OpenAI, model: str, messages: list) -> str:
    """Call OpenRouter with extended thinking enabled; return text content."""
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


# ---------------------------------------------------------------------------
# Verifiers (same logic as multilingual_qa, but use thinking instead of
# reasoning_effort so the calls are compatible with Claude via OpenRouter)
# ---------------------------------------------------------------------------

def _grade_faithfulness(
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
            "grounding": 1, "precision": 1, "numerical_fidelity": 1, "reason": "missing",
        }
        fallback["overall"] = _compute_faith_overall(fallback)
        normalised.append(fallback)

    return normalised


def _grade_quality(
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
            "search_bar_realism": 1, "specificity": 1, "phrasing_economy": 1,
            "focus": 1, "linguistic_quality": 1, "failure_type": "missing", "reason": "missing",
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
            "search_realism": 1, "lexical_distance": 1, "conceptual_framing": 1,
            "retrievability": 1, "linguistic_quality": 1, "failure_type": "missing", "reason": "missing",
        }

    while len(normalised) < 3:
        fallback = dict(default)
        fallback["overall"] = _compute_quality_overall(fallback, mode)
        normalised.append(fallback)

    return normalised


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_output_row(
    original: Dict[str, str],
    faith: Dict[str, Any],
    qual: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "mode": original["mode"],
        "strategy": original["strategy"],
        "strategy_name": original["strategy_name"],
        "corpus_id": original["corpus_id"],
        "publication_number": original["publication_number"],
        "question_language": original["question_language"],
        "context_language": original["context_language"],
        "question": original["question"],
        "answer": original["answer"],
        "question_type": original.get("question_type", ""),
        "framing": original.get("framing", ""),
        "faith_grounding": faith["grounding"],
        "faith_precision": faith["precision"],
        "faith_numerical_fidelity": faith["numerical_fidelity"],
        "faith_overall": faith["overall"],
    }

    if mode == MODE_TECHNICAL:
        row["qual_search_bar_realism"] = qual["search_bar_realism"]
        row["qual_specificity"] = qual["specificity"]
        row["qual_phrasing_economy"] = qual["phrasing_economy"]
        row["qual_focus"] = qual["focus"]
        row["qual_linguistic_quality"] = qual["linguistic_quality"]
        row["qual_overall"] = qual["overall"]
    else:
        row["qual_search_realism"] = qual["search_realism"]
        row["qual_lexical_distance"] = qual["lexical_distance"]
        row["qual_conceptual_framing"] = qual["conceptual_framing"]
        row["qual_retrievability"] = qual["retrievability"]
        row["qual_linguistic_quality"] = qual["linguistic_quality"]
        row["qual_overall"] = qual["overall"]

    row["qual_failure_type"] = qual.get("failure_type", "none")
    row["total_score"] = _compute_total_score(faith, qual, mode)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_regrading(
    input_path: Path,
    corpus_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
) -> int:
    corpus = load_multilingual_corpus(corpus_path)

    with input_path.open(encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))

    print(f"Read {len(input_rows)} rows from {input_path}")

    # Group rows by (publication_number, question_language); preserve insertion order
    # so the 3 candidates per group stay together.
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in input_rows:
        key = (row["publication_number"], row["question_language"])
        groups[key].append(row)

    print(f"Found {len(groups)} (publication, language) groups to re-grade")

    client = _get_openrouter_client()
    fieldnames = _output_fieldnames()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        out_f.flush()

        for (pub_num, lang), group_rows in tqdm(groups.items(), desc="Re-grading", unit="group"):
            mode = group_rows[0]["mode"]

            corpus_rows = corpus.get(pub_num, [])
            if not corpus_rows:
                tqdm.write(f"  {pub_num}: no corpus rows found, skipping")
                continue

            all_passages = _build_all_passages_text(corpus_rows)

            qa_pairs: List[Dict[str, str]] = []
            for r in group_rows:
                qa: Dict[str, str] = {"question": r["question"], "answer": r["answer"]}
                if mode == MODE_TECHNICAL:
                    qa["question_type"] = r.get("question_type", "")
                else:
                    qa["framing"] = r.get("framing", "")
                qa_pairs.append(qa)

            try:
                faith_grades = _grade_faithfulness(client, model, all_passages, qa_pairs)
            except Exception as exc:
                tqdm.write(f"  {pub_num} [{lang}]: faithfulness grading error: {exc}")
                continue

            try:
                qual_grades = _grade_quality(client, model, all_passages, qa_pairs, mode)
            except Exception as exc:
                tqdm.write(f"  {pub_num} [{lang}]: quality grading error: {exc}")
                continue

            group_out: List[Dict[str, Any]] = []
            for i, orig in enumerate(group_rows):
                out = _build_output_row(orig, faith_grades[i], qual_grades[i], mode)
                group_out.append(_normalize_row(out))

            group_out.sort(key=lambda r: int(r.get("total_score", 0) or 0), reverse=True)

            writer.writerows(group_out)
            out_f.flush()
            rows_written += len(group_out)

            best = group_out[0]["total_score"]
            tqdm.write(f"  {pub_num} [{lang}]: ok (best={best})")

    print(f"\nWrote {rows_written} regraded rows -> {output_path}")
    return rows_written


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Re-grade existing QAC questions with OpenRouter Claude thinking."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/google_patents/qac/balanced_100_qac_all_generated.csv"),
        help="Input CSV (the _all_generated file)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/google_patents/multilingual_corpus.csv"),
        help="Multilingual corpus CSV (needed for passage context)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/google_patents/qac/balanced_100_qac_all_generated_regraded.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    run_regrading(
        input_path=args.input,
        corpus_path=args.corpus,
        output_path=args.output,
        model=args.model,
    )


if __name__ == "__main__":
    main()
