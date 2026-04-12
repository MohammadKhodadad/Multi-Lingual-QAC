"""
Alternative multilingual Q&A generation for documents that already exist in
multiple languages.  No translation step needed — questions are generated
directly in the target language.

Four strategies for choosing the question language:
  1  RANDOM_ANY       — pick a random language from {en, de, fr, es}
  2  RANDOM_MISSING   — pick a random language NOT in the document's languages
  3  RANDOM_EXISTING  — pick a random language that IS in the document's languages
  4  ALL              — generate a question for ALL 4 languages
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LANGS = ["en", "de", "fr", "es"]
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}

STRATEGY_RANDOM_ANY = 1
STRATEGY_RANDOM_MISSING = 2
STRATEGY_RANDOM_EXISTING = 3
STRATEGY_ALL = 4

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_GENERATION_REASONING_EFFORT = "medium"
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env for Q&A generation.")
    return OpenAI(api_key=api_key)


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_multilingual_corpus(
    corpus_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load the multilingual corpus CSV and group rows by publication_number.

    Returns {publication_number: [row_dict, ...]}.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with Path(corpus_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["publication_number"]].append(dict(row))
    return dict(groups)


# ---------------------------------------------------------------------------
# Strategy: pick target language(s)
# ---------------------------------------------------------------------------


def pick_target_languages(
    strategy: int,
    available_langs: list[str],
) -> list[str]:
    """
    Given the set of languages a document exists in and a strategy number,
    return the list of target language(s) to generate questions in.
    """
    available_set = set(available_langs) & set(ALL_LANGS)
    missing = [l for l in ALL_LANGS if l not in available_set]

    if strategy == STRATEGY_RANDOM_ANY:
        return [random.choice(ALL_LANGS)]

    if strategy == STRATEGY_RANDOM_MISSING:
        if not missing:
            # Document exists in all 4 languages — fall back to random any
            return [random.choice(ALL_LANGS)]
        return [random.choice(missing)]

    if strategy == STRATEGY_RANDOM_EXISTING:
        existing = [l for l in ALL_LANGS if l in available_set]
        if not existing:
            # Should not happen in a well-formed corpus, fall back
            return [random.choice(ALL_LANGS)]
        return [random.choice(existing)]

    if strategy == STRATEGY_ALL:
        return list(ALL_LANGS)

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Context selection
# ---------------------------------------------------------------------------


def _pick_context(
    rows: List[Dict[str, Any]],
    target_lang: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Pick the best context row for generating a question in *target_lang*.

    Prefers the row whose language matches target_lang.  Falls back to any
    available row (preferring English if present).

    Returns (chosen_row, context_text).
    """
    by_lang = {r["language"]: r for r in rows}

    # Best: same language as the question
    if target_lang in by_lang:
        row = by_lang[target_lang]
        return row, row.get("context") or row.get("abstract") or row.get("title", "")

    # Fallback: prefer English, then any
    for fallback in ["en"] + list(by_lang.keys()):
        if fallback in by_lang:
            row = by_lang[fallback]
            return row, row.get("context") or row.get("abstract") or row.get("title", "")

    # Should never get here
    row = rows[0]
    return row, row.get("context") or row.get("abstract") or row.get("title", "")


# ---------------------------------------------------------------------------
# Q&A generation (language-aware)
# ---------------------------------------------------------------------------

_BASE_PROMPT = """You are an expert at creating chemistry and patent retrieval questions.

The source context may be in any language.

Generate exactly ONE question-answer pair from the context.

Rules:
- Do not copy the source language unless a chemical name, formula, identifier, or proper noun should remain unchanged.
- The question must read like a realistic retrieval query that a researcher, engineer, or technical reader might actually type into a search system.
- Prefer short, natural, user-like wording over patent-summary wording.
- Prefer a specific question about one of these: purpose, application, composition, method step, property, technical advantage, operating condition, material relationship, mechanism, effect, or functional role.
- The question must be answerable from the text and specific enough to be useful for retrieval.
- Prefer semantically challenging questions that dense retrieval should handle better than simple keyword matching.
- For this task, semantic reformulation is more important than choosing the easiest extractive fact.
- First look for a question about rationale, role, effect, mechanism, interaction, implication, or process purpose tied to a specific step.
- Only fall back to exact ranges, exact ratios, exact named lists, or exact composition tables when the context does not support a stronger semantic question.
- If both are possible, prefer the question that requires understanding what the detail does or why it matters, not the one that only asks for the raw value.
- Prefer a question about one core technical fact, not a bundled summary of multiple advantages or multiple properties, unless the source presents them as one inseparable claim.
- Ask about function, effect, mechanism, role, use condition, or technical implication when possible, not just surface wording.
- Vary the question form across examples. Do not default to "How does ..." if another natural opening fits the fact better.
- Match the question opening to the fact type. Use forms such as:
  - "Why is ..." for step rationale or process purpose tied to a specific step
  - "Which ..." for identified biomarker pairs, materials, components, or options
  - "What function does ..." for a component's role
  - "What condition ..." or "At what ..." for operating constraints or measured ranges
  - "What property allows ..." for enabling characteristics
  - "How does ..." for mechanism, effect, or interaction only when that is the most natural form
- If you choose a method-style question, name the actual step, material, condition, or operation from the context.
- Do not ask vague questions like "What is the purpose of the method?" or "What is something about the process?" when the method contains a specific named step that can be asked about directly.
- Avoid making the question easy for exact-match retrieval by simply lifting the most distinctive nouns from the source into a template question.
- Preserve technical terms only when they are necessary for faithfulness or the question would become unnatural or ambiguous without them.
- Prefer grounded paraphrase over direct lexical overlap.
- Avoid spec-sheet questions when a more semantic alternative exists, especially:
  - exact wt% or mol-ratio lookups
  - exact temperature, density, time, or concentration range lookups
  - exact component inventory or long named-list lookups
  - exact "what does the composition contain" questions
- A numeric question is acceptable only when the number itself is the important retrieval target and the context does not support a better question about function, rationale, or effect.
- Avoid broad fallback wording like:
  - "What is the purpose of ..."
  - "What advantages does ... offer ..."
  - "What benefits does ... provide ..."
  when you can instead ask what a step achieves, why it is done, what a component does, or what effect it has.
- Avoid turning classification/grouping text into a weak taxonomy question if the same text supports a more functional question.
- When the context contains both "what it is" and "what it does", prefer "what it does".
- Avoid generic questions such as:
  - "What is the main object of the invention?"
  - "What is the main feature of the invention?"
  - "What are the main components?"
  unless the text is too short for anything better.
- Avoid document-centered phrasing such as:
  - "described in the invention"
  - "mentioned in the invention"
  - "according to the invention"
  - "in the text"
  - "described in the text"
  - "described in the present disclosure"
  - "used in the invention"
  Rewrite those into natural user-style wording instead.
- Do not begin the question with broad template wording such as:
  - "What is the application of ..."
  - "What are the advantages of ..."
  - "What is the benefit of ..."
  - "What is the main technical advantage of ..."
  - "What is the purpose of ..."
  - "What types of products ..."
  - "What is the role of ..." when it only asks for a broad use summary
- Avoid these patterns especially when a more specific question can be asked about:
  - one process step
  - one operating condition
  - one material property
  - one mechanism
  - one component interaction
- Do not turn the title into a question.
- Do not simply wrap a copied title phrase or copied noun phrase in a question template.
- If a title-like wording comes to mind first, rewrite it into a more natural and more semantically reformulated question.
- Do not copy a sentence from the context nearly verbatim.
- Do not just wrap a copied noun phrase in a question template.
- Do not keep unusually high word overlap with the opening sentence or title unless a few technical anchor terms must remain for clarity.
- Do not make the question artificially difficult or obscure just to reduce word overlap.
- Before finalizing, ask yourself:
  - Did I choose the deepest answerable fact rather than the easiest extractive fact?
  - Would this still look like a good query if the exact numbers or list items were hidden?
  - Does this require some semantic understanding rather than simple table lookup?
  - Did I accidentally ask for a broad purpose/advantage summary when a narrower technical question was available?
- If the question starts with "What is the purpose of" or "What advantages does", rewrite it unless no narrower question is possible.
- If the answer to those checks is no, regenerate a better question.
- The answer must be concise (1-2 sentences) and strictly grounded in the context.
- Include a short supporting_text quote copied from the source context that justifies the answer.
- Include a question_type chosen from: purpose, application, composition, method, property, advantage, operating_condition, material_relationship, other.
- Output valid JSON only, no markdown:
  {{"question": "...", "answer": "...", "supporting_text": "...", "question_type": "..."}}

IMPORTANT: You MUST write the question in {lang_name}. The question must sound natural and fluent to a native {lang_name} speaker. The answer should be in English.
"""


def generate_qa_in_language(
    client: OpenAI,
    context: str,
    target_lang: str,
    *,
    previous_question: Optional[str] = None,
    previous_answer: Optional[str] = None,
    previous_feedback: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """
    Generate one Q&A pair where the question is in *target_lang* and the
    answer is in English.
    """
    lang_name = LANG_NAMES[target_lang]
    prompt = _BASE_PROMPT.format(lang_name=lang_name)

    retry_note = ""
    if previous_feedback:
        retry_note = (
            "\n\nPrevious attempt issue to fix:\n"
            f"{previous_feedback}\n"
            "Regenerate the question and answer so they fix that issue "
            "while staying fully grounded in the context."
        )
    previous_attempt_note = ""
    if previous_question or previous_answer:
        previous_attempt_note = (
            "\n\nPrevious failed attempt to improve upon:\n"
            f"Previous question: {previous_question or ''}\n"
            f"Previous answer: {previous_answer or ''}\n"
            "Use this only as feedback about what to avoid or improve. "
            "Do not lightly edit it or reuse its wording as a template. "
            "Generate a fresh corrected question-answer pair."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n\n{context[:4000]}"
                    f"{retry_note}"
                    f"{previous_attempt_note}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_GENERATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    return {
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "supporting_text": str(data.get("supporting_text", "")).strip(),
        "question_type": str(data.get("question_type", "other")).strip(),
    }


# ---------------------------------------------------------------------------
# Validation gates (faithfulness + quality)
# ---------------------------------------------------------------------------


def check_faithfulness(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    supporting_text: str,
    *,
    model: str = DEFAULT_MODEL,
) -> Tuple[bool, str]:
    """Validate that the answer is supported by the source context."""
    prompt = """You are a strict faithfulness checker for patent question-answer pairs.

The question may be in any language. The answer is in English.

Approve only if:
- the question is answerable from the context,
- the answer is fully supported by the context,
- the answer does not add unsupported details,
- the supporting_text is relevant evidence from the context.

Reject if the answer is generic, speculative, partially unsupported, or not clearly grounded.

Output valid JSON only:
{"approved": true, "reason": "..."}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}\n\n"
                    f"Supporting text: {supporting_text}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    return approved, reason


def check_question_quality(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    target_lang: str,
    *,
    model: str = DEFAULT_MODEL,
) -> Tuple[bool, str]:
    """Validate that the question is retrieval-useful and in the correct language."""
    lang_name = LANG_NAMES[target_lang]
    prompt = f"""You are a strict quality checker for retrieval questions built from technical patent text.

The question is expected to be in {lang_name}.

Reject the question if it is NOT written in {lang_name}.

Approve only if the question:
- is clearly written in {lang_name},
- sounds like a realistic search or retrieval query,
- is specific enough to distinguish the document,
- asks about a concrete technical point from the context,
- uses natural user-like wording rather than patent-summary wording,
- is phrased semantically rather than as an obvious exact-match template,
- is not too generic,
- is not nearly copied from the context verbatim,
- and is useful for retrieval benchmarking.

Reject broad patterns such as:
- "What is the main object of the invention?"
- "What is the purpose ...?" without naming a specific step or component
- "What are the advantages ...?" leading to a bundled summary
- document-centered wording ("described in the invention", "in the text")

If you reject:
- set failure_type to one of: wrong-language, title-lift, high-overlap, overly-extractive, broad-summary, bundled-facts, weak-query-shape
- provide a short better_direction hint

Output valid JSON only:
{{"approved": true, "reason": "...", "failure_type": "none", "better_direction": ""}}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    failure_type = str(data.get("failure_type", "")).strip()
    better_direction = str(data.get("better_direction", "")).strip()
    if failure_type and failure_type != "none":
        reason = f"{failure_type}: {reason}" if reason else failure_type
    if better_direction:
        reason = f"{reason} Better direction: {better_direction}".strip()
    return approved, reason


# ---------------------------------------------------------------------------
# Process one document group
# ---------------------------------------------------------------------------


def _process_document(
    pub_num: str,
    rows: List[Dict[str, Any]],
    *,
    strategy: int,
    model: str,
    max_attempts: int,
) -> List[Dict[str, str]]:
    """
    Generate Q&A pair(s) for one publication (one group of multilingual rows).

    Returns a list of output dicts, one per successfully generated Q&A.
    """
    available_langs = [r["language"] for r in rows]
    target_langs = pick_target_languages(strategy, available_langs)
    client = _get_client()
    results: List[Dict[str, str]] = []

    for target_lang in target_langs:
        context_row, context_text = _pick_context(rows, target_lang)
        if not context_text.strip():
            tqdm.write(f"  {pub_num} [{target_lang}]: skipped (empty context)")
            continue

        approved = False
        q = ""
        a = ""
        supporting_text = ""
        question_type = ""
        last_failure = ""
        retry_feedback: Optional[str] = None
        retry_question: Optional[str] = None
        retry_answer: Optional[str] = None

        for _attempt in range(1, max_attempts + 1):
            try:
                generated = generate_qa_in_language(
                    client,
                    context_text,
                    target_lang,
                    previous_question=retry_question,
                    previous_answer=retry_answer,
                    previous_feedback=retry_feedback,
                    model=model,
                )
            except Exception as exc:
                last_failure = f"generation error: {exc}"
                break

            q = generated["question"]
            a = generated["answer"]
            supporting_text = generated["supporting_text"]
            question_type = generated["question_type"]
            retry_question = q
            retry_answer = a

            # Gate 1: faithfulness
            try:
                faith_ok, faith_reason = check_faithfulness(
                    client, context_text, q, a, supporting_text, model=model,
                )
            except Exception as exc:
                last_failure = f"faithfulness check error: {exc}"
                break
            if not faith_ok:
                last_failure = f"faithfulness: {faith_reason}"
                retry_feedback = (
                    f"{last_failure}. Remove unsupported details and keep "
                    "the answer strictly grounded in the context."
                )
                continue

            # Gate 2: quality + language correctness
            try:
                qual_ok, qual_reason = check_question_quality(
                    client, context_text, q, a, target_lang, model=model,
                )
            except Exception as exc:
                last_failure = f"quality check error: {exc}"
                break
            if not qual_ok:
                last_failure = f"quality: {qual_reason}"
                retry_feedback = (
                    f"{last_failure}. Use the better direction above if "
                    "present. Regenerate one fresh question that is more "
                    "retrieval-useful, more specific, less generic, and "
                    f"written in natural {LANG_NAMES[target_lang]}."
                )
                continue

            approved = True
            break

        if approved:
            results.append({
                "corpus_id": context_row.get("id", ""),
                "publication_number": pub_num,
                "question_language": target_lang,
                "context_language": context_row.get("language", ""),
                "question": q,
                "answer": a,
                "question_type": question_type,
            })
            tqdm.write(
                f"  {pub_num} [{target_lang}]: ok ({question_type})"
            )
        else:
            tqdm.write(
                f"  {pub_num} [{target_lang}]: skipped ({last_failure})"
            )

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_multilingual_qa_pipeline(
    corpus_path: Path,
    output_path: Path,
    *,
    strategy: int = STRATEGY_RANDOM_ANY,
    model: str = DEFAULT_MODEL,
    max_attempts: int = MAX_ATTEMPTS,
    seed: int = 42,
    limit: Optional[int] = None,
) -> int:
    """
    Generate Q&A pairs from a multilingual corpus using the given strategy.

    Parameters
    ----------
    corpus_path : Path to multilingual_corpus.csv
    output_path : Path for the output QAC CSV
    strategy    : 1=random_any, 2=random_missing, 3=random_existing, 4=all
    model       : OpenAI model name
    max_attempts: retries per question
    seed        : random seed for reproducibility
    limit       : if set, only process this many documents (for testing)

    Returns number of QAC rows written.
    """
    random.seed(seed)
    groups = load_multilingual_corpus(corpus_path)
    pub_nums = list(groups.keys())

    if limit and limit < len(pub_nums):
        pub_nums = pub_nums[:limit]

    strategy_names = {
        STRATEGY_RANDOM_ANY: "random_any",
        STRATEGY_RANDOM_MISSING: "random_missing",
        STRATEGY_RANDOM_EXISTING: "random_existing",
        STRATEGY_ALL: "all",
    }
    print(
        f"Multilingual QA generation: {len(pub_nums)} documents, "
        f"strategy={strategy_names.get(strategy, strategy)}, model={model}"
    )

    all_rows: List[Dict[str, str]] = []
    progress = tqdm(pub_nums, desc="Generate Q&A", unit="doc")
    for pub_num in progress:
        rows = groups[pub_num]
        try:
            results = _process_document(
                pub_num,
                rows,
                strategy=strategy,
                model=model,
                max_attempts=max_attempts,
            )
            all_rows.extend(results)
        except Exception as exc:
            tqdm.write(f"  {pub_num}: error: {exc}")

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "corpus_id", "publication_number", "question_language",
        "context_language", "question", "answer", "question_type",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} QAC rows -> {output_path}")
    return len(all_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Generate multilingual Q&A from documents existing in multiple languages. "
            "No translation step — questions are generated directly in the target language."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/google_patents/multilingual_corpus.csv"),
        help="Path to multilingual corpus CSV (default: data/google_patents/multilingual_corpus.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/google_patents/qac/multilingual_qac.csv"),
        help="Output QAC CSV path (default: data/google_patents/qac/multilingual_qac.csv)",
    )
    parser.add_argument(
        "--strategy",
        type=int,
        default=STRATEGY_RANDOM_ANY,
        choices=[
            STRATEGY_RANDOM_ANY,
            STRATEGY_RANDOM_MISSING,
            STRATEGY_RANDOM_EXISTING,
            STRATEGY_ALL,
        ],
        help=(
            "Language selection strategy: "
            "1=random from {en,de,fr,es}, "
            "2=random from languages NOT in the document, "
            "3=random from languages IN the document, "
            "4=all 4 languages (default: 1)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_ATTEMPTS,
        help=f"Max retry attempts per question (default: {MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many documents (for testing)",
    )
    args = parser.parse_args()

    run_multilingual_qa_pipeline(
        corpus_path=args.corpus,
        output_path=args.output,
        strategy=args.strategy,
        model=args.model,
        max_attempts=args.max_attempts,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
