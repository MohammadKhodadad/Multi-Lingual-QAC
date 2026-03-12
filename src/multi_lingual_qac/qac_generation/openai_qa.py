"""
OpenAI-based Q&A generation (Option A: English first, translate to all languages).

Samples corpus, generates (question, answer) in English per document,
then translates to all target languages.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# Languages to translate into (exclude en; we generate in English)
DEFAULT_TARGET_LANGS = [
    "de", "fr", "es", "ja", "ko", "zh", "ru", "pt", "it", "nl", "ar", "tr", "pl", "hi",
]

LANG_NAMES = {
    "de": "German", "fr": "French", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ru": "Russian", "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
    "ar": "Arabic", "tr": "Turkish", "pl": "Polish", "hi": "Hindi", "en": "English",
}


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


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """Load corpus CSV into list of dicts."""
    rows: List[Dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def sample_corpus(
    rows: List[Dict[str, Any]],
    sample_size: int,
    *,
    stratify_by_language: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample rows from corpus. If stratify_by_language, take proportionally from each language.
    """
    if seed is not None:
        random.seed(seed)
    if sample_size >= len(rows):
        return rows
    if not stratify_by_language:
        return random.sample(rows, sample_size)
    # Stratify: group by language
    by_lang: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        lang = row.get("language", "en")
        by_lang.setdefault(lang, []).append(row)
    # Sample proportionally
    total = len(rows)
    sampled: List[Dict[str, Any]] = []
    for lang, lang_rows in by_lang.items():
        n = max(1, round(sample_size * len(lang_rows) / total))
        n = min(n, len(lang_rows))
        sampled.extend(random.sample(lang_rows, n))
    # If we got more or fewer, trim or pad
    random.shuffle(sampled)
    return sampled[:sample_size]


def generate_qa_english(
    client: OpenAI,
    context: str,
    *,
    model: str = "gpt-4o-mini",
) -> Dict[str, str]:
    """
    Generate one validated-target Q&A pair in English from the given context.
    Returns question, answer, and supporting_text.
    """
    prompt = """You are an expert at creating chemistry and patent retrieval questions.

The source context may be in any language, but your output must be in English only.

Generate exactly ONE question-answer pair from the context.

Rules:
- Output must be in natural English only.
- Do not copy the source language unless a chemical name, formula, identifier, or proper noun should remain unchanged.
- The question must read like a realistic retrieval query that a researcher, engineer, or technical reader might actually type into a search system.
- Prefer short, natural, user-like wording over patent-summary wording.
- Prefer a specific question about one of these: purpose, application, composition, method step, property, technical advantage, operating condition, or material relationship.
- The question must be answerable from the text and specific enough to be useful for retrieval.
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
  Rewrite those into natural user-style English instead.
- Do not copy a sentence from the context nearly verbatim.
- Do not make the question artificially difficult or obscure just to reduce word overlap.
- The answer must be concise (1-2 sentences) and strictly grounded in the context.
- Include a short supporting_text quote copied from the source context that justifies the answer.
- Include a question_type chosen from: purpose, application, composition, method, property, advantage, operating_condition, material_relationship, other.
- Good style examples:
  - "How should the hair-strengthening preparation be applied?"
  - "What products can use the polyamide-based microcapsules?"
  - "What does the shape deformation layer do in the artificial nail?"
- Bad style examples:
  - "What are the recommended application methods for the preparation described in the invention?"
  - "What type of products can include the microcapsules mentioned in the invention?"
- Output valid JSON only, no markdown:
  {"question": "...", "answer": "...", "supporting_text": "...", "question_type": "..."}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Context:\n\n{context[:4000]}"},
        ],
        temperature=0.3,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    return {
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "supporting_text": str(data.get("supporting_text", "")).strip(),
        "question_type": str(data.get("question_type", "other")).strip(),
    }


def check_english_language(
    client: OpenAI,
    question: str,
    answer: str,
    *,
    model: str = "gpt-4o-mini",
) -> Tuple[bool, str]:
    """
    Validate that the generated question and answer are written in English.
    Returns (approved, reason).
    """
    prompt = """You are a strict language checker.

Decide whether BOTH the question and answer are written mainly in English.

Approve only if:
- both are natural English,
- they are not primarily written in another language,
- they are not mixed-language outputs except for unavoidable chemical names, formulas, identifiers, or proper nouns.

Output valid JSON only:
{"approved": true, "reason": "..."}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer: {answer}",
            },
        ],
        temperature=0,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    return approved, reason


def check_faithfulness(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    supporting_text: str,
    *,
    model: str = "gpt-4o-mini",
) -> Tuple[bool, str]:
    """
    Validate that the answer is supported by the source context.
    Returns (approved, reason).
    """
    prompt = """You are a strict faithfulness checker for patent question-answer pairs.

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
        temperature=0,
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
    *,
    model: str = "gpt-4o-mini",
) -> Tuple[bool, str]:
    """
    Validate that the question is retrieval-useful, specific, and not overly generic.
    Returns (approved, reason).
    """
    prompt = """You are a strict quality checker for retrieval questions built from technical patent text.

Approve only if the question:
- sounds like a realistic search or retrieval query,
- is specific enough to distinguish the document,
- asks about a concrete technical point from the context,
- uses natural user-like wording rather than patent-summary wording,
- is not too generic,
- is not nearly copied from the context verbatim,
- and is useful for retrieval benchmarking.

Reject questions that are broad or repetitive patterns such as:
- "What is the main object of the invention?"
- "What is the main feature?"
- "What are the main components?"
- "What are the applications ...?" when a more specific application question is possible
- "What is the composition ...?" when a more targeted material or component question is possible
unless the context is too short for a better question.

Also reject document-centered wording such as:
- "described in the invention"
- "mentioned in the invention"
- "according to the invention"
- "in the text"

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
                    f"Answer: {answer}"
                ),
            },
        ],
        temperature=0,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    return approved, reason


def translate_qa(
    client: OpenAI,
    question: str,
    answer: str,
    target_langs: List[str],
    *,
    model: str = "gpt-4o-mini",
) -> Dict[str, Tuple[str, str]]:
    """
    Translate (question, answer) to target languages. Returns {lang: (q, a)}.
    """
    if not target_langs:
        return {}
    lang_list = ", ".join(LANG_NAMES.get(l, l) for l in target_langs)
    prompt = f"""Translate the following question and answer pair into these languages: {lang_list}.

For each language, produce a natural, retrieval-style translation.
Keep the same meaning, level of specificity, and technical terms where appropriate.
Do not make the question more generic than the original.

Output valid JSON only:
{{"translations": {{"de": {{"question": "...", "answer": "..."}}, "fr": {{...}}, ...}}}}

Languages to include: {json.dumps(target_langs)}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer: {answer}",
            },
        ],
        temperature=0.2,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    trans = data.get("translations", data)
    result: Dict[str, Tuple[str, str]] = {}
    for lang in target_langs:
        if lang in trans and isinstance(trans[lang], dict):
            q = trans[lang].get("question", "")
            a = trans[lang].get("answer", "")
            result[lang] = (str(q).strip(), str(a).strip())
    return result


def _process_sample_row(
    index: int,
    row: Dict[str, Any],
    *,
    target_languages: List[str],
    model: str,
    max_attempts: int,
) -> Dict[str, Any]:
    corpus_id = row.get("id", "")
    context = row.get("context", row.get("abstract", "")) or row.get("title", "")
    if not context.strip():
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": [],
            "status": "skipped (empty context)",
        }

    try:
        client = _get_client()
        approved = False
        q_en = ""
        a_en = ""
        supporting_text = ""
        question_type = ""
        last_failure = ""

        for _attempt in range(1, max_attempts + 1):
            generated = generate_qa_english(client, context, model=model)
            q_en = generated["question"]
            a_en = generated["answer"]
            supporting_text = generated["supporting_text"]
            question_type = generated["question_type"]

            lang_ok, lang_reason = check_english_language(
                client,
                q_en,
                a_en,
                model=model,
            )
            if not lang_ok:
                last_failure = f"language check failed: {lang_reason or 'not English enough'}"
                continue

            faithful_ok, faithful_reason = check_faithfulness(
                client,
                context,
                q_en,
                a_en,
                supporting_text,
                model=model,
            )
            if not faithful_ok:
                last_failure = f"faithfulness check failed: {faithful_reason or 'not grounded enough'}"
                continue

            quality_ok, quality_reason = check_question_quality(
                client,
                context,
                q_en,
                a_en,
                model=model,
            )
            if not quality_ok:
                last_failure = f"quality check failed: {quality_reason or 'question not useful enough'}"
                continue

            approved = True
            break

        if not approved:
            return {
                "index": index,
                "corpus_id": corpus_id,
                "rows": [],
                "status": f"skipped ({last_failure or 'validation failed'})",
            }

        qac_rows = [{
            "corpus_id": corpus_id,
            "language": "en",
            "question": q_en,
            "answer": a_en,
        }]
        trans = translate_qa(client, q_en, a_en, target_languages, model=model)
        for lang, (q, a) in trans.items():
            qac_rows.append({
                "corpus_id": corpus_id,
                "language": lang,
                "question": q,
                "answer": a,
            })
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": qac_rows,
            "status": f"ok ({question_type or 'validated'} en + {len(trans)} translations)",
        }
    except Exception as exc:
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": [],
            "status": f"error: {exc}",
        }


def run_qa_pipeline(
    corpus_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 50,
    target_languages: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    max_attempts: int = 3,
    batch_mode: bool = False,
) -> int:
    """
    Sample corpus, generate Q&A in English, translate to all target languages.
    Writes qac.csv (corpus_id, language, question, answer) to output_dir.
    Returns number of QAC rows written.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    target_languages = target_languages or DEFAULT_TARGET_LANGS
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus(corpus_path)
    sampled = sample_corpus(rows, sample_size, stratify_by_language=True, seed=42)
    print(f"Sampled {len(sampled)} documents from corpus ({len(rows)} total).")
    qac_rows: List[Dict[str, str]] = []
    results: List[Dict[str, Any]] = []

    if batch_mode and sampled:
        available_cpus = os.cpu_count() or 1
        workers = max(1, min(len(sampled), available_cpus))
        print(
            f"Running batched Q&A generation with {workers} worker(s) "
            f"based on {available_cpus} available CPU(s)."
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_sample_row,
                    index,
                    row,
                    target_languages=target_languages,
                    model=model,
                    max_attempts=max_attempts,
                )
                for index, row in enumerate(sampled)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"  [{completed}/{len(sampled)}] {result['corpus_id']}... {result['status']}"
                )
    else:
        if sampled:
            print("Running Q&A generation in single-threaded mode.")
        for index, row in enumerate(sampled, start=1):
            result = _process_sample_row(
                index - 1,
                row,
                target_languages=target_languages,
                model=model,
                max_attempts=max_attempts,
            )
            results.append(result)
            print(f"  [{index}/{len(sampled)}] {result['corpus_id']}... {result['status']}")

    for result in sorted(results, key=lambda item: item["index"]):
        qac_rows.extend(result["rows"])

    out_csv = output_dir / "qac.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus_id", "language", "question", "answer"])
        w.writeheader()
        w.writerows(qac_rows)

    print(f"Wrote {len(qac_rows)} QAC rows -> {out_csv}")
    return len(qac_rows)
