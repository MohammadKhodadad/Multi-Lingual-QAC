"""
OpenAI-based Q&A generation (Option A: English first, translate to all languages).

Samples corpus, generates (question, answer) in English per document,
then translates to all target languages.
"""

from __future__ import annotations

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


def generate_qa_english(client: OpenAI, context: str, *, model: str = "gpt-4o-mini") -> Tuple[str, str]:
    """
    Generate one (question, answer) pair in English from the given context.
    Returns (question, answer).
    """
    prompt = """You are an expert at creating reading comprehension questions from technical chemistry/patent texts.

Given the following patent abstract or context, generate exactly ONE question and its answer.

Rules:
- The question must be answerable from the text (extractive or short abstractive).
- The answer must be concise (1-3 sentences) and grounded in the context.
- Focus on chemistry, materials, processes, or technical details.
- Output valid JSON only, no markdown: {"question": "...", "answer": "..."}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Context:\n\n{context[:4000]}"},
        ],
        temperature=0.3,
    )
    text = response.choices[0].message.content or ""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    return (data.get("question", "").strip(), data.get("answer", "").strip())


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

For each language, produce a natural translation. Keep the same meaning and technical terms where appropriate.

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
    text = response.choices[0].message.content or ""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    trans = data.get("translations", data)
    result: Dict[str, Tuple[str, str]] = {}
    for lang in target_langs:
        if lang in trans and isinstance(trans[lang], dict):
            q = trans[lang].get("question", "")
            a = trans[lang].get("answer", "")
            result[lang] = (str(q).strip(), str(a).strip())
    return result


def run_qa_pipeline(
    corpus_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 50,
    target_languages: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
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

    client = _get_client()
    qac_rows: List[Dict[str, str]] = []

    for i, row in enumerate(sampled):
        corpus_id = row.get("id", "")
        context = row.get("context", row.get("abstract", "")) or row.get("title", "")
        if not context.strip():
            continue
        print(f"  [{i+1}/{len(sampled)}] {corpus_id}...", end=" ", flush=True)
        try:
            q_en, a_en = generate_qa_english(client, context, model=model)
            qac_rows.append({
                "corpus_id": corpus_id,
                "language": "en",
                "question": q_en,
                "answer": a_en,
            })
            trans = translate_qa(client, q_en, a_en, target_languages, model=model)
            for lang, (q, a) in trans.items():
                qac_rows.append({
                    "corpus_id": corpus_id,
                    "language": lang,
                    "question": q,
                    "answer": a,
                })
            print(f"ok (en + {len(trans)} translations)")
        except Exception as e:
            print(f"error: {e}")

    out_csv = output_dir / "qac.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus_id", "language", "question", "answer"])
        w.writeheader()
        w.writerows(qac_rows)

    print(f"Wrote {len(qac_rows)} QAC rows -> {out_csv}")
    return len(qac_rows)
