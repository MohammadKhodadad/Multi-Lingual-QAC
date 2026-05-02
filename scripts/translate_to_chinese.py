"""
Translate selected publications from the multilingual corpus to Chinese
(simplified Chinese) and append the translated rows to the same corpus CSV.

For each chosen publication a single new row is appended with language='zh'.
The source row is the English row when present, otherwise any other language
that already exists for that publication. Publications that already have a
Chinese row are skipped.

Row format matches the existing corpus exactly. The ``context`` field is
rebuilt in code with the same logic as the original dataloader
(``src/multi_lingual_qac/dataloaders/google_patents.py``):

    parts = []
    if title:       parts.append(f"Title: {title}")
    if abstract:    parts.append(f"Abstract: {abstract}")
    if first_claim: parts.append(f"First claim: {first_claim}")
    context = "\\n\\n".join(parts).strip()

so the structural labels stay in English exactly as they appear in every
other row, while the body is in Chinese. ``description`` is never part of
``context`` (it isn't in the dataloader either) and is copied through
unchanged. The model is only asked to translate ``title``, ``abstract`` and
``first_claim``.

Translation guidance (encoded in the prompt):
  - Translate fluent prose into natural simplified Chinese.
  - Preserve all chemical names, compound identifiers, trade names, reagent
    abbreviations, and every numerical value, unit, and range exactly as they
    appear in the source — never translate, round, or reformat them.
  - When a technical term has a widely accepted, unambiguous Chinese
    equivalent, use it. When it does not, leave the original term in place to
    avoid ambiguity.

Usage:
    python scripts/translate_to_chinese.py \\
        --corpus data/google_patents/multilingual_corpus.csv \\
        --count 20

The script appends new rows to --corpus in-place. Use --output to write to a
different file instead.

Requires OPENAI_API_KEY in environment (or .env file).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# CSV fields can carry very long patent descriptions; bump the limit so we
# don't truncate the corpus while reading.
csv.field_size_limit(sys.maxsize)

CORPUS_FIELDS = [
    "id",
    "language",
    "title",
    "abstract",
    "description",
    "first_claim",
    "context",
    "publication_number",
    "country_code",
    "publication_date",
    "source",
]

# Fields whose body the model translates. ``first_claim`` is currently always
# empty in the corpus, but the original dataloader does include it in
# ``context`` when populated, so we translate it for parity.
TRANSLATABLE_FIELDS = ("title", "abstract", "first_claim")
# Carried through unchanged. ``description`` is never part of ``context`` in
# the dataloader, so we leave it as-is.
COPY_AS_IS_FIELDS = ("description",)

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "low"

SYSTEM_PROMPT = """You are an expert technical translator specializing in chemistry and patent documents.

Translate the supplied patent fields into simplified Chinese (zh-CN). Translation rules:

1. Produce natural, fluent simplified Chinese suitable for a Chinese-speaking patent reader.
2. Preserve EXACTLY (do NOT translate, round, or reformat):
   - chemical names, IUPAC names, formulas, SMILES, and compound identifiers (e.g., "Compound 3a")
   - trade names, brand names, and proper nouns
   - reagent / catalyst abbreviations (e.g., "Pd/C", "DMSO", "THF")
   - all numerical values, units, ranges, tolerances, and percentages (e.g., "85–87°C", "0.1 wt%")
   - element symbols and reference numerals in parentheses (e.g., "(10)", "(20)")
3. Translate technical terminology only when there is a widely accepted, unambiguous Chinese equivalent
   (e.g., "催化剂" for "catalyst", "溶剂" for "solvent"). When a term has no settled, unambiguous Chinese
   form, or translating would introduce ambiguity, KEEP the original (typically English) term in place
   inside the Chinese sentence. Prefer keeping the original over guessing.
4. Translate ordinary prose around the technical terms — connectors, descriptions, explanations — into
   natural Chinese. Do not produce word-for-word translations and do not omit content.
5. Keep paragraph and list structure. Do not add commentary, headings, or notes.
6. Translate ONLY the body text in each field. Do NOT translate or invent structural labels — there
   are none in the input. If a field is empty in the source, return an empty string for that field.

You will receive a JSON object whose keys are field names and whose values are source text. Return a
JSON object with the same keys whose values are the Chinese translations following the rules above.
Output JSON only, no markdown fences, no extra text.
"""


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env for translation.")
    return OpenAI(api_key=api_key)


def _parse_json_response(text: str) -> Dict[str, str]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def load_corpus(corpus_path: Path) -> List[Dict[str, str]]:
    with corpus_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_by_publication(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["publication_number"]].append(row)
    return groups


def _pick_source_row(rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """English row preferred; otherwise the first non-Chinese row with any
    translatable content."""
    by_lang = {r["language"]: r for r in rows}
    if "en" in by_lang:
        return by_lang["en"]
    for lang in ("fr", "de", "es"):
        if lang in by_lang:
            return by_lang[lang]
    for r in rows:
        if r["language"] != "zh" and any((r.get(f) or "").strip() for f in TRANSLATABLE_FIELDS):
            return r
    return None


def _build_context(title: str, abstract: str, first_claim: str) -> str:
    """Reconstruct the ``context`` field exactly the way the original corpus
    builder does in ``src/multi_lingual_qac/dataloaders/google_patents.py``:
    English structural labels, only non-empty parts included, joined with a
    blank line, stripped. ``description`` is intentionally not part of the
    context (matches the dataloader)."""
    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if first_claim:
        parts.append(f"First claim: {first_claim}")
    return "\n\n".join(parts).strip()


def translate_row(
    client: OpenAI,
    source: Dict[str, str],
    *,
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """Translate the translatable fields of *source* into Chinese.

    Returns a new row dict with the same keys as the corpus schema, language
    set to 'zh', id suffixed with '_zh', and translatable fields replaced
    with their Chinese translations. The ``context`` field is rebuilt from
    the translated title/abstract using the same English structural labels
    used by every other row in the corpus.
    """
    payload = {field: source.get(field, "") or "" for field in TRANSLATABLE_FIELDS}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    translated = _parse_json_response(response.choices[0].message.content or "")

    new_row: Dict[str, str] = {field: source.get(field, "") for field in CORPUS_FIELDS}
    new_row["language"] = "zh"
    new_row["id"] = f"{source['publication_number']}_zh"
    for field in TRANSLATABLE_FIELDS:
        value = translated.get(field, "")
        new_row[field] = "" if value is None else str(value)
    for field in COPY_AS_IS_FIELDS:
        new_row[field] = source.get(field, "") or ""
    new_row["context"] = _build_context(
        new_row["title"], new_row["abstract"], new_row["first_claim"]
    )
    return new_row


def select_publications_to_translate(
    groups: Dict[str, List[Dict[str, str]]],
    *,
    count: int,
    seed: int,
    skip_publications: Optional[set[str]] = None,
) -> List[str]:
    """Pick publications that have non-Chinese content and lack a Chinese row."""
    skip = skip_publications or set()
    candidates: List[str] = []
    for pub_num, rows in groups.items():
        if pub_num in skip:
            continue
        languages = {r["language"] for r in rows}
        if "zh" in languages:
            continue
        source = _pick_source_row(rows)
        if source is None:
            continue
        candidates.append(pub_num)

    if count > len(candidates):
        raise ValueError(
            f"Requested {count} translations but only {len(candidates)} eligible publications "
            f"found (after skipping {len(skip)} excluded and existing zh rows)."
        )

    rng = random.Random(seed)
    candidates.sort()
    return rng.sample(candidates, count)


def append_rows(corpus_path: Path, new_rows: List[Dict[str, str]]) -> None:
    """Append rows to the corpus CSV. Header is assumed to already be present."""
    with corpus_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CORPUS_FIELDS, extrasaction="ignore")
        writer.writerows(new_rows)


def write_full(corpus_path: Path, all_rows: List[Dict[str, str]]) -> None:
    """Write the full CSV (used when --output points to a new file)."""
    with corpus_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CORPUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def run_translation(
    corpus_path: Path,
    *,
    count: int,
    output_path: Optional[Path] = None,
    publications: Optional[List[str]] = None,
    skip_publications: Optional[set[str]] = None,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
) -> List[Dict[str, str]]:
    rows = load_corpus(corpus_path)
    groups = group_by_publication(rows)

    if publications:
        chosen = list(publications)
        missing = [p for p in chosen if p not in groups]
        if missing:
            raise ValueError(f"Publications not found in corpus: {missing}")
    else:
        chosen = select_publications_to_translate(
            groups,
            count=count,
            seed=seed,
            skip_publications=skip_publications,
        )

    print(f"Translating {len(chosen)} publications to Chinese using {model}")

    client = _get_client()
    translated_rows: List[Dict[str, str]] = []
    for pub_num in tqdm(chosen, desc="Translate -> zh", unit="doc"):
        source = _pick_source_row(groups[pub_num])
        if source is None:
            tqdm.write(f"  {pub_num}: skipped (no translatable source row)")
            continue
        try:
            new_row = translate_row(client, source, model=model)
        except Exception as exc:
            tqdm.write(f"  {pub_num}: translation error: {exc}")
            continue
        translated_rows.append(new_row)
        tqdm.write(f"  {pub_num}: ok (source language={source['language']})")

    if output_path is None or output_path == corpus_path:
        append_rows(corpus_path, translated_rows)
        print(f"\nAppended {len(translated_rows)} Chinese rows -> {corpus_path}")
    else:
        write_full(output_path, rows + translated_rows)
        print(f"\nWrote {len(rows) + len(translated_rows)} rows -> {output_path}")

    return translated_rows


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Translate selected publications to simplified Chinese and append "
            "the translations to the multilingual corpus CSV."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/google_patents/multilingual_corpus.csv"),
        help="Path to multilingual corpus CSV (rows will be appended in place unless --output is set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, new rows are appended to --corpus in place.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of publications to translate (default: 20).",
    )
    parser.add_argument(
        "--publications",
        nargs="*",
        default=None,
        help="Optional explicit list of publication_numbers to translate (overrides --count and sampling).",
    )
    parser.add_argument(
        "--skip-from-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV file whose 'publication_number' column lists pubs to exclude from sampling. "
            "Used to avoid re-translating publications already covered by a previous QAC run."
        ),
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
        help="Random seed for sampling (default: 42).",
    )
    args = parser.parse_args()

    skip: Optional[set[str]] = None
    if args.skip_from_csv:
        with args.skip_from_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            skip = {row["publication_number"] for row in reader if row.get("publication_number")}

    run_translation(
        corpus_path=args.corpus,
        count=args.count,
        output_path=args.output,
        publications=args.publications,
        skip_publications=skip,
        model=args.model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
