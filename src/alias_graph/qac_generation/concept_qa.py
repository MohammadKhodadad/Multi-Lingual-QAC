"""
Concept-query QA generation for the Alias-Graph benchmark.

For each ChEBI concept in ``alias_graph.json`` we generate ONE technical search
query whose *answer is the concept itself* (concept = Aspirin -> a query that
describes aspirin from a gold patent without naming it; answer = "Aspirin"). The
query language is chosen with the same four strategies as the multilingual QAC
pipeline, and the two existing verifiers (faithfulness + technical quality) grade
the single (question, answer) pair with the same criteria.

This module deliberately reuses the verifiers, strategies, and helpers from
``multi_lingual_qac.qac_generation.multilingual_qa`` so the grading is identical;
only the generation step (one concept-centric query, with per-language prompts in
``concept_query_generation_prompts/``) is new.
"""

from __future__ import annotations

import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from src.alias_graph.matching import contains_name
from src.multi_lingual_qac.qac_generation.multilingual_qa import (
    ALL_LANGS,
    DEFAULT_GENERATION_REASONING_EFFORT,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    FAITHFULNESS_FIELDS,
    MODE_TECHNICAL,
    STRATEGY_NAMES,
    STRATEGY_RANDOM_ANY,
    TECHNICAL_QUALITY_FIELDS,
    _build_all_passages_text,
    _compute_faith_overall,
    _compute_quality_overall,
    _compute_total_score,
    _get_client,
    _parse_json_response,
    _pick_context,
    _serialize_context_languages,
    load_multilingual_corpus,
    pick_target_languages,
)

_BASE_DIR = Path(__file__).resolve().parent
_PROMPT_DIR = _BASE_DIR / "concept_query_generation_prompts"
_MAX_GEN_RETRIES = 3  # retry generation on empty/errored response (same document)
_prompt_cache: Dict[str, str] = {}

# Order in which to prefer a language when grounding the answer surface form.
_ANSWER_LANG_ORDER = ["en", "de", "fr", "es", "zh", "chebi"]


_FAITHFULNESS_PROMPT = _BASE_DIR / "faithfulness_prompt" / "faithfulness_prompt.txt"
_QUALITY_PROMPT = _BASE_DIR / "technical_quality_verifier_prompt" / "verifier.txt"


def _load_text(path: Path) -> str:
    key = str(path)
    if key not in _prompt_cache:
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        _prompt_cache[key] = path.read_text(encoding="utf-8").strip()
    return _prompt_cache[key]


def _load_generation_prompt(lang: str) -> str:
    return _load_text(_PROMPT_DIR / f"{lang}.txt")


def grade_faithfulness_single(
    client, all_passages: str, qa: Dict[str, str], *, model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """Faithfulness grade for ONE (question, answer) pair (same rubric, single-object output)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _load_text(_FAITHFULNESS_PROMPT)},
            {"role": "user", "content": f"{all_passages}\n\nQuestion: {qa['question']}\nAnswer: {qa['answer']}"},
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    if isinstance(data, list):
        data = data[0] if data else {}
    row = {
        "grounding": int(data.get("grounding", 1)),
        "precision": int(data.get("precision", 1)),
        "numerical_fidelity": int(data.get("numerical_fidelity", 1)),
        "reason": str(data.get("reason", "")).strip(),
    }
    row["overall"] = _compute_faith_overall(row)
    return row


def grade_quality_single(
    client, all_passages: str, qa: Dict[str, str], *, model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """Technical-quality grade for ONE question (same rubric, single-object output)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _load_text(_QUALITY_PROMPT)},
            {"role": "user", "content": f"{all_passages}\n\nQuestion: {qa['question']}\nAnswer: {qa['answer']}"},
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    if isinstance(data, list):
        data = data[0] if data else {}
    row = {
        "search_bar_realism": int(data.get("search_bar_realism", 1)),
        "specificity": int(data.get("specificity", 1)),
        "phrasing_economy": int(data.get("phrasing_economy", 1)),
        "focus": int(data.get("focus", 1)),
        "linguistic_quality": int(data.get("linguistic_quality", 1)),
        "failure_type": str(data.get("failure_type", "none")).strip(),
        "reason": str(data.get("reason", "")).strip(),
    }
    row["overall"] = _compute_quality_overall(row, MODE_TECHNICAL)
    return row


def load_alias_graph(json_path: Path) -> List[Dict[str, Any]]:
    with Path(json_path).open(encoding="utf-8") as fh:
        return json.load(fh)["concepts"]


def _names_for_lang(name_set: Dict[str, Any], lang: str) -> List[str]:
    value = name_set.get(lang)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _all_aliases(name_set: Dict[str, Any]) -> List[str]:
    seen: Dict[str, None] = {}
    for value in name_set.values():
        names = value if isinstance(value, list) else [value]
        for n in names:
            if n:
                seen.setdefault(n, None)
    return list(seen)


def _pick_answer(
    name_set: Dict[str, Any], target_lang: str, passages: str
) -> Tuple[str, str, bool]:
    """
    Choose the answer (the concept's name). Decision: answer = the concept's name
    in the query language. We use the query-language Wikipedia title when the
    concept has one (preferring a grounded variant), otherwise the **canonical
    ChEBI primary name** (English). We never scan the full ChEBI synonym list,
    because it contains brand names and formulas that are common words (e.g.
    "Action", "CO2") and would be wrongly picked up as grounded.
    Returns (answer, lang, grounded).
    """
    if target_lang != "chebi":
        target_names = _names_for_lang(name_set, target_lang)
        for nm in target_names:
            if contains_name(passages, nm):
                return nm, target_lang, True
        if target_names:
            return target_names[0], target_lang, False

    # No Wikipedia title in the query language: use the ChEBI primary name (the
    # first entry of the chebi bucket is the canonical name, not a synonym).
    chebi_names = _names_for_lang(name_set, "chebi")
    if chebi_names:
        primary = chebi_names[0]
        return primary, "chebi", contains_name(passages, primary)

    # Last resort: any Wikipedia title we have.
    for lang in _ANSWER_LANG_ORDER:
        names = _names_for_lang(name_set, lang)
        if names:
            return names[0], lang, contains_name(passages, names[0])
    return "", "", False


def generate_concept_query(
    client,
    all_passages: str,
    concept_name: str,
    aliases: Sequence[str],
    target_lang: str,
    *,
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """Generate ONE concept-centric technical query in ``target_lang``."""
    prompt = _load_generation_prompt(target_lang)
    alias_block = ", ".join(aliases)
    user = (
        f"CONCEPT (the answer): {concept_name}\n"
        f"ALIASES — never use any of these in the question (any language/spelling): {alias_block}\n\n"
        f"PASSAGES:\n{all_passages}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        reasoning_effort=DEFAULT_GENERATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    if isinstance(data, list):
        data = data[0] if data else {}
    return {
        "question": str(data.get("question", "")).strip(),
        "question_type": str(data.get("question_type", "other")).strip(),
    }


def _select_gold_pub(
    gold: Sequence[str],
    groups: Dict[str, List[Dict[str, Any]]],
    target_lang: str,
    rng: random.Random,
    name_set: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Pick a gold publication to ground generation, preferring one available in
    the target language AND whose passages actually contain a concept name (so the
    model has the concept to describe — guards against any residual false golds)."""
    in_corpus = [p for p in gold if p in groups]
    if not in_corpus:
        return None
    preferred = [
        p for p in in_corpus
        if any(r.get("language") == target_lang for r in groups[p])
    ]
    pool = list(preferred or in_corpus)
    rng.shuffle(pool)
    names = [n for v in (name_set or {}).values() for n in (v if isinstance(v, list) else [v])]
    if names:
        for pub in pool[:30]:
            text = _build_all_passages_text(groups[pub])
            if any(contains_name(text, nm) for nm in names):
                return pub
    return pool[0] if pool else None


def _build_row(
    entry: Dict[str, Any],
    qa: Dict[str, str],
    faith: Dict[str, Any],
    qual: Dict[str, Any],
    *,
    strategy: int,
    corpus_id: str,
    publication_number: str,
    question_language: str,
    answer_language: str,
    context_language: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "chebi_id": entry["chebi_id"],
        "concept_name": entry["name"],
        "mode": MODE_TECHNICAL,
        "strategy": strategy,
        "strategy_name": STRATEGY_NAMES.get(strategy, str(strategy)),
        "corpus_id": corpus_id,
        "publication_number": publication_number,
        "question_language": question_language,
        "context_language": context_language,
        "question": qa["question"],
        "answer": qa["answer"],
        "answer_language": answer_language,
        "question_type": qa.get("question_type", ""),
        "faith_grounding": faith["grounding"],
        "faith_precision": faith["precision"],
        "faith_numerical_fidelity": faith["numerical_fidelity"],
        "faith_overall": faith["overall"],
        "qual_search_bar_realism": qual["search_bar_realism"],
        "qual_specificity": qual["specificity"],
        "qual_phrasing_economy": qual["phrasing_economy"],
        "qual_focus": qual["focus"],
        "qual_linguistic_quality": qual["linguistic_quality"],
        "qual_overall": qual["overall"],
        "qual_failure_type": qual.get("failure_type", "none"),
        "total_score": _compute_total_score(faith, qual, MODE_TECHNICAL),
        "gold_publication_count": entry.get("n_gold", len(entry.get("gold", []))),
    }
    return row


def _process_item(
    client,
    entry: Dict[str, Any],
    doc_rows: List[Dict[str, Any]],
    pub: str,
    target_lang: str,
    name_set: Dict[str, Any],
    aliases: Sequence[str],
    *,
    strategy: int,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Generate + grade ONE (concept, gold publication, target language) item.

    All language variants of *doc_rows* (the same patent in en/de/fr/es/zh) are
    passed to the generator together; the question is written in *target_lang*.
    Returns the output row, or None if no valid question/answer could be formed.
    """
    all_passages = _build_all_passages_text(doc_rows)
    if not all_passages.strip():
        return None
    context_row, _ = _pick_context(doc_rows, target_lang)
    context_languages = _serialize_context_languages(doc_rows)

    answer, answer_lang, _grounded = _pick_answer(name_set, target_lang, all_passages)
    if not answer:
        return None

    # Retry generation on the same document if the model doesn't respond
    # (empty question) or errors, up to _MAX_GEN_RETRIES attempts.
    gen = None
    for attempt in range(1, _MAX_GEN_RETRIES + 1):
        try:
            candidate = generate_concept_query(
                client, all_passages, entry["name"], aliases, target_lang, model=model
            )
        except Exception as exc:
            tqdm.write(
                f"  {entry['chebi_id']} [{target_lang}]: generation error "
                f"(attempt {attempt}/{_MAX_GEN_RETRIES}): {exc}"
            )
            continue
        if candidate["question"]:
            gen = candidate
            break
    if gen is None:
        tqdm.write(
            f"  {entry['chebi_id']} [{target_lang}]: no question after "
            f"{_MAX_GEN_RETRIES} attempts; skipped"
        )
        return None

    qa_pair = {"question": gen["question"], "answer": answer}
    try:
        faith = grade_faithfulness_single(client, all_passages, qa_pair, model=model)
        qual = grade_quality_single(client, all_passages, qa_pair, model=model)
    except Exception as exc:
        tqdm.write(f"  {entry['chebi_id']} [{target_lang}]: grading error: {exc}")
        return None

    row = _build_row(
        entry,
        {"question": gen["question"], "answer": answer, "question_type": gen["question_type"]},
        faith,
        qual,
        strategy=strategy,
        corpus_id=context_row.get("id", ""),
        publication_number=pub,
        question_language=target_lang,
        answer_language=answer_lang,
        context_language=context_languages,
    )
    tqdm.write(
        f"  {entry['chebi_id']} ({entry['name']}) [{target_lang}]: "
        f"ok (total={row['total_score']})"
    )
    return row


def _process_concept(
    entry: Dict[str, Any],
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    strategy: int,
    model: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(f"{seed}:{entry['chebi_id']}")
    available_langs = entry.get("gold_langs", [])
    target_langs = pick_target_languages(strategy, available_langs)
    name_set = entry.get("name_set", {})
    aliases = _all_aliases(name_set)
    client = _get_client()
    rows: List[Dict[str, Any]] = []

    for target_lang in target_langs:
        pub = _select_gold_pub(entry.get("gold", []), groups, target_lang, rng, name_set)
        if pub is None:
            continue
        row = _process_item(
            client, entry, groups[pub], pub, target_lang, name_set, aliases,
            strategy=strategy, model=model,
        )
        if row is not None:
            rows.append(row)

    return rows


def _build_balanced_plan(
    entries: List[Dict[str, Any]],
    groups: Dict[str, List[Dict[str, Any]]],
    per_lang: int,
    rng: random.Random,
) -> List[Tuple[Dict[str, Any], str, str]]:
    """Build a language-balanced work list of (concept, gold pub, target language).

    Each language L in ``ALL_LANGS`` gets up to ``per_lang`` items, each grounded
    on a DISTINCT publication that (a) has a row in language L and (b) whose
    passages contain the concept name (same grounding check as ``_select_gold_pub``).
    Distinct concepts are preferred within a language; concept reuse is allowed
    only in a relaxed second pass when a language would otherwise fall short. If a
    language still cannot reach ``per_lang`` (e.g. zh, with few gold docs), it is
    capped at what's available and a warning is printed.
    """
    langs = list(ALL_LANGS)
    selected: Dict[str, List[Tuple[Dict[str, Any], str]]] = {L: [] for L in langs}
    used_pubs: Dict[str, Set[str]] = {L: set() for L in langs}
    used_concepts: Dict[str, Set[str]] = {L: set() for L in langs}
    # pub -> (combined passages text, set of ALL_LANGS present in that pub)
    pub_cache: Dict[str, Tuple[str, Set[str]]] = {}

    order = list(entries)
    rng.shuffle(order)

    def fill(allow_concept_reuse: bool) -> None:
        for entry in order:
            if all(len(selected[L]) >= per_lang for L in langs):
                return
            names = _all_aliases(entry.get("name_set", {}))
            if not names:
                continue
            cid = entry["chebi_id"]
            gold = [p for p in entry.get("gold", []) if p in groups]
            rng.shuffle(gold)
            for p in gold:
                if p not in pub_cache:
                    rows = groups[p]
                    text = _build_all_passages_text(rows)
                    present = {r.get("language") for r in rows} & set(langs)
                    pub_cache[p] = (text, present)
                text, present = pub_cache[p]
                if not present or not any(contains_name(text, nm) for nm in names):
                    continue
                for L in present:
                    if len(selected[L]) >= per_lang:
                        continue
                    if p in used_pubs[L]:
                        continue
                    if not allow_concept_reuse and cid in used_concepts[L]:
                        continue
                    selected[L].append((entry, p))
                    used_pubs[L].add(p)
                    used_concepts[L].add(cid)

    fill(allow_concept_reuse=False)
    if any(len(selected[L]) < per_lang for L in langs):
        fill(allow_concept_reuse=True)

    plan: List[Tuple[Dict[str, Any], str, str]] = []
    for L in langs:
        n = len(selected[L])
        if n < per_lang:
            print(f"  [balance] {L}: only {n}/{per_lang} eligible documents (capped + warned)")
        for entry, p in selected[L]:
            plan.append((entry, p, L))
    return plan


def _generate_per_concept(
    entries: List[Dict[str, Any]],
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    strategy: int,
    model: str,
    seed: int,
    workers: int,
) -> List[Dict[str, Any]]:
    """Default mode: one query per (concept, strategy-chosen language)."""
    print(
        f"Concept-query QA generation: {len(entries)} concepts, "
        f"strategy={STRATEGY_NAMES.get(strategy, strategy)}, model={model}, workers={workers}"
    )
    results: List[List[Dict[str, Any]]] = []
    if workers > 1 and entries:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_concept, entry, groups, strategy=strategy, model=model, seed=seed
                ): entry
                for entry in entries
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Concept Q&A", unit="concept"):
                results.append(future.result())
    else:
        for entry in tqdm(entries, desc="Concept Q&A", unit="concept"):
            results.append(_process_concept(entry, groups, strategy=strategy, model=model, seed=seed))
    return [row for group in results for row in group]


def _generate_balanced(
    entries: List[Dict[str, Any]],
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    strategy: int,
    model: str,
    seed: int,
    total: int,
    workers: int,
) -> List[Dict[str, Any]]:
    """Balanced mode: ``total`` queries split equally across ``ALL_LANGS``, each
    grounded on a document that exists in its query language."""
    langs = list(ALL_LANGS)
    per_lang = total // len(langs)
    if per_lang == 0:
        raise ValueError(f"--alias-qa-total {total} is below the number of languages ({len(langs)}).")
    if total % len(langs) != 0:
        print(
            f"  [balance] total={total} not divisible by {len(langs)} languages; "
            f"using {per_lang}/language ({per_lang * len(langs)} total)"
        )
    plan = _build_balanced_plan(entries, groups, per_lang, random.Random(seed))
    print(
        f"Concept-query QA (balanced): {len(plan)} items, target {per_lang}/language across "
        f"{', '.join(langs)}; strategy={STRATEGY_NAMES.get(strategy, strategy)}, model={model}, workers={workers}"
    )

    client = _get_client()

    def work(item: Tuple[Dict[str, Any], str, str]) -> Optional[Dict[str, Any]]:
        entry, pub, target_lang = item
        name_set = entry.get("name_set", {})
        return _process_item(
            client, entry, groups[pub], pub, target_lang, name_set, _all_aliases(name_set),
            strategy=strategy, model=model,
        )

    rows: List[Dict[str, Any]] = []
    if workers > 1 and plan:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(work, item) for item in plan]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Concept Q&A (balanced)", unit="item"):
                row = future.result()
                if row is not None:
                    rows.append(row)
    else:
        for item in tqdm(plan, desc="Concept Q&A (balanced)", unit="item"):
            row = work(item)
            if row is not None:
                rows.append(row)
    return rows


def _output_fieldnames() -> List[str]:
    return [
        "chebi_id", "concept_name", "mode", "strategy", "strategy_name",
        "corpus_id", "publication_number", "question_language", "context_language",
        "question", "answer", "answer_language", "question_type",
        *FAITHFULNESS_FIELDS, *TECHNICAL_QUALITY_FIELDS,
        "qual_failure_type", "total_score", "gold_publication_count",
    ]


def run_concept_qa(
    alias_json: Path,
    corpus_path: Path,
    output_path: Path,
    *,
    strategy: int = STRATEGY_RANDOM_ANY,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    limit: Optional[int] = None,
    workers: int = 1,
    total: Optional[int] = None,
) -> int:
    """Generate concept-centric technical queries and write them to ``output_path``.

    When ``total`` is given, run in balanced mode (``total`` queries split equally
    across ``ALL_LANGS``, grounded on documents that exist in each query language);
    otherwise generate one query per (concept, strategy-chosen language).
    """
    entries = load_alias_graph(alias_json)
    if limit is not None:
        entries = entries[:limit]
    groups = load_multilingual_corpus(corpus_path)
    random.seed(seed)

    if total is not None:
        all_rows = _generate_balanced(
            entries, groups, strategy=strategy, model=model, seed=seed, total=total, workers=workers
        )
    else:
        all_rows = _generate_per_concept(
            entries, groups, strategy=strategy, model=model, seed=seed, workers=workers
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_output_fieldnames())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} concept-query rows -> {output_path}")
    return len(all_rows)
