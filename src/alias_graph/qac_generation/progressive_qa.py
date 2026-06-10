"""
Query generation for the progressive (cumulative-ladder) code-switched variants.

Each base document produces ONE fixed query, about the concept whose term is
swapped at the **first** rung of the ladder (e.g. "carbon dioxide"). That query is
reused as the query for every depth 0..N of the ladder; the gold for the query is
the variant document at each depth, so the eval can measure how retrieval of the
(increasingly code-switched) document decays while the query stays constant.

Generation is **identical to the Alias-Graph concept-query pipeline**
(``concept_qa``): the exact same per-language prompt
(``concept_query_generation_prompts/<lang>.txt``), the same "describe the concept,
never name it or its aliases" contract, the same answer selection (``_pick_answer``)
and the same faithfulness + technical-quality verifiers. As in the alias graph, the
document is passed in **all of its available languages** at once
(``_build_all_passages_text`` over the full multilingual group for the
publication); only the query language differs — here it is the ladder's anchor
language (the language the variant documents are written in).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.alias_graph.qac_generation.concept_qa import (
    _all_aliases,
    _pick_answer,
    generate_concept_query,
    grade_faithfulness_single,
    grade_quality_single,
)
from src.alias_graph.qac_generation.variant_qa import _grade_fields
from src.multi_lingual_qac.qac_generation.multilingual_qa import (
    DEFAULT_MODEL,
    FAITHFULNESS_FIELDS,
    TECHNICAL_QUALITY_FIELDS,
    _build_all_passages_text,
    _get_client,
    load_multilingual_corpus,
)

_MAX_GEN_RETRIES = 3  # same retry budget as the alias-graph concept-query pipeline

OUTPUT_FIELDS: List[str] = [
    "base_id", "n_replacements", "concept_chebi_id", "concept_name", "query_language",
    "term_used", "question", "answer", "question_type",
    *FAITHFULNESS_FIELDS, *TECHNICAL_QUALITY_FIELDS, "qual_failure_type", "total_score",
    "gold_id", "source_id",
]


def _process_base(base: Dict[str, Any], groups: Dict[str, List[dict]],
                  name_set_by_cid: Dict[str, dict], model: str) -> List[Dict[str, Any]]:
    """One fixed concept-query per base doc, reused across its ladder of variants.

    The query is generated exactly like the alias graph: all language versions of
    the source publication are passed together as passages, and the prompt is told
    to describe the concept without ever naming it or its aliases.
    """
    doc_rows = groups.get(base["publication_number"])
    if not doc_rows:
        return []
    all_passages = _build_all_passages_text(doc_rows)
    if not all_passages.strip():
        return []
    client = _get_client()
    lang = base["query_language"]  # the ladder's anchor language
    cname = base["concept_name"]
    name_set = name_set_by_cid.get(base["concept_chebi_id"], {})
    aliases = _all_aliases(name_set)

    gen = None
    for _ in range(_MAX_GEN_RETRIES):
        try:
            cand = generate_concept_query(client, all_passages, cname, aliases, lang, model=model)
        except Exception as exc:
            tqdm.write(f"  {base['concept_chebi_id']} [{lang}]: generation error: {exc}")
            continue
        if cand["question"]:
            gen = cand
            break
    if gen is None:
        return []

    answer, _alang, _ground = _pick_answer(name_set, lang, all_passages)
    if not answer:
        return []
    qa = {"question": gen["question"], "answer": answer}
    faith = grade_faithfulness_single(client, all_passages, qa, model=model)
    qual = grade_quality_single(client, all_passages, qa, model=model)
    fields = _grade_fields(faith, qual)

    rows: List[Dict[str, Any]] = []
    for depth, gold_id in sorted(base["variants"]):
        rows.append({
            "base_id": base["source_id"], "n_replacements": depth,
            "concept_chebi_id": base["concept_chebi_id"], "concept_name": cname,
            "query_language": lang, "term_used": base["original_term"],
            "question": gen["question"], "answer": answer,
            "question_type": gen["question_type"], **fields,
            "gold_id": gold_id, "source_id": base["source_id"],
        })
    return rows


def run_progressive_qa(
    corpus_csv: Path,
    source_corpus: Path,
    alias_json: Path,
    output_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    limit: Optional[int] = None,
    workers: int = 1,
) -> int:
    """Generate the fixed per-base concept-query for the progressive variants.
    Returns rows written (one per (base, depth))."""
    from src.alias_graph.builder import _read_corpus

    rows = _read_corpus(corpus_csv)
    groups = load_multilingual_corpus(source_corpus)
    with Path(alias_json).open(encoding="utf-8") as fh:
        name_set_by_cid = {c["chebi_id"]: c.get("name_set", {}) for c in json.load(fh)["concepts"]}

    # Group ladder rows by base document; collect (depth, gold variant id).
    bases: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        base_id = r["base_id"]
        b = bases.setdefault(base_id, {
            "source_id": base_id,
            "publication_number": r["source_publication_number"],
            "concept_chebi_id": r["question_concept_chebi_id"],
            "concept_name": r["question_concept_name"],
            "query_language": r["anchor_language"],
            "original_term": r["question_original_term"],
            "variants": [],
        })
        b["variants"].append((int(r["n_replacements"]), r["id"]))

    base_list = list(bases.values())
    if limit is not None:
        base_list = base_list[:limit]
    print(f"Progressive QA: {len(base_list)} base docs, model={model}, workers={workers}")

    out_rows: List[Dict[str, Any]] = []

    def run_job(b):
        return _process_base(b, groups, name_set_by_cid, model)

    if workers > 1 and base_list:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_job, b) for b in base_list]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Progressive QA", unit="base"):
                out_rows.extend(fut.result())
    else:
        for b in tqdm(base_list, desc="Progressive QA", unit="base"):
            out_rows.extend(run_job(b))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    n_bases = len({r["base_id"] for r in out_rows})
    print(f"\nWrote {len(out_rows)} progressive-QA rows ({n_bases} questions) -> {output_path}")
    return len(out_rows)
