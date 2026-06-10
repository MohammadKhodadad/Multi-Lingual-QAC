"""
QA generation for the progressive (cumulative-ladder) code-switched variants.

Each base document produces ONE fixed question, about the **step-1 term** — the
term that is swapped out at the first rung of the ladder (e.g. "carbon dioxide").
The question uses that term verbatim and is reused as the query for every depth
0..N of the ladder; the gold for the query is the variant document at each depth.
This lets the eval measure how retrieval of the (increasingly code-switched)
document decays as the dose grows, while the query stays constant.

Reuses the term-query generator + single-query graders from the single-swap
variant pipeline (``variant_qa`` / ``concept_qa``); the only new logic is grouping
by base document and emitting one row per (query, depth, gold variant).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.alias_graph.builder import _read_corpus
from src.alias_graph.qac_generation.concept_qa import (
    _pick_answer,
    grade_faithfulness_single,
    grade_quality_single,
)
from src.alias_graph.qac_generation.variant_qa import (
    _grade_fields,
    generate_term_query,
)
from src.multi_lingual_qac.qac_generation.multilingual_qa import (
    DEFAULT_MODEL,
    FAITHFULNESS_FIELDS,
    TECHNICAL_QUALITY_FIELDS,
    _build_all_passages_text,
    _get_client,
)

OUTPUT_FIELDS: List[str] = [
    "base_id", "n_replacements", "concept_chebi_id", "concept_name", "query_language",
    "term_used", "question", "answer", "question_type",
    *FAITHFULNESS_FIELDS, *TECHNICAL_QUALITY_FIELDS, "qual_failure_type", "total_score",
    "gold_id", "source_id",
]


def _process_base(base: Dict[str, Any], source_by_id: Dict[str, dict],
                  name_set_by_cid: Dict[str, dict], model: str) -> List[Dict[str, Any]]:
    """One fixed term-query per base doc, reused across its ladder of variant docs."""
    src = source_by_id.get(base["source_id"])
    if src is None:
        return []
    all_passages = _build_all_passages_text([src])
    if not all_passages.strip():
        return []
    client = _get_client()
    lang = base["query_language"]
    term = base["term_used"]
    cname = base["concept_name"]
    gen = generate_term_query(client, all_passages, cname, term, lang, model=model)
    if gen is None:
        return []
    name_set = name_set_by_cid.get(base["concept_chebi_id"], {})
    answer, _alang, _ground = _pick_answer(name_set, lang, all_passages)
    qa = {"question": gen["question"], "answer": answer}
    faith = grade_faithfulness_single(client, all_passages, qa, model=model)
    qual = grade_quality_single(client, all_passages, qa, model=model)
    fields = _grade_fields(faith, qual)
    rows: List[Dict[str, Any]] = []
    for depth, gold_id in sorted(base["variants"]):
        rows.append({
            "base_id": base["source_id"], "n_replacements": depth,
            "concept_chebi_id": base["concept_chebi_id"], "concept_name": cname,
            "query_language": lang, "term_used": term, "question": gen["question"],
            "answer": answer, "question_type": gen["question_type"], **fields,
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
    """Generate the fixed per-base question for the progressive variants. Returns
    rows written (one per (base, depth))."""
    rows = _read_corpus(corpus_csv)
    source_by_id = {r["id"]: r for r in _read_corpus(source_corpus)}
    with Path(alias_json).open(encoding="utf-8") as fh:
        name_set_by_cid = {c["chebi_id"]: c.get("name_set", {}) for c in json.load(fh)["concepts"]}

    # Group ladder rows by base document; collect (depth, gold variant id).
    bases: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        base_id = r["base_id"]
        b = bases.setdefault(base_id, {
            "source_id": base_id,
            "concept_chebi_id": r["question_concept_chebi_id"],
            "concept_name": r["question_concept_name"],
            "query_language": r["anchor_language"],
            "term_used": r["question_original_term"],
            "variants": [],
        })
        b["variants"].append((int(r["n_replacements"]), r["id"]))

    base_list = list(bases.values())
    if limit is not None:
        base_list = base_list[:limit]
    print(f"Progressive QA: {len(base_list)} base docs, model={model}, workers={workers}")

    out_rows: List[Dict[str, Any]] = []

    def run_job(b):
        return _process_base(b, source_by_id, name_set_by_cid, model)

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
