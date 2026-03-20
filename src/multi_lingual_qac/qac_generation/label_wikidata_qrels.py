"""
Label multilingual retrieval qrels for Wikidata/Wikipedia chunks.

For each row in qac.csv, finds all other corpus chunks with the same Wikidata
QID (any language, including other chunks in the query language) and uses a
cheap OpenAI model to judge which passages can answer the question.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from src.multi_lingual_qac.qac_generation.openai_qa import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_REASONING_EFFORT,
    LANG_NAMES,
    _get_client,
    _parse_json_response,
)

# How many passages to send per judge call (balances cost vs context size).
DEFAULT_BATCH_SIZE = 5
# Truncate long chunks for the judge prompt.
MAX_PASSAGE_CHARS = 1800


def _load_corpus_full(path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    """Return (by_id, by_qid)."""
    by_id: Dict[str, Dict[str, str]] = {}
    by_qid: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = row.get("id", "").strip()
            qid = row.get("qid", "").strip()
            if not cid or not qid:
                continue
            by_id[cid] = row
            by_qid[qid].append(row)
    return by_id, dict(by_qid)


def _load_qac(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _batched(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _judge_batch(
    client: Any,
    *,
    question: str,
    question_lang: str,
    passage_lang: str,
    numbered_passages: List[Tuple[int, str, str]],
    model: str,
) -> Set[int]:
    """
    numbered_passages: (display_index, corpus_id, text)
    Returns set of display_index values judged relevant.
    """
    if not numbered_passages:
        return set()

    q_lang_name = LANG_NAMES.get(question_lang.lower(), question_lang)
    p_lang_name = LANG_NAMES.get(passage_lang.lower(), passage_lang)

    lines = []
    for idx, _cid, text in numbered_passages:
        snippet = (text or "")[:MAX_PASSAGE_CHARS]
        if len(text or "") > MAX_PASSAGE_CHARS:
            snippet += " [...]"
        lines.append(f"[{idx}] {snippet}")

    system = """You are a strict relevance judge for multilingual retrieval benchmarks.

Your job: decide which numbered passages could help a user answer the question.
A passage is relevant if it states or clearly implies the information needed to answer,
even if the passage is not in the same language as the question.

Rules:
- Mark a passage as relevant only if it actually supports answering the question (not just same topic).
- If none apply, return an empty list.
- Output valid JSON only, no markdown:
  {"relevant": [1, 3]}
Use 1-based indices exactly as shown in brackets [1], [2], ...
"""

    user = f"""Question language: {q_lang_name} ({question_lang})
Passage language: {p_lang_name} ({passage_lang})

Question:
{question}

Numbered passages (same Wikidata entity, possibly different section/chunk):
{chr(10).join(lines)}

Which numbers are relevant? JSON only."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
        raw = response.choices[0].message.content or ""
        data = _parse_json_response(raw)
        rel = data.get("relevant", [])
        if not isinstance(rel, list):
            return set()
        out: Set[int] = set()
        for x in rel:
            try:
                out.add(int(x))
            except (TypeError, ValueError):
                continue
        return out
    except Exception:
        return set()


def run_wikidata_qrels_labeling(
    *,
    corpus_full_path: Path,
    qac_path: Path,
    output_dir: Path,
    model: str = DEFAULT_GENERATION_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, int]:
    """
    Write queries.csv and qrels.csv under output_dir.

    queries columns: _id, text, language, source_corpus_id, qid
    qrels columns: query-id, corpus-id, score  (score 1.0 for each relevant doc)

    Returns simple stats dict.
    """
    corpus_full_path = Path(corpus_full_path)
    qac_path = Path(qac_path)
    output_dir = Path(output_dir)

    if not corpus_full_path.is_file():
        raise FileNotFoundError(f"Corpus full CSV not found: {corpus_full_path}")
    if not qac_path.is_file():
        raise FileNotFoundError(f"QAC CSV not found: {qac_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    by_id, by_qid = _load_corpus_full(corpus_full_path)
    qac_rows = _load_qac(qac_path)
    if not qac_rows:
        raise ValueError(f"No rows in {qac_path}")

    client = _get_client()

    queries_out: List[Dict[str, str]] = []
    qrels_out: List[Dict[str, Any]] = []
    stats = {
        "qac_rows": len(qac_rows),
        "queries_written": 0,
        "qrels_written": 0,
        "judge_api_calls": 0,
    }

    for qix, qrow in enumerate(tqdm(qac_rows, desc="Label qrels", unit="query")):
        source_cid = qrow.get("corpus_id", "")
        question = qrow.get("question", "")
        q_lang = (qrow.get("language") or "en").lower()

        if not source_cid or not question:
            continue
        if source_cid not in by_id:
            tqdm.write(f"  Skip: corpus_id not in corpus: {source_cid}")
            continue

        meta = by_id[source_cid]
        qid = (meta.get("qid") or "").strip()
        if not qid:
            tqdm.write(f"  Skip: missing qid for {source_cid}")
            continue

        query_id = f"{source_cid}_q_{q_lang}"
        if sum(1 for r in qac_rows[:qix] if r.get("corpus_id") == source_cid and r.get("language") == q_lang) > 0:
            query_id = f"{source_cid}_q_{q_lang}_{qix}"

        queries_out.append(
            {
                "_id": query_id,
                "text": question,
                "language": q_lang,
                "source_corpus_id": source_cid,
                "qid": qid,
            }
        )

        relevant_ids: Set[str] = {source_cid}
        siblings = [r for r in by_qid.get(qid, []) if r.get("id") != source_cid]

        by_passage_lang: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in siblings:
            lang = (row.get("language") or "").lower()
            if lang:
                by_passage_lang[lang].append(row)

        for passage_lang, rows in sorted(by_passage_lang.items(), key=lambda x: x[0]):
            batches = _batched(rows, batch_size)
            for batch in batches:
                numbered: List[Tuple[int, str, str]] = []
                index_to_cid: Dict[int, str] = {}
                for local_i, row in enumerate(batch, start=1):
                    cid = row.get("id", "")
                    text = row.get("context", "") or row.get("text", "") or row.get("abstract", "")
                    numbered.append((local_i, cid, text))
                    index_to_cid[local_i] = cid

                stats["judge_api_calls"] += 1
                good_indices = _judge_batch(
                    client,
                    question=question,
                    question_lang=q_lang,
                    passage_lang=passage_lang,
                    numbered_passages=numbered,
                    model=model,
                )
                for idx in good_indices:
                    cid_hit = index_to_cid.get(idx)
                    if cid_hit:
                        relevant_ids.add(cid_hit)
                time.sleep(0.05)

        for cid_rel in sorted(relevant_ids):
            qrels_out.append(
                {
                    "query-id": query_id,
                    "corpus-id": cid_rel,
                    "score": 1.0,
                }
            )

        stats["queries_written"] += 1
        stats["qrels_written"] += len(relevant_ids)

    queries_path = output_dir / "queries.csv"
    qrels_path = output_dir / "qrels.csv"
    stats_path = output_dir / "qrels_label_stats.json"

    with queries_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["_id", "text", "language", "source_corpus_id", "qid"],
        )
        w.writeheader()
        w.writerows(queries_out)

    with qrels_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["query-id", "corpus-id", "score"])
        w.writeheader()
        w.writerows(qrels_out)

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote {stats['queries_written']} queries -> {queries_path}")
    print(f"Wrote {stats['qrels_written']} qrel rows -> {qrels_path}")
    print(f"Judge API calls: {stats['judge_api_calls']} (model={model})")
    return stats
