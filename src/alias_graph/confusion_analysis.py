"""
"How often does a confusable *wrong* compound beat the right one?" — per language.

This is the analysis the alias graph was built for. For each query (about a
concept, in some language) the benchmark provides gold documents (the right
compound) and hard-negative documents (chemically-similar look-alike compounds,
each labelled with its neighbour concept + relation). We embed queries and the
corpus with an embedding model and ask, per query: does a look-alike document
score higher than every gold document? Aggregated per query language, that is the
confusion rate.

The published dataset already carries everything: `qrels` (gold = score 1,
look-alike = score 0) and a `hard_negatives` config (corpus-id ->
neighbour_chebi_id / neighbour_name / relation). Embedding-model loading + cache
are reused from `multi_lingual_qac.mteb.evaluation`.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset

from src.multi_lingual_qac.mteb.evaluation import (
    DEFAULT_MTEB_MODELS,
    _configure_local_model_cache,
)

DEFAULT_DATASET = "MehdiAstaraki/multi-lingual-qac-alias-graph"
_CONFIGS = ("corpus", "queries", "qrels", "hard_negatives")


def _load_config(dataset: str, config: str):
    """Load one config from a local hf_export dir or from the Hugging Face Hub."""
    path = Path(dataset)
    if path.is_dir():
        return load_dataset(
            "parquet", data_files=str(path / config / f"{config}.parquet"), split="train"
        )
    return load_dataset(dataset, config, split="train")


def _encode(model, texts: Sequence[str], *, prefix: str, batch_size: int) -> np.ndarray:
    payload = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
    return model.encode(
        payload, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )


def run_confusion_analysis(
    dataset: str = DEFAULT_DATASET,
    output_dir: Path = Path("reports/confusion_analysis"),
    *,
    models: Optional[Sequence[str]] = None,
    batch_size: int = 32,
    query_limit: Optional[int] = None,
) -> Path:
    from sentence_transformers import SentenceTransformer

    models = list(models) if models else list(DEFAULT_MTEB_MODELS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _configure_local_model_cache()

    # --- load dataset ---
    corpus = _load_config(dataset, "corpus")
    queries = _load_config(dataset, "queries")
    qrels = _load_config(dataset, "qrels")
    hard_neg = _load_config(dataset, "hard_negatives")

    corpus_ids = list(corpus["_id"])
    corpus_pos = {cid: i for i, cid in enumerate(corpus_ids)}
    corpus_text = [
        ((t or "") + " " + (x or "")).strip()
        for t, x in zip(corpus["title"], corpus["text"])
    ]

    # qrels: query-id -> (gold positions, hardneg positions)
    gold_by_q: Dict[str, List[int]] = defaultdict(list)
    hardneg_by_q: Dict[str, List[int]] = defaultdict(list)
    for r in qrels:
        qid, cid, score = r["query-id"], r["corpus-id"], float(r["score"])
        pos = corpus_pos.get(cid)
        if pos is None:
            continue
        (gold_by_q if score > 0 else hardneg_by_q)[qid].append(pos)

    # hard_negatives: (query-id, corpus-id) -> neighbour label
    neighbor: Dict[tuple, Dict[str, str]] = {}
    for r in hard_neg:
        neighbor[(r["query-id"], r["corpus-id"])] = {
            "chebi_id": r.get("neighbor_chebi_id", ""),
            "name": r.get("neighbor_name", ""),
            "relation": r.get("relation", ""),
        }

    q_rows = list(queries)
    if query_limit is not None:
        q_rows = q_rows[:query_limit]
    # keep only queries that have both gold and look-alike docs in the corpus
    q_rows = [
        q for q in q_rows
        if gold_by_q.get(q["_id"]) and hardneg_by_q.get(q["_id"])
    ]
    print(f"Confusion analysis: {len(q_rows)} queries, {len(corpus_ids)} docs, models={models}")

    per_query_rows: List[dict] = []

    for model_name in models:
        print(f"\nEncoding with `{model_name}` ...")
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        is_e5 = "e5" in model_name.lower()
        doc_emb = _encode(model, corpus_text, prefix="passage: " if is_e5 else "", batch_size=batch_size)
        q_emb = _encode(model, [q["text"] for q in q_rows], prefix="query: " if is_e5 else "", batch_size=batch_size)

        for i, q in enumerate(q_rows):
            qid = q["_id"]
            sims = doc_emb @ q_emb[i]
            gold_idx = np.array(gold_by_q[qid])
            hn_idx = np.array(hardneg_by_q[qid])
            best_gold = float(sims[gold_idx].max())
            hn_local = int(hn_idx[int(np.argmax(sims[hn_idx]))])
            best_hn = float(sims[hn_local])
            win = best_hn > best_gold
            lab = neighbor.get((qid, corpus_ids[hn_local]), {})
            per_query_rows.append({
                "model": model_name, "query_id": qid,
                "query_language": q.get("query_language", ""),
                "chebi_id": q.get("chebi_id", ""), "concept_name": q.get("concept_name", ""),
                "n_gold": len(gold_idx), "n_hardneg": len(hn_idx),
                "best_gold_rank": int((sims > best_gold).sum()) + 1,
                "best_hardneg_rank": int((sims > best_hn).sum()) + 1,
                "max_gold_sim": round(best_gold, 4), "max_hardneg_sim": round(best_hn, 4),
                "margin": round(best_hn - best_gold, 4), "win": int(win),
                "top_neighbor_chebi_id": lab.get("chebi_id", "") if win else "",
                "top_neighbor_name": lab.get("name", "") if win else "",
                "top_relation": lab.get("relation", "") if win else "",
            })

    _write_outputs(output_dir, per_query_rows, models)
    return output_dir


def _write_outputs(output_dir: Path, per_query: List[dict], models: Sequence[str]) -> None:
    pq_fields = list(per_query[0].keys()) if per_query else []
    with (output_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pq_fields)
        w.writeheader()
        w.writerows(per_query)

    # Aggregate per (model, language) + an ALL row.
    agg: Dict[tuple, List[dict]] = defaultdict(list)
    for r in per_query:
        agg[(r["model"], r["query_language"])].append(r)
        agg[(r["model"], "ALL")].append(r)

    rows = []
    for (model, lang), items in sorted(agg.items()):
        n = len(items)
        wins = sum(r["win"] for r in items)
        rows.append({
            "model": model, "query_language": lang, "n_queries": n,
            "confusion_rate": round(wins / n, 4) if n else 0.0,
            "n_wins": wins,
            "mean_best_gold_rank": round(sum(r["best_gold_rank"] for r in items) / n, 2),
            "mean_best_hardneg_rank": round(sum(r["best_hardneg_rank"] for r in items) / n, 2),
        })
    with (output_dir / "confusion_by_language.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["model", "query_language", "n_queries", "confusion_rate", "n_wins",
                            "mean_best_gold_rank", "mean_best_hardneg_rank"])
        w.writeheader()
        w.writerows(rows)

    # Markdown summary: per-language confusion rate per model + worst confusions.
    langs = sorted({r["query_language"] for r in per_query if r["query_language"]})
    by = {(r["model"], r["query_language"]): r for r in rows}
    lines = ["# Confusion analysis — does a wrong (look-alike) compound beat the right one?", "",
             "Confusion rate = fraction of queries where a hard-negative (chemically-similar "
             "wrong compound) scores higher than every gold document.", "",
             "| model | " + " | ".join(langs) + " | ALL |",
             "| --- | " + " | ".join(["---:"] * (len(langs) + 1)) + " |"]
    for model in models:
        cells = []
        for lang in langs + ["ALL"]:
            r = by.get((model, lang))
            cells.append(f"{r['confusion_rate']:.1%} (n={r['n_queries']})" if r else "—")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines += ["", "## Most frequent confusions (winning look-alike, all models)", ""]
    conf = Counter(
        (r["concept_name"], r["top_neighbor_name"] or r["top_neighbor_chebi_id"], r["top_relation"])
        for r in per_query if r["win"]
    )
    if conf:
        lines += ["| right compound | beaten by (look-alike) | relation | count |",
                  "| --- | --- | --- | ---: |"]
        for (right, wrong, rel), c in conf.most_common(30):
            lines.append(f"| {right} | {wrong} | {rel} | {c} |")
    else:
        lines.append("_No confusions (no look-alike outranked the gold)._")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote confusion analysis -> {output_dir}")
    print(f"  {output_dir / 'confusion_by_language.csv'}")
