#!/usr/bin/env python3
"""Re-score the SAVED alias-graph predictions under two relevance interpretations and write
them as separate result sets under reports/runs/alias_graph/interpretations/.

No models are re-run -- this only re-defines what counts as "relevant" and re-scores the
existing per-query rankings (reports/runs/alias_graph/predictions/<model>/...).

Two lenses (the dataset's queries are keyed by CONCEPT+language, e.g. CHEBI_15138__en):

  concept_level   (the benchmark as shipped): a query's gold = EVERY document of EVERY
                  publication that attests the concept (~109 docs/query, spanning ~50
                  publications). "Find all patents about compound X."

  per_publication (the ~2.3-docs-per-query reading): each (query, gold-publication) pair is
                  a separate retrieval unit whose gold = just that publication's language
                  variants (~2.3 docs). The query's OTHER gold publications are removed from
                  the candidate ranking so the model is scored on retrieving THIS patent's
                  translations above the hard-negatives/unrelated docs, not penalised for
                  surfacing sibling gold. "Find this patent's cross-language versions."

Run on a login node (needs internet to read the dataset qrels/corpus configs):
    .venv/bin/python scripts/alias_graph_interpretations.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytrec_eval
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_lingual_qac.mteb.question_analysis import _discover_models, _load_predictions  # noqa: E402

DATASET_REPO = "MehdiAstaraki/multi-lingual-qac-alias-graph"
RUN_DIR = PROJECT_ROOT / "reports" / "runs" / "alias_graph"
PRED_DIR = RUN_DIR / "predictions"
OUT_DIR = RUN_DIR / "interpretations"

# pytrec_eval measure -> output key
MEASURES = {"map", "recip_rank", "ndcg_cut.10", "recall.10", "recall.100", "P.10"}
OUT_KEYS = ["recall_10", "recall_100", "P_10", "ndcg_cut_10", "recip_rank", "map"]
PRETTY = {
    "recall_10": "Recall@10", "recall_100": "Recall@100", "P_10": "Precision@10",
    "ndcg_cut_10": "nDCG@10", "recip_rank": "MRR", "map": "MAP",
}


def _pub_of(doc_id: str, pub_lookup: dict[str, str]) -> str:
    """Publication of a doc id. Prefer the corpus column; else strip the `_<lang>` suffix
    (ids look like `WO-2025202541-A1_en`)."""
    if doc_id in pub_lookup:
        return pub_lookup[doc_id]
    return doc_id.rsplit("_", 1)[0]


def load_gold_and_pubs():
    """per-query gold docs (score>0) and a doc_id -> publication_number lookup for them."""
    qrels = load_dataset(DATASET_REPO, "qrels", split="train")
    corpus = load_dataset(DATASET_REPO, "corpus", split="train")
    cid = "_id" if "_id" in corpus.column_names else "corpus-id"
    pub_lookup = {str(r[cid]): str(r["publication_number"]) for r in corpus} if "publication_number" in corpus.column_names else {}
    gold: dict[str, set[str]] = defaultdict(set)
    for r in qrels:
        if float(r["score"]) > 0:
            gold[str(r["query-id"])].add(str(r["corpus-id"]))
    return gold, pub_lookup


def avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def score_concept(preds, gold):
    """Lens 1: gold = all concept docs. One eval unit per query."""
    qrels = {q: {d: 1 for d in docs} for q, docs in gold.items() if q in preds}
    run = {q: preds[q] for q in qrels}
    res = pytrec_eval.RelevanceEvaluator(qrels, MEASURES).evaluate(run)
    metrics = {k: avg([res[q][k] for q in res]) for k in OUT_KEYS}
    gold_sizes = [len(gold[q]) for q in qrels]
    return metrics, len(qrels), avg(gold_sizes)


def score_per_publication(preds, gold, pub_lookup):
    """Lens 2: one eval unit per (query, gold-publication); gold = that pub's variants;
    sibling gold docs removed from the candidate ranking."""
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    gold_sizes: list[int] = []
    for q, docs in gold.items():
        if q not in preds:
            continue
        ranking = preds[q]
        by_pub: dict[str, set[str]] = defaultdict(set)
        for d in docs:
            by_pub[_pub_of(d, pub_lookup)].add(d)
        for pub, pdocs in by_pub.items():
            siblings = docs - pdocs  # the concept's other gold docs
            pseudo = f"{q}@@{pub}"
            qrels[pseudo] = {d: 1 for d in pdocs}
            run[pseudo] = {d: s for d, s in ranking.items() if d not in siblings}
            gold_sizes.append(len(pdocs))
    res = pytrec_eval.RelevanceEvaluator(qrels, MEASURES).evaluate(run)
    metrics = {k: avg([res[p][k] for p in res]) for k in OUT_KEYS}
    return metrics, len(qrels), avg(gold_sizes)


def write_lens(name: str, blurb: str, rows: list[dict], n_units: int, avg_gold: float):
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: r["metrics"]["recall_10"], reverse=True)
    payload = {
        "interpretation": name,
        "description": blurb,
        "eval_units": n_units,
        "avg_relevant_per_unit": round(avg_gold, 2),
        "ranked_by": "recall_10",
        "models": [{"model_name": r["model"], "metrics": {k: round(r["metrics"][k], 5) for k in OUT_KEYS}} for r in ranked],
    }
    (d / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Interpretation: {name}", "", blurb, "",
        f"- Eval units: **{n_units}**  ·  avg relevant docs per unit: **{avg_gold:.1f}**",
        f"- Models ranked by Recall@10 (re-scored from the saved predictions; no model re-run).", "",
        "| Rank | Model | " + " | ".join(PRETTY[k] for k in OUT_KEYS) + " |",
        "| ---: | --- | " + " | ".join(["---:"] * len(OUT_KEYS)) + " |",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append("| " + " | ".join([str(i), f"`{r['model']}`", *[f"{r['metrics'][k]:.4f}" for k in OUT_KEYS]]) + " |")
    (d / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ranked


def main():
    if not PRED_DIR.is_dir():
        sys.exit(f"No predictions under {PRED_DIR}. Run cluster/combine_alias_graph.sh first.")
    print(">> loading qrels + corpus ...")
    gold, pub_lookup = load_gold_and_pubs()
    models = _discover_models(PRED_DIR, None)
    print(f">> {len(models)} model(s) with predictions")

    concept_rows, pp_rows = [], []
    c_units = c_gold = p_units = p_gold = 0
    for label, slug in models:
        preds = _load_predictions(PRED_DIR / slug)
        if not preds:
            print(f"   [skip] {label}: no predictions"); continue
        cm, c_units, c_gold = score_concept(preds, gold)
        pm, p_units, p_gold = score_per_publication(preds, gold, pub_lookup)
        concept_rows.append({"model": label, "metrics": cm})
        pp_rows.append({"model": label, "metrics": pm})
        print(f"   {label:48s} concept R@10={cm['recall_10']:.4f}  per-pub R@10={pm['recall_10']:.4f}")

    c_blurb = ("Gold = every document of every publication attesting the query's concept "
               "(\"find all patents about compound X\"). This is the benchmark as shipped; "
               "Recall@10 is mechanically capped because there are many relevant docs per query.")
    p_blurb = ("Gold = a single gold publication's language variants (\"find this patent's "
               "cross-language versions\"). Each (query, gold-publication) pair is one eval unit; "
               "the concept's other gold publications are excluded from the candidate ranking.")
    c_ranked = write_lens("concept_level", c_blurb, concept_rows, c_units, c_gold)
    p_ranked = write_lens("per_publication", p_blurb, pp_rows, p_units, p_gold)

    # Side-by-side comparison + README.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmp = ["# Alias-graph: two relevance interpretations", "",
           "Same models, same saved rankings — only the definition of *relevant* differs.", "",
           f"- **concept_level**: {c_units} queries, ~{c_gold:.0f} gold docs each.",
           f"- **per_publication**: {p_units} (query, publication) units, ~{p_gold:.1f} gold docs each.", "",
           "## Recall@10 / nDCG@10 / MRR side by side (ranked by per-publication Recall@10)", "",
           "| Model | concept R@10 | per-pub R@10 | concept nDCG@10 | per-pub nDCG@10 | concept MRR | per-pub MRR |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    cm = {r["model"]: r["metrics"] for r in concept_rows}
    pm = {r["model"]: r["metrics"] for r in pp_rows}
    for r in sorted(pp_rows, key=lambda r: r["metrics"]["recall_10"], reverse=True):
        m = r["model"]
        cmp.append(f"| `{m}` | {cm[m]['recall_10']:.4f} | {pm[m]['recall_10']:.4f} | "
                   f"{cm[m]['ndcg_cut_10']:.4f} | {pm[m]['ndcg_cut_10']:.4f} | "
                   f"{cm[m]['recip_rank']:.4f} | {pm[m]['recip_rank']:.4f} |")
    (OUT_DIR / "comparison.md").write_text("\n".join(cmp) + "\n", encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# Two relevance interpretations of the alias-graph run\n\n"
        f"{c_blurb}\n\n{p_blurb}\n\n"
        "Files:\n"
        "- `concept_level/` — leaderboard.md + summary.json (gold = all concept docs)\n"
        "- `per_publication/` — leaderboard.md + summary.json (gold = one publication's variants)\n"
        "- `comparison.md` — both lenses side by side\n\n"
        "Both are re-scored from `../predictions/` with pytrec_eval; the models were NOT re-run.\n",
        encoding="utf-8",
    )
    print(f"\n>> wrote {OUT_DIR}")
    print(f"   concept_level   : {OUT_DIR/'concept_level'/'leaderboard.md'}")
    print(f"   per_publication : {OUT_DIR/'per_publication'/'leaderboard.md'}")
    print(f"   comparison      : {OUT_DIR/'comparison.md'}")


if __name__ == "__main__":
    main()
