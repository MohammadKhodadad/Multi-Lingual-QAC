"""
Self-contained data layer for the *user-requested* chem-patents plots (524-query release).

WHY THIS EXISTS (and does not reuse common.core_per_query):
  common.gold_publication() reads the cached HF `qrels` config, which is STALE at the old
  137-query release. The current run has 524 queries / 1284 qrels (run_metadata.json), so the gold
  must be rebuilt. We reconstruct the COMPLETE gold per query from the (current) multilingual_GP
  corpus: gold(query) = every language version of the query's source patent that exists in the
  corpus, keyed by `publication_number`. This reproduces the authoritative per-query gold count
  `n_relevant` from question_analysis/question_level_metrics.csv for 522/524 queries; the 2 odd
  `q_*_zh` ids (whose source doc is not in the corpus) fall back to the rankings `relevance=='gold'`
  union.

Authoritative per-query attributes (model-independent) come from question_level_metrics.csv:
  query_language, mode in {technical, semantic}, strategy, is_synthetic_translation, recall_at_10.

gte-base is DROPPED everywhere (degenerate encoder: corpus-wide recall@10 ~= 0.005).

cross-vs-same (per the user's definition): for a query whose gold lives in >=2 languages, the query
is SAME-language if at least one gold doc is in the query language, else CROSS-language (all gold
foreign). Reuses common.py for loaders/metrics/style/colours.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # loaders, metrics, plot style, colours

# ---- models: drop the degenerate gte-base, keep canonical recall order ----
DROP_MODELS = {"Alibaba-NLP/gte-multilingual-base"}
MODELS = [m for m in C.MODEL_ORDER if m not in DROP_MODELS]
short = C.short
LANG_NAME = C.LANG_NAME
LANG_COLOR = C.LANG_COLOR
MODEL_COLOR = C.MODEL_COLOR
QUERY_LANGS = ["en", "de", "fr", "es", "zh"]

RUN_DIR = C.RUN_DIR
PLOTS_DIR = C.PLOTS_DIR
QLM_CSV = RUN_DIR / "question_analysis" / "question_level_metrics.csv"
OUT_DIR = PLOTS_DIR / "requested_plots"

INF = float("inf")


# --------------------------------------------------------------------------- gold reconstruction
@functools.lru_cache(maxsize=None)
def _corpus() -> pd.DataFrame:
    c = C._ds(C.HAYSTACK_REPO, "corpus").copy()
    c["corpus_language"] = c["corpus_language"].str.lower()
    return c


@functools.lru_cache(maxsize=None)
def _pub_to_ids() -> Dict[str, List[str]]:
    c = _corpus()
    return c.groupby("publication_number")["_id"].apply(list).to_dict()


@functools.lru_cache(maxsize=None)
def _id_to_pub() -> Dict[str, str]:
    c = _corpus()
    return dict(zip(c["_id"], c["publication_number"]))


@functools.lru_cache(maxsize=None)
def _id_to_lang() -> Dict[str, str]:
    c = _corpus()
    m = dict(zip(c["_id"], c["corpus_language"]))
    # fallback for any gold id only present in rankings
    r = C.rankings()
    for cid, lang in zip(r["corpus_id"], r["corpus_language"]):
        m.setdefault(str(cid), str(lang).lower())
    return m


def doc_lang(cid: str) -> str:
    return _id_to_lang().get(str(cid)) or C.lang_of(cid)


@functools.lru_cache(maxsize=None)
def _rankings_gold_union() -> Dict[str, Set[str]]:
    """query_id -> set of gold corpus_ids found in any model's top-1000 (fallback only)."""
    r = C.rankings()
    g = r[r["relevance"] == "gold"]
    return g.groupby("query_id")["corpus_id"].apply(lambda s: set(map(str, s))).to_dict()


@functools.lru_cache(maxsize=None)
def gold_complete() -> Dict[str, Set[str]]:
    """query_id -> COMPLETE gold set (all language versions of the source patent)."""
    p2i = _pub_to_ids()
    i2p = _id_to_pub()
    union = _rankings_gold_union()
    out: Dict[str, Set[str]] = {}
    for qid in query_meta().index:
        src = qid.split("_q_")[0]            # source doc corpus_id, e.g. EP-4633662-A1_en
        pub = i2p.get(src)
        gold = set(p2i.get(pub, [])) if pub else set()
        if not gold:                         # the 2 odd q_*_zh ids
            gold = set(union.get(qid, set()))
        out[qid] = gold
    return out


# --------------------------------------------------------------------------- per-query metadata
@functools.lru_cache(maxsize=None)
def query_meta() -> pd.DataFrame:
    """One row per query (524), indexed by query_id, model-independent attributes."""
    qm = pd.read_csv(QLM_CSV)
    meta = qm.drop_duplicates("query_id").set_index("query_id")[
        ["query_language", "mode", "strategy", "is_synthetic_translation"]
    ].copy()
    meta["query_language"] = meta["query_language"].str.lower()
    return meta


@functools.lru_cache(maxsize=None)
def per_query() -> pd.DataFrame:
    """One row per query with gold-language breakdown and the cross/same label."""
    meta = query_meta()
    gold = gold_complete()
    rows = []
    for qid, m in meta.iterrows():
        g = gold.get(qid, set())
        qlang = m["query_language"]
        glangs = sorted({doc_lang(d) for d in g})
        same = {d for d in g if doc_lang(d) == qlang}
        cross = g - same
        rows.append({
            "query_id": qid,
            "query_language": qlang,
            "mode": m["mode"],
            "strategy": m["strategy"],
            "is_synthetic_translation": bool(m["is_synthetic_translation"]),
            "origin": "synthetic" if m["is_synthetic_translation"] else "original",
            "n_gold": len(g),
            "gold_langs": glangs,
            "n_gold_same": len(same),
            "n_gold_cross": len(cross),
            "direction": "same" if same else "cross",
        })
    return pd.DataFrame(rows).set_index("query_id")


# --------------------------------------------------------------------------- per-(model, query) metrics
KS = (1, 5, 10, 20, 50, 100)


@functools.lru_cache(maxsize=None)
def core() -> pd.DataFrame:
    """One row per (model, query) over the 8 kept models, with recall on full/same/cross gold
    and first-gold ranks (for XRC/RRC/ARI). Gold is the reconstructed 524-query gold."""
    lists = C.ranked_lists()
    gold = gold_complete()
    pq = per_query()
    rows = []
    for model in MODELS:
        for qid, prow in pq.iterrows():
            ranked = lists.get((model, qid), [])
            g = gold.get(qid, set())
            qlang = prow["query_language"]
            same = {d for d in g if doc_lang(d) == qlang}
            cross = g - same
            row = {
                "model": model, "short": short(model), "query_id": qid,
                "query_language": qlang, "mode": prow["mode"],
                "origin": prow["origin"], "direction": prow["direction"],
                "n_gold": len(g), "n_gold_same": len(same), "n_gold_cross": len(cross),
                "first_gold_rank": C.first_gold_rank(ranked, g),
                "first_same_rank": C.first_gold_rank(ranked, same) if same else INF,
                "first_cross_rank": C.first_gold_rank(ranked, cross) if cross else INF,
            }
            for k in KS:
                row[f"recall_at_{k}"] = C.recall_at_k(ranked, g, k)
                row[f"same_recall_at_{k}"] = C.recall_at_k(ranked, same, k) if same else float("nan")
                row[f"cross_recall_at_{k}"] = C.recall_at_k(ranked, cross, k) if cross else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- io helpers
def out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def sem(values) -> float:
    v = np.asarray([x for x in values if x == x], dtype=float)
    return float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


# --------------------------------------------------------------------------- self-test / verification
if __name__ == "__main__":
    pq = per_query()
    print("per_query:", pq.shape)
    print("  mode:", pq["mode"].value_counts().to_dict())
    print("  direction:", pq["direction"].value_counts().to_dict())
    print("  origin:", pq["origin"].value_counts().to_dict())
    print("  query_language:", pq["query_language"].value_counts().to_dict())

    # cross-check reconstructed gold count == authoritative n_relevant
    qm = pd.read_csv(QLM_CSV).drop_duplicates("query_id").set_index("query_id")["n_relevant"]
    chk = pq.join(qm)
    chk["match"] = chk["n_gold"] == chk["n_relevant"]
    print(f"\n[CHECK] reconstructed n_gold == n_relevant: {chk['match'].sum()}/{len(chk)} "
          f"(total gold pairs {int(pq['n_gold'].sum())} vs metadata 1284)")

    # cross-check recall@10 vs question_level_metrics.csv (authoritative per-(model,query))
    cm = core()
    qlm = pd.read_csv(QLM_CSV)
    qlm = qlm[qlm["model"].isin(MODELS)][["model", "query_id", "recall_at_10"]]
    merged = cm.merge(qlm, on=["model", "query_id"], suffixes=("_recon", "_csv"))
    diff = (merged["recall_at_10_recon"] - merged["recall_at_10_csv"]).abs()
    print(f"[CHECK] recall@10 recon vs csv: max abs diff = {diff.max():.4g}, "
          f"mean abs diff = {diff.mean():.4g}, n mismatches>1e-6 = {(diff > 1e-6).sum()}/{len(merged)}")

    print("\nmean recall@10 by model (524-query, full gold):")
    print(cm.groupby("short")["recall_at_10"].mean().sort_values(ascending=False).round(4).to_string())
