"""
Shared infrastructure for the chem-patents multilingual retrieval analysis (10-round CLIR deep-dive).

Everything is loaded locally — no network:
  - 9 models' top-1000 rankings:  reports/runs/chem_patents/parts/<model>/retrieval_results/scored_rankings.parquet
  - dataset configs (cached HF):  MehdiAstaraki/multi-lingual-qac-chem-patents {queries, qrels, corpus}
                                  MehdiAstaraki/multilingual_GP {corpus}  (the actual retrieval haystack)

The benchmark stresses CROSS-LINGUAL retrieval: a question in language L should pull back the
relevant patent in its *other* language versions. Two query populations:
  * original   (57) — question written in the source patent's language; HAS one same-language gold.
  * synthetic  (80) — question machine-translated into another language; pure CLIR, NO same-language
                      gold (the patent has no document in the query language).
Spanish is a pure query-side language: 34 es queries, **zero** es gold documents.

Gold (multilingual qrels): every available language version of the query's source patent.
Split per query by direction relative to the query language:
  * SAME-language gold  -> MoLIR  (monolingual retrieval; defined only for the 57 originals)
  * CROSS-language gold -> CLIR   (cross-lingual retrieval; defined for all 137 queries)

Provides: cached loaders, the core per-(model,query) metric frame, a CLIR metric library
(recall@k, CLIR@k / MoLIR@k, first-foreign-rank, mate-MRR, language entropy, RBO), statistics
(bootstrap_ci, paired_perm_test, wilcoxon), a stable plot style + model/language colours, and the
append_findings() helper that maintains experimental_plots/FINDINGS.md.

Run a round from the repo root inside .venv, e.g.:
    python reports/runs/chem_patents/experimental_codes/round01_clir_leaderboard.py
"""
from __future__ import annotations

import functools
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# load everything from the local HF cache, never hit the network
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# --------------------------------------------------------------------------- paths
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "reports" / "runs" / "chem_patents" / "parts").is_dir():
            return parent
    raise RuntimeError("could not locate repo root (reports/runs/chem_patents/parts not found)")


REPO_ROOT = _find_repo_root()
RUN_DIR = REPO_ROOT / "reports" / "runs" / "chem_patents"
PARTS_DIR = RUN_DIR / "parts"
PLOTS_DIR = RUN_DIR / "experimental_plots"
CODES_DIR = RUN_DIR / "experimental_codes"
KEY_DIR = RUN_DIR / "key_findings"
FINDINGS_MD = PLOTS_DIR / "FINDINGS.md"

DATASET_REPO = "MehdiAstaraki/multi-lingual-qac-chem-patents"
HAYSTACK_REPO = "MehdiAstaraki/multilingual_GP"


# --------------------------------------------------------------------------- registry
# all five appear as query languages; with the 524-query dataset all five ALSO appear as gold-doc
# languages (Spanish now has gold docs — the earlier 137-query release had es=0 gold).
QUERY_LANGS = ["en", "de", "es", "fr", "zh"]
DOC_LANGS = ["en", "de", "es", "fr", "zh"]
LANG_NAME = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese"}

# canonical full model name -> short label, in accuracy (recall@10) order
MODELS: Dict[str, str] = {
    "google/embeddinggemma-300m": "embeddinggemma",
    "BAAI/bge-m3": "bge-m3",
    "Qwen/Qwen3-Embedding-0.6B": "qwen3-0.6B",
    "nomic-ai/nomic-embed-text-v2-moe": "nomic-v2-moe",
    "ibm-granite/granite-embedding-278m-multilingual": "granite-278m",
    "sentence-transformers/LaBSE": "LaBSE",
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": "SapBERT",
    "intfloat/multilingual-e5-large-instruct": "e5-large-instruct",
    # Alibaba-NLP/gte-multilingual-base EXCLUDED 2026-06-11: its results are a model-loading
    # artifact, not real performance — it fails trivial same-language self-retrieval and shows
    # degenerate, near-equidistant embeddings (collapses identically on alias_graph and these
    # patents, recall@10 ~0.005 on both). Dropped from the registry AND the rankings() loader so
    # no analysis or plot counts it. Raw parts/predictions are kept on disk as the eval record.
}
MODEL_ORDER = list(MODELS)
SHORT = MODELS

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
MODEL_COLOR = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(MODEL_ORDER)}
LANG_COLOR = {"en": "#1f77b4", "de": "#ff7f0e", "fr": "#2ca02c", "es": "#d62728", "zh": "#9467bd"}
ORIGIN_COLOR = {"original": "#2ca02c", "synthetic": "#d62728"}

INF = float("inf")


def short(model: str) -> str:
    return SHORT.get(model, model.split("/")[-1])


def lang_of(corpus_id: str) -> str:
    """Language suffix of a corpus id, e.g. 'EP-4633662-A1_zh' -> 'zh' (fallback parser)."""
    m = re.search(r"_(de|en|es|fr|zh)$", str(corpus_id))
    return m.group(1) if m else ""


def patent_of(corpus_id: str) -> str:
    """Strip the language suffix: 'EP-4633662-A1_zh' -> 'EP-4633662-A1'."""
    return re.sub(r"_(de|en|es|fr|zh)$", "", str(corpus_id))


# --------------------------------------------------------------------------- loaders
@functools.lru_cache(maxsize=None)
def _ds(repo: str, config: str) -> pd.DataFrame:
    from datasets import load_dataset
    return pd.DataFrame(load_dataset(repo, config, split="train")[:])


def queries_df() -> pd.DataFrame:
    return _ds(DATASET_REPO, "queries")


def qrels_df() -> pd.DataFrame:
    return _ds(DATASET_REPO, "qrels")


@functools.lru_cache(maxsize=None)
def rankings() -> pd.DataFrame:
    """All 9 models' top-1000 rankings, concatenated (~1.23M rows)."""
    frames = []
    for part in sorted(PARTS_DIR.iterdir()):
        p = part / "retrieval_results" / "scored_rankings.parquet"
        if p.is_file():
            frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"].isin(MODELS)].reset_index(drop=True)  # drop excluded models (e.g. gte)
    df["query_language"] = df["query_language"].str.lower()
    df["corpus_language"] = df["corpus_language"].str.lower()
    return df


# --------------------------------------------------------------------------- query / gold maps
@functools.lru_cache(maxsize=None)
def query_meta() -> pd.DataFrame:
    """One row per query, indexed by query id, with the handy columns."""
    q = queries_df().copy()
    # the config carries both `_id` and an identical `query_id`; use `_id` as the canonical index
    q["query_language"] = q["query_language"].str.lower()
    q["src_language"] = q["corpus_language"].str.lower()          # source patent's language ("home")
    q["origin"] = q["is_synthetic_translation"].map({True: "synthetic", False: "original"})
    q["patent"] = q["corpus_id"].map(patent_of)
    return q.set_index("_id")


@functools.lru_cache(maxsize=None)
def q_lang() -> Dict[str, str]:
    return query_meta()["query_language"].to_dict()


@functools.lru_cache(maxsize=None)
def q_src_lang() -> Dict[str, str]:
    return query_meta()["src_language"].to_dict()


@functools.lru_cache(maxsize=None)
def q_origin() -> Dict[str, str]:
    return query_meta()["origin"].to_dict()


@functools.lru_cache(maxsize=None)
def q_patent() -> Dict[str, str]:
    return query_meta()["patent"].to_dict()


@functools.lru_cache(maxsize=None)
def gold_publication() -> Dict[str, Set[str]]:
    """Full gold set per query (every language version of the source patent), from qrels score>0."""
    g = defaultdict(set)
    qr = qrels_df()
    for qid, cid, score in zip(qr["query-id"], qr["corpus-id"], qr["score"]):
        if float(score) > 0:
            g[str(qid)].add(str(cid))
    return dict(g)


@functools.lru_cache(maxsize=None)
def corpus_lang() -> Dict[str, str]:
    """corpus_id -> language. Built from the rankings (authoritative for retrieved docs) and the
    haystack corpus, with a suffix-parse fallback."""
    m: Dict[str, str] = {}
    try:
        c = _ds(HAYSTACK_REPO, "corpus")
        for cid, lang in zip(c["_id"], c["corpus_language"]):
            m[str(cid)] = str(lang).lower()
    except Exception:
        pass
    r = rankings()
    for cid, lang in zip(r["corpus_id"], r["corpus_language"]):
        m.setdefault(str(cid), str(lang).lower())
    return m


def doc_lang(corpus_id: str) -> str:
    return corpus_lang().get(str(corpus_id)) or lang_of(corpus_id)


def split_gold(gold: Set[str], qlang: str) -> Tuple[Set[str], Set[str]]:
    """(same-language gold, cross-language gold) relative to the query language."""
    same, cross = set(), set()
    for d in gold:
        (same if doc_lang(d) == qlang else cross).add(d)
    return same, cross


# --------------------------------------------------------------------------- ranking access
@functools.lru_cache(maxsize=None)
def ranked_lists() -> Dict[Tuple[str, str], List[str]]:
    """(model, query_id) -> list of corpus_ids ordered by rank (1=best)."""
    df = rankings().sort_values("rank")
    out: Dict[Tuple[str, str], List[str]] = {}
    for (model, qid), grp in df.groupby(["model", "query_id"], sort=False):
        out[(model, qid)] = grp["corpus_id"].tolist()
    return out


@functools.lru_cache(maxsize=None)
def score_lists() -> Dict[Tuple[str, str], Dict[str, float]]:
    """(model, query_id) -> {corpus_id: score}."""
    df = rankings()
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (model, qid), grp in df.groupby(["model", "query_id"], sort=False):
        out[(model, qid)] = dict(zip(grp["corpus_id"], grp["score"]))
    return out


@functools.lru_cache(maxsize=None)
def rank_of() -> Dict[Tuple[str, str], Dict[str, int]]:
    """(model, query_id) -> {corpus_id: rank}. Missing docs treated as rank=inf by callers."""
    out: Dict[Tuple[str, str], Dict[str, int]] = {}
    for key, docs in ranked_lists().items():
        out[key] = {d: i + 1 for i, d in enumerate(docs)}
    return out


def best_rank(model: str, qid: str, docs: Set[str]) -> float:
    r = rank_of().get((model, qid), {})
    return min((r.get(d, INF) for d in docs), default=INF)


# --------------------------------------------------------------------------- metrics
def recall_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(ranked[:k]) & gold) / len(gold)


def hit_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return 1.0 if set(ranked[:k]) & gold else 0.0


def mrr_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return float("nan")
    for i, d in enumerate(ranked[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return float("nan")
    dcg = sum((1.0 / math.log2(i + 1)) for i, d in enumerate(ranked[:k], start=1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def first_gold_rank(ranked: Sequence[str], gold: Set[str]) -> float:
    """1-based rank of the first gold doc (INF if none retrieved)."""
    for i, d in enumerate(ranked, start=1):
        if d in gold:
            return float(i)
    return INF


def jaccard_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def overlap_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / k


def foreign_share_at_k(ranked: Sequence[str], qlang: str, k: int) -> float:
    """Fraction of the top-k that is in a *different* language than the query."""
    top = ranked[:k]
    if not top:
        return float("nan")
    return sum(1 for d in top if doc_lang(d) != qlang) / len(top)


def lang_entropy_at_k(ranked: Sequence[str], k: int) -> float:
    """Shannon entropy (bits) of the language distribution of the top-k; 0=monolingual."""
    top = ranked[:k]
    if not top:
        return float("nan")
    counts: Dict[str, int] = defaultdict(int)
    for d in top:
        counts[doc_lang(d)] += 1
    n = len(top)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def rbo(a: Sequence[str], b: Sequence[str], p: float = 0.9, k: int | None = None) -> float:
    """Extrapolated Rank-Biased Overlap (RBO_ext, Webber et al. 2010): top-weighted list similarity
    in [0,1]; 1.0 identical, 0.0 disjoint. p~0.9 emphasises roughly the first 1/(1-p)=10 ranks."""
    A = list(a[:k]) if k else list(a)
    B = list(b[:k]) if k else list(b)
    if not A and not B:
        return float("nan")
    if not A or not B:
        return 0.0
    if len(A) <= len(B):
        S, L = A, B
    else:
        S, L = B, A
    s, l = len(S), len(L)
    Sset: Set[str] = set()
    Lset: Set[str] = set()
    X = [0] * (l + 1)
    overlap = 0
    for d in range(1, l + 1):
        if d <= s:
            x = S[d - 1]
            if x in Lset:
                overlap += 1
            Sset.add(x)
        y = L[d - 1]
        if y in Sset:
            overlap += 1
        Lset.add(y)
        X[d] = overlap
    Xs, Xl = X[s], X[l]
    sum1 = sum((X[d] / d) * (p ** d) for d in range(1, l + 1))
    sum2 = sum((Xs * (d - s) / (s * d)) * (p ** d) for d in range(s + 1, l + 1)) if l > s else 0.0
    term = ((Xl - Xs) / l + Xs / s) * (p ** l)
    return (1 - p) / p * (sum1 + sum2) + term


# --------------------------------------------------------------------------- core per-query frame
KS = (1, 5, 10, 20, 50, 100)


@functools.lru_cache(maxsize=None)
def core_per_query() -> pd.DataFrame:
    """One row per (model, query): overall / CLIR / MoLIR recall at several k, plus mate-retrieval
    and language-mix diagnostics. This is the workhorse frame several rounds build on."""
    lists = ranked_lists()
    gold = gold_publication()
    ql = q_lang()
    qsrc = q_src_lang()
    qorg = q_origin()
    qpat = q_patent()
    rows = []
    for model in MODEL_ORDER:
        for qid, g in gold.items():
            ranked = lists.get((model, qid), [])
            lang = ql.get(qid, "")
            same, cross = split_gold(g, lang)
            row = {
                "model": model, "short": short(model), "query_id": qid,
                "query_language": lang, "src_language": qsrc.get(qid, ""),
                "origin": qorg.get(qid, ""), "patent": qpat.get(qid, ""),
                "n_gold": len(g), "n_gold_same": len(same), "n_gold_cross": len(cross),
                "mrr_at_10": mrr_at_k(ranked, g, 10),
                "ndcg_at_10": ndcg_at_k(ranked, g, 10),
                "first_gold_rank": first_gold_rank(ranked, g),
                "first_cross_rank": first_gold_rank(ranked, cross),
                "mate_mrr": 1.0 / first_gold_rank(ranked, cross) if cross else float("nan"),
                "foreign_share_at_10": foreign_share_at_k(ranked, lang, 10),
                "lang_entropy_at_10": lang_entropy_at_k(ranked, 10),
            }
            for k in KS:
                row[f"recall_at_{k}"] = recall_at_k(ranked, g, k)
                row[f"clir_at_{k}"] = recall_at_k(ranked, cross, k) if cross else float("nan")
                row[f"molir_at_{k}"] = recall_at_k(ranked, same, k) if same else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- statistics
_RNG = np.random.default_rng(20260609)


def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 10000, alpha: float = 0.05, seed: int | None = None) -> Tuple[float, float, float]:
    """(point, lo, hi) for `stat` over `values` via the percentile bootstrap."""
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))],
                   dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat(v))
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = stat(v[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def bootstrap_ci_rate(successes: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                      seed: int | None = None) -> Tuple[float, float, float]:
    return bootstrap_ci(successes, stat=np.mean, n_boot=n_boot, alpha=alpha, seed=seed)


def paired_perm_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000,
                     seed: int | None = None) -> dict:
    """Two-sided paired sign-flip permutation test on mean(a-b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    n = d.size
    if n == 0:
        return {"diff": float("nan"), "p": float("nan"), "n": 0}
    obs = float(d.mean())
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * d).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(obs) - 1e-12).mean())
    return {"diff": obs, "p": p, "n": int(n)}


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired Wilcoxon signed-rank (scipy); falls back to the permutation test."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    try:
        from scipy.stats import wilcoxon as _w
        if a.size == 0 or np.allclose(a, b):
            return {"stat": 0.0, "p": 1.0, "n": int(a.size)}
        res = _w(a, b)
        return {"stat": float(res.statistic), "p": float(res.pvalue), "n": int(a.size)}
    except Exception:
        out = paired_perm_test(a, b)
        return {"stat": float("nan"), "p": out["p"], "n": out["n"]}


def stars(p: float) -> str:
    if p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# --------------------------------------------------------------------------- plotting / io
def set_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 140, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.fontsize": 8, "legend.frameon": False,
    })


def round_dir(slug: str) -> Path:
    d = PLOTS_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_findings(section_md: str) -> None:
    """Append (or refresh) a round's section in the evolving FINDINGS.md, delimited by
    '<!-- round:<slug> -->' markers so re-running a round replaces its own section."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    text = FINDINGS_MD.read_text(encoding="utf-8") if FINDINGS_MD.exists() else (
        "# Chem-patents multilingual retrieval — CLIR findings\n\n"
        "Iterative 10-round deep-dive on the 9-model chemistry-patent retrieval benchmark, centred on "
        "**cross-lingual retrieval (CLIR)**: can a question in one language pull back the relevant "
        "patent in its *other* language versions? Each section is one round "
        "(question -> plots -> numbers -> what we learned -> next questions).\n"
    )
    slug = re.search(r"round:(\S+)", section_md)
    if slug:
        marker = slug.group(1)
        pat = re.compile(rf"<!-- round:{re.escape(marker)} -->.*?<!-- /round:{re.escape(marker)} -->",
                         re.DOTALL)
        if pat.search(text):
            text = pat.sub(section_md.strip(), text)
        else:
            text = text.rstrip() + "\n\n" + section_md.strip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + section_md.strip() + "\n"
    FINDINGS_MD.write_text(text, encoding="utf-8")


def jdump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    df = rankings()
    print("rankings:", df.shape, "| models:", df["model"].nunique(), "| queries:", df["query_id"].nunique())
    g = gold_publication()
    print("queries with gold:", len(g), "| mean gold/query:", round(np.mean([len(v) for v in g.values()]), 2))
    same = sum(len(split_gold(s, q_lang()[qid])[0]) for qid, s in g.items())
    cross = sum(len(split_gold(s, q_lang()[qid])[1]) for qid, s in g.items())
    print(f"gold split: same-lang={same}  cross-lang={cross}")
    cpq = core_per_query()
    print("core_per_query:", cpq.shape)
    lead = cpq[cpq.model == "google/embeddinggemma-300m"]
    print("embeddinggemma  recall@10:", round(lead["recall_at_10"].mean(), 4),
          "| CLIR@10:", round(lead["clir_at_10"].mean(), 4),
          "| MoLIR@10:", round(lead["molir_at_10"].mean(), 4))
    print("RBO identical:", round(rbo(list("abcde"), list("abcde")), 3),
          "| disjoint:", round(rbo(list("abcde"), list("fghij")), 3))
    print("models:", [short(m) for m in MODEL_ORDER])
