"""
Shared infrastructure for the FINAL paper-figure suite (reports/runs/final_plots).

Everything is read from local CSV/parquet artifacts already produced by the eval +
analysis pipeline. No embedding model is ever loaded; no network is touched.

Story (five claims):
  A  technical questions score lower than semantic questions
  B  the technical<semantic / model-ranking pattern transfers across patent offices (GP vs EPO)
  C  per-language results, language-balanced and confound-decomposed
  D  alias-graph "distractor latch": find the right concept across languages while
     rejecting chemically-similar look-alikes
  E  deployment-cost wrap-up: XRC (reading cost) / RRC@K (re-ranker ceiling) / ARI

Sources
  reports/runs/chem_patents/question_analysis/question_level_metrics.csv   (per-query R@10, mode empty)
  reports/runs/epo/question_analysis/question_level_metrics.csv            (per-query R@10, mode filled)
  reports/runs/{chem_patents,epo}/mteb_tables/model_comparison.csv         (aggregate metric family)
  data/google_patents/qac/qac_chempatents.csv                             (recovers chem_patents mode)
  reports/runs/chem_patents/experimental_plots/extra_*                     (XRC / RRC / ARI / frontier)
  reports/runs/alias_graph/{confusion,experimental_plots}/...             (confusion / RBO / attractors / sep)

Conventions
  * one fixed colour per model and per language, shared across every figure
  * single-sentence titles; the explanation lives in the paper caption
  * every figure also writes the exact plotted numbers to data/<fig>.csv
  * candidates render to candidates/ as PNG (300 dpi) + PDF
"""
from __future__ import annotations

import functools as _functools
import math
import re
import re as _re
from collections import defaultdict as _defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- paths
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "reports" / "runs" / "chem_patents" / "parts").is_dir():
            return parent
    raise RuntimeError("could not locate repo root (reports/runs/chem_patents/parts not found)")


REPO_ROOT = _find_repo_root()
RUNS = REPO_ROOT / "reports" / "runs"
CP_RUN = RUNS / "chem_patents"
EPO_RUN = RUNS / "epo"
AG_RUN = RUNS / "alias_graph"
FINAL = RUNS / "final_plots"
CAND = FINAL / "candidates"
DATA = FINAL / "data"
MAIN = FINAL / "main"
APPX = FINAL / "appendix"
for _d in (CAND, DATA, MAIN, APPX):
    _d.mkdir(parents=True, exist_ok=True)

CP_EXTRA = CP_RUN / "experimental_plots"
AG_PLOTS = AG_RUN / "experimental_plots"


# --------------------------------------------------------------------------- registry
# Five target languages. EPO covers only de/en/fr.
LANG_NAME = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese"}
CP_LANGS = ["en", "de", "es", "fr", "zh"]
EPO_LANGS = ["en", "de", "fr"]

# full HF name -> short label, in recall@10 order (gte excluded: loading artifact, see memory)
MODELS: Dict[str, str] = {
    "google/embeddinggemma-300m": "embeddinggemma",
    "BAAI/bge-m3": "bge-m3",
    "Qwen/Qwen3-Embedding-0.6B": "qwen3-0.6B",
    "nomic-ai/nomic-embed-text-v2-moe": "nomic-v2-moe",
    "ibm-granite/granite-embedding-278m-multilingual": "granite-278m",
    "sentence-transformers/LaBSE": "LaBSE",
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": "SapBERT",
    "intfloat/multilingual-e5-large-instruct": "e5-large-instruct",
}
MODEL_ORDER = list(MODELS)
SHORT = MODELS

# models that are degenerate on the cross-lingual instruments (CLIR@10 < 0.10): drawn hollow/greyed
# in cross-lingual figures. e5 retrieves but siloes by language; gte is already excluded entirely.
DEGENERATE_CROSS = {"intfloat/multilingual-e5-large-instruct"}

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
MODEL_COLOR = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(MODEL_ORDER)}
LANG_COLOR = {"en": "#1f77b4", "de": "#ff7f0e", "fr": "#2ca02c", "es": "#d62728", "zh": "#9467bd"}
MODE_COLOR = {"technical": "#c44e52", "semantic": "#4c72b0"}


def short(model: str) -> str:
    return SHORT.get(model, model.split("/")[-1])


# --------------------------------------------------------------------------- per-query loaders
def _qlm(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "question_analysis" / "question_level_metrics.csv")
    df = df[df["model"].isin(MODELS)].copy()
    df["short"] = df["model"].map(SHORT)
    df["query_language"] = df["query_language"].str.lower()
    return df


def epo_per_query() -> pd.DataFrame:
    """EPO per-query metrics (198 queries; this summary is current, not stale). `mode` is populated;
    `question_type` is joined in from qac_epo_best.csv for the technical sub-type analysis."""
    df = _qlm(EPO_RUN)
    assert df["mode"].isin(["technical", "semantic"]).all(), "EPO mode column unexpectedly empty"
    best = pd.read_csv(REPO_ROOT / "data" / "EPO" / "qac" / "qac_epo_best.csv")
    best["query_id"] = best["corpus_id"] + "_q_" + best["question_language"]
    qt = best.dropna(subset=["question_type"]).drop_duplicates("query_id").set_index("query_id")["question_type"]
    df["question_type"] = df["query_id"].map(qt)
    return df


@_functools.lru_cache(maxsize=1)
def _epo_corpus_family():
    """(corpus_id -> language, publication -> {corpus_ids}) for EPO gold reconstruction."""
    from datasets import load_dataset
    corp = pd.DataFrame(load_dataset("MehdiAstaraki/multi-lingual-qac-epo", "corpus", split="train"))
    corp["corpus_language"] = corp["corpus_language"].str.lower()
    clang = dict(zip(corp["corpus_id"], corp["corpus_language"]))
    fam = _defaultdict(set)
    for cid, pub in zip(corp["corpus_id"], corp["publication_number"]):
        fam[str(pub)].add(cid)
    pub_of = {cid: str(pub) for cid, pub in zip(corp["corpus_id"], corp["publication_number"])}
    return clang, dict(fam), pub_of


@_functools.lru_cache(maxsize=1)
def epo_rankings() -> pd.DataFrame:
    """All 8 models' top-1000 EPO rankings (model, query_id, query_language, rank, corpus_id, corpus_language)."""
    frames = []
    for part in sorted((EPO_RUN / "parts").iterdir()):
        p = part / "retrieval_results" / "scored_rankings.parquet"
        if p.is_file():
            frames.append(pd.read_parquet(
                p, columns=["model", "query_id", "query_language", "rank", "corpus_id", "corpus_language"]))
    df = pd.concat(frames, ignore_index=True)
    return df[df["model"].isin(MODELS)].copy()


@_functools.lru_cache(maxsize=1)
def epo_gold() -> dict:
    """query_id -> (same-language gold set, cross-language gold set) for EPO."""
    clang, fam, pub_of = _epo_corpus_family()
    df = epo_rankings()
    meta = df.drop_duplicates("query_id")[["query_id", "query_language"]].copy()
    meta["pub"] = meta["query_id"].str.replace(r"_q_[a-z]{2}$", "", regex=True).map(lambda c: pub_of.get(c))
    out = {}
    for r in meta.itertuples():
        gold = fam.get(str(r.pub), set())
        same = frozenset(d for d in gold if clang.get(d) == r.query_language)
        out[r.query_id] = (same, frozenset(gold) - same)
    return out


@_functools.lru_cache(maxsize=1)
def epo_per_query_clir() -> pd.DataFrame:
    """EPO per-(model, query) recall/clir/molir over reconstructed gold (gold = the source patent's
    language versions in the EPO haystack). Validated: pooled Recall@10 matches MTEB exactly."""
    gold = epo_gold()
    df = epo_rankings()
    top = df[df["rank"] <= 10].groupby(["model", "query_id"])["corpus_id"].apply(set).to_dict()
    meta = df.drop_duplicates("query_id")[["query_id", "query_language"]]
    out = []
    for r in meta.itertuples():
        same, cross = gold[r.query_id]
        full = same | cross
        for m in MODEL_ORDER:
            t = top.get((m, r.query_id), set())
            out.append({
                "model": m, "short": short(m), "query_id": r.query_id, "query_language": r.query_language,
                "recall_at_10": len(t & full) / len(full) if full else float("nan"),
                "molir_at_10": len(t & same) / len(same) if same else float("nan"),
                "clir_at_10": len(t & cross) / len(cross) if cross else float("nan"),
            })
    return pd.DataFrame(out)


def _first_rank(rank_of: dict, docs) -> float:
    r = [rank_of[d] for d in docs if d in rank_of]
    return min(r) if r else float("inf")


def epo_first_ranks() -> pd.DataFrame:
    """Per-(model, query) first same-language-gold and first cross-language-gold rank (inf if the gold
    is never retrieved in top-1000), plus the gold counts. Mirrors the GP first-rank frame."""
    gold = epo_gold()
    df = epo_rankings()
    rows = []
    for (m, qid), grp in df.groupby(["model", "query_id"]):
        same, cross = gold[qid]
        rank_of = dict(zip(grp["corpus_id"], grp["rank"]))
        rows.append({
            "model": m, "query_id": qid,
            "first_same_rank": _first_rank(rank_of, same) if same else float("nan"),
            "first_cross_rank": _first_rank(rank_of, cross) if cross else float("nan"),
            "n_gold_same": len(same), "n_gold_cross": len(cross),
        })
    return pd.DataFrame(rows)


# ----- Google-Patents: full 524-query rebuild from rankings + _best.csv + reconstructed gold -----
# The shipped question_level_metrics.csv is a stale 135-query summary (old 137-query release) and the
# cached HF qrels are stale too (Spanish gold = 0). The actual eval covers all 524 queries: their
# top-1000 rankings are in parts/<model>/retrieval_results/scored_rankings.parquet (gold flagged), and
# the authoritative question set with modes is data/google_patents/qac/qac_chempatents_best.csv.
# We rebuild per-query metrics from those, reconstructing each query's gold as every language version
# of its source patent present in the haystack (validated: pooled Recall@10 = 0.5725 vs MTEB 0.5739).
import functools as _functools
import re as _re
from collections import defaultdict as _defaultdict

_GP_BEST = REPO_ROOT / "data" / "google_patents" / "qac" / "qac_chempatents_best.csv"
_LANG_SUFFIX = _re.compile(r"_(de|en|es|fr|zh)$")


def _patent_of(corpus_id: str) -> str:
    return _LANG_SUFFIX.sub("", str(corpus_id))


@_functools.lru_cache(maxsize=1)
def _gp_corpus_family():
    """(corpus_id -> language, publication -> {corpus_ids in haystack}) for gold reconstruction."""
    import sys
    sys.path.insert(0, str(CP_RUN / "experimental_codes"))
    import common as cpc
    clang = {str(k): str(v).lower() for k, v in cpc.corpus_lang().items()}
    fam = _defaultdict(set)
    for cid in clang:
        fam[_patent_of(cid)].add(cid)
    return clang, dict(fam)


@_functools.lru_cache(maxsize=1)
def gp_rankings() -> pd.DataFrame:
    """All 8 models' top-1000 GP rankings (model, query_id, rank, corpus_id, corpus_language)."""
    frames = []
    for part in sorted((CP_RUN / "parts").iterdir()):
        p = part / "retrieval_results" / "scored_rankings.parquet"
        if p.is_file():
            frames.append(pd.read_parquet(
                p, columns=["model", "query_id", "rank", "corpus_id", "corpus_language"]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"].isin(MODELS)].copy()
    df["corpus_language"] = df["corpus_language"].str.lower()
    return df


@_functools.lru_cache(maxsize=1)
def gp_query_table() -> pd.DataFrame:
    """One row per benchmark query (524): query_id, languages, mode, question_type, is_synthetic, and
    the reconstructed gold set. Maps _best.csv rows to the parquet's query_ids, including the eval's
    collision disambiguation (duplicate (corpus_id,qlang) -> `<qid>_<rowidx>`; two zh-source rows ->
    `q_<rowidx>_zh`)."""
    clang, fam = _gp_corpus_family()
    best = pd.read_csv(_GP_BEST).reset_index().rename(columns={"index": "row"})
    pqids = set(gp_rankings()["query_id"].unique())
    rows, used = [], set()
    for r in best.itertuples():
        nat = f"{r.corpus_id}_q_{r.question_language}"
        cands = [nat, f"{nat}_{r.row}", f"q_{r.row}_{r.question_language}"]
        qid = next((c for c in cands if c in pqids and c not in used), nat)
        used.add(qid)
        src = (_LANG_SUFFIX.search(str(r.corpus_id)) or [None])
        src_lang = src.group(1) if hasattr(src, "group") else None
        gold = fam.get(_patent_of(r.corpus_id), set())
        rows.append({
            "query_id": qid, "query_language": r.question_language, "src_language": src_lang,
            "mode": r.mode, "question_type": r.question_type,
            "is_synthetic": (r.question_language != src_lang),
            "corpus_id": r.corpus_id,
            "gold": frozenset(gold),
            "gold_same": frozenset(d for d in gold if clang.get(d) == r.question_language),
            "gold_cross": frozenset(d for d in gold if clang.get(d) != r.question_language),
        })
    tab = pd.DataFrame(rows)
    assert tab["query_id"].nunique() == len(tab) == 524, f"GP query table not 524 unique: {len(tab)}"
    return tab


@_functools.lru_cache(maxsize=1)
def cp_per_query() -> pd.DataFrame:
    """Google-Patents per-(model, query) metrics over all 524 queries, gold-correct.

    Columns: model, short, query_id, query_language, mode, question_type, is_synthetic,
    recall_at_10, rr_at_10, hit_at_10, clir_at_10 (cross-language gold), molir_at_10 (same-language
    gold), n_gold, n_gold_same, n_gold_cross. Recall uses the full reconstructed gold as denominator.
    """
    tab = gp_query_table().set_index("query_id")
    rk = gp_rankings()
    top = rk[rk["rank"] <= 10].sort_values("rank")
    top_lists = top.groupby(["model", "query_id"])["corpus_id"].apply(list).to_dict()
    gold = tab["gold"].to_dict()
    gold_same = tab["gold_same"].to_dict()
    gold_cross = tab["gold_cross"].to_dict()

    def _recall(t, g):
        return float(len(set(t) & g) / len(g)) if g else float("nan")

    def _mrr(t, g):
        for i, d in enumerate(t, 1):
            if d in g:
                return 1.0 / i
        return 0.0

    out = []
    for model in MODEL_ORDER:
        for qid, meta in tab.iterrows():
            t = top_lists.get((model, qid), [])
            g = gold[qid]
            if not g:
                continue
            ts = set(t)
            out.append({
                "model": model, "short": short(model), "query_id": qid,
                "query_language": meta["query_language"], "mode": meta["mode"],
                "question_type": meta["question_type"], "is_synthetic": meta["is_synthetic"],
                "recall_at_10": _recall(t, g), "rr_at_10": _mrr(t, g),
                "hit_at_10": 1.0 if ts & g else 0.0,
                "clir_at_10": _recall(t, gold_cross[qid]) if gold_cross[qid] else float("nan"),
                "molir_at_10": _recall(t, gold_same[qid]) if gold_same[qid] else float("nan"),
                "n_gold": len(g), "n_gold_same": len(gold_same[qid]), "n_gold_cross": len(gold_cross[qid]),
            })
    return pd.DataFrame(out)


def gp_denominators() -> pd.DataFrame:
    """Per-language query / gold-qrel / haystack-document counts (the real 524-query benchmark)."""
    clang, _ = _gp_corpus_family()
    tab = gp_query_table()
    n_q = tab["query_language"].value_counts().reindex(CP_LANGS).fillna(0).astype(int)
    golddocs = set().union(*tab["gold"]) if len(tab) else set()
    n_gold = pd.Series(_defaultdict(int)) if False else None
    from collections import Counter
    n_gold = pd.Series(Counter(clang.get(d) for d in golddocs)).reindex(CP_LANGS).fillna(0).astype(int)
    n_doc = pd.Series(Counter(clang.values())).reindex(CP_LANGS).fillna(0).astype(int)
    return pd.DataFrame({"query_language": CP_LANGS, "n_queries": n_q.to_numpy(),
                         "n_gold_qrels": n_gold.to_numpy(), "n_haystack_docs": n_doc.to_numpy()})


def model_comparison(run: str) -> pd.DataFrame:
    """Aggregate metric family (recall/ndcg/mrr/map at k) for 'chem_patents' or 'epo'."""
    run_dir = CP_RUN if run == "chem_patents" else EPO_RUN
    df = pd.read_csv(run_dir / "mteb_tables" / "model_comparison.csv")
    df = df[df["model_name"].isin(MODELS)].copy()
    df["short"] = df["model_name"].map(SHORT)
    return df


# --------------------------------------------------------------------------- aggregation helpers
_RNG_SEED = 20260611


def bootstrap_ci(values: Sequence[float], stat: Callable = np.mean, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = _RNG_SEED) -> Tuple[float, float, float]:
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))],
                   dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    point = float(stat(v))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = stat(v[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def lang_balanced_mean(df: pd.DataFrame, value: str, lang_col: str = "query_language") -> float:
    """Macro-average over languages: mean of per-language means (each language weighted equally),
    so unequal per-language question counts cannot bias the headline number."""
    per_lang = df.groupby(lang_col)[value].mean()
    return float(per_lang.mean())


def lang_balanced_ci(df: pd.DataFrame, value: str, lang_col: str = "query_language",
                     n_boot: int = 5000, seed: int = _RNG_SEED) -> Tuple[float, float, float]:
    """Bootstrap CI for the language-balanced mean: resample queries *within* each language, then
    average the per-language means. Keeps the equal-language weighting under resampling."""
    langs = sorted(df[lang_col].unique())
    groups = {g: df.loc[df[lang_col] == g, value].dropna().to_numpy() for g in langs}
    groups = {g: v for g, v in groups.items() if v.size}
    if not groups:
        return (float("nan"),) * 3
    point = float(np.mean([v.mean() for v in groups.values()]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        means = [v[rng.integers(0, v.size, v.size)].mean() for v in groups.values()]
        boot[b] = np.mean(means)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (point, float(lo), float(hi))


# --------------------------------------------------------------------------- plotting / io
def set_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10.5, "legend.fontsize": 8.5, "legend.frameon": False,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    })


def save(fig, name: str) -> None:
    """Save a candidate as PNG (300 dpi) + PDF under candidates/."""
    import matplotlib.pyplot as plt
    fig.savefig(CAND / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(CAND / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def dump_data(df: pd.DataFrame, name: str) -> None:
    df.to_csv(DATA / f"{name}.csv", index=False)


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    set_style()
    cp = cp_per_query()
    epo = epo_per_query()
    print("chem_patents per-query rows:", len(cp), "| unique queries:", cp["query_id"].nunique())
    print("  mode (unique queries):",
          cp.drop_duplicates("query_id")["mode"].value_counts(dropna=False).to_dict())
    print("  query_language:", cp.drop_duplicates("query_id")["query_language"].value_counts().to_dict())
    # validate reconstructed pooled Recall@10 against the official MTEB table
    mc = model_comparison("chem_patents").set_index("short")["recall_at_10"]
    print("  Recall@10 reconstructed vs MTEB (should match within ~0.005):")
    for m in MODEL_ORDER:
        r = cp[cp["model"] == m]["recall_at_10"].mean()
        print(f"    {short(m):18s} mine={r:.4f}  mteb={mc.get(short(m), float('nan')):.4f}")
    den = gp_denominators()
    print("  per-language denominators (524-query):")
    print(den.to_string(index=False))
    print("epo per-query unique queries:", epo["query_id"].nunique(),
          "| mode:", epo.drop_duplicates("query_id")["mode"].value_counts().to_dict())
    print("models:", [short(m) for m in MODEL_ORDER])
