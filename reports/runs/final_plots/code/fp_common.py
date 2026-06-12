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

import math
import re
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
    """EPO per-query metrics; `mode` is already populated (technical/semantic)."""
    df = _qlm(EPO_RUN)
    assert df["mode"].isin(["technical", "semantic"]).all(), "EPO mode column unexpectedly empty"
    return df


def cp_per_query() -> pd.DataFrame:
    """Google-Patents per-query metrics with `mode` recovered from the QAC source.

    The eval-time question_level_metrics.csv has an empty `mode`; we recover it by parsing the
    query id (`<corpus_id>_q_<qlang>[_mt]`) and joining (corpus_id, question_language) against
    data/google_patents/qac/qac_chempatents.csv. Of 135 unique queries, 132 join to the QAC source;
    1 of those maps to a key carrying both modes (ambiguous) and is set to NaN, leaving 131 with an
    unambiguous mode. The 3 unmatched + 1 ambiguous query get mode=NaN and are dropped by
    mode-stratified figures.
    """
    df = _qlm(CP_RUN)
    parsed = df["query_id"].str.extract(r"^(?P<corpus_id>.+)_q_(?P<qlang>[a-z]{2})$")
    df["corpus_id"] = parsed["corpus_id"]
    df["qlang"] = parsed["qlang"]
    # `is_synthetic_translation` is the authoritative human-vs-MT marker (the 80 synthetic queries
    # are MT-translated *questions* over a human-translated corpus, which the benchmark allows; only
    # they give es/zh cross-lingual coverage). Main figures keep all queries; the human-vs-MT split
    # is reported as an appendix robustness check.
    df["is_synthetic"] = df["is_synthetic_translation"].astype(bool)

    qac = pd.read_csv(REPO_ROOT / "data" / "google_patents" / "qac" / "qac_chempatents.csv")
    key = (qac.groupby(["corpus_id", "question_language"])["mode"]
              .agg(n_modes="nunique", mode_val=lambda s: s.iloc[0]).reset_index())
    key["mode_resolved"] = np.where(key["n_modes"] == 1, key["mode_val"], np.nan)
    df = df.merge(
        key[["corpus_id", "question_language", "mode_resolved"]],
        left_on=["corpus_id", "qlang"], right_on=["corpus_id", "question_language"], how="left",
    )
    df["mode"] = df["mode_resolved"]

    # integrity check (per-query, model-independent): 131 unambiguous modes recovered out of 135
    nq = df.drop_duplicates("query_id")
    resolved = int(nq["mode"].notna().sum())
    assert resolved == 131, f"chem_patents mode-join changed: {resolved}/135 unique queries resolved (expected 131)"
    return df.drop(columns=["mode_resolved", "question_language"])


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
    print("epo per-query rows:", len(epo), "| unique queries:", epo["query_id"].nunique())
    print("  mode:", epo.drop_duplicates("query_id")["mode"].value_counts().to_dict())
    # anchor: EPO embeddinggemma technical vs semantic recall@10
    g = epo[epo["short"] == "embeddinggemma"]
    print("  EPO gemma R@10 technical:", round(g[g["mode"] == "technical"]["recall_at_10"].mean(), 3),
          "| semantic:", round(g[g["mode"] == "semantic"]["recall_at_10"].mean(), 3))
    print("models:", [short(m) for m in MODEL_ORDER])
