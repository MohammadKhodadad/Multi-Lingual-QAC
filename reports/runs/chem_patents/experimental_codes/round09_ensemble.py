"""
Round 9 — Complementarity: oracle, rank-fusion, and per-language routing.

Different models collapse and under-score along different languages (Rounds 6-8). Are their errors
complementary enough that combining them recovers buried twins? We measure three things against the
best single model's CLIR@10:
  * RRF fusion    — Reciprocal Rank Fusion of the reliable models (a real, deployable ensemble).
  * per-language router — route each query language to its best model (an oracle upper bound on routing).
  * oracle        — a query counts if ANY reliable model surfaces the foreign gold in top-10 (ceiling).

Outputs (../experimental_plots/round09_ensemble/):
  per_query_ensemble.csv, summary.json, and figures:
    ensemble_overview.png        best-single vs RRF vs router vs oracle (CLIR@10)
    oracle_headroom_by_language.png  best-single vs oracle CLIR@10 per query language
    best_model_by_language.png   which model wins each query language
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round09_ensemble"
DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]
RRF_C = 60
DEPTH = 100
K = 10


def rrf_ranking(qid: str, models: list) -> list:
    lists = C.ranked_lists()
    score: dict = defaultdict(float)
    for m in models:
        for rank, d in enumerate(lists.get((m, qid), [])[:DEPTH], start=1):
            score[d] += 1.0 / (RRF_C + rank)
    return [d for d, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    lists = C.ranked_lists()
    gold = C.gold_publication()
    ql = C.q_lang()

    # per-model CLIR@10 by language (for the router)
    cpq = C.core_per_query()
    pool_cpq = cpq[cpq.model.isin(POOL)]
    clir_by_ml = (pool_cpq.groupby(["model", "query_language"])["clir_at_10"].mean().unstack("model"))
    best_model_per_lang = {lang: clir_by_ml.loc[lang].idxmax() for lang in clir_by_ml.index}

    model_clir = pool_cpq.groupby("model")["clir_at_10"].mean().sort_values(ascending=False)
    best_single_model = model_clir.index[0]
    TOP = list(model_clir.index[:4])      # fuse the strong models (naive all-model fusion backfires)

    rows = []
    for qid, gset in gold.items():
        lang = ql.get(qid, "")
        _, cross = C.split_gold(gset, lang)
        if not cross:
            continue
        # best single (global) model
        bs = C.recall_at_k(lists.get((best_single_model, qid), []), cross, K)
        # RRF fusion of the top-4 vs all reliable models
        rrf = C.recall_at_k(rrf_ranking(qid, TOP), cross, K)
        rrf_all = C.recall_at_k(rrf_ranking(qid, POOL), cross, K)
        # per-language router (oracle choice of model for this language)
        rmodel = best_model_per_lang.get(lang, best_single_model)
        router = C.recall_at_k(lists.get((rmodel, qid), []), cross, K)
        # oracle union over reliable models
        union = set()
        for m in POOL:
            union |= set(lists.get((m, qid), [])[:K])
        oracle = len(union & cross) / len(cross)
        rows.append({"query_id": qid, "qlang": lang, "best_single": bs, "rrf": rrf,
                     "rrf_all": rrf_all, "router": router, "oracle": oracle})
    pq = pd.DataFrame(rows)
    pq.to_csv(out / "per_query_ensemble.csv", index=False)

    means = {c: float(pq[c].mean()) for c in ["best_single", "rrf", "rrf_all", "router", "oracle"]}

    # ---- Fig 1: overview -------------------------------------------------------------------------
    labels = [f"best single\n({C.short(best_single_model)})", "RRF fusion\n(top-4)",
              "RRF fusion\n(all reliable)", "per-language\nrouter (oracle)", "oracle\n(any model)"]
    vals = [means["best_single"], means["rrf"], means["rrf_all"], means["router"], means["oracle"]]
    colors = ["#7f7f7f", "#1f77b4", "#aec7e8", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=colors)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CLIR@10 (cross-language recall)")
    ax.set_ylim(0, 1)
    ax.set_title("Complementarity headroom: fusing models recovers buried foreign twins\n"
                 "oracle = a query is solved if ANY reliable model finds the twin in top-10")
    fig.tight_layout()
    fig.savefig(out / "ensemble_overview.png")
    plt.close(fig)

    # ---- Fig 2: oracle headroom by language ------------------------------------------------------
    by = pq.groupby("qlang")[["best_single", "rrf", "oracle"]].mean().reindex(C.QUERY_LANGS)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    xs = np.arange(len(by))
    w = 0.27
    ax.bar(xs - w, by["best_single"], w, label=f"best single ({C.short(best_single_model)})", color="#7f7f7f")
    ax.bar(xs, by["rrf"], w, label="RRF fusion", color="#1f77b4")
    ax.bar(xs + w, by["oracle"], w, label="oracle (any model)", color="#2ca02c")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.LANG_NAME[l]}\n({l})" for l in by.index])
    ax.set_ylabel("CLIR@10")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Where is the headroom? Cross-lingual recall by query language")
    fig.tight_layout()
    fig.savefig(out / "oracle_headroom_by_language.png")
    plt.close(fig)

    # ---- Fig 3: best model per language ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    mat = clir_by_ml.reindex(index=C.QUERY_LANGS)[[m for m in POOL if m in clir_by_ml.columns]]
    data = mat.values.astype(float)
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0, vmax=np.nanmax(data))
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([C.short(m) for m in mat.columns], rotation=30, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"{l}" for l in mat.index])
    for i in range(mat.shape[0]):
        best_j = int(np.nanargmax(data[i]))
        for j in range(mat.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.3 else "black",
                    fontweight="bold" if j == best_j else "normal")
        ax.add_patch(plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CLIR@10")
    ax.set_xlabel("model")
    ax.set_ylabel("query language")
    ax.set_title("Best model differs by query language (red = winner) → routing has value")
    fig.tight_layout()
    fig.savefig(out / "best_model_by_language.png")
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------------------
    summary = {
        "best_single_model": C.short(best_single_model),
        "fused_models_top4": [C.short(m) for m in TOP],
        "best_single_clir10": round(means["best_single"], 4),
        "rrf_top4_clir10": round(means["rrf"], 4),
        "rrf_all_clir10": round(means["rrf_all"], 4),
        "router_clir10": round(means["router"], 4),
        "oracle_clir10": round(means["oracle"], 4),
        "rrf_top4_gain": round(means["rrf"] - means["best_single"], 4),
        "rrf_all_gain": round(means["rrf_all"] - means["best_single"], 4),
        "oracle_headroom": round(means["oracle"] - means["best_single"], 4),
        "best_model_per_language": {l: C.short(m) for l, m in best_model_per_lang.items()},
        "n_distinct_winners": len(set(best_model_per_lang.values())),
    }
    C.jdump(summary, out / "summary.json")

    md = f"""<!-- round:09 -->
## Round 9 — Complementarity: oracle, rank-fusion and routing

**Question.** Models fail along different languages. Do they fail on the *same* queries, or are their
errors complementary enough to combine? Baseline = best single model
({C.short(best_single_model)}, CLIR@10 = {means['best_single']:.2f}).

**What we learned.**
- **Untuned fusion does *not* beat a dominant model.** RRF over the top-4
  ({", ".join(summary['fused_models_top4'])}) lands at CLIR@10 = **{means['rrf']:.2f}**
  ({summary['rrf_top4_gain']:+.2f} vs the best single model), and fusing *all* reliable models is worse
  still ({means['rrf_all']:.2f}, {summary['rrf_all_gain']:+.2f}). `embeddinggemma` is strong enough
  that rank-averaging with weaker models only injects noise — a caution against reflexive ensembling.
- **Yet complementarity is real.** The oracle (any reliable model finds the foreign twin in top-10)
  reaches **{means['oracle']:.2f}** — a headroom of **{summary['oracle_headroom']:+.2f}** over the best
  single model. The twins *are* findable; no single model finds all of them, so a *score-aware* or
  *learned* combiner (not plain RRF) is where the gains live.
- **Routing helps, slightly.** The best model differs across query languages
  ({summary['n_distinct_winners']} distinct winners over 5 languages); an oracle per-language router
  edges past the best single model (CLIR@10 = {means['router']:.2f}). The remaining headroom is
  concentrated in the lower-resource / homeless languages, where no current model is strong.

**Next questions.**
1. We now have six orthogonal lenses (accuracy, CLIR, consistency, MT-robustness, collapse,
   separability). Fold them into one multilingual-robustness verdict and a final recommendation. → Round 10.
<!-- /round:09 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
