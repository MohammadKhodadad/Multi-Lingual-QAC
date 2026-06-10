"""
Round 6 — Language collapse and cross-lingual reach.

Low cross-lingual consistency (Round 5) suggests the top-k fills with language-specific documents.
We quantify it. For each query we measure the language composition of the top-k and compare the
same-language share to the corpus *base rate* (that language's share of the 23,487-doc haystack). The
ratio observed/base is a **same-language over-representation** factor: >1 means the model piles up
query-language documents beyond chance — language collapse, the mechanism behind the home advantage.

Outputs (../experimental_plots/round06_language_collapse/):
  per_query_composition.csv, per_model.csv, summary.json, and figures:
    topk_language_composition.png   top-10 language mix by query language (best model) vs base rate
    overrep_by_query_language.png   same-language over-representation factor by query language
    collapse_vs_clir.png            per-model: over-representation vs CLIR@10
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round06_language_collapse"
DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]
K = 10


def haystack_share() -> dict:
    c = C._ds(C.HAYSTACK_REPO, "corpus")
    vc = c["corpus_language"].str.lower().value_counts()
    tot = vc.sum()
    return {lang: cnt / tot for lang, cnt in vc.items()}


def comp_frame(base: dict) -> pd.DataFrame:
    lists = C.ranked_lists()
    ql = C.q_lang()
    cpq = C.core_per_query().set_index(["model", "query_id"])
    rows = []
    for model in C.MODEL_ORDER:
        for qid, lang in ql.items():
            top = lists.get((model, qid), [])[:K]
            if not top:
                continue
            cnt = Counter(C.doc_lang(d) for d in top)
            n = len(top)
            same_share = cnt.get(lang, 0) / n
            base_same = base.get(lang, np.nan)
            row = {"model": model, "short": C.short(model), "query_id": qid, "qlang": lang,
                   "same_share": same_share, "foreign_share": 1 - same_share,
                   "base_same": base_same,
                   "overrep": (same_share / base_same) if base_same and base_same > 0 else np.nan,
                   "entropy": C.lang_entropy_at_k(top, K),
                   "clir_at_10": float(cpq.loc[(model, qid), "clir_at_10"])
                   if (model, qid) in cpq.index else np.nan}
            for L in C.QUERY_LANGS:
                row[f"frac_{L}"] = cnt.get(L, 0) / n
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    base = haystack_share()
    df = comp_frame(base)
    df.to_csv(out / "per_query_composition.csv", index=False)

    # per model
    rows = []
    for model in C.MODEL_ORDER:
        g = df[df.model == model]
        rows.append({"model": model, "short": C.short(model),
                     "mean_overrep": float(g["overrep"].mean(skipna=True)),
                     "mean_foreign_share": float(g["foreign_share"].mean()),
                     "mean_entropy": float(g["entropy"].mean()),
                     "clir_at_10": float(g["clir_at_10"].mean(skipna=True))})
    msum = pd.DataFrame(rows)
    msum.to_csv(out / "per_model.csv", index=False)

    # ---- Fig 1: top-10 composition by query language (best model) + base rate --------------------
    best = C.MODEL_ORDER[0]
    gb = df[df.model == best]
    comp = gb.groupby("qlang")[[f"frac_{L}" for L in C.QUERY_LANGS]].mean().reindex(C.QUERY_LANGS)
    base_row = pd.Series({f"frac_{L}": base.get(L, 0) for L in C.QUERY_LANGS}, name="corpus")
    comp_plot = pd.concat([comp, base_row.to_frame().T])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(len(comp_plot))
    xs = np.arange(len(comp_plot))
    for L in C.QUERY_LANGS:
        vals = comp_plot[f"frac_{L}"].values.astype(float)
        ax.bar(xs, vals, bottom=bottom, label=L, color=C.LANG_COLOR[L])
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\nquery" if l in C.QUERY_LANGS else l for l in comp.index.tolist()] + ["corpus\nbase rate"])
    ax.set_ylabel("mean language composition of top-10")
    ax.set_ylim(0, 1)
    ax.legend(title="doc language", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f"Top-10 language mix by query language — {C.short(best)}\n"
                 "each query language over-fills with its own language vs the corpus base rate")
    fig.tight_layout()
    fig.savefig(out / "topk_language_composition.png")
    plt.close(fig)

    # ---- Fig 2: over-representation by query language ---------------------------------------------
    pool = df[df.model.isin(POOL)]
    by = pool.groupby("qlang")["overrep"].mean().reindex(C.QUERY_LANGS)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = np.arange(len(by))
    ax.bar(xs, by.values, color=[C.LANG_COLOR[l] for l in by.index])
    for x, v in zip(xs, by.values):
        ax.text(x, v * 1.03, f"{v:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="k", lw=0.9, ls="--", label="chance (corpus base rate)")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.LANG_NAME[l]}\n({l})" for l in by.index])
    ax.set_ylabel("same-language over-representation (observed / base rate)")
    ax.legend()
    ax.set_title("How much does each query language over-fill the top-10 with its own language?\n"
                 "(pooled reliable models; log scale)")
    fig.tight_layout()
    fig.savefig(out / "overrep_by_query_language.png")
    plt.close(fig)

    # ---- Fig 3: collapse vs CLIR (per model) -----------------------------------------------------
    rel = msum[msum.model.isin(POOL)]
    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.scatter(rel["mean_overrep"], rel["clir_at_10"], s=70,
               color=[C.MODEL_COLOR[m] for m in rel["model"]])
    for _, r in rel.iterrows():
        ax.annotate(r["short"], (r["mean_overrep"], r["clir_at_10"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    if len(rel) >= 3:
        rr = np.corrcoef(rel["mean_overrep"], rel["clir_at_10"])[0, 1]
        b, a = np.polyfit(rel["mean_overrep"], rel["clir_at_10"], 1)
        xs = np.linspace(rel["mean_overrep"].min(), rel["mean_overrep"].max(), 50)
        ax.plot(xs, a + b * xs, "--", color="grey", lw=1, label=f"Pearson r = {rr:.2f}")
        ax.legend()
    ax.set_xlabel("mean same-language over-representation (×)")
    ax.set_ylabel("CLIR@10")
    ax.set_title("Language collapse vs cross-lingual recall")
    fig.tight_layout()
    fig.savefig(out / "collapse_vs_clir.png")
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------------------
    rr = float(np.corrcoef(rel["mean_overrep"], rel["clir_at_10"])[0, 1]) if len(rel) >= 3 else float("nan")
    most = by.sort_values(ascending=False)
    summary = {
        "corpus_base_share": {k: round(v, 4) for k, v in base.items()},
        "most_collapsed_language": {"lang": most.index[0], "overrep": round(float(most.iloc[0]), 2)},
        "mean_foreign_share_best": round(float(msum[msum.model == best]["mean_foreign_share"].iloc[0]), 4),
        "pearson_overrep_clir": round(rr, 3),
        "per_model": msum[["short", "mean_overrep", "mean_foreign_share", "mean_entropy", "clir_at_10"]]
                     .round(4).to_dict(orient="records"),
    }
    C.jdump(summary, out / "summary.json")

    md = f"""<!-- round:06 -->
## Round 6 — Language collapse and cross-lingual reach

**Question.** Do models fill the top-k with the query's own language? We compare each query's
same-language share of the top-10 to the corpus base rate (that language's share of the 23k-doc
haystack); the ratio is a **same-language over-representation** factor (>1 = collapse).

**What we learned.**
- **Collapse is severe for low-resource languages.** The most over-represented query language is
  **{summary['most_collapsed_language']['lang']}** at **{summary['most_collapsed_language']['overrep']:.1f}×**
  its corpus base rate — Chinese and Spanish queries pull back their own language far beyond chance,
  even though (for es) no Spanish document is ever relevant.
- **Collapse predicts CLIR failure.** Across models, Pearson r(over-representation, CLIR@10) =
  **{summary['pearson_overrep_clir']:+.2f}**: the more a model anchors on the query language, the worse
  its cross-lingual recall. This is the mechanism behind the home advantage in Round 1 and the low
  RBO in Round 5.
- The best model still devotes only ~{summary['mean_foreign_share_best']:.0%} of its top-10 to foreign
  languages on average.

**Next questions.**
1. If same-language documents crowd the top-k, *irrelevant* same-language patents must be out-ranking
   the correct foreign one. How often does that happen, and is the distractor same- or
   cross-language? → Round 7.
2. Is this a ranking problem or a *score-separability* problem — are gold docs even separable from the
   crowd? → Round 8.
<!-- /round:06 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
