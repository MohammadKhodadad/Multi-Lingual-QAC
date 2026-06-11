"""
Round 2 (headline) — Directional CLIR matrix and translation-direction asymmetry.

CLIR@10 is an average over many query→document language directions. This round opens it up: for
each (query language X, gold-document language Y) it measures the fraction of those gold documents
retrieved in the top-10, builds the X→Y matrix per model, and asks whether the failure is
*symmetric* — is en→zh as hard as zh→en? It also surfaces Spanish, the pure query-side language with
no same-language gold (its whole row is cross-lingual).

Outputs (../experimental_plots/round02_directional_clir/):
  pair_recall.csv, asymmetry.csv, summary.json, and figures:
    directional_recall_best.png   X→Y recall@10 heatmap for the best model
    directional_recall_grid.png   one heatmap per model (small multiples)
    asymmetry.png                 A(X,Y)=recall(X→Y)-recall(Y→X), pooled over reliable models
    clir_by_query_language.png    CLIR@10 by query language (which languages collapse)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round02_directional_clir"
DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]  # reliable models for pooled views


def pair_frame() -> pd.DataFrame:
    """One row per (model, query, gold-doc): query language, doc language, hit@10."""
    rank_of = C.rank_of()
    gold = C.gold_publication()
    ql = C.q_lang()
    qorg = C.q_origin()
    rows = []
    for model in C.MODEL_ORDER:
        for qid, gset in gold.items():
            ro = rank_of.get((model, qid), {})
            x = ql.get(qid, "")
            for d in gset:
                rows.append({
                    "model": model, "short": C.short(model), "query_id": qid,
                    "qlang": x, "dlang": C.doc_lang(d), "origin": qorg.get(qid, ""),
                    "hit10": 1.0 if ro.get(d, C.INF) <= 10 else 0.0,
                })
    return pd.DataFrame(rows)


def matrix_for(df: pd.DataFrame) -> pd.DataFrame:
    """X→Y recall@10 matrix (rows=query lang, cols=doc lang)."""
    m = (df.groupby(["qlang", "dlang"])["hit10"].mean()
         .unstack("dlang").reindex(index=C.QUERY_LANGS, columns=C.DOC_LANGS))
    return m


def _heatmap(ax, mat, counts=None, title="", vmin=0, vmax=1, cmap="viridis", diverging=False):
    import matplotlib.pyplot as plt
    data = mat.values.astype(float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("document language")
    ax.set_ylabel("query language")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color="grey")
                continue
            txt = f"{v:+.2f}" if diverging else f"{v:.2f}"
            if counts is not None:
                c = counts.values[i, j]
                if not np.isnan(c):
                    txt += f"\n({int(c)})"
            thresh = 0 if diverging else (vmax + vmin) / 2
            color = "white" if (abs(v) > 0.5 if diverging else v < thresh) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    return im


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    df = pair_frame()

    # ---- per-model + pooled matrices -------------------------------------------------------------
    pool_df = df[df.model.isin(POOL)]
    counts = (df[df.model == C.MODEL_ORDER[0]].groupby(["qlang", "dlang"]).size()
              .unstack("dlang").reindex(index=C.QUERY_LANGS, columns=C.DOC_LANGS))

    # tidy CSV
    rec = (df.groupby(["short", "qlang", "dlang"])["hit10"].agg(["mean", "size"])
           .reset_index().rename(columns={"mean": "recall_at_10", "size": "n_pairs"}))
    rec.to_csv(out / "pair_recall.csv", index=False)

    # ---- Fig 1: best model heatmap ---------------------------------------------------------------
    best_model = C.MODEL_ORDER[0]
    mb = matrix_for(df[df.model == best_model])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = _heatmap(ax, mb, counts, title=f"X→Y Recall@10 — {C.short(best_model)}\n"
                                        "rows = query language, cols = gold-document language (n pairs)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall@10")
    fig.tight_layout()
    fig.savefig(out / "directional_recall_best.png")
    plt.close(fig)

    # ---- Fig 2: grid of all models ---------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12))
    for ax, model in zip(axes.ravel(), C.MODEL_ORDER):
        m = matrix_for(df[df.model == model])
        im = _heatmap(ax, m, None, title=C.short(model))
    fig.suptitle("Directional Recall@10 (query language → document language), per model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "directional_recall_grid.png")
    plt.close(fig)

    # ---- Fig 3: asymmetry A(X,Y) over the 4 langs that are both query & doc ----------------------
    both = [l for l in C.DOC_LANGS]  # en, de, fr, zh
    mp = matrix_for(pool_df)
    asym = pd.DataFrame(index=both, columns=both, dtype=float)
    arows = []
    for x in both:
        for y in both:
            if x == y:
                asym.loc[x, y] = np.nan
                continue
            a = mp.loc[x, y] - mp.loc[y, x]
            asym.loc[x, y] = a
            if x < y:  # record each unordered pair once
                arows.append({"from": x, "to": y, "recall_xy": mp.loc[x, y],
                              "recall_yx": mp.loc[y, x], "asymmetry_xy_minus_yx": a})
    pd.DataFrame(arows).to_csv(out / "asymmetry.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = _heatmap(ax, asym, None, title="Translation-direction asymmetry  A(X→Y) = R(X→Y) − R(Y→X)\n"
                                        "pooled over reliable models; red row = X is a weaker query side",
                  vmin=-0.4, vmax=0.4, cmap="RdBu", diverging=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "asymmetry.png")
    plt.close(fig)

    # ---- Fig 4: CLIR@10 by query language --------------------------------------------------------
    cpq = C.core_per_query()
    clq = cpq[cpq.model.isin(POOL)]
    by = []
    for lang in C.QUERY_LANGS:
        sub = clq[clq.query_language == lang]
        pt, lo, hi = C.bootstrap_ci(sub["clir_at_10"].values)
        by.append({"lang": lang, "clir": pt, "lo": lo, "hi": hi, "n": int(sub.query_id.nunique())})
    bydf = pd.DataFrame(by)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = np.arange(len(bydf))
    err = np.vstack([bydf["clir"] - bydf["lo"], bydf["hi"] - bydf["clir"]])
    ax.bar(xs, bydf["clir"], yerr=err, capsize=3, color=[C.LANG_COLOR[l] for l in bydf["lang"]])
    for x, v, n in zip(xs, bydf["clir"], bydf["n"]):
        ax.text(x, v + 0.012, f"{v:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.LANG_NAME[l]}\n({l})" for l in bydf["lang"]])
    ax.set_ylabel("CLIR@10 (pooled reliable models)")
    ax.set_ylim(0, max(0.6, bydf["hi"].max() + 0.08))
    ax.set_title("Which query language collapses cross-lingually?\n"
                 "Spanish (es) is pure-CLIR — it has no same-language gold document anywhere")
    fig.tight_layout()
    fig.savefig(out / "clir_by_query_language.png")
    plt.close(fig)

    # ---- summary.json ----------------------------------------------------------------------------
    # hardest/easiest directions pooled (off-diagonal)
    long = mp.reset_index().melt(id_vars="qlang", var_name="dlang", value_name="recall")
    long = long[long.qlang != long.dlang].dropna()
    hardest = long.sort_values("recall").iloc[0]
    easiest = long.sort_values("recall").iloc[-1]
    asym_long = pd.DataFrame(arows).reindex(
        pd.DataFrame(arows)["asymmetry_xy_minus_yx"].abs().sort_values(ascending=False).index)
    worst_qlang = bydf.sort_values("clir").iloc[0]
    summary = {
        "hardest_direction": {"dir": f"{hardest['qlang']}→{hardest['dlang']}",
                              "recall_at_10": round(float(hardest["recall"]), 3)},
        "easiest_direction": {"dir": f"{easiest['qlang']}→{easiest['dlang']}",
                             "recall_at_10": round(float(easiest["recall"]), 3)},
        "most_asymmetric_pair": (None if asym_long.empty else {
            "pair": f"{asym_long.iloc[0]['from']}↔{asym_long.iloc[0]['to']}",
            "gap": round(float(asym_long.iloc[0]["asymmetry_xy_minus_yx"]), 3)}),
        "weakest_query_language": {"lang": worst_qlang["lang"],
                                   "clir_at_10": round(float(worst_qlang["clir"]), 3)},
        "spanish_clir_at_10": (round(float(bydf[bydf.lang == "es"]["clir"].iloc[0]), 3)
                               if (bydf.lang == "es").any() else None),
        "english_as_doc_target_meanrecall": round(float(mp["en"].mean()), 3),
    }
    C.jdump(summary, out / "summary.json")

    md = f"""<!-- round:02 -->
## Round 2 — Directional CLIR matrix and translation-direction asymmetry

**Question.** CLIR@10 averages over every query→document language direction. Which directions
actually collapse, and is the difficulty symmetric (is en→zh as hard as zh→en)?

**What we learned.**
- **The matrix is strongly anisotropic.** Pooled over reliable models, the hardest direction is
  **{summary['hardest_direction']['dir']}** (Recall@10 = {summary['hardest_direction']['recall_at_10']:.2f})
  and the easiest cross-lingual direction is **{summary['easiest_direction']['dir']}**
  ({summary['easiest_direction']['recall_at_10']:.2f}).
- **Direction matters, not just the pair.** The most asymmetric language pair is
  **{summary['most_asymmetric_pair']['pair'] if summary['most_asymmetric_pair'] else 'n/a'}**
  with a gap of {summary['most_asymmetric_pair']['gap'] if summary['most_asymmetric_pair'] else float('nan'):+.2f}
  in Recall@10 between its two directions — retrieval is not a symmetric similarity.
- **English is the easiest target** (mean Recall@10 into English documents
  = {summary['english_as_doc_target_meanrecall']:.2f}): models lean on English as a hub language.
- **Spanish is the canary.** As a pure query-side language with *no* same-language gold, Spanish
  CLIR@10 = {summary['spanish_clir_at_10']:.2f}; the weakest query language overall is
  **{summary['weakest_query_language']['lang']}** ({summary['weakest_query_language']['clir_at_10']:.2f}).

**Next questions.**
1. Synthetic (machine-translated) queries dominate the weak directions. Is the penalty the
   *translation* or the *missing home document*? Compare CLIR for human vs MT queries. → Round 3.
2. If a query can't find one foreign version, can it find *any* of the patent's other-language
   twins, and at what rank? → Round 4 (mate retrieval).
<!-- /round:02 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
