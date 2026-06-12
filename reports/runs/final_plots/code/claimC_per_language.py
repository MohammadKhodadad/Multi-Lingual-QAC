"""
Claim C: per-language results, normalized so unequal language sizes don't mislead.

Candidates
  C1  model x query-language Recall@10 heatmap (Google Patents, 5 langs, human queries),
      with a language-balanced 'mean' column so the reader sees the normalized headline.
  C2  per-language Recall@10 bars with 95% bootstrap CIs; per-model dots overlaid.
  C3  the honest 'home advantage' view: each language's gold split into same-language (MoLIR)
      vs cross-language (CLIR) recall — English's apparent lead is mostly gold-at-home, not skill.
  C4  (appendix) EPO 3-language mirror of C1.
  C5  (appendix) denominator transparency: queries / gold qrels / haystack share per language —
      the justification for language-balancing rather than pooling.
  C6  directional matrix: 8-panel grid, one query-language x document-language Recall@10 heatmap per
      model (diagonal = same-language, off-diagonal = cross-lingual).
  C7  the same data as a single routes x models heatmap (all four dimensions in one panel).

Normalization
  * within-language means first, then language-balanced macro mean for any single headline number.
  * GP main panels use human-original queries; CLIR/MoLIR come from the shared `common` per-query frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fp_common as fp


def _gp():
    return fp.cp_per_query()


# --------------------------------------------------------------------------- C1
def c1_heatmap():
    df = _gp()
    mat = (df.groupby(["short", "query_language"])["recall_at_10"].mean()
             .unstack("query_language").reindex(index=[fp.short(m) for m in fp.MODEL_ORDER],
                                                columns=fp.CP_LANGS))
    bal = mat.mean(axis=1)  # language-balanced mean (equal weight per language)
    full = mat.copy()
    full["balanced"] = bal
    cols = fp.CP_LANGS + ["balanced"]
    arr = full[cols].to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=max(0.7, np.nanmax(arr)), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([fp.LANG_NAME.get(c, "balanced\nmean") for c in cols], rotation=25, ha="right")
    ax.set_yticks(range(len(full.index))); ax.set_yticklabels(full.index)
    ax.axvline(len(fp.CP_LANGS) - 0.5, color="white", lw=2.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v < 0.42 else "black",
                        fontweight="bold" if j == len(fp.CP_LANGS) else "normal")
    ax.set_title("Recall@10 by query language (Google Patents)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Recall@10")
    fp.save(fig, "claimC_C1_heatmap")
    fp.dump_data(full.reset_index(), "claimC_C1_heatmap")


# --------------------------------------------------------------------------- C2
def c2_bars():
    df = _gp()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    langs = fp.CP_LANGS
    rows = []
    xs = np.arange(len(langs))
    for li, lang in enumerate(langs):
        sub = df[df["query_language"] == lang]
        pt, lo, hi = fp.bootstrap_ci(sub["recall_at_10"].dropna().to_numpy())
        ax.bar(li, pt, color=fp.LANG_COLOR[lang], alpha=0.55, width=0.66, zorder=1)
        ax.errorbar(li, pt, yerr=[[pt - lo], [hi - pt]], color="k", capsize=4, lw=1.2, zorder=3)
        rows.append({"query_language": lang, "mean_recall10": pt, "lo": lo, "hi": hi})
        # per-model dots
        for model in fp.MODEL_ORDER:
            v = sub[sub["model"] == model]["recall_at_10"].mean()
            ax.scatter(li + np.random.uniform(-0.16, 0.16), v, s=26,
                       color=fp.MODEL_COLOR[model], edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([fp.LANG_NAME[l] for l in langs])
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 per query language — pooled bar (95% CI) with per-model dots")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=fp.MODEL_COLOR[m],
               label=fp.short(m), ms=6) for m in fp.MODEL_ORDER]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=7.5)
    fp.save(fig, "claimC_C2_bars")
    fp.dump_data(pd.DataFrame(rows), "claimC_C2_bars")


# --------------------------------------------------------------------------- C3
def c3_home_advantage():
    cpq = fp.cp_per_query()  # has clir_at_10 (cross-lang gold) & molir_at_10 (same-lang gold), 524 queries
    # language-balanced over models: per (query_language) average the per-query metric across all models
    rows = []
    for lang in fp.CP_LANGS:
        sub = cpq[cpq["query_language"] == lang]
        molir = sub["molir_at_10"].dropna()
        clir = sub["clir_at_10"].dropna()
        rows.append({"query_language": lang,
                     "molir": float(molir.mean()) if len(molir) else np.nan,
                     "clir": float(clir.mean()) if len(clir) else np.nan,
                     "n_same_gold_q": int(molir.shape[0]) // 8, "n_cross_gold_q": int(clir.shape[0]) // 8})
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    xs = np.arange(len(d))
    w = 0.38
    ax.bar(xs - w / 2, d["molir"], width=w, color="#4c72b0", label="same-language gold (MoLIR@10)")
    ax.bar(xs + w / 2, d["clir"], width=w, color="#c44e52", label="cross-language gold (CLIR@10)")
    for i, r in d.iterrows():
        if not np.isnan(r["molir"]):
            ax.text(i - w / 2, r["molir"] + 0.012, f"{r['molir']:.2f}", ha="center", fontsize=8)
        if not np.isnan(r["clir"]):
            ax.text(i + w / 2, r["clir"] + 0.012, f"{r['clir']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([fp.LANG_NAME[l] for l in d["query_language"]])
    ax.set_ylabel("Recall@10")
    ax.set_title("Per-language recall splits into gold-at-home vs cross-language gold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(0.8, np.nanmax(d[["molir", "clir"]].to_numpy()) * 1.2))
    fp.save(fig, "claimC_C3_home_advantage")
    fp.dump_data(d, "claimC_C3_home_advantage")


# --------------------------------------------------------------------------- C4 (appendix)
def c4_epo_heatmap():
    df = fp.epo_per_query()
    mat = (df.groupby(["short", "query_language"])["recall_at_10"].mean()
             .unstack("query_language").reindex(index=[fp.short(m) for m in fp.MODEL_ORDER],
                                                columns=fp.EPO_LANGS))
    full = mat.copy(); full["balanced"] = mat.mean(axis=1)
    cols = fp.EPO_LANGS + ["balanced"]
    arr = full[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=max(0.7, np.nanmax(arr)), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([fp.LANG_NAME.get(c, "balanced\nmean") for c in cols], rotation=25, ha="right")
    ax.set_yticks(range(len(full.index))); ax.set_yticklabels(full.index)
    ax.axvline(len(fp.EPO_LANGS) - 0.5, color="white", lw=2.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v < 0.42 else "black",
                        fontweight="bold" if j == len(fp.EPO_LANGS) else "normal")
    ax.set_title("Recall@10 by query language (EPO)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Recall@10")
    fp.save(fig, "claimC_C4_epo_heatmap")
    fp.dump_data(full.reset_index(), "claimC_C4_epo_heatmap")


# --------------------------------------------------------------------------- C5 (appendix)
def c5_denominators():
    den = fp.gp_denominators().set_index("query_language").reindex(fp.CP_LANGS)
    n_q = den["n_queries"]
    n_gold = den["n_gold_qrels"]
    n_doc = den["n_haystack_docs"]

    panels = [("queries (by query language)", n_q), ("gold qrels (by doc language)", n_gold),
              ("haystack documents", n_doc)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    for ax, (title, ser) in zip(axes, panels):
        ax.bar(range(len(ser)), ser.to_numpy(),
               color=[fp.LANG_COLOR[l] for l in fp.CP_LANGS])
        ax.set_xticks(range(len(ser)))
        ax.set_xticklabels([fp.LANG_NAME[l] for l in fp.CP_LANGS], rotation=30, ha="right")
        ax.set_title(title)
        for i, v in enumerate(ser.to_numpy()):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.18)
    fig.suptitle("Per-language denominators are unequal — why per-language means must be language-balanced",
                 fontsize=12, fontweight="bold", y=1.04)
    fp.save(fig, "claimC_C5_denominators")
    out = pd.DataFrame({"query_language": fp.CP_LANGS, "n_queries": n_q.to_numpy(),
                        "n_gold_qrels": n_gold.to_numpy(), "n_haystack_docs": n_doc.to_numpy()})
    fp.dump_data(out, "claimC_C5_denominators")


# --------------------------------------------------------------------------- C6/C7 data
def _query_doc_recall() -> pd.DataFrame:
    """Tidy frame: (model, query_language, doc_language) -> mean Recall@10 over the gold docs that
    live in doc_language, for queries asked in query_language. Diagonal = same-language gold."""
    from collections import defaultdict
    clang, _ = fp._gp_corpus_family()
    tab = fp.gp_query_table()
    rk = fp.gp_rankings()
    top = rk[rk["rank"] <= 10].groupby(["model", "query_id"])["corpus_id"].apply(set).to_dict()
    rows = []
    for model in fp.MODEL_ORDER:
        for q in tab.itertuples():
            t = top.get((model, q.query_id), set())
            by_lang = defaultdict(set)
            for d in q.gold:
                by_lang[clang.get(d)].add(d)
            for dl, gset in by_lang.items():
                if dl in fp.CP_LANGS:
                    rows.append({"model": model, "short": fp.short(model),
                                 "query_language": q.query_language, "doc_language": dl,
                                 "recall": len(t & gset) / len(gset)})
    return pd.DataFrame(rows)


def _cell(ax, mat, langs, title, annotate=True, vmax=0.8):
    arr = np.ma.masked_invalid(mat.to_numpy())
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#dddddd")
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(langs))); ax.set_yticks(range(len(langs)))
    ax.set_xticklabels([l.upper() for l in langs], fontsize=8)
    ax.set_yticklabels([l.upper() for l in langs], fontsize=8)
    for i in range(len(langs)):  # outline the same-language diagonal
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", lw=1.6))
        if annotate:
            for j in range(len(langs)):
                v = mat.to_numpy()[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v < 0.45 else "black")
    ax.set_title(title, fontsize=10)
    return im


# --------------------------------------------------------------------------- C6
def c6_query_doc_grid():
    df = _query_doc_recall()
    fp.dump_data(df, "claimC_C6_query_doc_grid")
    langs = fp.CP_LANGS
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2))
    im = None
    for ax, model in zip(axes.ravel(), fp.MODEL_ORDER):
        sub = df[df["model"] == model]
        mat = (sub.groupby(["query_language", "doc_language"])["recall"].mean()
                  .unstack("doc_language").reindex(index=langs, columns=langs))
        im = _cell(ax, mat, langs, fp.short(model))
    for ax in axes[1, :]:
        ax.set_xlabel("document language", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("query language", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Recall@10")
    fig.suptitle("Recall@10 by query language × document language, per model "
                 "(white box = same-language; off-diagonal = cross-lingual)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fp.save(fig, "claimC_C6_query_doc_grid")


# --------------------------------------------------------------------------- C7
def c7_route_model_matrix():
    df = _query_doc_recall()
    langs = fp.CP_LANGS
    # rows = routes (query→doc), same-language routes first then cross, grouped by query language
    routes = [(q, d) for q in langs for d in langs]
    routes = sorted(routes, key=lambda qd: (qd[0] != qd[1], langs.index(qd[0]), langs.index(qd[1])))
    rl = {r: i for i, r in enumerate(routes)}
    mat = np.full((len(routes), len(fp.MODEL_ORDER)), np.nan)
    g = df.groupby(["query_language", "doc_language", "model"])["recall"].mean()
    for (q, d, model), v in g.items():
        if (q, d) in rl and model in fp.MODEL_ORDER:
            mat[rl[(q, d)], fp.MODEL_ORDER.index(model)] = v
    n_same = sum(1 for q, d in routes if q == d)

    fig, ax = plt.subplots(figsize=(9.0, 9.5))
    arr = np.ma.masked_invalid(mat)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#dddddd")
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(fp.MODEL_ORDER)))
    ax.set_xticklabels([fp.short(m) for m in fp.MODEL_ORDER], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(routes)))
    ax.set_yticklabels([f"{q.upper()}→{d.upper()}" + ("  (same)" if q == d else "") for q, d in routes],
                       fontsize=8)
    ax.axhline(n_same - 0.5, color="#c44e52", lw=2)  # separate same-language routes from cross
    for i in range(len(routes)):
        for j in range(len(fp.MODEL_ORDER)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v < 0.45 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Recall@10")
    ax.set_title("Recall@10 for every query→document language route × model\n"
                 "(red line splits same-language routes from cross-lingual)",
                 fontsize=11, fontweight="bold")
    fp.save(fig, "claimC_C7_route_model_matrix")
    fp.dump_data(df, "claimC_C7_route_model_matrix")


def main():
    fp.set_style()
    np.random.seed(0)
    c1_heatmap()
    c2_bars()
    c3_home_advantage()
    c4_epo_heatmap()
    c5_denominators()
    c6_query_doc_grid()
    c7_route_model_matrix()
    print("claim C: C1-C7 written to candidates/")


if __name__ == "__main__":
    main()
