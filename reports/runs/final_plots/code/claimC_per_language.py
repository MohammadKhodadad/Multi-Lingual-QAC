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


def _cp_common():
    """Lazy import of the chem_patents analysis module (loads cached HF datasets)."""
    sys.path.insert(0, str(fp.CP_RUN / "experimental_codes"))
    import common as cpc  # noqa: E402
    return cpc


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
    cpc = _cp_common()
    cpq = cpc.core_per_query()  # has clir_at_10 (cross-lang gold) & molir_at_10 (same-lang gold)
    # language-balanced over models: per (query_language) average the per-query metric across all models
    rows = []
    for lang in fp.CP_LANGS:
        sub = cpq[cpq["query_language"] == lang]
        molir = sub["molir_at_10"].dropna()
        clir = sub["clir_at_10"].dropna()
        rows.append({"query_language": lang,
                     "molir": float(molir.mean()) if len(molir) else np.nan,
                     "clir": float(clir.mean()) if len(clir) else np.nan,
                     "n_same_gold_q": int(molir.shape[0]), "n_cross_gold_q": int(clir.shape[0])})
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
    cpc = _cp_common()
    qm = cpc.query_meta()  # all queries (the population the per-language figures use)
    n_q = qm["query_language"].value_counts().reindex(fp.CP_LANGS).fillna(0).astype(int)

    # gold qrels by gold-document language
    gold = cpc.gold_publication()
    gl = {l: 0 for l in fp.CP_LANGS}
    for qid, docs in gold.items():
        for d in docs:
            dl = cpc.doc_lang(d)
            if dl in gl:
                gl[dl] += 1
    n_gold = pd.Series(gl).reindex(fp.CP_LANGS).fillna(0).astype(int)

    # haystack share by document language
    clang = pd.Series(cpc.corpus_lang())
    hay = clang.value_counts()
    n_doc = hay.reindex(fp.CP_LANGS).fillna(0).astype(int)

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


def main():
    fp.set_style()
    np.random.seed(0)
    c1_heatmap()
    c2_bars()
    c3_home_advantage()
    c4_epo_heatmap()
    c5_denominators()
    print("claim C: C1-C5 written to candidates/")


if __name__ == "__main__":
    main()
