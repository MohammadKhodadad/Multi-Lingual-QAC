"""
Claim D: the alias-graph "distractor latch".

Question: given a chemical concept named in several languages, can a retriever find the right
document across languages while ignoring documents about chemically-similar look-alikes?

All inputs are precomputed under reports/runs/alias_graph/. Candidates:
  D1  confusion-rate heatmap, model x query-language (publication lens) + ALL column + mean row.
  D2  universal attractors: look-alike compounds that most often out-rank gold, coloured by relation.
  D2H (hero) D1 + D2 side by side: "how often it happens" next to "what does it".
  D3  cross-lingual RBO per model (consistency ceiling ~0.39): same concept, 5 languages -> same docs?
  D4  score collapse: separability AUC(gold>look-alike) for confused vs non-confused queries.
  D5  the structure-question trap: Recall@10 and confusion by question type.

confusion = a chemically-similar wrong compound out-ranks every gold document for that query
            (publication lens: gold = the query's own patent + its translations).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fp_common as fp

AG = fp.AG_RUN
SHORT_ORDER = [fp.short(m) for m in fp.MODEL_ORDER]


def _conf_pub() -> pd.DataFrame:
    df = pd.read_csv(AG / "experimental_plots" / "round02_confusion" / "confusion_rate_publication.csv",
                     index_col=0)
    return df.reindex(SHORT_ORDER)


# --------------------------------------------------------------------------- D1
def _heatmap(ax, df, langs, title):
    cols = langs + ["ALL"]
    arr = df[cols].to_numpy()
    im = ax.imshow(arr, cmap="OrRd", vmin=0, vmax=max(0.5, np.nanmax(arr)), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([fp.LANG_NAME.get(c, "ALL") for c in cols], rotation=25, ha="right")
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
    ax.axvline(len(langs) - 0.5, color="white", lw=2.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.28 else "black",
                        fontweight="bold" if cols[j] == "ALL" else "normal")
    ax.set_title(title)
    return im


def d1_confusion_heatmap():
    df = _conf_pub()
    langs = ["en", "de", "fr", "es", "zh"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    im = _heatmap(ax, df, langs, "Confusion rate (%): a look-alike out-ranks all gold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="confusion rate")
    fp.save(fig, "claimD_D1_confusion_heatmap")
    fp.dump_data(df.reset_index().rename(columns={"index": "short"}), "claimD_D1_confusion_heatmap")


# --------------------------------------------------------------------------- D2
def _attractors(top_n=14):
    att = pd.read_csv(AG / "experimental_plots" / "round06_confusion_network" / "attractor_strength.csv")
    edges = pd.read_csv(AG / "experimental_plots" / "round06_confusion_network" / "confusion_edges.csv")
    rel = (edges.groupby(["wrong", "relation"])["weight"].sum().reset_index()
                .sort_values("weight", ascending=False).drop_duplicates("wrong")
                .set_index("wrong")["relation"])
    att = att.head(top_n).copy()
    att["relation"] = att["wrong"].map(rel).fillna("sibling")
    att["label"] = att["wrong"].str.replace(" macromolecule", "", regex=False)
    return att


def d2_attractors():
    att = _attractors()
    rel_color = {"sibling": "#4c72b0", "parent": "#dd8452"}
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ys = np.arange(len(att))[::-1]
    ax.barh(ys, att["weight"], color=[rel_color[r] for r in att["relation"]])
    ax.set_yticks(ys); ax.set_yticklabels(att["label"])
    ax.set_xlabel("times this compound out-ranked gold (summed over models)")
    ax.set_title("A few look-alike compounds account for most of the latching")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in rel_color.values()]
    ax.legend(handles, [f"{k} of the true concept" for k in rel_color], loc="lower right")
    fp.save(fig, "claimD_D2_attractors")
    fp.dump_data(att, "claimD_D2_attractors")


def d2h_hero():
    df = _conf_pub()
    att = _attractors(top_n=12)
    rel_color = {"sibling": "#4c72b0", "parent": "#dd8452"}
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2),
                             gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.5})
    im = _heatmap(axes[0], df, ["en", "de", "fr", "es", "zh"],
                  "How often a look-alike beats all gold (%)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03, label="confusion rate")
    ax = axes[1]
    ys = np.arange(len(att))[::-1]
    ax.barh(ys, att["weight"], color=[rel_color[r] for r in att["relation"]])
    ax.set_yticks(ys); ax.set_yticklabels(att["label"])
    ax.set_xlabel("times it out-ranked gold (Σ models)")
    ax.set_title("Which look-alikes do it")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in rel_color.values()]
    ax.legend(handles, list(rel_color), loc="lower right")
    fig.suptitle("The distractor latch: chemically-similar compounds out-rank the right document",
                 fontsize=13, fontweight="bold", y=1.03)
    fp.save(fig, "claimD_D2H_hero")


# --------------------------------------------------------------------------- D3
def d3_rbo():
    agm = pd.read_csv(AG / "experimental_plots" / "round01_ranking_agreement" / "per_model_agreement.csv")
    agm = agm.set_index("short").reindex(SHORT_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ys = np.arange(len(agm))[::-1]
    for y, (_, r) in zip(ys, agm.iterrows()):
        model = [m for m in fp.MODEL_ORDER if fp.short(m) == r["short"]][0]
        ax.barh(y, r["rbo"], color=fp.MODEL_COLOR[model], alpha=0.85)
        ax.plot([r["rbo_lo"], r["rbo_hi"]], [y, y], color="k", lw=1.4)
    ax.set_yticks(ys); ax.set_yticklabels(agm["short"])
    ax.axvline(1.0, color="#2ca02c", ls="--", lw=1.4)
    ax.text(0.99, len(agm) - 0.5, "language-agnostic\nideal = 1.0", ha="right", va="top",
            fontsize=8.5, color="#2ca02c")
    best = agm["rbo"].max()
    ax.axvline(best, color="#888", ls=":", lw=1.2)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("cross-lingual RBO  (same concept in 5 languages → same ranked documents?)")
    ax.set_title("Even the best model returns mostly different documents per language (RBO ≈ 0.39)")
    fp.save(fig, "claimD_D3_rbo")
    fp.dump_data(agm, "claimD_D3_rbo")


# --------------------------------------------------------------------------- D6 (RBO + publication-lens confusion)
def d6_rbo_publication():
    """fig2's publication-lens confusion panel, with the concept-lens panel replaced by the
    cross-lingual RBO bars (D3): the two alias-graph failure modes side by side — inconsistent ranked
    lists across languages (left) and latching onto chemically-similar look-alikes (right)."""
    agm = (pd.read_csv(AG / "experimental_plots" / "round01_ranking_agreement" / "per_model_agreement.csv")
             .set_index("short").reindex(SHORT_ORDER).reset_index())
    conf = _conf_pub()
    langs = ["en", "de", "fr", "es", "zh"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.0, 1.05]})

    # (a) cross-lingual RBO bars — embeddinggemma at top, aligned with the heatmap rows
    ys = np.arange(len(agm))
    for y, (_, r) in zip(ys, agm.iterrows()):
        model = [m for m in fp.MODEL_ORDER if fp.short(m) == r["short"]][0]
        axL.barh(y, r["rbo"], color=fp.MODEL_COLOR[model], alpha=0.85)
        axL.plot([r["rbo_lo"], r["rbo_hi"]], [y, y], color="k", lw=1.4)
    axL.set_ylim(-0.5, len(agm) - 0.5); axL.invert_yaxis()
    axL.set_yticks(ys); axL.set_yticklabels(agm["short"])
    axL.axvline(1.0, color="#2ca02c", ls="--", lw=1.4)
    axL.text(0.99, -0.45, "language-agnostic\nideal = 1.0", ha="right", va="top", fontsize=8.5, color="#2ca02c")
    axL.axvline(float(agm["rbo"].max()), color="#888", ls=":", lw=1.2)
    axL.set_xlim(0, 1.02)
    axL.set_xlabel("cross-lingual RBO  (same concept, 5 languages → same ranked docs?)")
    axL.set_title("(a) Cross-lingual consistency (RBO ≈ 0.39 even at best)")

    # (b) publication-lens confusion heatmap — y-labels suppressed (shared with panel a, same order)
    im = _heatmap(axR, conf, langs, "(b) Publication-lens confusion: look-alike out-ranks all gold")
    axR.set_yticklabels([])
    fig.colorbar(im, ax=axR, fraction=0.046, pad=0.03, label="confusion rate")

    fig.suptitle("Alias-graph failure modes: rankings disagree across languages (a) and latch onto "
                 "chemical look-alikes (b)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fp.save(fig, "claimD_D6_rbo_publication")
    fp.dump_data(agm, "claimD_D6_rbo_publication")


# --------------------------------------------------------------------------- D4
def d4_score_collapse():
    sep = pd.read_csv(AG / "experimental_plots" / "round09_score_separability" / "separability_per_query.csv")
    conf = pd.read_csv(AG / "confusion" / "per_query.csv")
    m = sep.merge(conf[["model", "query_id", "win"]], on=["model", "query_id"], how="inner")
    confused = m[m["win"] == 1]["auc"].dropna().to_numpy()
    ok = m[m["win"] == 0]["auc"].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    parts = ax.violinplot([ok, confused], positions=[0, 1], showmeans=True, widths=0.8)
    for pc, c in zip(parts["bodies"], ["#4c72b0", "#c44e52"]):
        pc.set_facecolor(c); pc.set_alpha(0.55)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("k")
    ax.axhline(0.5, color="#888", ls="--", lw=1.2)
    ax.text(1.45, 0.505, "chance (0.5)", fontsize=8.5, color="#666", va="bottom")
    ax.scatter([0, 1], [ok.mean(), confused.mean()], color="k", zorder=5)
    ax.annotate(f"{ok.mean():.2f}", (0, ok.mean()), xytext=(8, 0),
                textcoords="offset points", fontsize=10, va="center")
    ax.annotate(f"{confused.mean():.2f}", (1, confused.mean()), xytext=(8, 0),
                textcoords="offset points", fontsize=10, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"not confused\n(n={len(ok)})", f"confused\n(n={len(confused)})"])
    ax.set_ylabel("separability AUC  (gold scored above look-alike)")
    ax.set_title("When the model is confused, gold and look-alike scores collapse to a coin-flip")
    fp.save(fig, "claimD_D4_score_collapse")
    fp.dump_data(pd.DataFrame({"group": ["not_confused", "confused"],
                               "mean_auc": [ok.mean(), confused.mean()],
                               "n": [len(ok), len(confused)]}), "claimD_D4_score_collapse")


# --------------------------------------------------------------------------- D5
def d5_structure_trap():
    f = pd.read_csv(AG / "experimental_plots" / "round07_question_surface" / "per_query_features.csv")
    order = ["role", "reaction", "parameter", "structure"]
    agg = (f.groupby("question_type")
             .agg(recall10=("recall10", "mean"), confusion=("win", "mean"), n=("win", "size"))
             .reindex(order))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    xs = np.arange(len(agg)); w = 0.38
    ax.bar(xs - w / 2, agg["recall10"], width=w, color="#55a868", label="Recall@10")
    ax.bar(xs + w / 2, agg["confusion"], width=w, color="#c44e52", label="confusion rate")
    for i, r in enumerate(agg.itertuples()):
        ax.text(i - w / 2, r.recall10 + 0.012, f"{r.recall10:.2f}", ha="center", fontsize=8.5)
        ax.text(i + w / 2, r.confusion + 0.012, f"{r.confusion:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t}\n(n={int(agg.loc[t, 'n'])})" for t in order])
    ax.set_ylabel("rate")
    ax.set_ylim(0, 0.8)
    ax.set_title("Structure-style questions are the trap: low recall, high confusion")
    ax.legend(loc="upper right")
    fp.save(fig, "claimD_D5_structure_trap")
    fp.dump_data(agg.reset_index(), "claimD_D5_structure_trap")


def main():
    fp.set_style()
    d1_confusion_heatmap()
    d2_attractors()
    d2h_hero()
    d3_rbo()
    d4_score_collapse()
    d5_structure_trap()
    d6_rbo_publication()
    print("claim D: D1, D2, D2H, D3, D4, D5, D6 written to candidates/")


if __name__ == "__main__":
    main()
