"""
Claim A: technical questions are retrieved worse than semantic questions.

Candidates
  A1  paired dumbbell per model, semantic -> technical Recall@10, two panels (GP | EPO),
      sorted by semantic, language-balanced means with 95% bootstrap CIs.
  A2  per-query Recall@10 distribution by mode: pooled ECDF + per-model semantic/technical means.
  A3  (appendix) gap = R@10(semantic) - R@10(technical), model x language heatmap (both sources).
  A4  (appendix) metric robustness: A1 repeated for MRR@10 and Hit@10.

Normalization
  * language-balanced means (macro over query languages) so unequal per-language counts can't bias.
  * Google-Patents keeps all queries (the 80 MT-translated questions give es/zh coverage and are
    allowed by the benchmark); A5 shows the human-vs-MT split as a robustness check.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fp_common as fp


def _gp():
    df = fp.cp_per_query()
    return df[df["mode"].isin(["technical", "semantic"])].copy()


def _epo():
    return fp.epo_per_query()


def _per_model_mode(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """(model, mode) -> language-balanced mean + 95% CI of `value`."""
    rows = []
    for model in fp.MODEL_ORDER:
        for mode in ("semantic", "technical"):
            sub = df[(df["model"] == model) & (df["mode"] == mode)]
            if sub.empty:
                continue
            pt, lo, hi = fp.lang_balanced_ci(sub, value)
            rows.append({"model": model, "short": fp.short(model), "mode": mode,
                         "value": pt, "lo": lo, "hi": hi, "n": len(sub)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- A1
def a1_dumbbell():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    out = []
    for ax, (df, title, tag) in zip(
        axes, [(_gp(), "Google Patents (5 languages)", "GP"),
               (_epo(), "EPO (3 languages)", "EPO")]):
        agg = _per_model_mode(df, "recall_at_10")
        agg["source"] = tag
        out.append(agg)
        piv = agg.pivot(index="short", columns="mode", values="value")
        piv = piv.sort_values("semantic")
        ys = np.arange(len(piv))
        for y, (name, r) in zip(ys, piv.iterrows()):
            ax.plot([r["technical"], r["semantic"]], [y, y], color="#bbbbbb", lw=2.4, zorder=1)
        ax.scatter(piv["technical"], ys, s=70, color=fp.MODE_COLOR["technical"],
                   zorder=3, label="technical", edgecolor="white", linewidth=0.8)
        ax.scatter(piv["semantic"], ys, s=70, color=fp.MODE_COLOR["semantic"],
                   zorder=3, label="semantic", edgecolor="white", linewidth=0.8)
        # CI whiskers
        for mode in ("technical", "semantic"):
            sub = agg[agg["mode"] == mode].set_index("short").reindex(piv.index)
            ax.hlines(ys, sub["lo"], sub["hi"], color=fp.MODE_COLOR[mode], lw=1.2, alpha=0.55, zorder=2)
        ax.set_yticks(ys)
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("Recall@10  (language-balanced)")
        ax.set_title(title)
        ax.set_xlim(0, max(0.8, piv.max().max() * 1.15))
    axes[0].legend(loc="lower right")
    fig.suptitle("Technical questions retrieve worse than semantic ones — every model, both offices",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fp.save(fig, "claimA_A1_dumbbell")
    fp.dump_data(pd.concat(out, ignore_index=True), "claimA_A1_dumbbell")


# --------------------------------------------------------------------------- A2
def a2_distribution():
    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)

    # left: pooled ECDF of per-query R@10 by mode (both sources stacked)
    ax0 = fig.add_subplot(gs[0])
    pooled = []
    for df, tag in [(_gp(), "GP"), (_epo(), "EPO")]:
        d = df.copy(); d["source"] = tag; pooled.append(d)
    pooled = pd.concat(pooled, ignore_index=True)
    for mode in ("semantic", "technical"):
        v = np.sort(pooled[pooled["mode"] == mode]["recall_at_10"].dropna().to_numpy())
        y = np.arange(1, len(v) + 1) / len(v)
        ax0.step(v, y, where="post", color=fp.MODE_COLOR[mode], lw=2.2, label=f"{mode}  (n={len(v)})")
    ax0.set_xlabel("per-query Recall@10")
    ax0.set_ylabel("cumulative fraction of queries")
    ax0.set_title("Per-query distribution (pooled)")
    ax0.legend(loc="lower right")

    # right: per-model semantic vs technical means, both sources as marker shapes
    ax1 = fig.add_subplot(gs[1])
    rows = []
    for df, tag, marker in [(_gp(), "GP", "o"), (_epo(), "EPO", "s")]:
        agg = _per_model_mode(df, "recall_at_10")
        piv = agg.pivot(index="short", columns="mode", values="value")
        for name, r in piv.iterrows():
            ax1.plot([r["technical"], r["semantic"]], [name, name],
                     color=fp.MODEL_COLOR.get([m for m in fp.MODEL_ORDER if fp.short(m) == name][0], "#999"),
                     lw=1.0, alpha=0.4, zorder=1)
            rows.append({"source": tag, "short": name,
                         "technical": r["technical"], "semantic": r["semantic"]})
        ax1.scatter(piv["technical"], piv.index, marker=marker, s=55,
                    facecolor=fp.MODE_COLOR["technical"], edgecolor="k", linewidth=0.4,
                    label=f"technical ({tag})", zorder=3)
        ax1.scatter(piv["semantic"], piv.index, marker=marker, s=55,
                    facecolor=fp.MODE_COLOR["semantic"], edgecolor="k", linewidth=0.4,
                    label=f"semantic ({tag})", zorder=3)
    ax1.set_xlabel("Recall@10  (language-balanced)")
    ax1.set_title("Semantic > technical, per model (o = GP, s = EPO)")
    ax1.legend(loc="lower right", fontsize=7)
    fig.suptitle("The technical penalty is a distribution shift, not a few hard queries",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fp.save(fig, "claimA_A2_distribution")
    fp.dump_data(pd.DataFrame(rows), "claimA_A2_distribution")


# --------------------------------------------------------------------------- A3 (appendix)
def a3_gap_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4),
                             gridspec_kw={"width_ratios": [5, 3], "wspace": 0.42})
    out = []
    for ax, (df, title, langs) in zip(
        axes, [(_gp(), "Google Patents", fp.CP_LANGS), (_epo(), "EPO", fp.EPO_LANGS)]):
        # gap per (model, language) = sem - tech, language-balanced not needed (per cell)
        g = (df.groupby(["short", "query_language", "mode"])["recall_at_10"].mean()
               .unstack("mode"))
        g["gap"] = g["semantic"] - g["technical"]
        mat = g["gap"].unstack("query_language").reindex(
            index=[fp.short(m) for m in fp.MODEL_ORDER], columns=langs)
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([fp.LANG_NAME[l] for l in langs], rotation=30, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.to_numpy()[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7.5,
                            color="black" if abs(val) < 0.32 else "white")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="R@10 gap (sem − tech)")
        m = mat.reset_index().melt(id_vars="short", var_name="language", value_name="gap")
        m["source"] = title
        out.append(m)
    fig.suptitle("Semantic-minus-technical Recall@10 gap is positive in nearly every cell",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fp.save(fig, "claimA_A3_gap_heatmap")
    fp.dump_data(pd.concat(out, ignore_index=True), "claimA_A3_gap_heatmap")


# --------------------------------------------------------------------------- A4 (appendix)
def a4_metric_robustness():
    metrics = [("rr_at_10", "MRR@10"), ("hit_at_10", "Hit@10")]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharex="col")
    out = []
    for col, (df, src) in enumerate([(_gp(), "Google Patents"), (_epo(), "EPO")]):
        for row, (value, label) in enumerate(metrics):
            ax = axes[row, col]
            agg = _per_model_mode(df, value)
            agg["source"] = src; agg["metric"] = label
            out.append(agg)
            piv = agg.pivot(index="short", columns="mode", values="value").sort_values("semantic")
            ys = np.arange(len(piv))
            for y, (_, r) in zip(ys, piv.iterrows()):
                ax.plot([r["technical"], r["semantic"]], [y, y], color="#bbbbbb", lw=2.2, zorder=1)
            ax.scatter(piv["technical"], ys, s=55, color=fp.MODE_COLOR["technical"], zorder=3,
                       edgecolor="white", linewidth=0.7, label="technical")
            ax.scatter(piv["semantic"], ys, s=55, color=fp.MODE_COLOR["semantic"], zorder=3,
                       edgecolor="white", linewidth=0.7, label="semantic")
            ax.set_yticks(ys); ax.set_yticklabels(piv.index)
            ax.set_title(f"{src} — {label}")
            if row == 1:
                ax.set_xlabel(f"{label}  (language-balanced)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("The technical penalty holds under MRR@10 and Hit@10 too",
                 fontsize=12.5, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fp.save(fig, "claimA_A4_metric_robustness")
    fp.dump_data(pd.concat(out, ignore_index=True), "claimA_A4_metric_robustness")


# --------------------------------------------------------------------------- A7 (MRR @ gold-language split)
def _lang_split_mrr(source: str) -> pd.DataFrame:
    """Per (model, query): MRR@10 computed separately over same-language gold and cross-language gold,
    with mode/query_language. rr@10 = 1/first-gold-rank if that gold is in the top-10 else 0; NaN when
    the query has no gold of that language (so it drops out of that row's domain, like C3's MoLIR/CLIR)."""
    import claimE_metrics as cem
    if source == "GP":
        fr = cem._first_ranks()[["model", "query_id", "first_same_rank", "first_cross_rank",
                                 "n_gold_same", "n_gold_cross", "query_language"]].copy()
        mode = fp.gp_query_table().set_index("query_id")["mode"]
    else:
        fr = fp.epo_first_ranks().copy()
        meta = fp.epo_per_query().drop_duplicates("query_id").set_index("query_id")
        fr["query_language"] = fr["query_id"].map(meta["query_language"])
        mode = meta["mode"]
    fr["mode"] = fr["query_id"].map(mode)
    fr["short"] = fr["model"].map(fp.SHORT)
    with np.errstate(divide="ignore", invalid="ignore"):
        for half, rank_col, n_col in [("rr_same", "first_same_rank", "n_gold_same"),
                                      ("rr_cross", "first_cross_rank", "n_gold_cross")]:
            rr = np.where(fr[rank_col] <= 10, 1.0 / fr[rank_col], 0.0)
            fr[half] = np.where(fr[n_col] > 0, rr, np.nan)
    return fr


def a7_mrr_same_cross():
    rows_def = [("rr_same", "same-language (MoLIR)"), ("rr_cross", "cross-language (CLIR)")]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharex=True)
    out = []
    for col, (source, src) in enumerate([("GP", "Google Patents"), ("EPO", "EPO")]):
        d = _lang_split_mrr(source)
        for row, (val, half_label) in enumerate(rows_def):
            ax = axes[row, col]
            sub = d.dropna(subset=[val])
            agg = _per_model_mode(sub, val)
            agg["source"] = src; agg["gold"] = half_label
            out.append(agg)
            piv = agg.pivot(index="short", columns="mode", values="value").sort_values("semantic")
            ys = np.arange(len(piv))
            for y, (_, r) in zip(ys, piv.iterrows()):
                ax.plot([r["technical"], r["semantic"]], [y, y], color="#bbbbbb", lw=2.2, zorder=1)
            ax.scatter(piv["technical"], ys, s=55, color=fp.MODE_COLOR["technical"], zorder=3,
                       edgecolor="white", linewidth=0.7, label="technical")
            ax.scatter(piv["semantic"], ys, s=55, color=fp.MODE_COLOR["semantic"], zorder=3,
                       edgecolor="white", linewidth=0.7, label="semantic")
            ax.set_yticks(ys); ax.set_yticklabels(piv.index)
            ax.set_title(f"{src} — MRR@10, {half_label} gold")
            if row == 1:
                ax.set_xlabel("MRR@10  (language-balanced)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("The technical penalty holds under MRR@10 for both same-language and cross-language gold",
                 fontsize=12.5, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fp.save(fig, "claimA_A7_mrr_same_cross")
    fp.dump_data(pd.concat(out, ignore_index=True), "claimA_A7_mrr_same_cross")


# --------------------------------------------------------------------------- A5 (appendix)
def a5_mt_robustness():
    """Human-original vs MT-translated questions retrieve about equally — the technical penalty is
    not an MT artifact. Google Patents only (EPO is all human)."""
    df = fp.cp_per_query()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    rows = []
    ys = np.arange(len(fp.MODEL_ORDER))
    for yi, model in enumerate(fp.MODEL_ORDER):
        sub = df[df["model"] == model]
        hum = sub[~sub["is_synthetic"]]["recall_at_10"]
        mt = sub[sub["is_synthetic"]]["recall_at_10"]
        h, m = float(hum.mean()), float(mt.mean())
        ax.plot([h, m], [yi, yi], color="#bbb", lw=2.2, zorder=1)
        ax.scatter(h, yi, s=70, color="#2ca02c", edgecolor="white", linewidth=0.8, zorder=3,
                   label="human-original" if yi == 0 else None)
        ax.scatter(m, yi, s=70, color="#d62728", edgecolor="white", linewidth=0.8, zorder=3,
                   label="MT-translated" if yi == 0 else None)
        rows.append({"short": fp.short(model), "human_recall10": h, "mt_recall10": m,
                     "n_human": int(hum.shape[0]), "n_mt": int(mt.shape[0])})
    ax.set_yticks(ys); ax.set_yticklabels([fp.short(m) for m in fp.MODEL_ORDER])
    ax.set_xlabel("Recall@10  (pooled)")
    ax.set_title("Human-original vs MT-translated questions retrieve comparably (Google Patents)")
    ax.legend(loc="lower right")
    fp.save(fig, "claimA_A5_mt_robustness")
    fp.dump_data(pd.DataFrame(rows), "claimA_A5_mt_robustness")


# --------------------------------------------------------------------------- A6 (new finding)
def a6_question_type():
    """The technical penalty is not uniform: among technical questions, parameter/condition and
    method questions are hardest; outcome and structure questions are easier. Semantic shown as a
    reference band. Both offices. (Enabled by the 524-query rebuild, which carries question_type.)"""
    order = ["parameter_or_condition", "method", "material", "structure", "outcome"]
    nice = {"parameter_or_condition": "parameter /\ncondition", "method": "method",
            "material": "material", "structure": "structure", "outcome": "outcome"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    out = []
    for ax, (df, title) in zip(axes, [(_gp(), "Google Patents"), (_epo(), "EPO")]):
        tech = df[df["mode"] == "technical"]
        sem_mean = df[df["mode"] == "semantic"]["recall_at_10"].mean()
        rows = []
        for qt in order:
            sub = tech[tech["question_type"] == qt]
            if sub.empty:
                continue
            pt, lo, hi = fp.bootstrap_ci(sub["recall_at_10"].dropna().to_numpy())
            rows.append({"question_type": qt, "value": pt, "lo": lo, "hi": hi, "n": len(sub)})
        d = pd.DataFrame(rows)
        xs = np.arange(len(d))
        ax.bar(xs, d["value"], color="#c44e52", width=0.66,
               yerr=[d["value"] - d["lo"], d["hi"] - d["value"]], capsize=3, ecolor="#444")
        ax.axhline(sem_mean, color="#4c72b0", ls="--", lw=1.6)
        ax.text(-0.45, sem_mean + 0.012, f"semantic mean {sem_mean:.2f}",
                ha="left", va="bottom", fontsize=8.5, color="#4c72b0")
        ax.set_xticks(xs); ax.set_xticklabels([nice[q] for q in d["question_type"]], fontsize=8.5)
        ax.set_title(title)
        for x, r in zip(xs, d.itertuples()):
            ax.text(x, r.hi + 0.012, f"{r.value:.2f}\nn={r.n}", ha="center", va="bottom", fontsize=7.5)
        d["source"] = title
        out.append(d)
    axes[0].set_ylabel("Recall@10")
    axes[0].set_ylim(0, 0.86)  # shared; headroom for EPO's small-n 'outcome' bar + CIs
    fig.suptitle("Within technical questions, parameter/condition and method are the hardest",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fp.save(fig, "claimA_A6_question_type")
    fp.dump_data(pd.concat(out, ignore_index=True), "claimA_A6_question_type")


def main():
    fp.set_style()
    a1_dumbbell()
    a2_distribution()
    a3_gap_heatmap()
    a4_metric_robustness()
    a5_mt_robustness()
    a6_question_type()
    a7_mrr_same_cross()
    print("claim A: A1-A7 written to candidates/")


if __name__ == "__main__":
    main()
