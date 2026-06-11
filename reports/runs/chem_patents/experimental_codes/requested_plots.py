"""
USER-REQUESTED chem-patents plots (524-query release, gte-base dropped everywhere).

Six deliverables, all written to experimental_plots/requested_plots/:

  1. avg_recall_by_query_language.png   — average model Recall@10 per query language
                                          (bar = mean over 8 models, err = SEM over models, model dots).
  2. cross_same_by_question_type.png    — Recall@10 in two clusters (technical / semantic); within each,
                                          cross- vs same-language bars; err = SEM over languages.
  3. cross_same_pairs_by_language.png   — query-language x gold-language pair structure
                                          (heatmap with same-language diagonal boxed + per-language
                                          same/cross stacked bars).
  4. xrc_rrc_ari_*.{csv,png}            — XRC, RRC@K, ARI@K recomputed on the full 524-query data.
  5. own_vs_foreign_gold.png            — own-language gold is retrieved far better than foreign gold
                                          (MoLIR@10 vs CLIR@10, paired on queries that have both).

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/requested_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import requested_common as R

MODELS = R.MODELS
SHORTS = [R.short(m) for m in MODELS]
LANG_ORDER = ["en", "de", "fr", "es", "zh"]
DIR_COLOR = {"same": "#4C72B0", "cross": "#DD8452"}
MODE_LABEL = {"technical": "Technical", "semantic": "Semantic"}
MAXRANK = 1000


# =========================================================================== plot 1
def plot_avg_by_query_language(out: Path):
    import matplotlib.pyplot as plt
    cm = R.core()
    # per (model, language) mean recall@10, then average over models with SEM over models
    pml = cm.groupby(["short", "query_language"])["recall_at_10"].mean().reset_index()
    langs = [l for l in LANG_ORDER if l in pml["query_language"].unique()]
    means = [pml[pml.query_language == l]["recall_at_10"].mean() for l in langs]
    sems = [R.sem(pml[pml.query_language == l]["recall_at_10"]) for l in langs]
    order = np.argsort(means)[::-1]
    langs = [langs[i] for i in order]; means = [means[i] for i in order]; sems = [sems[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    xs = np.arange(len(langs))
    bars = ax.bar(xs, means, yerr=sems, capsize=5, width=0.62,
                  color=[C.LANG_COLOR[l] for l in langs], edgecolor="black", linewidth=0.5,
                  error_kw=dict(ecolor="#333", lw=1.2), zorder=2)
    # per-model dots (the spread the SEM summarises)
    rng = np.random.default_rng(0)
    for xi, l in enumerate(langs):
        vals = pml[pml.query_language == l]["recall_at_10"].to_numpy()
        ax.scatter(np.full_like(vals, xi) + rng.uniform(-0.16, 0.16, vals.size), vals,
                   s=22, color="black", alpha=0.45, zorder=3)
    for xi, (m, s) in enumerate(zip(means, sems)):
        ax.text(xi, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{R.LANG_NAME[l]}\n({l})" for l in langs])
    ax.set_ylabel("Recall@10  (mean over 8 models)")
    ax.set_ylim(0, max(m + s for m, s in zip(means, sems)) + 0.07)
    ax.set_title("Average model retrieval performance by query language\n"
                 "(bar = mean over 8 models, error = SEM over models, dots = individual models)")
    fig.tight_layout(); fig.savefig(out / "avg_recall_by_query_language.png"); plt.close(fig)

    pml.pivot(index="short", columns="query_language", values="recall_at_10").round(4)\
        .to_csv(out / "avg_recall_by_query_language.csv")
    return dict(zip(langs, means))


# =========================================================================== plot 2
def plot_cross_same_by_question_type(out: Path):
    import matplotlib.pyplot as plt
    cm = R.core()
    modes = ["technical", "semantic"]
    dirs = ["cross", "same"]
    # cell value: per language mean recall@10 (over models x queries), bar=mean over langs, err=SEM over langs
    table = {}
    lang_means = {}
    for mode in modes:
        for d in dirs:
            sub = cm[(cm["mode"] == mode) & (cm["direction"] == d)]
            per_lang = sub.groupby("query_language")["recall_at_10"].mean()
            lang_means[(mode, d)] = per_lang
            table[(mode, d)] = (per_lang.mean(), R.sem(per_lang), len(per_lang))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cluster_w = 0.34
    centers = np.array([0.0, 1.1])
    for di, d in enumerate(dirs):
        offs = (di - 0.5) * (cluster_w + 0.02)
        xs = centers + offs
        means = [table[(m, d)][0] for m in modes]
        errs = [table[(m, d)][1] for m in modes]
        ax.bar(xs, means, yerr=errs, width=cluster_w, capsize=5,
               color=DIR_COLOR[d], edgecolor="black", linewidth=0.5,
               error_kw=dict(ecolor="#333", lw=1.2),
               label=("cross-language (no gold in query language)" if d == "cross"
                      else "same-language (gold exists in query language)"), zorder=2)
        # language dots
        rng = np.random.default_rng(di + 1)
        for xi, m in zip(xs, modes):
            vals = lang_means[(m, d)].to_numpy()
            ax.scatter(np.full_like(vals, xi) + rng.uniform(-0.07, 0.07, vals.size), vals,
                       s=20, color="black", alpha=0.4, zorder=3)
        for xi, (mn, er) in zip(xs, zip(means, errs)):
            ax.text(xi, mn + er + 0.012, f"{mn:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([MODE_LABEL[m] for m in modes], fontsize=11)
    ax.set_ylabel("Recall@10  (mean over 8 models)")
    ax.set_ylim(0, 0.95)
    ax.set_title("Cross- vs same-language retrieval by question type\n"
                 "(bar = mean over query languages, error = SEM over languages, dots = languages)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "cross_same_by_question_type.png"); plt.close(fig)

    rows = []
    for mode in modes:
        for d in dirs:
            mn, er, nl = table[(mode, d)]
            rows.append({"mode": mode, "direction": d, "recall_at_10_mean": round(mn, 4),
                         "sem_over_languages": round(er, 4), "n_languages": nl})
    pd.DataFrame(rows).to_csv(out / "cross_same_by_question_type.csv", index=False)
    return table


# =========================================================================== plot 3
def plot_cross_same_pairs(out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    pq = R.per_query()
    # build query-lang x gold-lang pair-count matrix
    rows = []
    for _, p in pq.iterrows():
        for gl in p["gold_langs"]:
            rows.append((p["query_language"], gl))
    pairs = pd.DataFrame(rows, columns=["qlang", "glang"])
    mat = pd.crosstab(pairs["qlang"], pairs["glang"]).reindex(index=LANG_ORDER, columns=LANG_ORDER).fillna(0).astype(int)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.3),
                                   gridspec_kw=dict(width_ratios=[1.05, 1.0]))

    # ---- panel A: heatmap, diagonal (same-language) boxed ----
    M = mat.to_numpy()
    im = axA.imshow(M, cmap="Blues", aspect="auto")
    for i in range(len(LANG_ORDER)):
        for j in range(len(LANG_ORDER)):
            v = M[i, j]
            same = (i == j)
            axA.text(j, i, str(v), ha="center", va="center", fontsize=10,
                     color="white" if v > M.max() * 0.55 else "#222",
                     fontweight="bold" if same else "normal")
        axA.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#c0392b", lw=2.4))
    axA.set_xticks(range(len(LANG_ORDER))); axA.set_xticklabels(LANG_ORDER)
    axA.set_yticks(range(len(LANG_ORDER))); axA.set_yticklabels(LANG_ORDER)
    axA.set_xlabel("gold-document language")
    axA.set_ylabel("query language")
    axA.set_title("Query x gold-document language pairs (count)\n"
                  "red diagonal = same-language pairs; off-diagonal = cross-language")
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04, label="# (query, gold) pairs")

    # ---- panel B: per query language, same vs cross pair counts (stacked) ----
    same_cnt = np.array([mat.loc[l, l] for l in LANG_ORDER], dtype=float)
    total_cnt = mat.sum(axis=1).reindex(LANG_ORDER).to_numpy(dtype=float)
    cross_cnt = total_cnt - same_cnt
    xs = np.arange(len(LANG_ORDER))
    axB.bar(xs, same_cnt, width=0.62, color=DIR_COLOR["same"], edgecolor="black", linewidth=0.5,
            label="same-language gold pairs")
    axB.bar(xs, cross_cnt, bottom=same_cnt, width=0.62, color=DIR_COLOR["cross"], edgecolor="black",
            linewidth=0.5, label="cross-language gold pairs")
    for x, (sc, cc, tot) in enumerate(zip(same_cnt, cross_cnt, total_cnt)):
        if sc > 0:
            axB.text(x, sc / 2, f"{int(sc)}\n({sc/tot:.0%})", ha="center", va="center", fontsize=8, color="white")
        axB.text(x, sc + cc / 2, f"{int(cc)}\n({cc/tot:.0%})", ha="center", va="center", fontsize=8, color="white")
        axB.text(x, tot + 4, f"n={int(tot)}", ha="center", va="bottom", fontsize=8, color="#333")
    axB.set_xticks(xs); axB.set_xticklabels([f"{R.LANG_NAME[l]}\n({l})" for l in LANG_ORDER])
    axB.set_ylabel("# (query, gold) pairs")
    axB.set_ylim(0, total_cnt.max() * 1.18)
    axB.set_title("Same- vs cross-language gold pairs per query language\n"
                  "(how much of each language's retrieval target is in another language)")
    axB.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "cross_same_pairs_by_language.png"); plt.close(fig)

    mat.to_csv(out / "cross_same_pairs_matrix.csv")
    return mat


# =========================================================================== plot 4: XRC / RRC / ARI
def _mean_depth(depths: np.ndarray):
    """Mean first-relevant reading depth. Censored (never-found) ranks are inf; we cap them at
    MAXRANK=1000 (the retrieved-list depth) so the mean is well-defined. With any censoring this is
    a LOWER BOUND on the true mean depth (the twin is at rank > 1000, counted as exactly 1000)."""
    d = np.asarray(depths, dtype=float)
    if d.size == 0:
        return float("nan"), False
    censored = bool(np.any(~np.isfinite(d)))
    return float(np.mean(np.minimum(d, MAXRANK))), censored


def compute_xrc_rrc_ari(out: Path):
    import matplotlib.pyplot as plt
    cm = R.core()
    KS_CURVE = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    K_PRIMARY = 100

    xrc_rows, rrc_rows, ari_rows, lang_rows = [], [], [], []
    curves = {}
    for m in MODELS:
        s = R.short(m)
        sub = cm[cm.model == m]
        cross = sub[sub.n_gold_cross > 0]["first_cross_rank"].to_numpy(float)
        same = sub[sub.n_gold_same > 0]["first_same_rank"].to_numpy(float)
        clir10 = float(sub["cross_recall_at_10"].mean())
        molir10 = float(sub["same_recall_at_10"].mean())

        # ---- XRC: MEAN reading-depth multiplier (censored ranks capped at MAXRANK) ----
        dmean_cross, cens_c = _mean_depth(cross)
        dmean_same, cens_s = _mean_depth(same)
        xrc_mean = (dmean_cross / dmean_same) if (dmean_same and dmean_same > 0) else float("nan")
        xrc_rows.append({"model": m, "short": s,
                         "n_cross": cross.size, "n_same": same.size,
                         "cross_censored_frac": round(float(np.mean(~np.isfinite(cross))), 4),
                         "same_censored_frac": round(float(np.mean(~np.isfinite(same))), 4),
                         "Dmean_same": round(dmean_same, 2),
                         "Dmean_cross": round(dmean_cross, 2),
                         "XRC_mean": round(xrc_mean, 3) if np.isfinite(xrc_mean) else xrc_mean,
                         "XRC_mean_censored": bool(cens_c or cens_s),
                         "clir_at_10": round(clir10, 4)})

        # ---- RRC(K) curve + L_inf + ARI ----
        rrc = np.array([float(np.mean(cross <= k)) for k in KS_CURVE])
        curves[s] = rrc
        rrc10 = float(np.mean(cross <= 10))
        rrc100 = float(np.mean(cross <= K_PRIMARY))
        rrc1000 = float(np.mean(cross <= 1000))
        l_inf = 1.0 - rrc1000
        rrc_rows.append({"model": m, "short": s, "n_cross": cross.size,
                         "RRC_at_10": round(rrc10, 4),
                         "RRC_at_100": round(rrc100, 4), "RRC_at_1000": round(rrc1000, 4),
                         "found_11_100": round(rrc100 - rrc10, 4),
                         "lost_at_100": round(1.0 - rrc100, 4),
                         "found_101_1000": round(rrc1000 - rrc100, 4),
                         "lost_at_1000": round(l_inf, 4)})
        resid = 1.0 - rrc100
        ari100 = (l_inf / resid) if resid > 0 else float("nan")
        ari_rows.append({"model": m, "short": s,
                         "RRC_at_100_cheap": round(rrc100, 4),
                         "deep_100_to_1000": round(rrc1000 - rrc100, 4),
                         "L_inf_floor": round(l_inf, 4),
                         "identity_sum": round(rrc100 + (rrc1000 - rrc100) + l_inf, 6),
                         "ARI_at_100": round(ari100, 4) if np.isfinite(ari100) else float("nan"),
                         "clir_at_10": round(clir10, 4)})

        # ---- per-language XRC (mean reading-depth multiplier) ----
        for lang in LANG_ORDER:
            cl = sub[(sub.n_gold_cross > 0) & (sub.query_language == lang)]["first_cross_rank"].to_numpy(float)
            sl = sub[(sub.n_gold_same > 0) & (sub.query_language == lang)]["first_same_rank"].to_numpy(float)
            if cl.size == 0:
                continue
            dmc, _ = _mean_depth(cl)
            dms, _ = _mean_depth(sl) if sl.size else (float("nan"), False)
            xrc_l = (dmc / dms) if (np.isfinite(dms) and dms) else float("nan")
            lang_rows.append({"model": m, "short": s, "query_language": lang,
                              "n_cross": int(cl.size), "n_same": int(sl.size),
                              "Dmean_cross": round(dmc, 2),
                              "Dmean_same": round(dms, 2) if np.isfinite(dms) else dms,
                              "XRC_mean": round(xrc_l, 3) if np.isfinite(xrc_l) else xrc_l})

    xrc_df = pd.DataFrame(xrc_rows)
    rrc_df = pd.DataFrame(rrc_rows)
    ari_df = pd.DataFrame(ari_rows)
    lang_df = pd.DataFrame(lang_rows)
    xrc_df.to_csv(out / "xrc_per_model.csv", index=False)
    rrc_df.to_csv(out / "rrc_per_model.csv", index=False)
    ari_df.to_csv(out / "ari_per_model.csv", index=False)
    lang_df.to_csv(out / "xrc_per_language.csv", index=False)

    # ---- figure: 3 panels (RRC ceiling | ARI decomposition | XRC vs CLIR) ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 6.0))

    # panel 1: RRC recoverability ceiling (stacked) — realistic re-ranker budgets K=10 / K=100
    o = rrc_df.sort_values("RRC_at_100", ascending=False).reset_index(drop=True)
    xs = np.arange(len(o))
    ax1.bar(xs, o.RRC_at_10, color="#228833", label="recoverable by re-rank @top-10")
    ax1.bar(xs, o.found_11_100, bottom=o.RRC_at_10, color="#aacc99", label="found in 11..100")
    ax1.bar(xs, o.lost_at_100, bottom=o.RRC_at_10 + o.found_11_100, color="#cc6677",
            label="never in top-100 (lost)")
    for x, l in zip(xs, o.lost_at_100):
        ax1.text(x, 1.005, f"{l:.0%}", ha="center", va="bottom", fontsize=7, color="#cc6677")
    ax1.set_xticks(xs); ax1.set_xticklabels(o["short"], rotation=40, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.1); ax1.set_ylabel("fraction of cross-gold queries")
    ax1.set_title("RRC — re-ranker recoverability ceiling (top-10 / top-100)")
    ax1.legend(fontsize=7, loc="lower right")

    # panel 2: ARI decomposition (stacked) over models in capability order
    oa = ari_df.sort_values("RRC_at_100_cheap", ascending=False).reset_index(drop=True)
    xs = np.arange(len(oa))
    ax2.bar(xs, oa.RRC_at_100_cheap, color="#228833", label="cheap (RRC@100)")
    ax2.bar(xs, oa.deep_100_to_1000, bottom=oa.RRC_at_100_cheap, color="#aacc99",
            label="deep (RRC@1000-RRC@100)")
    ax2.bar(xs, oa.L_inf_floor, bottom=oa.RRC_at_100_cheap + oa.deep_100_to_1000, color="#cc6677",
            label="alignment-only floor (L_inf)")
    for x, (fl, ari) in enumerate(zip(oa.L_inf_floor, oa.ARI_at_100)):
        ax2.text(x, 1.012, f"ARI {ari:.2f}", ha="center", va="bottom", fontsize=7,
                 color="#7a2b34", fontweight="bold")
    ax2.set_xticks(xs); ax2.set_xticklabels(oa["short"], rotation=40, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.12); ax2.set_ylabel("share of cross-gold queries")
    ax2.set_title("ARI — alignment-recoverability decomposition (K=100)")
    ax2.legend(fontsize=7, loc="lower left")

    # panel 3: XRC (mean reading-depth multiplier) vs CLIR@10 scatter
    name2model = {R.short(m): m for m in MODELS}
    for r in xrc_df.itertuples():
        y = r.XRC_mean if isinstance(r.XRC_mean, (int, float)) and np.isfinite(r.XRC_mean) else None
        if y is None:
            continue
        ax3.scatter(r.clir_at_10, y, s=95, color=R.MODEL_COLOR[name2model[r.short]],
                    edgecolor="black", linewidth=0.5, zorder=3,
                    marker="v" if r.XRC_mean_censored else "o")
        ax3.annotate(r.short, (r.clir_at_10, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax3.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax3.set_yscale("log")  # e5's huge multiplier compresses a linear axis
    ax3.set_xlabel("CLIR@10 (cross-lingual recall@10)")
    ax3.set_ylabel("XRC = mean D(cross) / mean D(same)\n(mean reading-cost multiplier, log scale)")
    ax3.set_title("XRC — cross-lingual reading-cost vs CLIR@10\n(dashed = parity; lower = cheaper to read across languages)")

    fig.suptitle("XRC / RRC / ARI on the full 524-query chem-patents benchmark (8 models)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "xrc_rrc_ari_overview.png"); plt.close(fig)

    # ---- figure: RRC(K) curves ----
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    Ks = np.array(KS_CURVE, float)
    for m in MODELS:
        s = R.short(m)
        ax.plot(Ks, curves[s], color=R.MODEL_COLOR[m], lw=1.6, marker="o", ms=3, label=s)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("re-ranker pool depth K (log scale)")
    ax.set_ylabel("RRC(K) = fraction of cross-gold twins inside top-K")
    ax.set_title("Re-ranker budget frontier: RRC(K) per model (524-query data)")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(out / "rrc_budget_frontier.png"); plt.close(fig)

    return xrc_df, rrc_df, ari_df


# =========================================================================== plot 5: own vs foreign gold
def plot_own_vs_foreign(out: Path):
    import matplotlib.pyplot as plt
    cm = R.core()
    # paired domain: queries that carry BOTH a same-language gold and a cross-language gold
    both = cm[(cm.n_gold_same > 0) & (cm.n_gold_cross > 0)].copy()
    rows = []
    for m in MODELS:
        s = R.short(m)
        sub = both[both.model == m]
        own = sub["same_recall_at_10"].to_numpy(float)
        foreign = sub["cross_recall_at_10"].to_numpy(float)
        st = C.wilcoxon(own, foreign)
        rows.append({"model": m, "short": s, "n_paired": len(sub),
                     "own_lang_recall_at_10": round(float(np.nanmean(own)), 4),
                     "foreign_lang_recall_at_10": round(float(np.nanmean(foreign)), 4),
                     "gap": round(float(np.nanmean(own) - np.nanmean(foreign)), 4),
                     "wilcoxon_p": st["p"], "stars": C.stars(st["p"])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "own_vs_foreign_gold.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    xs = np.arange(len(df))
    w = 0.38
    ax.bar(xs - w / 2, df.own_lang_recall_at_10, width=w, color="#2a9d8f", edgecolor="black",
           linewidth=0.5, label="own-language gold (same language as query)")
    ax.bar(xs + w / 2, df.foreign_lang_recall_at_10, width=w, color="#e76f51", edgecolor="black",
           linewidth=0.5, label="foreign-language gold (lives in another language)")
    for x, r in zip(xs, df.itertuples()):
        ax.text(x - w / 2, r.own_lang_recall_at_10 + 0.012, f"{r.own_lang_recall_at_10:.2f}",
                ha="center", va="bottom", fontsize=7.5)
        ax.text(x + w / 2, r.foreign_lang_recall_at_10 + 0.012, f"{r.foreign_lang_recall_at_10:.2f}",
                ha="center", va="bottom", fontsize=7.5)
        top = max(r.own_lang_recall_at_10, r.foreign_lang_recall_at_10)
        ax.text(x, top + 0.06, r.stars, ha="center", va="bottom", fontsize=9, color="#444")
    ax.set_xticks(xs); ax.set_xticklabels(df["short"], rotation=30, ha="right")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.05)
    ax.set_title("Queries retrieve their own-language gold far better than foreign-language gold\n"
                 f"(paired on the {int(df.n_paired.iloc[0])} queries that have both; "
                 "Wilcoxon: *** p<0.001)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "own_vs_foreign_gold.png"); plt.close(fig)
    return df


# =========================================================================== main
def main():
    C.set_style()
    out = R.out_dir()
    print(f"[requested_plots] models (gte-base dropped): {SHORTS}")
    print(f"[requested_plots] writing -> {out}\n")

    lang_means = plot_avg_by_query_language(out)
    print("1. avg recall@10 by query language:",
          {k: round(v, 3) for k, v in lang_means.items()})

    t2 = plot_cross_same_by_question_type(out)
    print("2. cross/same x question type (recall@10 mean | SEM over langs):")
    for k in [("technical", "cross"), ("technical", "same"), ("semantic", "cross"), ("semantic", "same")]:
        mn, er, nl = t2[k]
        print(f"     {k[0]:9s} {k[1]:5s}: {mn:.3f} +/- {er:.3f}  (n_langs={nl})")

    mat = plot_cross_same_pairs(out)
    same_pairs = sum(mat.loc[l, l] for l in LANG_ORDER)
    total_pairs = int(mat.to_numpy().sum())
    print(f"3. pair matrix: {total_pairs} pairs total | same-language {same_pairs} "
          f"({same_pairs/total_pairs:.0%}) | cross-language {total_pairs-same_pairs} "
          f"({(total_pairs-same_pairs)/total_pairs:.0%})")

    xrc_df, rrc_df, ari_df = compute_xrc_rrc_ari(out)
    print("4. XRC / RRC / ARI (per model):")
    show = rrc_df.merge(ari_df[["short", "ARI_at_100"]], on="short").merge(
        xrc_df[["short", "Dmean_same", "Dmean_cross", "XRC_mean"]], on="short")
    print(show[["short", "RRC_at_10", "RRC_at_100", "lost_at_100", "RRC_at_1000", "lost_at_1000",
                "ARI_at_100", "Dmean_same", "Dmean_cross", "XRC_mean"]].to_string(index=False))

    ovf = plot_own_vs_foreign(out)
    print("5. own- vs foreign-language gold recall@10 (paired):")
    print(ovf[["short", "own_lang_recall_at_10", "foreign_lang_recall_at_10", "gap", "stars"]]
          .to_string(index=False))
    print(f"\n[requested_plots] done -> {out}")


if __name__ == "__main__":
    main()
