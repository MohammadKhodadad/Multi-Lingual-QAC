"""
Round 4 (headline) — Mate retrieval: can the model find the patent's translated twin, and where?

A bitext lens on the benchmark. Each query's "foreign mates" are the same patent's documents in other
languages. We separate two questions the averaged metrics blur together:
  * CAN it find one?   mate-hit@k  — is any foreign mate in the top-k.
  * WHERE is it?       first-foreign-rank and mate-MRR = 1 / first-foreign-rank.

Outputs (../experimental_plots/round04_mate_retrieval/):
  per_query_mate.csv, summary.json, and figures:
    mate_overview.png             mate-hit@10, mate-hit@100, mate-MRR per model
    first_foreign_rank_cdf.png    survival curve: P(first foreign mate at rank ≤ x)
    mate_hit_by_query_language.png mate-hit@10 by query language (+ median first-foreign-rank)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round04_mate_retrieval"
DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    cpq = C.core_per_query().copy()
    cpq = cpq[cpq.n_gold_cross > 0]  # every query has cross-language mates, but be safe
    for k in (1, 5, 10, 20, 50, 100):
        cpq[f"mate_hit_{k}"] = (cpq["first_cross_rank"] <= k).astype(float)
    cpq.to_csv(out / "per_query_mate.csv", index=False)

    # ---- per-model summary -----------------------------------------------------------------------
    rows = []
    for model in C.MODEL_ORDER:
        g = cpq[cpq.model == model]
        h10, h10lo, h10hi = C.bootstrap_ci(g["mate_hit_10"].values)
        h100 = g["mate_hit_100"].mean()
        mrr, mrrlo, mrrhi = C.bootstrap_ci(g["mate_mrr"].fillna(0).values)
        lost = float((g["first_cross_rank"] == C.INF).mean())  # no foreign mate even in top-1000
        med = float(np.median(g.loc[g["first_cross_rank"] < C.INF, "first_cross_rank"])) \
            if (g["first_cross_rank"] < C.INF).any() else float("nan")
        rows.append({"model": model, "short": C.short(model),
                     "mate_hit10": h10, "h10_lo": h10lo, "h10_hi": h10hi,
                     "mate_hit100": h100, "mate_mrr": mrr, "mrr_lo": mrrlo, "mrr_hi": mrrhi,
                     "lost_share": lost, "median_first_foreign_rank": med})
    msum = pd.DataFrame(rows)
    msum.to_csv(out / "per_model.csv", index=False)

    # ---- Fig 1: overview -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = np.arange(len(msum))
    w = 0.27
    err10 = np.vstack([msum["mate_hit10"] - msum["h10_lo"], msum["h10_hi"] - msum["mate_hit10"]])
    ax.bar(xs - w, msum["mate_hit10"], w, yerr=err10, capsize=2.5, label="mate-hit@10 (any foreign twin in top-10)", color="#1f77b4")
    ax.bar(xs, msum["mate_hit100"], w, label="mate-hit@100", color="#aec7e8")
    errm = np.vstack([msum["mate_mrr"] - msum["mrr_lo"], msum["mrr_hi"] - msum["mate_mrr"]])
    ax.bar(xs + w, msum["mate_mrr"], w, yerr=errm, capsize=2.5, label="mate-MRR (1 / first-foreign-rank)", color="#ff7f0e")
    ax.set_xticks(xs)
    ax.set_xticklabels(msum["short"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title("Finding the translated twin: can the model surface any foreign version, and how high?")
    fig.tight_layout()
    fig.savefig(out / "mate_overview.png")
    plt.close(fig)

    # ---- Fig 2: first-foreign-rank survival ------------------------------------------------------
    grid = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for model in C.MODEL_ORDER:
        g = cpq[cpq.model == model]
        fr = g["first_cross_rank"].values
        surv = [(fr <= x).mean() for x in grid]
        ax.plot(grid, surv, color=C.MODEL_COLOR[model], label=C.short(model))
    ax.set_xscale("log")
    ax.set_xlabel("rank cutoff x")
    ax.set_ylabel("P(first foreign mate at rank ≤ x)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, fontsize=7)
    ax.set_title("How deep must you scan to surface the first foreign twin?")
    fig.tight_layout()
    fig.savefig(out / "first_foreign_rank_cdf.png")
    plt.close(fig)

    # ---- Fig 3: mate-hit@10 by query language ----------------------------------------------------
    pool = cpq[cpq.model.isin(POOL)]
    by = []
    for lang in C.QUERY_LANGS:
        sub = pool[pool.query_language == lang]
        pt, lo, hi = C.bootstrap_ci(sub["mate_hit_10"].values)
        med = float(np.median(sub.loc[sub["first_cross_rank"] < C.INF, "first_cross_rank"])) \
            if (sub["first_cross_rank"] < C.INF).any() else float("nan")
        by.append({"lang": lang, "hit10": pt, "lo": lo, "hi": hi, "median_rank": med,
                   "n": int(sub.query_id.nunique())})
    bydf = pd.DataFrame(by)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = np.arange(len(bydf))
    err = np.vstack([bydf["hit10"] - bydf["lo"], bydf["hi"] - bydf["hit10"]])
    ax.bar(xs, bydf["hit10"], yerr=err, capsize=3, color=[C.LANG_COLOR[l] for l in bydf["lang"]])
    for x, v, m in zip(xs, bydf["hit10"], bydf["median_rank"]):
        lbl = f"{v:.2f}\nmed.rk {int(m)}" if m == m else f"{v:.2f}"
        ax.text(x, v + 0.012, lbl, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.LANG_NAME[l]}\n({l})" for l in bydf["lang"]])
    ax.set_ylabel("mate-hit@10 (pooled reliable models)")
    ax.set_ylim(0, max(0.7, bydf["hi"].max() + 0.08))
    ax.set_title("From which query language can you reach a foreign twin in the top-10?")
    fig.tight_layout()
    fig.savefig(out / "mate_hit_by_query_language.png")
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------------------
    best = msum.sort_values("mate_mrr", ascending=False).iloc[0]
    pooled_lost = float(pool["first_cross_rank"].eq(C.INF).mean())
    pooled_hit10 = float(pool["mate_hit_10"].mean())
    summary = {
        "best_mate_mrr_model": best["short"], "best_mate_mrr": round(float(best["mate_mrr"]), 4),
        "best_mate_hit10": round(float(best["mate_hit10"]), 4),
        "best_median_first_foreign_rank": (None if best["median_first_foreign_rank"] != best["median_first_foreign_rank"]
                                           else int(best["median_first_foreign_rank"])),
        "pooled_mate_hit10": round(pooled_hit10, 4),
        "pooled_lost_share_top1000": round(pooled_lost, 4),
        "weakest_query_language": bydf.sort_values("hit10").iloc[0]["lang"],
        "per_model": msum[["short", "mate_hit10", "mate_hit100", "mate_mrr", "lost_share",
                           "median_first_foreign_rank"]].round(4).to_dict(orient="records"),
    }
    C.jdump(summary, out / "summary.json")

    md = f"""<!-- round:04 -->
## Round 4 — Mate retrieval: finding the translated twin

**Question.** Beyond "did it find a relevant doc", can the model surface the patent's *foreign twin*
at all, and how deep do you have to scan? Foreign mates = the same patent in other languages;
mate-hit@k = any foreign twin in the top-k; mate-MRR = 1 / rank of the first foreign twin.

**What we learned.**
- The best twin-finder is **{best['short']}** (mate-MRR = {best['mate_mrr']:.3f}, mate-hit@10
  = {best['mate_hit10']:.3f}); when it does surface a twin, the median first-foreign rank is
  **{summary['best_median_first_foreign_rank']}**.
- Pooled over reliable models, only **{pooled_hit10:.0%}** of queries surface a foreign twin in the
  top-10, and **{pooled_lost:.0%}** of (query, model) pairs never surface one even in the **top-1000**
  — those patents are effectively unreachable across the language barrier.
- The widening gap between mate-hit@10 and mate-hit@100 (Fig 1) shows the twin is often *present but
  buried*: a re-ranking or deeper cutoff recovers a meaningful share.
- Weakest entry language for twin-finding: **{summary['weakest_query_language']}**.

**Next questions.**
1. If the same question in different languages returns different twins, the rankings themselves must
   diverge. How consistent is the ranked list across languages (RBO)? → Round 5.
2. When the twin is buried, what outranks it — same-language noise? → Rounds 6 & 7.
<!-- /round:04 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
