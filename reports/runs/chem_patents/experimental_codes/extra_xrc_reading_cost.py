"""
EXTRA (round-1 additive analysis) — XRC + RRC.

Two deployment-facing, distribution-free numbers computed from the on-disk top-1000 rankings,
with NO new embedding run and NO API call:

  * XRC — Cross-Lingual Reading-Cost Multiplier.
      "To find the foreign twin at C% coverage you must scan XRC x more documents than the
       same-language copy."  XRC(m, C) = D_C(cross) / D_C(same), where D_C is the C-th-percentile
       first-relevant rank (reading depth). Reported at C=90 and C=95.
      - cross numerator domain: all 137 queries that have a cross-language gold (pure CLIR).
      - same  denominator domain: only the 57 originals carry a same-language gold (mirrors the
        Fig-1 MoLIR caveat) — stated explicitly as the denominator's population.
      Censoring: a rank is INF when the first foreign gold never appears in the top-1000. If the
      C-th percentile falls in that unfound tail we report it as right-censored (">1000") and the
      resulting XRC as a lower bound ("xrc >= ...").

  * RRC — Re-Ranker Recoverability Ceiling.
      RRC(m, K) = fraction of cross-gold queries whose foreign twin is already inside top-K.
      1 - RRC is provably UNRECOVERABLE by any re-ranker that only re-orders a top-K pool.
      Reported at K=100 (realistic re-ranker pool) and K=1000 (the full retrieved list).

Writes to a NEW dir, never touching the existing 10 rounds:
    reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/
Outputs: xrc_per_model.csv, xrc_per_language.csv, rrc_per_model.csv, summary.json,
         xrc_vs_clir.png, rrc_ceiling.png

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/extra_xrc_reading_cost.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_xrc_reading_cost"
MAXRANK = 1000  # depth of the retrieved list; INF ranks are censored at > MAXRANK
SEED = 20260609


def pct_depth(depths: np.ndarray, q: float):
    """C-th percentile reading depth over a vector that may contain np.inf (censored).

    Returns (value, censored_flag). We sort with inf at the top; if the percentile index lands on
    an inf entry the percentile is right-censored and we return (np.inf, True). Otherwise the finite
    percentile value and (value, False). np.percentile cannot be used directly because it propagates
    inf into linear interpolation; sorting+indexing on the *full* distribution is the honest version.
    """
    d = np.sort(np.asarray(depths, dtype=float))  # inf sorts last
    n = d.size
    if n == 0:
        return float("nan"), False
    # nearest-rank percentile index (lower), 0-based
    idx = int(np.ceil(q / 100.0 * n)) - 1
    idx = min(max(idx, 0), n - 1)
    val = d[idx]
    return (float("inf"), True) if not np.isfinite(val) else (float(val), False)


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    cpq = C.core_per_query()
    gold = C.gold_publication()
    ql = C.q_lang()
    lists = C.ranked_lists()
    # gte-base is the degenerate encoder excluded from round04's pooled figures; keep it in the
    # per-model table (clearly flagged by its 0.91 censored frac) but exclude it from POOLED checks.
    DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
    POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]

    # ---- same-language reading depths (57 originals only) per model ----
    same_depth = {m: [] for m in C.MODEL_ORDER}
    same_lang_rows = {m: {} for m in C.MODEL_ORDER}  # per query-language same depths (sparse)
    for m in C.MODEL_ORDER:
        for qid, g in gold.items():
            lang = ql.get(qid, "")
            same, _ = C.split_gold(g, lang)
            if same:
                r = C.first_gold_rank(lists.get((m, qid), []), same)
                same_depth[m].append(r)
                same_lang_rows[m].setdefault(lang, []).append(r)

    # ---- per-model XRC + RRC ----
    rows = []
    rrc_rows = []
    per_lang = []
    for m in C.MODEL_ORDER:
        sub = cpq[(cpq.model == m) & (cpq.n_gold_cross > 0)]
        cross = sub["first_cross_rank"].to_numpy(dtype=float)  # may contain inf
        same = np.asarray(same_depth[m], dtype=float)
        n_cross = cross.size
        n_same = same.size
        cross_cens = float(np.mean(~np.isfinite(cross)))
        same_cens = float(np.mean(~np.isfinite(same)))

        d50_cross, c50c = pct_depth(cross, 50)
        d75_cross, c75c = pct_depth(cross, 75)
        d90_cross, c90c = pct_depth(cross, 90)
        d95_cross, c95c = pct_depth(cross, 95)
        d50_same, c50s = pct_depth(same, 50)
        d75_same, c75s = pct_depth(same, 75)
        d90_same, c90s = pct_depth(same, 90)
        d95_same, c95s = pct_depth(same, 95)

        def ratio(num, den, num_c, den_c):
            if not np.isfinite(den) or den == 0:
                return float("nan"), True
            if not np.isfinite(num):
                # numerator censored -> XRC is a lower bound using MAXRANK as the floor on num
                return MAXRANK / den, True
            return num / den, bool(num_c or den_c)

        # XRC50 (median) is the robust, fully-finite headline; D90/D95 are censored tail bounds.
        xrc50, xrc50_cens = ratio(d50_cross, d50_same, c50c, c50s)
        xrc75, xrc75_cens = ratio(d75_cross, d75_same, c75c, c75s)
        xrc90, xrc90_cens = ratio(d90_cross, d90_same, c90c, c90s)
        xrc95, xrc95_cens = ratio(d95_cross, d95_same, c95c, c95s)

        clir10 = float(cpq[cpq.model == m]["clir_at_10"].mean())

        rows.append({
            "model": m, "short": C.short(m),
            "n_cross": n_cross, "n_same": n_same,
            "cross_censored_frac": round(cross_cens, 4), "same_censored_frac": round(same_cens, 4),
            "D50_same": d50_same, "D75_same": d75_same, "D90_same": d90_same, "D95_same": d95_same,
            "D50_cross": d50_cross, "D75_cross": d75_cross, "D90_cross": d90_cross, "D95_cross": d95_cross,
            "XRC50": round(xrc50, 3) if np.isfinite(xrc50) else xrc50, "XRC50_censored": xrc50_cens,
            "XRC75": round(xrc75, 3) if np.isfinite(xrc75) else xrc75, "XRC75_censored": xrc75_cens,
            "XRC90": round(xrc90, 3) if np.isfinite(xrc90) else xrc90, "XRC90_censored": xrc90_cens,
            "XRC95": round(xrc95, 3) if np.isfinite(xrc95) else xrc95, "XRC95_censored": xrc95_cens,
            "clir_at_10": round(clir10, 4),
        })

        # RRC
        rrc100 = float(np.mean(cross <= 100))
        rrc1000 = float(np.mean(cross <= 1000))
        rrc_rows.append({
            "model": m, "short": C.short(m),
            "n_cross": n_cross,
            "RRC_at_100": round(rrc100, 4), "RRC_at_1000": round(rrc1000, 4),
            "lost_at_1000": round(1.0 - rrc1000, 4),
            "recoverable_by_rerank_100": round(rrc100, 4),
            "found_101_1000": round(rrc1000 - rrc100, 4),
        })

        # per-language XRC (group cross depths by query language)
        for lang in C.QUERY_LANGS:
            cl = sub[sub.query_language == lang]["first_cross_rank"].to_numpy(dtype=float)
            if cl.size == 0:
                continue
            sl = np.asarray(same_lang_rows[m].get(lang, []), dtype=float)
            d95c, _ = pct_depth(cl, 95)
            d90c, _ = pct_depth(cl, 90)
            d95s, _ = pct_depth(sl, 95) if sl.size else (float("nan"), False)
            xrc_l = (d95c / d95s) if (np.isfinite(d95c) and np.isfinite(d95s) and d95s) else float("nan")
            per_lang.append({
                "model": m, "short": C.short(m), "query_language": lang,
                "n_cross": int(cl.size), "n_same": int(sl.size),
                "D95_cross": d95c, "D90_cross": d90c, "D95_same": d95s,
                "XRC95": round(xrc_l, 3) if np.isfinite(xrc_l) else xrc_l,
                "cross_censored_frac": round(float(np.mean(~np.isfinite(cl))), 4),
            })

    xrc_df = pd.DataFrame(rows)
    rrc_df = pd.DataFrame(rrc_rows)
    lang_df = pd.DataFrame(per_lang)
    xrc_df.to_csv(out / "xrc_per_model.csv", index=False)
    rrc_df.to_csv(out / "rrc_per_model.csv", index=False)
    lang_df.to_csv(out / "xrc_per_language.csv", index=False)

    # ---- figure 1: XRC vs CLIR@10 (W1 headline scatter). Use the robust, fully-finite XRC50
    # (median reading depth) as the y-axis; the 90/95 tail multipliers are mostly right-censored at
    # this sample size (see xrc_per_model.csv) and would be a fragile headline. ----
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for r in xrc_df.itertuples():
        y = r.XRC50 if isinstance(r.XRC50, (int, float)) and np.isfinite(r.XRC50) else None
        if y is None:
            continue
        col = C.MODEL_COLOR[r.model]
        mk = "v" if r.XRC50_censored else "o"
        ax.scatter(r.clir_at_10, y, s=90, color=col, marker=mk, zorder=3,
                   edgecolor="black", linewidth=0.4)
        ax.annotate(r.short, (r.clir_at_10, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax.set_yscale("log")  # one outlier (e5) at ~98x compresses a linear axis; log separates the cluster
    ax.set_xlabel("CLIR@10 (cross-lingual recall@10, higher = better)")
    ax.set_ylabel("XRC50  =  D50(cross-lang) / D50(same-lang)\n(median reading-cost multiplier, log scale, lower = better)")
    ax.set_title("Cross-Lingual Reading-Cost Multiplier vs CLIR@10\n"
                 "(median coverage; dashed = parity; better models cluster low-XRC / high-CLIR)")
    fig.tight_layout(); fig.savefig(out / "xrc_vs_clir.png"); plt.close(fig)

    # ---- figure 2: RRC ceiling stacked bar ----
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    order = rrc_df.sort_values("RRC_at_100", ascending=False)
    xs = np.arange(len(order))
    rec100 = order["RRC_at_100"].to_numpy()
    found_mid = order["found_101_1000"].to_numpy()
    lost = order["lost_at_1000"].to_numpy()
    ax.bar(xs, rec100, color="#228833", label="recoverable by re-rank @top-100")
    ax.bar(xs, found_mid, bottom=rec100, color="#aacc99", label="found in 101..1000 (needs deeper pool)")
    ax.bar(xs, lost, bottom=rec100 + found_mid, color="#cc6677", label="never in top-1000 (lost forever)")
    for x, l in zip(xs, lost):
        ax.text(x, 1.005, f"{l:.0%}", ha="center", va="bottom", fontsize=7, color="#cc6677")
    ax.set_xticks(xs); ax.set_xticklabels(order["short"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("fraction of cross-gold queries")
    ax.set_ylim(0, 1.08)
    ax.set_title("Re-Ranker Recoverability Ceiling — what a top-100 re-ranker can and cannot fix\n"
                 "(green = re-orderable; red = beyond any re-ranker over the retrieved list)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "rrc_ceiling.png"); plt.close(fig)

    # ---- summary + consistency checks ----
    eg = xrc_df[xrc_df.short == "embeddinggemma"].iloc[0]
    eg_rrc = rrc_df[rrc_df.short == "embeddinggemma"].iloc[0]
    pooled_cross_all = cpq[cpq.n_gold_cross > 0]["first_cross_rank"].to_numpy(dtype=float)
    pooled_lost1000_all = float(np.mean(pooled_cross_all > 1000))
    pooled_cross_pool = cpq[(cpq.model.isin(POOL)) & (cpq.n_gold_cross > 0)]["first_cross_rank"].to_numpy(dtype=float)
    pooled_lost1000_pool = float(np.mean(pooled_cross_pool > 1000))
    summary = {
        "method": "XRC = D_C(cross) / D_C(same); RRC = mean(first_cross_rank <= K)",
        "headline_coverage": "XRC50 (median reading depth) — fully finite, robust; D90/D95 reported "
                             "but right-censored at this sample size (treat as lower bounds).",
        "same_lang_population": "57 originals (only queries with a same-language gold)",
        "cross_lang_population": "137 queries with a cross-language gold",
        "embeddinggemma": {
            "D50_same": eg.D50_same, "D50_cross": eg.D50_cross, "XRC50": eg.XRC50,
            "D75_same": eg.D75_same, "D75_cross": eg.D75_cross, "XRC75": eg.XRC75,
            "D90_same": eg.D90_same, "D90_cross": eg.D90_cross, "XRC90": eg.XRC90,
            "XRC90_censored": bool(eg.XRC90_censored), "cross_censored_frac": eg.cross_censored_frac,
            "RRC_at_100": eg_rrc.RRC_at_100, "RRC_at_1000": eg_rrc.RRC_at_1000,
            "lost_at_1000": eg_rrc.lost_at_1000,
        },
        "consistency_checks": {
            "pooled_lost_share_top1000_8model_pool": round(pooled_lost1000_pool, 4),
            "expected_round04_pooled_lost_share_8model": 0.1542,
            "pooled_lost_share_top1000_all9": round(pooled_lost1000_all, 4),
            "egemma_lost_share_expected_0.0584": eg_rrc.lost_at_1000,
            "egemma_RRC1000_expected_0.9416": eg_rrc.RRC_at_1000,
        },
        "best_xrc50_model": str(xrc_df.loc[xrc_df["XRC50"].apply(
            lambda v: v if isinstance(v, (int, float)) and np.isfinite(v) else np.inf).idxmin(), "short"]),
        "n_models_xrc50_censored": int(xrc_df["XRC50_censored"].sum()),
        "n_models_xrc95_censored": int(xrc_df["XRC95_censored"].sum()),
    }
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] XRC per-model (D = reading depth in #docs; XRC = cross/same):")
    print(xrc_df[["short", "n_cross", "cross_censored_frac", "D50_same", "D50_cross",
                  "XRC50", "XRC75", "XRC90", "XRC95", "clir_at_10"]].to_string(index=False))
    print(f"\n[{SLUG}] RRC per-model:")
    print(rrc_df[["short", "RRC_at_100", "RRC_at_1000", "lost_at_1000"]].to_string(index=False))
    print(f"\n[{SLUG}] CONSISTENCY: pooled lost@1000 (8-model pool) = {pooled_lost1000_pool:.4f} "
          f"(round04 reported 0.1542); all-9 = {pooled_lost1000_all:.4f}; "
          f"egemma RRC@1000 = {eg_rrc.RRC_at_1000} (expect 0.9416)")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
