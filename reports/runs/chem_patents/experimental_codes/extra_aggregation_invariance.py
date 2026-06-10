"""
EXTRA (round-1 additive analysis) — Aggregation-invariance ribbon.

Question: does the deployment recommendation (embeddinggemma) depend on the specific CLIR-MRS
weighting, or is it rank-1 under *any* sensible aggregation of the same 6 normalized axes?

Reads ONLY one CSV already on disk (no parquet, no embeddings, no API):
    reports/runs/chem_patents/experimental_plots/round10_robustness_synthesis/robustness_axes_normalized.csv
    (cols: accuracy, clir, separability, consistency, mt_robust, lang_parity — min-max normalized)

Four aggregation schemes over the 6 axes:
  1. CLIR-MRS  (reproduced: capability=mean(acc,clir,sep), robustness=mean(cons,mt,parity),
                MRS = capability * (0.5 + 0.5*robustness)) — must match round10 ranks exactly.
  2. Borda count over per-axis ranks (MMTEB-style): sum of per-axis ranks; lower = better.
  3. Equal-weight mean of all 6 normalized axes.
  4. Per-axis winner-take-all: how many of the 6 axes each model wins (higher = better).

Emits per model the rank under each scheme and the rank RANGE (min..max across schemes).

Writes to a NEW dir:
    reports/runs/chem_patents/experimental_plots/extra_aggregation_invariance/
Outputs: aggregation_ranks.csv, summary.json, aggregation_ribbon.png

Run: /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
       reports/runs/chem_patents/experimental_codes/extra_aggregation_invariance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_aggregation_invariance"
AXES = ["accuracy", "clir", "separability", "consistency", "mt_robust", "lang_parity"]
CAP = ["accuracy", "clir", "separability"]
ROB = ["consistency", "mt_robust", "lang_parity"]


def ranks_from_score(score: pd.Series, ascending: bool) -> pd.Series:
    """1 = best. ascending=True means lower score is better (e.g. Borda sum)."""
    return score.rank(ascending=ascending, method="min").astype(int)


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    src = (C.PLOTS_DIR / "round10_robustness_synthesis" / "robustness_axes_normalized.csv")
    df = pd.read_csv(src).set_index("short")

    # ---- Scheme 1: CLIR-MRS (reproduce from the 6 axes) ----
    cap = df[CAP].mean(axis=1)
    rob = df[ROB].mean(axis=1)
    mrs = cap * (0.5 + 0.5 * rob)
    rank_mrs = ranks_from_score(mrs, ascending=False)

    # ---- Scheme 2: Borda count over per-axis ranks ----
    axis_ranks = pd.DataFrame({ax: df[ax].rank(ascending=False, method="min") for ax in AXES})
    borda_sum = axis_ranks.sum(axis=1)  # lower = better
    rank_borda = ranks_from_score(borda_sum, ascending=True)

    # ---- Scheme 3: equal-weight mean of the 6 axes ----
    eqw = df[AXES].mean(axis=1)
    rank_eqw = ranks_from_score(eqw, ascending=False)

    # ---- Scheme 4: per-axis winner-take-all ----
    wins = pd.Series(0, index=df.index)
    for ax in AXES:
        winner = df[ax].idxmax()
        wins[winner] += 1
    rank_wta = ranks_from_score(wins, ascending=False)

    res = pd.DataFrame({
        "model": df["model"],
        "rank_clir_mrs": rank_mrs,
        "rank_borda": rank_borda,
        "rank_equal_weight": rank_eqw,
        "rank_winner_take_all": rank_wta,
        "mrs": mrs.round(4),
        "borda_sum": borda_sum.astype(int),
        "equal_weight_mean": eqw.round(4),
        "axes_won": wins.astype(int),
    })
    scheme_cols = ["rank_clir_mrs", "rank_borda", "rank_equal_weight", "rank_winner_take_all"]
    res["rank_min"] = res[scheme_cols].min(axis=1)
    res["rank_max"] = res[scheme_cols].max(axis=1)
    res = res.sort_values("rank_clir_mrs")
    res.to_csv(out / "aggregation_ranks.csv")

    # ---- verification: scheme-1 ranks must reproduce round10 ----
    r10 = pd.read_csv(C.PLOTS_DIR / "round10_robustness_synthesis" / "robustness_scores.csv").set_index("short")
    r10_rank = r10["rank"].to_dict()
    repro_ok = all(int(res.loc[s, "rank_clir_mrs"]) == int(r10_rank[s]) for s in res.index)

    egemma_ranks = {c: int(res.loc["embeddinggemma", c]) for c in scheme_cols}
    egemma_rank1_count = sum(1 for v in egemma_ranks.values() if v == 1)
    egemma_min = int(res.loc["embeddinggemma", "rank_min"])
    egemma_max = int(res.loc["embeddinggemma", "rank_max"])

    # which middle-of-field models swap across schemes?
    swaps = {}
    for s in res.index:
        rng = (int(res.loc[s, "rank_min"]), int(res.loc[s, "rank_max"]))
        if rng[1] - rng[0] >= 2:
            swaps[s] = rng

    # which model wins mt_robust / lang_parity (often the degenerate encoder, an artifact)
    parity_winner = df["lang_parity"].idxmax()
    mt_winner = df["mt_robust"].idxmax()

    summary = {
        "scheme1_reproduces_round10_ranks": bool(repro_ok),
        "embeddinggemma_ranks": egemma_ranks,
        "embeddinggemma_rank1_under_n_of_4_schemes": egemma_rank1_count,
        "embeddinggemma_rank_range": [egemma_min, egemma_max],
        "headline": (f"embeddinggemma is rank-1 under {egemma_rank1_count}/4 schemes; "
                     f"its rank range is [{egemma_min},{egemma_max}]."),
        "honest_finding": (
            "The deployment recommendation is AGGREGATION-SENSITIVE, not invariant: embeddinggemma "
            "leads all 3 capability axes (accuracy/clir/separability) but is weak on the robustness "
            "axes (mt_robust, lang_parity), so equal-weight and Borda demote it (rank 4 / rank 3). "
            "It is rank-1 only when capability is up-weighted (CLIR-MRS) or by axes-won. The robust, "
            "defensible claim is per-axis dominance on the capability axes, NOT composite invariance."),
        "caveat_winner_take_all": (
            f"axes-won is contaminated by the degenerate encoder: lang_parity is won by '{parity_winner}' "
            f"and mt_robust by '{mt_winner}' — both are 'robust/parity' only because they retrieve almost "
            f"nothing (trivially uniform / unchanged under MT). Treat axes-won with care."),
        "models_with_rank_range_ge_2": swaps,
    }
    C.jdump(summary, out / "summary.json")

    # ---- figure: rank-range ribbon ----
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    order = res.sort_values("rank_min")
    ys = np.arange(len(order))[::-1]
    markers = {"rank_clir_mrs": ("o", "CLIR-MRS"), "rank_borda": ("s", "Borda"),
               "rank_equal_weight": ("^", "equal-weight"), "rank_winner_take_all": ("D", "axes-won")}
    for y, (s, row) in zip(ys, order.iterrows()):
        lo, hi = int(row["rank_min"]), int(row["rank_max"])
        ax.plot([lo, hi], [y, y], color="#999999", lw=3, alpha=0.5, zorder=1)
        for c, (mk, _) in markers.items():
            ax.scatter(int(row[c]), y, marker=mk, s=55, zorder=3,
                       color=C.MODEL_COLOR[row["model"]], edgecolor="black", linewidth=0.3)
    ax.set_yticks(ys); ax.set_yticklabels(order.index)
    ax.set_xlabel("rank across 4 aggregation schemes (1 = best, at left)")
    ax.set_xticks(range(1, len(order) + 1))
    handles = [plt.Line2D([0], [0], marker=mk, color="grey", linestyle="", label=lbl)
               for mk, lbl in markers.values()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="scheme")
    ax.set_title("Aggregation-invariance ribbon — rank range across 4 schemes\n"
                 "(narrow ribbon = recommendation does not depend on the weighting)")
    fig.tight_layout(); fig.savefig(out / "aggregation_ribbon.png"); plt.close(fig)

    print(f"[{SLUG}] scheme-1 reproduces round10 ranks: {repro_ok}")
    print(res[scheme_cols + ["rank_min", "rank_max", "axes_won"]].to_string())
    print(f"[{SLUG}] HEADLINE: {summary['headline']}")
    print(f"[{SLUG}] models with rank range >= 2: {swaps}")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
