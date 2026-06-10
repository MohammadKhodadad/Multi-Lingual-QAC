"""
EXTRA (round-1 additive analysis) — n=9 -> n=7 correlation robustness (drop the collapsers).

Do the load-bearing mechanism correlations survive dropping the two degenerate encoders
(gte-base, e5-large-instruct)? Reports Pearson + Spearman on all 9 and on the 7 non-collapsed.

Reads ONLY on-disk per-model CSVs (no parquet, no API):
  round08_separability/per_model.csv   -> auc_cross, clir_at_10
  round05_consistency/per_model.csv    -> rbo, home_adv
  round06_language_collapse/per_model.csv -> mean_overrep, clir_at_10

Load-bearing pairs checked:
  (auc_cross, clir_at_10)   expected Pearson +0.96 (round08)
  (mean_overrep, clir_at_10) expected Pearson -0.60 (round06)
  (home_adv, rbo)           expected Pearson -0.85 (round05)

Outputs -> reports/runs/chem_patents/experimental_plots/extra_correlation_robustness/
  correlation_robustness.csv (pair, pearson_n9, pearson_n7, spearman_n9, spearman_n7), summary.json

Run: /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
       reports/runs/chem_patents/experimental_codes/extra_correlation_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_correlation_robustness"
# IMPORTANT: the published round05/06/08 correlations are computed over the 8-model RELIABLE POOL
# (gte-base excluded), NOT all 9. So the baseline is n=8; the robustness check additionally drops
# e5-large-instruct (n=7). all-9 is shown for transparency (it exposes how fragile home_adv~rbo is
# to the single degenerate encoder).
GTE = "gte-base"
SECOND_COLLAPSER = "e5-large-instruct"


def _corr(x, y, method):
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan"), float("nan")
    if method == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return float(r), float(p)


def main() -> None:
    out = C.round_dir(SLUG)
    sep = pd.read_csv(C.PLOTS_DIR / "round08_separability" / "per_model.csv")
    con = pd.read_csv(C.PLOTS_DIR / "round05_consistency" / "per_model.csv")
    col = pd.read_csv(C.PLOTS_DIR / "round06_language_collapse" / "per_model.csv")

    # assemble a per-model frame keyed by short label
    f = sep[["short", "auc_cross", "clir_at_10"]].merge(
        con[["short", "rbo", "home_adv"]], on="short").merge(
        col[["short", "mean_overrep"]], on="short")

    pairs = [
        ("auc_cross", "clir_at_10", "+0.961 (round08, n=8 pool)"),
        ("mean_overrep", "clir_at_10", "-0.600 (round06, n=8 pool)"),
        ("home_adv", "rbo", "-0.846 (round05, n=8 pool)"),
    ]

    f8 = f[f["short"] != GTE]                                   # published baseline (8-model pool)
    f7 = f8[f8["short"] != SECOND_COLLAPSER]                    # robustness: drop second collapser
    f9 = f                                                      # all 9 (transparency)

    rows = []
    for a, b, expected in pairs:
        pr8, _ = _corr(f8[a], f8[b], "pearson")
        pr7, _ = _corr(f7[a], f7[b], "pearson")
        pr9, _ = _corr(f9[a], f9[b], "pearson")
        sp8, _ = _corr(f8[a], f8[b], "spearman")
        sp7, _ = _corr(f7[a], f7[b], "spearman")
        rows.append({
            "pair": f"{a} ~ {b}", "expected": expected,
            "pearson_n8_pool": round(pr8, 3), "pearson_n7": round(pr7, 3),
            "pearson_n9_all": round(pr9, 3),
            "spearman_n8_pool": round(sp8, 3), "spearman_n7": round(sp7, 3),
            "n8": int(f8[[a, b]].dropna().shape[0]), "n7": int(f7[[a, b]].dropna().shape[0]),
        })
    res = pd.DataFrame(rows)
    res.to_csv(out / "correlation_robustness.csv", index=False)

    # gates: each pair's n8-pool Pearson must reproduce the published number
    gate_auc = abs(res.iloc[0]["pearson_n8_pool"] - 0.961) <= 0.01
    gate_over = abs(res.iloc[1]["pearson_n8_pool"] - (-0.600)) <= 0.01
    gate_home = abs(res.iloc[2]["pearson_n8_pool"] - (-0.846)) <= 0.01

    # flag pairs that weaken materially when the second collapser is dropped (n8 -> n7)
    weakened = [r["pair"] for r in rows
                if abs(r["pearson_n8_pool"]) - abs(r["pearson_n7"]) > 0.20]

    summary = {
        "baseline_population": "8-model reliable pool (gte-base excluded) — matches published rounds",
        "robustness_population": "7 models (also drop e5-large-instruct, the 2nd degenerate encoder)",
        "gates_reproduce_published_n8": {
            "auc_cross~clir = +0.961": bool(gate_auc),
            "mean_overrep~clir = -0.600": bool(gate_over),
            "home_adv~rbo = -0.846": bool(gate_home),
        },
        "rows": res.to_dict("records"),
        "pairs_weakened_on_n7": weakened,
        "key_observation": (
            "home_adv~rbo is the most fragile: Pearson -0.846 on the n=8 pool but only "
            f"{res.iloc[2]['pearson_n9_all']:.3f} when gte-base is re-included (all 9) — a single "
            "degenerate encoder dominates it. On the n=7 set (drop e5 too) it is "
            f"{res.iloc[2]['pearson_n7']:.3f}. The auc_cross~clir and over-rep~clir mechanisms are "
            "more robust (see Spearman n7)."),
    }
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] correlation robustness (baseline = 8-model pool, matches published):")
    print(res.to_string(index=False))
    print(f"[{SLUG}] GATES n8-pool reproduce published: auc~clir={gate_auc}, "
          f"overrep~clir={gate_over}, homeadv~rbo={gate_home}")
    print(f"[{SLUG}] pairs weakened (|r| drop >0.20 on n7): {weakened}")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
