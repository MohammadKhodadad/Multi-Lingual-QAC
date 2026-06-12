"""
Recompute XRC / RRC@K / ARI on the FULL 524-query chem-patents gold.

The shipped extra_*.csv tables under chem_patents/experimental_plots were computed off the stale
137-query gold (cached qrels with Spanish gold = 0). This module re-derives the same quantities,
replicating the reference implementations exactly (extra_xrc_reading_cost.py,
extra_rrc_budget_frontier.py, extra_ari_decomposition.py), but on the correct 524-query reconstructed
gold from fp_common. Outputs go to final_plots/data/ and are consumed by claimE.

Definitions (verbatim from the reference scripts):
  first_cross_rank = min rank among the query's cross-language gold (foreign twins); inf if none in top-1000.
  first_same_rank  = min rank among same-language gold; inf if none.
  D_C              = nearest-rank C-th percentile reading depth (inf-censored).
  XRC50            = D50_cross / D50_same   (per model).
  RRC@K            = mean(first_cross_rank <= K) over cross-gold queries.
  K*               = max-distance-to-chord knee of RRC(K) on log10 K.
  L_inf            = 1 - RRC@1000;  ARI@100 = L_inf / (1 - RRC@100).
  degenerate       = CLIR@10 < 0.10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fp_common as fp

KS = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
DEG_CLIR_CUTOFF = 0.10
MAXRANK = 1000


def _first_ranks() -> pd.DataFrame:
    """(model, query_id) -> first_same_rank, first_cross_rank, n_gold_same, n_gold_cross, query_language."""
    tab = fp.gp_query_table()
    recs = []
    for r in tab.itertuples():
        for d in r.gold_same:
            recs.append((r.query_id, d, "same"))
        for d in r.gold_cross:
            recs.append((r.query_id, d, "cross"))
    gold_long = pd.DataFrame(recs, columns=["query_id", "corpus_id", "kind"])
    rk = fp.gp_rankings()[["model", "query_id", "rank", "corpus_id"]]
    merged = rk.merge(gold_long, on=["query_id", "corpus_id"], how="inner")
    first = (merged.groupby(["model", "query_id", "kind"])["rank"].min()
                   .unstack("kind").reset_index())
    # full (model, query) grid so non-retrieved golds become inf
    grid = pd.MultiIndex.from_product([fp.MODEL_ORDER, tab["query_id"]],
                                      names=["model", "query_id"]).to_frame(index=False)
    meta = tab.set_index("query_id")
    grid["n_gold_same"] = grid["query_id"].map(meta["gold_same"].apply(len))
    grid["n_gold_cross"] = grid["query_id"].map(meta["gold_cross"].apply(len))
    grid["query_language"] = grid["query_id"].map(meta["query_language"])
    out = grid.merge(first, on=["model", "query_id"], how="left")
    out["first_same_rank"] = np.where(out["n_gold_same"] > 0, out.get("same", np.nan), np.nan)
    out["first_cross_rank"] = np.where(out["n_gold_cross"] > 0, out.get("cross", np.nan), np.nan)
    # within-domain non-retrieved -> inf (censored)
    out.loc[(out["n_gold_same"] > 0) & out["first_same_rank"].isna(), "first_same_rank"] = np.inf
    out.loc[(out["n_gold_cross"] > 0) & out["first_cross_rank"].isna(), "first_cross_rank"] = np.inf
    return out


def _pct_depth(depths: np.ndarray, q: float):
    d = np.sort(np.asarray(depths, dtype=float))
    n = d.size
    if n == 0:
        return float("nan"), False
    idx = min(max(int(np.ceil(q / 100.0 * n)) - 1, 0), n - 1)
    val = d[idx]
    return (float("inf"), True) if not np.isfinite(val) else (float(val), False)


def _knee_log(Ks: np.ndarray, rrc: np.ndarray):
    x = np.log10(Ks.astype(float)); y = rrc.astype(float)
    xn = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
    yn = (y - y[0]) / (y[-1] - y[0]) if y[-1] != y[0] else np.zeros_like(y)
    j = int(np.argmax(np.abs(xn - yn) / np.sqrt(2.0)))
    return float(Ks[j]), float(rrc[j])


def compute():
    fr = _first_ranks()
    cpq = fp.cp_per_query()
    clir = {fp.short(m): float(cpq[cpq["model"] == m]["clir_at_10"].mean()) for m in fp.MODEL_ORDER}

    cost_rows, knee_rows, curve_rows, ari_rows = [], [], [], []
    Ks = np.array(KS, dtype=float)
    for m in fp.MODEL_ORDER:
        s = fp.short(m)
        sub = fr[fr["model"] == m]
        cross = sub.loc[sub["n_gold_cross"] > 0, "first_cross_rank"].to_numpy(float)
        same = sub.loc[sub["n_gold_same"] > 0, "first_same_rank"].to_numpy(float)
        d50c, c50c = _pct_depth(cross, 50)
        d50s, c50s = _pct_depth(same, 50)
        if not np.isfinite(d50s) or d50s == 0:
            xrc50, xrc_cens = float("nan"), True
        elif not np.isfinite(d50c):
            xrc50, xrc_cens = MAXRANK / d50s, True
        else:
            xrc50, xrc_cens = d50c / d50s, bool(c50c or c50s)
        degen = clir[s] < DEG_CLIR_CUTOFF

        rrc = np.array([float(np.mean(cross <= k)) for k in KS])
        for i, k in enumerate(KS):
            curve_rows.append({"model": m, "short": s, "K": int(k), "RRC": round(float(rrc[i]), 6)})
        k_star, rrc_kstar = _knee_log(Ks, rrc)
        rrc100, rrc1000 = float(rrc[KS.index(100)]), float(rrc[KS.index(1000)])
        l_inf = 1.0 - rrc1000
        ari_100 = l_inf / (1.0 - rrc100) if (1.0 - rrc100) > 0 else float("nan")

        cost_rows.append({"model": m, "short": s, "clir_at_10": round(clir[s], 4),
                          "XRC50": round(xrc50, 3) if np.isfinite(xrc50) else xrc50,
                          "XRC50_censored": xrc_cens, "degenerate_clir": degen})
        knee_rows.append({"model": m, "short": s, "K_star": int(k_star),
                          "RRC_at_Kstar": round(rrc_kstar, 4), "L_inf": round(l_inf, 4),
                          "RRC_at_100": round(rrc100, 4), "RRC_at_1000": round(rrc1000, 4),
                          "clir_at_10": round(clir[s], 4), "degenerate_clir": degen})
        ari_rows.append({"model": m, "short": s, "degenerate_clir": degen,
                         "clir_at_10": round(clir[s], 4), "K_star": int(k_star),
                         "RRC_at_100": round(rrc100, 4), "deep_100_to_1000": round(rrc1000 - rrc100, 4),
                         "L_inf": round(l_inf, 4), "ARI_at_100": round(ari_100, 4)})

    cost = pd.DataFrame(cost_rows)
    # Pareto frontier among non-degenerate: max clir, min XRC50
    nd = cost[~cost["degenerate_clir"]].copy()
    on = []
    for r in nd.itertuples():
        dominated = ((nd["clir_at_10"] >= r.clir_at_10) & (nd["XRC50"] <= r.XRC50) &
                     ((nd["clir_at_10"] > r.clir_at_10) | (nd["XRC50"] < r.XRC50))).any()
        on.append(not dominated)
    nd["on_frontier"] = on
    cost = cost.merge(nd[["short", "on_frontier"]], on="short", how="left")
    cost["on_frontier"] = cost["on_frontier"].fillna(False)

    fp.dump_data(cost, "E_cost_frontier")
    fp.dump_data(pd.DataFrame(curve_rows), "E_rrc_curve")
    fp.dump_data(pd.DataFrame(knee_rows), "E_rrc_knee")
    fp.dump_data(pd.DataFrame(ari_rows), "E_ari_decomposition")
    return cost, pd.DataFrame(knee_rows), pd.DataFrame(curve_rows), pd.DataFrame(ari_rows)


if __name__ == "__main__":
    cost, knee, curve, ari = compute()
    print("XRC/RRC/ARI recomputed on 524-query gold:")
    print(cost.to_string(index=False))
    print()
    print(ari[["short", "clir_at_10", "K_star", "RRC_at_100", "L_inf", "ARI_at_100"]].to_string(index=False))
