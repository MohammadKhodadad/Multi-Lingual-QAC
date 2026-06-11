"""
EXTRA (round-2 additive analysis) — the RE-RANKER-BUDGET FRONTIER (DO-NOW-2).

Turns RRC (Re-Ranker Recoverability Ceiling) from two scalars (RRC@100, RRC@1000) into a full CURVE
per model, with the three planning objects the novelty critic named as the highest-leverage upgrade:

  * RRC(K)        = fraction of cross-gold queries whose first foreign twin is within top-K
                    (== mean(first_cross_rank <= K)); monotone non-decreasing in K.
  * dRRC/dK       = marginal recoverability, the discrete forward difference normalized per-candidate;
                    reported per-100-candidates for readability. Tells you the payoff of a deeper pool.
  * knee K*       = the elbow: argmax perpendicular distance from (K, RRC(K)) to the chord
                    (K_min, RRC_min) -> (K_max, RRC_max), with K normalized by log10 (RRC saturates
                    fast). Beyond K* extra re-ranker budget buys little.
  * L_inf floor   = 1 - RRC(1000) == lost_at_1000: the STRUCTURAL floor no re-ranker over the
                    retrieved top-1000 can ever recover. This is the unrecoverable tax.

Inputs: raw per-query first-foreign ranks via common.core_per_query() (loads the 9 on-disk parquet
rankings; HF_HUB_OFFLINE=1; NO network, NO embedding, NO API). Regression-checked against the
round-1 rrc_per_model.csv (must reproduce RRC@100 / RRC@1000 exactly).

Writes to a NEW dir:
    reports/runs/chem_patents/experimental_plots/extra_rrc_budget_frontier/
Outputs: rrc_curve.csv, rrc_knee.csv, summary.json, rrc_budget_frontier.png, rrc_xrc_plane.png
Paper figure target name: cp_fig19_rrc_budget.png

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/extra_rrc_budget_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_rrc_budget_frontier"
SEED = 20260610
MAXRANK = 1000
KS = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
DEG_CLIR_CUTOFF = 0.10


def knee_log(Ks: np.ndarray, rrc: np.ndarray) -> tuple[float, float]:
    """Max-distance-to-chord elbow with K normalized on a log10 scale (RRC saturates fast).
    Returns (K_star, RRC_at_Kstar). Deterministic; no extra dependency."""
    x = np.log10(Ks.astype(float))
    y = rrc.astype(float)
    x0, x1 = x[0], x[-1]
    y0, y1 = y[0], y[-1]
    xn = (x - x0) / (x1 - x0) if x1 != x0 else np.zeros_like(x)
    yn = (y - y0) / (y1 - y0) if y1 != y0 else np.zeros_like(y)
    # perpendicular distance from each (xn, yn) to the chord (0,0)->(1,1): |xn - yn| / sqrt(2)
    dist = np.abs(xn - yn) / np.sqrt(2.0)
    j = int(np.argmax(dist))
    return float(Ks[j]), float(rrc[j])


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    cpq = C.core_per_query()

    # regression source (round-1) — read only to CHECK, not to compute from.
    rrc_ref = pd.read_csv(C.PLOTS_DIR / "extra_xrc_reading_cost" / "rrc_per_model.csv")
    xrc_ref = pd.read_csv(C.PLOTS_DIR / "extra_xrc_reading_cost" / "xrc_per_model.csv")
    ref = {r.short: r for r in rrc_ref.itertuples()}
    clir = {C.short(m): float(cpq[cpq.model == m]["clir_at_10"].mean()) for m in C.MODEL_ORDER}

    Ks = np.array(KS, dtype=float)
    curve_rows = []
    knee_rows = []
    curves: dict[str, np.ndarray] = {}
    check_fails = []

    for m in C.MODEL_ORDER:
        s = C.short(m)
        cross = cpq[(cpq.model == m) & (cpq.n_gold_cross > 0)]["first_cross_rank"].to_numpy(float)
        # (count is data-derived; was hardcoded to 137 for the old release)
        rrc = np.array([float(np.mean(cross <= k)) for k in KS])
        curves[s] = rrc

        # monotonicity check
        if not np.all(np.diff(rrc) >= -1e-12):
            check_fails.append(f"{s}: RRC not monotone non-decreasing")

        # regression check vs round-1 rrc_per_model.csv at K=100 and K=1000
        i100, i1000 = KS.index(100), KS.index(1000)
        exp100, exp1000 = float(ref[s].RRC_at_100), float(ref[s].RRC_at_1000)
        if abs(rrc[i100] - exp100) > 1e-3:
            check_fails.append(f"{s}: RRC@100 {rrc[i100]:.4f} != ref {exp100:.4f}")
        if abs(rrc[i1000] - exp1000) > 1e-3:
            check_fails.append(f"{s}: RRC@1000 {rrc[i1000]:.4f} != ref {exp1000:.4f}")

        # marginal dRRC/dK per 100 candidates (forward difference, attributed to the left K)
        dKs = np.diff(Ks)
        dRRC = np.diff(rrc)
        drrc_per100 = (dRRC / dKs) * 100.0
        for i, k in enumerate(KS):
            curve_rows.append({
                "model": m, "short": s, "K": int(k), "RRC": round(float(rrc[i]), 6),
                "dRRC_per100": round(float(drrc_per100[i]), 6) if i < len(drrc_per100) else "",
            })

        k_star, rrc_kstar = knee_log(Ks, rrc)
        l_inf = float(1.0 - rrc[i1000])
        knee_rows.append({
            "model": m, "short": s,
            "K_star": int(k_star), "RRC_at_Kstar": round(rrc_kstar, 4),
            "L_inf": round(l_inf, 4),
            "RRC_at_100": round(float(rrc[i100]), 4),
            "RRC_at_1000": round(float(rrc[i1000]), 4),
            "lost_at_1000_ref": round(float(ref[s].lost_at_1000), 4),
            "clir_at_10": round(clir[s], 4),
            "degenerate_clir": bool(clir[s] < DEG_CLIR_CUTOFF),
        })
        # L_inf == lost_at_1000 check
        if abs(l_inf - float(ref[s].lost_at_1000)) > 1e-3:
            check_fails.append(f"{s}: L_inf {l_inf:.4f} != lost_at_1000 {ref[s].lost_at_1000:.4f}")

    curve_df = pd.DataFrame(curve_rows)
    knee_df = pd.DataFrame(knee_rows)
    curve_df.to_csv(out / "rrc_curve.csv", index=False)
    knee_df.to_csv(out / "rrc_knee.csv", index=False)

    # ---- figure 1: RRC(K) curves on log-K, knee marked, L_inf floor shaded, non-deg models ----
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    nondeg = [m for m in C.MODEL_ORDER if clir[C.short(m)] >= DEG_CLIR_CUTOFF]
    for m in nondeg:
        s = C.short(m)
        rrc = curves[s]
        col = C.MODEL_COLOR[m]
        ax.plot(Ks, rrc, color=col, lw=1.6, marker="o", ms=3.2, label=s, zorder=3)
        kr = knee_df[knee_df.short == s].iloc[0]
        ax.scatter([kr.K_star], [kr.RRC_at_Kstar], s=110, facecolor="none",
                   edgecolor=col, linewidth=1.8, zorder=4)
    # L_inf floor of the best (egemma) and worst non-deg model shaded
    eg = knee_df[knee_df.short == "embeddinggemma"].iloc[0]
    ax.axhline(1.0 - eg.L_inf, color="#228833", ls=":", lw=0.9)
    ax.text(1.1, 1.0 - eg.L_inf - 0.02, f"egemma ceiling 1-L_inf = {1.0 - eg.L_inf:.3f}",
            fontsize=7, color="#228833", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("re-ranker pool depth K (log scale)")
    ax.set_ylabel("RRC(K) = fraction of cross-gold twins inside top-K")
    ax.set_ylim(0, 1.02)
    ax.set_title("Re-ranker budget frontier: RRC(K) per non-degenerate model\n"
                 "(open ring = knee K*; above the curve at K=1000 is the unrecoverable L_inf floor)")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(out / "rrc_budget_frontier.png"); plt.close(fig)

    # ---- figure 2: the single planning plane (x = XRC50 reading-depth cost, y = L_inf floor) ----
    xrc_map = {}
    for r in xrc_ref.itertuples():
        v = pd.to_numeric(pd.Series([r.XRC50]), errors="coerce").iloc[0]
        xrc_map[r.short] = float(v) if np.isfinite(v) else np.nan
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    name2model = {C.short(m): m for m in C.MODEL_ORDER}
    for kr in knee_df.itertuples():
        x = xrc_map.get(kr.short, np.nan)
        if not np.isfinite(x):
            continue
        col = C.MODEL_COLOR[name2model[kr.short]]
        mk = "X" if kr.degenerate_clir else "o"
        edge = "#d62728" if kr.degenerate_clir else "black"
        ax.scatter(x, kr.L_inf, s=120, color=col, marker=mk, zorder=3,
                   edgecolor=edge, linewidth=1.4 if kr.degenerate_clir else 0.5)
        ax.annotate(kr.short, (x, kr.L_inf), fontsize=7.5, xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("reading-depth cost  =  XRC50  (log)")
    ax.set_ylabel("structural floor  L_inf = 1 - RRC(1000)  (unrecoverable)")
    ax.set_title("Single planning object: reading-depth cost (XRC50) vs structural floor (L_inf)\n"
                 "(bottom-left = cheap to read AND little lost beyond re-ranking)")
    fig.tight_layout(); fig.savefig(out / "rrc_xrc_plane.png"); plt.close(fig)

    eg_knee = knee_df[knee_df.short == "embeddinggemma"].iloc[0]
    e5_knee = knee_df[knee_df.short == "e5-large-instruct"].iloc[0]
    summary = {
        "method": ("RRC(K)=mean(first_cross_rank<=K) on a K-grid; knee=max-distance-to-chord with "
                   "log10-K normalization; L_inf=1-RRC(1000); dRRC/dK forward diff per 100 cands."),
        "k_grid": KS,
        "knee_normalization": "log10(K)",
        "embeddinggemma": {
            "K_star": int(eg_knee.K_star), "RRC_at_Kstar": float(eg_knee.RRC_at_Kstar),
            "L_inf": float(eg_knee.L_inf), "RRC_at_100": float(eg_knee.RRC_at_100),
            "RRC_at_1000": float(eg_knee.RRC_at_1000),
        },
        "e5_large_instruct": {
            "K_star": int(e5_knee.K_star), "L_inf": float(e5_knee.L_inf),
        },
        "L_inf_by_model": {r.short: float(r.L_inf) for r in knee_df.itertuples()},
        "K_star_by_model": {r.short: int(r.K_star) for r in knee_df.itertuples()},
        "regression_checks_passed": len(check_fails) == 0,
        "regression_failures": check_fails,
        "verify_egemma_RRC100_0.7445": float(eg_knee.RRC_at_100),
        "verify_egemma_RRC1000_0.9416": float(eg_knee.RRC_at_1000),
        "verify_egemma_Linf_0.0584": float(eg_knee.L_inf),
        "verify_e5_Linf_0.3723": float(e5_knee.L_inf),
    }
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] RRC knee / floor per model:")
    print(knee_df[["short", "K_star", "RRC_at_Kstar", "L_inf", "RRC_at_100", "RRC_at_1000",
                   "clir_at_10"]].to_string(index=False))
    if check_fails:
        print(f"\n[{SLUG}] !!! REGRESSION CHECK FAILURES:")
        for f in check_fails:
            print("   ", f)
    else:
        print(f"\n[{SLUG}] all regression checks PASSED (RRC@100/@1000 reproduce ref; L_inf==lost@1000; "
              f"RRC monotone).")
    print(f"[{SLUG}] egemma knee K*={eg_knee.K_star} RRC(K*)={eg_knee.RRC_at_Kstar} L_inf={eg_knee.L_inf}")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
