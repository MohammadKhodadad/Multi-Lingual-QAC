"""
EXTRA (round-3 additive analysis) — M2: the ARI ALIGNMENT-RECOVERABILITY DECOMPOSITION.

Operationalizes the paper's thesis ("align, don't re-rank") as a *measured per-model split* of the
cross-lingual shortfall (1 = perfect cross-lingual recall) into three exhaustive, mutually-exclusive
buckets, read directly off the already-emitted RRC(K) curve:

  * recoverable-cheaply  = RRC@K      — fraction whose foreign twin is inside a top-K re-ranker pool;
                                        recovered by re-ranking a *cheap* (top-K) candidate set.
  * recoverable-deeply   = RRC@1000 - RRC@K
                                      — fraction recoverable only by a *deep* (K..1000) pool.
  * alignment-only floor = L_inf = 1 - RRC@1000
                                      — the structural floor NO re-ranker over the retrieved top-1000
                                        can move; only representation alignment can.

Identity (closes to 1.0 per model, by construction):
    RRC@K + (RRC@1000 - RRC@K) + L_inf == 1.

ARI (Alignment-Recoverability Index) = L_inf / (1 - RRC@K)
    = the share of the *post-cheap-re-ranking residual* that is alignment-only (un-rerankable).
    HIGH ARI => what is left after a cheap re-ranker is mostly an alignment problem, not a depth one.

Two split points are reported per model:
  * K = 100  (PRIMARY)  — the "realistic top-100 re-ranker" the paper already names.
  * K = K*   (per-model knee)  — the elbow of the RRC curve; the practical-budget read.

This is a PURE TRANSFORM of two on-disk CSVs already emitted by extra_rrc_budget_frontier.py:
    .../extra_rrc_budget_frontier/rrc_curve.csv  (RRC at every K)
    .../extra_rrc_budget_frontier/rrc_knee.csv   (RRC@100, RRC@1000, L_inf, K_star, degenerate_clir)
It recomputes NOTHING from parquet (so it is auto-consistent with cp_fig19), hits NO network, NO API,
NO embedding. HF_HUB_OFFLINE=1 inherited via common.

Writes to a NEW dir, never touching existing outputs:
    reports/runs/chem_patents/experimental_plots/extra_ari_decomposition/
Outputs: ari_decomposition.csv, summary.json, ari_decomposition.png
Paper figure target name: cp_fig22_ari_decomposition.png

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/extra_ari_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_ari_decomposition"
SEED = 20260610
K_PRIMARY = 100  # the "realistic top-100 re-ranker" budget the paper already names
TOL = 1e-6


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    src_dir = C.PLOTS_DIR / "extra_rrc_budget_frontier"
    curve_path = src_dir / "rrc_curve.csv"
    knee_path = src_dir / "rrc_knee.csv"
    for p in (curve_path, knee_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing input CSV: {p} (run extra_rrc_budget_frontier.py first)")

    curve = pd.read_csv(curve_path)
    knee = pd.read_csv(knee_path)

    # RRC lookup at arbitrary K, straight off the curve (auto-consistent with cp_fig19).
    def rrc_at(short: str, k: int) -> float:
        row = curve[(curve["short"] == short) & (curve["K"] == k)]
        if row.empty:
            raise KeyError(f"no RRC for {short} at K={k} in rrc_curve.csv")
        return float(row["RRC"].iloc[0])

    rows = []
    check_fails = []
    for kr in knee.itertuples():
        short = kr.short
        k_star = int(kr.K_star)

        rrc100 = rrc_at(short, K_PRIMARY)
        rrc1000 = rrc_at(short, 1000)
        l_inf = 1.0 - rrc1000

        # --- decomposition at the PRIMARY budget K=100 ---
        cheap_100 = rrc100
        deep_100 = rrc1000 - rrc100
        floor = l_inf
        ident_100 = cheap_100 + deep_100 + floor
        # ARI@100 = alignment-only share of the post-cheap-re-rank residual (1 - RRC@100)
        resid_100 = 1.0 - rrc100
        ari_100 = (l_inf / resid_100) if resid_100 > 0 else float("nan")

        # --- decomposition at the per-model knee K=K* ---
        rrc_kstar = rrc_at(short, k_star)
        cheap_kstar = rrc_kstar
        deep_kstar = rrc1000 - rrc_kstar
        ident_kstar = cheap_kstar + deep_kstar + floor
        resid_kstar = 1.0 - rrc_kstar
        ari_kstar = (l_inf / resid_kstar) if resid_kstar > 0 else float("nan")

        # cross-check every RRC@100/@1000/L_inf against rrc_knee.csv. rrc_curve.csv carries 6-decimal
        # RRC; rrc_knee.csv is rounded to 4 decimals, so the agreement tolerance is rounding-level
        # (1e-3), not bit-level — the curve is the authoritative higher-precision source.
        if abs(rrc100 - float(kr.RRC_at_100)) > 1e-3:
            check_fails.append(f"{short}: RRC@100 curve {rrc100} != knee {kr.RRC_at_100}")
        if abs(rrc1000 - float(kr.RRC_at_1000)) > 1e-3:
            check_fails.append(f"{short}: RRC@1000 curve {rrc1000} != knee {kr.RRC_at_1000}")
        if abs(l_inf - float(kr.L_inf)) > 1e-3:
            check_fails.append(f"{short}: L_inf computed {l_inf:.4f} != knee {kr.L_inf}")
        # identity must close to 1.0 (both budgets)
        if abs(ident_100 - 1.0) > TOL:
            check_fails.append(f"{short}: K=100 identity {ident_100:.8f} != 1.0")
        if abs(ident_kstar - 1.0) > TOL:
            check_fails.append(f"{short}: K=K* identity {ident_kstar:.8f} != 1.0")
        # buckets must be non-negative (deep>=0 because RRC monotone non-decreasing)
        if deep_100 < -TOL or deep_kstar < -TOL:
            check_fails.append(f"{short}: negative deep bucket (RRC not monotone?)")

        rows.append({
            "model": kr.model, "short": short,
            "degenerate_clir": bool(kr.degenerate_clir),
            "clir_at_10": round(float(kr.clir_at_10), 4),
            "K_star": k_star,
            # K=100 split
            "RRC_at_100": round(cheap_100, 4),
            "deep_100_to_1000": round(deep_100, 4),
            "L_inf": round(floor, 4),
            "identity_sum_100": round(ident_100, 6),
            "ARI_at_100": round(ari_100, 4) if np.isfinite(ari_100) else float("nan"),
            # K=K* split
            "RRC_at_Kstar": round(cheap_kstar, 4),
            "deep_Kstar_to_1000": round(deep_kstar, 4),
            "identity_sum_Kstar": round(ident_kstar, 6),
            "ARI_at_Kstar": round(ari_kstar, 4) if np.isfinite(ari_kstar) else float("nan"),
        })

    ari_df = pd.DataFrame(rows)
    # keep models in the canonical recall order
    order_map = {C.short(m): i for i, m in enumerate(C.MODEL_ORDER)}
    ari_df["_o"] = ari_df["short"].map(order_map)
    ari_df = ari_df.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    ari_df.to_csv(out / "ari_decomposition.csv", index=False)

    # ---- figure: stacked bar over the 7 NON-DEGENERATE models (cheap / deep / floor at K=100) ----
    nondeg = ari_df[~ari_df["degenerate_clir"]].copy()
    deg = ari_df[ari_df["degenerate_clir"]].copy()
    # order the bars by capability (RRC@100 desc) for readability
    nondeg = nondeg.sort_values("RRC_at_100", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    xs = np.arange(len(nondeg))
    cheap = nondeg["RRC_at_100"].to_numpy()
    deep = nondeg["deep_100_to_1000"].to_numpy()
    floor = nondeg["L_inf"].to_numpy()

    ax.bar(xs, cheap, color="#228833", label="recoverable cheaply  (RRC@100: foreign twin in top-100)")
    ax.bar(xs, deep, bottom=cheap, color="#aacc99",
           label="recoverable deeply  (RRC@1000 - RRC@100: needs a deeper pool)")
    ax.bar(xs, floor, bottom=cheap + deep, color="#cc6677",
           label="alignment-only floor  (L_inf = 1 - RRC@1000: un-rerankable)")
    # annotate the alignment floor and the ARI@100 scalar above each bar
    for x, (fl, ari) in enumerate(zip(floor, nondeg["ARI_at_100"])):
        ax.text(x, 1.012, f"L_inf {fl:.1%}", ha="center", va="bottom", fontsize=7, color="#cc6677")
        ax.text(x, 1.055, f"ARI {ari:.2f}", ha="center", va="bottom", fontsize=7.5,
                color="#7a2b34", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(nondeg["short"], rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel("share of cross-gold queries (decomposition of the cross-lingual shortfall)")
    ax.set_ylim(0, 1.13)
    ax.axhline(1.0, color="grey", ls="--", lw=0.7)
    ax.set_title(
        "ARI alignment-recoverability decomposition (7 non-degenerate models, split at K=100)\n"
        "cheap (re-rankable) + deep (deeper pool) + alignment-only floor = 1; "
        "ARI = L_inf / (1 - RRC@100)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ari_decomposition.png")
    plt.close(fig)

    # ---- summary ----
    eg = ari_df[ari_df["short"] == "embeddinggemma"].iloc[0]
    summary = {
        "method": (
            "Pure transform of rrc_curve.csv / rrc_knee.csv. Per model: "
            "cheap=RRC@K, deep=RRC@1000-RRC@K, floor=L_inf=1-RRC@1000; "
            "ARI@K = L_inf / (1 - RRC@K). Identity cheap+deep+floor == 1."),
        "K_primary": K_PRIMARY,
        "n_models_total": int(len(ari_df)),
        "n_models_nondegenerate": int((~ari_df["degenerate_clir"]).sum()),
        "degenerate_models_excluded_from_figure": deg["short"].tolist(),
        "identity_closes_all_models": bool(
            (ari_df["identity_sum_100"].sub(1.0).abs() < TOL).all()
            and (ari_df["identity_sum_Kstar"].sub(1.0).abs() < TOL).all()),
        "embeddinggemma": {
            "RRC_at_100_cheap": float(eg.RRC_at_100),
            "deep_100_to_1000": float(eg.deep_100_to_1000),
            "L_inf_floor": float(eg.L_inf),
            "identity_sum_100": float(eg.identity_sum_100),
            "ARI_at_100": float(eg.ARI_at_100),
            "K_star": int(eg.K_star),
            "RRC_at_Kstar_cheap": float(eg.RRC_at_Kstar),
            "deep_Kstar_to_1000": float(eg.deep_Kstar_to_1000),
            "ARI_at_Kstar": float(eg.ARI_at_Kstar),
        },
        "ARI_at_100_by_model": {r.short: float(r.ARI_at_100) for r in ari_df.itertuples()},
        "ARI_at_Kstar_by_model": {r.short: float(r.ARI_at_Kstar) for r in ari_df.itertuples()},
        "L_inf_by_model": {r.short: float(r.L_inf) for r in ari_df.itertuples()},
        "verify_egemma_cheap_0.7445": float(eg.RRC_at_100),
        "verify_egemma_deep_0.1971": float(eg.deep_100_to_1000),
        "verify_egemma_floor_0.0584": float(eg.L_inf),
        "verify_egemma_ARI100_0.229": float(eg.ARI_at_100),
        "checks_passed": len(check_fails) == 0,
        "check_failures": check_fails,
    }
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] ARI decomposition (split at K={K_PRIMARY}):")
    print(ari_df[["short", "degenerate_clir", "RRC_at_100", "deep_100_to_1000", "L_inf",
                  "identity_sum_100", "ARI_at_100", "K_star", "ARI_at_Kstar"]].to_string(index=False))
    print(f"\n[{SLUG}] identity (cheap+deep+floor==1) closes for all models: "
          f"{summary['identity_closes_all_models']}")
    print(f"[{SLUG}] egemma: cheap={eg.RRC_at_100} deep={eg.deep_100_to_1000} floor={eg.L_inf} "
          f"-> sum={eg.identity_sum_100}  ARI@100={eg.ARI_at_100}")
    if check_fails:
        print(f"\n[{SLUG}] !!! CHECK FAILURES:")
        for f in check_fails:
            print("   ", f)
    else:
        print(f"[{SLUG}] all identity / cross-CSV checks PASSED.")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
