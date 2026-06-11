"""
EXTRA (round-4 additive analysis) — APPENDIX robustness table (DO-NOW-2 + DO-NOW-5).

Converts the three load-bearing scalars from prose-level small-n hedging into "every load-bearing
scalar survives resampling," as ONE appendix table referenced once from §7. NO body float (a table,
no png). CPU-only, 0 API: reads on-disk per-model CSVs + reconstructs per-query depth populations via
common.core_per_query() (loads from the offline HF cache in seconds).

Three resampled blocks (each: point estimate + percentile 95% CI + stability metric + n):

  A1 — separability r=+0.96 (n=7) robustness.
       Input round08_separability/per_model.csv (auc_cross, clir_at_10). Drop the two degenerates
       (gte-base, e5-large-instruct) -> n=7. MODEL-LEVEL bootstrap: resample the 7 model rows with
       replacement N_BOOT times, recompute Pearson r per draw (skip degenerate draws with <3 distinct
       x or y). Report point r, percentile 95% CI, and SIGN-STABILITY = fraction of draws with r>0
       (the load-bearing read — lead with it).

  A6 — XRC50 depth bootstrap for the 3 frontier members (embeddinggemma, bge-m3, granite-278m).
       Per-query depth populations from core_per_query() (first_cross_rank over the 137 cross-gold
       queries) AND the same-language depths reconstructed exactly as extra_xrc_reading_cost.py does
       (the 57 originals with a same-language gold, via first_gold_rank on the same-language gold
       split). For each frontier model: resample the cross-depth population (n=137) and the same-depth
       population (n=57) INDEPENDENTLY with replacement N_BOOT times, recompute XRC50 =
       pct_depth(cross,50)/pct_depth(same,50) per draw using the SAME censoring/percentile rule.
       Report point + percentile 95% CI + censored-draw fraction (a resample whose median lands in
       the right-censored tail yields an inf-bounded ratio — counted, not dropped).

  A5 — ARI@100 egemma-vs-qwen3 0.004-gap order stability.
       ARI@K = L_inf/(1-RRC@K) is a pure transform of RRC; RRC@K = mean(first_cross_rank <= K) over
       the 137 cross queries. PER-QUERY PAIRED bootstrap: resample the 137 cross-query indices with
       replacement N_BOOT times; for egemma and qwen3 recompute RRC@100 and RRC@1000 on the SAME
       resampled index set (paired), then L_inf=1-RRC@1000 and ARI@100=L_inf/(1-RRC@100). Report
       P(ARI@100_egemma < ARI@100_qwen3) and the bootstrap CI on the gap (qwen3 - egemma).

  W2 (DO-NOW-5) — separability partial-r controlling for Recall@10.
       partial r(auc_cross, CLIR@10 | Recall@10) on the n=7 non-degenerate set. HONEST per the
       troubleshooter: this comes back weak/non-significant (~+0.30, n.s.) -> reported descriptively,
       NOT used to harden C3. One row + a summary key with explicit honesty guidance.

Writes to a NEW dir, never touching existing rounds:
    reports/runs/chem_patents/experimental_plots/extra_robustness_appendix/
Outputs: robustness_table.csv (rows: scalar, point, lo, hi, stability_metric, stability_value, n) +
         summary.json. NO png.

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/extra_robustness_appendix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_robustness_appendix"
N_BOOT = 10000
SEED = 20260610
ALPHA = 0.05

DEGENERATES_SHORT = {"gte-base", "e5-large-instruct"}
FRONTIER = ["embeddinggemma", "bge-m3", "granite-278m"]
EG, QW = "embeddinggemma", "qwen3-0.6B"


def pct_depth_vec(depths: np.ndarray, q: float):
    """C-th percentile reading depth over a 1-D vector that may contain np.inf (censored).
    Mirrors extra_xrc_reading_cost.pct_depth EXACTLY (nearest-rank, inf sorts last, right-censoring).
    Returns (value, censored_flag)."""
    d = np.sort(np.asarray(depths, dtype=float))  # inf sorts last
    n = d.size
    if n == 0:
        return float("nan"), False
    idx = int(np.ceil(q / 100.0 * n)) - 1
    idx = min(max(idx, 0), n - 1)
    val = d[idx]
    return (float("inf"), True) if not np.isfinite(val) else (float(val), False)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, NaN if <3 distinct points on either axis (degenerate draw)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3 or np.unique(x).size < 3 or np.unique(y).size < 3:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def partial_r_xy_z(x, y, z) -> float:
    """Partial correlation r(x,y | z): correlate the residuals of x~z and y~z."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)

    def resid(a, b):  # residual of a regressed on [1, b]
        A = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef

    rx, ry = resid(x, z), resid(y, z)
    return pearson_r(rx, ry)


def _pval_from_partial_r(r: float, n: int, k: int) -> float:
    """Two-sided p for a partial r with k controls (df = n - 2 - k)."""
    from scipy.stats import t as tdist
    df = n - 2 - k
    if df <= 0 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan")
    tstat = r * np.sqrt(df / (1.0 - r * r))
    return float(2.0 * tdist.sf(abs(tstat), df))


def main() -> None:
    out = C.round_dir(SLUG)
    rng = np.random.default_rng(SEED)
    rows = []
    summary = {"method": "percentile bootstrap (common.bootstrap_ci family); headline = sign/order "
                         "stability, not CI width. NO BCa (no helper exists; troubleshooter option a). "
                         f"N_BOOT={N_BOOT}, seed={SEED}, alpha={ALPHA}."}

    # ===================================================================== A1: separability r (n=7)
    sep = pd.read_csv(C.PLOTS_DIR / "round08_separability" / "per_model.csv")
    nd = sep[~sep["short"].isin(DEGENERATES_SHORT)].copy()  # n=7
    x = nd["auc_cross"].to_numpy(float)
    y = nd["clir_at_10"].to_numpy(float)
    n7 = x.size
    r_point = pearson_r(x, y)
    # also reproduce n=9 for transparency
    r_n9 = pearson_r(sep["auc_cross"].to_numpy(float), sep["clir_at_10"].to_numpy(float))
    # model-level bootstrap: resample the 7 rows with replacement
    boot_r = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n7, size=n7)
        boot_r[i] = pearson_r(x[idx], y[idx])
    valid = boot_r[np.isfinite(boot_r)]
    n_degenerate_draws = int(N_BOOT - valid.size)
    a1_lo, a1_hi = np.percentile(valid, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    lo, hi = a1_lo, a1_hi
    sign_stability = float(np.mean(valid > 0))
    rows.append({
        "scalar": "A1: separability Pearson r(auc_cross, CLIR@10), n=7 non-degenerate",
        "point": round(r_point, 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
        "stability_metric": "sign-stability P(r>0)", "stability_value": round(sign_stability, 4),
        "n": n7, "note": f"r(n9)={r_n9:.4f}; {n_degenerate_draws} degenerate draws skipped",
    })
    gate_a1 = abs(r_point - 0.958) <= 0.01
    summary["A1_separability_r"] = {
        "point_r_n7": round(r_point, 4), "r_n9": round(r_n9, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "sign_stability_P_r_gt_0": round(sign_stability, 4),
        "n_degenerate_bootstrap_draws": n_degenerate_draws, "n": n7,
        "headline": "sign-stability (P(r>0)) is the load-bearing read, not the CI width",
        "gate_r_point_approx_0.958": bool(gate_a1),
    }

    # ===================================================================== A6: XRC50 depth bootstrap
    cpq = C.core_per_query()
    gold = C.gold_publication()
    ql = C.q_lang()
    lists = C.ranked_lists()
    short2model = {C.short(m): m for m in C.MODEL_ORDER}

    # same-language reading depths (57 originals only), reconstructed EXACTLY as extra_xrc_reading_cost
    same_depth = {sh: [] for sh in FRONTIER}
    for sh in FRONTIER:
        m = short2model[sh]
        for qid, g in gold.items():
            lang = ql.get(qid, "")
            same, _ = C.split_gold(g, lang)
            if same:
                same_depth[sh].append(C.first_gold_rank(lists.get((m, qid), []), same))

    xrc_ref = pd.read_csv(C.PLOTS_DIR / "extra_xrc_reading_cost" / "xrc_per_model.csv")
    summary["A6_XRC50_depth_bootstrap"] = {}
    for sh in FRONTIER:
        m = short2model[sh]
        sub = cpq[(cpq.model == m) & (cpq.n_gold_cross > 0)]
        cross = sub["first_cross_rank"].to_numpy(float)         # n=137, may contain inf
        same = np.asarray(same_depth[sh], dtype=float)          # n=57, may contain inf
        nc, ns = cross.size, same.size
        d50c, _ = pct_depth_vec(cross, 50)
        d50s, _ = pct_depth_vec(same, 50)
        xrc_point = d50c / d50s if (np.isfinite(d50c) and np.isfinite(d50s) and d50s) else float("nan")
        boot_xrc = np.empty(N_BOOT)
        censored = 0
        for i in range(N_BOOT):
            ci = rng.integers(0, nc, size=nc)
            si = rng.integers(0, ns, size=ns)
            dc, fc = pct_depth_vec(cross[ci], 50)
            ds, fs = pct_depth_vec(same[si], 50)
            if not np.isfinite(ds) or ds == 0:
                boot_xrc[i] = np.nan; censored += 1
            elif not np.isfinite(dc):
                boot_xrc[i] = np.inf; censored += 1   # numerator censored -> inf-bounded ratio
            else:
                boot_xrc[i] = dc / ds
        finite = boot_xrc[np.isfinite(boot_xrc)]
        cens_frac = float((N_BOOT - finite.size) / N_BOOT)
        lo, hi = np.percentile(finite, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        ref = float(pd.to_numeric(xrc_ref[xrc_ref.short == sh]["XRC50"], errors="coerce").iloc[0])
        rows.append({
            "scalar": f"A6: XRC50 ({sh})",
            "point": round(xrc_point, 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "stability_metric": "censored-draw fraction", "stability_value": round(cens_frac, 4),
            "n": f"{nc} cross / {ns} same",
            "note": f"ref XRC50={ref}; CI={'lower bound (>5% censored)' if cens_frac > 0.05 else 'finite'}",
        })
        summary["A6_XRC50_depth_bootstrap"][sh] = {
            "point_XRC50": round(xrc_point, 4), "ref_XRC50": ref,
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "censored_draw_frac": round(cens_frac, 4),
            "ci_is_lower_bound": bool(cens_frac > 0.05),
            "n_cross": nc, "n_same": ns,
            "gate_matches_ref": bool(abs(xrc_point - ref) <= 0.01),
        }

    # ===================================================================== A5: ARI@100 egemma vs qwen3
    eg_cross = cpq[(cpq.short == EG) & (cpq.n_gold_cross > 0)]["first_cross_rank"].to_numpy(float)
    qw_cross = cpq[(cpq.short == QW) & (cpq.n_gold_cross > 0)]["first_cross_rank"].to_numpy(float)
    assert eg_cross.size == qw_cross.size, (eg_cross.size, qw_cross.size)  # cross-model equality (count data-derived)
    n_q = eg_cross.size

    def ari100(cross_vec, idx):
        c = cross_vec[idx]
        rrc100 = np.mean(c <= 100)
        rrc1000 = np.mean(c <= 1000)
        linf = 1.0 - rrc1000
        denom = 1.0 - rrc100
        return linf / denom if denom > 0 else np.nan

    eg_point = ari100(eg_cross, np.arange(n_q))
    qw_point = ari100(qw_cross, np.arange(n_q))
    gap_point = qw_point - eg_point   # qwen3 - egemma (positive => egemma lower => paper's claim)
    boot_gap = np.empty(N_BOOT)
    eg_lt_qw = 0
    for i in range(N_BOOT):
        idx = rng.integers(0, n_q, size=n_q)  # PAIRED: same index set for both models
        a_eg = ari100(eg_cross, idx)
        a_qw = ari100(qw_cross, idx)
        boot_gap[i] = a_qw - a_eg
        if a_eg < a_qw:
            eg_lt_qw += 1
    p_eg_lower = float(eg_lt_qw / N_BOOT)
    gfin = boot_gap[np.isfinite(boot_gap)]
    glo, ghi = np.percentile(gfin, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    gap_ci_includes_zero = bool(glo <= 0.0 <= ghi)
    rows.append({
        "scalar": "A5: ARI@100 gap (qwen3 - embeddinggemma)",
        "point": round(gap_point, 4), "ci_lo": round(float(glo), 4), "ci_hi": round(float(ghi), 4),
        "stability_metric": "order-prob P(ARI_egemma < ARI_qwen3)", "stability_value": round(p_eg_lower, 4),
        "n": n_q,
        "note": f"egemma={eg_point:.4f}, qwen3={qw_point:.4f}; gap CI {'INCLUDES 0' if gap_ci_includes_zero else 'excludes 0'}",
    })
    summary["A5_ARI100_gap_egemma_vs_qwen3"] = {
        "ari100_egemma": round(eg_point, 4), "ari100_qwen3": round(qw_point, 4),
        "gap_qwen3_minus_egemma": round(gap_point, 4),
        "gap_ci95": [round(float(glo), 4), round(float(ghi), 4)],
        "gap_ci_includes_zero": gap_ci_includes_zero,
        "order_prob_P_egemma_lower": round(p_eg_lower, 4),
        "n_cross_queries": n_q,
        "gate_egemma_0.2286": bool(abs(eg_point - 0.2286) <= 0.001),
        "gate_qwen3_0.2326": bool(abs(qw_point - 0.2326) <= 0.001),
        "honest_read": ("the 0.004 ARI@100 gap is NOT a reliable ordering: the paired-bootstrap CI "
                        "on the gap straddles 0 and the order-probability is near a coin-flip; report "
                        "the two ARI@100 values as effectively tied, not as a strict egemma<qwen3 win"
                        if gap_ci_includes_zero else
                        "the ARI@100 ordering egemma<qwen3 is stable under the paired bootstrap"),
    }

    # ===================================================================== W2 (DO-NOW-5): partial r
    clir = pd.read_csv(C.PLOTS_DIR / "round01_clir_leaderboard" / "per_model.csv")
    w2 = nd[["short", "auc_cross", "clir_at_10"]].merge(
        clir[["short", "recall_at_10"]], on="short")  # n=7 already (nd is non-degenerate)
    pr = partial_r_xy_z(w2["auc_cross"].to_numpy(float),
                        w2["clir_at_10"].to_numpy(float),
                        w2["recall_at_10"].to_numpy(float))
    zero_order = pearson_r(w2["auc_cross"].to_numpy(float), w2["clir_at_10"].to_numpy(float))
    p_partial = _pval_from_partial_r(pr, n=len(w2), k=1)
    rows.append({
        "scalar": "W2: partial r(auc_cross, CLIR@10 | Recall@10), n=7",
        "point": round(pr, 4), "ci_lo": "", "ci_hi": "",
        "stability_metric": "two-sided p", "stability_value": round(p_partial, 4),
        "n": len(w2),
        "note": f"zero-order r={zero_order:.4f}; WEAK/N.S. once Recall@10 partialled out -> descriptive only",
    })
    summary["W2_separability_partial_r_controlling_recall10"] = {
        "partial_r": round(pr, 4), "p_two_sided": round(p_partial, 4),
        "zero_order_r": round(zero_order, 4), "n": len(w2), "n_controls": 1,
        "honest_guidance": ("Cross-language AUC and overall Recall@10 are strongly collinear across the "
                            "7 non-degenerate models; at this n the separability->CLIR link cannot be "
                            f"statistically disentangled from general capability (partial r={pr:+.3f}, "
                            "n.s.). Frame the separability->floor link as DESCRIPTIVE, not as an effect "
                            "net of capability. Do NOT claim 'not a capability artifact'."),
    }

    # ===================================================================== emit
    table = pd.DataFrame(rows, columns=["scalar", "point", "ci_lo", "ci_hi",
                                        "stability_metric", "stability_value", "n", "note"])
    table.to_csv(out / "robustness_table.csv", index=False)
    summary["gates_all_pass"] = bool(
        gate_a1
        and all(summary["A6_XRC50_depth_bootstrap"][s]["gate_matches_ref"] for s in FRONTIER)
        and summary["A5_ARI100_gap_egemma_vs_qwen3"]["gate_egemma_0.2286"]
        and summary["A5_ARI100_gap_egemma_vs_qwen3"]["gate_qwen3_0.2326"]
    )
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] APPENDIX robustness table:")
    print(table.to_string(index=False))
    print(f"\n[{SLUG}] A1 r(n7)={r_point:.4f} (gate {gate_a1}); sign-stability P(r>0)={sign_stability:.4f}; "
          f"CI=[{a1_lo:.3f},{a1_hi:.3f}]")
    for s in FRONTIER:
        d = summary["A6_XRC50_depth_bootstrap"][s]
        print(f"[{SLUG}] A6 XRC50({s})={d['point_XRC50']} CI={d['ci95']} censored={d['censored_draw_frac']} "
              f"(ref {d['ref_XRC50']}, gate {d['gate_matches_ref']})")
    a5 = summary["A5_ARI100_gap_egemma_vs_qwen3"]
    print(f"[{SLUG}] A5 ARI@100 egemma={a5['ari100_egemma']} qwen3={a5['ari100_qwen3']} "
          f"gap={a5['gap_qwen3_minus_egemma']} CI={a5['gap_ci95']} includes0={a5['gap_ci_includes_zero']} "
          f"P(egemma<qwen3)={a5['order_prob_P_egemma_lower']}")
    print(f"[{SLUG}] W2 partial r={pr:+.4f} p={p_partial:.4f} (zero-order {zero_order:+.4f}) -> descriptive only")
    print(f"[{SLUG}] all gates pass: {summary['gates_all_pass']}")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
