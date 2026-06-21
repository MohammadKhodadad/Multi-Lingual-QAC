"""
Compare human-annotator QAC scores (Evaluated data.xlsx) against LLM
auto-grader scores (balanced_100_qac_regraded.csv).

Produces JSON-friendly summary stats used by the Markdown report under
reports/human_eval/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# Human annotations cover only a 97-row subset; the agreement metrics are
# computed on that subset (inner-joined to the LLM grades).
HUMAN_XLSX = ROOT / "Evaluated data_by_annotator.xlsx"
LLM_CSV = ROOT / "data" / "google_patents" / "qac" / "qac_chempatents_best.csv"
OUT_DIR = ROOT / "reports" / "human_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bucket(score: float, scale: int) -> str:
    """Map a score to {poor, ok, good} on a 0–scale axis using the
    user-specified bands (0–3 poor, 4–6 ok, 7–10 good on a /10 scale)."""
    s10 = score * 10.0 / scale
    if s10 <= 3:
        return "poor"
    if s10 <= 6:
        return "ok"
    return "good"


def icc_two_way(rater_a: np.ndarray, rater_b: np.ndarray, alpha: float = 0.05) -> dict:
    """Two-rater intra-class correlation (Shrout & Fleiss 1979 / McGraw & Wong
    1996), single-measurement forms, with 95% CIs.

    - ICC(2,1) = absolute agreement: penalizes systematic rater offset.
    - ICC(3,1) = consistency: rank/scaling agreement, ignores the offset
      (≈ Pearson on a paired design); use this to compare against Pearson r.

    Inputs are paired scores for n subjects; they must be on the SAME scale
    (here: the 0–1 normalized human/LLM percentages). ICC is scale-invariant.
    """
    x = np.column_stack([np.asarray(rater_a, float), np.asarray(rater_b, float)])
    n, k = x.shape  # subjects, raters (k=2)

    grand = x.mean()
    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()      # between subjects
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()      # between raters
    ss_err = ss_total - ss_rows - ss_cols

    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))

    icc2 = (msr - mse) / (msr + (k - 1) * mse + (k / n) * (msc - mse))
    icc3 = (msr - mse) / (msr + (k - 1) * mse)

    # CI for ICC(3,1) — consistency
    f3 = msr / mse
    fl = f3 / stats.f.ppf(1 - alpha / 2, n - 1, (n - 1) * (k - 1))
    fu = f3 * stats.f.ppf(1 - alpha / 2, (n - 1) * (k - 1), n - 1)
    icc3_lo = (fl - 1) / (fl + (k - 1))
    icc3_hi = (fu - 1) / (fu + (k - 1))

    # CI for ICC(2,1) — absolute agreement (McGraw & Wong 1996)
    a = (k * icc2) / (n * (1 - icc2))
    b = 1 + (k * icc2 * (n - 1)) / (n * (1 - icc2))
    v = (a * msc + b * mse) ** 2 / (
        (a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1))
    )
    f_u = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f_l = stats.f.ppf(1 - alpha / 2, v, n - 1)
    icc2_lo = (
        n * (msr - f_u * mse)
        / (f_u * (k * msc + (k * n - k - n) * mse) + n * msr)
    )
    icc2_hi = (
        n * (f_l * msr - mse)
        / (k * msc + (k * n - k - n) * mse + n * f_l * msr)
    )

    return {
        "n": n,
        "icc2_1_absolute_agreement": round(float(icc2), 3),
        "icc2_1_ci95": [round(float(icc2_lo), 3), round(float(icc2_hi), 3)],
        "icc3_1_consistency": round(float(icc3), 3),
        "icc3_1_ci95": [round(float(icc3_lo), 3), round(float(icc3_hi), 3)],
    }


def weighted_kappa(a: pd.Series, b: pd.Series, order=("poor", "ok", "good")) -> float:
    """Linear-weighted Cohen's kappa on ordered categorical buckets.

    Chance-corrected: ~0 means agreement is no better than expected from the
    marginal rates alone. Collapses toward 0 when one rater has (near-)zero
    spread across buckets, which the raw agreement % hides.
    """
    idx = {c: i for i, c in enumerate(order)}
    n, m = len(a), len(order)
    ai = a.map(idx).to_numpy()
    bi = b.map(idx).to_numpy()
    obs = np.zeros((m, m))
    for x, y in zip(ai, bi):
        obs[x, y] += 1
    row, col = obs.sum(1), obs.sum(0)
    exp = np.outer(row, col) / n
    w = np.array([[abs(i - j) / (m - 1) for j in range(m)] for i in range(m)])
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return round(float(1 - (w * obs).sum() / denom), 3)


def main() -> None:
    human = pd.read_excel(HUMAN_XLSX, sheet_name="qac_with_modes")
    llm = pd.read_csv(LLM_CSV)

    # LLM total_score range = faith_overall (max 15) + qual_overall (max 25) = /40
    llm_max = 40
    human_max = 10

    joined = human.rename(columns={"total_score": "human_total"}).merge(
        llm[["corpus_id", "question", "total_score"]].rename(
            columns={"total_score": "llm_total"}
        ),
        on=["corpus_id", "question"],
        how="inner",
    )
    assert len(joined) == len(human), "join lost rows"

    joined["human_bucket"] = joined["human_total"].apply(lambda s: bucket(s, human_max))
    joined["llm_bucket"] = joined["llm_total"].apply(lambda s: bucket(s, llm_max))
    joined["human_norm"] = joined["human_total"] / human_max
    joined["llm_norm"] = joined["llm_total"] / llm_max

    summary: dict = {}

    # ---- 1. Human-only stats ------------------------------------------------
    summary["overall_human"] = {
        "n": len(joined),
        "mean": float(joined["human_total"].mean()),
        "median": float(joined["human_total"].median()),
        "min": int(joined["human_total"].min()),
        "max": int(joined["human_total"].max()),
        "bucket_counts": joined["human_bucket"]
        .value_counts()
        .reindex(["poor", "ok", "good"], fill_value=0)
        .to_dict(),
        "score_distribution": joined["human_total"]
        .value_counts()
        .sort_index()
        .to_dict(),
    }

    by_mode = (
        joined.groupby("mode")["human_total"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )
    summary["human_by_mode"] = by_mode.reset_index().to_dict(orient="records")

    by_strategy = (
        joined.groupby(["mode", "strategy_name"])["human_total"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )
    summary["human_by_strategy"] = by_strategy.reset_index().to_dict(orient="records")

    # Bucket share by strategy
    bucket_by_strategy = (
        joined.groupby(["mode", "strategy_name", "human_bucket"]).size().unstack(fill_value=0)
    )
    for col in ["poor", "ok", "good"]:
        if col not in bucket_by_strategy.columns:
            bucket_by_strategy[col] = 0
    bucket_by_strategy = bucket_by_strategy[["poor", "ok", "good"]]
    bucket_by_strategy["n"] = bucket_by_strategy.sum(axis=1)
    bucket_by_strategy["good_pct"] = (
        bucket_by_strategy["good"] / bucket_by_strategy["n"] * 100
    ).round(1)
    summary["human_bucket_by_strategy"] = bucket_by_strategy.reset_index().to_dict(
        orient="records"
    )

    # Per-dimension means by mode
    tech_dims = [
        "faith_grounding",
        "faith_precision",
        "faith_numerical_fidelity",
        "faith_overall",
        "qual_search_bar_realism",
        "qual_specificity",
        "qual_phrasing_economy",
        "qual_focus",
        "qual_linguistic_quality",
        "qual_overall",
    ]
    sem_dims = [
        "faith_grounding",
        "faith_precision",
        "faith_numerical_fidelity",
        "faith_overall",
        "qual_search_realism",
        "qual_lexical_distance",
        "qual_conceptual_framing",
        "qual_retrievability",
        "qual_linguistic_quality_1",
        "qual_overall_2",
    ]
    tech_df = joined[joined["mode"] == "technical"]
    sem_df = joined[joined["mode"] == "semantic"]
    summary["human_dim_means_technical"] = (
        tech_df[tech_dims].mean().round(2).to_dict()
    )
    summary["human_dim_means_semantic"] = (
        sem_df[sem_dims].mean().round(2).to_dict()
    )

    # ---- 2. Human vs LLM ----------------------------------------------------
    pearson_overall = float(joined["human_norm"].corr(joined["llm_norm"], method="pearson"))
    spearman_overall = float(
        joined["human_norm"].corr(joined["llm_norm"], method="spearman")
    )

    summary["compare_overall"] = {
        "n": len(joined),
        "llm_max": llm_max,
        "human_max": human_max,
        "human_mean_pct": round(joined["human_norm"].mean() * 100, 1),
        "llm_mean_pct": round(joined["llm_norm"].mean() * 100, 1),
        "pearson": round(pearson_overall, 3),
        "spearman": round(spearman_overall, 3),
        "mean_abs_diff_pct": round(
            (joined["human_norm"] - joined["llm_norm"]).abs().mean() * 100, 2
        ),
        "mean_signed_diff_pct_human_minus_llm": round(
            (joined["human_norm"] - joined["llm_norm"]).mean() * 100, 2
        ),
        "icc": icc_two_way(joined["human_norm"].to_numpy(), joined["llm_norm"].to_numpy()),
    }

    # Per-mode
    rows = []
    for mode_name, sub in joined.groupby("mode"):
        rows.append(
            {
                "mode": mode_name,
                "n": len(sub),
                "human_mean_pct": round(sub["human_norm"].mean() * 100, 1),
                "llm_mean_pct": round(sub["llm_norm"].mean() * 100, 1),
                "pearson": round(
                    float(sub["human_norm"].corr(sub["llm_norm"], method="pearson")), 3
                ),
                "spearman": round(
                    float(sub["human_norm"].corr(sub["llm_norm"], method="spearman")),
                    3,
                ),
                "mean_abs_diff_pct": round(
                    (sub["human_norm"] - sub["llm_norm"]).abs().mean() * 100, 2
                ),
                "mean_signed_diff_pct": round(
                    (sub["human_norm"] - sub["llm_norm"]).mean() * 100, 2
                ),
                "icc": icc_two_way(
                    sub["human_norm"].to_numpy(), sub["llm_norm"].to_numpy()
                ),
            }
        )
    summary["compare_by_mode"] = rows

    # Per-strategy
    rows = []
    for (mode_name, strat), sub in joined.groupby(["mode", "strategy_name"]):
        rows.append(
            {
                "mode": mode_name,
                "strategy": strat,
                "n": len(sub),
                "human_mean_pct": round(sub["human_norm"].mean() * 100, 1),
                "llm_mean_pct": round(sub["llm_norm"].mean() * 100, 1),
                "pearson": round(
                    float(sub["human_norm"].corr(sub["llm_norm"], method="pearson")), 3
                )
                if len(sub) >= 3
                else None,
                "spearman": round(
                    float(sub["human_norm"].corr(sub["llm_norm"], method="spearman")), 3
                )
                if len(sub) >= 3
                else None,
                "mean_abs_diff_pct": round(
                    (sub["human_norm"] - sub["llm_norm"]).abs().mean() * 100, 2
                ),
                "mean_signed_diff_pct": round(
                    (sub["human_norm"] - sub["llm_norm"]).mean() * 100, 2
                ),
            }
        )
    summary["compare_by_strategy"] = rows

    # Bucket agreement (cross-tab)
    crosstab = pd.crosstab(
        joined["human_bucket"], joined["llm_bucket"], dropna=False
    ).reindex(index=["poor", "ok", "good"], columns=["poor", "ok", "good"], fill_value=0)
    summary["bucket_crosstab"] = crosstab.to_dict()
    agree = (joined["human_bucket"] == joined["llm_bucket"]).mean()
    summary["bucket_agreement_pct"] = round(float(agree) * 100, 1)
    summary["bucket_weighted_kappa"] = weighted_kappa(
        joined["human_bucket"], joined["llm_bucket"]
    )

    # Save artifacts
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    joined.to_csv(OUT_DIR / "joined_scores.csv", index=False)

    print("Wrote", OUT_DIR / "summary.json")
    print("Wrote", OUT_DIR / "joined_scores.csv")
    print()
    print("--- key numbers ---")
    print("Overall human:", summary["overall_human"])
    print("Overall compare:", summary["compare_overall"])
    print("Bucket agreement %:", summary["bucket_agreement_pct"])


if __name__ == "__main__":
    main()
