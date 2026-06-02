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

ROOT = Path(__file__).resolve().parents[1]
HUMAN_XLSX = ROOT / "Evaluated data.xlsx"
LLM_CSV = ROOT / "balanced_100_qac_regraded.csv"
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
