"""
Curate the reviewed candidates into main/ and appendix/ with stable, paper-ready names.

Selection rationale lives in ../README.md. Re-running is idempotent: main/ and appendix/ are
cleared and repopulated from candidates/. Each entry copies both .png and .pdf.
"""
from __future__ import annotations

import shutil

import fp_common as fp

# candidate basename -> paper-ready basename
MAIN = {
    "claimA_A1_dumbbell":        "fig_A_technical_vs_semantic",
    "claimB_B1_scatter":         "fig_B_source_transfer",
    "claimC_C1_heatmap":         "fig_C_per_language_recall",
    "claimC_C3_home_advantage":  "fig_C_home_advantage",
    "claimD_D2H_hero":           "fig_D_distractor_latch",
    "claimE_E1_scatter":         "fig_E_cost_capability",
}

APPENDIX = {
    "claimA_A6_question_type":   "figA6_technical_subtype",
    "claimA_A2_distribution":    "figA2_per_query_distribution",
    "claimA_A3_gap_heatmap":     "figA3_gap_by_language",
    "claimA_A5_mt_robustness":   "figA5_mt_robustness",
    "claimB_B3_gap_transfer":    "figB3_penalty_transfer",
    "claimC_C4_epo_heatmap":     "figC4_epo_per_language",
    "claimC_C5_denominators":    "figC5_denominators",
    "claimD_D3_rbo":             "figD3_cross_lingual_rbo",
    "claimD_D4_score_collapse":  "figD4_score_collapse",
    "claimE_E4_rrc_curves":      "figE4_rrc_budget_curves",
    "claimE_E5_ari_stack":       "figE5_ari_decomposition",
}


def _copy(mapping, dest):
    for src, name in mapping.items():
        for ext in ("png", "pdf"):
            s = fp.CAND / f"{src}.{ext}"
            if s.exists():
                shutil.copy2(s, dest / f"{name}.{ext}")


def main():
    for d in (fp.MAIN, fp.APPX):
        for f in d.glob("*"):
            f.unlink()
    _copy(MAIN, fp.MAIN)
    _copy(APPENDIX, fp.APPX)
    print(f"curated {len(MAIN)} main + {len(APPENDIX)} appendix figures")
    print("main:", sorted(p.stem for p in fp.MAIN.glob("*.png")))
    print("appendix:", sorted(p.stem for p in fp.APPX.glob("*.png")))


if __name__ == "__main__":
    main()
