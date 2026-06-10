"""
Curate the paper-ready key_findings/ folder from the 10 rounds:
  * copy the headline figures,
  * assemble a per-model headline_numbers table (CSV + JSON),
  * render EXECUTIVE_SUMMARY.md from the round summaries.

    python reports/runs/chem_patents/experimental_codes/build_key_findings.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

PLOTS = C.PLOTS_DIR
KEY = C.KEY_DIR
FIGS = KEY / "figures"

FIGURES = [
    ("round01_clir_leaderboard", "clir_leaderboard.png", "fig01_clir_leaderboard.png"),
    ("round01_clir_leaderboard", "home_advantage.png", "fig02_home_advantage.png"),
    ("round02_directional_clir", "directional_recall_best.png", "fig03_directional_clir_matrix.png"),
    ("round02_directional_clir", "asymmetry.png", "fig04_clir_direction_asymmetry.png"),
    ("round03_mt_penalty", "mt_penalty_paired.png", "fig05_mt_penalty.png"),
    ("round04_mate_retrieval", "mate_overview.png", "fig06_mate_retrieval.png"),
    ("round04_mate_retrieval", "first_foreign_rank_cdf.png", "fig07_first_foreign_rank.png"),
    ("round05_consistency", "rbo_vs_home_advantage.png", "fig08_consistency_vs_bias.png"),
    ("round06_language_collapse", "overrep_by_query_language.png", "fig09_language_collapse.png"),
    ("round07_distractor_dominance", "blocker_language_by_query_language.png", "fig10_distractor_language.png"),
    ("round08_separability", "auc_by_model.png", "fig11_separability.png"),
    ("round09_ensemble", "ensemble_overview.png", "fig12_ensemble_headroom.png"),
    ("round10_robustness_synthesis", "mrs_leaderboard.png", "fig13_clir_mrs_leaderboard.png"),
    ("round10_robustness_synthesis", "robustness_radar.png", "fig14_robustness_radar.png"),
]


def _summary(slug: str) -> dict:
    p = PLOTS / slug / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _csv(slug: str, name: str) -> pd.DataFrame:
    return pd.read_csv(PLOTS / slug / name)


def main() -> None:
    KEY.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---- copy figures ----------------------------------------------------------------------------
    for slug, src, dst in FIGURES:
        s = PLOTS / slug / src
        if s.exists():
            shutil.copy(s, FIGS / dst)
        else:
            print(f"  (missing figure {s})")

    # ---- headline numbers table ------------------------------------------------------------------
    r1 = _csv("round01_clir_leaderboard", "per_model.csv")[["short", "recall_at_10", "clir_at_10", "molir_at_10", "home_adv"]]
    r4 = _csv("round04_mate_retrieval", "per_model.csv")[["short", "mate_mrr", "mate_hit10", "lost_share"]]
    r5 = _csv("round05_consistency", "per_model.csv")[["short", "rbo"]]
    r6 = _csv("round06_language_collapse", "per_model.csv")[["short", "mean_overrep", "mean_foreign_share"]]
    r8 = _csv("round08_separability", "per_model.csv")[["short", "auc_cross"]]
    r10 = _csv("round10_robustness_synthesis", "robustness_scores.csv")[["short", "capability", "robustness", "MRS", "rank"]]
    h = (r1.merge(r4, on="short").merge(r5, on="short").merge(r6, on="short")
         .merge(r8, on="short").merge(r10, on="short").sort_values("rank"))
    h = h.rename(columns={"mean_overrep": "same_lang_overrep", "mean_foreign_share": "foreign_share_top10",
                          "auc_cross": "separability_auc"})
    h = h.round(4)
    h.to_csv(KEY / "headline_numbers.csv", index=False)
    (KEY / "headline_numbers.json").write_text(json.dumps(h.to_dict(orient="records"), indent=2))

    # ---- gather scalars for the prose ------------------------------------------------------------
    s1, s2, s3 = _summary("round01_clir_leaderboard"), _summary("round02_directional_clir"), _summary("round03_mt_penalty")
    s4, s5, s6 = _summary("round04_mate_retrieval"), _summary("round05_consistency"), _summary("round06_language_collapse")
    s7, s8, s9 = _summary("round07_distractor_dominance"), _summary("round08_separability"), _summary("round09_ensemble")
    s10 = _summary("round10_robustness_synthesis")

    tbl = "\n".join(
        f"| {int(r['rank'])} | `{r['short']}` | {r['recall_at_10']:.3f} | {r['clir_at_10']:.3f} | "
        f"{r['home_adv']:+.2f} | {r['mate_mrr']:.3f} | {r['separability_auc']:.2f} | {r['MRS']:.2f} |"
        for _, r in h.iterrows())

    summary_md = f"""# Chem-patents multilingual retrieval — key findings (CLIR deep-dive)

*Benchmark:* `MehdiAstaraki/multi-lingual-qac-chem-patents` (`multilingual` variant) — **137**
chemistry-patent questions in **5 languages** (en/de/es/fr/zh; 57 human-original + 80
machine-translated), retrieved against the shared **`multilingual_GP`** haystack (**23,487** docs);
**9** multilingual embedding models. Gold = every language version of a question's source patent.

This study adds **CLIR@k** and a family of cross-lingual metrics the standard MTEB report omits, then
iterates over 10 rounds to explain *why* cross-lingual retrieval fails here and *which model to trust*.

## The benchmark's defining property

Two query populations, and they are not symmetric:
- **original (57)** — question in the source patent's language; has exactly one *same-language* gold.
- **synthetic (80)** — question machine-translated into another language; **no same-language gold at
  all** — pure cross-lingual retrieval.
- **Spanish is a pure query-side language**: 34 Spanish queries, **zero** Spanish gold documents. It
  is the benchmark's built-in "no-home" CLIR stress test.

## Seven headline results

1. **Cross-lingual recall trails same-language recall in every model.** Best CLIR@10 =
   **{s1.get('best_clir_at_10', float('nan')):.2f}** ({s1.get('best_clir_model','?')}); the
   home-advantage (MoLIR−CLIR) reaches **+{s1.get('largest_home_adv', float('nan')):.2f}** for the most
   biased model — a same-language copy is far easier than its foreign twin. *(Round 1)*
2. **Retrieval direction is anisotropic.** Hardest direction
   {s2.get('hardest_direction',{}).get('dir','?')} (R@10={s2.get('hardest_direction',{}).get('recall_at_10',float('nan')):.2f});
   English is the easiest *target* language; the most asymmetric pair is
   {s2.get('most_asymmetric_pair',{}).get('pair','?')} (gap {s2.get('most_asymmetric_pair',{}).get('gap',float('nan')):+.2f}).
   Retrieval is not a symmetric similarity. *(Round 2)*
3. **Machine translation of the question is *not* the problem.** Controlling for the patent, the paired
   human−MT difference in cross-lingual reach is
   {(s3.get('paired_mean_diff') if s3.get('paired_mean_diff') is not None else float('nan')):+.3f}
   (p={s3.get('paired_p_value', float('nan')):.2f}) — statistically insignificant. The project's
   "MT-is-fine-for-the-question" assumption holds. *(Round 3)*
4. **Foreign twins are often buried or lost.** Pooled mate-hit@10 = {s4.get('pooled_mate_hit10', float('nan')):.2f};
   **{s4.get('pooled_lost_share_top1000', float('nan')):.0%}** of (query, model) pairs never surface a
   foreign twin even in the **top-1000**. Best twin-finder: {s4.get('best_mate_mrr_model','?')}
   (median first-foreign rank {s4.get('best_median_first_foreign_rank','?')}). *(Round 4)*
5. **Same question, different languages → different rankings.** Cross-lingual RBO ceiling is only
   **{s5.get('ceiling_rbo', float('nan')):.2f}**, and r(home-advantage, RBO) =
   **{s5.get('pearson_homeadv_rbo', float('nan')):+.2f}** — bias drives inconsistency. *(Round 5)*
6. **Language collapse is the mechanism.** Low-resource query languages over-fetch their own language
   up to **{s6.get('most_collapsed_language',{}).get('overrep', float('nan')):.0f}×** the corpus base
   rate; same-language noise out-ranks the gold on
   **{s7.get('pooled_same_language_confusion_rate', float('nan')):.0%}** of queries; and across models
   r(over-representation, CLIR@10) = {s6.get('pearson_overrep_clir', float('nan')):+.2f}. *(Rounds 6-7)*
7. **It's a separability problem, and complementarity is the lever.** r(cross-language AUC, CLIR@10) =
   **{s8.get('pearson_auccross_clir', float('nan')):+.2f}**: foreign golds are under-scored, not just
   mis-ranked. No single model finds every twin — the oracle reaches CLIR@10 =
   **{s9.get('oracle_clir10', float('nan')):.2f}** (headroom +{s9.get('oracle_headroom', float('nan')):.2f}),
   though plain RRF fusion does **not** beat the dominant model. *(Rounds 8-9)*

## Verdict — CLIR-MRS leaderboard

CLIR-MRS = capability {{accuracy, CLIR, separability}} × (0.5 + 0.5·robustness {{consistency,
MT-robust, language-parity}}). Capability is the spine; robustness modulates ±50%.

| rank | model | R@10 | CLIR@10 | home adv | mate-MRR | sep-AUC | CLIR-MRS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{tbl}

**Most robust multilingual retriever: `{s10.get('winner','?')}`** (CLIR-MRS = {s10.get('winner_mrs', float('nan')):.2f}
[{s10.get('winner_mrs_ci',[float('nan'),float('nan')])[0]:.2f}, {s10.get('winner_mrs_ci',[float('nan'),float('nan')])[1]:.2f}]).

## What to do about it

- **Deploy `{s10.get('winner','?')}`** as the single model; report **CLIR@10 and language-parity next
  to recall** — average recall hides the cross-lingual collapse.
- **Don't reflexively ensemble.** Untuned RRF underperformed the best model here; the oracle headroom
  is real but needs a *score-aware* / learned combiner, or **per-language routing** for the homeless
  / low-resource languages (es, zh).
- **The fix is alignment, not re-ranking.** Foreign twins are under-scored at the embedding level;
  a monolingual re-ranker cannot recover them.
- **Machine-translating the questions is safe** — invest human-translation effort in the source
  patents, not the generated Q/A.

## Figures (key_findings/figures/)

1. `fig01_clir_leaderboard.png` — overall vs CLIR vs MoLIR recall per model.
2. `fig02_home_advantage.png` — same-language home advantage per model.
3. `fig03_directional_clir_matrix.png` — query→document language recall matrix (best model).
4. `fig04_clir_direction_asymmetry.png` — translation-direction asymmetry.
5. `fig05_mt_penalty.png` — paired human vs machine-translated queries.
6. `fig06_mate_retrieval.png` — mate-hit@10/100 and mate-MRR.
7. `fig07_first_foreign_rank.png` — depth to the first foreign twin.
8. `fig08_consistency_vs_bias.png` — RBO consistency vs home-advantage.
9. `fig09_language_collapse.png` — same-language over-representation by query language.
10. `fig10_distractor_language.png` — language of the documents that bury the gold.
11. `fig11_separability.png` — AUC(gold>non-gold), same vs cross language.
12. `fig12_ensemble_headroom.png` — best-single vs fusion vs router vs oracle.
13. `fig13_clir_mrs_leaderboard.png` — CLIR-MRS leaderboard.
14. `fig14_robustness_radar.png` — robustness profiles (radar).

*Reproduce:* `python reports/runs/chem_patents/experimental_codes/run_all.py`
"""
    (KEY / "EXECUTIVE_SUMMARY.md").write_text(summary_md, encoding="utf-8")
    print(f"[build_key_findings] wrote {KEY}")


if __name__ == "__main__":
    main()
