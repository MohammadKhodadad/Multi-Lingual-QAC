# Reporter handoff (round 1) -> feeds story + writer (round 2)

## TL;DR
Round 1 added 8 CPU-only analyses (0 API calls) across two run families (chem_patents, alias_graph), all written to NEW `extra_*/` dirs with no existing round/key_findings touched; I opened every output file and **every claimed number matches its source CSV/JSON exactly**. The results are real but partly *adversarial to the dreamer's hopes*: aggregation-invariance FAILS (egemma rank range [1,4]), the home-advantage<->availability slope is NEGATIVE (-0.57, so home advantage is residual encoder bias, not an availability artifact), and 2 of 3 "load-bearing" correlations are FRAGILE (only auc_cross~clir survives). The new positive contributions are the XRC reading-cost multiplier (egemma XRC50 = 3.5x) and the RRC re-ranker ceiling (egemma recovers <=74% at top-100, 5.84% lost forever).

## Verified new results
Format: value -> source path -> figure basename -> paper section/claim it affects. All paths relative to repo root; all VERIFIED against the file.

1. **XRC50 (median cross-lingual reading-cost multiplier): embeddinggemma = 3.50** (D50_same 2 docs, D50_cross 7 docs). granite 1.25 (lowest), nomic 11.5, e5-large-instruct 97.75 (catastrophic), gte-base degenerate (all inf).
   - src: `reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/xrc_per_model.csv` + `summary.json`
   - fig: `cp_fig15_xrc_reading_cost.png` (log-y XRC vs CLIR)
   - affects: new "cost of cross-linguality" result; D90/D95 are RIGHT-CENSORED (6-16% first-foreign ranks = inf), so the headline MUST be the finite median XRC50, not D90/D95 (those are lower bounds). Populations: 57 same-lang-gold queries, 137 cross-lang queries.

2. **RRC re-ranker recoverability ceiling: egemma RRC@100 = 0.7445, RRC@1000 = 0.9416, lost@1000 = 0.0584.** Worst non-degenerate: e5 loses 37.2% (RRC@1000 0.6277); gte-base degenerate (lost 91.2%).
   - src: `reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/rrc_per_model.csv` + `summary.json`
   - fig: `cp_fig16_rrc_reranker_ceiling.png`
   - affects: "a top-100 re-ranker reaches at most 74% of foreign twins; 5.84% are unrecoverable" -> supports the argument that retrieval (not re-ranking) is the bottleneck. Consistency gate: 8-model pooled lost@1000 = 0.1542 (exact match to round04; all-9 = 0.2384 reported separately).

3. **Aggregation-invariance FAILS (CONTRADICTS DREAMER).** embeddinggemma is rank-1 under only **2 of 4** schemes (CLIR-MRS=1, winner-take-all=1, Borda=3, equal-weight=4); **rank range [1,4]**.
   - src: `reports/runs/chem_patents/experimental_plots/extra_aggregation_invariance/summary.json` + `aggregation_ranks.csv`
   - fig: `cp_fig17_aggregation_ribbon.png`
   - affects: M6 framing. The DEFENSIBLE claim is **per-axis capability dominance** (egemma leads accuracy/clir/separability individually -- Table 1), NOT composite invariance. Present the ribbon as a RANGE. Caveat: winner-take-all is contaminated (gte-base "wins" mt_robust & lang_parity by retrieving almost nothing).

4. **B1 fix -- directional hub scores (incl. diagonal, 8-model pool):** en 0.367, fr 0.375, zh 0.350, de 0.309. So **fr (0.375) > en (0.367) -> "English is the easiest target" is FALSE.** Hardest directed edge en->de R@10 = 0.125; most asymmetric pair de<->zh gap +0.234.
   - src: `reports/runs/chem_patents/experimental_plots/extra_directional_hub/hub_scores.csv` + `summary.json`
   - fig: none (numbers + replacement sentence in summary.json `writer_replacement_sentence`)
   - affects: must REPLACE the round02 claim (L217-218). Copy-ready replacement sentence is in summary.json. Corpus-composition caveat annotated (en 46% / zh 0.4% of docs), so asymmetry partly tracks corpus size, not only encoder behaviour.

5. **Availability-adjusted home advantage: slope NEGATIVE -0.572 (CONTRADICTS DREAMER/troubleshoot framing).** Pearson -0.87, R^2=0.76, n=5 languages. zh (8% availability) has the LARGEST home advantage (+0.475). Mean home advantage 0.324.
   - src: `reports/runs/alias_graph/experimental_plots/extra_availability_residual/availability_regression.csv` + `summary.json`
   - fig: `ag_fig11_availability_residual.png`
   - affects: the +0.32 home advantage is **residual encoder bias, NOT an availability artifact** -- the "availability explains it away" branch does NOT fire; this STRENGTHENS the encoder-bias narrative. Label DESCRIPTIVE (n=5), not inferential.

6. **Drop-the-collapsers correlation robustness: only auc_cross~clir is ROBUST (CONTRADICTS DREAMER).**
   - auc_cross~clir: +0.961 (n8) -> +0.958 (n7), Spearman 0.976 -> 0.964 -- **survives.**
   - mean_overrep~clir: -0.600 (n8) -> **+0.419 (n7)** -- **FLIPS SIGN**; note Spearman was already only -0.048 on n8.
   - home_adv~rbo: -0.846 (n8) -> **+0.186 (n7)**, and -0.187 if gte-base re-included -- **most fragile.**
   - src: `reports/runs/chem_patents/experimental_plots/extra_correlation_robustness/correlation_robustness.csv` + `summary.json`
   - fig: none
   - affects: the writer must HEDGE the over-rep~clir and home_adv~rbo correlations; lean on **auc_cross~clir as the single robust mechanism.**

7. **A6 joint failure mode (n=257 confused concept-lens cases):** modal failure is a **same-language sibling = 114/257 = 44.4%**. Siblings 79.4% of confusions; same-language winners 55.6%.
   **A8 universal-blind core: 16/132 = 12%; 14/16 are STRUCTURE questions** (2 role). By language: fr 5, zh 4, de 3, es 3, en 1.
   - src: `reports/runs/alias_graph/experimental_plots/extra_joint_failure/summary.json`, `joint_failure_modes.csv`, `universal_blind_profile.csv`, `universal_blind_ids.csv`
   - fig: `ag_fig12_joint_failure_modes.png`
   - affects: confusion-taxonomy section ("both traps at once" is modal) and the universal-blind profile ("structure questions are the universal trap"). The 16 count matches round08; 12% earns a clean profile.

8. **Two-level confusion severity: siblings do the damage -- pooled sibling win-rate 18.1% vs parent 6.2% (2.9x ratio).** embeddinggemma 6.1% vs 1.5% (4x).
   - src: `reports/runs/alias_graph/experimental_plots/extra_confusion_severity/severity_split.csv` + `summary.json`
   - fig: none
   - affects: confusion-severity claim. SCOPE NOTE (honest): this is a TWO-LEVEL split only (on-disk `relation` field is binary); the graded ChEBI hop-distance decay law is BACKLOG (CCI-hop-distance-law).

## Discrepancies / unverifiable claims
**No fabrication found. Every headline number matches its source file.** Minor items for the writer's awareness:
- **(minor, internal tension -- not a fabrication)** In `extra_correlation_robustness/summary.json`, the `pairs_weakened_on_n7` list names only `home_adv ~ rbo`, and `key_observation` calls over-rep~clir "more robust." But the CSV shows over-rep~clir Pearson FLIPS -0.600 -> +0.419 on n7 (Spearman -0.048 -> +0.429). The implement_report's prose (DO-NOW-6 headline) correctly classifies BOTH over-rep~clir and home_adv~rbo as fragile -- trust the prose/CSV, not the JSON `key_observation` phrasing. Net: only auc_cross~clir is robust; the writer should soften BOTH others.
- **(expected, handled honestly)** XRC D90/D95 are right-censored for most models (XRC90 NaN for 6/9, XRC95 censored for all 9). The implementer correctly moved the headline to the finite XRC50. Writer must NOT quote D90/D95 as point estimates.
- **(resolved, not a bug)** The pooled lost@1000 "discrepancy" (0.2384 vs round04's 0.1542) is just an 8-vs-9-model pool difference; the 8-model pool reproduces 0.1542 exactly. No action needed.

## Changed files this round (git diff --stat summary)
- **Tracked diff:** only `.gitignore` (+2/-1). Nothing else tracked was modified -- no existing round/key_findings/FINDINGS file touched (matches the report's claim).
- **New untracked code (7):** chem_patents `extra_{xrc_reading_cost,aggregation_invariance,correlation_robustness,directional_hub}.py` + alias_graph `extra_{availability_residual,joint_failure,confusion_severity}.py`.
- **New untracked output dirs (6):** chem_patents `extra_{xrc_reading_cost,aggregation_invariance,correlation_robustness,directional_hub}/`; alias_graph `extra_{availability_residual,joint_failure,confusion_severity}/` -- all CSVs/JSONs/PNGs present and verified.
- **New figures in `paper/figures/` (5 new this round):** `cp_fig15_xrc_reading_cost.png`, `cp_fig16_rrc_reranker_ceiling.png`, `cp_fig17_aggregation_ribbon.png`, `ag_fig11_availability_residual.png`, `ag_fig12_joint_failure_modes.png` -- all valid PNGs (verified via `file`). (The full `paper/` tree is untracked -- first commit of the paper scaffold.)

## Backlogged (forthcoming) experiments to mention as pending
From `paper/loop/needs_eval.md` (all tagged r1; critics must treat as DONE going forward):
- **W4-formula-injection** -- causal formula-token intervention; needs new query embeddings. Upgrades the formula-token observation into a causal rescue claim.
- **CLIRMRS-external-validation** -- human/RAG external utility signal to validate CLIR-MRS vs mean-recall. The one thing that converts CLIR-MRS from a demoted convenience into a validated contribution.
- **XRC-conformal-M2** -- split-conformal XRC with finite-sample coverage guarantee; deferred (only 57 same-lang-gold queries, too thin). Empirical XRC ships now; this is the guarantee upgrade.
- **CCI-hop-distance-law** -- ChEBI graph build + traversal for graded confusion-vs-hop-distance law; CPU but graph-construction risk, binary on-disk relation field can't yield the graded law.
- **equivalence-audit-spotcheck** -- expert annotation that parallel human-translated golds are claim-level equivalent; pre-empts the hostile "how do you know your golds are equivalent?" review.

## Recommended next-round focus (for the story architect)
1. **Rebuild the deployment-recommendation narrative around per-axis dominance, NOT composite invariance.** The headline "egemma is best under any aggregation" is dead (rank range [1,4]). Reframe M6 as "rankings are aggregation-sensitive; egemma's claim rests on leading the three CAPABILITY axes individually," and present the aggregation ribbon as a range. This is the single biggest structural change.
2. **Promote XRC50 + RRC as the new positive contributions** (reading-cost multiplier and re-ranker ceiling) -- they are clean, finite, and verified; they give the paper a quantitative "what cross-linguality costs" result. Pair them with the censoring discipline (median headline, D90/D95 as lower bounds).
3. **Tighten the mechanism story to the one robust correlation (auc_cross~clir) and the strengthened encoder-bias claim** (negative availability slope -> home advantage is residual bias). Soften/hedge over-rep~clir and home_adv~rbo throughout; fix the B1 "English easiest target" sentence using the supplied replacement.
