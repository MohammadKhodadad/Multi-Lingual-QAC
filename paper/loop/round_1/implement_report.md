# Implementation report (round 1)

All DO-NOW items (1–8) executed. CPU-only, **0 API calls**. No `--evaluate-mteb`, no `run_all.py`,
no `build_key_findings.py`. Every new output written to NEW `experimental_plots/extra_*/` dirs; no
existing round / key_findings / FINDINGS file modified (verified via `git status`). Venv:
`.venv/bin/python` (Python 3.13.7, pandas/scipy/sklearn/matplotlib present).

## Done (what / command / output paths / verification / api)

### DO-NOW-1 + DO-NOW-2 — XRC reading-cost multiplier + RRC re-ranker ceiling (one script)
- **Command:** `.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_xrc_reading_cost.py`
- **Outputs:** `reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/`
  `xrc_per_model.csv`, `xrc_per_language.csv`, `rrc_per_model.csv`, `summary.json`,
  `xrc_vs_clir.png` (log-y), `rrc_ceiling.png`.
- **Verification (both gates PASS):**
  - Pooled lost@1000 over the **8-model reliable pool = 0.1542** — exact match to round04 (the
    0.1542 in round04 is over the POOL, gte-base excluded; all-9 pooled = 0.2384, reported for
    transparency). This resolved an apparent discrepancy: round04's headline pools 8, not 9.
  - egemma **RRC@1000 = 0.9416**, lost@1000 = **0.0584** — exact match to round04's `lost_share`.
    Every per-model lost@1000 matches round04 (bge-m3 0.1095, qwen 0.073, nomic 0.0876, …).
- **Headline numbers (REAL):**
  - **XRC50 (median reading depth, fully finite — the robust headline): embeddinggemma = 3.50**
    (median 7 docs cross vs 2 docs same). granite lowest at 1.25; nomic 11.5; e5-large-instruct
    a catastrophic **97.75**. The 90th/95th-percentile XRC are RIGHT-CENSORED for most models
    (first-foreign rank is INF >1000 for 6–16% of queries), so D90/D95 are reported but flagged as
    lower bounds — I deliberately moved the headline to the median, which carries no censoring. This
    is the small-n tail discipline the troubleshoot asked for.
  - **RRC (re-ranker recoverability ceiling): embeddinggemma RRC@100 = 0.7445, RRC@1000 = 0.9416**
    → a top-100 re-ranker can reach at most 74% of foreign twins; **5.84% are lost forever** (never
    in top-1000, unrecoverable by any re-ranker over the retrieved list). Worst non-degenerate:
    e5 loses 37%; gte-base 91%.
- **api: 0.**

### DO-NOW-3 — Aggregation-invariance ribbon
- **Command:** `.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_aggregation_invariance.py`
- **Outputs:** `…/extra_aggregation_invariance/` `aggregation_ranks.csv`, `summary.json`,
  `aggregation_ribbon.png`.
- **Verification:** scheme-1 (CLIR-MRS reproduced from the 6 normalized axes) reproduces round10
  ranks **exactly** (egemma=1, bge-m3=2, granite=3, …) — confirms the normalization read.
- **HEADLINE (HONEST, contradicts the dreamer's hope):** **the recommendation is
  AGGREGATION-SENSITIVE, not invariant.** embeddinggemma is rank-1 under only **2 of 4** schemes
  (CLIR-MRS, axes-won); its **rank range is [1,4]** — it falls to rank 3 under Borda and rank 4
  under equal-weight, because it leads all 3 capability axes (accuracy/clir/separability) but is
  weak on the robustness axes (mt_robust 0.195, lang_parity 0.295). Caveat flagged: axes-won is
  contaminated by gte-base "winning" mt_robust & lang_parity purely by retrieving almost nothing.
  **For the writer:** the defensible claim is *per-axis capability dominance* (egemma leads
  accuracy, clir, separability individually — Table 1), NOT composite invariance. This actually
  *supports* the M6 "rankings are aggregation-sensitive / no composite is load-bearing" framing,
  but the ribbon must be presented as a range, not as "rank-1 everywhere."
- **api: 0.**

### DO-NOW-4 — B1 fix: directional hub-and-spoke numbers
- **Command:** `.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_directional_hub.py`
- **Outputs:** `…/extra_directional_hub/` `hub_scores.csv`, `summary.json`.
- **Verification (GATE PASS):** reproduces the critic's recomputation **exactly** — hub scores
  (incl. diagonal, 8-model pool, matrix-cell mean = round02 `mp`): **en 0.367, fr 0.375, zh 0.350,
  de 0.309**, so **fr (0.375) > en (0.367)** → "English is the easiest target" is FALSE.
- **Replacement numbers for the writer:** hardest directed edge **en→de = 0.125**; most asymmetric
  pair **de↔zh, gap +0.234**; corpus-composition caveat (en 46% / zh 0.4% of docs) annotated.
  Writer replacement sentence is in `summary.json["writer_replacement_sentence"]`.
- **api: 0.**

### DO-NOW-5 — Availability-adjusted home advantage (alias side)
- **Command:** `.venv/bin/python reports/runs/alias_graph/experimental_codes/extra_availability_residual.py`
- **Outputs:** `…/alias_graph/…/extra_availability_residual/` `availability_regression.csv`,
  `summary.json`, `availability_residual.png`.
- **HEADLINE (HONEST, OPPOSITE of the dreamer's framing):** across the n=5 language points the OLS
  slope is **NEGATIVE (−0.572, Pearson −0.87, R²=0.76)** — home advantage does **not** shrink with
  same-language availability; if anything it is *largest* for the lowest-availability languages
  (zh: 8% availability, +0.475 home advantage). **Mean home advantage 0.324 is residual encoder
  bias, NOT an availability artifact.** This refutes the "availability explains it away" hypothesis
  and strengthens the encoder-bias narrative. Labeled DESCRIPTIVE (n=5), not an inferential test.
- **api: 0.**

### DO-NOW-6 — Drop-the-collapsers correlation robustness
- **Command:** `.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_correlation_robustness.py`
- **Outputs:** `…/extra_correlation_robustness/` `correlation_robustness.csv`, `summary.json`.
- **Verification (3/3 gates PASS):** the published correlations are over the **8-model pool**
  (gte-base excluded), NOT all 9 — I confirmed and reproduced: auc_cross~clir **+0.961**,
  mean_overrep~clir **−0.600**, home_adv~rbo **−0.846** (all match round05/06/08 to 3dp).
- **HEADLINE (HONEST robustness flag):** only **(auc_cross~clir) is ROBUST** — 0.961 (n8) → 0.958
  (n7), Spearman 0.976 → 0.964. The other two are **FRAGILE**: dropping the 2nd collapser
  (e5-large-instruct) flips them to **+0.419** (over-rep~clir) and **+0.186** (home_adv~rbo);
  home_adv~rbo also collapses to −0.187 if gte-base is re-included. The writer must soften the
  over-rep and home_adv/rbo correlations (lean on auc/clir as the one robust mechanism).
- **api: 0.**

### DO-NOW-7 — Joint failure mode (A6) + universal-blind profile (A8)
- **Command:** `.venv/bin/python reports/runs/alias_graph/experimental_codes/extra_joint_failure.py`
- **Outputs:** `…/alias_graph/…/extra_joint_failure/` `joint_failure_modes.csv`,
  `universal_blind_profile.csv`, `universal_blind_ids.csv`, `summary.json`, `joint_failure_modes.png`.
- **A6 (REAL, n=257 confused concept-lens cases):** the **modal failure is a same-language sibling:
  114/257 = 44.4%** ("both traps at once"). Siblings = 79.4% of all confusions; same-language
  winners = 55.6%. Winner language recovered by re-deriving the top-ranked hard-negative corpus-id.
- **A8 (REAL):** **16/132 = 12%** universal-blind core (re-derived from round03 metrics, matches
  round08's count of 16). **14/16 are STRUCTURE questions** (2 role); by language fr 5, zh 4, de 3,
  es 3, en 1. Earns the 12% number with a clean profile (structure questions are the universal trap).
- **api: 0.**

### DO-NOW-8 — Two-level confusion severity (sibling vs parent)
- **Command:** `.venv/bin/python reports/runs/alias_graph/experimental_codes/extra_confusion_severity.py`
- **Outputs:** `…/alias_graph/…/extra_confusion_severity/` `severity_split.csv`, `summary.json`.
- **HEADLINE (REAL):** **siblings do the damage** — pooled, a sibling outranks the gold **18.1%**
  vs **6.2%** for a parent (**2.9× ratio**). embeddinggemma 6.1% vs 1.5% (4×). Honest scope note in
  the summary: this is a TWO-LEVEL split only; the graded ChEBI hop-distance law is BACKLOG-EVAL
  (the on-disk `relation` field is binary).
- **api: 0.**

## Backlogged to needs_eval.md (id + reason)
Appended verbatim under the marker in `paper/loop/needs_eval.md` (all tagged `r1`):
- `W4-formula-injection` — needs new query embeddings (causal formula-token intervention).
- `CLIRMRS-external-validation` — needs human/RAG external utility signal.
- `XRC-conformal-M2` — split-conformal XRC; too thin (57 same-lang queries) for a credible guarantee.
- `CCI-hop-distance-law` — ChEBI graph build + traversal; CPU but edge-case-prone, binary relation
  field on disk cannot yield the graded law.
- `equivalence-audit-spotcheck` — needs expert annotation of parallel-gold equivalence.

## New figures copied to paper/figures/ (basename -> source)
- `cp_fig15_xrc_reading_cost.png` -> `reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/xrc_vs_clir.png`
- `cp_fig16_rrc_reranker_ceiling.png` -> `reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/rrc_ceiling.png`
- `cp_fig17_aggregation_ribbon.png` -> `reports/runs/chem_patents/experimental_plots/extra_aggregation_invariance/aggregation_ribbon.png`
- `ag_fig11_availability_residual.png` -> `reports/runs/alias_graph/experimental_plots/extra_availability_residual/availability_residual.png`
- `ag_fig12_joint_failure_modes.png` -> `reports/runs/alias_graph/experimental_plots/extra_joint_failure/joint_failure_modes.png`

## Failures / surprises (verbatim real outcomes)
1. **Pooled lost@1000 mismatch (resolved, not a bug).** First run printed
   `pooled lost@1000 = 0.2384 (round04 reported 0.1542)`. Cause: round04's
   `pooled_lost_share_top1000` is computed over the 8-model **POOL** (gte-base excluded, see
   round04 L28/L124), not all 9. Fixed the consistency check to compare the 8-model pool (= 0.1542,
   exact) and report all-9 separately. egemma RRC@1000 = 0.9416 matched on the first try.
2. **D90/D95 XRC are right-censored at this sample size (expected, handled honestly).** With 6–16%
   of first-foreign ranks = INF (>1000), the 90th/95th-percentile reading depth lands in the
   unfound tail for most models, so XRC90/XRC95 are lower bounds / NaN. I moved the headline to the
   fully-finite **median (XRC50)** and report the censored fraction per model; the figure uses XRC50.
3. **Aggregation-invariance does NOT hold (honest, contradicts the dreamer).** embeddinggemma is
   rank-1 under only 2/4 schemes; rank range [1,4]. Reported truthfully — the recommendation is
   aggregation-sensitive; the defensible claim is per-axis capability dominance. (This still serves
   M6, but as a *range*, not as invariance.)
4. **Availability regression slope is NEGATIVE, not positive (honest, contradicts the framing in the
   troubleshoot).** Home advantage rises as availability falls (slope −0.57, R²=0.76). So the +0.32
   home advantage is residual encoder bias, NOT an availability artifact. The troubleshoot's
   "if residual ≈ 0 → availability artifact" branch does not fire; the residual is large.
5. **Two of three load-bearing correlations are FRAGILE.** auc_cross~clir survives (robust); but
   mean_overrep~clir and home_adv~rbo flip sign / collapse when the 2nd collapser is dropped or
   gte-base re-included. The writer must hedge these two.

## API calls used this round: 0/20
