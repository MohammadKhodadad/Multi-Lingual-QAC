# Correctness review (round 4)

Reviewer #2 (correctness). Re-audited every numeric claim and figure in the **round-4 revision** of
`paper/main.tex` against the canonical data under `reports/runs/{chem_patents,alias_graph}/`, with
special attention to the **two NEW round-4 analyses** that did not exist in round-3:

- chem: `extra_ari_decomposition/` (Fig.22, §ssec:cp + §Metrics + §Analysis + §Deployment + Limitations)
- chem: `extra_per_route_frontier/` (Fig.23, §Deployment per-route paragraph + Limitations)

I recomputed the ARI identity and formula from `ari_decomposition.csv`, cross-checked the per-route
corners/flips against `per_route_frontier.csv` + `frontier_membership_by_route.csv` +
`decision_flip_by_route.csv` + `summary.json`, re-verified the RRC source the ARI is a pure transform
of, re-ran the full 81-cell leaderboard check, confirmed the two round-3 closing fixes landed, and
checked the two new cascade citations are real papers.

## Headline verdict

**Every round-4 number traces exactly. Both round-3 closing fixes landed. The two new cascade
citations are real papers. Zero MISMATCH, zero UNTRACEABLE.**

- **ARI formula is correct.** `ARI@K = L_inf / (1 - RRC@K)` recomputes to the on-disk `ARI_at_100`
  for all 9 models from the on-disk RRC values (max abs error < 1e-3; the only deviations are
  4th-decimal rounding of the displayed CSV).
- **ARI@100 egemma = 0.2286 → paper 0.229**, the lowest non-degenerate ARI@100; **next qwen3-0.6B =
  0.2326 → paper 0.233**. Both claims hold (egemma is the min, qwen3 the 2nd-min, over the 7
  non-deg).
- **L_inf egemma = 0.0584 → paper 0.058** is the smallest floor of any non-degenerate model
  (next-smallest non-deg is qwen3 0.073). Holds.
- **Identity cheap+deep+floor sums to 1.0 for all 9 models** (`identity_closes_all_models: true`;
  CSV sums = 1.0, with SapBERT/e5 showing 1.0001 purely from 4-decimal display rounding — the JSON
  records the exact identity as closed).
- **Per-route: 3 distinct max-CLIR corners** (en→qwen3-0.6B, de/es/zh→embeddinggemma,
  fr→nomic-v2-moe), **decision flips on 2/5 routes** (en, fr), **egemma wins 3/5 routes** (de, es,
  zh = the routes where it is the max-CLIR corner), **same-lang gold n: de=7, zh=2, es=0**, and
  **es XRC is undefined/NaN** (no same-lang gold, never imputed). All trace.
- **Round-3 C-NEW RESOLVED:** Fig.21 caption no longer says "sibling-confusion rate"; it now reads
  "(confusion rate, alias-graph benchmark)." No "sibling-" mislabel survives anywhere.
- **"eight non-degenerate" relabel RESOLVED:** there is no remaining "eight non-degenerate" string.
  The two surviving "eight models" tokens (L585, L714) correctly describe the *eight models with a
  defined cross-lingual recall / RRC curve* and explicitly say "all but the degenerate
  \texttt{gte-base}" — i.e., eight = nine minus the one off-plane model, NOT a claim that eight are
  non-degenerate. "seven non-degenerate" is used correctly everywhere it appears (L489, L631, L746,
  L759).
- **New cascade citations are real:** `nogueira2019multistage` = "Multi-Stage Document Ranking with
  BERT" (Nogueira/Yang/Cho/Lin, arXiv:1910.14424, 2019); `gao2021rethink` = "Rethink Training of
  BERT Rerankers in Multi-Stage Retrieval Pipeline" (Gao/Dai/Callan, ECIR 2021, LNCS 12657,
  pp.280–286, DOI 10.1007/978-3-030-72240-1_26). Both genuine, both correctly used to support the
  cascade/knee result (recall capped by first-stage depth) while the paper claims only the
  cross-lingual quantification + L_inf floor as its own.
- **81 leaderboard cells (45 CP + 36 alias) still match** their `headline_numbers.csv` sources.
- **All 27 referenced figures exist** (up from 25; the two new are `cp_fig22_ari_decomposition.png`
  and `cp_fig23_per_route_frontier.png`).
- **No active `\todo{}` red flags;** the previously-fenced un-traced numbers (corpus dedup 14,401;
  human-eval 8.33/10, 97/100, +4.3pp) remain confined to `%` comment lines, never in rendered prose.

**Counts: MISMATCH = 0. UNTRACEABLE (not already comment-fenced) = 0. NEW non-blocking nuance = 1
(D-NEW, ARI caption "all nine models" describes the identity, but the figure plots only 7 bars).**

---

## Blocking issues (MISMATCH / UNTRACEABLE) — none

No MISMATCH and no UNTRACEABLE numbers in the round-4 draft. The two round-3 closing fixes (C-NEW
sibling-mislabel; "eight non-degenerate" relabel) both landed. All prior blockers (round-2 N1/N2,
round-3 none) remain fixed.

## Non-blocking nuance — optional clarity fix

### D-NEW (caption wording, value correct). Fig.22 "sum to 1.0 for all nine models" vs. 7 bars plotted.
- **Where:** Fig.22 caption (L703): "The three sum to $1.0$ for all nine models."
- **What the data says:** `extra_ari_decomposition/summary.json` records `identity_closes_all_models:
  true` and the per-row `identity_sum_100 = 1.0` for all 9 (verified by recompute) — so the
  *statement is true*. BUT the same JSON also records `degenerate_models_excluded_from_figure:
  [e5-large-instruct, gte-base]`, i.e. the **plotted figure shows only 7 bars**. A reader could
  parse "for all nine models" as "nine bars are drawn," which the figure does not show.
- **Severity:** Low. The identity claim is mathematically true for all nine; only the juxtaposition
  with a 7-bar plot is potentially confusing. This is a clarity/cohesion item, not a number error.
- **Minimal fix (optional):** "The three sum to $1.0$ for every model (the identity closes for all
  nine; the figure shows the seven non-degenerate)." Do NOT change any value.

---

## Number-trace table (round-4 NEW numbers + conductor checkpoints + re-spot-checks)

Tolerance ≤ 0.006 on [0,1] metrics; depths/counts exact. "source" = named file under
`reports/runs/chem_patents/experimental_plots/extra_*/` unless noted.

| claim (paper) | paper value | source file | source value | status |
|---|---|---|---|---|
| **ARI formula ARI@K = L_inf/(1−RRC@K)** | (formula) | `ari_decomposition.csv` recompute, all 9 | recomputes to file ARI@100, err<1e-3 | MATCH |
| **ARI@100 egemma** | 0.229 | `ari_decomposition.csv` / `summary.json` | 0.2286 | MATCH |
| **ARI@100 egemma = lowest non-deg** | lowest | `ARI_at_100_by_model` (7 non-deg) | egemma 0.2286 = min | MATCH |
| **next ARI@100 = qwen3-0.6B** | 0.233 | `ARI_at_100_by_model` | qwen3 0.2326 (2nd-min) | MATCH |
| **L_inf egemma smallest non-deg** | 0.058 | `L_inf_by_model` | egemma 0.0584 = min non-deg | MATCH |
| egemma RRC@100 / deep / floor (ARI parts) | 0.7445 / 0.1971 / 0.0584 | `summary.json` egemma | 0.7445 / 0.1971 / 0.0584 | MATCH |
| **identity sums to 1.0 all 9** | 1.0 (all nine) | `summary.json` `identity_closes_all_models` + per-row CSV | true; sums=1.0 (display-rounding only) | MATCH |
| egemma ARI@Kstar | 0.1127 | `ari_decomposition.csv` | 0.1127 | MATCH |
| ARI source = pure transform of RRC | RRC@100 0.7445 / RRC@1000 0.9416 / K*5 | `extra_rrc_budget_frontier/rrc_knee.csv` egemma | 0.7445 / 0.9416 / 5 | MATCH |
| **per-route 3 distinct max-CLIR corners** | 3 | `per_route summary.json` `n_distinct_max_clir_corners` | 3 | MATCH |
| en corner | qwen3-0.6B | `max_clir_corner_by_route.en` | qwen3-0.6B | MATCH |
| de/es/zh corner | embeddinggemma | `max_clir_corner_by_route` de/es/zh | embeddinggemma (×3) | MATCH |
| fr corner | nomic-v2-moe | `max_clir_corner_by_route.fr` | nomic-v2-moe | MATCH |
| **decision flips 2/5 (en, fr)** | 2/5 | `decision_flip_by_route.csv` / `n_routes_flipped` | en=T, fr=T, others F; n=2 | MATCH |
| **egemma wins 3/5 routes (de, es, zh)** | 3/5 | `max_clir_corner_by_route` egemma routes | de, es, zh | MATCH |
| recall-only picks egemma on all 5 | all five | `decision_flip_detail.recall_pick` | egemma ×5 | MATCH |
| **same-lang gold n: de=7** | 7 | `n_same_by_route.de` | 7 | MATCH |
| **same-lang gold n: zh=2** | 2 | `n_same_by_route.zh` | 2 | MATCH |
| **same-lang gold n: es=0** | 0 | `n_same_by_route.es` | 0 | MATCH |
| cross-side n 22–34 | 22–34 | `n_cross_by_route` | min 22(zh) max 34(es) | MATCH |
| **es XRC undefined (NaN, not imputed)** | undefined | `per_route_frontier.csv` es XRC50 blank; `xrc_axis_status.es` | "undefined (n_same=0)" | MATCH |
| global pareto / egemma global corner | {bge-m3,egemma,granite}; max CLIR 0.50 | `global_pareto_set_reference` / `verify_egemma_global_clir` | same / 0.5024 | MATCH |
| **Fig.21 caption = "confusion rate" (no "sibling-")** | confusion rate | `main.tex` L759 grep | no "sibling-" anywhere | RESOLVED |
| **"eight non-degenerate" relabeled** | absent | `main.tex` grep | 0 hits; "eight models … all but degenerate gte" only | RESOLVED |
| cascade cite nogueira2019multistage real | real | `custom.bib` L368 | Multi-Stage Doc Ranking w/ BERT, arXiv:1910.14424 | MATCH |
| cascade cite gao2021rethink real | real | `custom.bib` L357 | Rethink BERT Rerankers, ECIR 2021 LNCS 12657 | MATCH |
| **CP leaderboard (45 cells)** | 45 | CP `headline_numbers.csv` | all 45 within tol | MATCH |
| **alias leaderboard (36 cells)** | 36 | alias `headline_numbers.csv` | all 36 within tol | MATCH |
| granite two-tax (carried) | 1.25× / 0.182 | `two_tax_table.csv` granite | 1.25 / 0.1818 | MATCH |
| best CLIR@10 / home +0.55 (carried) | 0.50 / +0.55 | CP `headline_numbers.csv` egemma/e5 | 0.5024 / 0.5526 | MATCH |
| RRC@100 / RRC@1000 / L_inf / K* (carried) | 0.7445 / 0.942 / 0.058 / 5 | `rrc_knee.csv` egemma | 0.7445/0.9416/0.0584/5 | MATCH |
| RBO ceilings alias / CP (carried) | 0.39 / 0.19 | alias/CP headline | 0.387 / 0.1934 | MATCH |
| separability r robust (carried) | +0.96 / +0.958 n=7 | `extra_correlation_robustness` | 0.961 / 0.958 | MATCH |
| two-tax / trap rho (carried) | −0.59 (n=7,p=.16) / +0.29 (n=7,p=.53) | `extra_two_tax_degeneracy` | −0.5946/0.1591 ; +0.2857/0.5345 | MATCH |
| DEG gate {gte,e5}; SapBERT floor 0.179 (carried) | {gte,e5}; 0.179 | `extra_two_tax_degeneracy` | [e5,gte]; SapBERT 0.1788 | MATCH |
| corpus dedup 14,401 / human-eval 8.33 | (comment-fenced) | not under reports/ | — | FENCED (OK) |

---

## Design-soundness findings (per headline claim)

1. **ARI decomposition (Fig.22, §Metrics Eq.5, §Analysis, §Deployment, Limitations).** Sound and
   the cleanest new contribution. It is an honest *re-presentation* of the RRC curve, not a new
   measurement: `cheap = RRC@K`, `deep = RRC@1000−RRC@K`, `floor = L_inf = 1−RRC@1000`, which sum to
   1 by construction, and `ARI@K = L_inf/(1−RRC@K)` is the share of the *post-re-rank residual* that
   is alignment-bound. The formula recomputes exactly, the identity closes for all 9, and the
   egemma/qwen3 ordering (0.229 < 0.233 < …) is correct. The paper does NOT over-sell it: it is
   correctly framed as "the quantitative form of 'align, don't re-rank'" and as upside structure for
   a future alignment probe, with the RRC budget being the underlying measured object. Crucially the
   Limitations (L1207–1209) explicitly states "the ARI decomposition (which closes to 1.0 for all
   nine models) … is regression-checked and stated firmly," which matches `checks_passed: true`.
   **Leave the numbers and framing alone; only D-NEW (caption "nine models" vs 7 bars) is optional.**

2. **Per-route frontier (Fig.23, §Deployment, Limitations).** Sound and — importantly — honestly
   hedged. The robust spine is the per-route `CLIR@10_ell` axis (mean per-query CLIR@10 over each
   route's cross-gold queries, n_cross 22–34), and the paper rests its per-route win claims on that
   axis, not on the thin XRC axis. The three corners, the 2/5 flip, and egemma's 3/5 wins all trace.
   The paper correctly flags the thin denominators (de n_same=7, zh n_same=2, es n_same=0), states
   the per-route XRC is "indicative only," and explicitly says es XRC is undefined and "never
   imputed." The conclusion drawn — "a per-route router is upside *headroom*, not a recommendation"
   — is the correct, conservative reading at these sample sizes and does not overturn the
   single-model recommendation. **This is exactly the honesty a hostile reviewer wants on thin
   per-language cells. Leave alone.**

3. **"egemma wins 3/5 routes incl. the two hardest cross-only routes" (L1098).** Verified: egemma is
   the max-CLIR corner on de, es, zh; es (n_same=0, the pure no-home route) and zh (the thinnest,
   n_cross=22) are the two hardest, and egemma owns both. Defensible. The framing correctly notes
   the recall-only dashboard picks egemma on all five, so the *frontier* picture (where it loses en
   to qwen3 and fr to nomic) is the more honest one. Sound.

4. **Cascade citations (L447).** The paper attributes the *knee / diminishing-returns-with-depth*
   shape to the established multi-stage cascade literature (nogueira2019multistage, gao2021rethink)
   and claims only "its cross-lingual quantification and the structural floor L_inf" as novel. This
   is the correct attribution boundary — it does not claim to invent the cascade-depth result, which
   would be a novelty over-claim. Both papers are real and on-topic. Sound.

5. **Round-3 fixes verified held.** C-NEW (Fig.21 "sibling-" mislabel) is gone; the value 0.182
   (general `confusion_rate`) is retained and correctly labeled. The "eight non-degenerate" overclaim
   is gone; "eight models with a defined RRC/CLIR" (= nine minus degenerate gte) is the correct
   surviving usage, and "seven non-degenerate" is used wherever the truly-non-degenerate set is
   meant. Sound.

6. **Leaderboards (Tables 1–2).** All 81 cells re-checked and still match after the writer's edits;
   no drift introduced by the round-4 additions. Sound.

7. **Carried-correct items (XRC population caveat, MT-null framing, concept-lens mechanical cap,
   degeneracy gate, separability +0.96 robust, non-significant correlations quarantined out of
   abstract/intro/conclusion).** All re-spot-checked and unchanged from the round-3 verified state.
   Sound.

---

## Overlooked / confounds / threats-to-validity (with the minimal fix each)

**T1 — Home-advantage vs gold-availability confound (HANDLED, carried).** §Results L562 ties +0.55 to
"much of which gold availability shapes … though a residual encoder bias remains"; §Analysis carries
the availability-residual slope (−0.57, descriptive). Resolved.

**T2 — Directional asymmetry as corpus composition (HANDLED, carried).** §Results L587 "What asymmetry
exists partly tracks corpus composition (en 46% vs zh 0.4%) … not only encoder behaviour." Resolved.

**T3 — Per-route thinness (NEW, HANDLED).** The single most important new threat. The per-route
corners rest on de n_same=7, zh n_same=2, es n_same=0 — tiny cells. The paper pre-empts a hostile
reviewer twice: in the Deployment paragraph (L1104–1110, "the per-route XRC axis is *indicative*
only … es is XRC-undefined … never imputed … we present routing as the honest upside, not a
reversal") and in a dedicated Limitations paragraph (L1211–1219). The robust signals (CLIR@10_ell,
frontier membership, *existence* of corner movement) are correctly separated from the indicative XRC
axis. Resolved; no fix needed.

**T4 — ARI is a re-presentation, not an independent measurement (NEW, HANDLED).** A reviewer could
object that ARI adds no information beyond RRC. The paper does not claim otherwise: it presents ARI as
"a natural exhaustive reading of the same shortfall" (L453) and "the quantitative form of 'align,
don't re-rank'" (L466), with RRC as the measured object. The identity-closure and regression checks
are surfaced. No over-claim. No fix needed.

**T5 — Parallel-gold equivalence audit (HONESTLY DEFERRED, carried).** Limitations L1234–1238 still
flags claim-level equivalence and a parallel-gold spot-check as future work; listed in needs_eval, so
per the critic contract this is DONE/deferred, not a missing experiment.

**T6 — Stale baseline run never cited (CLEAN, carried).** The `20260601-235117_137questions` baseline
is still never pulled into the paper; all CLIR/RRC/ARI numbers come from the 23,487-corpus key_findings
+ extra_* dirs. No leak.

**T-MINOR (non-blocking, carried) — universal-blind language ordering.** §Analysis L996 says the blind
core is "predominantly French and Chinese" — this round it has been trimmed to the two unambiguous
top languages (fr 5, zh 4), avoiding the de/es 3-tie the round-3 critic flagged. Improved; no fix.

**D-NEW (non-blocking, NEW) — Fig.22 caption "all nine models" vs 7 plotted bars** (see above). The
identity claim is true for all nine; only the wording could imply nine bars. Optional one-clause fix.

---

## Verified-correct (leave these alone)

- **All round-4 NEW numbers** — ARI@100 (egemma 0.2286→0.229 lowest non-deg, qwen3 0.2326→0.233 next),
  ARI formula `L_inf/(1−RRC@K)` (recomputes for all 9), identity sum=1.0 (all 9), L_inf egemma 0.0584
  smallest; per-route 3 corners (en→qwen3, de/es/zh→egemma, fr→nomic), 2/5 flips (en,fr), egemma 3/5
  wins (de,es,zh), same-lang n de=7/zh=2/es=0, es XRC NaN — **trace exactly. Do not touch.**
- **Round-3 C-NEW RESOLVED:** Fig.21 caption is "confusion rate," no "sibling-" anywhere; value 0.182
  retained.
- **"eight non-degenerate" relabel RESOLVED:** no such string; "eight models … all but degenerate
  gte" is the correct surviving usage; "seven non-degenerate" used correctly.
- **New cascade citations real and correctly bounded** (nogueira2019multistage, gao2021rethink);
  novelty claimed is only the cross-lingual quantification + L_inf floor.
- **Both leaderboard tables (81 cells)** match `headline_numbers.csv` exactly after the edits.
- **All 27 figures** referenced exist under `paper/figures/` (incl. cp_fig22, cp_fig23).
- **No active `\todo{}`;** comment-fenced numbers (corpus dedup 14,401; human-eval) remain fenced —
  do NOT un-fence until dumped to `reports/`.
- **Carried-correct framing** (XRC population caveat, MT-null, concept-lens cap, degeneracy gate,
  separability +0.96 robust, non-significant correlations quarantined) unchanged and correct.

---

## Bottom line for the writer

**Every round-4 number traces exactly; both round-3 closing fixes (Fig.21 "sibling-" mislabel and the
"eight non-degenerate" relabel) have landed; and the two new cascade citations are real, on-topic
papers used with the correct novelty boundary.** The ARI decomposition is mathematically sound (formula
recomputes, identity closes to 1.0 for all nine, egemma 0.229 is the lowest non-degenerate ARI@100,
qwen3 0.233 next, L_inf 0.058 smallest), and the per-route frontier is honestly hedged (3 corners, 2/5
flips, egemma 3/5 wins, thin de=7/zh=2/es=0 cells flagged, es XRC undefined and never imputed). The
81-cell leaderboards still match. The ONE new item is non-blocking and optional: Fig.22's caption says
the identity sums to 1.0 "for all nine models" (true) while the figure plots only the seven
non-degenerate bars — a one-clause clarity tweak, no value change. Everything else should NOT be
"fixed."
