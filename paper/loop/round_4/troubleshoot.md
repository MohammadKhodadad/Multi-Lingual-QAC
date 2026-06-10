# Troubleshooting plan (round 4)

**Operating mode.** All three critics converged: the spine is frozen, novelty is
defensible (2nd clean round), and the **only** live liability is **float overload**
(27 figs + 2 tables = 29 floats, 17 in §6, against an 8-page body). The dreamer
biased everything toward **trim + harden, no new objects**. This plan does the same.
Channel (b) = **FREEZE** (confirmed; the metric family is closed). The round's whole
value is: (1) the float cut/merge + four cheap text fixes, and (2) CPU-only resamples
of the load-bearing scalars collected into **one appendix table** (outside the
8-page body budget).

I verified on disk, read-only, that **the dreamer's top-3 are CPU-only from existing
data**, and I ran a sanity numeric pass in the venv (no artifacts written). Two of
the proposed hardening numbers come back *weaker than the dreamer assumed* — I flag
both prominently below so the implementer reports them honestly rather than being
surprised. Repo facts grounding the plan:

- `common.bootstrap_ci(values, stat, n_boot, alpha, seed)` exists (common.py L405) —
  **percentile** bootstrap, NOT BCa. `bootstrap_ci_rate`, `paired_perm_test`,
  `wilcoxon`, `stars` also present. There is **no BCa helper**. See DO-NOW-2 note.
- `common.core_per_query()` (L364) loads all 9 parquet (~13 MB total, 9 files) and
  produces per-(model,query) `first_cross_rank`, `clir_at_*`, `recall_at_*`. Loads in
  **seconds**, not minutes. This is the per-query population A5/A6 need.
- Tiny on-disk CSVs (no parquet) already carry the per-model inputs for A1/A3/W2:
  - `round08_separability/per_model.csv` → `auc_cross`, `clir_at_10` (A1, W2)
  - `round01_clir_leaderboard/per_model.csv` → `recall_at_10`, `clir_at_10` (W2 control)
  - `extra_xrc_reading_cost/xrc_per_model.csv` → `XRC50`, `clir_at_10`, per-query depths
    are *not* persisted per-query, only the percentile summary (A3 needs only the summary)
  - `extra_rrc_budget_frontier/rrc_curve.csv` + `rrc_knee.csv` → RRC@K curve, L_inf, K* (A5)
  - `extra_ari_decomposition/ari_decomposition.csv` → ARI@100 per model (A5 reference)
- The float-merge sources (cp_fig06/07, cp_fig09/10) are **key_findings** figures, not
  outputs of any re-runnable `extra_*` script (figures_manifest.md L17–18, L29–30). The
  per-panel PNGs already exist in `paper/figures/`. So a 2-panel merge has two clean
  routes: LaTeX `subfigure` (no new asset, WRITER) or a trivial PNG-stitch script that
  loads the two existing PNGs side-by-side (DO-NOW, optional). Neither requires
  re-plotting from data. **Do NOT re-plot the panels from the round04/round06 per-query
  CSVs — that risks the merged panel diverging from the verified key_findings figure.**

API budget for the whole round: **0**. Nothing here calls a model or an API.
Never `--evaluate-mteb`.

---

## DO-NOW (ordered) — each: goal / files / exact commands / inputs(exist?) / outputs / runtime / verify / api-cost

### DO-NOW-1 — The four cheap text fixes (ρ_k cite + fig22 caption + §6.1 ARI trim + bib entry)
**These are WRITER edits in essence (see WRITER-ONLY), but I list the *bib mechanics* here
because they touch a tracked file.** If the implementer is also the writer this round,
do these first; otherwise hand straight to WRITER-ONLY. Zero computation.
- **goal.** Close the round's only real residual (the ρ_k citation) + the 3 bookkeeping
  seams all three critics named.
- **files.** `paper/custom.bib` (add 1 entry), `paper/main.tex` (3 one-clause edits at
  ~L466 ARI paragraph, L703 fig22 caption, L687–690 §6.1 opener).
- **inputs (exist?).** Verbatim bib entry + exact rewrites are in dreamer C-a/C-b/C-c and
  novelty critic §"Missing citations". Yes, all on hand.
- **outputs.** Edited `.bib` + `.tex`; net **line saving** (C-c removes ~1.5 lines).
- **runtime.** ~10 min of editing. No build needed beyond the writer's normal pass.
- **verify.** `grep -n residualrerank2026 paper/custom.bib paper/main.tex` resolves;
  fig22 caption no longer reads "all nine models" unqualified; §6.1 opens at the result.
- **api-cost.** 0.

### DO-NOW-2 — Appendix robustness table: the three load-bearing scalars resampled (A1 + A6 + A5)
This is dreamer **W1 ← {A1, A6, A5}** and the single highest-leverage hardening move.
One NEW script, one appendix table, **no body float** (appendix is outside the 8-page
body budget). Reports each as a point estimate + resample interval + sign/order
stability + n.

- **goal.** Convert prose-level small-n hedging into "every load-bearing scalar survives
  resampling," as one appendix table referenced once from §7.
- **files (new).** `reports/runs/chem_patents/experimental_codes/extra_robustness_appendix.py`
  (model it on `extra_correlation_robustness.py` — same import/IO pattern, `import common as C`).
- **what it computes (three blocks):**
  1. **A1 — separability r=+0.96 (n=7) robustness.** Input
     `round08_separability/per_model.csv` (`auc_cross`, `clir_at_10`), drop the two
     degenerates (`gte-base`, `e5-large-instruct`) → **n=7**. **Model-level bootstrap**:
     resample the 7 model rows with replacement 10 000×, recompute Pearson r per draw
     (skip degenerate draws where <3 distinct points). Report **point r, percentile 95%
     CI, and sign-stability = fraction of draws with r>0** (this is the load-bearing read
     per the dreamer's caveat — lead with it). I verified r(n7)=**0.958** reproduces the
     published number; r(n9)=0.888.
  2. **A6 — XRC50 depth bootstrap for the 3 frontier members** (embeddinggemma, bge-m3,
     granite-278m). Input: per-query depth populations from `common.core_per_query()`
     (`first_cross_rank` over the 137 cross-gold queries) **and** the same-language depths
     reconstructed exactly as `extra_xrc_reading_cost.py` does it (the 57 originals with a
     same-language gold, via `first_gold_rank` on the same-language gold split — copy that
     block, L80–90 of that script). For each frontier model: resample the cross-depth
     population (n=137) and the same-depth population (n=57) **independently** with
     replacement 10 000×, recompute XRC50 = median(cross)/median(same) per draw using the
     SAME censoring/percentile rule (`pct_depth`, copy it), report point + percentile 95%
     CI. Censoring caveat: a resample whose median lands in the right-censored tail yields
     an inf-bounded ratio — count those draws and report the censored-draw fraction
     alongside the CI (do not silently drop them; report "CI lower bound" if >5% censored).
     Expected points to reproduce: egemma 3.5, bge-m3 2.0, granite 1.25.
  3. **A5 — ARI@100 egemma-vs-qwen3 0.004-gap order stability.** ARI@K = L_inf/(1−RRC@K)
     is a pure transform of RRC, and RRC@K = mean(first_cross_rank ≤ K) over the 137 cross
     queries. **Per-query paired bootstrap**: resample the 137 cross-query indices with
     replacement 10 000×; for egemma and qwen3 recompute RRC@100 and RRC@1000 on the
     resampled indices, then L_inf=1−RRC@1000 and ARI@100=L_inf/(1−RRC@100); report
     **P(ARI@100_egemma < ARI@100_qwen3)** = fraction of draws egemma's ARI is strictly
     lower, plus the bootstrap CI on the *gap* (qwen3−egemma). Use the SAME per-query
     `first_cross_rank` vectors for both models on each resampled index set (paired). Point
     reference: egemma 0.2286 < qwen3 0.2326 (gap 0.004); reproduce from
     `ari_decomposition.csv`.
- **inputs (exist?).** YES — all four CSVs + the parquet for `core_per_query()` confirmed
  on disk. The only thing NOT on disk is a per-query depth CSV, so A6/A5 must call
  `core_per_query()` (cheap, seconds).
- **BCa caveat (IMPORTANT).** The dreamer asks for **BCa**; `common.bootstrap_ci` is
  **percentile only**. Two acceptable options, in order of preference:
  (a) **Report percentile CI + sign/order-stability** (the load-bearing read is the
  stability vote, not CI width — dreamer A1 caveat says exactly this). This reuses
  `common.bootstrap_ci` directly and is sufficient. **Recommended.**
  (b) If the implementer wants BCa specifically, add a ~25-line `bca_ci()` helper to the
  new script (bias-correction z0 from the proportion of boot < point; acceleration a via
  jackknife) — standard, CPU-only, no new dependency beyond `scipy.stats.norm`. Optional;
  do NOT block on it. Either way, **sign-stability / order-probability is the headline**,
  not the interval.
- **outputs.** `reports/runs/chem_patents/experimental_plots/extra_robustness_appendix/`
  → `robustness_appendix.csv` (rows: scalar, point, lo, hi, stability_metric, n) +
  `summary.json`. NO png needed (it is a table). The writer turns the CSV into one
  appendix `\begin{table}`.
- **runtime.** ~1–3 min (3×10k bootstraps on ≤137-vectors + one parquet load).
- **verify.** Script self-checks: r(n7)≈0.958 (gate ±0.01 vs published); egemma XRC50
  point ≈3.5, bge-m3 ≈2.0, granite ≈1.25 (gate vs `xrc_per_model.csv`); ARI@100 egemma
  point ≈0.2286, qwen3 ≈0.2326 (gate vs `ari_decomposition.csv`). Add an `asserts/gates`
  block exactly like the other `extra_*` scripts (e.g. cost_frontier `verify_*` keys).
- **api-cost.** 0.

### DO-NOW-3 — τ-sensitivity of the cost frontier (A3): admitted-set stability interval
This is dreamer **A3** — an inline result (one sentence / 3-row micro-table), no float.
- **goal.** Replace the "τ=0.40 untuned constant" footnote with a stated stability
  interval, pre-empting "you tuned the threshold."
- **files.** Either extend `extra_cost_frontier.py` (add a τ-sweep block + emit
  `tau_sweep.csv`) OR a tiny standalone `extra_tau_sweep.py`. Prefer **extending
  cost_frontier** so the τ logic lives next to the existing TAU=0.40 read-off (L116–127).
- **what it computes.** Sweep τ over the grid **{0.30, 0.35, 0.40, 0.45, 0.50}** (and
  optionally 0.025 steps for the boundary). For each τ: admitted set = {short :
  clir_at_10 ≥ τ}; cheapest-admitted = argmin XRC50 over admitted (finite XRC only);
  record. Then report **(i)** the τ-range over which the admitted set = {bge-m3, qwen3,
  embeddinggemma} is stable, **(ii)** the τ-range over which cheapest-admitted = bge-m3,
  **(iii)** confirm embeddinggemma is the unique global max-CLIR corner for *all* τ
  (τ-invariant by construction).
- **inputs (exist?).** YES — `extra_xrc_reading_cost/xrc_per_model.csv` only. Tiny.
- **HONEST FINDING (I already ran this in the venv — flag to writer):** the admitted set
  and "cheapest = bge-m3" are **NOT as τ-robust as the dreamer assumed**. Verified:
  - τ=0.30 → admitted {bge-m3, egemma, granite, nomic, qwen3}, **cheapest = granite-278m**
    (XRC50 1.25 < bge-m3 2.0). The recommendation *flips* here.
  - τ=0.35 → admitted {bge-m3, egemma, nomic, qwen3}, cheapest = bge-m3.
  - τ=0.40 → admitted {bge-m3, egemma, qwen3}, cheapest = bge-m3. (paper's stated set ✓)
  - τ=0.45, 0.50 → admitted {egemma} only, cheapest = egemma.
  So the honest claim is: **"cheapest admitted reader = bge-m3 holds for τ∈[0.35, ~0.43];
  below 0.35 granite (a lower-recall but cheaper-to-read model) enters and becomes
  cheapest, and at τ≥~0.44 only embeddinggemma is admitted. embeddinggemma is the unique
  max-CLIR corner for all τ (τ-invariant)."** This is a *narrower* and more honest stability
  band than "τ=0.40 is arbitrary, the rule is robust." It still pre-empts the objection,
  but the writer must NOT overstate robustness — the rule is τ-sensitive at the low end.
  The τ-invariance of egemma's corner is the clean, unconditional part.
- **outputs.** `extra_cost_frontier/tau_sweep.csv` + 3 keys in `summary.json`
  (`tau_admitted_stable_range`, `tau_cheapest_bge_range`, `egemma_corner_tau_invariant`).
- **runtime.** seconds.
- **verify.** τ=0.40 row reproduces the existing `tau_admitted_set` and
  `tau_admitted_min_xrc_model='bge-m3'` already in `extra_cost_frontier/summary.json`.
- **api-cost.** 0.

### DO-NOW-4 (OPTIONAL, low priority) — pre-stitched 2-panel PNGs for the two merges
Only do this if the writer prefers a single `\includegraphics` over LaTeX `subfigure`.
- **goal.** Produce `cp_fig06_07_mate.png` and `cp_fig09_10_collapse.png` as 2-up panels.
- **files (new).** `reports/runs/chem_patents/experimental_codes/stitch_merged_panels.py`
  (one-shot; ~25 lines, matplotlib `imread` + 1×2 subplot of the two existing PNGs).
- **method.** **Load the two existing rendered PNGs** (`paper/figures/cp_fig06_*.png` +
  `cp_fig07_*.png`; and `cp_fig09_*.png` + `cp_fig10_*.png`) and place them side by side
  with `ax.imshow`/`imread`, `axis('off')`, save at the same DPI. **Do NOT re-plot from
  the round04/round06 per-query CSVs** — that risks divergence from the verified
  key_findings panels (correctness has signed off on the current panels).
- **inputs (exist?).** YES — the four per-panel PNGs are in `paper/figures/`.
- **outputs.** `paper/figures/cp_fig06_07_mate.png`, `paper/figures/cp_fig09_10_collapse.png`.
- **runtime.** seconds.
- **verify.** Both new PNGs open and show both panels legibly; both source panels visible.
- **api-cost.** 0.
- **ASSESSMENT (asked by the conductor).** A stitch script is **clean but not clearly
  better** than LaTeX `subfigure` here, because the panels come from key_findings (not a
  re-runnable extra_ script), so a stitch just re-photographs two PNGs — it adds a derived
  asset to maintain. **Recommendation: default to LaTeX `subfigure` (WRITER-ONLY) and keep
  this stitch script as a fallback** only if the writer finds `subfigure` layout fights
  the column width. List it as optional DO-NOW, not part of the base bundle.

### DO-NOW-5 (RUN-IT-BUT-HANDLE-WITH-CARE) — W2 separability partial-r controlling for Recall@10
Dreamer **W2**. CPU-only, inputs on disk, **but the result comes back weak and could
backfire** — so I have already computed it and give the implementer explicit guidance.
- **goal.** Test whether cross-language AUC predicts CLIR@10 *after* partialling out
  overall Recall@10 (the capability proxy), to pre-empt "good models are just good."
- **files.** Add a `partial_r` block to the new `extra_robustness_appendix.py` (DO-NOW-2)
  or a 15-line standalone. Inputs: `round08_separability/per_model.csv` (`auc_cross`,
  `clir_at_10`) + `round01_clir_leaderboard/per_model.csv` (`recall_at_10`), merged on
  `short`, drop the 2 degenerates → n=7.
- **inputs (exist?).** YES, both tiny CSVs confirmed.
- **HONEST FINDING (I already ran it):** partial r(auc_cross, CLIR@10 | Recall@10) on n=7
  = **+0.295 (p=0.52)** — i.e. once overall Recall@10 is partialled out, the separability→
  CLIR link is **weak and non-significant**. At n=7 with one control there are ~4 residual
  df; the zero-order r=0.958 is **largely collinear with overall recall**. This does NOT
  show "separability is not a capability artifact"; if anything it shows we *cannot
  statistically separate* the two at this n.
- **GUIDANCE — pick ONE of:**
  - **(preferred) Report it in the appendix table honestly and DO NOT use it to harden
    C3.** Phrase: "Cross-language AUC and overall Recall@10 are strongly collinear across
    the 7 non-degenerate models; at this n the separability signal cannot be statistically
    disentangled from general capability (partial r=+0.30, n.s.). We therefore frame the
    separability→floor link as descriptive, not as an effect net of capability." This is
    the *honest* outcome and is still a credibility win (it shows the team probed its own
    mechanism), but the writer must NOT claim the dreamer's intended "not a capability
    artifact" line — the data refutes it at n=7.
  - **(acceptable) Omit W2 entirely.** The appendix table (A1+A5+A6) stands on its own;
    W2's weak partial-r is not load-bearing and the paper already hedges C3 as
    correlational in Limitations. Dropping it loses nothing.
  - **(do NOT) Spin it as supporting C3.** It does not.
- **outputs.** one row in `robustness_appendix.csv` (if kept) + a summary key.
- **runtime.** seconds.
- **verify.** partial r ≈ +0.30, p ≈ 0.52 (reproduces my sanity run).
- **api-cost.** 0.
- **RISK FLAG.** This is the one DO-NOW that could *weaken* a paper claim if mishandled.
  Sequenced last and gated behind explicit honesty guidance. If in doubt, **omit (option
  b)** rather than risk the writer over-reading it.

---

## BACKLOG-EVAL (exact commands + rationale for needs_eval.md)

**Nothing new this round.** Channel (b) is FREEZE; the dreamer minted no new metric and
proposed no new embedding run. The existing `needs_eval.md` backlog (W3 alignment causal
probe, W4 formula-injection, CLIRMRS-external-validation, XRC-conformal, CCI-hop-distance,
equivalence-audit) is unchanged and remains the correct deferred set. The novelty critic's
"W3 alignment causal probe with ARI as before/after target" is **already in needs_eval.md
(W3-alignment-causal-probe, r2, updated r3 to name the ARI floor as the target)** — do NOT
re-add it. The only paper-side action for W3 this round is the WRITER prediction-box (W3
below in WRITER-ONLY), which costs nothing and runs no eval.

---

## WRITER-ONLY (reframes/citations to pass forward)

The conductor's 3 named text fixes plus the dreamer's framing-only items. All ~1 clause each.

1. **ρ_k residual-decomposition citation (dreamer C-a / novelty near-mandatory).** Add the
   `residualrerank2026` bib entry (arXiv:2604.01506, verbatim in dreamer C-a) and cite once
   in the ARI paragraph (§4, ~L466) with the credit-and-distinguish clause: "Normalizing a
   re-ranking remainder by a recoverable gap has a precedent in long-tailed reranking
   \citep{residualrerank2026}; we invert it (the un-rerankable share rather than the
   reranker's recoverable gain) and tie it to representation alignment cross-lingually, with
   an alignment-only floor ρ_k has no analogue for." **This is the round's only
   near-mandatory missing cite.** (Bib mechanics also noted in DO-NOW-1.)
2. **cp_fig22 caption "nine vs. seven" reconcile (dreamer C-b / correctness D-NEW / cohesion).**
   Change L703 from "The three sum to $1.0$ for all nine models." to "The three sum to $1.0$
   for every model (the identity closes for all nine; the figure shows the seven
   non-degenerate)." **Change no value.** The 2 omitted models are e5-large-instruct,
   gte-base (`degenerate_models_excluded_from_figure`).
3. **§6.1 ARI read-off trim (dreamer C-c / cohesion seam #1).** Replace the L687–690
   re-definition opener with the connective clause in dreamer C-c: "The ARI decomposition
   (Figure~\ref{fig:ari}) reports this split per model. For `embeddinggemma` the
   alignment-only floor is the smallest of any non-degenerate model, and its post-re-rank
   residual is the lowest: $\mathrm{ARI@}100 = 0.229$ (next `qwen3-0.6B` at $0.233$)…"
   Net line saving. No number changes.
4. **The float cut/merge (dreamer C-d / cohesion — the dominant ask), 29 → 25:**
   - **CUT cp_fig14 + ag_fig10 (the two radars)** (−2). Absorb their "where each model
     wins" half-sentence via dreamer A7: append to the leaderboard paragraph
     "(`embeddinggemma` leads consistency and separability; per-axis detail is in Tables 1–2
     and Figure~\ref{fig:cp_ribbon})." Tables 1–2 + cp_fig17 cover the lost content.
   - **MERGE cp_fig06 + cp_fig07 (mate-retrieval) → one 2-panel** (−1): wrap as two
     `subfigure`s under one `\begin{figure}`/`\caption`/`\label`, collapse the two `\ref`s.
     (Or use DO-NOW-4's `cp_fig06_07_mate.png` if preferred.)
   - **MERGE cp_fig09 + cp_fig10 (language-collapse) → one 2-panel** (−1): same mechanics;
     both already cited in the single parenthetical at L958 — point it at the merged label.
   - **STAYS (explicit):** cp_fig22 (ARI ×3), cp_fig23 (per-route ×2), cp_fig18 (cost
     frontier), cp_fig19 (RRC budget), cp_fig20 (degeneracy gate), cp_fig17 (ribbon — it
     covers the radars), teaser, both leaderboard tables. **Stretch (last resort only):**
     cp_fig17 ribbon — but cutting it undermines the radar cut's cover story, so hold.
5. **Float-order swap (dreamer C-e / cohesion seam #3, cosmetic).** Swap the `\begin{figure}`
   blocks so cp_fig19 (RRC budget) precedes cp_fig22 (ARI). Lowest priority; do not let it
   cost a rewrite. Letting LaTeX float them is acceptable since prose order is already right.
6. **W3 prediction-box (dreamer W3, framing-only).** Tighten the Limitations sentence
   (~L1250–1255) into one falsifiable line naming the exact ARI/L∞ targets: "an alignment
   intervention should drop embeddinggemma's $L_\infty$ (0.058) / ARI@100 (0.229) while
   leaving RRC@K-under-re-ranking flat — the experiment that converts the correlational
   'align, don't re-rank' into a causal claim." No new float; do NOT run the probe.
7. **Appendix table reference (from DO-NOW-2).** Once `robustness_appendix.csv` exists, add
   one appendix `\begin{table}` and reference it once from §7 ("every load-bearing scalar
   survives resampling; see Appendix Table~\ref{tab:robust}"). Lead each row with the
   stability metric (sign-stability / order-probability), not the CI width.
8. **(carryover, cosmetic) "home advantage" hyphenation** — harmonize to unhyphenated on
   the final polish pass (cohesion, 3-round carryover). Don't spend a content edit on it.

---

## Round API budget plan (target 0, cap 20)

**Total planned API calls: 0.** No item calls a model, an LLM, or `--evaluate-mteb`. All
computation is CPU-only over on-disk CSVs + the 13 MB parquet via `core_per_query()`.
Well under the ≤20 cap; meets the target of 0.

---

## Risks & sequencing notes

1. **Do the cut/merge + 4 text fixes FIRST (DO-NOW-1 + WRITER 1–5).** This is the #1 ask
   from all three critics and the only pre-submission blocker (page budget). It is pure
   editing, zero compute, and lands the paper inside the budget (29→25). Everything else is
   additive hardening that can follow.
2. **DO-NOW-2 (appendix table) is the highest-value compute.** It is additive and
   appendix-only — it **cannot destabilize any body figure or number** because it reads the
   same RRC/separability/XRC sources the body already cites and re-expresses them as
   intervals. Gate it with the `verify_*` reproductions so a regression is caught instantly.
3. **TWO hardening numbers come back WEAKER than the dreamer hoped — flagged, not hidden:**
   - **W2 partial-r = +0.30 (n.s.)** (DO-NOW-5): does NOT support "not a capability
     artifact." Report honestly or omit; never spin. This is the round's one backfire risk.
   - **τ-sweep (DO-NOW-3): "cheapest = bge-m3" holds only for τ∈[0.35, ~0.43]**, flips to
     granite at τ=0.30. Report the honest narrow band; egemma's max-CLIR corner is the only
     τ-invariant part. Do not overstate robustness.
   Both are still *net positive* (they pre-empt the obvious objections and show the team
   probed its own claims) — but only if framed honestly. The implementer must read the
   GUIDANCE blocks before writing prose.
4. **Figure-merge: prefer LaTeX `subfigure` over the stitch script (DO-NOW-4).** The panels
   are key_findings outputs, not re-runnable extra_ outputs; a stitch script just
   re-photographs verified PNGs and adds a derived asset to maintain. Keep it as a fallback.
   **Never re-plot the merged panels from the per-query CSVs** — risks divergence from the
   correctness-verified figures.
5. **BCa vs percentile (DO-NOW-2):** `common.bootstrap_ci` is percentile-only; there is no
   BCa helper. Use percentile + sign/order-stability (sufficient and what the dreamer's own
   caveat says is load-bearing). Add a small BCa function only if specifically wanted; do
   not block on it.
6. **Channel (b) = FREEZE confirmed.** No new metric, no new body float, no new embedding
   run. The metric family (CLIR@k / home / directional / mate / RBO / collapse / sep-AUC /
   XRC / RRC / ARI / DEG) is closed. The round's value is trim (top-1) + harden-existing
   (top-2/3), exactly the conductor's bias.
