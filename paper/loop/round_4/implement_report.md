# Implementation report (round 4)

Operating as **The Implementer**. Executed the CPU-only, 0-API DO-NOW items from
`paper/loop/round_4/troubleshoot.md`: DO-NOW-2 (appendix robustness table, incl. DO-NOW-5
partial-r folded in), DO-NOW-3 (τ-sweep), DO-NOW-4 (optional 2-panel stitch). DO-NOW-1 is
WRITER-ONLY (bib/tex) and was left to the writer (its `.bib`/`.tex` edits were already present
in the working tree and were not touched by me).

All numbers below are ACTUAL computed values. Two hardening numbers came back weaker than the
dreamer assumed (τ-band narrower; partial-r n.s.) — reported honestly per the troubleshooter's
guidance, not spun.

## Done

### DO-NOW-2 — Appendix robustness table (A1 + A6 + A5) + DO-NOW-5 (W2 partial-r)
- **what:** New script resampling the three load-bearing scalars (percentile bootstrap via the
  `common.bootstrap_ci` family; NO BCa — none exists, troubleshooter option (a)). Headline is
  sign/order stability, not CI width. n_boot=10000, seed=20260610, alpha=0.05. W2 partial-r
  folded in as one extra row.
- **command run:**
  `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_robustness_appendix.py`
- **output paths:**
  - `reports/runs/chem_patents/experimental_codes/extra_robustness_appendix.py` (new script)
  - `reports/runs/chem_patents/experimental_plots/extra_robustness_appendix/robustness_table.csv`
  - `reports/runs/chem_patents/experimental_plots/extra_robustness_appendix/summary.json`
  - NO png (this is an appendix TABLE, not a body float — as instructed).
- **quick verification (all self-check gates PASS):**
  - A1 r(n7)=0.9577 (gate vs published 0.958, ±0.01 ✓); r(n9)=0.8877.
  - A6 XRC50 points reproduce `xrc_per_model.csv` exactly: egemma 3.5, bge-m3 2.0, granite 1.25 (gates ✓).
  - A5 ARI@100 egemma 0.2286, qwen3 0.2326 (gates vs `ari_decomposition.csv`, ±0.001 ✓).
  - `gates_all_pass: True`.
- **REAL numbers:**
  - **A1 (separability r, n=7 model-level bootstrap):** point r = **0.9577**, 95% CI **[0.730, 0.998]**,
    **sign-stability P(r>0) = 0.9997** (32/10000 degenerate draws skipped). The sign is essentially
    certain; the CI is wide at n=7 (small-n), which is exactly why sign-stability is the headline.
  - **A6 (XRC50 depth bootstrap, 3 frontier members; cross n=137 & same n=57 resampled
    independently):**
    - embeddinggemma: 3.5, 95% CI **[0.909, 12.0]**, censored-draw frac **0.0** (finite CI).
    - bge-m3: 2.0, 95% CI **[0.529, 7.0]**, censored-draw frac **0.0** (finite CI).
    - granite-278m: 1.25, 95% CI **[0.284, 12.25]**, censored-draw frac **0.0** (finite CI).
    - None exceeded the 5% censored-draw threshold, so all three CIs are genuine (not lower bounds).
      CIs are wide (median-of-discrete-depths bootstrap at this n) but all stay finite.
  - **A5 (ARI@100 egemma-vs-qwen3 0.004 gap, per-query PAIRED bootstrap n=137):** gap
    (qwen3−egemma) point = **0.0040**, 95% CI **[-0.174, 0.176]** — **CI INCLUDES 0**.
    Order-prob **P(ARI_egemma < ARI_qwen3) = 0.519** (near coin-flip). Honest read: the 0.004 gap is
    **not a reliable ordering**; report the two ARI@100 values as effectively tied, not as a strict
    egemma<qwen3 win.
  - **W2 (DO-NOW-5, partial-r):** partial r(auc_cross, CLIR@10 | Recall@10), n=7 = **+0.2948**,
    two-sided **p=0.5706** (n.s.); zero-order r=+0.9577. AUC and overall Recall@10 are strongly
    collinear; the separability→CLIR link **cannot be statistically disentangled from general
    capability at this n**. Reported DESCRIPTIVELY only — does NOT support a "not a capability
    artifact" claim (do not spin).
- **api-calls-used:** 0 (reads on-disk CSVs + `core_per_query()` from the offline HF cache).

### DO-NOW-3 — τ-sensitivity sweep of the cost frontier
- **what:** Standalone `extra_tau_sweep.py` (kept `extra_cost_frontier.py` unmodified to avoid
  destabilizing the verified τ=0.40 read-off). Sweeps τ ∈ {0.30,0.35,0.40,0.45,0.50} (coarse) and
  a 0.005-step fine grid for boundary detection. Admit logic and cheapest=argmin-finite-XRC50
  match `extra_cost_frontier` exactly.
- **command run:**
  `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_tau_sweep.py`
- **output paths:**
  - `reports/runs/chem_patents/experimental_codes/extra_tau_sweep.py` (new script)
  - `reports/runs/chem_patents/experimental_plots/extra_cost_frontier/tau_sweep.csv`
  - `reports/runs/chem_patents/experimental_plots/extra_cost_frontier/tau_sweep_summary.json`
  - (original `cost_frontier.csv`/`.png`/`summary.json` UNTOUCHED — confirmed by timestamps.)
- **quick verification:** τ=0.40 row reproduces the existing `extra_cost_frontier/summary.json`:
  admitted = {bge-m3, embeddinggemma, qwen3-0.6B} (matches `tau_admitted_set` ✓), cheapest = bge-m3
  (matches `tau_admitted_min_xrc_model` ✓).
- **REAL numbers (the three summary keys + the honest narrow band):**
  - **(i) tau_admitted_stable_range** (admitted == {bge-m3, qwen3, egemma}): **τ ∈ [0.385, 0.430]**.
  - **(ii) tau_cheapest_bge_range** (cheapest-admitted == bge-m3): **τ ∈ [0.330, 0.435]**. (Slightly
    wider than the troubleshooter's eyeball [0.35,~0.43]; the 0.005 grid shows bge-m3 stays cheapest
    down to 0.330.)
  - **(iii) egemma_corner_tau_invariant = True** — embeddinggemma is the unique global max-CLIR@10
    corner for ALL τ (τ-invariant by construction).
  - **Honest narrow band (confirmed, not spun):** at **τ ≤ 0.3285** granite-278m (CLIR@10=0.3285,
    XRC50 1.25 < bge-m3's 2.0) enters the admitted set and becomes cheapest — the
    cheapest-reader recommendation **FLIPS** to granite. At **τ ≥ 0.45** only embeddinggemma is
    admitted. So "cheapest admitted = bge-m3" holds only over τ∈[0.330, 0.435]; the τ-invariance of
    the egemma max-CLIR corner is the only unconditional part. The writer must NOT overstate
    robustness at the low end.
  - **Separability partial-r controlling for Recall@10 (the inline number the conductor asked for):
    partial r = +0.2948, p = 0.5706 (n.s.), zero-order r = +0.9577** — computed in the W2 block of
    DO-NOW-2 (no float, inline only).
- **api-calls-used:** 0 (reads `xrc_per_model.csv` only).

### DO-NOW-4 (OPTIONAL) — pre-stitched 2-panel PNGs
- **what:** One-shot `stitch_merged_panels.py` loads the 4 already-rendered, correctness-verified
  per-panel PNGs and places each pair side-by-side via `imread`/`imshow` (axis off). NO re-plot
  from per-query CSVs (avoids divergence from key_findings). Provides the writer a single
  `\includegraphics` option as a fallback to LaTeX `subfigure`.
- **command run:**
  `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python reports/runs/chem_patents/experimental_codes/stitch_merged_panels.py`
- **output paths (written directly to paper/figures/):**
  - `paper/figures/cp_fig06_07_mate.png` (cp_fig06_mate_retrieval + cp_fig07_first_foreign_rank)
  - `paper/figures/cp_fig09_10_collapse.png` (cp_fig09_language_collapse + cp_fig10_distractor_language)
  - `reports/runs/chem_patents/experimental_codes/stitch_merged_panels.py` (new script)
- **quick verification:** both merged PNGs open and show both source panels legibly (visually
  confirmed: mate-retrieval bars + first-foreign-rank curves; over-rep bars + distractor-language
  stacks). Source per-panel PNGs untouched.
- **api-calls-used:** 0.

## Backlogged to needs_eval.md
- **None.** Channel (b) is FROZEN; the troubleshoot's BACKLOG-EVAL section is "Nothing new this
  round." Every DO-NOW item was CPU-only and completed. `needs_eval.md` left unchanged.

## New figures copied to paper/figures/
- `cp_fig06_07_mate.png` -> stitched from `cp_fig06_mate_retrieval.png` + `cp_fig07_first_foreign_rank.png`
- `cp_fig09_10_collapse.png` -> stitched from `cp_fig09_language_collapse.png` + `cp_fig10_distractor_language.png`
- (The robustness table is a CSV, not a figure — no body/appendix float PNG, as instructed.)

## Failures / surprises (real outcomes)
- **No run failures.** All three scripts ran clean; all self-check gates PASS (`gates_all_pass: True`).
- **Cosmetic stdout bug found & fixed:** the A1 CI in the final `print` initially reused the
  last-loop `lo,hi` (granite XRC values [0.284,12.25]); the CSV/JSON were always correct
  ([0.730,0.998]). Renamed the A1 percentile vars to `a1_lo/a1_hi` and re-ran — stdout now prints
  the correct A1 CI [0.730,0.998].
- **Surprise vs dreamer (both pre-flagged by the troubleshooter, both confirmed):**
  1. **τ-band narrower than assumed:** cheapest=bge-m3 only over τ∈[0.330,0.435]; flips to granite
     at τ≤0.3285. Only the egemma max-CLIR corner is τ-invariant.
  2. **W2 partial-r weak/n.s.:** +0.2948 (p=0.5706); separability is collinear with overall recall
     at n=7 — descriptive framing only, do not claim "not a capability artifact."
- **A5 gap CI includes 0:** the 0.004 ARI@100 egemma-vs-qwen3 ordering is NOT statistically reliable
  (CI [-0.174,0.176], P(egemma<qwen3)=0.519). Treat the two as effectively tied.
- **Path note:** the conductor's example path `extra_robustness_resamples/` was a non-binding "e.g.";
  I used the troubleshoot.md authoritative dir `extra_robustness_appendix/` with the conductor's
  requested filename `robustness_table.csv`.

## API calls used this round: 0/20
