# Implementation report (round 2)

All three Top-3 DO-NOW items executed. **0 API calls.** No `--evaluate-mteb`, no `run_all.py`,
no `build_key_findings.py`, no GPU, no network (everything read the local HF cache /
`HF_HUB_OFFLINE=1` or pure on-disk CSVs). Every new output went to a **new** `extra_*` dir; no
existing round-1 output, `key_findings/`, or figure was modified. DO-NOW-4 (paired XRC) was **not**
run — it is optional and lower-priority; the WRITER-ONLY one-clause caveat (W-COR-TNEW) closes
T-NEW without it, and the round budget/scope was satisfied by the Top-3.

Interpreter: `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python` (symlink → /usr/bin/python3),
scipy 1.17.1 (spearmanr available).

---

## Done

### DO-NOW-1 — cross-lingual COST FRONTIER (closes N2; upgrades C4; delivers M2 + W2)
- **what.** New `extra_cost_frontier.py`. Loads `extra_xrc_reading_cost/xrc_per_model.csv` only
  (no parquet). Computes the 2-objective Pareto frontier (max CLIR@10, min XRC50) over finite-XRC
  models; gte-base (blank XRC50) excluded from the sweep and plotted as an off-plane censored
  marker. W2 trap = Spearman ρ(XRC50, CLIR@10) over the non-degenerate set (DEG gate
  clir<0.10 → drops {gte, e5}). τ=0.40 stated (not tuned) deployment read-off.
- **command.**
  `.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_cost_frontier.py`
- **outputs.** `reports/runs/chem_patents/experimental_plots/extra_cost_frontier/`
  → `cost_frontier.csv`, `summary.json`, `cost_frontier.png`.
- **REAL numbers / verification.**
  - **Pareto frontier members = {embeddinggemma, bge-m3, granite-278m}.** embeddinggemma ∈ frontier
    ✓ (verify asserted), granite-278m ∈ frontier ✓. Interior-finite models qwen3 (dominated by
    bge-m3), nomic-v2-moe, LaBSE, SapBERT, e5 are all dominated.
  - Verify values match CSV: egemma XRC50=3.5 / CLIR@10=0.5024; granite XRC50=1.25 / CLIR@10=0.3285.
  - **τ=0.40 admitted set = {bge-m3, qwen3-0.6B, embeddinggemma}; min-XRC50 = bge-m3 (2.0), NOT
    embeddinggemma.** This is the empirical confirmation that the N2 "cheapest deployable = egemma"
    superlative is FALSE and must not be revived. The honest claim baked into summary.json: egemma
    is the unique **max-CLIR@10 corner** of the frontier and is Pareto-optimal.
  - **W2 trap Spearman ρ(XRC50, CLIR@10), non-deg n=7 = +0.2857, p=0.5345 → POSITIVE → trap
    SUPPORTED** (`W2_trap_supported_positive_rho: true`). CAVEAT for the writer: positive sign
    supports the "cheapest reader can be the worst retriever" framing, but the magnitude is weak and
    NOT statistically significant at n=7 (p=0.53). Use the trap sentence as illustrative/directional,
    not as a strong statistical claim.
- **figure check.** Rendered and visually verified: frontier polyline granite→bge-m3→egemma; egemma
  at top-right max-CLIR corner; dominated models as X; e5 with red degenerate edge; gte off-plane ▼.
- **api-calls-used.** 0.

### DO-NOW-3 — DEG gate + TWO-TAX non-redundancy (closes P3/P4; hardens N2; delivers M1 + M4)
- **what.** New `extra_two_tax_degeneracy.py`. Reads three on-disk CSVs (chem xrc, chem rrc, alias
  severity_split); no parquet, no alias `common` import (alias CSV read by absolute path). Emits
  three DEG candidate rules + membership; builds the two-tax table (reading-cost tax = XRC50,
  confusability tax = confusion_rate) joined on `short`; computes cross-model Spearman on all-9 and
  the n=7 non-degenerate set.
- **command.**
  `.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_two_tax_degeneracy.py`
- **outputs.** `reports/runs/chem_patents/experimental_plots/extra_two_tax_degeneracy/`
  → `deg_flags.csv`, `two_tax_table.csv`, `summary.json`, `degeneracy_gap.png`, `two_tax_scatter.png`.
- **REAL numbers / verification.**
  - **DEG_strict (clir<0.10 AND rrc1000<0.10) flags ONLY {gte-base}** — confirms the dreamer's
    literal AND-gate misses e5 (e5 RRC@1000=0.6277 ≥ 0.10).
  - **DEG_clir_only (clir<0.10) flags exactly {gte-base, e5-large-instruct}** →
    `matches_paper_exclusions_{gte,e5}: true`. **This is the recommended single criterion**; clean
    gap SapBERT 0.1788 vs e5 0.0766. The writer should adopt `DEG = CLIR@10 < 0.10`.
  - **Two-tax Spearman ρ(reading-cost tax, confusability tax):**
    - all-9 finite (n=8, gte XRC50 is NaN/blank so dropped): ρ = **-0.1557**, p=0.7128.
    - **n=7 non-deg: ρ = -0.5946, p=0.1591** → `nonredundant_supported_abs_rho_lt_0.6: true`.
  - Join sanity confirmed by `short`: egemma XRC50=3.5 / conf=0.0682; granite XRC50=1.25 / conf=0.1818.
  - **HONESTY FLAG for the writer (important):** the n=7 ρ is **-0.5946** — it is *negative* (a mild
    anti-correlation, not the "positive-but-weak" the plan loosely anticipated) and its magnitude
    sits **just under** the stated 0.6 non-redundancy threshold (and p=0.16 is not significant at
    n=7). The "two taxes are non-redundant / both benchmarks necessary" claim is **technically
    supported** by the stated |ρ|<0.6 gate, but it is borderline. Recommended phrasing: *"the two
    taxes are only weakly (and if anything inversely) rank-correlated across the seven non-degenerate
    models (Spearman ρ = −0.59, n=7, p=0.16, n.s.), so neither benchmark is a proxy for the other."*
    Do NOT state a strong/significant non-redundancy claim. The all-9 ρ (−0.16) is even weaker and
    can corroborate.
- **figure check.** Both rendered and verified: degeneracy_gap bar with 0.10 cutoff (e5 red below;
  gte at 0.000); two_tax_scatter with egemma low-low, e5 degenerate X top-right.
- **api-calls-used.** 0.

### DO-NOW-2 — RRC BUDGET FRONTIER (closes P2 / novelty #1; delivers M3)
- **what.** New `extra_rrc_budget_frontier.py`. Loads raw `first_cross_rank` via
  `C.core_per_query()` (the one parquet load; lru_cached). Builds RRC(K) on a 16-point K-grid,
  marginal dRRC/dK (per 100 candidates), the knee K* (max-distance-to-chord with **log10-K**
  normalization), and L∞ = 1 − RRC(1000). Regression-checked against round-1 `rrc_per_model.csv`.
- **command.**
  `.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_rrc_budget_frontier.py`
- **outputs.** `reports/runs/chem_patents/experimental_plots/extra_rrc_budget_frontier/`
  → `rrc_curve.csv`, `rrc_knee.csv`, `summary.json`, `rrc_budget_frontier.png`, `rrc_xrc_plane.png`.
- **REAL numbers / verification.**
  - **All regression checks PASSED** (`regression_checks_passed: true`, empty failures list): for all
    9 models RRC@100 and RRC@1000 reproduce `rrc_per_model.csv` to <1e-3; L∞ == lost_at_1000; RRC
    monotone non-decreasing in K. (egemma RRC@100=0.7445 / RRC@1000=0.9416 / L∞=0.0584;
    e5 L∞=0.3723 — all verified.)
  - **embeddinggemma knee K*=5, RRC(K*)=0.4818, L∞=0.0584.**
  - K* by model (non-deg): egemma 5, bge-m3 5, qwen3 10, nomic 2, granite 20, LaBSE 20, SapBERT 20;
    e5 30; gte 100. Interpretation: the recoverability elbow lands shallow (K*≈5–20) for the strong
    models — most of the re-ranker payoff is in a small top-K pool — but L∞ (the unrecoverable floor)
    ranges from 0.058 (egemma) to 0.372 (e5), the structural tax no re-ranker over the retrieved
    top-1000 can touch.
- **figure check.** Both rendered and verified: 8 non-deg RRC(K) curves on log-K with knee rings and
  egemma's 0.942 ceiling line; the XRC50×L∞ planning plane with egemma bottom-left (cheap to read +
  little lost).
- **api-calls-used.** 0.

---

## Backlogged to needs_eval.md
- `W3-alignment-causal-probe` (added, r2): fit a per-language linear alignment map on ONE model
  (e.g. LaBSE), re-embed queries+corpus, re-retrieve over multilingual_GP, recompute XRC50 + RRC@100
  before/after on the same 137 cross queries. Reason: requires re-embedding → NEW eval (out of the
  CPU-only / 0-API scope). Marked UPSIDE-ONLY — the paper must not depend on it. The dreamer's other
  needs-eval ideas (M5 gXRC → `XRC-conformal-M2`; CLIRMRS external validation) were already present
  from r1 and were NOT re-added (per troubleshoot instruction).

## New figures copied to paper/figures/ (basename → source)
- `cp_fig18_cost_frontier.png`   → `extra_cost_frontier/cost_frontier.png`
- `cp_fig19_rrc_budget.png`      → `extra_rrc_budget_frontier/rrc_budget_frontier.png`
- `cp_fig20_degeneracy_gap.png`  → `extra_two_tax_degeneracy/degeneracy_gap.png`
- `cp_fig21_two_tax.png`         → `extra_two_tax_degeneracy/two_tax_scatter.png`
(Copied with `cp -n` — none pre-existed, nothing overwritten. The secondary panels
`rrc_xrc_plane.png` and the per-model CSVs/JSONs remain in the `extra_*` dirs for the writer to wire
if wanted; not copied to paper/figures.)

## Failures / surprises (verbatim / real outcomes)
- **No hard failures.** All three scripts ran clean on the first execution; all asserts and
  regression checks passed.
- **Surprise 1 (data-conditional claim, resolved honestly):** the two-tax n=7 Spearman is
  **ρ = −0.5946** (negative, |ρ| just under 0.6, p=0.16 n.s.), not the loosely-anticipated weak
  positive. Non-redundancy is *technically* supported by the stated |ρ|<0.6 gate but is **borderline
  and non-significant** — flagged above for the writer to phrase cautiously (no strong claim).
- **Surprise 2 (W2 trap weak):** the W2 trap ρ = +0.2857 is positive (so the trap *direction* holds
  and the framing is permitted) but weak and non-significant (p=0.53). Use illustratively, not as a
  statistical result.
- **Cosmetic (non-blocking):** in `cp_fig18_cost_frontier.png` the legend and the gte off-plane
  annotation slightly overlap in the top-left corner; still legible. In `cp_fig20_degeneracy_gap.png`
  the gte-base bar is invisible because CLIR@10=0.000 (correct; it is labeled). Neither affects
  correctness; left as-is.
- harmless stderr from HF datasets ("Using the latest cached version … offline mode is enabled") —
  expected, confirms no network was hit.

## API calls used this round: 0/20
