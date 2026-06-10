# Reporter handoff (round 2) -> feeds story + writer (round 3)

## TL;DR
Round 2 added three CPU-only, 0-API experiments (cost frontier, RRC budget frontier, two-tax
degeneracy) plus four figures (cp_fig18-21). Every claimed value was opened and confirmed against the
JSON/CSV -- no fabrication, no discrepancies, figures are byte-identical (md5) to their source plots.
The new results give the paper a clean deployment story (a 3-model Pareto frontier, a re-ranker
budget knee, and a principled degeneracy gate DEG = CLIR@10 < 0.10 that exactly recovers the {gte,
e5} exclusions). Two headline correlations are NOT statistically significant (two-tax rho=-0.59
p=0.16; W2 trap rho=+0.29 p=0.53) and MUST be written as directional/descriptive, not as established
effects.

## Verified new results
Each: value -> source path -> figure basename -> paper section/claim it affects. All values below
were read directly from the listed file and match the implement_report and the conductor brief.

### Cost frontier (DO-NOW-1)
- Pareto frontier members = {embeddinggemma, bge-m3, granite-278m} (max CLIR@10, min XRC50 over
  finite-XRC models). embeddinggemma in frontier, granite in frontier -- both true in JSON.
  - src: reports/runs/chem_patents/experimental_plots/extra_cost_frontier/summary.json
    (frontier_members), .../cost_frontier.csv (on_frontier column).
  - fig: cp_fig18_cost_frontier.png.
  - Anchor values (verified): embeddinggemma XRC50=3.5 / CLIR@10=0.5024; bge-m3 XRC50=2.0 /
    CLIR@10=0.4367; granite-278m XRC50=1.25 / CLIR@10=0.3285. Dominated finite: qwen3-0.6B (by
    bge-m3), nomic-v2-moe, LaBSE, SapBERT, e5-large-instruct. gte-base = off-plane censored (blank
    XRC50).
  - paper claim it affects: the deployment/cost-tradeoff section (was N2/C4). Closes N2 and upgrades
    C4 to a real frontier figure.
- embeddinggemma is the UNIQUE max-CLIR@10 corner of the frontier and is Pareto-optimal -- but it is
  NOT the cheapest deployable model. At the stated (untuned) deployment threshold tau=0.40 the
  admitted set is {bge-m3, qwen3-0.6B, embeddinggemma}, and the min-XRC50 admitted model is bge-m3
  (XRC50=2.0), not embeddinggemma (3.5).
  - src: same summary.json (unique_top_clir_model: "embeddinggemma", tau_admitted_set,
    tau_admitted_min_xrc_model: "bge-m3", tau_admitted_min_xrc_value: 2.0, HONEST_CLAIM).
  - paper claim it affects: this is the empirical kill-switch for the old "cheapest deployable =
    embeddinggemma" superlative (N2). The writer MUST NOT revive it. Correct phrasing: egemma is the
    Pareto-optimal capability corner; bge-m3 is the cheaper-to-read admitted alternative.

### Two-tax non-redundancy + DEG gate (DO-NOW-3)
- DEG gate recommendation: DEG = CLIR@10 < 0.10, which flags exactly {gte-base, e5-large-instruct}
  -- matches the paper's existing degenerate-model exclusions (matches_paper_exclusions_{gte,e5}:
  true). The dreamer's literal AND-gate (clir<0.10 AND rrc1000<0.10) flags ONLY {gte} because e5
  RRC@1000=0.6277 >= 0.10; the single-criterion CLIR gate is the clean rule (gap: SapBERT 0.1788 vs
  e5 0.0766).
  - src: reports/runs/chem_patents/experimental_plots/extra_two_tax_degeneracy/summary.json
    (deg_gate block), .../deg_flags.csv.
  - fig: cp_fig20_degeneracy_gap.png (0.10 cutoff line; e5 in red below cutoff; gte at 0.000).
  - paper claim it affects: methods/definitions -- gives a principled, reproducible justification for
    dropping {gte, e5} that previously may have looked ad hoc. Hardens N2.
- Two-tax (reading-cost tax = XRC50 vs confusability tax = alias-graph confusion_rate) cross-model
  Spearman:
  - n=7 non-degenerate: rho = -0.5946, p = 0.1591 (NOT significant).
  - all-9 finite (n=8, gte XRC50 NaN dropped): rho = -0.1557, p = 0.7128.
  - src: same summary.json (two_tax block: spearman_rho_n7_nondeg, spearman_p_n7,
    spearman_rho_all9_finite, spearman_p_all9), .../two_tax_table.csv.
  - fig: cp_fig21_two_tax.png (title already states "n=7 non-deg Spearman rho = -0.59").
  - paper claim it affects: the "two benchmarks are non-redundant / neither is a proxy for the other"
    argument. CRITICAL HONESTY CAVEAT below -- this is borderline and non-significant.
  - Join anchors (verified): embeddinggemma XRC50=3.5 / conf=0.0682 (low-low); granite XRC50=1.25 /
    conf=0.1818.

### RRC budget frontier (DO-NOW-2)
- All round-1 regression checks PASSED (regression_checks_passed: true, empty failures): RRC@100 and
  RRC@1000 reproduce round-1 rrc_per_model.csv to <1e-3 for all 9 models; L_inf == lost_at_1000; RRC
  monotone non-decreasing in K.
  - src: reports/runs/chem_patents/experimental_plots/extra_rrc_budget_frontier/summary.json,
    .../rrc_knee.csv (note lost_at_1000_ref == L_inf column-for-column).
- embeddinggemma knee K*=5, RRC(K*)=0.4818, L_inf=0.0584 (RRC@100=0.7445, RRC@1000=0.9416).
  K* by model: egemma 5, bge-m3 5, qwen3 10, nomic 2, granite 20, LaBSE 20, SapBERT 20, e5 30, gte
  100. L_inf floor ranges 0.058 (egemma) -> 0.372 (e5) -> 0.912 (gte).
  - src: same summary.json (embeddinggemma, K_star_by_model, L_inf_by_model), .../rrc_knee.csv.
  - fig: cp_fig19_rrc_budget.png (8 non-deg RRC(K) curves on log-K, knee rings, egemma 0.942 ceiling
    line). Secondary panel rrc_xrc_plane.png exists in the extra dir (NOT copied to paper/figures) if
    the writer wants the XRC50 x L_inf planning plane.
  - paper claim it affects: novelty #1 (P2) -- re-ranker budgeting. Supports "most re-ranker payoff
    is in a shallow top-K (K*~5-20), but L_inf is a structural floor no re-ranker over the top-1000
    can touch." This is a clean, significant-by-construction result (regression-checked), safe to
    state firmly.

## Discrepancies / unverifiable claims
None. Every numeric claim in implement_report.md was confirmed against the source file. The four
paper figures are md5-identical to their extra_* source PNGs (verified). The two reported cosmetic
issues are real and harmless: (1) cp_fig18 legend slightly overlaps the gte off-plane annotation in
the top-left -- still legible; (2) cp_fig20 gte-base bar is invisible because CLIR@10=0.000 --
correct and labeled. Minor note for the writer, not a discrepancy: cp_fig18's title legend says "X =
dominated", but degenerate models (e5, nomic) also render with red X edges; the marker semantics are
clear in context but the writer may want one clarifying clause in the caption.

## HONESTY CAVEATS THE WRITER MUST HONOR (round 3)
These two correlations are the anti-fabrication focus of this round. Do not upgrade either into a
statistically established effect.
1. Two-tax non-redundancy: rho = -0.59, n=7, p=0.16 (n.s.). It is negative (mild anti-correlation),
   not positive, and |rho| sits just under the stated 0.6 non-redundancy threshold. It is technically
   consistent with the |rho|<0.6 "non-redundant" gate but is borderline and non-significant.
   Recommended phrasing (from implementer, endorsed): "the two taxes are only weakly -- and if
   anything inversely -- rank-correlated across the seven non-degenerate models (Spearman rho =
   -0.59, n=7, p=0.16, n.s.), so neither benchmark is a clean proxy for the other." Frame as
   descriptive/motivating, not as a demonstrated independence result. The all-9 rho=-0.16 (p=0.71)
   can corroborate the weakness but is even weaker.
2. W2 trap (XRC50 vs CLIR@10): rho = +0.29, n=7, p=0.53 (n.s.). The sign is positive, so the
   directional framing ("the cheapest reader can be the worst retriever") is permitted as an
   illustration, but the magnitude is weak and the result is not significant. Use it
   illustratively/directionally only -- never as a statistical claim.
3. "Cheapest deployable" superlative is empirically FALSE (bge-m3, not egemma, is the min-XRC50
   admitted model at tau=0.40). Do not state or imply egemma is the cheapest. Egemma = unique
   max-CLIR Pareto corner only.
4. tau=0.40 is a stated, not tuned threshold -- say so; the admitted-set conclusion is conditional on
   it.

## Changed files this round (git diff --stat summary)
Tracked edits (paper text/refs, not new results):
- paper/main.tex (+655/-208 region; substantial restructure)
- paper/custom.bib (+91)
- paper/loop/needs_eval.md (+1: the W3 backlog entry)
New untracked artifacts (this round's deliverables):
- 3 scripts: reports/runs/chem_patents/experimental_codes/extra_{cost_frontier,rrc_budget_frontier,two_tax_degeneracy}.py
- 3 plot dirs: reports/runs/chem_patents/experimental_plots/extra_{cost_frontier,rrc_budget_frontier,two_tax_degeneracy}/
  (each with its CSVs + summary.json + PNGs)
- 4 figures: paper/figures/cp_fig18_cost_frontier.png, cp_fig19_rrc_budget.png,
  cp_fig20_degeneracy_gap.png, cp_fig21_two_tax.png
- paper/loop/round_2/ (loop docs incl. this file)
No round-1 output, key_findings/, or pre-existing figure was modified (confirmed: only extra_* dirs
and cp_fig18-21 are new; cp's are byte-identical to sources). 0 API calls used (0/20).

## Backlogged (forthcoming) experiments to mention as pending
- W3-alignment-causal-probe (added r2, in paper/loop/needs_eval.md): fit a per-language linear
  alignment map on ONE model (e.g. LaBSE), re-embed queries+corpus, re-retrieve over multilingual_GP,
  recompute XRC50 + RRC@100 before/after on the same 137 cross queries. Would elevate "align, don't
  re-rank" (C3) from correlational to a demonstrated intervention. Marked UPSIDE-ONLY -- the paper
  must NOT depend on it; mention as forthcoming only. Requires re-embedding -> a new eval, out of the
  CPU-only/0-API round scope.
- Pre-existing r1 backlog still open (not re-added per troubleshoot instruction): M5 gXRC ->
  XRC-conformal-M2; CLIRMRS external validation.

## Recommended next-round focus (for the story architect)
- Promote the deployment narrative to a first-class contribution. The 3-model Pareto frontier
  (cp_fig18) + the RRC knee/L_inf budgeting (cp_fig19) + the principled DEG gate (cp_fig20) now form
  a coherent "what to deploy and how to budget a re-ranker" story. Restructure so these land
  together, and replace the dead "cheapest = egemma" superlative with the honest "egemma = capability
  corner, bge-m3 = cheaper-read alternative" framing.
- Demote the two-tax and trap correlations to motivation, not findings. Keep cp_fig21 and the
  two-tax point, but the section text must carry the n=7 / p=0.16 (and p=0.53 for the trap) caveats
  inline. Story should not build a load-bearing claim on either.
- Lock the DEG definition once (CLIR@10 < 0.10) and reuse it everywhere {gte, e5} are excluded,
  citing cp_fig20 -- this turns prior ad-hoc exclusions into a defensible, reproducible rule.
