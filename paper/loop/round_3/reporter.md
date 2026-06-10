# Reporter handoff (round 3) -> feeds story + writer (round 4)

## TL;DR
Round 3 added two CPU-only, 0-API analyses, both fully verified against their CSV/JSON outputs.
(1) **ARI decomposition** (cp_fig22): every model's cross-lingual shortfall splits exhaustively into
cheap (RRC@100) + deep (RRC@1000-RRC@100) + alignment-only floor (L_inf), and the identity sums to
**1.0 for all 9 models**; the scalar **ARI@K = L_inf/(1-RRC@K)** isolates the share of the residual a
re-ranker *cannot* recover. egemma has the **lowest ARI@100 (0.229)** and **smallest L_inf (5.8%)** of
any non-degenerate model -- i.e. after a cheap top-100 re-rank, its residual is the least alignment-bound.
(2) **Per-route frontier** (cp_fig23): the capability corner is **not stationary** -- across the 5 query
languages there are **3 distinct max-CLIR corners** (en->qwen3-0.6B, de/es/zh->embeddinggemma,
fr->nomic-v2-moe), and a recall-only dashboard's pick **flips on 2/5 routes** (en, fr). This nuances --
does NOT overturn -- the single-model "deploy embeddinggemma" recommendation (C4): egemma still wins
3/5 routes incl. the two hardest cross-only ones (es, zh), is the global capability corner, and has the
lowest alignment-only floor; a per-route router is genuine **upside headroom**, not a free win (per-route
n is thin: de=27 cross/7 same, zh=22 cross/2 same).

## Verified new results

### ARI decomposition (DO-NOW-1 / M2) -- cp_fig22_ari_decomposition.png
- **Identity closes to 1.0 for ALL 9 models** -- VERIFIED. identity_sum_100 == 1.0 for every row of
  ari_decomposition.csv; summary.json identity_closes_all_models: true. Independently recomputed
  egemma from the authoritative 6-decimal source (extra_rrc_budget_frontier/rrc_curve.csv:
  RRC@100=0.744526, RRC@1000=0.941606) -> cheap 0.7445 + deep 0.1971 + floor 0.0584 = 1.000000. PASS.
- **egemma ARI@100 = 0.2286 (~0.229), lowest non-degenerate** -- VERIFIED. CSV/JSON give 0.2286; my
  recompute L_inf/(1-RRC@100) = 0.0584/(1-0.744526) = 0.2286. It is the minimum ARI@100 among the 7
  non-degenerate models (next: qwen3-0.6B 0.233). PASS.
- **egemma L_inf floor = 0.0584 (smallest of all 9)** -- VERIFIED via L_inf_by_model in summary.json.
- **Per-model ARI@100 (non-degenerate, recall order)** -- VERIFIED exactly against ARI_at_100_by_model:
  embeddinggemma 0.2286, qwen3-0.6B 0.2326, nomic-v2-moe 0.2791, bge-m3 0.3659, LaBSE 0.3492,
  granite-278m 0.4167, SapBERT 0.4189. Degenerate (figure-excluded): e5-large-instruct 0.5204,
  gte-base 0.9124.
  NOTE one wording cleanup for the writer: the implement_report inline prose lists these out of recall
  order; the numbers all match the CSV -- only the narrated ordering is loose. Use the CSV order above.
- **Source:** reports/runs/chem_patents/experimental_plots/extra_ari_decomposition/ari_decomposition.csv
  + summary.json. **Figure basename:** cp_fig22_ari_decomposition.png (present in paper/figures/,
  116535 B, byte-identical to plot source per implementer md5).
- **Paper claim it affects:** strengthens C3 ("align, don't re-rank") with a quantitative split -- the
  paper can now state how much of the gap is re-ranker-recoverable vs. an irreducible alignment floor, and
  that egemma's floor is the lowest. Natural before/after target for the W3 alignment causal probe.

### Per-route frontier + decision-flip (DO-NOW-2 / A1 + DO-NOW-3 / W1) -- cp_fig23_per_route_frontier.png
- **Pooled CLIR@10 reproduces cost_frontier (egemma 0.5024)** -- VERIFIED. summary.json
  verify_egemma_global_clir_0.5024: 0.5024, matches rrc_knee.csv clir_at_10.
- **Capability corner MOVES -- 3 distinct max-CLIR corners** -- VERIFIED. max_clir_corner_by_route =
  {en: qwen3-0.6B (0.4444), de: embeddinggemma (0.6358), es: embeddinggemma (0.5441),
  fr: nomic-v2-moe (0.3889), zh: embeddinggemma (0.553)}; n_distinct_max_clir_corners: 3;
  capability_corner_moves_across_routes: true. Independently recounted distinct set =
  {qwen3-0.6B, embeddinggemma, nomic-v2-moe} = 3. PASS.
- **Decision flips on 2/5 routes (en, fr)** -- VERIFIED. decision_flip_by_route = {en: true, de: false,
  es: false, fr: true, zh: false}; n_routes_flipped: 2. Recall-only pick is embeddinggemma on every
  route; frontier/CLIR pick differs on en (->qwen3-0.6B) and fr (->nomic-v2-moe). Both flip targets sit
  on their own route's frontier (qwen3-0.6B in en frontier; nomic-v2-moe in fr frontier) -- constraint holds.
- **es XRC is NaN (no same-lang gold), never imputed** -- VERIFIED. All 9 es rows in
  per_route_frontier.csv have empty XRC50 + xrc_censored=True; n_same_by_route.es: 0;
  xrc_axis_status.es: "undefined (n_same=0)". PASS.
- **Per-route sample sizes (load-bearing caveat)** -- VERIFIED from n_cross_by_route / n_same_by_route:
  cross = {en27, de27, es34, fr27, zh22}; same = {en21, de7, es0, fr27, zh2}. XRC axis credibility:
  en/fr credible, de/zh thin (indicative), es undefined.
- **Frontier membership by route** -- VERIFIED against frontier_membership_by_route.csv:
  en={qwen3-0.6B, granite-278m, LaBSE}; de={embeddinggemma, bge-m3, qwen3-0.6B, granite-278m, LaBSE};
  es={embeddinggemma} (CLIR-only, 1-D, n_same=0); fr={bge-m3, nomic-v2-moe, granite-278m};
  zh={embeddinggemma} (n_same=2, thin). Global Pareto reference set {bge-m3, embeddinggemma, granite-278m}.
- **Source:** reports/runs/chem_patents/experimental_plots/extra_per_route_frontier/
  {per_route_frontier.csv, frontier_membership_by_route.csv, decision_flip_by_route.csv, summary.json}.
  **Figure basename:** cp_fig23_per_route_frontier.png (present, 186343 B, byte-identical per md5).
- **Paper claim it affects:** nuances C4 (single-model deployment recommendation). See framing below.

## HOW THE WRITER MUST FRAME THE PER-ROUTE FINDING (critical)
The per-route corner movement is real and must be reported, but it is an UPSIDE refinement of C4, not a
reversal. Honest framing the writer MUST hold to:
1. **embeddinggemma is still the single-model recommendation.** It is the global capability corner
   (max pooled CLIR@10 = 0.5024), it has the lowest alignment-only floor (L_inf=0.058) and lowest
   ARI@100 (0.229), and per-route it wins 3/5 routes (de, es, zh) -- including the two hardest
   cross-only routes (es has no same-lang gold; zh is the thinnest). The recall-only dashboard picks it
   on all five routes.
2. **A per-route router is HEADROOM, not a delivered win.** Present it as consistent with the earlier
   oracle/ensemble headroom story (a router *could* add the en->qwen3-0.6B and fr->nomic-v2-moe corners),
   NOT as "routing beats egemma for free."
3. **Do NOT overstate routing.** Route corners are estimated on thin per-language samples
   (de n_same=7, zh n_same=2, es n_same=0; cross-side n=22-34). Attach the thin-n caveat to any
   route-specific claim; do NOT promote a per-route router to a headline result or deployed recommendation.
   The defensible spine is: ROBUST CLIR@10_ell axis + frontier membership + the *existence* of corner
   movement; the per-route XRC y-axis is explicitly INDICATIVE and must be labeled so.
4. **Suggested one-liner:** "A single model (embeddinggemma) is the right default -- it is the global
   capability corner, owns the lowest irreducible alignment floor, and leads 3 of 5 query-language routes
   including the hardest. A per-route router that swaps in qwen3-0.6B for en and nomic-v2-moe for fr is a
   plausible upside, but on our per-language sample sizes (n_same as low as 2) we report it as headroom
   rather than a recommendation."

## Discrepancies / unverifiable claims
- **None material.** Every numeric claim in implement_report.md reproduces from the output files within
  rounding. One cosmetic note: the implementer's inline per-model ARI list is narrated out of recall
  order; values are all correct -- writer should pull from ari_decomposition.csv, not the prose. The
  implementer's first-run "17 CHECK FAILURES" were a 1e-9-vs-rounding tolerance artifact (curve stores
  6 decimals, knee 4), self-diagnosed and fixed to 1e-3; underlying numbers and identity closure were
  correct on run 1. No fabrication, no overwrite of any existing CSV/figure/paper number (the two new
  figures are untracked additions; no prior figure touched).

## Changed files this round (git diff --stat summary)
- paper/main.tex -- +218 / -80 (296 lines). NOTE: this is WRITER prose/integration work, not part of the
  three DO-NOW compute items; the round-3 implementer claims only CPU artifacts + 2 figures. Story/writer
  next round should reconcile what main.tex already absorbed vs. what still needs cp_fig22/23 wired in.
- paper/loop/needs_eval.md -- +1/-1: single cross-reference appended to the existing
  W3-alignment-causal-probe (r2) entry noting it == dreamer F7 / novelty "route 2", and that cp_fig22's
  L_inf floor is its natural before/after target. No NEW backlog item added.
- Untracked (new this round):
  - reports/runs/chem_patents/experimental_codes/extra_ari_decomposition.py
  - reports/runs/chem_patents/experimental_codes/extra_per_route_frontier.py
  - reports/runs/chem_patents/experimental_plots/extra_ari_decomposition/ (csv, json, png)
  - reports/runs/chem_patents/experimental_plots/extra_per_route_frontier/ (4 csv/json + png)
  - paper/figures/cp_fig22_ari_decomposition.png, paper/figures/cp_fig23_per_route_frontier.png

## Backlogged (forthcoming) experiments to mention as pending
(From needs_eval.md -- all GPU/human-eval, treated as DONE per critic contract; mention as forthcoming.)
- W3-alignment-causal-probe (r2) -- fit a per-language alignment map on one model, re-embed/re-retrieve,
  recompute XRC50 + RRC@100 before/after. Now explicitly == dreamer F7 / novelty "route 2"; cp_fig22's
  per-model L_inf floor is its target metric. UPSIDE ONLY -- paper must not depend on it.
- equivalence-audit-spotcheck (r1) -- expert spot-check that parallel human-translated golds are
  claim-level equivalent (== dreamer W4 "equivalence-audit-lite"). Needs human annotation.
- XRC-conformal-M2 (r1) -- split-conformal XRC with finite-sample coverage; deferred (only 57
  same-lang-gold queries -> calibration/test split too thin).
- CCI-hop-distance-law (r1) -- ChEBI hop-distance vs confusion-rate decay law; CPU but needs non-trivial
  graph build, deferred.

## Recommended next-round focus (for the story architect)
1. Wire cp_fig22 + cp_fig23 into the narrative and reconcile main.tex (already +218 lines this round) --
   confirm ARI decomposition and per-route frontier are cited, and that C3 (alignment floor) and C4
   (single-model deployment) read consistently with the verified numbers above.
2. Lock the C4 framing using the 4-point "single model default + router-as-headroom" guidance above. This
   is the highest-risk-of-overstatement section -- no draft should promote routing to a recommendation or
   drop the thin-n caveat (de=7, zh=2, es=0 same-lang gold).
3. Tie the new L_inf floor to the W3 probe as the paper's "next lever" -- the alignment-only floor is the
   quantity the forthcoming causal probe aims to move; frame as motivated next experiment, not a gap.
