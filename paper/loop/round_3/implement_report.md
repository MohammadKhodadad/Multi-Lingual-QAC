# Implementation report (round 3)

Executed the three DO-NOW items from `paper/loop/round_3/troubleshoot.md`. All CPU-only,
0 API, `HF_HUB_OFFLINE=1`, writing to NEW `extra_*` dirs and NEW `cp_fig22/23` names. No
existing CSV/figure/paper number was overwritten. WRITER-ONLY items (F1/F2/F3/F4/F5/F6) are
prose/bib edits left for next round's writer — no compute, not touched here.

## Done

### DO-NOW-1 — M2: ARI alignment-recoverability decomposition (stacked bar + table)
- **What.** New standalone script that re-reads the already-emitted RRC curve and decomposes
  each model's cross-lingual shortfall into three exhaustive buckets:
  cheap=RRC@100 / deep=RRC@1000−RRC@100 / floor=L_inf=1−RRC@1000, with the scalar
  ARI@K = L_inf/(1−RRC@K) at both K=100 (primary) and K=K* (per-model knee). Pure two-CSV
  transform (no parquet, no recompute) → auto-consistent with cp_fig19.
- **Command.**
  `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_ari_decomposition.py`
- **Output paths.**
  - `reports/runs/chem_patents/experimental_codes/extra_ari_decomposition.py` (new)
  - `reports/runs/chem_patents/experimental_plots/extra_ari_decomposition/ari_decomposition.csv`
  - `.../extra_ari_decomposition/summary.json`
  - `.../extra_ari_decomposition/ari_decomposition.png`
- **Quick verification (ALL PASS).** Identity cheap+deep+floor == 1.0 closes for ALL 9 models
  (±1e-6) at BOTH K=100 and K=K* (`identity_closes_all_models: true`). egemma reads exactly
  the troubleshoot targets: cheap RRC@100=**0.7445**, deep=**0.1971**, floor L_inf=**0.0584**,
  sum=**1.0**, **ARI@100=0.2286** (≈ the 0.229 target). Every RRC@100/@1000/L_inf cross-checks
  against `rrc_knee.csv` within rounding tolerance (1e-3; `rrc_curve.csv` carries 6-decimal
  precision vs the 4-decimal knee CSV — the curve is the authoritative source).
  Per-model ARI@100 (non-degenerate, in recall order):
  embeddinggemma **0.229**, bge-m3 0.366, qwen3-0.6B 0.233, nomic-v2-moe 0.279,
  granite-278m 0.417, LaBSE 0.349, SapBERT 0.419. (Degenerate, figure-excluded:
  e5-large-instruct 0.520, gte-base 0.912.) Lowest ARI = embeddinggemma → after a cheap top-100
  re-ranker, its residual is the *least* alignment-bound; it also has the smallest L_inf floor (5.8%).
- **api-calls-used: 0.**

### DO-NOW-2 — A1: per-route (per query-language) cost frontier + route-membership table
### DO-NOW-3 — W1: decision-flip count (folded into the same script, one extra block + CSV)
- **What.** New standalone script reusing `pareto_frontier()`/`dominators()` (imported from
  `extra_cost_frontier.py`), `pct_depth()` (imported from `extra_xrc_reading_cost.py`), and
  `common.core_per_query()`. Routes = query languages {en,de,es,fr,zh}. x = CLIR@10_ell (robust,
  n=22–34/route); y = XRC50_ell using the MEDIAN D50 (indicative), carrying n_same + censored +
  thin flags on every value. es has no same-language gold → XRC undefined (NOT imputed). Per-route
  Pareto frontier + frontier membership + W1 decision-flip (argmax recall@10_ell vs argmax
  CLIR@10_ell).
- **Command.**
  `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_per_route_frontier.py`
- **Output paths.**
  - `reports/runs/chem_patents/experimental_codes/extra_per_route_frontier.py` (new)
  - `.../extra_per_route_frontier/per_route_frontier.csv`
  - `.../extra_per_route_frontier/frontier_membership_by_route.csv`
  - `.../extra_per_route_frontier/decision_flip_by_route.csv`
  - `.../extra_per_route_frontier/summary.json`
  - `.../extra_per_route_frontier/per_route_frontier.png`
- **Quick verification (ALL PASS, `checks_passed: true`).**
  - (a) Pooled CLIR@10 per model reproduces `cost_frontier.csv` (egemma 0.5024) for all 9 models.
  - (b) Per-route cross-gold counts == {de27, en27, es34, fr27, zh22}; same-lang == {en21, de7, es0, fr27, zh2}.
  - (c) es XRC50 is NaN for every model (never imputed).
  - Global Pareto set {bge-m3, embeddinggemma, granite-278m} recorded and appears union-ish
    across routes.
  - **Capability corner MOVES across routes: TRUE — 3 distinct max-CLIR corners.**
    en→qwen3-0.6B (0.444), de→embeddinggemma (0.636), es→embeddinggemma (0.544),
    fr→nomic-v2-moe (0.389), zh→embeddinggemma (0.553).
  - **Frontier membership by route:** en={qwen3-0.6B, granite-278m, LaBSE};
    de={embeddinggemma, bge-m3, qwen3-0.6B, granite-278m, LaBSE}; es={embeddinggemma} (CLIR-only,
    1-D); fr={bge-m3, nomic-v2-moe, granite-278m}; zh={embeddinggemma} (thin, n_same=2).
  - **Decision-flip = 2/5 routes.** A recall-only dashboard picks embeddinggemma on EVERY route,
    but the CLIR/frontier pick differs on **en** (→qwen3-0.6B) and **fr** (→nomic-v2-moe);
    de/es/zh do not flip. Both frontier_pick models are on their route's frontier (qwen3-0.6B∈en
    frontier, nomic-v2-moe∈fr frontier) — verify constraint satisfied.
  - es panel renders with explicit "no same-language gold → reading-cost undefined (CLIR-only
    route)" annotation; de (n_same=7) and zh (n_same=2) panels carry "XRC indicative … thin"
    warnings. y-axis framed INDICATIVE in title + per-panel.
- **api-calls-used: 0.**

## Backlogged to needs_eval.md (id + reason + exact command)
None NEW required (per troubleshoot BACKLOG section). The dreamer's only eval items
(F7 alignment causal probe; W4 equivalence-audit-lite) are already present as
`W3-alignment-causal-probe` (r2) and `equivalence-audit-spotcheck` (r1) and are treated as DONE
per the critic contract. Added a single one-line cross-reference to the existing
`W3-alignment-causal-probe` entry noting it == dreamer F7 / novelty "route 2" and that the new
cp_fig22 L_inf floor is the natural before/after target for that probe. No new command added.

## New figures copied to paper/figures/ (basename -> source)
- `cp_fig22_ari_decomposition.png` -> `reports/runs/chem_patents/experimental_plots/extra_ari_decomposition/ari_decomposition.png` (md5 cb373d1e… match)
- `cp_fig23_per_route_frontier.png` -> `reports/runs/chem_patents/experimental_plots/extra_per_route_frontier/per_route_frontier.png` (md5 a36b49a5… match)

## Failures / surprises (verbatim errors, real outcomes)
- First run of `extra_ari_decomposition.py` emitted 17 spurious "CHECK FAILURES" of the form
  `embeddinggemma: RRC@100 curve 0.744526 != knee 0.7445`. ROOT CAUSE: my cross-CSV equality
  tolerance was 1e-9, but `rrc_curve.csv` stores RRC at 6 decimals while `rrc_knee.csv` rounds to
  4 — they differ only by rounding, not value. The numbers themselves (and the identity closure)
  were correct on the first run. FIX: loosened the cross-CSV check to 1e-3 (rounding-level), with
  a code comment that the curve is the authoritative higher-precision source. Re-ran → "all
  identity / cross-CSV checks PASSED", numbers unchanged. No other failures.
- No surprises in DO-NOW-2/3: all verify gates passed on the first run; the parquet load
  (`core_per_query()`) completed in well under a minute.
- Real outcome note (honest framing, not a failure): the per-route XRC50_ell y-axis is genuinely
  noisy where the same-language denominator is thin (e.g. egemma en XRC50=7.5 vs fr=12.0 off
  D50_same=2 in en/fr; granite fr XRC50=1.159; some thin-denominator cells like nomic zh=27.0 are
  driven by an n_same=2 median). This is exactly the troubleshoot's load-bearing caveat — the
  DEFENSIBLE A1 claim rides on the ROBUST CLIR@10_ell axis + frontier membership + the
  route-moving capability corner, not on precise per-route XRC numbers. The figure and CSVs carry
  the flags to keep the y-axis honest.

## API calls used this round: 0/20
