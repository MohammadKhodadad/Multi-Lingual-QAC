# Troubleshooting plan (round 3)

Bridge from critics+dreamer to the implementer. **Read-only verification done.** The
paper is converging (0 correctness mismatches, novelty defensible, cohesion near-clean),
so this round is: (1) land the two CPU-only ceiling-raisers the critics+dreamer both
elevated (**A1 per-route cost frontier**, **M2 ARI**), and (2) close the three named
writer fixes (F1/F2/F3). Everything below is grounded in the actual scripts and on-disk
CSVs — feasibility is **verified**, not assumed.

**Environment (verified):** `.venv/bin/python` has numpy 2.4.3 / pandas 3.0.1 / scipy
1.17.1 / matplotlib 3.10.9. The `extra_*` scripts are **standalone** (NOT in `run_all.py`;
run each directly). They import `common.py` via `sys.path.insert`. All loads are
`HF_HUB_OFFLINE=1` (no network, no API, no embedding). Figure naming convention is
`paper/figures/cp_figNN_name.png`.

---

## Verification of the dreamer's top-2 (the conductor's explicit ask)

### A1 — per-route cost frontier: COMPUTABLE from existing data, with one quantified caveat
- **x-axis CLIR@10ℓ — fully robust.** `common.core_per_query()` emits one row per
  (model, query) with `query_language` and per-query `clir_at_10`. Grouping by
  `query_language` over the cross-gold domain (`n_gold_cross>0`) gives CLIR@10ℓ with
  healthy cell sizes: **de=27, en=27, es=34, fr=27, zh=22** cross-gold queries per route
  (verified). e.g. egemma CLIR@10 by route = {de 0.636, en 0.407, es 0.544, fr 0.370,
  zh 0.553}. No new data needed.
- **y-axis XRC50ℓ — computable but THIN/CENSORED on 3 of 5 routes (the load-bearing
  caveat).** XRC = D50(cross)/D50(same). The cross numerator is fine (22–34/route). The
  **same-language denominator** exists only for the 57 originals and is sparse per route:
  **en=21, fr=27, de=7, zh=2, es=0** (verified). Consequences the implementer MUST honor:
  - **es route has NO XRC axis** (zero same-language gold — the "homeless" route). Its
    panel carries **only** the CLIR coordinate; do not fabricate a y-value.
  - **zh denominator = 2 queries** → D50 is a 2-sample median; **de = 7** → fragile.
    Only **en (21)** and **fr (27)** give a credible XRC50ℓ.
  - The existing `xrc_per_language.csv` (emitted by `extra_xrc_reading_cost.py`, lines
    156–171) already proves the split is mechanically valid — but it uses **D95**, which
    is **mostly blank/censored** at these cell sizes (most XRC95ℓ cells are empty). A1
    must use **D50** (not D95), carry an explicit per-cell `n_same` + censored flag, and
    frame per-route XRC **descriptively / indicative** (the dreamer's and Limitations'
    own caveat, now quantified).
  - **VERDICT: A1 is feasible-now, CPU-only, no new embedding runs** — but it is a
    *CLIR-route frontier with a thin XRC axis*, not a clean 5×5 directed-pair matrix.
    Build it as **per query-language** (5 routes), not per directed pair (a 5×4 XRC
    matrix would be far too thin to read). The headline object survives: CLIR@10ℓ moves
    a lot across routes (0.37–0.64 for egemma alone), so "the capability corner is
    route-dependent" is demonstrable from the robust axis even if XRC is indicative.

### M2 — ARI (Alignment Recoverability Index): COMPUTABLE, trivially, no new data
- **Three reads off an existing curve.** `extra_rrc_budget_frontier/rrc_curve.csv` carries
  RRC at every K∈{1,2,3,5,10,20,30,50,75,100,150,200,300,500,750,1000} for all 9 models
  (verified on disk); `rrc_knee.csv` carries RRC@100, RRC@1000, L_inf, K_star per model.
  The decomposition at a chosen budget K is exactly:
  `RRC@K (recoverable-cheaply) + (RRC@1000 − RRC@K) (recoverable-deeply) + L_inf
  (alignment-only) = 1` — all three terms are direct lookups in `rrc_curve.csv`.
  ARI = `L_inf / (1 − RRC@K)`.
- **VERDICT: M2 is feasible-now, CPU-only, no embedding runs, no API.** It re-reads the
  RRC curve the round-2 work already emitted; zero risk to existing numbers (additive
  new dir + new figure). This is the lowest-risk, highest-thesis-payoff item on the board.

Both top-2 confirmed: **per-query-language splitting of CLIR exists; cross-gold-depth +
score data exist to split XRC by query language (with the thin-denominator caveat); and
the RRC/first_cross_rank curve fully supports the K*/1000/L∞ ARI decomposition.**

---

## DO-NOW (ordered) — each: goal / files / commands / inputs(exist?) / outputs / runtime / verify / api-cost

### DO-NOW-1 — M2: ARI alignment-recoverability decomposition (stacked bar + table)
- **Goal.** Operationalize "align, don't re-rank" as a *measured per-model split*:
  recoverable-cheaply (RRC@K*) / recoverable-deeply (RRC@1000 − RRC@K*) / alignment-only
  (L_inf), plus the scalar ARI = L_inf/(1−RRC@K*). Closes novelty C3/C2 (turns the L∞
  adjective into a decomposition); the paper's single most self-explanatory new figure.
- **Files.** New script `reports/runs/chem_patents/experimental_codes/extra_ari_decomposition.py`
  (mirror `extra_rrc_budget_frontier.py`'s I/O scaffold + `import common as C`).
- **Exact command.**
  `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_ari_decomposition.py`
- **Inputs (exist? YES).** `reports/runs/chem_patents/experimental_plots/extra_rrc_budget_frontier/rrc_curve.csv`
  (RRC at all K) and `.../rrc_knee.csv` (RRC@100, RRC@1000, L_inf, K_star, degenerate_clir).
  Read these CSVs directly — do NOT recompute from parquet (keeps it a pure transform and
  auto-consistent with cp_fig19).
- **Method detail (make the K choice explicit and defensible).** Use a single **stated
  budget K=100** (the "realistic top-100 re-ranker" the paper already names) as the primary
  split point, AND report the same triple at the per-model **K=K*** (knee) as a second
  column so the decomposition is shown at both the practical and the knee budgets. For
  each model: cheap=RRC@100; deep=RRC@1000−RRC@100; floor=L_inf; ARI=L_inf/(1−RRC@100).
  Restrict the figure to the **7 non-degenerate models** (exclude {gte-base,
  e5-large-instruct} via the `degenerate_clir` column already in `rrc_knee.csv`); a
  hostile reviewer must not see e5/gte inside a "non-degenerate" object (this is exactly
  the P1/F1 discipline). Optionally show e5 as a separately-labeled "degenerate
  illustration" bar.
- **Outputs.** `reports/runs/chem_patents/experimental_plots/extra_ari_decomposition/`
  → `ari_decomposition.csv` (model, RRC@100, deep, L_inf, ARI@100, RRC@K*, ARI@K*),
  `summary.json` (egemma triple + ARI; pooled), `ari_decomposition.png`
  (stacked bar, 7 non-deg models, segments cheap/deep/floor). Paper figure target:
  `cp_fig22_ari_decomposition.png`.
- **Runtime.** < 5 s (two small CSV reads + one plot).
- **Verify.** `summary.json` must satisfy `RRC@100 + deep + L_inf == 1.0` per model
  (±1e-6); egemma must read RRC@100=0.7445, L_inf=0.0584 → cheap 0.7445 / deep 0.1971 /
  floor 0.0584, ARI@100 = 0.0584/0.2555 = **0.229**. Cross-check every RRC@100 / RRC@1000
  against `rrc_knee.csv` (they must be byte-identical since it reads them).
- **API cost.** 0.

### DO-NOW-2 — A1: per-route (per query-language) cost frontier + route-membership table
- **Goal.** Convert the global cost frontier into a **per-route deployment map**: recompute
  CLIR@10ℓ (robust) and XRC50ℓ (indicative, thin) per query language, compute the Pareto
  frontier per route, and report **frontier membership by route**. Closes novelty #1
  over-claim ("standard Pareto plot") and earns the route-dependent-corner claim that F4
  only disclaims. Pair with the W1 decision-flip count (DO-NOW-3).
- **Files.** New script
  `reports/runs/chem_patents/experimental_codes/extra_per_route_frontier.py` reusing the
  exact machinery already written: `pareto_frontier()` and `dominators()` from
  `extra_cost_frontier.py` (copy or import), `pct_depth()` from `extra_xrc_reading_cost.py`
  for the censored-median, and `common.core_per_query()` for per-query CLIR + the
  per-query `first_cross_rank`/same-depth split.
- **Exact command.**
  `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python reports/runs/chem_patents/experimental_codes/extra_per_route_frontier.py`
- **Inputs (exist? YES).** `core_per_query()` (built from the 9 on-disk
  `scored_rankings.parquet`, schema verified: model/query_id/query_language/rank/
  corpus_id/corpus_language/score/relevance). The per-route same-language depths must be
  rebuilt the same way `extra_xrc_reading_cost.py` does it (lines 80–90): for each query
  with a same-language gold, `first_gold_rank` over the same set, grouped by
  `query_language`.
- **Method detail (HARD REQUIREMENTS — these protect the paper).**
  1. **Routes = query languages** {en, de, es, fr, zh}; NOT directed pairs (too thin).
  2. **x = CLIR@10ℓ** = mean per-query `clir_at_10` over cross-gold queries in route ℓ.
     Robust (n=22–34). This is the load-bearing axis.
  3. **y = XRC50ℓ** = `pct_depth(cross_ℓ, 50)/pct_depth(same_ℓ, 50)` using the **median
     (D50)**, NOT D95. Emit `n_same_ℓ` and a `xrc_censored_ℓ` flag with EVERY value.
  4. **es route: XRC undefined** (n_same=0). Plot es with its CLIR coordinate only and an
     explicit "no same-language gold → reading-cost undefined" annotation. Do NOT impute.
  5. **zh (n_same=2), de (n_same=7): mark `thin_denominator=True`.** Report their XRC50ℓ
     but caption it as indicative.
  6. **Frame the whole y-axis as "indicative per-route"** in the figure title + caption
     (matches the Limitations thinness caveat and the directional-matrix precedent). The
     CLAIM the figure earns is the **route-dependent capability corner** read off the
     robust CLIR axis + frontier membership, not a precise per-route XRC number.
- **Outputs.** `reports/runs/chem_patents/experimental_plots/extra_per_route_frontier/`
  → `per_route_frontier.csv` (route, model, CLIR@10ℓ, XRC50ℓ, n_cross_ℓ, n_same_ℓ,
  xrc_censored_ℓ, on_frontier_ℓ, dominated_by_ℓ), `frontier_membership_by_route.csv`
  (route → frontier members; the headline table), `summary.json` (per-route corner model,
  whether the max-CLIR corner changes across routes), `per_route_frontier.png`
  (small-multiples 5-panel grid OR one panel with arrows; small-multiples is safer to
  read). Paper figure target: `cp_fig23_per_route_frontier.png`.
- **Runtime.** ~30–60 s (one `core_per_query()` build — it loads ~1.23M ranking rows — then
  pure aggregation; `core_per_query()` is `lru_cache`d).
- **Verify.** (a) Pooling all routes must reproduce the global numbers: pooled CLIR@10 per
  model == `cost_frontier.csv` clir_at_10 (egemma 0.5024); pooled XRC50 == global XRC50
  (egemma 3.5). (b) Per-route cross-gold counts must equal {de27,en27,es34,fr27,zh22}.
  (c) es XRC50ℓ must be NaN/blank, never a number. (d) The global Pareto set
  {egemma,bge-m3,granite} must appear as the **union-ish** baseline; report explicitly
  whether the max-CLIR corner is the same model on every route (the novelty hook).
- **API cost.** 0.

### DO-NOW-3 — W1: decision-flip count (per route, recall-only vs frontier choice)
- **Goal.** Turn the paper's thesis ("the recall dashboard is wrong") into a counted
  per-route fact: for each route ℓ, which model a **recall-only** dashboard picks (argmax
  recall@10ℓ) vs which the **CLIR/frontier** picks (argmax CLIR@10ℓ on-frontier), and
  count the routes where they disagree. Tiny add-on to A1; high rhetorical payoff.
- **Files.** Fold into `extra_per_route_frontier.py` (one extra block + one CSV) — do NOT
  spin a separate script.
- **Inputs (exist? YES).** Same `core_per_query()` — it carries per-query `recall_at_10`
  (overall) AND `clir_at_10` (cross). Group both by `query_language`.
- **Outputs.** `decision_flip_by_route.csv` (route, recall_only_pick, frontier_pick,
  flipped[bool]) + a `n_routes_flipped` field in A1's `summary.json`.
- **Runtime.** negligible (reuses the A1 aggregation).
- **Verify.** Count is in [0,5]; each row names two real model shorts; the "frontier_pick"
  must be a model that is on that route's frontier in `per_route_frontier.csv`.
- **API cost.** 0.

> **Sequencing:** DO-NOW-1 (M2) first — it is the safest and most thesis-central, pure CSV
> transform. Then DO-NOW-2 (A1) + DO-NOW-3 (W1) together (W1 rides on A1's frame). All three
> write to **new** `extra_*` dirs and **new** `cp_fig22/23` names — they cannot perturb any
> existing figure, CSV, or paper number.

---

## BACKLOG-EVAL (exact commands + rationale for needs_eval.md)

Nothing NEW needs adding this round — the dreamer's only `needs-eval` items (**F7** = the
alignment causal probe; **W4** = equivalence-audit-lite via corpus embeddings) are **already
in `needs_eval.md`** as `W3-alignment-causal-probe` (r2) and `equivalence-audit-spotcheck`
(r1) respectively. Per the critic contract these are treated as DONE; do not re-plan them.
The novelty critic's "route 2" (causal alignment probe) maps to the same `W3-alignment-causal-probe`
entry. **No new backlog append is required.** (If the implementer wants the label to match
the dreamer's F7 framing, it may add a one-line cross-reference, but it is not necessary.)

---

## WRITER-ONLY (reframes/citations to pass forward — no computation)

These are pure prose/bib edits for next round's writer. **No numbers change.**

- **F1 (cohesion #1, the round's must-fix).** Relabel the two "**eight** non-degenerate
  models" sites so they drop the gate-bound term (the DEG gate reserves "non-degenerate"
  for the precise 7-set; the count 8 is correct for these all-but-`gte` populations, the
  *word* is wrong):
  - line ~557 (directional-hub pooling): → "pooled over the **eight models with a defined
    cross-lingual recall** (all but the degenerate `gte-base`), French (0.375) and English
    (0.367) …".
  - line ~657 (cp_fig19 caption): → "… for the **eight models with a defined RRC curve**
    (all but `gte-base`, whose candidate pool is empty) …".
  ~6 words each, no figure/number change.
- **F2 (correctness C-NEW).** Fig.21 caption/prose: delete "**sibling-**". The plotted/quoted
  quantity is the general `confusion_rate` (granite 0.182, egemma 0.068 — correct values),
  NOT `sibling_win_rate` (granite 0.144). Caption → "(confusion rate, alias-graph
  benchmark)". Leave the separate, correctly-labeled sibling-vs-parent severity split
  (18.1% vs 6.2%, §ssec:ag) untouched. **Verified against `severity_split.csv`:**
  confusion_rate column = 0.0682/0.1818 for egemma/granite (matches paper); sibling_win_rate
  = 0.0606/0.1439 (different) — so the figure's own axis label contradicts "sibling-".
- **F3 (novelty #2, near-mandatory cite).** In §2 (the RRC/related-ranking paragraph) add
  ONE cascade / re-rank-depth / recall-ceiling reference + one sentence crediting the knee
  shape to prior art and claiming only the cross-lingual quantification + L∞:
  *"The knee and diminishing-returns-with-depth shape is the established cascade /
  re-rank-depth result \citep{<cascade>}; our contribution is its cross-lingual
  quantification and the structural floor L∞."* Prefer a **peer-reviewed multi-stage /
  cascade dense-retrieval paper that states the first-stage recall ceiling** over the
  Elastic semantic-reranker blog (the blog is the named fallback if a venue-acceptable
  peer-reviewed cite isn't found). Candidates to source: a two-stage retrieve-and-rerank /
  cascade-ranking paper (e.g. Matveeva-style cascade ranking, or a multi-stage BERT
  re-ranking paper that reports recall@first-stage-depth saturation). This is the single
  citation that converts the knee from "uncredited" to "credited-and-extended."

**Also pass forward (the dreamer's optional polish, writer's discretion — all
paper-framing-only, no compute):** F4 (add "standard Pareto frame, XRC is the new axis"
half-clause to the C2 bullet / Fig.18 caption; optional syftr arXiv:2505.20266 or RAG
cost-frontier arXiv:2511.09545 cite — **now partly earned by A1**, so F4 can soften from
a disclaimer to "and we resolve it per route, §<A1>"); F5 (surface the inferential step
behind "alignment-only floor" once: "a floor no re-ranker can move; only representation
alignment can, per §7"); F6 (relabel e5 as the *degenerate illustration* at the L∞-range
line ~649; universal-blind list → "French and Chinese" dropping the de/es 3-tie — **verified**
in the correctness review's A8 source; home-advantage hyphenation harmonize to unhyphenated;
cp_fig19 float order cosmetic, do not spend a rewrite).

---

## Round API budget plan (target 0, cap 20)

**Total planned API calls: 0.** All three DO-NOW items are CPU-only transforms/aggregations
of on-disk CSVs and parquet rankings with `HF_HUB_OFFLINE=1`. No `--evaluate-mteb`, no LLM,
no network. The WRITER-ONLY items are prose/bib edits (the F3 citation is sourced by the
writer's own knowledge / a single optional web lookup, not an API spend in the budget sense).

---

## Risks & sequencing notes

- **A1's biggest risk is over-reading the thin XRC axis.** The implementer MUST emit per-cell
  `n_same` + censored flags and frame XRC50ℓ as indicative; es is CLIR-only (no denominator),
  zh (n=2) and de (n=7) are fragile. The *defensible* A1 claim rides on the robust CLIR@10ℓ
  axis (n=22–34/route) + frontier membership, exactly as the directional-CLIR matrix is
  framed. If XRC50ℓ looks too noisy to plot, fall back to plotting CLIR@10ℓ frontier
  membership alone + an XRC50ℓ table with censoring flags — the route-dependent-corner claim
  survives either way.
- **Additivity protects the paper.** All three DO-NOW items write to NEW `extra_*` dirs and
  NEW `cp_fig22/23` names; they read existing CSVs/parquet but never overwrite them. Zero
  risk to the 81 leaderboard cells, the cost frontier, the RRC budget, the DEG gate, or any
  traced number. Re-runnable and idempotent (each script `round_dir()`-creates its own dir).
- **M2 is the safe anchor; run it first.** It is a two-CSV read with a closed-form identity
  check (cheap+deep+floor=1); if it passes, the thesis-central figure is banked before the
  riskier A1.
- **Do not touch the verified-correct numbers.** Correctness review lists everything to leave
  alone (all round-3 NEW numbers trace exactly). The only sanctioned edits are F1/F2/F3 (+
  optional F4/F5/F6), and the two NEW additive figures.
- **`extra_*` scripts are NOT auto-run by `run_all.py`** — the implementer invokes the two
  new scripts directly with the exact venv command above; no run_all.py edit is needed
  (consistent with how the other 7 `extra_*` scripts are wired).
