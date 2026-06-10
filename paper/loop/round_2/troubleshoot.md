# Troubleshooting plan (round 2)

Bridge from the round-2 critics + dreamer to the implementer's marching orders. I verified
feasibility read-only against the on-disk data. **Headline: the dreamer's Top-3 are all
CPU-only and computable from data already on disk — 0 new embedding runs, 0 API calls.** The
three deliver: (1) the cross-lingual cost **frontier** (fixes the only hard MISMATCH, N2, and
gives the paper a new deployment object), (2) the **RRC budget frontier** (the novelty critic's
#1 upgrade), and (3) the **two-tax non-redundancy** measurement **+ a stated DEG gate** (closes
cohesion #3/#4 and hardens N2). Everything else is ~5 one-line WRITER-ONLY critic fixes.

## Feasibility verification (read-only, done this round)

| needed input | file confirmed on disk | columns/keys confirmed |
|---|---|---|
| XRC50 + CLIR@10 per model (A1) | `reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/xrc_per_model.csv` | `short`, `XRC50`, `XRC50_censored`, `clir_at_10`, `D50_same`, `D50_cross` (all 9 models) |
| mate-hit@K / RRC@K per model (A4, DEG) | same dir `rrc_per_model.csv` | `short`, `RRC_at_100`, `RRC_at_1000`, `lost_at_1000` |
| raw per-query first-foreign rank (A4 curve, A7 paired) | `reports/runs/chem_patents/parts/<9 models>/retrieval_results/scored_rankings.parquet` (all **9 present**) via `common.core_per_query()` → column `first_cross_rank` (per model×query, 137 cross-gold rows/model) | reads local HF cache only; `HF_HUB_OFFLINE=1` already set in `common.py`; **no network, no embedding** |
| confusability tax per model (A5) | `reports/runs/alias_graph/experimental_plots/extra_confusion_severity/severity_split.csv` | `short`, `confusion_rate`, `sibling_win_rate`, `sibling_to_parent_ratio` (all 9) |
| join key chem↔alias | both `common.py` `SHORT` maps | identical `short` labels (embeddinggemma, bge-m3, … gte-base) → trivial join |

The XRC generator (`extra_xrc_reading_cost.py`) already computes RRC@K as `np.mean(cross <= K)`
over `cpq["first_cross_rank"]`; re-using that same vector on a K-grid yields the full RRC(K)
curve — confirmed in code (lines 96–145). **No re-eval needed for any Top-3 item.**

---

## DO-NOW (ordered) — each: goal / files / exact commands / inputs(exist?) / outputs / runtime / verify / api-cost

> All DO-NOW scripts are **new additive `extra_*` scripts** that write to **new** plot dirs and
> never touch the 10 rounds or existing `extra_*` outputs (same isolation discipline the round-1
> `extra_*` scripts already follow). Run each with the repo venv from repo root:
> `/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python <script>`. Each imports the existing
> `common as C` (so `sys.path.insert(0, parent)` then `import common as C`).

### DO-NOW-1 — A1+M2+W2: the cross-lingual **cost frontier** (the round; closes N2, upgrades C4) [api: 0]
- **goal.** Replace the false superlative "lowest non-degenerate reading cost (egemma)" with a
  *true frontier* claim: plot every model in (x = CLIR@10 ↑, y = XRC50 ↓) space, compute the
  **Pareto frontier** (minimize XRC50, maximize CLIR@10), mark on-frontier vs dominated, and
  emit the W2 inversion number (Spearman ρ between XRC50 and CLIR@10 across the **non-degenerate**
  models — the "cheapest reader may be the worst retriever" trap).
- **new file.** `reports/runs/chem_patents/experimental_codes/extra_cost_frontier.py`
  (`SLUG = "extra_cost_frontier"`).
- **inputs (exist?).** `extra_xrc_reading_cost/xrc_per_model.csv` — YES (read directly with
  pandas; do **not** recompute XRC). Columns `XRC50`, `XRC50_censored`, `clir_at_10`, `short`,
  `D50_cross`. Optionally read `severity_split.csv` only if you want marker styling — not required.
- **method (exact).**
  1. Load `xrc_per_model.csv`. Coerce `XRC50` to float; gte has blank XRC50 + `XRC50_censored=True`
     → treat as **degenerate / off-frontier**, plot at top with a censored marker (▼) but
     **exclude from the Pareto sweep**.
  2. Pareto set over finite-XRC models: model *m* is on-frontier iff no other finite model has
     `CLIR@10 ≥ m.CLIR@10` **and** `XRC50 ≤ m.XRC50` with at least one strict. (Standard 2-obj
     non-dominated sweep: sort by CLIR@10 desc, walk keeping running-min XRC50.)
  3. Compute **W2 trap correlation**: Spearman ρ(`XRC50`, `clir_at_10`) over the **non-degenerate
     set** (apply the DEG gate from DO-NOW-3; if DO-NOW-3 not yet run, use the explicit list
     `{gte-base, e5-large-instruct}` excluded). Report sign + value. Expected positive-ish (cheap
     XRC co-occurs with low CLIR among the weak tail) — that is the quotable trap.
  4. Emit the deployment read-off string: among models with `CLIR@10 ≥ τ` (set τ = 0.40, a stated
     threshold that admits embeddinggemma/bge-m3/qwen3), report the min-XRC50 model. **From the
     data: at τ=0.40 the admitted set is {embeddinggemma 0.50/3.5, bge-m3 0.437/2.0, qwen3
     0.433/3.25}; bge-m3 has the lower XRC50, so do NOT claim egemma is cheapest — instead state
     egemma is the unique top-right Pareto point (max CLIR@10) and is on the frontier.** Verify the
     Pareto membership in code and let the script print who is on-frontier; the writer phrases the
     claim from the script's output, not from a guess.
- **outputs.** `reports/runs/chem_patents/experimental_plots/extra_cost_frontier/`:
  `cost_frontier.csv` (per model: XRC50, CLIR@10, on_frontier bool, dominated_by list),
  `summary.json` (frontier members, τ-admitted set + min-XRC within it, W2 Spearman ρ + sign),
  `cost_frontier.png` (scatter, frontier line drawn through non-dominated points, egemma + granite
  annotated, degenerate models flagged). Figure target name for the paper: **`cp_fig18_cost_frontier.png`** (copy into `paper/figures/` only when the writer wires it).
- **runtime.** < 5 s (pure pandas + one matplotlib panel; no parquet load).
- **verify.** `summary.json` frontier must include `embeddinggemma` and `granite-278m` (granite is
  the XRC-cheapest finite point, egemma the CLIR-richest → both non-dominated; the *interior*
  finite models bge-m3/qwen3 may or may not be dominated — let the sweep decide and assert egemma
  ∈ frontier). Cross-check XRC50/CLIR@10 values printed equal the CSV (egemma 3.5/0.5024, granite
  1.25/0.3285). The W2 ρ sign must be **positive** over the non-degenerate set or the trap framing
  is wrong — if it comes back negative, the writer must NOT use the trap sentence (flag in summary).
- **closes.** P1, **N2** (the only MISMATCH), upgrades C4, delivers M2 + W2.

### DO-NOW-2 — A4+M3 (C-RRC): the **re-ranker-budget frontier** (RRC curve + knee K* + L∞) [api: 0]
- **goal.** Turn RRC from two scalars into a **curve** per model, with (i) marginal recoverability
  dRRC/dK, (ii) the **knee** K* (max distance-to-chord elbow), (iii) the **structural floor**
  L∞ = 1 − RRC@K_max. This is the novelty critic's explicitly-named "highest-leverage upgrade."
- **new file.** `reports/runs/chem_patents/experimental_codes/extra_rrc_budget_frontier.py`
  (`SLUG = "extra_rrc_budget_frontier"`).
- **inputs (exist?).** Raw first-foreign ranks via `C.core_per_query()` → `first_cross_rank` for
  rows with `n_gold_cross > 0` (137/model). Backing parquet: all 9 present — YES. **No new eval.**
  (Sanity: `rrc_per_model.csv` RRC@100/RRC@1000 must reproduce exactly from the curve at K=100/1000
  — use it as the regression check, do not re-read it as a source.)
- **method (exact).**
  1. `cpq = C.core_per_query()`; for each model `m`, `cross = cpq[(cpq.model==m) &
     (cpq.n_gold_cross>0)]["first_cross_rank"].to_numpy(float)` (137 values, may contain inf).
  2. K-grid: `Ks = [1,2,3,5,10,20,30,50,75,100,150,200,300,500,750,1000]` (or a denser log grid).
     `RRC(K) = mean(cross <= K)`. (K_max = 1000 = MAXRANK, the retrieved-list depth.)
  3. **dRRC/dK** = discrete forward difference normalized per-candidate:
     `(RRC(K_{i+1}) − RRC(K_i)) / (K_{i+1} − K_i)`. Report per 100 candidates for readability.
  4. **Knee K\*** = argmax perpendicular distance from each `(K, RRC(K))` point (normalize K to
     [0,1] via K/1000 or log10(K)/log10(1000) — **use log10-K normalization** since RRC saturates
     fast; state which in summary) to the chord from `(K_min,RRC_min)` to `(K_max,RRC_max)`.
     Report K* and RRC(K*). (Kneedle is overkill; the max-distance-to-chord elbow is standard and
     deterministic — no extra dependency.)
  5. **L∞(m)** = `1 − RRC(1000)` = `lost_at_1000` (cross-check vs `rrc_per_model.csv`).
- **outputs.** `reports/runs/chem_patents/experimental_plots/extra_rrc_budget_frontier/`:
  `rrc_curve.csv` (long: model, K, RRC, dRRC_per100), `rrc_knee.csv` (model, K_star, RRC_at_Kstar,
  L_inf, RRC_at_100, RRC_at_1000), `summary.json` (egemma K*, RRC@K*, L∞ 0.0584; e5 L∞ 0.3723),
  `rrc_budget_frontier.png` (RRC(K) curves per non-degenerate model on a log-K x-axis, knee marked
  per curve, L∞ floor shaded). Figure target name: **`cp_fig19_rrc_budget.png`**.
  Optionally also a small 2-D panel (x = XRC50 reading-depth cost, y = L∞ structural floor) per
  model — the "single planning object" the dreamer/novelty critic describe; emit as
  `rrc_xrc_plane.png` if cheap.
- **runtime.** ~30–60 s (one `core_per_query()` build ≈ loads 9 parquet ≈ 1.2M rows; it is
  `lru_cache`d). Acceptable; it is the same load every round script does.
- **verify.** Assert `RRC(100)` and `RRC(1000)` from the curve equal `rrc_per_model.csv` to 1e-6
  for all 9 models (egemma 0.7445 / 0.9416; e5 0.2847 / 0.6277). Assert L∞ == `lost_at_1000`.
  Assert RRC monotone non-decreasing in K per model.
- **closes.** P2, **novelty #1**, delivers M3. Pair in prose with the one-clause honesty hedge
  (WRITER-ONLY W-NOV1 below) so the *current* claim is honest and the *curve* is the novelty.

### DO-NOW-3 — A3+M1 (C-DEG) + A5+M4 (C-P4): the **DEG gate** and the **two-tax non-redundancy** [api: 0]
- **goal.** (a) Define "degenerate" **operationally** and apply the gate so the 4 load-bearing
  "non-degenerate" uses + the WTA footnote are anchored; (b) show the **two taxes are
  non-redundant** across models (low rank correlation) → both benchmarks are necessary, not padded.
- **new file.** `reports/runs/chem_patents/experimental_codes/extra_two_tax_degeneracy.py`
  (`SLUG = "extra_two_tax_degeneracy"`). *(Lives under chem_patents because the join pulls the
  chem XRC/RRC tables; it reads the alias CSV by absolute/relative path — no alias `common`
  import needed, avoiding a cross-package import.)*
- **inputs (exist?).**
  - chem `extra_xrc_reading_cost/xrc_per_model.csv` (`clir_at_10`, `XRC50`, `short`) — YES.
  - chem `extra_xrc_reading_cost/rrc_per_model.csv` (`RRC_at_1000` = mate-hit@1000, `short`) — YES.
  - alias `extra_confusion_severity/severity_split.csv` (`confusion_rate`, `sibling_win_rate`,
    `short`) — YES. Path:
    `reports/runs/alias_graph/experimental_plots/extra_confusion_severity/severity_split.csv`.
  - Join key `short` is identical across the three tables (verified). Read all three with pandas.
- **method (exact).**
  1. **DEG gate (M1).** `DEG(m) = (clir_at_10 < 0.10) AND (RRC_at_1000 < 0.10)`. Apply:
     - gte-base: CLIR@10 0.0 (<0.10) AND RRC@1000 0.0876 (<0.10) → **DEG = 1** ✓
     - e5-large-instruct: CLIR@10 0.0766 (<0.10) AND RRC@1000 0.6277 (**≥0.10**) → **DEG = 0** by
       this AND-gate. **IMPORTANT MISMATCH WITH THE DRAFT/DREAMER:** the dreamer's stated gate
       (CLIR@10<0.10 AND <10% twins recovered@1000) **only flags gte, not e5** — but the paper
       excludes *both* gte and e5 from "non-degenerate" summaries. Resolve in code by reporting
       **two columns**: `DEG_strict` (the AND-gate above → only gte) and a documented
       **`DEG_paper`** that also catches e5 via an OR with `clir_at_10 < 0.10` alone (e5 CLIR@10
       0.0766 < 0.10 → flagged). Recommend the writer adopt **`DEG = clir_at_10 < 0.10`** as the
       single clean criterion (it flags exactly {gte, e5} and nothing else — the gap to SapBERT
       at 0.1788 is wide and clean), and keep the mate-hit@1000 floor as a corroborating second
       signal in the caption, not the gate. **Let the script emit both and print the membership so
       the writer picks the one that matches the paper's actual exclusions ({gte, e5}).**
  2. **Degeneracy gap plot (A3).** Bar/strip of `clir_at_10` for all 9 sorted desc, with a dashed
     cutoff line at 0.10; annotate the clean gap (SapBERT 0.179 vs e5 0.077). One inset-sized fig.
  3. **Two-tax table + correlation (A5/M4).** Build a per-model table with **reading-cost tax** =
     `XRC50` (chem) and **confusability tax** = `confusion_rate` (alias). Compute **Spearman rank
     correlation across models**, both (i) all 9 and (ii) **non-degenerate only** (drop {gte,e5} —
     the honest set, mirroring the correlation-robustness discipline already in the paper).
     Report both ρ. The claim "two taxes weakly correlated → non-redundant" must rest on the
     **n=7 non-degenerate** ρ (dropping the collapsed encoders, which co-inflate both taxes and
     would manufacture a spurious correlation). Also emit the scatter (A8): x = XRC50, y =
     confusion_rate, annotate egemma (low/low), e5 (excluded-degenerate), granite (cheap-XRC /
     mid-confusion).
- **outputs.** `reports/runs/chem_patents/experimental_plots/extra_two_tax_degeneracy/`:
  `deg_flags.csv` (short, clir_at_10, RRC_at_1000, DEG_strict, DEG_clir_only, DEG_paper),
  `two_tax_table.csv` (short, reading_cost_tax=XRC50, confusability_tax=confusion_rate, DEG),
  `summary.json` (DEG membership under each rule, Spearman ρ all-9 and n=7 + interpretation),
  `degeneracy_gap.png` (the CLIR@10 bar + 0.10 cutoff), `two_tax_scatter.png` (the A8 panel).
  Figure target names: **`cp_fig20_degeneracy_gap.png`**, **`cp_fig21_two_tax.png`**.
- **runtime.** < 5 s (three small CSVs; no parquet).
- **verify.** DEG_clir_only membership must equal exactly `{gte-base, e5-large-instruct}` (matches
  the paper's actual exclusions). Two-tax ρ on n=7 should be **near zero / weak** (|ρ| well below
  the +0.96 separability correlation) — if it comes back |ρ|>0.6 the "non-redundant" claim is
  unsupported and the writer must soften to the minimal C-P4 sentence (see WRITER-ONLY). Print the
  XRC50 and confusion_rate joined values to confirm the join aligned by `short` (egemma 3.5 /
  0.068; granite 1.25 / 0.182).
- **closes.** P3 (cohesion #4, hardens N2), P4 (cohesion #3), delivers M1 + M4.

### DO-NOW-4 (optional, lower priority) — A7+W4: **paired XRC** + the **no-home Spanish** slice [api: 0]
- **goal.** Answer correctness **T-NEW** (XRC is population-level, not paired) with a *paired*
  within-query XRC on the 57 originals, and showcase the no-home Spanish stress slice (W4).
- **new file.** `reports/runs/chem_patents/experimental_codes/extra_paired_xrc.py`
  (`SLUG = "extra_paired_xrc"`).
- **inputs (exist?).** `C.core_per_query()` gives, per (model, query): `first_cross_rank` and a
  same-language first-gold rank (recompute via `C.first_gold_rank(ranked, same)` exactly as the XRC
  script does at lines 83–90; the 57 originals are those with `n_gold_same>0`). All on disk — YES.
- **method (exact).** Restrict to the **57 original queries that have BOTH a same-language gold and
  a cross-language gold**. Per query: depth-to-same-gold `d_same`, depth-to-first-foreign `d_cross`.
  **Paired XRC** = median over the 57 of the per-query ratio `d_cross/d_same` (drop pairs where
  either is inf, report the censored fraction), AND the ratio-of-paired-medians as a robustness
  variant. Report alongside the population XRC50 (3.5×). For **W4**: the 34 Spanish queries have
  **zero es gold** → no same-language denominator; report only **absolute** D_cross depth on the
  Spanish slice vs the rest (sidesteps the ratio objection entirely).
- **outputs.** `extra_paired_xrc/`: `paired_xrc_per_model.csv` (short, n_pairs, paired_xrc_median,
  ratio_of_medians, censored_frac, population_xrc50), `nohome_spanish.csv` (model, D_cross_es_abs,
  D_cross_rest_abs), `summary.json`, optional `paired_vs_population.png`.
- **runtime.** ~30–60 s (one `core_per_query()` build).
- **verify.** Population XRC50 column must equal `xrc_per_model.csv` (egemma 3.5). Paired egemma
  XRC should be in a plausible range (likely close to 3.5×; if it lands ~2–6× the writer reports
  it as the headline paired number per A7). n_pairs ≤ 57.
- **closes.** P6 / T-NEW (paired upgrade), W4 (no-home showcase). **Optional** — the minimal
  WRITER-ONLY one-clause caveat (W-COR-TNEW below) fully closes T-NEW without this; DO-NOW-4 is the
  *upgrade* that makes XRC a within-query cost like the +0.55 home advantage. Do it only after
  DO-NOW-1..3 land.

---

## BACKLOG-EVAL (exact commands + rationale for needs_eval.md)

Nothing new is *required* this round — the Top-3 are all DO-NOW. The dreamer's two needs-eval
ideas are already in `needs_eval.md`; do **not** re-add. For completeness, map them:

- **M5 gXRC (conformal guaranteed XRC)** → already present as `XRC-conformal-M2` (added r1).
  Rationale unchanged: only 57 same-language-gold queries → calibration/test split too thin for a
  credible distribution-free guarantee. Leave as backlog; the empirical XRC (DO-NOW-1) ships now.
- **W3 alignment causal probe** (fit a cheap cross-lingual linear map on one model, re-embed,
  re-retrieve, show XRC↓/RRC↑) → **append to needs_eval.md** if the implementer wants the causal
  upside, but it requires re-embedding one model → NEW eval. Suggested entry:
  `- [ ] W3-alignment-causal-probe | Take ONE model (e.g. LaBSE), fit a per-language mean-centering / linear alignment map on a held-out parallel slice, re-embed queries+corpus for that model only, re-retrieve over multilingual_GP, and recompute XRC50 + RRC@100 before/after on the SAME 137 cross queries. | Elevates "align, don't re-rank" from correlational (C3) to a demonstrated intervention: the recommended lever measurably moves the cost metrics. Needs re-embedding -> eval. | r2`
  (Do NOT make the paper depend on it; it is upside only.)
- **CLIRMRS-external-validation** already in backlog; the novelty critic's route-2 ("point that
  eval at XRC instead") is a *framing* note for the writer, not a new eval — see W-NOV2 below.

Never propose `--evaluate-mteb` as DO-NOW (it is the GPU re-eval path). None proposed.

---

## WRITER-ONLY (reframes/citations to pass forward — ~5 one-line critic fixes + the hedges)

The five mandatory one-line critic fixes the conductor named (all pure prose, no computation):

- **W-COR-N2 (drop the false superlative).** §Deployment ~line 898–899. Remove "has the **lowest
  non-degenerate reading cost** (XRC50 3.5×)". Replace with the correctness critic's defensible
  line: *"is the best twin-finder (median first-foreign rank 5) and keeps a low reading cost
  (XRC50 3.5×, vs 11.5×–97.75× for nomic/e5)."* If DO-NOW-1 lands, upgrade to the frontier claim:
  *"is the unique Pareto-optimal point in the (CLIR@10, XRC50) cost–capability frontier; the
  lower-XRC models read shallowly only because they retrieve less (granite CLIR@10 0.329 vs
  0.50)."* **This is the floor — do it regardless of whether DO-NOW-1 lands.**
- **W-COR-N1 (soften the directional near-tie).** §Results ~line 509. "French (0.375) and English
  (0.367) are **statistically indistinguishable** as targets" → "are **nearly tied** (fr 0.375 ≈
  en 0.367, within 0.01)". No test was run; do not assert one. (Dreamer A6b bootstrap is optional;
  not requested this round — minimal wording fix suffices.)
- **W-COR-TNEW (XRC population-level caveat).** §6.1 XRC paragraph or the cp_fig15 caption. Add one
  clause: *"D_same is taken over the 57 same-language-gold queries and D_cross over the 137
  cross-gold queries — a population-level, not paired, ratio."* (DO-NOW-4 is the optional paired
  upgrade; this clause alone closes T-NEW.)
- **W-COH-B2 (line 605 "best model" → "any model").** §6.2 ~line 605. Rewrite *"Even the best
  model reaches a cross-lingual RBO of only 0.39"* → *"The best cross-lingual RBO any of the nine
  models reaches is only 0.39 (Figure~\ref{fig:ag_rbo}) — a ceiling no model beats."* Matches the
  abstract/intro/conclusion "any model" phrasing and fixes the rhetorical inversion (0.39 is a
  ceiling, not an achievement). This is the one real internal-framing contradiction left (B2).
- **W-COH-deZH (close the de↔zh orphan).** §6.1 / cp_fig03 caption. Fold the named-but-orphaned
  asymmetry number in: *"…the most asymmetric directed pair is de↔zh (+0.23; asymmetry panel
  cp_fig04)."* Either reference cp_fig04 or fold the number into the cp_fig03 caption — ~6 words,
  closes cohesion #2.

Plus the novelty/cohesion framing hedges that pair with the DO-NOW figures (carry forward; no
computation):

- **W-NOV1 (RRC honesty hedge — pairs with DO-NOW-2).** §4 RRC definition, add one clause:
  *"RRC@K is the cumulative first-foreign-twin hit rate (mate-hit@K on cross-lingual queries); our
  contribution is reading it as a per-model re-ranker ceiling: 1−RRC@K is provably
  unrecoverable."* Then the DO-NOW-2 curve/knee makes it the *budget object*, not the rename.
- **W-NOV2 (XRC monotone-invariance, one line — novelty #2 / P7).** §4 XRC definition, add:
  *"a ratio of retrieval depths is invariant to any monotone re-scaling of similarities, unlike an
  AUC or a weighted composite — XRC measures how deep you read, which no score normalization can
  hide."*
- **W-NOV3 (CLIRMRS-external-validation reframe).** Where future-work mentions CLIRMRS validation,
  note the external-utility eval can instead validate **XRC** (the novelty critic's route 2). One
  sentence; no eval this round.
- **W-COH-P4 (two-tax spine sentence).** Head of §6.2 (or end §6.1): *"If §6.1 measured what
  cross-linguality costs to read, the alias-graph benchmark measures what it costs in precision —
  the second line-item of the same bill: a look-alike compound that out-ranks the gold."* If
  DO-NOW-3 lands, back it with the measured non-redundancy (n=7 Spearman ρ) and Table.
- **W-COH-DEG (define "degenerate" once).** §6.1 first use (~line 534) or §5 Setup: *"We call a
  model degenerate if its cross-lingual recall collapses below CLIR@10 0.10 (gte-base 0.0,
  e5-large-instruct 0.077; it retrieves almost nothing distinctive), and exclude these two from
  'non-degenerate' summaries throughout."* Anchors all 4 uses + the WTA footnote. DO-NOW-3 emits
  the exact membership + the gap figure to back this.
- **W-COR-TMINOR (non-blocking).** §Analysis universal-blind list: "predominantly French, Chinese,
  and German" → "predominantly French and Chinese (German and Spanish tied behind)" (de/es both 3).
- **W-COH-cosmetic (non-blocking).** Harmonize "home advantage" hyphenation to the unhyphenated
  dominant form on a final pass; optionally reorder cp_fig16 float after cp_fig07. Low priority.

---

## Round API budget plan (target 0, cap 20)

**Total API calls this round: 0.** All four DO-NOW items read only on-disk CSVs and the existing
9 local parquet rankings (via `common.core_per_query()`, `HF_HUB_OFFLINE=1`); none calls any model
or LLM. No `--evaluate-mteb`. No `--cs-generate-qa` / variant E. Budget headroom (20) untouched —
keep it that way; the WRITER-ONLY items are prose and consume nothing.

---

## Risks & sequencing notes

1. **Sequence: DO-NOW-1 → DO-NOW-3 → DO-NOW-2 → (optional) DO-NOW-4.** DO-NOW-1 closes the only
   hard MISMATCH (N2) and is the round's headline; do it first. DO-NOW-3 defines the DEG gate that
   DO-NOW-1's W2 correlation and the paper's "non-degenerate" language both depend on — run it
   early so the gate membership ({gte, e5}) is fixed before any prose is written. DO-NOW-2 is the
   biggest novelty win but is independent; it can run in parallel. DO-NOW-4 is an upgrade, last.
2. **DEG-gate mismatch is the one real design decision (flagged, resolved).** The dreamer's literal
   AND-gate (CLIR@10<0.10 AND mate-hit@1000<0.10) flags **only gte**, not e5 — but the paper
   excludes **both**. Resolution baked into DO-NOW-3: emit all three candidate rules and recommend
   the single clean **`DEG = CLIR@10 < 0.10`** (flags exactly {gte, e5}; clean gap to SapBERT
   0.179). The writer must use whichever rule reproduces the paper's actual {gte, e5} exclusions —
   the script prints membership so this is mechanical, not a judgment call. Do **not** ship a gate
   that contradicts the paper's exclusions.
3. **Two claims are data-conditional — the script must gate the prose.** (a) DO-NOW-1's W2 "trap"
   sentence is only valid if Spearman ρ(XRC50, CLIR@10) is **positive** on the non-degenerate set;
   (b) DO-NOW-3's "two taxes non-redundant" is only valid if the n=7 ρ is **weak**. Both scripts
   must print the sign/value and `summary.json` must record it; the writer must read the number
   before writing the claim. If either comes back the wrong way, fall back to the minimal
   WRITER-ONLY sentence and drop the strong claim. This protects the paper from a new overclaim.
4. **τ in DO-NOW-1 is a stated threshold, not a tuned one.** Fix τ=0.40 in the summary and state it
   in the caption. Do not search τ to make egemma "win" — the honest claim is egemma's **Pareto
   membership** (max CLIR@10, undominated), which holds for any τ. Avoid re-introducing a
   superlative ("cheapest among deployable") unless the τ-admitted min-XRC is actually egemma — at
   τ=0.40 it is **bge-m3** (XRC50 2.0 < 3.5), so the writer must phrase via the frontier, not a new
   "cheapest" claim. This is the subtle trap that created N2 in the first place; the plan steers
   around it.
5. **Isolation.** Every new script writes to a **new** `extra_*` dir and reads existing CSVs/parquet
   read-only. No round-1 output, figure, or `key_findings/` file is modified. Zero risk to existing
   results. New paper figures are copied into `paper/figures/` only by the writer when wiring them,
   under the `cp_fig18..21` names reserved above.
6. **Runtime budget.** DO-NOW-1 and DO-NOW-3 are < 5 s each (CSV-only). DO-NOW-2 and DO-NOW-4 each
   pay one `core_per_query()` parquet load (~30–60 s, lru_cached within a process). Total well under
   a few minutes; nothing GPU.
