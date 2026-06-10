# Story (round 3)

## Changes since round 2

Round 2 was the *defense-and-honest-demotion* round (it closed the six "you
reinvented X" novelty surfaces, demoted CLIR-MRS to a table-ordering convenience,
and added XRC/RRC). Round 3 is the **consolidation round**: the reporter verified
(byte-identical figures, every number opened against its JSON/CSV) three new
CPU-only / 0-API objects that turn last round's loose ends into first-class,
*defensible* paper objects — but two of the supporting correlations are
**non-significant**, so this round's whole discipline is keeping the new framing
sharp *without* upgrading borderline correlations into established effects. The
six `needs_eval.md` items (W3-alignment-causal-probe, W4-formula-injection,
CLIRMRS-external-validation, XRC-conformal-M2, CCI-hop-distance-law,
equivalence-audit-spotcheck) are all treated as DONE per the critic contract; the
paper must stand without them.

The seven structural changes the writer must execute:

1. **COST-FRONTIER FRAMING — embeddinggemma is Pareto-optimal, NOT cheapest. This
   kills the N2 superlative for good.** The verified frontier in `(XRC50, CLIR@10)`
   space has **three members: {embeddinggemma, bge-m3, granite-278m}**
   (`chem_patents/.../extra_cost_frontier/summary.json` `frontier_members`;
   `cost_frontier.csv` `on_frontier`; fig `cp_fig18_cost_frontier.png`).
   - embeddinggemma is the **unique max-CLIR@10 corner** of the frontier
     (XRC50 3.5, CLIR@10 0.5024) — `unique_top_clir_model: "embeddinggemma"`.
   - It is **NOT the cheapest deployable model.** At the *stated, untuned* threshold
     $\tau=0.40$, the admitted set is {bge-m3, qwen3-0.6B, embeddinggemma} and the
     **min-XRC50 admitted model is bge-m3 (2.0×), not embeddinggemma (3.5×)**
     (`tau_admitted_min_xrc_model: "bge-m3"`). The writer **MUST NOT** revive any
     "lowest non-degenerate reading cost = embeddinggemma" superlative (this was N2).
   - Correct phrasing, locked: *embeddinggemma is the Pareto-optimal **capability
     corner** (no model is both cheaper-to-read AND higher-CLIR); bge-m3 is the
     cheaper-to-read admitted alternative on the same frontier.* The lower-XRC
     dominated models (granite 1.25× is on the frontier but at CLIR@10 0.329;
     qwen3, nomic, LaBSE, SapBERT, e5-large-instruct are dominated finite; gte
     is off-plane / censored).
   - **W2 "cheapest reader can be the worst retriever" is DIRECTIONAL ILLUSTRATION
     ONLY.** The XRC50~CLIR@10 trap correlation is **ρ=+0.29, n=7, p=0.53 (n.s.)**
     (`extra_two_tax_degeneracy`/W2 block). The sign is positive so the *picture*
     (granite/SapBERT are cheap-to-read precisely because they retrieve little) may
     be used as a quotable, motivating illustration — but **never as a statistical
     claim**. No "the cheapest reader IS the worst retriever" — only "can be," and
     only as the read-off of the frontier figure.

2. **RRC BUDGET FRONTIER — the knee K* and the L∞ alignment-only floor. This is the
   safe, significant-by-construction novelty upgrade for P2/novelty-#1.** The
   reporter confirmed **all round-1 RRC regression checks PASSED** (RRC@100/@1000
   reproduce to <1e-3; L∞ == lost_at_1000; RRC monotone in K) — so this is
   *regression-checked, safe to state firmly* (`extra_rrc_budget_frontier/summary.json`
   `regression_checks_passed: true`; `rrc_knee.csv`; fig `cp_fig19_rrc_budget.png`).
   - For **embeddinggemma**: knee **K\*=5**, RRC(K\*)=0.4818, RRC@100=0.7445,
     RRC@1000=0.9416, and the structural floor **L∞ = 0.0584** (=1−RRC@1000).
   - K\* by model: egemma 5, bge-m3 5, qwen3 10, nomic 2, granite 20, LaBSE 20,
     SapBERT 20, e5 30, gte 100. L∞ floor ranges 0.058 (egemma) → 0.372 (e5) →
     0.912 (gte).
   - Deployment read-off (locked): *most re-ranker payoff sits in a shallow top-K
     (K\*≈5 for the deployed model, ≤20 for nearly all non-degenerate models); past
     the knee a deeper pool buys almost nothing; and **L∞ ≈ 0.058 is a structural
     floor that no re-ranker over the top-1000 can touch — the alignment-only floor**.*
     This is the object that converts RRC from "renamed recall" into a per-model
     re-ranker-budget planning tool (novelty critic's #1 route).

3. **DEG-GATE — the principled definition of "degenerate," locked once and reused
   everywhere {gte, e5} are excluded.** Reporter-verified rule:
   **DEG(m) = 1 iff CLIR@10(m) < 0.10**, which flags **exactly {gte-base,
   e5-large-instruct}** and matches the paper's existing degenerate-model exclusions
   (`extra_two_tax_degeneracy/summary.json` `matches_paper_exclusions_{gte,e5}: true`;
   `deg_flags.csv`; fig `cp_fig20_degeneracy_gap.png`, 0.10 cutoff line).
   - **Use the SINGLE-criterion CLIR@10 gate, not the dreamer's literal AND-gate.**
     The reporter checked: `clir<0.10 AND rrc1000<0.10` flags **only {gte}** because
     e5 RRC@1000=0.6277 ≥ 0.10; the clean rule that recovers both exclusions is the
     one-criterion CLIR@10<0.10 gate (gap is wide: SapBERT 0.1788 vs e5 0.0766).
   - The writer must state this **once** (Setup §5 or first XRC use in §6.1) and then
     every later "non-degenerate" / "we exclude {gte, e5}" / the winner-take-all
     contamination footnote is anchored to it, citing `cp_fig20`. This turns the
     four prior ad-hoc "degenerate" uses (cohesion #4) into a defensible, reproducible
     gate. gte's bar in cp_fig20 is invisible because CLIR@10=0.000 — correct/labeled.

4. **TWO-TAX SPINE — integrate as a MEASURED-BUT-WEAK claim: "neither benchmark is a
   clean proxy for the other," NOT "the two taxes are independent."** This is the
   cohesion #3 fix (the unifying spine between the two benchmarks) delivered as a
   measured non-redundancy claim instead of a bare sentence — but it is the round's
   single most dangerous honesty trap and the writer must honor it strictly.
   - Reading-cost tax = XRC50 (cross-lingual benchmark); confusability tax =
     alias-graph sibling-confusion rate. Cross-model Spearman over the **n=7
     non-degenerate** models: **ρ = −0.59, p = 0.16 (NOT significant)**
     (`extra_two_tax_degeneracy/summary.json` `spearman_rho_n7_nondeg`,
     `spearman_p_n7`; `two_tax_table.csv`; fig `cp_fig21_two_tax.png`, title already
     reads "n=7 non-deg Spearman rho = −0.59"). All-9-finite (n=8) corroborates the
     weakness: ρ=−0.16, p=0.71.
   - **Locked phrasing (endorsed by reporter):** *"the two taxes are only weakly —
     and if anything inversely — rank-correlated across the seven non-degenerate
     models (Spearman ρ = −0.59, n=7, p=0.16, n.s.), so neither benchmark is a clean
     proxy for the other."* Frame as **descriptive/motivating** for the
     two-benchmark design, **not** as a demonstrated independence/non-redundancy
     result. The sign is **negative** (mild anti-correlation), and |ρ| sits just
     under the 0.6 "non-redundant" threshold; do NOT claim independence, do NOT claim
     statistical significance, do NOT build any load-bearing claim on it.
   - Join anchors (verified): embeddinggemma XRC50=3.5 / conf=0.068 (low-low);
     granite XRC50=1.25 / conf=0.182.

5. **N1 SOFTEN (writer fix).** §6.1 line ~509: "French (0.375) and English (0.367)
   are **statistically indistinguishable** as targets" → "**nearly tied** (fr 0.375
   ≈ en 0.367, within 0.01)." No significance test was run
   (`extra_directional_hub.py` computes only means + a `fr>en` boolean). Keep the
   correct point (no clean easiest target); drop the untested stat claim.

6. **B2 LINE 605 (writer fix).** §6.2 line 605 "Even **the best model** reaches a
   cross-lingual RBO of only 0.39" → match the abstract/intro/conclusion "any model"
   phrasing and the ceiling (not achievement) direction: *"The best cross-lingual
   RBO **any of the nine models** reaches is only 0.39 (Figure~\ref{fig:ag_rbo}) — a
   ceiling no model beats."* This is the one residual internal-framing contradiction.

7. **de↔zh ORPHAN + XRC POPULATION CAVEAT (writer fixes).**
   - de↔zh asymmetry orphan (cohesion #2): fold "+0.23" into the cp_fig03 caption —
     *"…the most asymmetric directed pair is de↔zh (+0.23; asymmetry panel not
     shown)"* — or add a half-sentence referencing cp_fig04. ~6 words; do not leave
     a named instrument with no panel and no caption mention.
   - XRC population caveat (correctness T-NEW): add one clause to the XRC paragraph
     or caption — *"$D_{\text{same}}$ is over the 57 same-language-gold queries and
     $D_{\text{cross}}$ over the 137 cross-gold queries (a **population-level, not
     paired**, ratio)."* Pre-empts the "you scan 3.5× as many documents" being
     misread as paired; mirrors the existing B3 home-advantage footnote discipline.

**Floor (do regardless):** N1 soften, B2 line-605, de↔zh caption, XRC population
clause. The three new objects (cost frontier, RRC budget frontier, DEG gate) are
the upgrades that turn the N2 patch into a contribution; the two-tax spine is the
cohesion upgrade — all four land with the non-significance caveats inline.

---

## Thesis (industrial framing)

> **A chemistry-patent search team must deploy exactly one multilingual embedding
> model, and the number their dashboard shows — average Recall@10 — is the one
> number that hides the failure they will ship.** Average recall is inflated by
> *same-language* hits; the moment a German chemist's query must reach an English
> or Chinese patent (the normal case in a patent family), recall collapses, and no
> two language versions of the same question return the same documents. We make
> this collapse measurable on two content-controlled, patent-grounded benchmarks,
> quantify *what cross-linguality costs* (you read ~3.5× deeper to find a foreign
> twin; a top-100 re-ranker recovers at most ~74% of them, and ~5.8% are
> structurally unrecoverable by any re-ranker), place the deployable models on a
> **cost-vs-capability frontier** (embeddinggemma is the Pareto-optimal capability
> corner; bge-m3 is the cheaper-to-read alternative on the same frontier), and show
> the durable fix is **representation alignment at indexing time, not a monolingual
> re-ranker at query time** — because foreign gold is *under-scored*, not merely
> mis-ordered, and L∞ ≈ 0.058 is a floor only alignment can move.

**Framing overlay — "the cross-lingual tax has two line-items" (cohesion spine,
MEASURED-but-weak).** Cross-lingual retrieval pays a **reading-cost tax** (XRC, the
depth multiplier, on the cross-lingual benchmark) and a **confusability tax** (the
look-alike that out-ranks the gold, on the alias-graph benchmark). The two taxes
are only weakly — and if anything inversely — rank-correlated across the seven
non-degenerate models (ρ=−0.59, n=7, p=0.16, n.s.), **so neither benchmark is a
clean proxy for the other** — the motivating reason both benchmarks are reported,
not a demonstrated independence result. Use this as connective tissue with the
caveat inline; do not make it load-bearing.

Three industrial pillars, each grounded in a file:

1. **The collapse is real, large, and costly.** Best cross-lingual Recall@10 is
   **0.50** (embeddinggemma) against a same-language home advantage up to **+0.55**
   for the most biased model (`chem_patents/.../EXECUTIVE_SUMMARY.md`). The price is
   quantified: **XRC50 = 3.5×** more reading depth to a foreign twin
   (`extra_cost_frontier`, `extra_rrc_budget_frontier`). Spanish — **34 queries, 0
   Spanish gold** — is the built-in no-home stress test.
2. **The collapse is mis-rankable two ways, and both are deployment bugs.** Same
   question in five languages returns *different* patents (cross-lingual RBO ceiling
   **0.39** alias-graph / **0.19** cross-lingual — any model, not "the best model"),
   and a chemically-confusable wrong compound out-ranks every gold on **14–78%** of
   queries. The modal confusion is a **same-language sibling** (44.4%): language bias
   and chemical confusability compound (`extra_joint_failure`).
3. **The cause is separable representations, so the fix is alignment — and it has a
   floor.** r(cross-language AUC, CLIR@10) = **+0.96**, robust to dropping the two
   collapsed encoders (+0.958, n=7, `extra_correlation_robustness`): foreign gold is
   *under-scored*. A re-ranker reads a list it cannot repair, and we bound exactly
   how much: knee **K\*=5**, **RRC@100 ≤ 0.74**, and **L∞ = 0.058 unrecoverable by
   any re-ranker** (`extra_rrc_budget_frontier`).

---

## Contributions (numbered, each with a one-line novelty claim)

**C1. Two content-controlled, patent-grounded multilingual chemistry-retrieval
benchmarks built only from human-translated patent text.** *(unchanged from round
2 — novelty critic: NOVEL, well-defended.)*
- *What:* (a) the **alias-graph** benchmark — 132 questions, 24 ChEBI compounds × 5
  languages, two co-equal relevance lenses, each compound shipping a graph of
  chemically-confusable neighbours (siblings/parents) as hard negatives; (b) the
  **cross-lingual (CLIR)** benchmark — 137 questions in en/de/es/fr/zh (57
  human-original + 80 MT-cross-lingual) over a **23,487-doc** shared `multilingual_GP`
  haystack, Spanish a pure no-home query language.
- *Novelty claim:* **The first cross-lingual, content-controlled,
  chemistry-ontology-grounded patent-retrieval benchmark whose gold is genuinely
  parallel human-translated patents + ChEBI ontology membership (not
  `publication_number` equivalence, not machine-translated documents) and whose
  negatives are chemically-confusable neighbours.** Bounded against **CLEF-IP**
  (prior-art relevance gold, not parallel translation) and **DAPFAM** (family-level
  gold = the equivalence we reject). *Mandatory cites:* CLEF-IP, DAPFAM, CLIRMatrix,
  ChEBI.

**C2. A cross-lingual robustness-metric family reported co-equally with recall —
anchored by deployment-legible cost objects (XRC, RRC) and now framed as a
cost-vs-capability frontier with a re-ranker-budget knee.**
- *What:* CLIR@k vs MoLIR + home-advantage; directional CLIR matrix +
  hub/asymmetry; mate-retrieval; cross-lingual RBO; language-collapse /
  over-representation; separability AUC (same vs cross); **XRC** (reading-cost
  multiplier) and **RRC** (re-ranker recoverability ceiling), now reported as **(i)
  a cost-vs-capability Pareto frontier** in (XRC50, CLIR@10) space and **(ii) an
  RRC(K) budget curve with a knee K\* and an L∞ alignment-only floor**; the **DEG
  gate** (CLIR@10<0.10) that defines "degenerate"; CLIR-MRS / MRS demoted to a
  table-ordering convenience.
- *Novelty claim:* **A retrieval-side, ranking-level robustness suite for CLIR whose
  two headline cost instruments are new deployment objects: XRC as a
  distribution-free reading-depth multiplier presented on a capability-conditioned
  Pareto frontier (no model is both cheaper-to-read and higher-CLIR than the corner
  model), and RRC as a per-model re-ranker-budget curve with a knee K\* and a
  structural floor L∞ that bounds what any re-ranker can recover.** The composite is
  explicitly NOT a contribution; per-axis dominance + the frontier carry the result.
  Cross-lingual RBO cites Bailey et al. 2017; separability AUC is standard machinery
  whose novelty is the same-vs-cross decomposition + the re-ranker corollary; the
  directional matrix cites CLIRMatrix. *(Optional novelty-hardening cites: a
  two-stage/recall-ceiling cascade reference for RRC's qualitative ancestor; LaBSE/
  Artetxe-Schwenk for the borrowed "mate-retrieval" term.)*

**C3. A mechanism finding, confirmed on a content-controlled corpus and made
falsifiable: cross-lingual chemistry-retrieval failure is an embedding-level
separability deficit, so the lever is alignment, not re-ranking — with a measured
floor.**
- *What:* availability sets the stage (English's reachable gold 42% vs 8–10% for
  de/es/zh) but a **residual encoder bias remains** (availability-adjusted slope
  −0.57, n=5, DESCRIPTIVE); the modal confusion is a **same-language sibling**
  (44.4%); structure-style questions are the trap (R@10 0.26, confusion 51%; formula
  token p<0.01); confusion **is** a separability collapse (AUC 0.55 vs 0.70) and
  across models r(cross-language AUC, CLIR@10) = **+0.96**, robust on n=7. Bound:
  knee **K\*=5**, **RRC@100 ≤ 0.74**, **L∞ = 0.058** unrecoverable.
- *Novelty claim:* **We *confirm* the alignment-not-translation finding of
  [2511.19324, 2507.07543] on a content-controlled parallel patent corpus that
  removes the translationese/content confounds those studies could not, and add (i)
  a chemistry-specific same-language-sibling confusability trap and (ii) a
  separability-AUC + RRC-floor test that turns "alignment is the fix" into a
  falsifiable per-model re-ranker bound with an explicit structural floor (L∞).**
  *Mandatory cites:* 2511.19324, 2507.07543.

**C4. A concrete, audited deployment decision with an operating rule and a
cost-frontier justification.**
- *What:* deploy **embeddinggemma** — it is the **Pareto-optimal capability corner**
  of the (XRC50, CLIR@10) frontier (the unique max-CLIR model; no model is both
  cheaper-to-read AND higher-CLIR), and leads CLIR@10, separability, and
  twin-finding *individually* (so the recommendation does not rest on a composite);
  **bge-m3 is the cheaper-to-read admitted alternative** on the same frontier (the
  decision is a frontier choice, not a single superlative). Report
  XRC/RRC/CLIR@10/language-parity *next to* recall; budget the re-ranker by the knee
  (K\*≈5) and respect the L∞ floor; **do not reflexively ensemble** (untuned RRF
  underperformed; oracle headroom real — CLIR@10 0.61 / alias 88% — but needs a
  score-aware combiner or per-language routing); **machine-translating the question
  is safe** (paired diff −0.044, p=0.13).
- *Novelty claim:* **An industry-track deployment decision grounded in a
  capability-conditioned cost frontier and a re-ranker-budget knee (not mean recall
  or a hand-weighted composite), with a negative ensemble result and a QT-vs-DT
  budget rule (Oard 1998; Saleh & Pecina 2020) re-derived for embedding retrieval
  over patents and quantified as an insignificant null.** *Mandatory cites:* Oard
  1998, Saleh & Pecina 2020, 2605.24297.

**C5. (Supporting) A reproducible QAC generation + audit pipeline the benchmarks
rest on.** *(unchanged — kept as support, softened to "human validation summarized
in the system description"; `\todo`→`% TODO`.)*

---

## Section map

### Abstract — purpose / beats / figures+numbers / links
- *Purpose:* state the deployment problem, the two content-controlled benchmarks +
  metric family, the cost numbers + the frontier, the headline model, and the
  alignment-not-re-ranking payoff in ~180 words.
- *Beats:* (1) average recall hides cross-lingual collapse. (2) two
  content-controlled patent-grounded benchmarks + a CLIR robustness suite. (3)
  headline numbers: best CLIR@10 0.50; home advantage up to +0.55; **RBO ceiling
  0.39 (alias-graph) / 0.19 (cross-lingual)** — both named; confusion 14–78%;
  **XRC50 3.5×**; **RRC@100 ≤ 0.74**, **L∞ 0.058**. (4) cause is a separability
  deficit (r(cross-language AUC, CLIR@10) +0.96, spelled in full); fix is alignment
  not re-ranking; **embeddinggemma is the Pareto-optimal capability corner** of the
  cost frontier (NOT "cheapest"); MT-of-question is safe.
- *Numbers:* both `EXECUTIVE_SUMMARY.md` + `extra_cost_frontier/summary.json` +
  `extra_rrc_budget_frontier/summary.json`.
- *Links:* sets up the Introduction's contributions list.
- *Honesty note:* do NOT put either non-significant correlation (two-tax ρ=−0.59;
  trap ρ=+0.29) in the abstract.

### 1 Introduction — purpose / beats / figures+numbers / links
- *Purpose:* motivate the deployment question; enumerate C1–C5.
- *Beats:* (1) team picks ONE model; dashboard shows mean Recall@10. (2) inflated by
  same-language hits — open with Spanish (34 queries, 0 Spanish gold). (3) Two
  failures behind the average: **inconsistency** (RBO ceiling 0.39 alias-graph /
  0.19 cross-lingual — "the best cross-lingual RBO *any model* achieves," never "the
  best model") and **confusion** (14–78%). (4) the cost framing in one line (read
  ~3.5× deeper; align-not-rerank; the choice is a cost-vs-capability frontier, not a
  single cheapest model). (5) explicit numbered contributions C1–C5. (6) one-line
  spoiler of the deployment rule.
- *Figures:* teaser `cp_fig01_clir_leaderboard.png`.
- *Links:* each contribution forward-references its home section.

### 2 Related Work — purpose / beats / figures+numbers / links
*(unchanged from round 2 — the six "you reinvented X" surfaces are CLOSED; do not
re-open. Six paragraphs, each a defended boundary, ending on a forward bridge into
Benchmarks.)*
- *Beats:* (1) Multilingual/CLIR benchmarks (MIRACL/MMTEB/NeuCLIR/MTEB) +
  **CLIRMatrix** + **MMTEB/2605.31142** (aggregation-sensitivity → why we test the
  ribbon and demote the composite). (2) Cross-lingual RAG / language preference
  (BordIRlines, XRAG, Linguistic Nepotism) — content-vs-language confound our
  parallel patents control. (3) Calibration / conformal / fairness-OT (TRAQ,
  CONFLARE, Conformal-RAG) — they score calibration, we score ranking robustness;
  Conformal-RAG is the future-work machinery cite for XRC-conformal (PENDING-EVAL).
  (4) Patent IR & family non-equivalence — **CLEF-IP** + **DAPFAM** (the single most
  dangerous round-1 omission, now fixed). (5) Chemistry IR / entity models (SapBERT,
  PaECTER) + **2605.24297** (English-only patent eval, distinguish on cross-lingual).
  (6) When to translate — **Oard 1998; Saleh & Pecina 2020**; cross-lingual RBO
  **Bailey et al. 2017**.
- *Optional add (novelty-hardening, not mandatory):* a two-stage/recall-ceiling
  cascade reference so RRC cites its qualitative ancestor and claims only the
  cross-lingual quantification.
- *Links:* END with the forward bridge into Benchmarks (already in place).

### 3 Benchmarks — purpose / beats / figures+numbers / links
*(unchanged from round 2.)* Deliver C1: shared corpus (`multilingual_GP`, 23,487
docs; corpus-construction stats deferred to the system description with `% TODO`),
the alias-graph benchmark (two lenses, confusable-neighbour negatives), the
cross-lingual benchmark (57 original + 80 synthetic, Spanish pure query-side),
honesty-by-design (human-translated source only; ontology/translation gold, not
`publication_number`; MT only for questions), and the C5 pipeline (softened).

### 4 Metrics — purpose / beats / figures+numbers / links
- *Purpose:* deliver C2 — define each robustness axis. **This round adds two upgrades
  inside the existing XRC/RRC definitions and locks the DEG gate.**
- *Beats:* (1) CLIR@k vs MoLIR + home-advantage. (2) Directional CLIR + hub/asymmetry
  (cite CLIRMatrix). (3) Mate-retrieval. (4) Cross-lingual RBO (cite Bailey 2017).
  (5) Language-collapse / over-representation. (6) Separability AUC (same vs cross).
  (7) **XRC — reading-cost multiplier.** Eq.~\ref{eq:xrc} stays; **add the
  population-level clause** ($D_{\text{same}}$ over 57, $D_{\text{cross}}$ over 137;
  population, not paired — closes T-NEW) and keep the monotone-invariance half-
  sentence. Then point forward to the **cost frontier** in Results: "we report XRC
  not as a scalar to minimize but on a capability-conditioned (XRC50, CLIR@10)
  frontier (\S\ref{ssec:cp}), because the global XRC minimum is a model that
  retrieves almost nothing." (8) **RRC — re-ranker recoverability ceiling.**
  Eq.~\ref{eq:rrc} stays; **add the budget-object framing**: "RRC@K is the cumulative
  first-foreign-twin hit rate (mate-hit@K on cross-lingual queries); our contribution
  is reading it as a per-model re-ranker *budget curve* — the knee K\* past which a
  deeper pool buys almost nothing, and L∞ = 1−RRC@K_max, the floor no re-ranker over
  the top-1000 can move." (9) **DEG gate — define "degenerate" ONCE here (or in §5).**
  State: "We call a model *degenerate* if CLIR@10 < 0.10; this flags exactly
  `gte-base` and `e5-large-instruct` (Figure~\ref{fig:cp_deg}), and we exclude them
  from all 'non-degenerate' summaries." Anchors every later use + the WTA
  contamination footnote. (10) **CLIR-MRS / MRS — demoted** (unchanged: table-
  ordering convenience, per-axis numbers carry the argument).
- *Figures/numbers:* DEG gate `cp_fig20_degeneracy_gap.png` may be introduced here
  or in Results; XRC/RRC formulae; the population clause; the frontier/budget
  forward-pointers.
- *Links:* Setup says which models/data; Results applies these metrics.

### 5 Experimental Setup — purpose / beats / figures+numbers / links
*(unchanged + one line.)* 9 models, shared 23,487-doc haystack, two lenses /
original-synthetic split, reproduce commands. Add a one-line pointer that the new
analyses regenerate via the `experimental_plots/extra_*.py` scripts, now including
`extra_cost_frontier.py`, `extra_rrc_budget_frontier.py`,
`extra_two_tax_degeneracy.py`. **Place the DEG-gate definition here if not in §4** —
exactly once, then reused.

### 6 Results — purpose / beats / figures+numbers / links
- *Purpose:* cross-lingual headlines, then alias-graph headlines, then leaderboards.
  Keep §6.1 cross-lingual / §6.2 alias-graph / §6.3 leaderboards separated; do not
  interleave numbers within a paragraph.
- *Beats (§6.1 cross-lingual):*
  (1) collapse — `cp_fig01` (CLIR@10 0.50) + `cp_fig02_home_advantage.png` (+0.55),
  with the T1 hedge ("much of which availability shapes, §7, though a residual
  encoder bias remains").
  (2) **anisotropy as a hub-and-spoke graph (N1 FIX)** — `cp_fig03`. Hardest directed
  edge en→de (R@10 0.12), most asymmetric pair de↔zh (gap +0.23 — **fold +0.23 into
  the cp_fig03 caption** to close the orphan), *no clean "easiest target"*: **fr 0.375
  ≈ en 0.367, "nearly tied" within 0.01** (NOT "statistically indistinguishable"),
  corpus-composition caveat (en 46% / zh 0.4%). Source:
  `extra_directional_hub/summary.json`.
  (3) **the cost of cross-linguality + the cost frontier (UPGRADED)** — the XRC
  paragraph keeps XRC50=3.5× (median depth 2→7) **and adds the population clause**,
  then presents **`cp_fig18_cost_frontier.png`**: the (XRC50, CLIR@10) Pareto frontier
  {embeddinggemma, bge-m3, granite-278m}. **embeddinggemma = Pareto-optimal capability
  corner (unique max-CLIR); NOT cheapest** (at τ=0.40, bge-m3 is the cheaper-to-read
  admitted model). The "cheapest reader CAN be the worst retriever" is a
  **directional read-off of the figure only** (trap ρ=+0.29, p=0.53 n.s. — state the
  caveat inline if the correlation is mentioned at all). Source:
  `extra_cost_frontier/summary.json` + `cost_frontier.csv`. Replaces the old
  cp_fig15 XRC-vs-CLIR scatter (cp_fig15 may be retired in favor of cp_fig18, or kept
  as the per-model bar and cp_fig18 as the frontier — writer's call, but cp_fig18 is
  the load-bearing one now).
  (4) MT-of-question safe — `cp_fig05_mt_penalty.png` (−0.044, p=0.13; null).
  (5) **twins buried + the re-ranker BUDGET FRONTIER (UPGRADED)** —
  `cp_fig06_mate_retrieval.png` (mate-hit@10 0.38), `cp_fig07_first_foreign_rank.png`
  (15% never in top-1000), and **`cp_fig19_rrc_budget.png`** as the headline:
  RRC(K) curves with the knee K\*=5 for embeddinggemma, RRC@100=0.7445,
  RRC@1000=0.9416, and **L∞=0.0584 the alignment-only floor**. State the budget
  read-off ("most payoff in a shallow top-K, K\*≈5; L∞ is structural"). This is
  regression-checked — safe to state firmly. Source: `extra_rrc_budget_frontier/`.
  (cp_fig16 RRC bar may be retired in favor of cp_fig19, or kept as the simple per-
  model ceiling with cp_fig19 as the budget curve.)
- *Beats (§6.2 alias-graph):*
  (0) **TWO-TAX SPINE BRIDGE (cohesion #3 FIX)** — open §6.2 with the bridging
  sentence: *"If §6.1 measured what cross-linguality costs to read, the alias-graph
  benchmark measures what it costs in precision — the second line-item of the same
  bill,"* and back it with the **measured-but-weak** non-redundancy:
  `cp_fig21_two_tax.png` (XRC50 reading-cost tax × sibling-confusion tax per model),
  with the caveat **inline**: "the two taxes are only weakly — and if anything
  inversely — rank-correlated across the seven non-degenerate models (ρ=−0.59, n=7,
  p=0.16, n.s.), so neither benchmark is a clean proxy for the other." Frame as the
  motivation for two benchmarks, NOT an independence result.
  (6) inconsistency — `ag_fig1_cross_lingual_rbo.png` (RBO ceiling 0.39; **B2 line-605
  FIX**: "any of the nine models," ceiling not achievement).
  (7) confusion — `ag_fig2_confusion_both_lenses.png` (14–78%),
  `ag_fig5_universal_attractors.png`, two-level severity split (sibling 18.1% vs
  parent 6.2%, 2.9×; egemma 6.1% vs 1.5%; `extra_confusion_severity/`) — graded
  hop-distance law PENDING-EVAL.
- *Beats (§6.3 leaderboards):* both tables ordered by CLIR-MRS / MRS, the
  teaser-reorder signpost, and the **aggregation-ribbon caveat**
  (`cp_fig17_aggregation_ribbon.png`, rank range [1,4]); the winner-take-all column
  contamination footnote now **anchored to the DEG gate** (gte "wins" parity/MT-robust
  precisely because CLIR@10=0.000 < 0.10). Radars (`cp_fig14`, `ag_fig10`) keep the
  interpretive clause.
- *Links:* Analysis explains *why*.

### 7 Analysis — purpose / beats / figures+numbers / links
- *Purpose:* deliver C3 — the mechanism, the trap, the separability diagnosis, with
  the RRC floor as the falsifiable bound.
- *Beats:*
  (1) **joint failure leads the section** — modal confusion is a same-language
  sibling (114/257=44.4%; siblings 79.4%; same-language winners 55.6%),
  `ag_fig12_joint_failure_modes.png`. Unifying thesis that lets the next sentences
  SPLIT cleanly by benchmark.
  (2) **availability sets the stage, a residual encoder bias remains** — alias
  sentence + footnote (own-lang 0.63–0.82 vs foreign 0.35–0.47; English reachable
  42% vs 8–10%; availability-adjusted slope −0.57, n=5, DESCRIPTIVE, zh carries the
  largest home advantage; `extra_availability_residual/`,
  `ag_fig11_availability_residual.png`); cross-lingual sentence + footnote
  (over-fetch up to 49×, same-language noise out-ranks gold on 60%; `cp_fig09`,
  `cp_fig10`). Two sentences, two footnotes.
  (3) **structure-question trap** — `ag_fig6_question_type_effect.png` (structure
  R@10 0.26 / confusion 51% vs role 0.60/25%; formula token p<0.01).
  (4) **bias↔inconsistency, hedged** — keep r(cross-language AUC, CLIR@10)=+0.96 as
  the load-bearing, robust mechanism (+0.958, n=7); keep the two fragile correlations
  (home-adv~RBO, over-rep~CLIR) as descriptive observations that do NOT survive
  dropping the collapsers (unchanged from round 2 — do not re-touch; correctness
  critic called this the best round-2 improvement).
  (5) **separability deficit ⇒ re-ranker FLOOR** — `cp_fig11_separability.png` (+0.96
  text statistic) + `ag_fig8_confusion_is_separability.png` (AUC 0.55 vs 0.70) ⇒
  under-scoring ⇒ the RRC floor (back-reference §6.1: knee K\*=5, RRC@100 ≤ 0.74,
  **L∞=0.058 unrecoverable by any re-ranker**). This is the crux that sets up
  Deployment's "align, don't re-rank."
- *Links:* the separability + RRC-floor beat sets up Deployment.

### 8 Deployment Recommendation — purpose / beats / figures+numbers / links
- *Purpose:* deliver C4 — the single decision, justified by the frontier and budgeted
  by the knee.
- *Beats:*
  (1) **Deploy embeddinggemma — the Pareto-optimal capability corner (N2 FIX).**
  REWRITE the current line ~898 ("lowest non-degenerate reading cost (XRC50 3.5×)")
  to: *"…leads CLIR@10 (0.50), separability, and twin-finding (median first-foreign
  rank 5) individually, and sits on the cost-vs-capability frontier as its **unique
  max-CLIR corner** — no model is both cheaper-to-read AND higher-CLIR
  (Figure~\ref{fig:cost_frontier}). It is **not the cheapest** deployable model:
  bge-m3 reads shallower (XRC50 2.0× vs 3.5×) at a still-admitted CLIR@10, so the
  decision is a frontier choice — pick the capability corner unless the
  reading-budget is binding, in which case bge-m3 is the cheaper-to-read frontier
  alternative."* Keep the composite-rank-range [1,4] caveat. **Do NOT write any
  "lowest reading cost = embeddinggemma" superlative.**
  (2) **Report robustness next to recall** (XRC/RRC/CLIR@10/language-parity), never
  average alone; Spanish no-home is the reason.
  (3) **Budget the re-ranker by the knee** — NEW deployment read-off:
  `cp_fig19_rrc_budget.png` — most payoff is in a shallow top-K (K\*≈5 for the
  deployed model, ≤20 for nearly all non-degenerate models); a deeper pool past the
  knee buys almost nothing; **L∞=0.058 is the alignment-only floor**.
  (4) **Do not reflexively ensemble** — `cp_fig12_ensemble_headroom.png` (oracle
  0.61, RRF loses) + `ag_fig7_ensemble_headroom.png` (88% vs 76%, Chinese largest
  headroom); the 12% universal-blind core is 14/16 structure questions — needs
  chemistry-aware help.
  (5) **Align, do not re-rank** — follows from the separability deficit, bounded by
  RRC: L∞=5.84% lost forever, top-100 leaves ~25% on the table.
  (6) **Budget rule** — MT the query, human-translate the corpus (Oard 1998 / Saleh &
  Pecina 2020 re-derivation).
- *Links:* Limitations bounds these; Conclusion restates.

### 9 Limitations — purpose / beats / figures+numbers / links
*(round-2 list + the two new non-significance caveats made explicit.)*
- (1) **Scale & statistical reach** — 132 + 137 questions, 24 compounds; thin
  directional cells; XRC50 the robust headline, D90/D95 right-censored lower bounds;
  availability slope −0.57 DESCRIPTIVE on 5 languages. **ADD:** the two-tax
  non-redundancy (ρ=−0.59, n=7, p=0.16) and the cheapest-reader trap (ρ=+0.29, n=7,
  p=0.53) are **non-significant** and reported as descriptive/motivating, not
  established effects. (2) **Domain transfer** — chemistry-patent-specific; no
  general-domain companion yet (PENDING-EVAL). (3) **Composite validation** — CLIR-MRS
  a reporting convenience, not externally validated (PENDING-EVAL). (4) **Judge
  dependence & gold equivalence** (PENDING-EVAL: equivalence-audit). (5) **Severity
  law** — two-level split; graded hop-distance law PENDING-EVAL. (6) **5 languages**,
  one non-Latin script; conformal-XRC future work. **ADD (forthcoming, upside-only):**
  the W3 alignment causal probe (fit a per-language alignment map on one model,
  re-retrieve, recompute XRC50/RRC@100 before/after) is the natural next step that
  XRC/RRC were built to measure — mention as forthcoming; the paper does NOT depend
  on it.
- *Links:* Conclusion.

### 10 Conclusion — purpose / beats / figures+numbers / links
- *Beats:* two content-controlled patent-grounded benchmarks + a CLIR robustness
  suite reveal that average recall hides a cross-lingual collapse that *costs* ~3.5×
  the reading budget (XRC) and that a top-100 re-ranker cannot fully recover (RRC,
  knee K\*=5, floor L∞=0.058); **embeddinggemma is the Pareto-optimal capability
  corner** of the cost frontier (bge-m3 the cheaper-to-read alternative); the cause
  is an embedding-level separability deficit (r(cross-language AUC, CLIR@10) +0.96,
  robust); so the fix is alignment at index time, not re-ranking; budget rule = MT
  the query, human-translate the corpus. Restate the two ceilings verbatim (0.39
  alias-graph / 0.19 cross-lingual). Report robustness next to recall.
- *Honesty note:* no non-significant correlation in the conclusion.
- *Links:* closes the loop opened in the Introduction.

---

## Open narrative risks (for critics to watch)

1. **The "cheapest = embeddinggemma" superlative must stay dead (correctness).** N2.
   embeddinggemma is the Pareto **capability corner**, NOT the cheapest; bge-m3
   (XRC50 2.0×) is the min-XRC admitted model at τ=0.40. The writer must not revive
   any "lowest non-degenerate reading cost" phrasing. Source:
   `extra_cost_frontier/summary.json` (`HONEST_CLAIM`,
   `tau_admitted_min_xrc_model: "bge-m3"`).

2. **Two-tax is MEASURED-BUT-WEAK, not independence (correctness + cohesion).**
   ρ=−0.59, n=7, p=0.16 (n.s.), negative sign, |ρ| just under 0.6. Phrase as "neither
   benchmark is a clean proxy for the other," descriptive/motivating only. Never:
   "the two taxes are independent / non-redundant (proven)" and never significant.
   Caveat must be **inline** at the figure, and it must NOT appear in
   abstract/intro/conclusion. Source: `extra_two_tax_degeneracy/summary.json`
   (`two_tax` block).

3. **The cheapest-reader "trap" is DIRECTIONAL ILLUSTRATION ONLY (correctness).**
   XRC50~CLIR@10 trap ρ=+0.29, n=7, p=0.53 (n.s.). Permitted as a quotable read-off
   of the frontier picture ("the cheapest reader CAN be the worst retriever"), never
   as a statistical claim, never "IS." Source: same summary.json (W2 block).

4. **τ=0.40 is STATED, not tuned (correctness).** The admitted-set / bge-m3
   conclusion is conditional on a stated threshold; say so explicitly when the
   admitted set is used.

5. **DEG gate is the SINGLE-criterion CLIR@10<0.10 rule (correctness).** Not the
   AND-gate (which would flag only gte). Stated once, reused everywhere {gte, e5} are
   excluded, citing `cp_fig20`. Source: `extra_two_tax_degeneracy/summary.json`
   (`deg_gate`, `matches_paper_exclusions_*`).

6. **RRC budget frontier is regression-checked and safe to state firmly (this is the
   one new object with no significance caveat).** Knee K\*=5, L∞=0.0584; all
   round-1 RRC checks PASSED. Do not over-hedge this one — it is the clean novelty
   upgrade. Source: `extra_rrc_budget_frontier/summary.json`
   (`regression_checks_passed: true`).

7. **Two RBO ceilings (correctness + cohesion).** 0.39 (alias-graph) / 0.19
   (cross-lingual) are different benchmarks; the intro/conclusion say "any model"
   (B2), and §6.2 line 605 must now too. Never average/conflate.

8. **MT-of-question is a NULL (correctness).** −0.044, p=0.13; "no significant
   penalty," never "MT helps."

9. **Availability slope is DESCRIPTIVE n=5 (correctness).** −0.57 strengthens the
   encoder-bias claim but is on 5 points; availability still sets which gold is
   reachable, it just does not explain away the same-language bias.

10. **Figure retirement bookkeeping (cohesion).** cp_fig18 (frontier) is the new
    load-bearing cost figure and cp_fig19 (RRC budget) the new load-bearing
    re-ranker figure; the writer should decide whether cp_fig15/cp_fig16 are retired
    or kept as simpler companions, and must not leave a referenced-but-absent figure
    or an orphan panel. cp_fig20 (DEG) and cp_fig21 (two-tax) must each be referenced
    and interpreted where introduced.
