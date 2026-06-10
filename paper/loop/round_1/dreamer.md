# Dreams (round 1)

The Dreamer's job: out-imagine the critics. For every weakness, a safe fix AND a
stretch idea; plus a few nobody asked for. Every idea is tagged
`[feasible-now | needs-eval | paper-framing-only]` with rough cost and novelty
payoff. **Core principle honored throughout: the paper must stand without any
code-switch / new-eval result** — every `needs-eval` idea is a bonus, never a load
bearer.

---

## Problems on the table (distilled from the 3 critics)

**Novelty (R#1):**
- N1. CLIR-MRS/MRS is an arbitrary hand-weighted composite with no external
  validation → demote or validate (the single biggest exposure).
- N2. "First decomposition / alignment-is-the-lever" already exists in 2511.19324 +
  2507.07543 → must reframe as *confirm-on-content-controlled* + add chemistry delta.
- N3. Directional CLIR matrix = CLIRMatrix (EMNLP 2020); separability AUC standard;
  home advantage = documented same-language bias; QT-vs-DT rule = Oard 1998 / Saleh &
  Pecina 2020 → cite, narrow "first" to "content-controlled."
- N4. The genuinely novel assets are: the **content-controlled parallel patent
  asset** (C1), the **confusion-rate metric** ("a look-alike beats ALL gold"), the
  **structure-question trap + formula-token intervention**, and the **separability →
  re-ranker-cannot-help corollary**. Critic explicitly hands the dreamer: turn the
  separability deficit into a **conformal cross-lingual reading-cost multiplier** —
  "the single best novelty upgrade available, already half-computed."

**Correctness (R#2):**
- B1 (MISMATCH). "English is the easiest target" is false — French (0.375) > English
  (0.367) pooled; the shown best-model figure makes fr-target (0.56) > en-target (0.49).
- B2 (attribution). "even the best model reaches RBO 0.19" — 0.19 is granite's ceiling,
  embeddinggemma is 0.154; say "best across models / ceiling," not "the best model."
- B3 (presentation). Fig 1 green MoLIR bars are the 57-original subpopulation; the
  +0.55 home advantage itself is a clean paired contrast — add one caption sentence.
- T1. +0.55 home advantage led with as encoder property; §6 later attributes it to
  availability — tie the two.
- T2. Directional anisotropy partly = corpus composition (en 46% / zh 0.4%), not encoder.
- T5. r=+0.96, −0.85/−0.87 are n=9; two collapsed encoders drive the spread.

**Cohesion (R#3):**
- G1. §6 "availability confound" paragraph fuses alias-graph numbers with chem-patents
  figures under one alias-only footnote — split by benchmark.
- G2. Abstract gives one RBO ceiling (0.39); body gives two (0.39 / 0.19).
- G3. Related Work → Benchmarks transition ends on a future-work disclaimer, not a handoff.
- G4. Two `\todo` markers sit in load-bearing body prose (visible promise-then-retract).
- G5. cp_fig11 caption claims a correlation the bar chart doesn't plot; radars read as decoration.
- G6. Teaser orders by recall, leaderboard by CLIR-MRS — same models, two orders, no signpost.
- Orphans: C5 has no in-text number; "~12% universal-blind core" appears first in Deployment.

---

## (a) New analyses
*Additional cuts of the EXISTING data that reveal something new. Most are feasible-now.*

### [feasible-now] A1 — The Cross-Lingual Reading-Cost Curve (separability → depth)
- **what/how:** The separability AUCs already exist per model, per same/cross split.
  Convert each into a *reading-cost multiplier*: under a fixed coverage target (e.g.
  "find the foreign twin with 95% probability"), how many documents must a reader scan
  cross-language vs same-language? Operationally, for each model take the rank
  distribution of the first foreign twin (round04 already has first-foreign-rank) and
  of the first same-language gold; the ratio of the 95th-percentile depths is the
  multiplier. Plot multiplier vs CLIR@10 across the 9 models.
- **cost:** CPU-only, reuses round04 + round08 outputs. Closes N1 (gives a
  non-arbitrary deployment number to *replace* the composite's role), N4.
- **novelty payoff:** lets the paper claim a *deployment-legible, distribution-free
  cost of cross-linguality* ("to find the German twin at 95% coverage you read 9× more
  documents than for the English copy") — concrete where the composite was abstract.
  This is the safe-now, no-conformal-math version of the wild card W1.

### [feasible-now] A2 — Availability-adjusted home advantage (partial out the confound)
- **what/how:** Critic T1 says +0.55 is led-with as encoder skill but §6 calls it an
  availability artifact. Make the adjustment *quantitative*: regress per-language
  home advantage on per-language in-corpus gold-availability share (42% / 8–10% already
  computed) across the language points, and report the **residual** home advantage —
  the part NOT explained by where the gold lives. If the residual is ≈0, the honest
  headline becomes "home advantage is almost entirely an availability artifact"; if
  positive, "even after controlling for availability, encoders retain a +X same-language
  bias." Either outcome is a cleaner claim than the raw +0.55.
- **cost:** CPU-only, two columns already on disk (home_adv per lang, availability share
  per lang); a single OLS / partial correlation. Closes T1, T2, N2.
- **novelty payoff:** turns a confounded headline into a *decomposed* one — "X% of the
  home advantage is corpus composition, Y% is encoder bias" — which is exactly the
  content-control delta the paper claims over 2507.07543/2511.19324.

### [feasible-now] A3 — Leave-the-collapsers-out robustness check (defuse T5)
- **what/how:** The load-bearing r=+0.96 (AUC,CLIR@10) and r=−0.85/−0.87 (bias,RBO) are
  n=9 with gte-base and e5 collapsed. Recompute each correlation (i) on all 9, (ii) on
  the 7 non-collapsed models, (iii) Spearman as well as Pearson, and report all three.
  Add a one-line "the relationship survives dropping the two degenerate encoders
  (r=… on n=7)" or honestly flag if it doesn't.
- **cost:** CPU-only, recompute correlations from existing per-model tables. Closes T5,
  and the novelty critic's "small-n mechanism" worry.
- **novelty payoff:** upgrades the mechanism from "a correlation two outliers could be
  driving" to "a relationship robust to the obvious confound" — makes the falsifiable
  re-ranker claim defensible.

### [feasible-now] A4 — Confusion as a function of ontological distance (sharpen the trap)
- **what/how:** Confusion is "driven almost entirely by *sibling* compounds rather than
  parent classes." Make that gradient explicit: bin the winning distractor by its
  ontology hop-distance to the gold concept (sibling=1 parent, parent=up, cousin, etc.,
  all available in the alias graph) and plot confusion rate vs hop-distance. Expectation:
  monotone decay with distance — a clean, citable "confusability is a graph-distance
  phenomenon" curve.
- **cost:** CPU-only IF the per-query winning-distractor identity + its ontology relation
  is already logged in round02/round06 outputs (likely, since the sibling-vs-parent split
  is already reported). Pure re-aggregation. Closes N4 (deepens the confusion finding the
  critic says to "feature more prominently").
- **novelty payoff:** "confusion rate decays with ChEBI hop-distance" is a new,
  domain-specific, mechanistic curve — strengthens the most original contribution.

### [feasible-now] A5 — The directional matrix, re-read as a hub-and-spoke graph
- **what/how:** B1 kills "English is easiest target." Replace the weak target-ranking
  claim with a *graph* reading of the 5×5 directional matrix: treat languages as nodes,
  recall as edge weight, and report (i) hardest **directed edge** (en→de 0.12, verified),
  (ii) most asymmetric **pair** (de↔zh +0.23, verified), and (iii) a node-level "hub
  score" = mean *incoming* recall (how reachable each language is as a target) annotated
  with its corpus share. This reframes anisotropy as a property of the language graph
  rather than a single "easiest" language, and naturally folds in the T2 corpus-share caveat.
- **cost:** CPU-only, pure re-presentation of pair_recall.csv. Closes B1, T2, N3.
- **novelty payoff:** converts a false claim into a defensible structural one ("the
  retrieval language-graph is anisotropic and its reachability tracks corpus share"),
  and lets the paper cite CLIRMatrix as the directional precedent while claiming the
  graph/hub reading + corpus-composition control as the patent-domain delta.

### [feasible-now] A6 — Per-query "who buried me" attribution audit (chimeric-field lite)
- **what/how:** For the confused queries, the distractor that wins is already known.
  Cross-tabulate: was the burying document (i) a same-language non-gold, (ii) a
  cross-language sibling, or (iii) a same-language sibling? This separates the two
  failure modes the paper currently states separately (language collapse vs chemical
  confusability) and shows whether they *co-occur* — i.e. is the worst case a
  same-language sibling (both traps at once)?
- **cost:** CPU-only if the per-query top-1 distractor + its language + its ontology
  relation are logged (round07 logs the same-language-confusion rate, so the join exists).
  Closes N4, G1 (gives the two-benchmark mechanism a single joint cut).
- **novelty payoff:** "the modal failure is a same-language sibling — language bias and
  chemical confusability compound" is a new, memorable, joint-mechanism claim that ties
  the two benchmarks together (also helps cohesion G1).

### [feasible-now] A7 — Aggregation-sensitivity ribbon for the leaderboard (kills the composite attack pre-emptively)
- **what/how:** Recompute the model ranking under *every* sensible aggregation: (i)
  current CLIR-MRS weights, (ii) Borda count over the per-axis ranks (MMTEB-style), (iii)
  equal-weight mean of min-max axes, (iv) per-axis raw winner-take-all. Report the *rank
  range* each model occupies across all schemes. If embeddinggemma is rank-1 under all
  four, the claim becomes "the winner is invariant to aggregation," which is far stronger
  than "our weighting picks it."
- **cost:** CPU-only, reuses the per-axis headline_numbers.csv. Closes N1 directly, T5.
- **novelty payoff:** the deployment recommendation stops resting on a hand-weighted
  number; it rests on *invariance under aggregation* — exactly the consistency-aware
  framing MMTEB/2605.31142 endorse.

### [feasible-now] A8 — "Where the oracle still loses": the universal-blind 12% characterized
- **what/how:** The ~12% universal-blind core (16/132) is currently an orphan number in
  Deployment (cohesion orphan). Characterize it: what compounds/question-types/languages
  make up those 16 queries? Tabulate them. This both *earns* the 12% in Analysis (fixes
  the orphan) and yields a new claim about what no encoder can do.
- **cost:** CPU-only, the 16 query ids are in round08 output. Closes orphan, N4.
- **novelty payoff:** "the irreducible failures are concentrated in {structure questions
  about methyl/sulfide siblings, in zh/de}" — a precise residual-error portrait, the kind
  reviewers reward, and the setup for the "chemistry-aware help, not more encoders" line.

---

## (b) New metric definitions
*Each: formula/intuition | what it captures that existing metrics don't | cost. Several
are minimal realizable versions of the level-2 CTC/CERC/LSR/ELI/ARGF ideas.*

### [feasible-now] M1 — XRC: Cross-lingual Reading-cost Multiplier (the headline upgrade, depth form)
- **formula:** For model m and target language ℓ, let D₉₅(same) and D₉₅(cross→ℓ) be the
  retrieval depth at which the gold (resp. foreign twin) is found for 95% of queries
  (empirical 95th percentile of first-relevant rank). **XRC(m,ℓ) = D₉₅(cross→ℓ) /
  D₉₅(same).** Report per model (pooled over ℓ) and per language.
- **captures:** a *deployment cost in documents-read*, monotone-invariant and unitful,
  where AUC/recall are unitless quality scores. Recall says "you miss it"; XRC says "you
  pay 9× the reading budget to not miss it." No existing CLIR metric expresses the cost
  as a reading-depth ratio.
- **cost:** CPU-only; the rank distributions are in round04 (first-foreign-rank) and a
  same-language analogue. This is the *empirical* (non-conformal) form of W1 — ship this
  even if the conformal version is backlogged. Closes N1, N4.

### [needs-eval-free but stretch] M2 — XRC-conformal: the distribution-free guarantee (CERC, minimal)
- **formula:** Split-conformal. On a calibration split, for the *same-language* setting
  compute the score threshold τ giving 95% coverage; the implied retrieval depth is the
  same-language conformal budget. Repeat for cross-language. **XRC-conf = depth_cross(95%)
  / depth_same(95%)** with the split-conformal coverage *guarantee* attached (not just an
  empirical percentile).
- **captures:** the same cost as M1 but with a *distribution-free finite-sample coverage
  guarantee* — citing Conformal-RAG (SIGIR 2025) as machinery while asking a genuinely new
  cross-lingual-retrieval-cost question. This is the novelty critic's "single best upgrade."
- **cost:** CPU-only (conformal is a quantile on held-out scores; the per-(query,doc)
  scores must be on disk — likely are, since separability AUC needs them). Needs a
  calibration/test split of the 137 (or 132) queries; small-n is a caveat to state, not a
  blocker. Closes N1 fully, N4, and converts the weakest contribution into the strongest.
- **honest tag:** `feasible-now` IF raw scores are dumped; degrade gracefully to M1 if not.

### [feasible-now] M3 — CCI: Chemical Confusability Index (operationalize the trap as a clean axis)
- **formula:** CCI(m) = E_q[ 1 − (rank of first gold) / (rank of first winning sibling) ]
  clamped to [0,1], or more simply the *confusion rate weighted by sibling ontology
  distance* (cf. A4): Σ_q 1[sibling beats all gold] · w(hop-distance). The hop-distance
  weight turns "any look-alike wins" into "a *near* look-alike wins," which is the
  dangerous case.
- **captures:** a *severity-graded* confusion measure — current confusion rate is binary
  (a look-alike beats all gold: yes/no); CCI grades by how chemically-close the winner is,
  so it distinguishes "beaten by a near-twin" (catastrophic) from "beaten by a distant
  cousin" (noise). No prior metric grades hard-negative wins by ontology distance.
- **cost:** CPU-only, needs the winning-distractor identity + hop-distance (A4 join).
  Closes N4 (makes the confusion finding a *metric*, not just a rate).

### [feasible-now] M4 — LPG: Language-Parity Gini (replace ad-hoc language-parity sub-axis)
- **formula:** Over the per-target-language recall vector r = (r_en,…,r_zh), LPG = 1 −
  Gini(r) (1 = perfectly balanced across languages, 0 = all recall in one language).
- **captures:** a single principled *balance* number with a standard inequality semantics,
  replacing whatever bespoke "language-parity" term currently feeds the composite. Lets the
  paper say "embeddinggemma is the most language-equitable encoder (LPG = …)" with a
  fairness-literature-grounded measure (ties to the fairness-OT cites already present).
- **cost:** CPU-only, one line from pair_recall column means. Closes N1 (de-arbitrarizes a
  composite sub-axis), supports the fairness framing.

### [feasible-now] M5 — RRC: Re-Ranker Recoverability Ceiling (make the falsifiable claim a number)
- **formula:** RRC(m) = fraction of queries whose foreign gold appears within the top-K
  candidate pool a re-ranker would see (e.g. K=100). The paper's central mechanism claim is
  "a monolingual re-ranker cannot recover under-scored foreign twins." RRC is the literal
  *upper bound* on what any re-ranker could achieve: 1 − RRC is provably unrecoverable by
  re-ranking. Already half-present as "15% never in top-1000."
- **captures:** turns the qualitative "re-ranking cannot fix it" into a *quantitative
  ceiling per model* — "even a perfect re-ranker tops out at RRC = 0.62 because the rest
  never enters the candidate pool." This is the falsifiable, deployment-relevant number the
  critic wants the mechanism story to produce.
- **cost:** CPU-only, directly from the first-foreign-rank distribution (round04). Closes
  N2, N4 — and makes "align, don't re-rank" an evidenced bound rather than a slogan.

### [paper-framing-only] M6 — Demote CLIR-MRS to "ordering convenience," promote per-axis dominance
- **what:** Not a new metric — a *re-statement*. Keep CLIR-MRS only to order table rows;
  add one sentence "the winner leads on CLIR@10, RBO, mate-rank, AND separability
  individually (Table 1), so no composite weighting is load-bearing." Pair with A7's
  invariance result.
- **captures:** removes the single highest novelty risk at zero compute. Closes N1.
- **cost:** writing only.

---

## (c) Answers to the feedback
*Every critic problem gets at least one concrete fix; bolder alternatives where useful.*

### [feasible-now] C-B1 — Fix "English is the easiest target" — closes: B1, N3, T2
- **safe fix:** Replace the sentence with the verified pair: "the hardest direction is
  en→de (R@10 0.12) and the most asymmetric pair is de↔zh (gap +0.23)," dropping any
  "easiest target" claim.
- **stretch:** Replace it with A5's hub-and-spoke reading + the T2 corpus-share annotation,
  so the corrected claim is also *more interesting* than the wrong one. **cost:** writing +
  A5 re-aggregation. **payoff:** turns a falsifiable error into a defended structural finding.

### [paper-framing-only] C-B2 — "best model" → "ceiling across models" for RBO 0.19 — closes: B2, G2
- Change intro/conclusion to "the best cross-lingual RBO any model achieves is 0.39
  (alias-graph) / 0.19 (cross-lingual)." Simultaneously fixes the abstract's single-ceiling
  vs body's two-ceiling seam (G2): abstract writes "(0.39 alias-graph; 0.19 cross-lingual)".
  **cost:** ~8 words. **payoff:** abstract↔body↔conclusion become verbatim-consistent.

### [paper-framing-only] C-B3 — One caption sentence on Fig 1 MoLIR population — closes: B3
- Add to Fig 1 caption: "MoLIR@10 is defined only on the 57 original queries (the only ones
  with a same-language gold); the +0.55 home advantage is measured paired within those
  queries." **cost:** one sentence. **payoff:** pre-empts the "green bar is a different N" reviewer.

### [paper-framing-only] C-N1 — The composite: demote + Borda invariance — closes: N1 (highest risk)
- Combine M6 (demote in prose) + A7 (aggregation-sensitivity ribbon, feasible-now) + a
  bootstrap CI (already present). Headline becomes "embeddinggemma is rank-1 under every
  sensible aggregation (Borda, equal-weight, per-axis), so the recommendation does not
  depend on our weights." Cite MMTEB (Borda) and 2605.31142 (rankings are
  aggregation-sensitive — exactly why we test invariance). **cost:** A7 compute + writing.
  **payoff:** converts the weakest contribution into an *invariance* claim reviewers respect.

### [paper-framing-only] C-N2 — Reframe the mechanism as confirm-on-content-controlled — closes: N2
- Rewrite C3 from "first decomposition" to: "we **confirm** the alignment-not-translation
  finding of [2511.19324, 2507.07543] on a **content-controlled parallel corpus that removes
  the translationese/content confounds those studies could not**, and add (i) a
  chemistry-specific confusability trap and (ii) a separability test that turns 'alignment is
  the fix' into a falsifiable re-ranker bound (RRC, M5)." Add both citations. **cost:**
  paragraph rewrite + 2 cites. **payoff:** the content-control is the defensible delta; the
  paper leads with it instead of an attackable "first."

### [paper-framing-only] C-N3 — Cite the precedents inline, narrow every "first" — closes: N3
- One pass adding the mandatory citations *in the same sentence* as each claim: CLEF-IP +
  DAPFAM (C1, narrow "first" to "first content-controlled chemistry-ontology-grounded"),
  CLIRMatrix (directional matrix), Bailey et al. 2017 (cross-lingual RBO lineage),
  AUC-as-separability (standard machinery), Oard 1998 + Saleh & Pecina 2020 (QT-vs-DT budget
  rule), 2605.24297 (English-only patent-embedding overlap — distinguish on cross-lingual).
  **cost:** ~9 bib entries + inline edits. **payoff:** removes all six "you reinvented X"
  attack surfaces at once; this is mostly mechanical but it is the bulk of the novelty defense.

### [feasible-now] C-T1 — Availability-adjusted home advantage — closes: T1
- Ship A2 (residual home advantage after partialling out availability). Adds one clause to
  the abstract/§Results: "(much of which is a gold-availability artifact, §6, where the
  availability-adjusted residual is +X)." **cost:** A2 compute. **payoff:** §Results and §6
  stop reading as contradictory; the home advantage becomes *decomposed*, not double-claimed.

### [paper-framing-only] C-T2 — Corpus-composition caveat on anisotropy — closes: T2
- One sentence in §Results/Limitations: "directional asymmetry partly reflects corpus
  language composition (en 46% / zh 0.4% of documents), not only encoder behavior." Fold into
  A5's hub annotation. **cost:** one sentence. **payoff:** defends against the
  composition/contamination reviewer.

### [feasible-now] C-T5 — n=9 robustness pass — closes: T5
- Ship A3 (drop-the-collapsers recomputation + Spearman). Annotate the load-bearing r's:
  "(n=9 models; r=… on the 7 non-collapsed encoders; Spearman ρ=…)". **cost:** A3 compute.
  **payoff:** the mechanism survives its own most obvious confound.

### [paper-framing-only] C-G1 — Split the §6 fused paragraph by benchmark — closes: G1
- Two sentences, two footnotes (one alias, one chem-patents), exactly as the cohesion critic
  drafted. Optionally lead the rewritten paragraph with A6's *joint* cut ("the modal failure
  is a same-language sibling") so the split paragraph gains a unifying thesis instead of just
  separating. **cost:** writing (+ A6 if used). **payoff:** removes the worst cohesion seam;
  fixes the mis-cited-figure risk.

### [paper-framing-only] C-G3 — Forward bridge Related Work → Benchmarks — closes: G3
- Move the future-work disclaimer up into the calibration paragraph; end §2 with the bridge
  the critic drafted ("Having positioned our four contributions, we now build the benchmarks
  that deliver C1…"). **cost:** move 2 sentences. **payoff:** removes the only hard
  section edge; re-earns C1 right before §3 pays it.

### [paper-framing-only] C-G4 — Move `\todo` to LaTeX comments, single clean deferral sentence — closes: G4
- Replace both in-text red `\todo` blocks with one descriptive sentence each ("Detailed
  corpus-construction statistics are deferred to the system description; all load-bearing
  sizes come from the two benchmark datasets.") and keep the trace note as `% TODO`. **cost:**
  edit two spots. **payoff:** removes the only promise-then-retract-in-real-time moments.

### [paper-framing-only] C-G5 — Fix cp_fig11 caption + name the radar axis — closes: G5
- Re-caption cp_fig11 to describe what the bars show, with +0.96 clearly a text statistic the
  figure *motivates* ("per model, cross-language gold (red) is harder to separate than
  same-language (green); the model-level AUC–CLIR@10 correlation is +0.96, text"). Add to each
  radar one clause naming the winning axis ("embeddinggemma leads on consistency and
  separability, not raw recall"). **cost:** caption edits. **payoff:** figures stop
  under-delivering their captions.

### [paper-framing-only] C-G6 — Signpost the teaser-vs-leaderboard reordering — closes: G6
- One sentence near Table 1: "the order changes from Fig. 1 — ranking by CLIR-MRS rather than
  recall reshuffles the middle of the field, which is the point of the paper." **cost:** one
  sentence. **payoff:** converts an apparent inconsistency into evidence for the thesis.

### [paper-framing-only] C-G-orphans — Earn C5 and the 12% in-text — closes: C5 orphan, 12% orphan
- C5: soften "the reproducible, human-validated pipeline" to "a reproducible pipeline (human
  validation summarized in the system description)" until the numbers are under reports/, so
  the abstract/intro stop promising an unshown payoff. 12%: introduce it in Analysis alongside
  the universal-attractors beat (or via A8's characterization), labeled as an oracle-residual.
  **cost:** writing (+ A8 if used). **payoff:** no orphan promises remain.

### [needs-eval] C-N1-validate — Validate CLIR-MRS against an external criterion (the gold-plated fix)
- **what/how:** The novelty critic's route #1: show CLIR-MRS *predicts* an external utility
  better than mean recall. Minimal version: a small held-out set of human cross-jurisdiction
  search-satisfaction judgments (or end-to-end RAG answer-correctness on a slice), then show
  rank-correlation(CLIR-MRS, utility) > rank-correlation(mean-recall, utility). **cost:**
  needs new human judgments or a downstream RAG eval — backlog. **payoff:** the only thing
  that makes the composite a *validated* contribution rather than demoted; high payoff but not
  required (A7+M6 already de-risk it without this).

---

## Wild cards (highest upside, clearly tagged)

### [feasible-now] W1 — The Cross-Lingual Reading-Cost Multiplier as the paper's NEW headline number
- Promote M1/M2 (XRC / XRC-conformal) from a metric to **the** memorable contribution,
  replacing CLIR-MRS as the thing the abstract leads with: *"finding a patent's foreign-
  language twin at 95% coverage costs N× the reading budget of finding its same-language
  copy; embeddinggemma has the lowest multiplier (N≈…)."* This is the novelty critic's
  explicit "single best upgrade," it's distribution-free, monotone-invariant, deployment-
  legible, half-computed, and directly cashes out "align, don't re-rank." **cost:** A1/M1
  compute now; M2 conformal if scores are on disk. **payoff:** a genuinely new
  cross-lingual-retrieval-cost question with a one-line headline — the paper's most
  defensible and most quotable contribution.

### [feasible-now] W2 — Confusion-rate as a benchmark *axis* with the ontology-distance law
- Elevate the confusion finding (critic: "one of the most original and defensible") to a
  named, reusable benchmark axis (CCI, M3) with the empirical *law* from A4: "confusion rate
  decays with ChEBI hop-distance, r=…". A metric + an empirical regularity is a sticky
  contribution other chemistry-IR benchmarks would adopt. **cost:** A4+M3 compute. **payoff:**
  the paper donates a metric *and* a law to the field, not just numbers.

### [paper-framing-only] W3 — Frame the whole paper as "the cross-lingual tax has two line-items"
- A unifying narrative device that solves the two-benchmark cohesion seam at the story level:
  cross-lingual retrieval pays a **reading-cost tax** (XRC, the depth multiplier — measured on
  the cross-lingual benchmark) and a **confusability tax** (CCI, the look-alike — measured on
  the alias-graph benchmark). Each benchmark *measures one line-item of the same bill*. This
  gives the interleaved benchmarks a single spine ("two taxes, two instruments, one decision")
  and makes the structure feel designed rather than seamed. **cost:** framing only. **payoff:**
  the strongest possible answer to G1/G2 — the benchmarks stop competing for the reader's model.

### [needs-eval] W4 — Causal surgery on the formula token (minimal intervention experiment)
- The formula-token result (p<0.01) is currently observational. A minimal *interventional*
  version: take the structure-question queries that fail, *inject* the language-independent
  formula token, re-retrieve, and measure the recall/confusion delta on the SAME queries
  (paired). If the injection recovers recall, it's a causal claim ("adding H₂S to the query
  measurably rescues retrieval"), not a correlation. **cost:** new retrieval passes on existing
  models (embedding-model runs) — needs-eval, backlog. **payoff:** upgrades a clean
  observation into a causal, actionable query-rewriting rule.

### [paper-framing-only] W5 — "The re-ranker ceiling" as a one-figure falsification kit
- Package M5 (RRC) into a single figure: per model, the bar of "recoverable by re-ranking"
  (foreign gold in top-K) vs "lost forever" (never in top-K). The slogan "align, don't
  re-rank" becomes a *picture of a ceiling* — the most concrete possible form of the
  mechanism claim. **cost:** RRC compute (feasible-now) + one figure. **payoff:** the
  deployment thesis gets its own falsifiable visual.

### [needs-eval] W6 — Cross-lingual QPP: predict the reading-cost from the query alone
- Stretch toward the level-2 cross-lingual QPP idea: can a pre-retrieval signal (query
  language, question-type=structure/role, presence of a formula token) *predict* a query's
  XRC multiplier? A small logistic/regression over the existing per-query features → a
  *router* that decides when to spend extra reading budget. **cost:** modeling over existing
  features (CPU) but to be persuasive wants a held-out eval — tag needs-eval to be safe.
  **payoff:** the "per-language routing could win" hypothesis becomes a *predictive* tool,
  the modern QPP-routing horizon the level-2 PDF marks as future work, partially realized.

---

## Top-3 recommended for this round (editorial pick across channels)

1. **W1/M1 — Cross-Lingual Reading-Cost Multiplier (XRC), feasible-now empirical form.**
   This is the novelty critic's explicitly-named single best upgrade and it closes the
   biggest exposure (N1, the arbitrary composite) by *replacing* its deployment role with a
   distribution-free, monotone-invariant, deployment-legible number that's already half-
   computed from round04/round08. Ship the empirical D₉₅-ratio now (A1/M1); attempt the
   conformal guarantee (M2) only if raw scores are on disk. Highest novelty payoff, low cost.

2. **A7 + M6 — Aggregation-invariance ribbon + demote CLIR-MRS to ordering convenience.**
   Directly neutralizes the highest-risk over-claim: instead of defending a hand-weighted
   number, prove embeddinggemma is rank-1 under Borda / equal-weight / per-axis (cite MMTEB,
   2605.31142). The recommendation then rests on invariance, not on our weights. CPU-only,
   reuses headline_numbers.csv, and is reviewer-disarming.

3. **C-N2 + M5 (RRC) — Reframe the mechanism as "confirm-on-content-controlled" + give
   "align, don't re-rank" a falsifiable ceiling.** Rewrite C3 to cite 2511.19324/2507.07543
   and lead with the content-control delta, and turn the slogan into the Re-Ranker
   Recoverability Ceiling (1 − RRC is provably unrecoverable by re-ranking, computed from the
   first-foreign-rank distribution already on disk). This closes N2 *and* the small-n
   mechanism worry by making the claim a per-model bound, not a single correlation.

**Bundle note:** items 1–3 plus the paper-framing-only fixes (C-B1, C-B2, C-B3, C-N3, C-T1,
C-T2, C-T5, C-G1, C-G3, C-G4, C-G5, C-G6, orphans) close *every* critic problem, are almost
all feasible-now/writing-only, and — critically — none of the three top picks depends on any
code-switch or new-eval result. W3 ("two taxes, one bill") is the recommended *framing
overlay* to bind the two benchmarks if cohesion needs a story-level fix beyond the joint splits.
