# Writer notes — round 3 (consolidation round)

Round 3 turned last round's loose ends (the dead "cheapest = embeddinggemma"
superlative, the four ad-hoc "degenerate" uses, the missing two-benchmark spine)
into first-class, defensible paper objects, landing four new figures
(cp_fig18–21) with their non-significance caveats **inline**. No fabricated
numbers; every value traces to an `extra_*/summary.json` opened this round. Self-
lint clean (braces 756/756; 25/25 figures; 2/2 tables; 3/3 equations; all
`\includegraphics` exist; all `\cite` keys in custom.bib; zero red `\todo` in
body; no ref/label orphans).

## Sections touched
- **Abstract** — added L∞=5.8% to the cost sentence; rewrote the closing model-
  pick clause to "Pareto-optimal capability corner (not the cheapest reader)".
  No non-significant correlation present (honesty note honored).
- **Introduction** — C2/C4 bullets upgraded (frontier + budget-knee framing;
  "CLIRMatrix-style" attribution added to the directional matrix in C2 per
  novelty nit #3; DEG gate named in C2). Cost clause now ends "the model choice is
  a cost-vs-capability frontier, not a single cheapest model." No leaks.
- **§4 Metrics** — (a) XRC paragraph: added the **population-level (not paired)**
  clause (D_same over 57, D_cross over 137; closes correctness T-NEW) plus the
  monotone-invariance one-liner (novelty nit #2). (b) RRC paragraph: added the
  "mate-hit@K renamed; our contribution is the budget-curve reading (knee K*,
  floor L∞)" hedge (novelty over-claim #1). (c) NEW **degeneracy-gate paragraph**:
  locked DEG = CLIR@10 < 0.10 ONCE, flags exactly {gte, e5}, with the
  single-vs-AND-criterion footnote; introduced **cp_fig20** here.
- **§5 Setup** — reproducibility line now names the three new
  `extra_{cost_frontier,rrc_budget_frontier,two_tax_degeneracy}.py` scripts.
- **§6.1** — cost beat rewritten around **cp_fig18 cost frontier** (3 members
  {egemma,bge-m3,granite}; egemma unique max-CLIR corner; NOT cheapest, bge-m3 at
  τ=0.40 is min-XRC admitted; τ stated-not-tuned said explicitly; trap framed as
  directional read-off with ρ=+0.29 p=0.53 n.s. inline). RRC beat rewritten around
  **cp_fig19 budget curve** (knee K*=5, RRC@100=0.7445, RRC@1000=0.9416,
  L∞=0.058 alignment-only floor; floor range 0.058→0.372; regression-checked →
  stated firmly). cp_fig15 and cp_fig16 retired (replaced).
- **§6.2** — NEW two-tax **bridge** opener with **cp_fig21**: "second line-item of
  the same bill"; caveat inline ("ρ=−0.59, n=7, p=0.16, n.s. → neither benchmark
  is a clean proxy for the other"; framed descriptive/motivating, NOT
  independence). **B2 line-605 fix**: "Even the best model" → "the best
  cross-lingual RBO any of the nine models reaches … a ceiling no model beats."
- **§6.3** — winner-take-all footnote re-anchored to the DEG gate (gte CLIR@10=0.000
  below the gate of §4).
- **§7 Analysis** — N1 soften ("statistically indistinguishable" → "nearly tied,
  within 0.01"; no test was run); de↔zh +0.23 folded into cp_fig03 caption (orphan
  closed). Separability crux back-references the RRC floor with K*=5 / RRC@100=0.74
  / L∞=5.84% so it sets up Deployment.
- **§8 Deployment** — N2 superlative killed; rewritten to "capability corner of the
  cost frontier … not the cheapest deployable model: bge-m3 reads shallower
  (2.0× vs 3.5×) … a frontier choice." NEW "Budget the re-ranker by the knee"
  paragraph (K*≈5, ≤20 for nearly all). "Align" paragraph now names L∞ as the
  alignment-only floor.
- **§9 Limitations** — added the two non-significant correlations explicitly
  (two-tax ρ=−0.59 p=0.16; trap ρ=+0.29 p=0.53; both descriptive/motivating) and
  contrasted with the regression-checked RRC budget objects. Added the forthcoming
  W3 alignment causal probe as upside-only ("none of the results above depend on
  it").
- **§10 Conclusion** — added the knee K*=5 / floor L∞=0.058 to the RRC clause and
  the "cost-vs-capability frontier … embeddinggemma Pareto-optimal capability
  corner (bge-m3 the cheaper-to-read alternative)" framing. Two RBO ceilings kept
  verbatim. No non-significant correlation present.

## Critic points addressed (round 2 → round 3)
### Correctness
- **N1 (overclaim)** — FIXED: "statistically indistinguishable" → "nearly tied
  (within 0.01)". §7 anisotropy.
- **N2 (MISMATCH, the dead superlative)** — FIXED everywhere: Deployment §8
  rewritten to the Pareto-corner framing; abstract/intro/conclusion all carry
  "capability corner, not cheapest"; bge-m3 named as the cheaper-to-read admitted
  alternative at the **stated** τ=0.40.
- **T-NEW (XRC population)** — FIXED: population-level (not paired) clause added to
  the §4 XRC definition.
- **T-MINOR (universal-blind language order)** — left as "predominantly French,
  Chinese, and German"; non-blocking per critic. `% writer-response:` not added
  since the critic explicitly marked it optional/harmless. (Deferred, not ignored.)

### Cohesion
- **#1 (B2 line-605 "best model")** — FIXED: "any of the nine models … ceiling no
  model beats."
- **#2 (de↔zh orphan)** — FIXED: +0.23 folded into cp_fig03 caption.
- **#3 (two-tax spine missing)** — FIXED: §6.2 opens with the "two line-items of
  the same bill" bridge backed by cp_fig21, caveat inline.
- **#4 ("degenerate" undefined, used 4×)** — FIXED: DEG gate defined ONCE in §4
  (cp_fig20), every later use anchored to it.
- **#5 (RRC two-K signposting)** — FIXED: §6.1 now says "leaving ~25% on the
  table … L∞=0.058 unrecoverable by any re-ranker" in one breath.
- **home-advantage hyphenation** — cosmetic, left for a final copy-edit pass (not
  load-bearing).

### Novelty
- **Over-claim #1 (RRC = renamed recall)** — FIXED: §4 RRC paragraph now says
  "mate-hit@K restricted to cross-lingual queries; our contribution is the
  budget-curve reading, not the quantity."
- **#2 (XRC monotone-invariance asserted)** — FIXED: one-line reason added in §4.
- **#3 (directional matrix reads proprietary in C2)** — FIXED: "(CLIRMatrix-
  style)" added to the C2 bullet.
- **Dreamer route #1 (RRC budget frontier)** — REALIZED as cp_fig19 (knee + L∞),
  which is exactly the "re-ranker-budget planning tool" the novelty critic flagged
  as the highest-leverage upgrade still on the table.
- Optional bib adds (cascade recall-ceiling ref; Artetxe/Schwenk mate lineage) —
  NOT added; both were explicitly optional and the paper is citation-defensible
  without them. Deferred to a future round if a reviewer presses.

## Honesty discipline (anti-fabrication focus of this round)
- Two-tax ρ=−0.59 (n=7, p=0.16) and trap ρ=+0.29 (n=7, p=0.53): both stated as
  **n.s.**, descriptive/motivating only, **inline** at their figures and in
  Limitations; **absent** from abstract/intro/conclusion.
- "Neither benchmark is a clean proxy for the other" — used; "the two taxes are
  independent / non-redundant (proven)" — never used.
- Trap: "the cheapest reader CAN be among the worst retrievers" (read-off of the
  figure), never "IS".
- τ=0.40 stated-not-tuned: said explicitly at the admitted-set claim.
- RRC budget object (K*, L∞): regression-checked, stated firmly (no hedge) — the
  one new object with no significance caveat.
- DEG gate: single-criterion CLIR@10<0.10 (not the AND-gate), stated once.

## Open \todo / % TODO items (carried, all honestly fenced — none new)
1. Corpus dedup count (14,401) + GP/EPO/JRC coverage matrix — `% TODO` (§3 corpus
   paragraph). Source is system slides, not yet under reports/.
2. Human-eval numbers (mean 8.33/10, items reviewed, auto-grader strictness) —
   `% TODO` ×2 (§3 pipeline paragraph + Appendix C5). System slides only.
3. MMTEB/MIRACL/NeuCLIR general-domain transfer measurement — `% TODO` (§9 domain
   transfer). Backlogged in needs_eval.
None of these are load-bearing for any headline claim; all four are `% TODO`
LaTeX comments (invisible to the reader), zero red `\todo{}` markers in body text.

## Retired figures
cp_fig15 (old XRC scatter) and cp_fig16 (old RRC bar) — superseded by cp_fig18 and
cp_fig19; both delabeled and dereferenced (0 occurrences in main.tex).
