# Dreams (round 5)

**Editorial stance up front: the paper is SHIP-READY and the dominant recommendation
this round is FREEZE.** All three round-5 critics converge on "done": novelty approved
and *strengthened* by the softenings (Reviewer #1: "Freeze the spine and submit");
correctness clean except one low-severity rounding nuance (Reviewer #2: MISMATCH=1
low-severity, UNTRACEABLE=0); cohesion the cleanest in five rounds with 25 in-body
floats and only three trivial optional nits (Reviewer #3: "ship after a one-line
typesetting pass; no content edit required"). The honest dreamer move is therefore
**not** to invent new compute — it is to (a) bundle the one correctness fix + three nits
into a single tight cleanup list, and (b) actively *defend the freeze* against the
temptation to add a "nice-to-have" analysis that would add noise, a new float, or a new
claim surface a hostile reviewer could attack. Below I still produce ideas in all three
channels per my mandate, but every channel's headline is: the load-bearing version is
already in the paper, so the recommendation is FREEZE-and-polish.

## Problems on the table (distilled from the 3 critics)

There are **no novelty, no correctness-blocking, and no cohesion problems**. The entire
residual surface is five cosmetic/rounding items:

- **M1 (correctness, low, NEW):** body L632–633 and `cp_fig18` caption L657 display the
  τ-bands rounded to the 0.005 grid — admitted-stable `[0.39,0.43]` vs raw `[0.385,0.43]`;
  bge-cheapest `[0.33,0.44]` vs raw `[0.33,0.435]`. The 0.385→0.39 rounding *narrows*
  (conservative); the 0.435→0.44 rounding *widens* the bge band by one grid step (mildly
  anti-conservative). Changes no conclusion. Reviewer #2 wants either the raw values
  restored or a one-clause "(rounded to the 0.005 τ-grid)" disclosure.
- **N1 (cohesion, trivial, inherited):** `cp_fig22` (ARI) float block at L713 sits *before*
  `cp_fig19` (RRC budget) at L728, while the prose reads RRC→ARI. Pure source-order
  float-placement nit; LaTeX will likely co-locate them anyway. Deferred since round 3.
- **N2 (cohesion, trivial):** `cp_fig09_10_collapse` caption (L973) and Analysis prose
  (L955) say "49×" while the rendered left panel labels the Chinese bar "48.7×". Same
  number, defensible round; a sharp reader sees a 0.3 mismatch caption-vs-panel.
- **N3 (cohesion, trivial, carried):** $L_\infty$ appears in two notations — "0.058"
  (692/703/719/734/1036/1088/1289) and "5.84%" (1030/1191). Same number, locally clear.
- **(Watch, not a defect — Reviewer #1 #1):** the C2 $L_\infty$=0.058 distinction must not
  silently collapse into the now-tied ARI@100 in a camera-ready copy-edit. Correct as
  written at all five sites; flagged only so the polish pass preserves the separation. **No
  edit now.**

That is the complete problem set. Everything below is graded against: does it beat doing
*nothing but* fixing these five?

---

## (a) New analyses

### [paper-framing-only | FREEZE] Holistic "load-bearing scalar + resample CI" ledger — ALREADY COVERED, do not add
- **what / how.** The conductor asked whether a single holistic "every load-bearing scalar
  + its resample CI" summary is missing. It is **not missing** — it is exactly the round-4
  `tab:robust` appendix table (six rows: separability r, three frontier XRC50 medians,
  ARI@100 gap, partial-r), verified cell-for-cell by Reviewer #2 and referenced from the
  body *five times* (§6.1 L707, §7 L1023 & L1042, cp_fig11 caption L1052, §9 L1212).
  Reviewer #3 calls this "the round's cleanest new joint" and "a model credibility move."
  | cost: zero (already shipped). | payoff: none new — adding a *second* ledger or
  expanding this one would duplicate a clean asset, add words, and risk a sixth-vs-seventh
  scalar inconsistency. **Recommendation: this idea is satisfied; do not extend the table.**

### [feasible-now | DO NOT RUN — noise, not signal] Leave-one-language-out (LOLO) on the headline separability r
- **what / how.** The headline $r(\text{cross-AUC},\text{CLIR@10})=+0.96$ is already
  stress-tested two ways that *dominate* a LOLO: (i) the drop-the-two-collapsed-encoders
  jackknife (n=7→+0.958, n=9→+0.888) and (ii) the sign-stability bootstrap
  ($P(r>0)=0.9997$, wide 95% CI [0.73,1.00]). Note this correlation is *across the seven
  non-degenerate **encoders***, not across languages, so a "leave-one-language-out" on
  *this* scalar is not even well-defined — the resample unit is the model, and the
  model-jackknife is already in `tab:robust`. A LOLO would only apply to the *per-language*
  descriptive slopes (home-advantage −0.57 over n=5; availability), which are already
  fenced as descriptive/n=5 and deliberately kept out of abstract/intro/conclusion. | cost:
  CPU-only, ~1 hr re-resample. | payoff: **negative** — it would (1) re-confirm an
  already-fenced descriptive slope with an even smaller n, (2) tempt a new appendix row or
  float, (3) invite a reviewer to ask "why LOLO here but not on the encoder axis?"
  **Recommendation: DO NOT RUN. The model-jackknife already does the strictly-stronger job.**

### [paper-framing-only | OPTIONAL one-liner, NOT a float] Single sign-test sentence as the universal robustness frame
- **what / how.** If the writer wants *one* extra defensive sentence (not a float, not a
  number table), §7/§9 could state once that "every headline scalar that enters the
  abstract or conclusion is reported with a resample interval or a sign-stability
  probability in Appendix~\ref{app:robust}." This is a *framing* sentence that turns the
  existing `tab:robust` into an explicit promise. | cost: one sentence, zero compute, zero
  float. closes: nothing (no critic asked) — pure reviewer-confidence varnish. | payoff:
  marginal; only worth it if a co-author feels the ledger is under-advertised. Reviewer #3
  already judged it well-advertised (5 body refs). **Recommendation: optional, low value;
  skip unless a human wants it.**

---

## (b) New metric definitions

### [needs-eval | FREEZE — post-submission only] No new metric this round
- **what / how.** The metric family (XRC, RRC-knee, $L_\infty$-floor, ARI-reading, DEG
  gate, CLIR-MRS) is approved by Reviewer #1 as a NOVEL axis + INCREMENTAL knee + NOVEL
  floor, with ARI now ρ_k-credited and CLIR-MRS correctly *not* claimed as a contribution.
  The level-2 directions (CTC, CERC, LSR, ELI, ARGF) and the W3 alignment causal probe all
  require **new embedding-model runs** and are explicitly held as post-submission UPSIDE by
  both the story freeze and Reviewer #1's "hand to dreamer" note. | cost: new evals
  (backlogged, not run this round). | payoff: would be a *future* paper's headline, not
  this one's. **Recommendation: FREEZE. A new metric this late adds a claim surface with no
  time to harden it — the exact thing rounds 1–4 worked to eliminate.**

### [paper-framing-only | the ONE thing that would beat freeze IF it were free — it isn't] W3 alignment causal probe → makes "alignment-only" causal
- **what / how (for the backlog, not this round).** Reviewer #1's named upside: fit a
  per-language alignment map on one model, re-retrieve, recompute XRC50 / RRC@100 / ARI@100
  / $L_\infty$. If $L_\infty$ *drops* under alignment while staying flat under re-ranking,
  the "alignment-only" adjective becomes a **causal** headline and ARI becomes the measured
  movable quantity of an intervention. The round-5 C3 softening (separability now
  *descriptive* / collinear) makes this *more* valuable, because an intervention that moves
  $L_\infty$ is exactly what converts the descriptive bridge into a causal one. | cost:
  **needs-eval** — new retrieval runs + an alignment fit; cannot be CPU-only-from-existing-
  data. | payoff: high, but for the *next* paper. **Recommendation: leave frozen in
  Limitations as forthcoming (L1263–1275). Nothing in the current paper depends on it, and
  the current submission stands without it — which is the correct posture.**

---

## (c) Answers to the feedback

### [feasible-now | DO THIS — the round's only required edit] M1 τ-band fix (pick ONE option)
- **closes:** M1 (Reviewer #2's single low-severity MISMATCH). | **what/how:** the cleaner
  of Reviewer #2's two options for a polished industry-track camera-ready is to **restore
  the raw values** so display==source and the question never arises:
  - L632: `stable for $\tau \in [0.39, 0.43]$` → `stable for $\tau \in [0.385, 0.43]$`
  - L633: `cheapest admitted reader for $\tau \in [0.33, 0.44]$` → `... $\tau \in [0.33, 0.435]$`
  - L657 (`cp_fig18` caption): `stable for $\tau \in [0.33, 0.44]$` → `... $\tau \in [0.33, 0.435]$`

  (Alternative, equally accepted by Reviewer #2: keep the 2dp display and append once at
  first use "(rounded to the 0.005 τ-grid; raw $[0.385,0.43]$ / $[0.33,0.435]$)". The
  restore-raw option is shorter and removes the discrepancy entirely, so it is preferred.)
  **Do NOT touch** the granite-flip clause (granite CLIR@10=0.3285 → admitted at τ≤0.3285 ≈
  0.33), the "only egemma above τ≈0.45," or the "egemma corner τ-invariant" — Reviewer #2
  verified all three correct against source. | cost: 3 single-token edits, zero compute. |
  payoff: closes the only logged correctness mismatch; display now matches source exactly.

### [feasible-now | DO THIS — trivial harmonization] N2 49× ↔ 48.7× caption/panel
- **closes:** cohesion nit #2. | **what/how:** harmonize the two prose sites to the panel's
  exact rendered value so caption==panel:
  - L955 (Analysis prose): `up to $49\times$` → `up to $48.7\times$`
  - L973 (`cp_fig09_10_collapse` caption): `up to $49\times$ the corpus base rate` →
    `up to $48.7\times$ the corpus base rate`

  (Reviewer #3's alternative "$\sim$$49\times$" also acceptable; matching the panel's
  48.7× exactly is cleanest and removes the 0.3 caption-vs-panel gap entirely.) | cost: 2
  single-token edits. | payoff: removes the only caption-vs-figure number mismatch; sharp
  reviewers see consistency.

### [feasible-now | DO THIS — one-time standardization] N3 $L_\infty$ dual notation
- **closes:** cohesion nit #3. | **what/how:** standardize to "$0.058$ ($5.84\%$)" at the
  **first body use only** (L692) and leave "$0.058$" everywhere thereafter; convert the two
  bare-percentage sites (L1030, L1191) to "$0.058$" for one notation in running prose. |
  cost: ~3 edits, zero compute. | payoff: single notation for the paper's most-cited
  alignment-floor scalar; cosmetic tightness.

### [paper-framing-only | DO THIS ON TYPESET PASS, not now] N1 float order cp_fig22 ↔ cp_fig19
- **closes:** cohesion nit #1. | **what/how:** on the final typesetting pass, move the
  `cp_fig19` (RRC budget) `\begin{figure}` block (L728) above the `cp_fig22` (ARI) block
  (L713) so source order matches prose order (RRC→ARI). Or let LaTeX float — Reviewer #3
  says it will likely co-locate them anyway. | cost: one block swap, no content. | payoff:
  cosmetic; lowest priority; explicitly deferrable.

### [paper-framing-only | NO EDIT — flag for camera-ready only] Preserve the $L_\infty$ vs ARI@100 separation
- **closes:** Reviewer #1's "Watch" #1. | **what/how:** make **no change now** — the
  separation is correct at all five sites. Add a note to the camera-ready checklist that any
  future copy-edit must keep embeddinggemma's *smallest-$L_\infty$=0.058 floor* (the C4
  differentiator) textually apart from the now-*tied* ARI@100. | cost: zero. | payoff:
  prevents a future regression that would cost C4 its only non-tied capability-axis
  differentiator. **This is a guardrail, not an edit.**

---

## Wild cards (highest upside, clearly tagged) — all DEFERRED, none for this round

### [needs-eval | post-submission] Equivalence Audit on the parallel gold
- The T5 deferred parallel-gold claim-level equivalence audit + spot-check (Limitations
  L1263+) would convert "content-controlled" from a construction guarantee into a *measured*
  one. High credibility payoff, but needs human annotation / new eval — correctly deferred.
  **Not this round.**

### [needs-eval | post-submission] ARI as the before/after target of a causal alignment surgery
- Same as the W3 probe in (b): the single highest-novelty future move, makes the headline
  causal. Backlog. **Not this round.**

### [paper-framing-only | DECLINE] A "contamination audit" sidebar
- Could pre-empt a "the encoders saw these patents in pretraining" reviewer question with a
  framing paragraph. But the content-controlled design + the human-translation-only source
  constraint already blunt this, and adding a defensive sidebar this late risks *inviting*
  the very doubt it answers. **Decline — the silence is stronger than the sidebar here.**

---

## Top-3 recommended for this round (editorial pick across the three channels)

1. **[feasible-now] Ship the tight cleanup list (5 edits, zero compute, zero new float,
   zero new claim):**
   - **M1** — restore raw τ-bands: L632 `[0.39,0.43]`→`[0.385,0.43]`; L633 `[0.33,0.44]`→
     `[0.33,0.435]`; L657 caption `[0.33,0.44]`→`[0.33,0.435]`. (Do not touch granite-flip
     / egemma-corner clauses.)
   - **N2** — 49×→48.7× at L955 (prose) and L973 (caption) to match the rendered panel.
   - **N3** — $L_\infty$ to "$0.058$ ($5.84\%$)" at first use (L692), "$0.058$" elsewhere
     (incl. L1030, L1191).
   - **N1** — defer to the typesetting pass (swap cp_fig19 above cp_fig22, or let LaTeX float).
   - **Guardrail (no edit)** — camera-ready must keep $L_\infty$=0.058 separate from the
     tied ARI@100.

   This closes 100% of the logged residual surface and is the entire required scope of the
   round.

2. **[paper-framing-only] FREEZE the analysis spine — explicitly decline LOLO and any new
   metric/float.** The holistic scalar+CI ledger the conductor asked about already exists as
   `tab:robust` (6 rows, 5 body refs, verified cell-for-cell); a leave-one-language-out on
   the headline is *ill-defined* (the separability resample unit is the encoder, and the
   model-jackknife already covers it) and would only re-touch an already-fenced n=5
   descriptive slope. Any further CPU analysis adds noise, not signal, and risks a new
   inconsistency at the exact moment the paper has none.

3. **[needs-eval] Park the W3 alignment causal probe as the named post-submission upside**
   (already in Limitations). It is the one genuinely high-value next move — the round-5 C3
   softening *increases* its value by making "alignment-only" a descriptive bridge that an
   intervention moving $L_\infty$ would turn causal — but it requires new evals, the current
   paper stands without it, and adding it now would re-open a hardening cycle. Backlog it;
   do not run.

---

### Freeze recommendation (one line)
**FREEZE: do only the 5-item cleanup list (M1 raw-τ-band + the 3 trivial nits + the
no-edit $L_\infty$/ARI guardrail), add no analysis, no metric, no float — the paper is
done and further compute adds noise not signal; wind the chain down to polish-only.**
