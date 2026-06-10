# Novelty review (round 5)

Reviewer #1 — Novelty Critic. EMNLP Industry Track. **Final pre-submission round.**
Round 5 is explicitly a *non-analysis* round: honesty-softening of four claims (C1–C4),
a 4-float net cut, three cheap residuals (R1 ρ_k cite, R2 fig22 caption, R3 §6.1 trim),
and one appendix robustness table. **No new claim is added.** My job this round is the
narrow but important one the conductor named: confirm that the softenings did **not**
accidentally retreat a genuinely-novel contribution *below* the publication bar, and
that nothing new was over-claimed in the process. I re-grounded against my round-4
review (no new web search needed — round 5 adds zero assertions and the closest prior
art for every object was already pinned down in rounds 3–4: CLIRMatrix, BordIRlines/XRAG,
whatdrivesclir2025/crosslingualcost2025, CLEF-IP/DAPFAM, the nogueira/gao cascade cites,
and the ρ_k long-tailed-reranking decomposition).

## Verdict summary (1 paragraph): is the paper's novelty defensible as written?

**Yes — and the softenings *strengthen* the submission rather than weaken it, because
not one of them touched the load-bearing novel core of any contribution.** The four
honesty corrections all retreat on the *inferential strength* of claims whose novelty
was never inferential to begin with: C1 and C3 retreat on the *separability* finding,
but separability is C3's *confirmation* layer (the genuinely-novel parts of C3 — the
chemistry confusability trap, the same-language-sibling modal failure, and the *measured*
alignment-only floor $L_\infty$ — are untouched and remain firm); C2 demotes the
embeddinggemma-vs-qwen3 ARI@100 *ordering* to a tie, but ARI was never claimed as a
contribution (it is "a reading of the RRC curve," now ρ_k-credited), and the C2 retreat
*preserves* embeddinggemma's genuinely-distinct smallest-$L_\infty$ win, which is the
quantity C4's recommendation actually leans on; C4 retreats the τ-rule to a stated band
but *keeps firm* the one unconditional, τ-invariant claim (embeddinggemma is the unique
max-CLIR@10 Pareto corner) that is the deployment take-away. **The benchmarks (C1) — the
paper's safest and strongest novelty — were frozen and untouched.** Critically, I checked
the inverse risk the conductor flagged: a softening can *over-retreat* into a new
negative over-claim ("separability is independent of capability / not a tautology" would
be a *new* claim; an *unqualified* "the rule is robust" would be a *surviving* over-claim).
Neither happened — §7 reads "a descriptive correlate of the cross-lingual floor, not an
effect net of general retrieval capability," the τ-rule states its band and the granite
low-end flip explicitly, and the protected surfaces (abstract / intro contributions /
conclusion) carry **none** of the ARI@100 number, the partial-r, the τ-band, or a routing
claim (verified by grep: every ARI@100 / τ / partial-r site sits in §4/§6/§7/§8/§9/Appendix,
never before line 461). The ρ_k residual-decomposition cite — my round-4 near-mandatory
ask — **is present (line 467), resolves in `custom.bib` (line 390), and is correctly
bounded**: it credits the borrowed "normalize a re-ranking remainder by a recoverable gap"
shape, then distinguishes ARI as the *inverse* ratio with an alignment-only floor "that
$\rho_k$ has no analogue for." **No remaining claim lets a hostile reviewer reject on
novelty grounds, and no last citation is missing.**

---

## The conductor's two questions, answered

### Q-A. Did the softenings (separability now descriptive/collinear; ARI a tie) weaken a genuinely-novel contribution below the bar?

**No. In every case the softening hit a layer that was already credited as
INCREMENTAL/confirmation, not the layer carrying the contribution's novelty.** Walking
the two softened pillars:

- **C1 + C3 (separability → descriptive/collinear).** The separability finding
  $r(\text{cross-AUC},\text{CLIR@10})=+0.96$ was *never* the novel core of C3. In my
  rounds 1–4 verdicts it has always been the **INCREMENTAL-confirmation** layer — the
  paper itself frames "alignment, not translation" as *confirming*
  whatdrivesclir2025/crosslingualcost2025 on a content-controlled corpus, explicitly *not*
  discovering it (§2, lines 216–226). The genuinely-novel parts of C3 are (i) the
  chemistry-specific confusability trap (sibling out-ranks gold $18.1\%$ vs.\ parent
  $6.2\%$; universal attractors), (ii) the same-language-**sibling** modal failure
  ($44.4\%$ of confused cases — the *joint* language×chemistry mode), and (iii) the
  *measured* alignment-only floor $L_\infty=0.058$ tied to a falsifiable per-model
  re-ranker bound (RRC). **All three are untouched and stated firmly** (lines 793–804,
  923–939, 692–698). Softening separability to "a descriptive correlate... not an effect
  net of general retrieval capability" (C3) and to sign-stability-not-magnitude (C1)
  *removes* an over-claim (a causal / capability-independent reading) without removing any
  novel finding. **C3 stays NOVEL on its real contributions; the softening only retires a
  layer that was already INCREMENTAL.** Net: C3 is *safer*, not weaker.

- **C2 (ARI@100 egemma-vs-qwen3 → a tie).** ARI was correctly subordinated in round 4 as
  "a reading of the RRC curve, never a fourth cost object," and that has *not* changed —
  the C2 bullet still lists ARI as "the re-ranker-irreducible share," a framing not a
  method claim. The tie retreat affects only the *ordering* of two models on ARI@100; it
  does **not** touch ARI's defensible novelty (the cross-lingual, mate-twin-resolved
  instantiation of the recall-ceiling decomposition with an alignment floor — INCREMENTAL
  but genuinely new in this setting). Crucially the writer obeyed the story's open-risk #1:
  embeddinggemma **keeps** its separate, still-distinct **smallest alignment-only floor**
  ($L_\infty=0.058$, next qwen3 $0.073$) at every site (lines 703–704, 719, 1088, 1108–1109),
  textually held apart from the now-tied ARI@100. That distinction — not the ARI@100
  ordering — is what C4's deployment recommendation actually rests on, so the recommendation
  spine is intact. **C2 is more honest and loses no novelty.**

**Net on Q-A:** the softenings retreated exactly the inferential/ordering surfaces and
left the novel cores (benchmarks, confusability trap, joint mode, measured $L_\infty$
floor, the τ-invariant max-CLIR corner) firm. No contribution dropped below the bar.

### Q-B. Did the softening process introduce any *new* over-claim, and is any last citation missing?

**No new over-claim, and no last missing citation.** I checked the three ways a softening
can backfire:

1. **Over-retreat into a new negative claim.** A statement that separability is
   "*independent* of capability" / "not a tautology" would itself be a *new* (and false,
   given the n.s. partial-r) claim. **Did not happen** — grep for `independent` / `tautolog`
   / `capability artifact` returns only "language-independent formula token" (unrelated,
   line 985) and the correct "not an effect net of general retrieval capability" (line 1041).
   The framing is descriptive, exactly as the partial-r supports.
2. **Surviving over-claim.** An unqualified "the τ-rule is robust" would survive C4 as an
   over-claim. **Did not happen** — §6.1 (lines 632–638) and the cp_fig18 caption (655–659)
   state the stable band $[0.39,0.43]$, the cheapest-reader band $[0.33,0.44]$, the granite
   low-end flip below $\approx0.33$, and the egemma-only region above $\approx0.45$; the one
   firm claim is the τ-invariant max-CLIR corner. Honest and correctly fenced.
3. **A residual "you reinvented X" surface.** The single such surface was ARI vs.\ the ρ_k
   long-tailed-reranking decomposition. My round-4 near-mandatory cite is now in
   (`residualrerank2026`, custom.bib line 390; cited line 467) and **correctly bounded** —
   it credits the shape and distinguishes ARI as the inverse ratio with an alignment floor
   $\rho_k$ has no analogue for. This was the last citation I would block on; it is closed.

The figure cut also lost no argument: the two radar `\ref`s are gone (only a CUT-NOTE
*comment* remains, line 911), the CUT-NOTE clause carries "embeddinggemma leads
consistency and separability" into the leaderboard paragraph (lines 831–834), the two
stitched panels are each referenced once with one caption/label, and 23 `\includegraphics`
+ 3 tables = 26 floats (25 in-body) with **zero** dangling references to cut/merged figures.
None of this is a novelty surface, but it confirms the trim introduced no orphaned or
inflated claim.

---

## Claim-by-claim (final verdicts; Δ = change since round 4)

| C | Final verdict | Δ this round | Exposure |
|---|---|---|---|
| **C1** two content-controlled patent-grounded benchmarks | **NOVEL, well-defended** | FROZEN, untouched | none — the paper's strongest novelty; narrowed correctly vs.\ CLEF-IP/DAPFAM |
| **C2** metric family (XRC / RRC-knee / $L_\infty$-floor / ARI-reading / DEG gate) | XRC **NOVEL axis** (reading-depth cost on a Pareto frame); RRC **INCREMENTAL knee + NOVEL floor**; **ARI INCREMENTAL-reading, ρ_k-credited**; CLIR-MRS correctly *not* a contribution | ARI@100 demoted to a tie (ordering only); ρ_k cite added | none — ARI never claimed as a method; tie is honest |
| **C3** mechanism (confusability trap + joint sibling mode + separability ⇒ measured $L_\infty$) | **NOVEL** on the trap / joint mode / measured floor; **INCREMENTAL-confirmation** on alignment-not-translation; separability now **descriptive correlate** | separability softened to sign-stable + collinear/descriptive | none — softening retired an over-claim, not a finding |
| **C4** deployment decision (capability corner + τ-band + per-route headroom) | **INCREMENTAL rule + NOVEL frontier-decision** (τ-invariant max-CLIR corner); per-route = bounded headroom | τ-rule banded; ARI@100 tie absorbed; $L_\infty$ win kept | none — the unconditional claim is fenced and firm |
| **C5** QAC pipeline | **INCREMENTAL** (supporting) | FROZEN | none |

**Nothing is exposed to a novelty rejection.** The dead risks from rounds 1–4 stay dead
(CLIR-MRS-as-contribution; "first decomposition"; "cheapest = embeddinggemma"; the
uncredited cascade knee; the ARI residual-decomposition surface). The round-5 softenings
add no new exposure — they remove two.

## Highest-risk over-claims (ranked) — all already handled this round

1. **(Watch, not a defect) The C2 $L_\infty$ distinction must not silently erode into the
   ARI@100 tie in a future copy-edit pass.** Right now it is correctly held apart at all
   five sites. The novel quantity C4 leans on is the *smallest-$L_\infty$* win, not the
   (now tied) ARI@100; if a later trim collapses the two, C4's recommendation would lose
   its only non-tied capability-axis differentiator on the alignment dimension. **No action
   needed now — it is correct as written; flagged only so the camera-ready pass preserves
   the separation.**
2. **(Carried, unchanged) "alignment-only" remains an inference, not an intervention.** C1/C3
   *lowered* this risk by reframing separability as descriptive; the $L_\infty$/ARI floor
   is still correctly tied to the forthcoming causal probe in Limitations (lines 1263–1275).
   Low risk, handled.

No over-claim rises to "a hostile reviewer rejects on novelty."

## Missing citations the paper should add (bib-ready)

**None.** The round-4 near-mandatory ρ_k cite (`residualrerank2026`, arXiv:2604.01506) is
present, resolves, and is correctly bounded. The optional carries from round 4
(CLIRMatrix back-reference for per-route direction; Artetxe/LaBSE Tatoeba "mate accuracy"
lineage) remain genuinely optional and **non-blocking** — the paper ships without them.
There is no new citation surface this round, because there is no new claim.

## What WOULD make the weakest contribution clearly novel (hand to dreamer)

Unchanged from round 4, and correctly held as UPSIDE-only: **the W3 alignment causal
probe with ARI as the before/after target.** Fit a per-language alignment map on one
model, re-retrieve, recompute XRC50 / RRC@100 / **ARI@100 / $L_\infty$**; if the
alignment-only floor *drops* under alignment while staying flat under re-ranking, the
"alignment-only" adjective becomes the paper's headline **causal** result and ARI becomes
the *measured movable quantity* of an intervention (genuinely novel), rather than a
re-expression of $L_\infty$. The round-5 C3 softening makes this *more* valuable: now that
separability is explicitly a descriptive correlate collinear with capability, an
intervention that moves $L_\infty$ is precisely what would convert the descriptive bridge
into a causal one. The story's freeze is right — this is a post-submission run; nothing in
the paper depends on it.

**Convergence note (novelty axis):** the paper is **done on novelty**, and round 5 left it
stronger, not weaker — every softening retired an over-claim while preserving the novel
core, the one residual citation is in and bounded, and the protected surfaces are clean.
Freeze the spine and submit.

---

### Sources (re-grounded from round 4; no new web search needed — zero new claims)
- ρ_k residual decomposition (ARI's credited cousin) — arXiv:2604.01506 (verified present in custom.bib + cited line 467)
- Cascade recall-ceiling (RRC knee) — Nogueira et al. arXiv:1910.14424; Gao et al. arXiv:2101.08751
- Alignment-not-translation (C3 confirmation layer) — *What Drives Cross-lingual Ranking?* arXiv:2511.19324; *The Cross-Lingual Cost* (crosslingualcost2025)
- Content-vs-language confound (C1 defense) — BordIRlines / XRAG / Linguistic Nepotism
- Multilingual patent IR boundary (C1 narrowing) — CLEF-IP, DAPFAM
