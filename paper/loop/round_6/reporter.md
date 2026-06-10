# Reporter handoff (round 6) -> CLOSING HANDOFF (no round 7 unless user resumes)

## TL;DR (what's new and what it means)
Round 6 was a length-compliance restructure + bib fix — ZERO new compute. The experiment
sub-chain is a confirmed no-op (frozen since round 5; nothing under reports/ changed). The
paper is now submission-ready AND restructured to fit the real EMNLP Industry Track 6-page
body limit: the body carries exactly 4 figures + 2 tables (was 7 body floats), with ~19
figures + tab:robust pushed behind a single \appendix. The bibliography is verified (0
author={Anonymous}, real authors/titles/DOIs). Cohesion critic estimates the body at
~5.8-6.05 pages (lucky compile) up to ~6.3 (unlucky) — so the ONE remaining action is a real
Overleaf compile to confirm exact page count; if it overflows, a pre-staged prose cut (Cut 2)
is ready.

## Verification results (all four checks PASS)

### 1. git diff --stat — only the expected changes this round
- Working tree vs HEAD: paper/main.tex ONLY (367 insertions / 434 deletions — the restructure).
- paper/custom.bib was committed earlier this session as 5e25d8d ("paper: verify+fix
  bibliography metadata"; replaced 7 Anonymous + 2 weak author fields, fixed fabricated titles).
- paper/loop/round_6/* is untracked (the round's loop artifacts).
- NO new reports/runs/extra_* dirs; git status reports/ clean; NO untracked files outside
  round_6/; NO new figures generated.

### 2. Body float count — 4 figures + 2 tables (6 body floats); \appendix once
- \appendix appears exactly once (main.tex line 1019).
- Total in file: 23 \begin{figure} + 3 \begin{table}.
- Before line 1019 (body): figures L172, 517, 572, 801 = 4 figures; tables L649, L674 =
  2 tables -> 6 body floats. (4 spine figures: teaser, cost_frontier, rrc_budget, cp_sep
  + 2 tables carrying C1-C4.)
- After line 1019 (appendix): 19 figures (L1029-1221) + 1 table (L1246) = tab:robust
  (label confirmed L1267). Matches "~19 figures + tab:robust after the appendix."

### 3. The 3 round-6 critics all passed
- Correctness (critic_correctness.md): headline "0 MISMATCH, 0 UNTRACEABLE"; Blocking
  issues — NONE.
- Novelty (critic_novelty.md): "Freeze and submit." The 7p->1p Related-Work compression is
  a zero-key-diff (all 28 \cite keys survive, all 8 novelty-boundary keys intact); corrected
  titles introduce no citation-claim mismatch; several hedges sharpened.
- Cohesion (critic_cohesion.md): reads as one story; the 2 fixes are applied — (a) ag_fig2
  (fig:ag_conf) moved to Appendix B (now L1112; cut-note L703; 14-78% confusion still in
  prose + tab:ag_board conf column; single body \ref retained), and (b) appendix-ref labels
  present (tab:robust referenced in body L1234, labeled in appendix L1267).

### 4. R5 cleanup persisted + bib clean
- tau-bands: [0.385, 0.43] (L502) and [0.33, 0.435] (L503, L527, L834).
- 48.7x over-fetch (L741, L1088).
- L-inf: 0.058 (5.84%) (L553, \mathbf{0.058} + (5.84\%); reused at L556/564/580/787/793/
  837/858/906/1004/1077).
- Bib: 0 author={Anonymous} (and 0 case-insensitive "anonymous").

## Discrepancies / unverifiable claims
- None that affect soundness. Two notes for the record:
  - No implement_report.md in round_6/; the round's implementation is the main.tex restructure
    itself (captured in draft.tex + writer_notes.md + figures_manifest.md). Expected for a
    no-compute restructure round — not a discrepancy.
  - The "~5.8-6.05pp" estimate is a line-density estimate, not a compiled measurement (cohesion
    critic basis: ~830 counted prose lines ~= 4.3-4.4 pages text + ~1.7-1.9 pages floats +
    ~0.35 page front matter = ~6.0-6.3pp; lucky-compile lands at/under 6). This is the single
    thing a human must confirm by compiling.

## Changed files this round (git diff --stat summary)
  paper/main.tex     | 367++/434-- (restructure: RW 7p->1p, 3 floats->appendix, 1 \appendix)
  paper/custom.bib   | committed earlier as 5e25d8d (7 Anonymous + 2 weak authors; titles)
  paper/loop/round_6/* (untracked loop artifacts: 3 critics, draft.tex, story.md, manifests)
No reports/, no figures/, no data/ changes. Compute remains frozen.

## The ONE remaining verification the user must do
Compile paper/main.tex on Overleaf (ACL [review] style) and read the exact body page count.
Estimate ~5.8-6.05pp (lucky) to ~6.3pp (unlucky); limit is a hard 6.
- If <= 6 pp: done — submit.
- If > 6 pp: apply the cohesion critic's pre-staged Cut 2 (prose, not floats) — do NOT cut
  below the 4 spine figures + 2 tables:
  1. Trim the Deployment per-route paragraph (~L852-877): delete the thin-sample mechanics
     (~L873-877) that are verbatim-redundant with the Limitations per-route paragraph
     (~L949-957) — same n_same=7/2/0, same "indicative XRC axis," same "es undefined, never
     imputed." Let Limitations own them. Saves ~0.1 page, zero info loss.
  2. In the "Deploy embeddinggemma" paragraph (~L827-850), compress the tau-band restatement
     (~L838-841) to "...over the stated tau-band (sec ref ssec:cp)" and drop the inline
     [0.33,0.435]/tau~0.33 repetition (already stated L502-503 + caption L526-528).

## Backlogged (forthcoming) experiments — 6 items, UNCHANGED this round
All eval-gated (need new embeddings/human/RAG signal); none touched in round 6. Per the
needs_eval contract, the paper already describes these as forthcoming and critics treat them
as done-for-now. Source: paper/loop/needs_eval.md.
1. W4-formula-injection (r1) — re-retrieve structure-style alias queries with the chemical
   formula token injected; upgrades the p<0.01 observation to a causal intervention.
2. CLIRMRS-external-validation (r1) — held-out human/RAG utility signal; converts CLIR-MRS
   from convenience metric to validated contribution (novelty route #1).
3. XRC-conformal-M2 (r1, OPTIONAL) — split-conformal XRC; deferred (only 57 same-lang-gold
   queries -> calibration/test split too thin for a credible guarantee).
4. CCI-hop-distance-law (r1) — ChEBI-graph hop-distance vs confusion-rate "decay law";
   CPU-only but needs non-trivial graph build (on-disk relation field is binary).
5. equivalence-audit-spotcheck (r1) — expert spot-check that parallel golds are claim-
   equivalent; pre-empts the "how do you know golds are equivalent?" review.
6. W3-alignment-causal-probe (r2; UPSIDE ONLY) — fit a per-language alignment map on one model,
   re-embed, recompute XRC50/RRC@100 before/after; elevates "align, don't re-rank" from
   correlational to demonstrated. cp_fig22's per-model L-inf floor is the natural target.

## Recommended next-round focus (only if user resumes the loop)
- No story/writer round is needed to submit. The next action is operational, not narrative:
  compile on Overleaf -> confirm <=6pp -> submit (apply Cut 2 only if over).
- If the user wants to strengthen rather than ship: highest-leverage backlog item is
  W3-alignment-causal-probe (re-embed of one model) — the single experiment that flips the
  paper's central recommendation ("align, don't re-rank") from correlational to causal, and
  the target (cp_fig22 L-inf=0.058 floor) is already in the paper. Second-highest:
  CLIRMRS-external-validation (needs new human/RAG eval) to validate the demoted CLIR-MRS.
- Do NOT add body floats or restart compute — the round-5 freeze holds and the page budget is
  the binding constraint.
