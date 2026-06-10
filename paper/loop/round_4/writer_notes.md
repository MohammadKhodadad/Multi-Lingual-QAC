# Writer notes (round 4)

Round 4 is the last substantive-addition round: wire cp_fig22 (ARI) + cp_fig23
(per-route frontier), land the three closing fixes F1–F3, then freeze the spine.
All numbers traced to `reports/runs/chem_patents/experimental_plots/extra_*/`;
no fabricated values. Self-lint passed (see bottom).

## Sections touched
- **Abstract** — one tight clause on the $L_\infty$ sentence: "a measured share
  recoverable only by alignment, not re-ranking." No new sentence; word count
  unchanged (316). No ARI number, no routing, no n.s. correlation. (story-permitted)
- **§1 Introduction (contributions list)** — C2 bullet: added the ARI clause
  (re-ranker-recoverable / deeper-pool / alignment-only share summing to one).
  C4 bullet: added the per-route-frontier upside-headroom clause. Body prose
  untouched (FREEZE).
- **§2 Related Work** — F3 cascade cite is placed at the RRC definition in §4 (see
  below) rather than §2; story allowed either. (No §2 prose change; the six
  boundaries stayed CLOSED.)
- **§4 Metrics** — (a) **F3:** added the cascade/recall-ceiling credit sentence to
  the RRC paragraph with `\citep{nogueira2019multistage,gao2021rethink}` — claims
  only the cross-lingual quantification + $L_\infty$. (b) **NEW ARI paragraph**
  ("ARI: the alignment-recoverability decomposition") immediately after the RRC
  budget-curve paragraph: states the additive identity (three parts sum to one)
  and the scalar Eq.~\ref{eq:ari} $\mathrm{ARI@}K=L_\infty/(1-\mathrm{RRC@}K)$,
  cites fig:ari.
- **§5 Experimental Setup** — reproducibility paragraph now lists
  `extra_ari_decomposition.py` and `extra_per_route_frontier.py`.
- **§6.1** — (a) **F4:** cost-frontier paragraph now credits "the standard
  cost-vs-capability Pareto frame `\citep{syftr2025}`, with XRC ... as the cost
  axis." (b) **F1 site #1** (was line 557): "eight non-degenerate models" →
  "the eight models with a defined cross-lingual recall (all but the degenerate
  `gte-base`)." (c) **F6:** the $L_\infty$ range endpoint relabeled "$0.372$ for
  the degenerate `e5-large-instruct`, which the gate excludes." (d) **NEW ARI
  read-off paragraph** + fig:ari float: egemma $\mathrm{ARI@}100=0.229$ lowest
  non-deg, smallest floor $0.058$, next qwen3 $0.233$, identity sums to 1.0 for
  all nine.
- **§6.1 fig:rrc_budget caption** — **F1 site #2** (was line 657): "eight
  non-degenerate models" → "the eight models with a defined RRC curve (all but
  `gte-base`, whose candidate pool is empty)."
- **§6.2 fig:two_tax caption** — **F2:** "(sibling-confusion rate ...)" →
  "(confusion rate, alias-graph benchmark)". Value 0.182/0.068 untouched; the
  separate sibling-vs-parent severity split (§ssec:ag, 18.1% vs 6.2%) left alone.
- **§7 Analysis** — separability-floor crux: added the ARI back-reference
  ("the ARI decomposition (§6.1) shows this floor is the *only* alignment-bound
  part ... just $\mathrm{ARI@}100=0.229$ ... lowest of any non-degenerate model").
  The two hedged fragile correlations and the robust +0.96 were NOT touched.
- **§7** — T-MINOR polish: universal-blind language list "French, Chinese, and
  German" → "predominantly French and Chinese" (de/es tie at 3).
- **§8 Deployment** — (a) deploy-egemma paragraph: one-clause ARI reinforcement
  ("smallest alignment-only floor and lowest $\mathrm{ARI@}100$ ($0.229$)").
  (b) **NEW per-route paragraph** + fig:per_route float under the four-point
  honesty contract: single-model default (egemma global corner, wins 3/5 incl.
  hardest es/zh) + router-as-headroom (en→qwen3, fr→nomic; flips 2/5) + thin-n
  caveat (de=7/zh=2/es=0 same-lang gold; cross n=22–34) + XRC axis INDICATIVE,
  es undefined never imputed + upside-not-reversal.
- **Limitations** — (a) RRC-firmness sentence now also names the ARI identity
  (closes to 1.0). (b) NEW "Per-route frontier thinness" paragraph (thin-n,
  indicative XRC, es undefined, router=headroom). (c) causal-probe paragraph now
  ties the W3 probe target explicitly to cp_fig22's per-model $L_\infty$/ARI as
  the before/after quantity; recompute XRC50/RRC@100/ARI.
- **Conclusion** — one optional clause: the floor $L_\infty=0.058$ "is the only
  part of the gap a re-ranker cannot move, recoverable by alignment alone." No ARI
  number, no routing, no n.s. correlation.
- **custom.bib** — added `gao2021rethink` (ECIR 2021, multi-stage cascade),
  `nogueira2019multistage` (arXiv:1910.14424), and `syftr2025` (arXiv:2505.20266,
  Pareto frontier). All three resolve.

## Critic points addressed (round 3)
- **Correctness C-NEW (F2):** fixed — fig21 caption "sibling-confusion rate" →
  "confusion rate." Value untouched.
- **Correctness T-MINOR:** fixed — universal-blind list → "predominantly French
  and Chinese."
- **Cohesion joint #1 (F1, the round-3 must-fix):** fixed — both "eight
  non-degenerate models" sites relabeled to "models with a defined recall / RRC
  curve (all but `gte-base`)." The DEG gate's 7-set discipline no longer
  self-contradicts.
- **Cohesion joint #2 (F6):** fixed — e5 $L_\infty$ endpoint relabeled as the
  degenerate model the gate excludes.
- **Cohesion joint #3 (float order):** acceptable as-is per critic; not disturbed.
- **Novelty over-claim #1 (F4):** addressed — Pareto frame credited
  (`\citep{syftr2025}`), XRC claimed as the new axis.
- **Novelty over-claim #2 / missing-cite (F3):** addressed — cascade/recall-ceiling
  cite added; only cross-lingual quantification + $L_\infty$ claimed.
- **Novelty over-claim #3 ("alignment-only floor" inferential step):** the §6.1
  floor sentence now reads "a floor only representation alignment can move
  (§7)," surfacing the inferential step once rather than baking it into the
  adjective.

## Numbers — all traced (no fabrication)
- ARI@100 egemma 0.229 (CSV 0.2286, min non-deg), next qwen3 0.233 (0.2326);
  $L_\infty$ egemma 0.058 (0.0584, smallest of 9); identity_sum_100 == 1.0 for all
  9 rows. Source: `extra_ari_decomposition/ari_decomposition.csv` + `summary.json`.
- Per-route: 3 distinct max-CLIR corners (en→qwen3-0.6B, de/es/zh→embeddinggemma,
  fr→nomic-v2-moe); flips 2/5 (en, fr); egemma wins 3/5 (de, es, zh); pooled
  CLIR@10 0.5024; n_same de=7/zh=2/es=0, n_cross 22–34; es XRC undefined.
  Source: `extra_per_route_frontier/summary.json` (+ 4 csv). Recompute matched
  reporter.md exactly.

## Self-lint (no compiler installed)
- `\begin`/`\end` balanced: figure 27/27, table 2/2, equation 4/4, abstract 1/1,
  itemize 1/1, document 1/1.
- Global brace balance (escaped braces ignored): 861 open / 861 close, diff 0.
- All 27 `\includegraphics` targets exist on disk (incl. cp_fig22, cp_fig23).
- All `\cite*` keys resolve to custom.bib / anthology.bib (0 missing); the three
  new keys present.
- New labels each defined once and referenced: fig:ari (ref ×3), fig:per_route
  (ref ×2), eq:ari (ref ×1).
- No new `\todo{}` added; the 22 string-matches are the macro definition (L22)
  and pre-existing `% TODO`-fenced comments (corpus dedup 14,401; human-eval
  8.33/10), which remain honestly fenced.

## Open \todo / deferred items (unchanged from round 3, none new)
- `% TODO trace:` corpus dedup count 14,401 + GP/EPO/JRC coverage matrix (system
  slides, not under reports/) — still fenced in §3.
- `% TODO trace:` human-eval numbers (mean 8.33/10, items reviewed, auto-grader
  strictness) — still fenced in §3 and Appendix.
- `% TODO trace:` MMTEB/MIRACL/NeuCLIR general-domain transfer measurement — still
  fenced in Limitations (backlogged).
- Forthcoming (needs_eval, framed as upside, paper does not depend): W3 alignment
  causal probe (now with cp_fig22 $L_\infty$/ARI as explicit before/after target),
  equivalence-audit spot-check, XRC-conformal, CCI hop-distance law.

## Freeze status
Spine frozen per story: C1 benchmarks (§3), DEG gate, N2 frontier framing, the two
RBO ceilings + "any model," MT null, +0.96 robust correlation, two-tax
MEASURED-but-weak, thesis/abstract/intro/conclusion claims — none re-opened beyond
the named one-clause additions. cp_fig22 + cp_fig23 are the last new analytical
objects; round 5+ is polish/tightening only.
