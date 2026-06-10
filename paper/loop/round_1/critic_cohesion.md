# Cohesion review (round 1)

Reviewer #3 (Cohesion). Scope: flow, framing, internal consistency. I did not
re-verify numbers against `reports/` (that is Reviewer #2) nor judge novelty
(Reviewer #1). I read `paper/main.tex` (identical to `round_1/draft.tex`), the
referenced figures, `story.md`, `figures_manifest.md`, and the two
`EXECUTIVE_SUMMARY.md` files only to confirm which *benchmark* a number/figure
belongs to (a cohesion question, not a fact-check).

## Overall: does it read as one story? (1 paragraph)

Yes — this reads as one tight, well-glued industrial story, and it is unusually
disciplined for a round-1 draft. The spine ("a team must deploy one model;
average Recall@10 hides the cross-lingual collapse; we make it measurable, name
the survivor, and trace it to a separability deficit so the fix is alignment not
re-ranking") is stated in the abstract, set up in the intro, instrumented in
Metrics, demonstrated in Results, explained in Analysis, and cashed out in
Deployment — the same four claims with the same emphasis in all five places.
Every promised contribution C1–C5 has a home section and is referenced there,
and every results/analysis subsection traces back to a stated failure
(collapse / inconsistency / confusion / separability). The two-benchmark
structure is the one persistent strain on cohesion: the paper interleaves
alias-graph and cross-lingual numbers continuously, and in two places it
actually *blends them inside a single paragraph* (the Analysis "availability
confound" paragraph and the two-RBO-ceiling thread) in a way a careful reader
cannot disentangle from the surrounding figure/footnote. Fixing the
benchmark-attribution joints, tightening the Related Work transition into
Benchmarks, and resolving the abstract's single-RBO-ceiling vs body's
two-ceilings would take this from "cohesive with seams" to "seamless." Nothing
here requires re-architecting the paper.

## Broken glue joints (ranked, each: location -> problem -> concrete fix)

**1. Analysis §6 "availability confound" paragraph (lines 578–596) silently
mixes two benchmarks under one footnote.**
- *Problem.* The paragraph opens with alias-graph mechanism numbers — "own-
  language gold ($0.63$–$0.82$) vs foreign ($0.35$–$0.47$)", "$42\%$ of an
  English query's gold is reachable in-English vs $8$–$10\%$ for de/es/zh" — and
  footnotes them to "alias-graph key findings 'mechanism.'" It then continues in
  the *same paragraph* with chem-patents numbers ("over-fetch up to $49\times$",
  "same-language noise out-ranks the gold on $60\%$ of queries") and points to
  the two chem-patents figures (`cp_fig09`, `cp_fig10`). I verified the split:
  the $42\%$/$8$–$10\%$/$0.63$–$0.82$ figures are from the **alias-graph**
  summary (Fig 3/4); the $49\times$/$60\%$ are from the **chem-patents**
  summary. So one paragraph fuses alias-graph evidence with chem-patents figures
  under an alias-only footnote — the reader cannot tell which benchmark the
  "collapse" claim rests on, and a fact-checker will read the footnote as
  mis-citing the chem-patents figures.
- *Fix.* Split into two sentences with explicit benchmark labels: "On the
  alias-graph benchmark, each language retrieves its own-language gold
  ($0.63$–$0.82$) far better than foreign ($0.35$–$0.47$)… [alias footnote]. The
  same availability bias drives the cross-lingual benchmark's *language
  collapse*: low-resource query languages over-fetch their own language up to
  $49\times$ the base rate and same-language noise out-ranks the gold on $60\%$
  of queries (Figs~\ref{fig:cp_collapse},~\ref{fig:cp_distractor})
  [chem-patents footnote]." Two footnotes, one per benchmark.

**2. Abstract (line 64) gives one RBO ceiling ($0.39$); body (lines 99–100) and
Conclusion (line 774) give two ($0.39$ alias / $0.19$ cross-lingual).**
- *Problem.* The abstract says "five language versions of a query agree on barely
  a third of their top-10 (cross-lingual RBO ceiling $0.39$)" — that is the
  *alias-graph* number — with no mention of the $0.19$ cross-lingual ceiling. A
  reader who meets $0.19$ for the first time in the intro (line 100) has to
  reconcile it against the abstract's single $0.39$. story.md risk #5 explicitly
  flags that these two ceilings must never be conflated; the abstract reads as if
  there is one ceiling. Mild promise/payoff mismatch: the abstract under-promises
  the inconsistency result the body over-delivers.
- *Fix.* Either (a) keep the abstract at one number but make it unambiguous which
  benchmark — "(cross-lingual RBO ceiling $0.39$ on the alias-graph benchmark,
  $0.19$ on the cross-lingual benchmark)" — matching the body's phrasing exactly;
  or (b) if space is tight, write "RBO ceiling $\le 0.39$" so the body's lower
  $0.19$ is consistent with the abstract rather than contradicting it. Option (a)
  is preferred; it costs ~8 words and removes the only abstract↔body number that
  does not line up verbatim.

**3. Related Work → Benchmarks transition is abrupt; §2 ends on a future-work
disclaimer, not a handoff.**
- *Problem.* §2 (Related Work) closes (lines 206–208) with "calibration-flavoured
  cross-lingual metrics… are a natural future-work horizon; we do not claim them
  here." §3 (Benchmarks) then opens cold with "We release two benchmarks…". The
  role file requires each section to end by setting up the next; here the last
  beat of Related Work points *away* from the paper (to future work) instead of
  *into* Benchmarks, so the reader hits a hard edge. Every other section
  transition in the paper does set up the next (Benchmarks→Metrics line 210,
  Metrics→Setup, Results→Analysis line 575, Analysis→Deployment line 656).
- *Fix.* Move the future-work disclaimer earlier (fold it into the calibration
  paragraph, lines 175–184, where it belongs) and end §2 with a forward bridge,
  e.g. "Having positioned our four contributions, we now build the benchmarks
  that deliver C1 — and the two design choices (no MT source docs, no
  publication-number gold) that the patent-IR and cross-lingual-RAG lines above
  show are necessary." That bridge also re-earns the C1 promise immediately
  before §3 pays it.

**4. Two unresolved `\todo{}` markers sit in load-bearing body prose, breaking
the "every number traces to a file" contract mid-sentence (lines 229–231,
278–280; appendix 801–802).**
- *Problem.* These are honest flags, but as written they interrupt the reader: a
  paragraph about the shared corpus (line 220) suddenly contains a bracketed
  red note about untraced dedup counts. For cohesion (not facts), the issue is
  that the Benchmarks §3 "Shared corpus" paragraph *promises* corpus-construction
  detail ("source counts after de-duplication; IPC/A61 distribution… reported in
  the supplementary system description") and then immediately withdraws it via
  `\todo`, leaving an orphan promise the reader sees fail in real time. story.md
  risk #1 anticipates exactly this.
- *Fix.* Demote the unavailable numbers to a single clean sentence with no
  in-line `\todo` ("Detailed corpus-construction statistics are deferred to the
  system description; all load-bearing sizes below come from the two benchmark
  datasets."), and keep the `\todo` as a LaTeX comment (`% TODO`) invisible to
  the reader rather than a red in-text marker. Same for the C5 human-eval markers
  (lines 278–280, 801–802): one descriptive sentence, `\todo` moved to a comment.
  This removes the only places where the prose visibly promises-then-retracts.

**5. Figure cp_fig11 (separability) is captioned with a correlation it does not
plot (lines 658–665), so the figure under-delivers its own caption.**
- *Problem.* The caption claims "$r(\mathrm{AUC},\mathrm{CLIR@10})=+0.96$", but
  the figure is a per-model bar chart of AUC (same vs cross-language); the
  correlation across models is a derived scalar the reader cannot see in the
  bars. Same pattern, milder, for the radar figures (cp_fig14, ag_fig10): the
  text says they show "*where* each top model wins" (line 501) but never names a
  single axis to look at, so they read as decoration.
- *Fix.* For cp_fig11, either change the caption to describe what is shown ("Per
  model, cross-language gold (red) is systematically harder to separate than
  same-language gold (green); the model-level AUC–CLIR@10 correlation is $+0.96$,
  Eq./text") so the $+0.96$ is clearly a text statistic the figure motivates, or
  swap in the actual scatter if one exists under `reports/`. For the radars, add
  one interpretive clause naming the axis where the winner separates from the
  pack ("embeddinggemma leads on consistency and separability, not on raw
  recall").

**6. The teaser figure (cp_fig01) orders models by overall recall; the
leaderboard table (tab:cp_board) orders by CLIR-MRS — same nine models, two
orders, no signpost.**
- *Problem.* In Fig.~1 the model order is embeddinggemma, bge-m3, qwen3, nomic,
  granite, LaBSE, SapBERT, e5, gte (by overall R@10). In Table~1 the order is
  embeddinggemma, bge-m3, granite, nomic, qwen3, LaBSE, SapBERT, e5, gte (by
  CLIR-MRS). granite and qwen3 swap places. A reader cross-referencing the teaser
  against the leaderboard sees the same models reshuffle with no note that the
  axis changed. Minor, but it momentarily undercuts the "robustness reorders the
  recall ranking" thesis at exactly the figure meant to introduce it.
- *Fix.* One sentence near Table~1 ("note the order changes from Fig.~1: ranking
  by CLIR-MRS rather than recall reshuffles the middle of the field — the point
  of the paper"), turning an inconsistency into evidence for the thesis.

## Unmet promises / orphan results

- **C5 (generation+validation pipeline) is promised in two section pointers
  (§Benchmarks, Appendix) but never actually delivers a number in-text** — both
  payoff sites are `\todo`-gated (lines 278–280, 801–802). C5 is currently a
  promise with no payoff visible to the reader. Either deliver the human-eval
  numbers (if dumped to `reports/`) or soften C5's intro phrasing from "the
  reproducible, human-validated… pipeline" to "a reproducible… pipeline (human
  validation summarized in the system description)" so the abstract/intro do not
  promise a validation the body cannot show. As-is this is the weakest
  promise/payoff link in the paper.
- **Directional asymmetry is over-promised relative to its figure.** §Metrics
  introduces "direction-asymmetry" as a named instrument (line 299–302), and
  Results states "the most asymmetric pair is de↔zh (gap $+0.23$)" (line 401–402),
  but the asymmetry figure (`cp_fig04`) is deferred (manifest line 44) — only the
  combined matrix `cp_fig03` is shown. The asymmetry claim is thus a number with
  no figure. Acceptable, but add "(not shown)" or fold the $+0.23$ into the
  matrix caption so the reader does not hunt for a missing panel.
- **No true orphan results found.** Every figure that *is* included is referenced
  and interpreted in text, and every Results/Analysis beat maps to a stated
  failure or contribution. The four copied-but-unreferenced figures (manifest
  lines 38–49) are correctly excluded, not dropped in.
- **"Universal-blind ~12% core" (line 705) appears only in Deployment** with no
  prior setup — it is introduced as if known. It traces to the alias attractors
  (Fig.~5) but the "~12%" figure itself is new at point of use. Either introduce
  it in Analysis (alongside the universal-attractors beat, line 474–476) or label
  it clearly as an oracle-residual derived in Deployment.

## Terminology & notation inconsistencies

- **Benchmark naming is consistent and good.** "alias-graph benchmark" and
  "cross-lingual (CLIR) benchmark" are used uniformly; "chem-patents" appears
  only in figure-source footnotes (e.g. "chem-patents key findings, Fig.~2"),
  never as the benchmark's name in body prose — that separation is clean and
  should be kept. One nit: the abstract (line 57) calls it "a cross-lingual
  (CLIR) benchmark" while §3 header (line 247) calls it "Cross-lingual (CLIR)
  benchmark" — consistent; no action.
- **MoLIR vs MoLIR@k vs MoLIR@10.** Used as MoLIR@k (defn, line 294), MoLIR@10
  (Fig.~1 caption, line 143), and "MoLIR" bare (line 296 in the home-advantage
  defn). Consistent in meaning; fine. Same for CLIR@k / CLIR@10. No fix needed,
  just confirming it is not drifting.
- **"home advantage" notation.** Written as $\mathrm{MoLIR}{-}\mathrm{CLIR}$
  consistently (lines 296, 383, 391, 523). Good. One spelling variance: hyphenated
  "home-advantage" in the contributions list (line 118) and metrics intent, vs
  unhyphenated "home advantage" elsewhere — cosmetic, harmonize to one form.
- **CLIR-MRS / MRS.** The composite is consistently "CLIR-MRS" for the
  cross-lingual benchmark and "MRS" for alias-graph (lines 328, 498–499, 549).
  Good — the role file's named worry (CLIR-MRS vs MRS) is handled correctly. Keep
  it; do not collapse the two names.
- **"mate-retrieval" / "foreign twin".** "mate-retrieval", "mate-hit@k",
  "mate-MRR", "foreign twin", "first foreign twin/rank" all used consistently
  (lines 304–308, 429–433). The metric "mate" and the prose "twin" coexist
  cleanly because the metric §defines the link (line 305). No fix.
- **The two correlation-r claims use different x-variables across benchmarks and
  this is *correct* but easy to misread.** Line 609: cross-lingual benchmark uses
  $r(\text{home advantage}, \text{RBO})=-0.85$; line 610–611: alias-graph uses
  $r(\text{same-language share}, \text{RBO})=-0.87$. story.md C3 (line 91) summarizes
  both as "r(same-language share, RBO) = −0.85 to −0.87", flattening the variable
  difference. The body is right to distinguish them; just make sure no later
  edit copies story.md's flattened phrasing into the abstract/conclusion. Add a
  half-clause ("two related bias proxies") so the reader sees they are deliberately
  different, not a typo.
- **Language codes** (en/de/es/fr/zh) are used uniformly; figures referenced in
  ascending order within each subsection. No notation drift found.

## Abstract/Conclusion alignment issues

- **Strong alignment overall.** Abstract and Conclusion make the same four claims
  with the same emphasis: collapse is real (CLIR@10 $0.50$, home up to $+0.55$),
  inconsistency ("barely a third of their top-10"), confusion ($14$–$78\%$),
  separability cause → "alignment at index time, not re-ranking at query time",
  plus the budget rule (MT the query, human-translate the corpus). The conclusion
  does not introduce any claim the abstract lacks, and vice versa. This is the
  paper's strongest cohesion property.
- **One asymmetry (see joint #2):** the abstract gives a single RBO ceiling
  ($0.39$); the conclusion (line 774) repeats "barely a third of their top-10"
  but the *intro* (line 100) introduced $0.19$. So abstract↔conclusion agree, but
  both under-state relative to the body's two-ceiling result. Harmonize per
  joint #2.
- **The "$r=+0.96$" lives in the abstract (line 66) and is the load-bearing
  mechanism claim, but the word "separability" is the only bridge to it in the
  conclusion** ("embedding-level separability deficit", line 778). The conclusion
  drops the number. That is fine for a conclusion, but ensure the abstract's
  $+0.96$ is the *same* quantity named in Analysis (line 652) — it is
  ($r(\text{cross-language AUC}, \text{CLIR@10})$); the abstract abbreviates it to
  $r(\mathrm{AUC},\mathrm{CLIR@10})$, dropping "cross-language". Spell it
  "cross-language AUC" in the abstract too, so the abstract claim and the Analysis
  claim are textually identical, not just conceptually.
- **MT-null-result emphasis matches** across abstract ("safe… no significant
  penalty"), Results (line 412–418, "null result"), and Deployment/Conclusion
  ("budget rule"). story.md risk #7 (must read as null, never as "MT helps") is
  honored everywhere — abstract line 70 says "is safe… no significant penalty",
  Results line 416–417 explicitly says "we do not claim it helps". Good, leave it.

## What's already cohesive (leave alone)

- **The four-claim spine is consistent across all five touchpoints**
  (abstract / intro / results / analysis / conclusion). Do not touch the framing.
- **Contributions C1–C4 each have a clearly labeled home section and are
  referenced there** (`\S\ref` forward pointers in the contributions list,
  lines 110–132, all resolve). Promise/payoff for C1–C4 is intact.
- **The industrial "which model to deploy + measure robustness honestly" thread
  is visible in every required section** — Intro (lines 77–104), Metrics (the
  whole §, framed as "what a single recall averages away"), Results, Analysis
  ("the lever is at the embedding level"), and a dedicated Deployment section
  with five concrete rules. The paper never drifts into pure-academic framing.
- **Section transitions (except Related Work→Benchmarks, joint #3) all set up
  the next section** — Results→Analysis ("Why does cross-lingual retrieval
  collapse here", line 575) and Analysis→Deployment ("the lever is at the
  embedding level" → "Align, do not re-rank") are model transitions; keep them.
- **The "don't ensemble" vs "oracle headroom is real" tension (story.md risk #6)
  is handled cohesively** — lines 696–706 state both facts in the same paragraph
  ("the two facts coexist"), so it does not read as self-contradiction. Leave it.
- **Figure→text integration is tight** for the included figures: each is
  introduced by a sentence that states its single takeaway before the
  `\includegraphics`, and captions carry the headline number + source. (Captions
  are mostly self-contained; only cp_fig11's caption over-claims, joint #5.)
- **Length/balance is appropriate for an 8-page industry track.** Results +
  Analysis + Deployment (the payoff) get the most space; Related Work is six tight
  paragraphs, each defending one contribution boundary; Limitations is honest and
  proportionate. No section is bloated or starved. Do not cut to rebalance.
