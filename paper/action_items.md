# Action Items for the Short Paper

Priority goal: finish the light polishing needed for a credible EMNLP Industry
Track submission. The latest review rates the draft around `8.3/10`, with
potential to reach about `8.5/10` after targeted polish. No major restructuring
is recommended.

## Completed

- Added a dedicated `Deployment Decision` section after Results.
- Refreshed the main-body figures with the updated two-source results.
- Reordered the main-body figure story around the four key claims:
  1. technical questions are harder than semantic ones;
  2. same-language gold is easier than cross-language gold;
  3. chemically confusable wrong documents can outrank gold evidence;
  4. deployment cost decomposes into reading depth, re-ranker recovery, and residual first-stage failure.
- Moved supporting query-language and attractor-detail plots to the appendix.
- Added additional appendix diagnostics for question-type robustness,
  cross-source transfer, language denominators, and deployment budget.
- Added a compact benchmark statistics table for both Google Patents and EPO.
- Updated the abstract to say the release contains two public multilingual
  chemistry QAC retrieval benchmarks: Google Patents and EPO.
- Updated the contribution list to avoid the broad `first public benchmark`
  claim and instead name the two chemistry-patent retrieval benchmarks.
- Updated the main empirical numbers to the current result set:
  - best cross-language Recall@10 about `0.55`;
  - same-language advantage visible in every query language;
  - chemistry-confusability failures `14--48%`.

## P0: Final Light Polish

1. Add a compact main findings table if page budget allows.
   - Location: `sections/05_results.tex`, likely near the start or end of
     Results.
   - Purpose: Give reviewers a fast summary of the core empirical claims.
   - Suggested rows:
     - QAC quality: `8.33/10`, `94/97` good.
     - Cross-language retrieval: best Recall@10 about `0.55`.
     - Same-language gap: about `0.21--0.28` Recall@10.
     - Chemistry confusion: `14--48%`.
     - Deployment frontier: `embeddinggemma`, `bge-m3`, `granite-278m`.
   - If space is tight, skip this rather than weakening the narrative.

2. Add a short related-work sentence on LLM-assisted benchmark construction.
   - Location: `sections/02_related_work.tex`.
   - Suggested wording:
     `LLM-assisted benchmark construction has become a practical way to create
     domain-specific evaluation data, but generated items require provenance,
     filtering, and human audit before they can support reliable conclusions.
     Our pipeline follows this direction while making each generated QAC
     traceable to a source document, language, verifier scores, and retrieval
     relevance judgments.`

3. Define `XRC50` more explicitly.
   - Location: `sections/06_deployment.tex`, before or at first use.
   - Suggested wording:
     `XRC50 measures the median multiplier in reading depth needed to reach
     cross-language evidence compared with same-language evidence.`

4. Replace generic abstract ending with more concrete deployment language.
   - Location: `sections/00_abstract.tex`.
   - Current concern: `practical framework` is acceptable but generic.
   - Suggested wording:
     `The benchmark turns cross-language and chemistry-confusability failures
     into measurable model-selection criteria for high-trust technical retrieval.`

## P1: Consistency and Submission Checks

5. Check all updated numbers across release surfaces.
   - Locations: abstract, Results, captions, appendix, dataset cards, GitHub
     README, and any report text.
   - Current paper numbers to verify:
     - best cross-language Recall@10 about `0.55`;
     - same-vs-cross gap about `0.21--0.28` Recall@10;
     - chemistry confusion `14--48%`;
     - Google Patents: corpus `23,787`, queries `524`, qrels `1,284`,
       cross-language qrels `1,023`;
     - EPO: corpus `11,315`, queries `198`, qrels `594`,
       cross-language qrels `396`.

6. Recheck the page budget after any final polish.
   - Compile `short_main.tex`.
   - If the body becomes too crowded, first remove or move the optional compact
     findings table rather than cutting the four-priority-figure story.

7. Verify data/code availability.
   - Check that both Hugging Face links and the GitHub link are live.
   - Confirm anonymization expectations for the target venue.
   - Confirm dataset cards describe the two releases clearly.

8. Add practical model attributes only if verified.
   - Do not invent cost, latency, license, model size, or serving constraints.
   - Keep deployment cost framed through retrieval-depth / reading-cost behavior
     unless external model-card facts are explicitly checked.

## Evidence Already Available

- Human validation: `reports/human_eval/summary.json`
  - `n = 97`
  - human mean `8.33/10`
  - `94/97` good items
  - LLM mean `79.0%`, human mean `83.3%`
- Released benchmark statistics:
  - Google Patents: corpus `23,787`, queries `524`, qrels `1,284`,
    cross-language qrels `1,023`
  - EPO: corpus `11,315`, queries `198`, qrels `594`,
    cross-language qrels `396`
- Current result figures:
  - `claimA_A3_gap_heatmap.png`
  - `claimC_C3_home_advantage.png`
  - `claimD_D1_confusion_heatmap.png`
  - `claimE_E2_triptych.png`
  - `cp_fig18_cost_frontier.png`

## Current Recommendation

Submit after one final polish pass. The highest-impact edits are adding the
LLM-assisted benchmark-construction sentence, defining `XRC50` cleanly, replacing
the generic abstract ending, and checking number consistency across the paper and
release pages. The compact findings table is useful, but optional if it hurts
page budget.
