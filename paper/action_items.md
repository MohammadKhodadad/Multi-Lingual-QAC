# Action Items for the Short Paper

Priority goal: finish the light polishing needed for a credible EMNLP Industry
Track submission. The revised draft is now around `8.0/10`; the remaining work is
mostly writing-level clarity rather than conceptual restructuring.

## Completed

- Added a dedicated `Deployment Decision` section after Results.
- Added the cost-vs-capability figure to the body.
- Added a compact benchmark statistics table.
- Added a compact main results table.
- Made the abstract more result-driven with concrete findings:
  - best cross-language Recall@10 `0.50`
  - same-language advantage up to `+0.55`
  - chemistry-confusability failures `14--78%`
- Tightened redundant Evaluation, Results, Limitations, and Conclusion prose.
- Made the conclusion more practitioner-oriented.

## P0: Final Writing Polish

1. Strengthen the Results prose without expanding much.
   - Location: `sections/05_results.tex`.
   - Purpose: Table 2 is strong, but key numbers should also appear in prose.
   - Add back the following numbers in the relevant paragraphs:
     - `97` human-reviewed QACs
     - mean human score `8.33/10`
     - `94/97` good QACs
     - LLM vs human scores `79.0%` vs `83.3%`
     - best cross-language Recall@10 `0.50`
     - home advantage up to `+0.55`
     - chemistry confusion `14--78%`
   - Keep this concise; do not undo the recent page-budget cuts.

2. Define `XRC50` more clearly.
   - Location: `sections/06_deployment.tex`.
   - Add one short sentence before the frontier discussion:
     `XRC50 measures how much deeper a user must inspect the ranked list,
     relative to same-language retrieval, before reaching cross-language
     evidence.`

3. Make the deployment recommendation explicitly conditional.
   - Location: `sections/06_deployment.tex`.
   - Soften benchmark-specific wording:
     - Use `Under a capability-first criterion in this benchmark,
       embeddinggemma is the recommended choice.`
     - Add that `bge-m3` becomes preferable under a stricter reading-cost budget
       if the CLIR@10 threshold is acceptable.
     - Keep `granite-278m` framed as a lower-capability, cheaper frontier point.

4. Add one practical user-workflow sentence.
   - Best location: `sections/01_introduction.tex`, first paragraph or just
     after it.
   - Suggested wording:
     `In the target workflow, a chemist or technical analyst issues a query in
     one working language and expects the retrieval system to surface the
     strongest patent evidence, even when that evidence appears in another
     language or near chemically similar documents.`

## P1: Optional If Page Budget Allows

5. Strengthen the "why patents?" explanation.
   - Location: `sections/01_introduction.tex` or `sections/03_benchmark.tex`.
   - Clarify that patents are not the only target application; they are the
     benchmark substrate because they combine multilingual publication variants,
     controlled source identity, chemistry-rich language, and realistic
     technical retrieval difficulty.

6. Sharpen the contribution list.
   - Location: `sections/01_introduction.tex`.
   - Make the contributions more explicit:
     1. multilingual chemistry-patent QAC benchmark with query/document language metadata;
     2. LLM-assisted QAC generation and verification with human validation;
     3. language-aware retrieval evaluation separating same-language and cross-language evidence;
     4. chemistry-confusability stress test using alias / ontology neighbors;
     5. deployment-oriented model-selection analysis showing why aggregate recall is insufficient.

7. Clarify QAC pipeline details.
   - Location: `sections/02_pipeline.tex`.
   - Add only concise details that are already supported:
     - multiple candidates per document/language/mode;
     - technical and semantic question modes;
     - verifier grades faithfulness, answerability, language quality, and
       retrieval usefulness;
     - human validation sample size: `97` reviewed QACs.

## P2: Submission Checks

8. Verify data/code availability.
   - Check that the Hugging Face and GitHub links are live.
   - Confirm anonymization expectations for the target venue.
   - Confirm the dataset card is clear.
   - Confirm paper numbers match released artifacts.

9. Add practical model attributes only if verified.
   - Do not invent cost, latency, license, or model-size values.
   - If verified model-card data is not available, keep deployment cost framed
     through retrieval-depth / reading-cost behavior rather than serving latency
     or licensing.

10. Recheck page budget after final polish.
    - Compile `short_main.tex`.
    - If still too long, move `cp_fig09_10_collapse.png` from Results back to the
      appendix; this remains the cleanest large space-saving option.

## Evidence Already Available

- Human validation: `reports/human_eval/summary.json`
  - `n = 97`
  - human mean `8.33/10`
  - `94/97` good items
  - LLM mean `79.0%`, human mean `83.3%`
- Released chem-patents dataset card:
  - corpus `1.11k` rows
  - qac / queries `137`
  - qrels `322`
  - cross-language qrels `265`
- Existing full-paper / reports-backed results:
  - best cross-language Recall@10 `0.50`
  - home advantage up to `+0.55`
  - alias-graph confusion `14--78%`
  - cost-vs-capability figure available: `cp_fig18_cost_frontier.png`
  - reranker recoverability figure available: `cp_fig19_rrc_budget.png`

## Current Recommendation

Do one final light revision bundle: strengthen Results prose, define `XRC50`,
make the deployment choice explicitly conditional, and add one user-workflow
sentence. Then compile and check page budget before submission.
