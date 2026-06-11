# Action Items for the Short Paper

Priority goal: strengthen the paper from a benchmark-construction story into a
clear workshop / Industry Track story about deployment-oriented model selection
for multilingual chemistry retrieval.

## P0: Must Do Before Next Commit

1. Add a short `Deployment Decision` section.
   - Location: after `sections/05_results.tex`, before `sections/06_limitations.tex`.
   - Update `short_main.tex` to include the new section.
   - Purpose: make the paper answer the reviewer's main question: what practical
     retriever choice follows from the benchmark?
   - Content to include:
     - Decision context: choose one multilingual embedding model for a shared
       chemistry technical retrieval index.
     - Old signal: aggregate Recall@10 alone can make deployment readiness look
       better than it is.
     - New signal: cross-language recall, home advantage, language collapse, and
       chemistry-confusability expose different risks.
     - Honest recommendation: `embeddinggemma` is the strongest capability corner;
       `bge-m3` is a cheaper-to-read alternative in the cost-vs-capability view;
       do not claim a universal winner without stating the operating threshold.
     - Practical lesson: aggregate recall alone is not a deployment dashboard.

2. Add a compact main results table.
   - Location: `sections/05_results.tex`.
   - Purpose: make the main findings scannable without forcing reviewers to
     infer them from prose and figures.
   - Suggested columns: `Finding`, `Metric`, `Takeaway`.
   - Include:
     - QAC quality: human mean `8.33/10`.
     - Human validation: `94/97` good items.
     - Human vs LLM grader: LLM mean `79.0%` vs human `83.3%`.
     - Cross-language ceiling: best cross-language Recall@10 `0.50`.
     - Same-language bias: home advantage up to `+0.55`.
     - Chemistry confusion: `14--78%` on publication-lens alias-graph queries.

3. Make the abstract more result-driven.
   - Location: `sections/00_abstract.tex`.
   - Add one concise empirical/deployment sentence.
   - Possible wording:
     `Aggregate Recall@10 overstates deployment readiness: the best
     cross-language Recall@10 is 0.50, same-language advantage reaches +0.55,
     and chemically confusable documents outrank gold evidence on 14--78% of
     publication-lens queries.`
   - Keep the abstract from becoming too long; trim a generic sentence if needed.

## P1: High-Value Clarity Improvements

4. Strengthen the "why patents?" explanation.
   - Location: `sections/01_introduction.tex` or `sections/03_benchmark.tex`.
   - Add a compact paragraph clarifying:
     - patents are not the only target application;
     - they are the substrate because they combine multilingual publication
       variants, controlled source identity, chemistry-rich language, and
       realistic technical retrieval difficulty;
     - this gives stronger content control than unrelated multilingual corpora.

5. Sharpen the contribution list.
   - Location: `sections/01_introduction.tex`.
   - Current contribution paragraph is good but broad.
   - Make the contributions more explicit:
     1. multilingual chemistry-patent QAC benchmark with query/document language metadata;
     2. LLM-assisted QAC generation and verification with human validation;
     3. language-aware retrieval evaluation separating same-language and cross-language evidence;
     4. chemistry-confusability stress test using alias / ontology neighbors;
     5. deployment-oriented model-selection analysis showing why aggregate recall is insufficient.

6. Clarify QAC pipeline details.
   - Location: `sections/02_pipeline.tex`.
   - Add only concise details that are already supported:
     - multiple candidates per document/language/mode;
     - technical and semantic question modes;
     - verifier grades faithfulness, answerability, language quality, and
       retrieval usefulness;
     - human validation sample size: `97` reviewed QACs.
   - Avoid adding model/prompt/threshold specifics unless verified from code or reports.

7. Reduce repeated "aggregate recall hides failure" phrasing.
   - Locations: abstract, introduction, evaluation, results, conclusion.
   - Keep the thesis, but make each occurrence do different work:
     - Abstract: empirical takeaway.
     - Introduction: motivation.
     - Method/Benchmark: how the benchmark makes it observable.
     - Results: evidence.
     - Conclusion: practitioner lesson.

## P2: Optional / Needs Verification

8. Add practical model attributes only if verified.
   - Possible location: Evaluation Setup or Deployment Decision.
   - Do not invent cost, latency, license, or model-size values.
   - If verified model-card data is not available, say explicitly that this paper
     treats deployment cost through retrieval-depth / candidate-pool behavior
     rather than measured serving latency or licensing.

9. Consider moving one deployment-oriented appendix figure into the body.
   - Candidate: `cp_fig18_cost_frontier.png`.
   - Only do this if adding the Deployment Decision section and if page budget allows.
   - Purpose: visually support the `embeddinggemma` vs `bge-m3` trade-off.

10. Improve Limitations structure if space allows.
    - Location: `sections/06_limitations.tex`.
    - Split into clearer categories:
      - data/domain limitations;
      - validation limitations;
      - deployment limitations;
      - generalization limitations.

11. Make the conclusion more practitioner-oriented.
    - Location: `sections/07_conclusion.tex`.
    - Possible final sentence:
      `For high-trust multilingual retrieval, the central question is not whether
      the system retrieves something relevant on average, but whether it
      retrieves the intended evidence when language and technical similarity
      create tempting shortcuts.`

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

Do not fully restructure the paper yet. First perform the targeted revision:
Deployment Decision section + results table + abstract update + why-patents
paragraph. This should address the largest review concerns with minimal churn.
