# Experimental analysis — chem-patents CLIR deep-dive

A 10-round iterative analysis of the 9-model chemistry-patent retrieval benchmark, centred on the
question the standard MTEB report never asks: **can a question in one language pull back the relevant
patent in its *other* language versions (CLIR)?** Each round asks a sharper question, adds plots +
CSVs, writes a section into `../experimental_plots/FINDINGS.md`, and feeds the curated
`../key_findings/` deliverable.

All inputs are **local** (no network): the 9 models' `scored_rankings.parquet` under
`../parts/*/retrieval_results/`, and the cached HF dataset configs
(`MehdiAstaraki/multi-lingual-qac-chem-patents` {queries, qrels, corpus} and the
`MehdiAstaraki/multilingual_GP` haystack), loaded with `HF_HUB_OFFLINE=1`.

## Run

```bash
# from the repo root, inside .venv
python reports/runs/chem_patents/experimental_codes/run_all.py
```

Rounds have cross-dependencies (Round 10 reads Rounds 1/3/5/8), so run in order — `run_all.py` does
this and then builds the curated deliverable.

## New metrics (beyond the existing MTEB recall/ndcg/map/mrr@k)

- **CLIR@k / MoLIR@k** — recall over *cross-language* vs *same-language* gold, and the **home-advantage
  index** (MoLIR − CLIR).
- **Directional CLIR matrix + asymmetry** — query→document language recall and A(X→Y)=R(X→Y)−R(Y→X).
- **MT penalty** — paired human-original vs machine-translated `foreign_reach@10` (home-confound removed).
- **Mate retrieval** — mate-hit@k, first-foreign-rank, mate-MRR (the bitext twin lens).
- **Cross-lingual RBO** — ranking consistency of the same question across languages.
- **Same-language over-representation** — top-k same-language share ÷ corpus base rate (collapse).
- **Distractor dominance** — language of the documents that out-rank the first gold.
- **Cross-language separability AUC** — AUC(cross-language gold > non-gold), and the same−cross gap.
- **Oracle / RRF fusion / per-language routing** — complementarity headroom.
- **CLIR-MRS** — capability × (0.5 + 0.5·robustness) synthesis with a query-bootstrap CI.

## Files

- `common.py` — local loaders, the `core_per_query()` frame, the CLIR metric library, statistics
  (`bootstrap_ci`, `paired_perm_test`, `wilcoxon`), model/language registry, plot style, and the
  `append_findings()` helper.
- `round01..round10_*.py` — one re-runnable script per round; each writes to
  `../experimental_plots/roundNN_*/` and a section into `FINDINGS.md`.
- `build_key_findings.py` — curates `../key_findings/` (14 figures + `EXECUTIVE_SUMMARY.md` +
  `headline_numbers.csv/json`).

## Rounds

| # | Question |
| --- | --- |
| 1 | Split recall into same- vs cross-language — how big is the home advantage? |
| 2 | Which query→document directions collapse, and is failure symmetric? |
| 3 | Is a machine-translated question worse than a human one (home-confound removed)? |
| 4 | Can the model find the patent's translated twin, and at what rank? |
| 5 | Same question, different languages — same ranked patents (RBO)? |
| 6 | Do models pile up the query's own language beyond the corpus base rate? |
| 7 | What buries the foreign twin — same-language noise or genuine distractors? |
| 8 | Are foreign golds separable from the crowd in score space (AUC)? |
| 9 | Are model errors complementary (oracle / fusion / routing)? |
| 10 | Fold it all into one robustness verdict (CLIR-MRS) and recommend. |
