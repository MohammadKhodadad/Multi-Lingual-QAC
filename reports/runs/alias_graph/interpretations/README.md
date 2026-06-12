# Two relevance interpretations of the alias-graph run

Gold = every document of every publication attesting the query's concept ("find all patents about compound X"). This is the benchmark as shipped; Recall@10 is mechanically capped because there are many relevant docs per query.

Gold = the query's OWN source publication's language variants -- the single patent each query was generated from (the dataset's `source_publication` column). One eval unit per query (~2-3 gold docs). Standard full-corpus ranking: every other document, including the concept's other gold patents, counts as non-relevant.

Files:
- `concept_level/` — leaderboard.md + summary.json (gold = all of the concept's patents)
- `per_publication/` — leaderboard.md + summary.json (gold = the query's own source patent's variants)
- `comparison.md` — both lenses side by side

Both are re-scored from `../predictions/` with pytrec_eval; the models were NOT re-run.
The per_publication lens uses the dataset's `source_publication` column (the single patent each
query was generated from), so each query has only ~2-3 gold docs and Recall@10 is not capped.
