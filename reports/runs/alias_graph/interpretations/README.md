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

## What this shows

- **Per-document Recall@10 is large and discriminative (0.24–0.67)** — confirming the tiny
  concept-level Recall@10 (~0.08) was purely a many-positives artifact (~109 gold/query), not poor
  retrieval. With the true ~2.4 gold/query, the top models retrieve a query's source patent's
  cross-language variants well (embeddinggemma 0.67, Qwen3 0.64, bge-m3 0.60, nomic 0.58).
- **Model ranking is the same under both lenses**, so the leaderboard is robust to the relevance
  definition.
- **gte-multilingual-base is a genuine outlier** (per-doc R@10 0.047 vs 0.5–0.67), ~10× below the
  pack under *both* lenses — its weak score is real, not an artifact of the position_ids fix.
- This per_publication lens (gold = the query's own source patent) is the meaningful headline for
  this benchmark; concept_level Recall@10 should not be used as the main score.
