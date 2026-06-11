# Retrieval results (embedding-model rankings)

`scored_rankings.parquet` — one row per ranked (query, document) pair, top-K per query per model.

Columns:
- `model` — embedding model name
- `query_id`, `query_language`, `chebi_id` — the query and its concept
- `rank` (1 = top), `corpus_id`, `corpus_language`, `score` (cosine, higher = better)
- `relevance` — `gold` (right document), `hard_negative` (chemically-similar look-alike), or `` (not judged)

Compute @k metrics by filtering `rank <= k`. CLIR@k (cross-lingual recall@k) = of the gold
documents whose `corpus_language != query_language`, the fraction with `rank <= k`.
