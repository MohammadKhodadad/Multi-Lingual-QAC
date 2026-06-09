# Two relevance interpretations of the alias-graph run

Gold = every document of every publication attesting the query's concept ("find all patents about compound X"). This is the benchmark as shipped; Recall@10 is mechanically capped because there are many relevant docs per query.

Gold = a single gold publication's language variants ("find this patent's cross-language versions"). Each (query, gold-publication) pair is one eval unit; the concept's other gold publications are excluded from the candidate ranking.

Files:
- `concept_level/` — leaderboard.md + summary.json (gold = all concept docs)
- `per_publication/` — leaderboard.md + summary.json (gold = one publication's variants)
- `comparison.md` — both lenses side by side

Both are re-scored from `../predictions/` with pytrec_eval; the models were NOT re-run.

## What this shows

- **Model ranking is the same under both lenses** (embeddinggemma > Qwen3 > nomic > bge-m3 > … > gte),
  so the leaderboard is robust to the choice of relevance definition.
- **Per-publication Recall@10 is *lower*, not higher** (~0.04 vs ~0.08), even though each unit has
  only ~2.2 gold docs. Reason: the queries are **concept-generic** (`CHEBI_x__lang`), so a model has
  no signal to prefer one of the concept's ~50 attesting publications. It concentrates retrieval on a
  few easy publications, so a *typical* publication's specific cross-language variants rank low. So the
  low Recall@10 is **not merely an artifact of the ~109-gold count** — retrieving a *specific* patent's
  translations from a concept-only query is intrinsically hard.
- The big drop in MRR/nDCG (concept ≈0.7/0.39 → per-pub ≈0.03/0.03) says the same thing: models
  reliably put *some* concept doc near the top, but rarely a *given* publication's doc.

## Caveat on the per-publication lens
The dataset does not store which single publication each query was generated from, so "per publication"
is realised by treating every (query, gold-publication) pair as an independent unit. A *true*
per-document benchmark (one query tied to one source publication; gold = that publication's ~2.3
translations) would require regenerating `queries`+`qrels` to record the source publication.
