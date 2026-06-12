# Interpretation: concept_level

Gold = every document of every publication attesting the query's concept ("find all patents about compound X"). This is the benchmark as shipped; Recall@10 is mechanically capped because there are many relevant docs per query.

- Eval units: **132**  ·  avg relevant docs per unit: **109.3**
- Models ranked by Recall@10 (re-scored from the saved predictions; no model re-run).

| Rank | Model | Recall@10 | Recall@100 | Precision@10 | nDCG@10 | MRR | MAP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `google-embeddinggemma-300m` | 0.0807 | 0.1515 | 0.3076 | 0.3937 | 0.7305 | 0.1128 |
| 2 | `qwen-qwen3-embedding-0-6b` | 0.0643 | 0.1262 | 0.2788 | 0.3623 | 0.7346 | 0.0793 |
| 3 | `nomic-ai-nomic-embed-text-v2-moe` | 0.0570 | 0.1121 | 0.2356 | 0.3187 | 0.6959 | 0.0677 |
| 4 | `sentence-transformers-labse` | 0.0561 | 0.0972 | 0.1947 | 0.2652 | 0.5741 | 0.0618 |
| 5 | `ibm-granite-granite-embedding-278m-multilingual` | 0.0561 | 0.1057 | 0.2045 | 0.2751 | 0.5811 | 0.0663 |
| 6 | `baai-bge-m3` | 0.0542 | 0.1133 | 0.2265 | 0.3114 | 0.7100 | 0.0645 |
| 7 | `cambridgeltl-sapbert-umls-2020ab-all-lang-from-xlmr` | 0.0496 | 0.1020 | 0.1432 | 0.1961 | 0.4525 | 0.0591 |
| 8 | `intfloat-multilingual-e5-large-instruct` | 0.0292 | 0.0732 | 0.1371 | 0.1788 | 0.4630 | 0.0351 |
