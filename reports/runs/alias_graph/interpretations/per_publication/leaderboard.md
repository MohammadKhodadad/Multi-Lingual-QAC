# Interpretation: per_publication

Gold = the query's OWN source publication's language variants -- the single patent each query was generated from (the dataset's `source_publication` column). One eval unit per query (~2-3 gold docs). Standard full-corpus ranking: every other document, including the concept's other gold patents, counts as non-relevant.

- Eval units: **132**  ·  avg relevant docs per unit: **2.4**
- Models ranked by Recall@10 (re-scored from the saved predictions; no model re-run).

| Rank | Model | Recall@10 | Recall@100 | Precision@10 | nDCG@10 | MRR | MAP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `google-embeddinggemma-300m` | 0.6705 | 0.8396 | 0.1674 | 0.6075 | 0.6335 | 0.5722 |
| 2 | `qwen-qwen3-embedding-0-6b` | 0.6414 | 0.7778 | 0.1591 | 0.5734 | 0.6284 | 0.5159 |
| 3 | `baai-bge-m3` | 0.6035 | 0.7778 | 0.1477 | 0.5600 | 0.6455 | 0.5100 |
| 4 | `nomic-ai-nomic-embed-text-v2-moe` | 0.5795 | 0.7551 | 0.1417 | 0.5285 | 0.6121 | 0.4738 |
| 5 | `ibm-granite-granite-embedding-278m-multilingual` | 0.5290 | 0.6932 | 0.1311 | 0.4702 | 0.5111 | 0.4290 |
| 6 | `sentence-transformers-labse` | 0.5114 | 0.7008 | 0.1265 | 0.4653 | 0.5299 | 0.4176 |
| 7 | `cambridgeltl-sapbert-umls-2020ab-all-lang-from-xlmr` | 0.3523 | 0.5391 | 0.0856 | 0.3112 | 0.3807 | 0.2692 |
| 8 | `intfloat-multilingual-e5-large-instruct` | 0.2374 | 0.4419 | 0.0583 | 0.2308 | 0.3702 | 0.1836 |
