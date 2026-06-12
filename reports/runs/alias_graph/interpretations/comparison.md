# Alias-graph: two relevance interpretations

Same models, same saved rankings — only the definition of *relevant* differs.

- **concept_level**: 132 queries, ~109 gold docs each (all patents about the concept).
- **per_publication**: 132 queries, ~2.4 gold docs each (the query's own source patent's language variants).

## Recall@10 / nDCG@10 / MRR side by side (ranked by per-publication Recall@10)

| Model | concept R@10 | per-doc R@10 | concept nDCG@10 | per-doc nDCG@10 | concept MRR | per-doc MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `google-embeddinggemma-300m` | 0.0807 | 0.6705 | 0.3937 | 0.6075 | 0.7305 | 0.6335 |
| `qwen-qwen3-embedding-0-6b` | 0.0643 | 0.6414 | 0.3623 | 0.5734 | 0.7346 | 0.6284 |
| `baai-bge-m3` | 0.0542 | 0.6035 | 0.3114 | 0.5600 | 0.7100 | 0.6455 |
| `nomic-ai-nomic-embed-text-v2-moe` | 0.0570 | 0.5795 | 0.3187 | 0.5285 | 0.6959 | 0.6121 |
| `ibm-granite-granite-embedding-278m-multilingual` | 0.0561 | 0.5290 | 0.2751 | 0.4702 | 0.5811 | 0.5111 |
| `sentence-transformers-labse` | 0.0561 | 0.5114 | 0.2652 | 0.4653 | 0.5741 | 0.5299 |
| `cambridgeltl-sapbert-umls-2020ab-all-lang-from-xlmr` | 0.0496 | 0.3523 | 0.1961 | 0.3112 | 0.4525 | 0.3807 |
| `intfloat-multilingual-e5-large-instruct` | 0.0292 | 0.2374 | 0.1788 | 0.2308 | 0.4630 | 0.3702 |
