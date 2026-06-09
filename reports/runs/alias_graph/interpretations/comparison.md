# Alias-graph: two relevance interpretations

Same models, same saved rankings — only the definition of *relevant* differs.

- **concept_level**: 132 queries, ~109 gold docs each.
- **per_publication**: 6609 (query, publication) units, ~2.2 gold docs each.

## Recall@10 / nDCG@10 / MRR side by side (ranked by per-publication Recall@10)

| Model | concept R@10 | per-pub R@10 | concept nDCG@10 | per-pub nDCG@10 | concept MRR | per-pub MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `google-embeddinggemma-300m` | 0.0807 | 0.0389 | 0.3937 | 0.0290 | 0.7305 | 0.0325 |
| `qwen-qwen3-embedding-0-6b` | 0.0643 | 0.0275 | 0.3623 | 0.0221 | 0.7346 | 0.0274 |
| `nomic-ai-nomic-embed-text-v2-moe` | 0.0570 | 0.0248 | 0.3187 | 0.0202 | 0.6959 | 0.0259 |
| `baai-bge-m3` | 0.0542 | 0.0233 | 0.3114 | 0.0186 | 0.7100 | 0.0234 |
| `ibm-granite-granite-embedding-278m-multilingual` | 0.0561 | 0.0203 | 0.2751 | 0.0162 | 0.5811 | 0.0198 |
| `sentence-transformers-labse` | 0.0561 | 0.0181 | 0.2652 | 0.0151 | 0.5741 | 0.0198 |
| `intfloat-multilingual-e5-large-instruct` | 0.0292 | 0.0148 | 0.1788 | 0.0114 | 0.4630 | 0.0193 |
| `cambridgeltl-sapbert-umls-2020ab-all-lang-from-xlmr` | 0.0496 | 0.0142 | 0.1961 | 0.0112 | 0.4525 | 0.0161 |
| `alibaba-nlp-gte-multilingual-base` | 0.0062 | 0.0027 | 0.0391 | 0.0020 | 0.1342 | 0.0037 |
