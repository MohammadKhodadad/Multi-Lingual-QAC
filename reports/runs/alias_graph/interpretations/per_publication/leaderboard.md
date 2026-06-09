# Interpretation: per_publication

Gold = a single gold publication's language variants ("find this patent's cross-language versions"). Each (query, gold-publication) pair is one eval unit; the concept's other gold publications are excluded from the candidate ranking.

- Eval units: **6609**  ·  avg relevant docs per unit: **2.2**
- Models ranked by Recall@10 (re-scored from the saved predictions; no model re-run).

| Rank | Model | Recall@10 | Recall@100 | Precision@10 | nDCG@10 | MRR | MAP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `google-embeddinggemma-300m` | 0.0389 | 0.1179 | 0.0095 | 0.0290 | 0.0325 | 0.0281 |
| 2 | `qwen-qwen3-embedding-0-6b` | 0.0275 | 0.0872 | 0.0066 | 0.0221 | 0.0274 | 0.0213 |
| 3 | `nomic-ai-nomic-embed-text-v2-moe` | 0.0248 | 0.0766 | 0.0060 | 0.0202 | 0.0259 | 0.0190 |
| 4 | `baai-bge-m3` | 0.0233 | 0.0745 | 0.0056 | 0.0186 | 0.0234 | 0.0178 |
| 5 | `ibm-granite-granite-embedding-278m-multilingual` | 0.0203 | 0.0659 | 0.0048 | 0.0162 | 0.0198 | 0.0160 |
| 6 | `sentence-transformers-labse` | 0.0181 | 0.0581 | 0.0044 | 0.0151 | 0.0198 | 0.0145 |
| 7 | `intfloat-multilingual-e5-large-instruct` | 0.0148 | 0.0527 | 0.0035 | 0.0114 | 0.0193 | 0.0090 |
| 8 | `cambridgeltl-sapbert-umls-2020ab-all-lang-from-xlmr` | 0.0142 | 0.0586 | 0.0034 | 0.0112 | 0.0161 | 0.0108 |
| 9 | `alibaba-nlp-gte-multilingual-base` | 0.0027 | 0.0099 | 0.0007 | 0.0020 | 0.0037 | 0.0015 |
