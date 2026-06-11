# MTEB Model Comparison

## Leaderboard

### Overview

- Dataset: `MehdiAstaraki/multi-lingual-qac-epo`
- Models compared: `9`
- Best model by `recall_at_10`: `google/embeddinggemma-300m` (0.5791)

### Ranking

| Rank | Model | Main score | Recall@10 | Recall@100 | MAP@10 | MAP@100 | MAP | nDCG@10 | nDCG@100 | Same-lang irr@100 | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `google/embeddinggemma-300m` | **0.5791** | **0.5791** | **0.8064** | **0.4625** | **0.4807** | **0.4821** | **0.5263** | **0.5904** | 0.5398 | 230.1 |
| 2 | `BAAI/bge-m3` | 0.5454 | 0.5454 | 0.7609 | 0.4209 | 0.4396 | 0.4412 | 0.4988 | 0.5611 | 0.4393 | 433.1 |
| 3 | `nomic-ai/nomic-embed-text-v2-moe` | 0.5101 | 0.5101 | 0.7121 | 0.4069 | 0.4234 | 0.4251 | 0.4758 | 0.5351 | 0.5035 | 158.7 |
| 4 | `Qwen/Qwen3-Embedding-0.6B` | 0.4680 | 0.4680 | 0.7087 | 0.3723 | 0.3895 | 0.3910 | 0.4338 | 0.5020 | 0.5601 | 204.9 |
| 5 | `ibm-granite/granite-embedding-278m-multilingual` | 0.3889 | 0.3889 | 0.5960 | 0.3106 | 0.3273 | 0.3287 | 0.3529 | 0.4129 | 0.3678 | 45.1 |
| 6 | `intfloat/multilingual-e5-large-instruct` | 0.2963 | 0.2963 | 0.4512 | 0.2281 | 0.2390 | 0.2409 | 0.3067 | 0.3523 | 0.9597 | 171.2 |
| 7 | `sentence-transformers/LaBSE` | 0.2727 | 0.2727 | 0.5252 | 0.1860 | 0.2036 | 0.2056 | 0.2382 | 0.3103 | **0.3307** | 74.0 |
| 8 | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | 0.2205 | 0.2205 | 0.4125 | 0.1505 | 0.1629 | 0.1651 | 0.1969 | 0.2512 | 0.4948 | 119.7 |
| 9 | `Alibaba-NLP/gte-multilingual-base` | 0.0050 | 0.0050 | 0.0253 | 0.0012 | 0.0026 | 0.0029 | 0.0029 | 0.0092 | 0.8056 | 54.4 |

### Metric Winners

| Metric | Best model | Score |
| --- | --- | ---: |
| `Main score` | `google/embeddinggemma-300m` | 0.5791 |
| `Recall@10` | `google/embeddinggemma-300m` | 0.5791 |
| `Recall@100` | `google/embeddinggemma-300m` | 0.8064 |
| `MAP@10` | `google/embeddinggemma-300m` | 0.4625 |
| `MAP@100` | `google/embeddinggemma-300m` | 0.4807 |
| `MAP` | `google/embeddinggemma-300m` | 0.4821 |
| `nDCG@10` | `google/embeddinggemma-300m` | 0.5263 |
| `nDCG@100` | `google/embeddinggemma-300m` | 0.5904 |
| `Same-lang irr@100` | `sentence-transformers/LaBSE` | 0.3307 |
