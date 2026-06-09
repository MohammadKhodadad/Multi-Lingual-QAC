# MTEB Model Comparison

## Leaderboard

### Overview

- Dataset: `MehdiAstaraki/multi-lingual-qac-alias-graph`
- Models compared: `9`
- Best model by `recall_at_10`: `google/embeddinggemma-300m` (0.0807)

### Ranking

| Rank | Model | Main score | Recall@10 | Recall@100 | MAP@10 | MAP@100 | MAP | nDCG@10 | nDCG@100 | Same-lang irr@100 | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `google/embeddinggemma-300m` | **0.0807** | **0.0807** | **0.1515** | **0.2668** | **0.1187** | **0.1128** | **0.3937** | **0.2467** | 0.3627 | 324.2 |
| 2 | `Qwen/Qwen3-Embedding-0.6B` | 0.0643 | 0.0643 | 0.1262 | 0.2281 | 0.0792 | 0.0793 | 0.3623 | 0.2034 | 0.3514 | 291.1 |
| 3 | `nomic-ai/nomic-embed-text-v2-moe` | 0.0570 | 0.0570 | 0.1121 | 0.1981 | 0.0707 | 0.0677 | 0.3187 | 0.1828 | 0.3963 | 279.3 |
| 4 | `sentence-transformers/LaBSE` | 0.0561 | 0.0561 | 0.0972 | 0.1645 | 0.0628 | 0.0618 | 0.2651 | 0.1538 | **0.2587** | 125.4 |
| 5 | `ibm-granite/granite-embedding-278m-multilingual` | 0.0561 | 0.0561 | 0.1057 | 0.1698 | 0.0670 | 0.0663 | 0.2751 | 0.1635 | 0.2615 | 67.1 |
| 6 | `BAAI/bge-m3` | 0.0541 | 0.0541 | 0.1133 | 0.1855 | 0.0632 | 0.0645 | 0.3114 | 0.1792 | 0.3057 | 607.8 |
| 7 | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | 0.0496 | 0.0496 | 0.1020 | 0.1146 | 0.0583 | 0.0591 | 0.1961 | 0.1425 | 0.3725 | 179.1 |
| 8 | `intfloat/multilingual-e5-large-instruct` | 0.0292 | 0.0292 | 0.0732 | 0.0957 | 0.0360 | 0.0351 | 0.1788 | 0.1139 | 0.9027 | 195.2 |
| 9 | `Alibaba-NLP/gte-multilingual-base` | 0.0062 | 0.0062 | 0.0147 | 0.0161 | 0.0051 | 0.0050 | 0.0389 | 0.0236 | 0.7502 | 91.2 |

### Metric Winners

| Metric | Best model | Score |
| --- | --- | ---: |
| `Main score` | `google/embeddinggemma-300m` | 0.0807 |
| `Recall@10` | `google/embeddinggemma-300m` | 0.0807 |
| `Recall@100` | `google/embeddinggemma-300m` | 0.1515 |
| `MAP@10` | `google/embeddinggemma-300m` | 0.2668 |
| `MAP@100` | `google/embeddinggemma-300m` | 0.1187 |
| `MAP` | `google/embeddinggemma-300m` | 0.1128 |
| `nDCG@10` | `google/embeddinggemma-300m` | 0.3937 |
| `nDCG@100` | `google/embeddinggemma-300m` | 0.2467 |
| `Same-lang irr@100` | `sentence-transformers/LaBSE` | 0.2587 |
