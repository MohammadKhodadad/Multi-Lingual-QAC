# Question-level analysis (MehdiAstaraki/multi-lingual-qac-chem-patents, `multilingual`, Recall@10 / MRR@10)

## Dataset structure
- Queries with relevance judgements: 524
- Original: 257  |  synthetic-translation: 267
- Questions by mode: technical=265, semantic=259
- Questions by strategy: random=127, random_missing=128, random_existing=121, all=128, forced_zh=20
- Queries by language: de=112, en=119, es=103, fr=98, zh=92
- Relevant (query, doc) pairs: 1284 (avg 2.45/query)
- Models analysed: Alibaba-NLP/gte-multilingual-base, BAAI/bge-m3, cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR, google/embeddinggemma-300m, ibm-granite/granite-embedding-278m-multilingual, intfloat/multilingual-e5-large-instruct, nomic-ai/nomic-embed-text-v2-moe, Qwen/Qwen3-Embedding-0.6B, sentence-transformers/LaBSE

## 1) Recall@10 by query language
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 112 | 0.000 | 0.445 | 0.220 | 0.539 | 0.379 | 0.163 | 0.412 | 0.450 | 0.211 |
| en | 119 | 0.004 | 0.525 | 0.240 | 0.613 | 0.464 | 0.377 | 0.527 | 0.595 | 0.401 |
| es | 103 | 0.003 | 0.518 | 0.228 | 0.560 | 0.395 | 0.129 | 0.488 | 0.469 | 0.237 |
| fr | 98 | 0.014 | 0.574 | 0.252 | 0.596 | 0.474 | 0.271 | 0.531 | 0.463 | 0.265 |
| zh | 92 | 0.004 | 0.487 | 0.252 | 0.558 | 0.359 | 0.123 | 0.364 | 0.487 | 0.391 |

##    MRR@10 by query language
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 112 | 0.000 | 0.411 | 0.195 | 0.462 | 0.341 | 0.179 | 0.395 | 0.383 | 0.238 |
| en | 119 | 0.002 | 0.532 | 0.268 | 0.598 | 0.471 | 0.530 | 0.563 | 0.582 | 0.354 |
| es | 103 | 0.005 | 0.468 | 0.199 | 0.505 | 0.376 | 0.170 | 0.459 | 0.426 | 0.233 |
| fr | 98 | 0.008 | 0.536 | 0.218 | 0.577 | 0.428 | 0.417 | 0.539 | 0.479 | 0.281 |
| zh | 92 | 0.002 | 0.488 | 0.255 | 0.509 | 0.328 | 0.202 | 0.372 | 0.463 | 0.365 |

## 2) Recall@10 by question mode (technical vs semantic)
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| technical | 265 | 0.002 | 0.370 | 0.185 | 0.408 | 0.268 | 0.167 | 0.336 | 0.342 | 0.241 |
| semantic | 259 | 0.008 | 0.652 | 0.292 | 0.744 | 0.567 | 0.271 | 0.601 | 0.653 | 0.362 |

## 3) Recall@10 by question strategy
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 127 | 0.008 | 0.544 | 0.236 | 0.602 | 0.425 | 0.267 | 0.496 | 0.547 | 0.356 |
| random_missing | 128 | 0.000 | 0.507 | 0.227 | 0.535 | 0.387 | 0.094 | 0.450 | 0.471 | 0.258 |
| random_existing | 121 | 0.007 | 0.585 | 0.296 | 0.680 | 0.541 | 0.380 | 0.560 | 0.569 | 0.348 |
| all | 128 | 0.005 | 0.410 | 0.198 | 0.491 | 0.329 | 0.164 | 0.391 | 0.408 | 0.230 |
| forced_zh | 20 | 0.000 | 0.475 | 0.225 | 0.533 | 0.325 | 0.075 | 0.317 | 0.450 | 0.392 |

## 4) Recall@10 by query origin (original vs synthetic-translation)
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 257 | 0.010 | 0.532 | 0.266 | 0.624 | 0.466 | 0.351 | 0.512 | 0.525 | 0.341 |
| synthetic-translation | 267 | 0.000 | 0.487 | 0.211 | 0.526 | 0.367 | 0.091 | 0.424 | 0.468 | 0.262 |

## 5) Cross-lingual targets: same- vs cross-language (mean Recall@10)
| Target | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|
| same-language target | 0.023 | 0.659 | 0.372 | 0.716 | 0.533 | 0.697 | 0.682 | 0.651 | 0.421 |
| cross-language target | 0.000 | 0.464 | 0.197 | 0.541 | 0.390 | 0.094 | 0.410 | 0.451 | 0.271 |

## 6) Language-pair Recall@10 matrix — google/embeddinggemma-300m (best model)
Rows = query language, Cols = relevant-doc language; cell = fraction of those
relevant docs retrieved in the top 10 (n = #relevant pairs).

| q\d | de | en | es | fr | zh |
|---|---|---|---|---|---|
| **de** | 0.77 (13) | 0.55 (112) | 0.42 (24) | 0.54 (90) | 0.49 (39) |
| **en** | 0.18 (11) | 0.69 (119) | 0.60 (40) | 0.57 (80) | 0.47 (36) |
| **es** | 0.52 (23) | 0.53 (103) | 0.78 (27) | 0.54 (78) | 0.42 (31) |
| **fr** | 0.23 (13) | 0.55 (97) | 0.63 (19) | 0.72 (80) | 0.31 (32) |
| **zh** | 0.70 (10) | 0.55 (92) | 0.54 (28) | 0.54 (65) | 0.73 (22) |

