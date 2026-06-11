# Question-level analysis (MehdiAstaraki/multi-lingual-qac-epo, `multilingual`, Recall@10 / MRR@10)

## Dataset structure
- Queries with relevance judgements: 198
- Original: 198  |  synthetic-translation: 0
- Questions by mode: technical=100, semantic=98
- Questions by strategy: random=51, random_missing=49, random_existing=50, all=48
- Queries by language: de=58, en=72, fr=68
- Relevant (query, doc) pairs: 594 (avg 3.00/query)
- Models analysed: Alibaba-NLP/gte-multilingual-base, BAAI/bge-m3, cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR, google/embeddinggemma-300m, ibm-granite/granite-embedding-278m-multilingual, intfloat/multilingual-e5-large-instruct, nomic-ai/nomic-embed-text-v2-moe, Qwen/Qwen3-Embedding-0.6B, sentence-transformers/LaBSE

## 1) Recall@10 by query language
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 58 | 0.006 | 0.592 | 0.161 | 0.586 | 0.345 | 0.213 | 0.483 | 0.517 | 0.241 |
| en | 72 | 0.005 | 0.574 | 0.264 | 0.634 | 0.472 | 0.444 | 0.602 | 0.546 | 0.389 |
| fr | 68 | 0.005 | 0.475 | 0.225 | 0.515 | 0.338 | 0.211 | 0.436 | 0.343 | 0.176 |

##    MRR@10 by query language
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 58 | 0.006 | 0.645 | 0.234 | 0.586 | 0.336 | 0.500 | 0.610 | 0.563 | 0.287 |
| en | 72 | 0.003 | 0.646 | 0.314 | 0.608 | 0.455 | 0.558 | 0.644 | 0.554 | 0.362 |
| fr | 68 | 0.002 | 0.469 | 0.221 | 0.537 | 0.341 | 0.439 | 0.451 | 0.427 | 0.200 |

## 2) Recall@10 by question mode (technical vs semantic)
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| technical | 100 | 0.003 | 0.460 | 0.277 | 0.460 | 0.267 | 0.233 | 0.437 | 0.387 | 0.273 |
| semantic | 98 | 0.007 | 0.633 | 0.163 | 0.701 | 0.514 | 0.361 | 0.585 | 0.551 | 0.272 |

## 3) Recall@10 by question strategy
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 51 | 0.013 | 0.608 | 0.261 | 0.654 | 0.399 | 0.301 | 0.490 | 0.471 | 0.261 |
| random_missing | 49 | 0.000 | 0.524 | 0.156 | 0.551 | 0.299 | 0.272 | 0.463 | 0.429 | 0.320 |
| random_existing | 50 | 0.000 | 0.467 | 0.180 | 0.473 | 0.353 | 0.273 | 0.453 | 0.427 | 0.207 |
| all | 48 | 0.007 | 0.583 | 0.285 | 0.639 | 0.507 | 0.340 | 0.639 | 0.549 | 0.306 |

## 4) Recall@10 by query origin (original vs synthetic-translation)
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 198 | 0.005 | 0.545 | 0.221 | 0.579 | 0.389 | 0.296 | 0.510 | 0.468 | 0.273 |

## 5) Cross-lingual targets: same- vs cross-language (mean Recall@10)
| Target | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|
| same-language target | 0.010 | 0.702 | 0.313 | 0.697 | 0.434 | 0.626 | 0.672 | 0.631 | 0.293 |
| cross-language target | 0.003 | 0.467 | 0.174 | 0.520 | 0.366 | 0.131 | 0.429 | 0.386 | 0.263 |

## 6) Language-pair Recall@10 matrix — google/embeddinggemma-300m (best model)
Rows = query language, Cols = relevant-doc language; cell = fraction of those
relevant docs retrieved in the top 10 (n = #relevant pairs).

| q\d | de | en | fr |
|---|---|---|---|
| **de** | 0.72 (58) | 0.48 (58) | 0.55 (58) |
| **en** | 0.58 (72) | 0.71 (72) | 0.61 (72) |
| **fr** | 0.43 (68) | 0.46 (68) | 0.66 (68) |

