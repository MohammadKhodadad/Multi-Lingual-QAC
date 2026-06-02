# Question-level analysis (MehdiAstaraki/multi-lingual-qac-chem-patents, `multilingual`, Recall@10 / MRR@10)

## Dataset structure
- Queries with relevance judgements: 137
- Original: 57  |  synthetic-translation: 80
- Questions by mode: technical=70, semantic=67
- Questions by strategy: random=30, random_missing=30, random_existing=24, all=33, forced_zh=20
- Queries by language: de=27, en=27, es=34, fr=27, zh=22
- Relevant (query, doc) pairs: 322 (avg 2.35/query)
- Models analysed: BAAI/bge-m3, intfloat/multilingual-e5-large, sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, sentence-transformers/paraphrase-multilingual-mpnet-base-v2

## 1) Recall@10 by query language
| Group | n | BAAI/bge-m3 | intfloat/multilingual-e5-large | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
|---|---|---|---|---|---|
| de | 27 | 0.784 | 0.543 | 0.506 | 0.623 |
| en | 27 | 0.654 | 0.667 | 0.463 | 0.506 |
| es | 34 | 0.740 | 0.676 | 0.520 | 0.623 |
| fr | 27 | 0.704 | 0.494 | 0.506 | 0.611 |
| zh | 22 | 0.689 | 0.258 | 0.470 | 0.568 |

##    MRR@10 by query language
| Group | n | BAAI/bge-m3 | intfloat/multilingual-e5-large | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
|---|---|---|---|---|---|
| de | 27 | 0.742 | 0.471 | 0.517 | 0.564 |
| en | 27 | 0.643 | 0.690 | 0.402 | 0.475 |
| es | 34 | 0.700 | 0.588 | 0.433 | 0.545 |
| fr | 27 | 0.639 | 0.617 | 0.419 | 0.575 |
| zh | 22 | 0.622 | 0.128 | 0.484 | 0.485 |

## 2) Recall@10 by question mode (technical vs semantic)
| Group | n | BAAI/bge-m3 | intfloat/multilingual-e5-large | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
|---|---|---|---|---|---|
| technical | 70 | 0.545 | 0.410 | 0.364 | 0.417 |
| semantic | 67 | 0.896 | 0.687 | 0.632 | 0.769 |

## 3) Recall@10 by question strategy
| Group | n | BAAI/bge-m3 | intfloat/multilingual-e5-large | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
|---|---|---|---|---|---|
| random | 30 | 0.739 | 0.639 | 0.506 | 0.567 |
| random_missing | 30 | 0.750 | 0.578 | 0.472 | 0.656 |
| random_existing | 24 | 0.792 | 0.653 | 0.576 | 0.604 |
| all | 33 | 0.636 | 0.530 | 0.444 | 0.556 |
| forced_zh | 20 | 0.675 | 0.250 | 0.500 | 0.558 |

## 4) Recall@10 by query origin (original vs synthetic-translation)
| Group | n | BAAI/bge-m3 | intfloat/multilingual-e5-large | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
|---|---|---|---|---|---|
| original | 57 | 0.722 | 0.605 | 0.509 | 0.588 |
| synthetic-translation | 80 | 0.712 | 0.502 | 0.485 | 0.590 |

## 5) Cross-lingual targets: same- vs cross-language (mean Recall@10)
| Target | BAAI/bge-m3 | intfloat/multilingual-e5-large | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
|---|---|---|---|---|
| same-language target | 0.842 | 0.877 | 0.614 | 0.632 |
| cross-language target | 0.682 | 0.461 | 0.462 | 0.574 |

## 6) Language-pair Recall@10 matrix — BAAI/bge-m3 (best model)
Rows = query language, Cols = relevant-doc language; cell = fraction of those
relevant docs retrieved in the top 10 (n = #relevant pairs).

| q\d | de | en | fr | zh |
|---|---|---|---|---|
| **de** | 0.71 (7) | 0.86 (21) | 0.74 (27) | 0.75 (8) |
| **en** | 0.43 (7) | 0.95 (21) | 0.48 (27) | 0.50 (8) |
| **es** | 0.67 (12) | 0.70 (27) | 0.76 (34) | 0.54 (13) |
| **fr** | 0.44 (9) | 0.74 (19) | 0.78 (27) | 0.50 (8) |
| **zh** | 0.75 (4) | 0.84 (19) | 0.55 (22) | 1.00 (2) |

