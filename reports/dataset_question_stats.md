# Dataset Question Statistics

Statistics are computed from the public Hugging Face releases using the
`corpus`, `queries`, `qrels`, and `cross_language-qrels` configs.

| Dataset | Corpus rows | Queries | Qrels | Cross-lang qrels | Languages | Mean question tokens | Std. question tokens | Question vocab | Mean corpus tokens | Std. corpus tokens | Corpus vocab |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Google Patents | 23,787 | 524 | 1,284 | 1,023 | de, en, es, fr, zh | 11.3 | 6.3 | 2,742 | 145.0 | 63.8 | 83,333 |
| EPO | 11,315 | 198 | 594 | 396 | de, en, fr | 14.1 | 5.4 | 1,371 | 178.2 | 71.1 | 72,256 |

Dataset URLs:

- Google Patents: https://huggingface.co/datasets/MehdiAstaraki/multi-lingual-qac-chem-patents
- EPO: https://huggingface.co/datasets/MehdiAstaraki/multi-lingual-qac-epo
