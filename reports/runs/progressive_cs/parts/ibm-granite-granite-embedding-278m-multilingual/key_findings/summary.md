# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `ibm-granite/granite-embedding-278m-multilingual`: recall@10 0.55 → 0.54 (Δ=+0.01); mean cosine 0.717 → 0.714.

## Per-step cosine drop by swap mode
- `ibm-granite/granite-embedding-278m-multilingual`: B=-0.0007, C=0.0005, D=0.0007, F=0.0011
