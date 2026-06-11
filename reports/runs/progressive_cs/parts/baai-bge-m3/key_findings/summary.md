# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `BAAI/bge-m3`: recall@10 0.66 → 0.64 (Δ=+0.02); mean cosine 0.574 → 0.568.

## Per-step cosine drop by swap mode
- `BAAI/bge-m3`: B=-0.0008, C=0.0027, D=0.0003, F=0.0025
