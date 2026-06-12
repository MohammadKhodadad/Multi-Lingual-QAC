# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `sentence-transformers/LaBSE`: recall@10 0.37 → 0.37 (Δ=+0.01); mean cosine 0.516 → 0.508.

## Per-step cosine drop by swap mode
- `sentence-transformers/LaBSE`: B=-0.0002, C=0.0013, D=0.0016, F=0.0025
