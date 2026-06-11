# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `intfloat/multilingual-e5-large-instruct`: recall@10 0.34 → 0.35 (Δ=-0.01); mean cosine 0.833 → 0.833.

## Per-step cosine drop by swap mode
- `intfloat/multilingual-e5-large-instruct`: B=-0.0015, C=-0.0010, D=0.0002, F=0.0008
