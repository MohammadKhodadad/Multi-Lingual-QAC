# Case-B fixability via the ChEBI Wikipedia name cache

- ChEBI concepts in cache with a Chinese name: **7623**
- Case B (zh docs with >=1 Latin word): **304** of 400

| Definition of "fixed" | Count | % of case B |
|---|---|---|
| >=1 non-Chinese word fixable (concept has zh) | 80 | 26% |
| Fully cleaned (all Latin words replaceable) | 3 | 1% |
| Latin word-TYPES fixable | 148/1983 | 7% |
| Hard ceiling: >=1 word maps to ANY ChEBI concept | 103 | 34% |

**Hard ceiling**: 201/304 (66%) case-B docs contain NO ChEBI-nameable Latin word (only units / element symbols / acronyms / English prose), so the ">=1 fixable" rate cannot exceed the ceiling no matter how complete the cache is.

## Unfixable Latin token-types by category

| Category | Token-types | Occurrences | Examples |
|---|---|---|---|
| other non-chemical Latin | 1321 | 2344 | aluminium, argon, converter, helium, krypton, magnesium, mao, oxygen, quasi, resonance |
| element symbol | 120 | 225 | al, at, co, cr, fe, li, lu, mn, nb, rb |
| english prose (needs MT) | 114 | 313 | and, by, from, less, more, no, not, of, or, than |
| unit | 103 | 239 | gpa, mass, mg, ml, mm, mol, mw, ng, nm, ppm |
| acronym / markup / roman | 103 | 246 | dna, dpp, fc, formula, id, ii, iii, iv, nmr, pcr |
| chemical (ChEBI match, no Chinese Wikipedia article) | 74 | 168 | acrylate, anthraquinone, antigen, cobalamins, filler, halides, maleate, methacrylate, methyl, shikimate |
