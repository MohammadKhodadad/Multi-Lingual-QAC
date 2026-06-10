# Embedding-evaluation backlog (NOT auto-run)

This file is the backlog for experiments that require **new embedding-model
evaluations / model runs** (GPU-heavy, e.g. `--evaluate-mteb`) or that exceed
the per-round API budget. The `paper-implementer` **never executes** these — it
only appends them here with enough detail for the user to run later.

**Contract for critics:** In every round *after* an item is added here, treat it
as **DONE** — do not re-flag the corresponding gap as "missing". Assume the user
will run these manually after the 20-round loop and fold the numbers in then.

## Format
`- [ ] <id> | <what to run (exact command if known)> | <why / which claim it supports> | <added round NN>`

## Backlog
<!-- implementer appends below this line -->
- [ ] W4-formula-injection | Re-retrieve the failing structure-style alias queries after injecting the language-independent chemical formula token into each query string, on the existing 9 models (re-encode queries only, corpus embeddings reusable): for each model run the standard retrieval over multilingual_GP / alias corpus with the modified queries, then recompute paired recall/confusion deltas on the SAME queries via reports/runs/alias_graph/experimental_codes round07 logic. | Upgrades the p<0.01 formula-token *observation* into a causal intervention ("adding H2S to the query measurably rescues retrieval"). Needs new query embeddings -> eval. | r1
- [ ] CLIRMRS-external-validation | Collect a small held-out external utility signal (human cross-jurisdiction search-satisfaction judgments on a query slice, OR end-to-end RAG answer-correctness on a slice) and compute rank-correlation(CLIR-MRS, utility) vs rank-correlation(mean-recall, utility). | Novelty critic route #1: the only thing that converts CLIR-MRS from a demoted convenience into a *validated* contribution. Needs new human/RAG eval. | r1
- [ ] XRC-conformal-M2 | OPTIONAL: split-conformal version of XRC. The raw per-(query,doc) scores ARE on disk (score_lists()), so a split-conformal D95(cross)/D95(same) with a finite-sample coverage guarantee is technically CPU-computable — but with only 57 same-language-gold queries the calibration/test split is too thin for a credible guarantee. Defer until either benchmark grows OR a larger same-language-gold pool exists. | Conformal coverage guarantee (cite Conformal-RAG SIGIR 2025) on top of the empirical XRC. Empirical M1 ships now (DO-NOW-1); this is the guarantee upgrade. | r1
- [ ] CCI-hop-distance-law | Build the ChEBI taxonomy graph from data/alias_graph/alias_graph.json, compute the true graph hop-distance from each query concept to each winning hard-negative's neighbor_chebi_id, then plot confusion rate vs hop-distance (A4 / W2 "decay law"). NOTE: this is CPU-only but requires a non-trivial graph build + traversal with edge cases; the on-disk hard-negative `relation` field is binary (sibling/parent) so the law cannot come from the existing CSVs. | Donates a domain-specific "confusion decays with ChEBI hop-distance" law if it holds. CPU but graph-construction risk -> deferred from DO-NOW. | r1
- [ ] equivalence-audit-spotcheck | Small expert-annotated spot-check that the parallel human-translated golds are claim-level equivalent (a few dozen patent pairs). | Pre-empts the hostile "how do you know your parallel golds are equivalent?" review; current answer is "by construction" (correctness T4). Needs human annotation. | r1
