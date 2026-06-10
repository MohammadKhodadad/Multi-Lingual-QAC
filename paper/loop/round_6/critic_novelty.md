# Novelty review (round 6)

Reviewer #1 — Novelty Critic. EMNLP Industry Track. **Length-compliance round.**
Round 6 compressed Related Work 7¶→1¶ (lines 184–222 of `paper/main.tex`) and fixed the
bibliography (real authors/titles for the 7 previously-Anonymous + 2 weak entries). No new
claim was added; no analysis changed. My focused job: confirm the compression did not (a)
drop a citation load-bearing for a novelty boundary, or (b) over-claim by removing a hedge,
and confirm the three corrected titles still match the prose. I did this by a hard
key-level diff of the R5 (pre-compression) vs R6 (post-compression) Related Work, a
phrase-level diff of every novelty-boundary hedge, and a title-vs-prose check on the three
flagged cites. No new web search was needed — round 6 asserts nothing new, and every prior-art
boundary was already pinned in rounds 1–5.

## Verdict (1 paragraph)

**The compression fully preserved the novelty positioning, and the corrected titles
introduce no citation–claim mismatch — ship it.** The 7¶→1¶ collapse is a **zero-key-diff**:
all 28 Related Work `\cite` keys survive exactly (verified by `comm` — empty in both
directions), including all eight novelty-boundary keys the conductor named (clefip2013,
dapfam2025, clirmatrix2020, whatdrivesclir2025, crosslingualcost2025, nepotism2025, xrag2025,
bordirlines2024). Every load-bearing hedge that fences a novelty boundary survived and several
are *sharpened*: the alignment-not-translation boundary now reads "which we *confirm*, not
discover" (the "not discover" disclaimer is more explicit than R5's "we confirm it on …"),
the C1-defense survives intact ("isolate the language effect" + "content-controlled" +
"removes their confounds"), DAPFAM's "may differ in their claims … we reject" is intact,
CLEF-IP's "not chemistry-grounded" is intact, and patentembeddings2026's "English-only … a
different ordering" is intact. On the three corrected titles: **whatdrivesclir2025** — the
prose ("attributes the gap to weak alignment over translation") is squarely supported by a
paper titled *What Drives Cross-lingual Ranking?*, no over-reach; **crosslingualcost2025** —
the now-Arabic-English-specific title is *matched* by an inline "(Arabic--English)" qualifier
added directly before the claim, and the prose attributes only "documents the same-language
head start," so there is no over-generalization (this was the one real mismatch risk and the
writer fenced it correctly); **conformalrag2025** — the prose no longer uses the framework
name "Conformal-RAG" at all (it is now a bare key inside the grouped "Conformal/calibration-IR
tools \citep{traq2023,conflare2024,conformalrag2025}"), so the framework-name-vs-real-title
concern is *moot* post-compression — there is zero mismatch surface left in the body. No
hedge was lost, no boundary citation dropped, no contribution silently widened.

## Checks run (all pass)

1. **Key diff R5→R6 Related Work.** 28 keys both sides, `comm -23` and `comm -13` both empty.
   Nothing dropped, nothing silently added. The eight flagged boundary keys all present.
2. **Hedge/boundary phrase diff.** All R5 boundary phrases reappear in R6: `content-controlled`,
   `isolate the language`, `confirm}, not discover` (sharpened from R5's `confirm} it on`),
   `removes their confounds`, `complementary`, `not chemistry-grounded`, `reject`,
   `may differ in their claims`, `English-only`, `different ordering`. No hedge deleted.
3. **Corrected-title vs prose** (3 cites) — all match; crosslingualcost2025 fenced with an
   inline "(Arabic--English)"; conformalrag2025's framework name removed from prose (no
   mismatch surface).
4. **No orphaned/over-claimed key elsewhere.** The compression did not remove the sole use of
   any key a later section relies on. The conformal-version-of-XRC *horizon* (not a claim)
   was relocated out of Related Work but still lives in Limitations (line 986), correctly
   flagged as future work — not lost.
5. **Bibliography.** Zero `Anonymous`/`anon` entries remain. The only uncited keys are the
   stock ACL template stubs (Aho:72, APA:83, Chandra:81, andrew2007scalable, Gusfield:97,
   rasooli-tetrault-2015, Ando2005) — boilerplate, never intended to be cited, not a regression.

## Highest-risk items (ranked)

1. **(Watch, not a defect) crosslingualcost2025 over-generalization.** The corrected title
   restricts the study to Arabic–English; the prose is correctly scoped by the inline
   "(Arabic--English)" qualifier and claims only the same-language head start. Camera-ready
   must keep that parenthetical — if a future trim drops it, the cite would over-generalize a
   single-language-pair result to "the cross-lingual cost." Correct as written; flag only.
2. **(Carried, unchanged) "alignment-only" is still an inference, not an intervention.** Tied
   to the forthcoming causal probe in Limitations; the compression did not touch this. Low.

No item rises to "a hostile reviewer rejects on novelty grounds."

## Missing citations

**None.** No new claim, no new citation surface. The round-5 ρ_k cite (residualrerank2026)
remains present and correctly bounded; the bibliography is now fully de-anonymized.

## Convergence note (novelty axis)

The paper is **done on novelty.** Round 6 is a pure length-compliance pass that lost no
citation, no hedge, and no boundary, and the de-anonymized titles now back the prose more
precisely than before (the Arabic–English scope is now visible at the cite site). Freeze and
submit.
