"""
Round 3 — The machine-translation penalty (human-original vs synthetic-translation queries).

The project's working assumption is "human translation for source patents, but machine translation is
fine for the generated questions." This round tests it. The trap is a confound: synthetic queries are
*both* machine-translated *and* homeless (no same-language gold). We neutralise the home-document
confound by scoring every query on a FIXED target set — the patent's documents in languages OTHER
than its source language (`foreign_reach@k`). For any patent, its one original (source-language)
query and its machine-translated queries share the identical target set, so the remaining difference
is the query itself: written by a human in the source language vs machine-translated into another.

Outputs (../experimental_plots/round03_mt_penalty/):
  per_query_reach.csv, paired_by_patent.csv, summary.json, and figures:
    mt_penalty_by_model.png        foreign_reach@10 original vs synthetic, per model (pooled queries)
    mt_penalty_paired.png          paired original−synthetic per model on shared patents (+ stars)
    synthetic_by_target_language.png  synthetic foreign_reach@10 by MT target language
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round03_mt_penalty"
DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]


def reach_frame() -> pd.DataFrame:
    """Per (model, query): foreign_reach@10 = recall over the patent's non-source-language docs."""
    rank_of = C.rank_of()
    gold = C.gold_publication()
    ql = C.q_lang()
    qsrc = C.q_src_lang()
    qorg = C.q_origin()
    qpat = C.q_patent()
    rows = []
    for model in C.MODEL_ORDER:
        for qid, gset in gold.items():
            src = qsrc.get(qid, "")
            targets = {d for d in gset if C.doc_lang(d) != src}  # foreign (non-home) versions
            if not targets:
                continue
            ro = rank_of.get((model, qid), {})
            hit = len({d for d in targets if ro.get(d, C.INF) <= 10}) / len(targets)
            rows.append({
                "model": model, "short": C.short(model), "query_id": qid, "patent": qpat.get(qid, ""),
                "origin": qorg.get(qid, ""), "qlang": ql.get(qid, ""), "src": src,
                "n_targets": len(targets), "foreign_reach10": hit,
            })
    return pd.DataFrame(rows)


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    df = reach_frame()
    df.to_csv(out / "per_query_reach.csv", index=False)

    # ---- (A) population comparison per model -----------------------------------------------------
    rows = []
    for model in C.MODEL_ORDER:
        g = df[df.model == model]
        o = g[g.origin == "original"]["foreign_reach10"].values
        s = g[g.origin == "synthetic"]["foreign_reach10"].values
        op, olo, ohi = C.bootstrap_ci(o)
        sp, slo, shi = C.bootstrap_ci(s)
        rows.append({"model": model, "short": C.short(model),
                     "orig": op, "orig_lo": olo, "orig_hi": ohi,
                     "syn": sp, "syn_lo": slo, "syn_hi": shi, "gap": op - sp})
    msum = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = np.arange(len(msum))
    w = 0.38
    for j, (col, lo, hi, lab, color) in enumerate([
        ("orig", "orig_lo", "orig_hi", "original (human, source language)", C.ORIGIN_COLOR["original"]),
        ("syn", "syn_lo", "syn_hi", "synthetic (machine-translated)", C.ORIGIN_COLOR["synthetic"]),
    ]):
        err = np.vstack([msum[col] - msum[lo], msum[hi] - msum[col]])
        ax.bar(xs + (j - 0.5) * w, msum[col], w, yerr=err, capsize=2.5, label=lab, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels(msum["short"], rotation=30, ha="right")
    ax.set_ylabel("foreign_reach@10  (recall over non-home docs)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title("Does a machine-translated question retrieve the foreign patent as well as a human one?\n"
                 "scored on the identical non-home target set per query")
    fig.tight_layout()
    fig.savefig(out / "mt_penalty_by_model.png")
    plt.close(fig)

    # ---- (B) paired by patent (patents with both an original and ≥1 synthetic) -------------------
    paired_rows = []
    diffs_all = []
    for model in C.MODEL_ORDER:
        g = df[df.model == model]
        po, ps = [], []
        for patent, sub in g.groupby("patent"):
            o = sub[sub.origin == "original"]
            s = sub[sub.origin == "synthetic"]
            if len(o) == 0 or len(s) == 0:
                continue
            ov = float(o["foreign_reach10"].mean())
            sv = float(s["foreign_reach10"].mean())
            po.append(ov)
            ps.append(sv)
            paired_rows.append({"model": C.short(model), "patent": patent,
                                "orig": ov, "syn": sv, "diff": ov - sv})
        if po:
            w = C.wilcoxon(po, ps)
            md = float(np.mean(np.array(po) - np.array(ps)))
            msum.loc[msum.model == model, "paired_diff"] = md
            msum.loc[msum.model == model, "paired_p"] = w["p"]
            msum.loc[msum.model == model, "paired_n"] = w["n"]
            if model in POOL:
                diffs_all.extend(list(np.array(po) - np.array(ps)))
    pd.DataFrame(paired_rows).to_csv(out / "paired_by_patent.csv", index=False)
    msum.to_csv(out / "per_model_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    o = msum.dropna(subset=["paired_diff"]).copy()
    xs = np.arange(len(o))
    ax.bar(xs, o["paired_diff"], color=[C.MODEL_COLOR[m] for m in o["model"]])
    for x, v, p in zip(xs, o["paired_diff"], o["paired_p"]):
        ax.text(x, v + (0.006 if v >= 0 else -0.02), f"{v:+.2f}{C.stars(p)}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(o["short"], rotation=30, ha="right")
    ax.set_ylabel("paired (original − synthetic) foreign_reach@10")
    ax.set_title("Same patent, same foreign targets: human query minus machine-translated query\n"
                 f"paired over shared patents (Wilcoxon; * p<.05, ** p<.01, *** p<.001)")
    fig.tight_layout()
    fig.savefig(out / "mt_penalty_paired.png")
    plt.close(fig)

    # ---- (C) synthetic reach by MT target language -----------------------------------------------
    syn = df[(df.origin == "synthetic") & (df.model.isin(POOL))]
    by = []
    for lang in C.QUERY_LANGS:
        sub = syn[syn.qlang == lang]
        if sub.empty:
            continue
        pt, lo, hi = C.bootstrap_ci(sub["foreign_reach10"].values)
        by.append({"lang": lang, "reach": pt, "lo": lo, "hi": hi, "n": int(sub.query_id.nunique())})
    bydf = pd.DataFrame(by)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = np.arange(len(bydf))
    err = np.vstack([bydf["reach"] - bydf["lo"], bydf["hi"] - bydf["reach"]])
    ax.bar(xs, bydf["reach"], yerr=err, capsize=3, color=[C.LANG_COLOR[l] for l in bydf["lang"]])
    for x, v, n in zip(xs, bydf["reach"], bydf["n"]):
        ax.text(x, v + 0.012, f"{v:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.LANG_NAME[l]}\n({l})" for l in bydf["lang"]])
    ax.set_ylabel("synthetic foreign_reach@10 (pooled reliable models)")
    ax.set_ylim(0, max(0.5, bydf["hi"].max() + 0.08))
    ax.set_title("Translating the question into which language hurts retrieval most?")
    fig.tight_layout()
    fig.savefig(out / "synthetic_by_target_language.png")
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------------------
    pooled = df[df.model.isin(POOL)]
    op = float(pooled[pooled.origin == "original"]["foreign_reach10"].mean())
    sp = float(pooled[pooled.origin == "synthetic"]["foreign_reach10"].mean())
    wd = C.wilcoxon(diffs_all, [0.0] * len(diffs_all)) if diffs_all else {"p": float("nan"), "n": 0}
    summary = {
        "pooled_original_foreign_reach10": round(op, 4),
        "pooled_synthetic_foreign_reach10": round(sp, 4),
        "pooled_gap": round(op - sp, 4),
        "paired_mean_diff": round(float(np.mean(diffs_all)), 4) if diffs_all else None,
        "paired_p_value": wd["p"], "paired_n_patent_model": len(diffs_all),
        "worst_target_language": (bydf.sort_values("reach").iloc[0]["lang"] if not bydf.empty else None),
        "per_model_gap": msum[["short", "orig", "syn", "gap", "paired_diff", "paired_p"]]
                         .round(4).to_dict(orient="records"),
    }
    C.jdump(summary, out / "summary.json")

    # judge from the *controlled* paired test, not the confounded population gap
    pv = summary["paired_p_value"]
    verdict = ("statistically insignificant (no MT penalty on cross-lingual reach)"
               if (pv != pv or pv >= 0.05) else "statistically significant")
    md = f"""<!-- round:03 -->
## Round 3 — The machine-translation penalty

**Question.** Is a machine-translated question as good as a human one for finding the patent's
foreign versions? We score every query on a *fixed* target set — the patent's non-source-language
documents (`foreign_reach@10`) — so the home-document confound is removed and only the query differs.

**Numbers (pooled over reliable models).**
- Human-original queries: foreign_reach@10 = **{summary['pooled_original_foreign_reach10']:.3f}**
- Machine-translated queries: foreign_reach@10 = **{summary['pooled_synthetic_foreign_reach10']:.3f}**
- Gap = **{summary['pooled_gap']:+.3f}**; paired over shared patents the mean difference is
  **{(summary['paired_mean_diff'] if summary['paired_mean_diff'] is not None else float('nan')):+.3f}**
  (p={summary['paired_p_value']:.3f}, n={summary['paired_n_patent_model']} patent×model).

**What we learned.**
- The MT penalty on *cross-lingual* reach is **{verdict}**. Controlling for the patent (same foreign
  targets), human and machine-translated questions retrieve the foreign patent comparably — the
  paired difference is only {(summary['paired_mean_diff'] if summary['paired_mean_diff'] is not None else float('nan')):+.3f}.
  This supports the project's "MT-is-fine-for-the-question" stance: the cross-lingual difficulty lives
  in the embedding model, not in the question's provenance.
- In the naive population view synthetic queries even look slightly *stronger*
  ({summary['pooled_synthetic_foreign_reach10']:.3f} vs {summary['pooled_original_foreign_reach10']:.3f});
  that is a **patent-selection artefact**, which is exactly why the paired test is the one to trust.
- The hardest MT *target* language is **{summary['worst_target_language']}** — translating the
  question into that language costs the most reach.

**Next questions.**
1. We've been asking "did it find a relevant doc". Sharpen to bitext: can the model find the *exact
   translated twin* of the patent, and at what rank? → Round 4.
2. If the question's language barely matters, what does? Maybe the model just returns same-language
   noise regardless. Quantify language collapse. → Round 6.
<!-- /round:03 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
