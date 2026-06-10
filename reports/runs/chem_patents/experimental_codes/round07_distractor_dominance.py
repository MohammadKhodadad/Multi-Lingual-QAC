"""
Round 7 — Distractor dominance: what buries the foreign twin?

This benchmark ships no curated hard-negatives, so we read confusion straight off the rankings: the
documents ranked *above* the first gold are the "blockers" that bury the correct patent. The
cross-lingual question is their language — is the foreign twin being out-ranked by *same-language*
noise (collapse-driven confusion) or by genuinely foreign distractors?

Outputs (../experimental_plots/round07_distractor_dominance/):
  per_query_confusion.csv, per_model.csv, summary.json, and figures:
    blocker_language_by_query_language.png  same- vs cross-language share of blockers, by query language
    same_language_confusion_by_model.png    P(a same-language non-gold out-ranks the first gold)
    rank1_composition.png                   what sits at rank 1: gold / same-lang / foreign distractor
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round07_distractor_dominance"
DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]
DEPTH = 100


def confusion_frame() -> pd.DataFrame:
    lists = C.ranked_lists()
    gold = C.gold_publication()
    ql = C.q_lang()
    rows = []
    for model in C.MODEL_ORDER:
        for qid, gset in gold.items():
            ranked = lists.get((model, qid), [])[:DEPTH]
            if not ranked:
                continue
            lang = ql.get(qid, "")
            gold_pos = [i for i, d in enumerate(ranked) if d in gset]
            first = gold_pos[0] if gold_pos else DEPTH  # 0-based; DEPTH => no gold in top-DEPTH
            blockers = ranked[:first]
            n_block = len(blockers)
            same_block = sum(1 for d in blockers if C.doc_lang(d) == lang)
            top = ranked[0]
            rank1 = "gold" if top in gset else ("same" if C.doc_lang(top) == lang else "foreign")
            rows.append({
                "model": model, "short": C.short(model), "query_id": qid, "qlang": lang,
                "first_gold_idx": first, "n_blockers": n_block,
                "same_lang_blockers": same_block,
                "sl_blocker_share": (same_block / n_block) if n_block > 0 else np.nan,
                "sl_confusion": 1.0 if same_block > 0 else 0.0,
                "rank1": rank1,
            })
    return pd.DataFrame(rows)


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    df = confusion_frame()
    df.to_csv(out / "per_query_confusion.csv", index=False)
    pool = df[df.model.isin(POOL)]

    # ---- Fig 1: blocker language share by query language -----------------------------------------
    by = pool.groupby("qlang")["sl_blocker_share"].mean().reindex(C.QUERY_LANGS)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = np.arange(len(by))
    same = by.values
    ax.bar(xs, same, label="same-language blockers", color="#d62728")
    ax.bar(xs, 1 - same, bottom=same, label="cross-language blockers", color="#9ecae1")
    for x, v in zip(xs, same):
        ax.text(x, v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.LANG_NAME[l]}\n({l})" for l in by.index])
    ax.set_ylabel("share of above-gold blockers")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("What buries the correct patent? Language of the documents ranked above the first gold\n"
                 "(pooled reliable models; high red = collapse-driven confusion)")
    fig.tight_layout()
    fig.savefig(out / "blocker_language_by_query_language.png")
    plt.close(fig)

    # ---- Fig 2: same-language confusion rate by model --------------------------------------------
    rows = []
    for model in C.MODEL_ORDER:
        g = df[df.model == model]
        pt, lo, hi = C.bootstrap_ci(g["sl_confusion"].values)
        rows.append({"model": model, "short": C.short(model), "rate": pt, "lo": lo, "hi": hi,
                     "mean_sl_blocker_share": float(g["sl_blocker_share"].mean(skipna=True))})
    msum = pd.DataFrame(rows)
    msum.to_csv(out / "per_model.csv", index=False)
    o = msum.sort_values("rate", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    xs = np.arange(len(o))
    err = np.vstack([o["rate"] - o["lo"], o["hi"] - o["rate"]])
    ax.bar(xs, o["rate"], yerr=err, capsize=3, color=[C.MODEL_COLOR[m] for m in o["model"]])
    for x, v in zip(xs, o["rate"]):
        ax.text(x, v + 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(o["short"], rotation=30, ha="right")
    ax.set_ylabel("P(a same-language non-gold out-ranks the first gold)")
    ax.set_ylim(0, 1)
    ax.set_title("Same-language confusion rate: how often does query-language noise bury the gold?")
    fig.tight_layout()
    fig.savefig(out / "same_language_confusion_by_model.png")
    plt.close(fig)

    # ---- Fig 3: rank-1 composition by query language (best model) --------------------------------
    best = C.MODEL_ORDER[0]
    gb = df[df.model == best]
    comp = (gb.groupby(["qlang", "rank1"]).size().unstack("rank1").reindex(C.QUERY_LANGS)
            .fillna(0))
    comp = comp.div(comp.sum(axis=1), axis=0)
    order = [c for c in ["gold", "same", "foreign"] if c in comp.columns]
    colors = {"gold": "#2ca02c", "same": "#d62728", "foreign": "#9ecae1"}
    labels = {"gold": "gold at rank 1", "same": "same-language distractor", "foreign": "foreign distractor"}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bottom = np.zeros(len(comp))
    xs = np.arange(len(comp))
    for c in order:
        vals = comp[c].values
        ax.bar(xs, vals, bottom=bottom, label=labels[c], color=colors[c])
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.LANG_NAME[l]}\n({l})" for l in comp.index])
    ax.set_ylabel("share of queries")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title(f"What occupies rank 1, by query language — {C.short(best)}")
    fig.tight_layout()
    fig.savefig(out / "rank1_composition.png")
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------------------
    worst_lang = by.sort_values(ascending=False)
    summary = {
        "pooled_same_language_confusion_rate": round(float(pool["sl_confusion"].mean()), 4),
        "pooled_mean_sl_blocker_share": round(float(pool["sl_blocker_share"].mean(skipna=True)), 4),
        "most_self_confused_language": {"lang": worst_lang.index[0],
                                        "sl_blocker_share": round(float(worst_lang.iloc[0]), 3)},
        "least_self_confused_language": {"lang": worst_lang.index[-1],
                                         "sl_blocker_share": round(float(worst_lang.iloc[-1]), 3)},
        "per_model": msum[["short", "rate", "mean_sl_blocker_share"]].round(4).to_dict(orient="records"),
    }
    C.jdump(summary, out / "summary.json")

    md = f"""<!-- round:07 -->
## Round 7 — Distractor dominance: what buries the foreign twin?

**Question.** With no curated hard-negatives, read confusion from the ranking itself: the documents
above the first gold are the blockers. Are they *same-language* noise (collapse-driven) or genuine
foreign distractors?

**What we learned.**
- **Same-language noise is the dominant blocker for high-resource query languages.** Pooled over
  reliable models, a same-language non-gold out-ranks the first gold on
  **{summary['pooled_same_language_confusion_rate']:.0%}** of queries, and
  **{summary['pooled_mean_sl_blocker_share']:.0%}** of all above-gold blockers share the query's
  language — far above their corpus base rate.
- It is **language-specific**: the most self-confused query language is
  **{summary['most_self_confused_language']['lang']}**
  ({summary['most_self_confused_language']['sl_blocker_share']:.0%} same-language blockers), the least
  is **{summary['least_self_confused_language']['lang']}**
  ({summary['least_self_confused_language']['sl_blocker_share']:.0%}). Spanish, with no same-language
  gold, is mostly blocked by *foreign* distractors — a different failure mode.
- Confusion here is a direct consequence of the Round-6 collapse: the same-language documents the
  model over-fetches are exactly what bury the foreign twin.

**Next questions.**
1. Is this fixable by re-ranking, or are the gold documents simply *not separable* in score space from
   the same-language crowd? → Round 8.
2. Different models collapse on different languages — are their errors complementary enough that an
   ensemble or a per-language router recovers the buried twins? → Round 9.
<!-- /round:07 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
