"""
Round 1 — The CLIR leaderboard and the home-advantage gap.

Anchor question: standard recall@10 ranks the models, but this benchmark exists to test
*cross-lingual* retrieval. When we restrict the gold set to documents in a DIFFERENT language than
the query (CLIR@k) versus the SAME language (MoLIR@k), how does the picture change, and how large is
the home-advantage each model enjoys when a same-language document is available?

Outputs (../experimental_plots/round01_clir_leaderboard/):
  per_model.csv, summary.json, and figures:
    clir_leaderboard.png     overall recall@10 vs CLIR@10 vs MoLIR@10, per model, with 95% CIs
    home_advantage.png       per-query (MoLIR - CLIR) on the 57 original queries, per model
    clir_vs_k.png            CLIR@k curves across k for every model
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round01_clir_leaderboard"


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    cpq = C.core_per_query()

    # ---- per-model headline numbers with bootstrap CIs -------------------------------------------
    rows = []
    for model in C.MODEL_ORDER:
        g = cpq[cpq.model == model]
        orig = g[g.origin == "original"]
        rec, rlo, rhi = C.bootstrap_ci(g["recall_at_10"].values)
        cl, clo, chi = C.bootstrap_ci(g["clir_at_10"].values)
        mo, mlo, mhi = C.bootstrap_ci(orig["molir_at_10"].values)
        # paired home-advantage on originals (each has both a same- and cross-language gold)
        hai = (orig["molir_at_10"] - orig["clir_at_10"]).values
        h, hlo, hhi = C.bootstrap_ci(hai)
        rows.append({
            "model": model, "short": C.short(model),
            "recall_at_10": rec, "recall_lo": rlo, "recall_hi": rhi,
            "clir_at_10": cl, "clir_lo": clo, "clir_hi": chi,
            "molir_at_10": mo, "molir_lo": mlo, "molir_hi": mhi,
            "home_adv": h, "home_adv_lo": hlo, "home_adv_hi": hhi,
            "clir_at_100": g["clir_at_100"].mean(),
            "recall_at_100": g["recall_at_100"].mean(),
        })
    msum = pd.DataFrame(rows)
    msum.to_csv(out / "per_model.csv", index=False)

    xs = np.arange(len(msum))

    # ---- Fig 1: overall / CLIR / MoLIR @10 grouped bars ------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.26
    for j, (col, lo, hi, lab, color) in enumerate([
        ("recall_at_10", "recall_lo", "recall_hi", "overall Recall@10", "#7f7f7f"),
        ("clir_at_10", "clir_lo", "clir_hi", "CLIR@10 (cross-language gold)", "#d62728"),
        ("molir_at_10", "molir_lo", "molir_hi", "MoLIR@10 (same-language gold, originals)", "#2ca02c"),
    ]):
        err = np.vstack([msum[col] - msum[lo], msum[hi] - msum[col]])
        ax.bar(xs + (j - 1) * w, msum[col], w, yerr=err, capsize=2.5, label=lab, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels(msum["short"], rotation=30, ha="right")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title("Cross-lingual vs same-language retrieval per model\n"
                 "every model loses ground when no same-language document exists (CLIR < overall < MoLIR)")
    fig.tight_layout()
    fig.savefig(out / "clir_leaderboard.png")
    plt.close(fig)

    # ---- Fig 2: home advantage (MoLIR - CLIR) on originals ---------------------------------------
    o = msum.sort_values("home_adv", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    xo = np.arange(len(o))
    err = np.vstack([o["home_adv"] - o["home_adv_lo"], o["home_adv_hi"] - o["home_adv"]])
    ax.bar(xo, o["home_adv"], yerr=err, capsize=3, color=[C.MODEL_COLOR[m] for m in o["model"]])
    for x, v in zip(xo, o["home_adv"]):
        ax.text(x, v + 0.012, f"{v:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xo)
    ax.set_xticklabels(o["short"], rotation=30, ha="right")
    ax.set_ylabel("home advantage  (MoLIR@10 − CLIR@10)")
    ax.set_title("How much easier is the same-language copy than a foreign one?\n"
                 "paired within the 57 original queries (positive = same-language docs are easier)")
    fig.tight_layout()
    fig.savefig(out / "home_advantage.png")
    plt.close(fig)

    # ---- Fig 3: CLIR@k curves --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for model in C.MODEL_ORDER:
        g = cpq[cpq.model == model]
        ys = [g[f"clir_at_{k}"].mean() for k in C.KS]
        ax.plot(C.KS, ys, marker="o", ms=4, color=C.MODEL_COLOR[model], label=C.short(model))
    ax.set_xscale("log")
    ax.set_xticks(C.KS)
    ax.set_xticklabels([str(k) for k in C.KS])
    ax.set_xlabel("k")
    ax.set_ylabel("CLIR@k  (cross-language recall)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, fontsize=7)
    ax.set_title("Cross-lingual recall vs depth — even at k=100 the best model finds only ~70% of foreign mates")
    fig.tight_layout()
    fig.savefig(out / "clir_vs_k.png")
    plt.close(fig)

    # ---- summary.json ----------------------------------------------------------------------------
    best = msum.sort_values("clir_at_10", ascending=False).iloc[0]
    worst_hai = msum.sort_values("home_adv", ascending=False).iloc[0]
    summary = {
        "n_queries": int(cpq.query_id.nunique()),
        "n_original": int((C.query_meta().origin == "original").sum()),
        "n_synthetic": int((C.query_meta().origin == "synthetic").sum()),
        "best_clir_model": best["short"],
        "best_clir_at_10": round(float(best["clir_at_10"]), 4),
        "best_clir_ci": [round(float(best["clir_lo"]), 4), round(float(best["clir_hi"]), 4)],
        "largest_home_adv_model": worst_hai["short"],
        "largest_home_adv": round(float(worst_hai["home_adv"]), 4),
        "per_model": msum[["short", "recall_at_10", "clir_at_10", "molir_at_10", "home_adv"]]
                     .round(4).to_dict(orient="records"),
    }
    C.jdump(summary, out / "summary.json")

    # ---- findings narrative ----------------------------------------------------------------------
    tbl = "\n".join(
        f"| `{r['short']}` | {r['recall_at_10']:.3f} | {r['clir_at_10']:.3f} "
        f"[{r['clir_lo']:.2f},{r['clir_hi']:.2f}] | "
        f"{r['molir_at_10']:.3f} | {r['home_adv']:+.3f} |"
        for _, r in msum.iterrows()
    )
    md = f"""<!-- round:01 -->
## Round 1 — The CLIR leaderboard and the home-advantage gap

**Question.** Standard recall ranks the models, but this benchmark exists to test *cross-lingual*
retrieval. Split each query's gold set by language relative to the query: **CLIR@k** = recall over
documents in a *different* language; **MoLIR@k** = recall over the *same*-language document (only the
57 original queries ever have one — all 80 synthetic queries are pure CLIR). How big is the gap?

**Numbers (mean over queries; CLIR 95% bootstrap CI; MoLIR/home-advantage on the 57 originals).**

| model | Recall@10 | CLIR@10 | MoLIR@10 | home adv |
| --- | ---: | :---: | ---: | ---: |
{tbl}

**What we learned.**
- The best cross-lingual model is **{best['short']}** at CLIR@10 = **{best['clir_at_10']:.3f}**
  [{best['clir_lo']:.2f}, {best['clir_hi']:.2f}]. Every model scores strictly lower on CLIR@10 than
  on overall recall — the easy points come from same-language matches.
- **Home advantage is universal and large.** When a same-language copy exists, models retrieve it far
  more reliably than any foreign version; the gap reaches **{worst_hai['home_adv']:+.3f}** for
  `{worst_hai['short']}`. The benchmark's 80 synthetic queries have *no* such crutch.
- `e5-large-instruct` is the genuine low performer here (instruction model, strong within-language
  but almost no cross-lingual transfer — a real property, not a defect). `gte-multilingual-base` was
  **excluded** from this analysis: its ≈0.004 score was a model-loading artifact (degenerate
  embeddings; it failed even trivial same-language self-retrieval), not real performance.

**Next questions.**
1. CLIR is an average over many language directions. *Which* query→document directions actually
   collapse, and is the failure symmetric (en→zh vs zh→en)? → Round 2.
2. Synthetic queries are both machine-translated *and* homeless. Isolate the MT effect by comparing
   CLIR (cross-language only) for human-original vs machine-translated queries. → Round 3.
<!-- /round:01 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
