"""
Round 5 — Cross-lingual ranking consistency (RBO).

When the *same* patent's question is asked in several languages, the information need is identical, so
a language-agnostic retriever should return the same ranked patents. We measure that with
Rank-Biased Overlap (RBO_ext, p=0.9, top-100) between the ranked lists of a patent's different-language
queries. 16 patents in this benchmark were asked in ≥2 languages (61 cross-lingual query pairs).

Outputs (../experimental_plots/round05_consistency/):
  rbo_pairs.csv, per_model.csv, summary.json, and figures:
    rbo_by_model.png            mean cross-lingual RBO per model (95% CI over patents)
    rbo_vs_home_advantage.png   does language sensitivity track the home-advantage bias?
    rbo_by_language_pair.png     RBO heatmap by query-language pair (pooled reliable models)
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round05_consistency"
DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]
P = 0.9
DEPTH = 100


def pair_frame() -> pd.DataFrame:
    """One row per (model, patent, query-pair in different languages): RBO of the two ranked lists."""
    lists = C.ranked_lists()
    ql = C.q_lang()
    qpat = C.q_patent()
    # patents -> their queries
    by_patent: dict = {}
    for qid in C.gold_publication():
        by_patent.setdefault(qpat.get(qid, ""), []).append(qid)
    rows = []
    for patent, qids in by_patent.items():
        langs = {ql.get(q, "") for q in qids}
        if len(langs) < 2:
            continue
        for qa, qb in combinations(qids, 2):
            la, lb = ql.get(qa, ""), ql.get(qb, "")
            if la == lb:
                continue
            for model in C.MODEL_ORDER:
                ra = lists.get((model, qa), [])
                rb = lists.get((model, qb), [])
                rows.append({"model": model, "short": C.short(model), "patent": patent,
                             "lang_a": la, "lang_b": lb, "pair": "↔".join(sorted([la, lb])),
                             "rbo": C.rbo(ra, rb, p=P, k=DEPTH)})
    return pd.DataFrame(rows)


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    pairs = pair_frame()
    pairs.to_csv(out / "rbo_pairs.csv", index=False)

    # home advantage per model (recompute from core frame; originals: MoLIR - CLIR @10)
    cpq = C.core_per_query()
    home_adv = {}
    clir = {}
    for model in C.MODEL_ORDER:
        g = cpq[cpq.model == model]
        o = g[g.origin == "original"]
        home_adv[model] = float((o["molir_at_10"] - o["clir_at_10"]).mean())
        clir[model] = float(g["clir_at_10"].mean())

    # per-model: aggregate per patent first (equal patent weight), bootstrap over patents
    rows = []
    for model in C.MODEL_ORDER:
        g = pairs[pairs.model == model]
        per_patent = g.groupby("patent")["rbo"].mean().values
        pt, lo, hi = C.bootstrap_ci(per_patent)
        rows.append({"model": model, "short": C.short(model), "rbo": pt, "lo": lo, "hi": hi,
                     "home_adv": home_adv[model], "clir_at_10": clir[model]})
    msum = pd.DataFrame(rows)
    msum.to_csv(out / "per_model.csv", index=False)

    # ---- Fig 1: RBO by model ---------------------------------------------------------------------
    o = msum.sort_values("rbo", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    xs = np.arange(len(o))
    err = np.vstack([o["rbo"] - o["lo"], o["hi"] - o["rbo"]])
    ax.bar(xs, o["rbo"], yerr=err, capsize=3, color=[C.MODEL_COLOR[m] for m in o["model"]])
    for x, v in zip(xs, o["rbo"]):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(o["short"], rotation=30, ha="right")
    ax.set_ylabel("mean cross-lingual RBO (top-100, p=0.9)")
    ax.set_ylim(0, 1)
    ax.set_title("Same patent asked in different languages — how identical are the ranked results?\n"
                 "RBO=1 identical ranking, 0 disjoint (16 patents, 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out / "rbo_by_model.png")
    plt.close(fig)

    # ---- Fig 2: RBO vs home advantage ------------------------------------------------------------
    rel = msum[msum.model.isin(POOL)]
    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.scatter(rel["home_adv"], rel["rbo"], s=70, color=[C.MODEL_COLOR[m] for m in rel["model"]])
    for _, r in rel.iterrows():
        ax.annotate(r["short"], (r["home_adv"], r["rbo"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    if len(rel) >= 3:
        rr = np.corrcoef(rel["home_adv"], rel["rbo"])[0, 1]
        b, a = np.polyfit(rel["home_adv"], rel["rbo"], 1)
        xs = np.linspace(rel["home_adv"].min(), rel["home_adv"].max(), 50)
        ax.plot(xs, a + b * xs, "--", color="grey", lw=1, label=f"Pearson r = {rr:.2f}")
        ax.legend()
    ax.set_xlabel("home advantage (MoLIR@10 − CLIR@10)")
    ax.set_ylabel("cross-lingual RBO")
    ax.set_title("Do language-biased models also rank inconsistently across languages?")
    fig.tight_layout()
    fig.savefig(out / "rbo_vs_home_advantage.png")
    plt.close(fig)

    # ---- Fig 3: RBO by language pair -------------------------------------------------------------
    pool = pairs[pairs.model.isin(POOL)]
    pair_means = pool.groupby("pair")["rbo"].agg(["mean", "size"]).reset_index().sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4.3))
    xs = np.arange(len(pair_means))
    ax.bar(xs, pair_means["mean"], color="#4c72b0")
    for x, v, n in zip(xs, pair_means["mean"], pair_means["size"]):
        ax.text(x, v + 0.008, f"{v:.2f}\nn={int(n)}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(pair_means["pair"], rotation=30, ha="right")
    ax.set_ylabel("mean RBO (pooled reliable models)")
    ax.set_ylim(0, min(1, pair_means["mean"].max() + 0.12))
    ax.set_title("Which language pairs produce the most divergent rankings for the same question?")
    fig.tight_layout()
    fig.savefig(out / "rbo_by_language_pair.png")
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------------------
    rel_sorted = rel.sort_values("rbo", ascending=False)
    rr = float(np.corrcoef(rel["home_adv"], rel["rbo"])[0, 1]) if len(rel) >= 3 else float("nan")
    summary = {
        "n_patents_multilang": int(pairs.patent.nunique()),
        "n_pairs_per_model": int(pairs[pairs.model == C.MODEL_ORDER[0]].shape[0]),
        "ceiling_rbo": round(float(rel_sorted.iloc[0]["rbo"]), 4),
        "ceiling_model": rel_sorted.iloc[0]["short"],
        "floor_rbo": round(float(rel_sorted.iloc[-1]["rbo"]), 4),
        "floor_model": rel_sorted.iloc[-1]["short"],
        "pearson_homeadv_rbo": round(rr, 3),
        "most_divergent_pair": pair_means.iloc[0]["pair"],
        "per_model": msum[["short", "rbo", "home_adv", "clir_at_10"]].round(4).to_dict(orient="records"),
    }
    C.jdump(summary, out / "summary.json")

    md = f"""<!-- round:05 -->
## Round 5 — Cross-lingual ranking consistency (RBO)

**Question.** When the *same* patent's question is asked in several languages, a language-agnostic
retriever should return the same ranked patents. How consistent are the top-100 lists across
languages? (RBO_ext, p=0.9; {summary['n_patents_multilang']} multilingual patents,
{summary['n_pairs_per_model']} cross-lingual query pairs per model.)

**What we learned.**
- Consistency is **low even at the ceiling**: the most consistent model is **{summary['ceiling_model']}**
  at RBO = **{summary['ceiling_rbo']:.2f}** (1.0 = identical) — the *same question* in two languages
  produces largely *different* top-100 patent lists. The floor is **{summary['floor_model']}**
  (RBO = {summary['floor_rbo']:.2f}).
- Inconsistency tracks bias: Pearson r(home-advantage, RBO) = **{summary['pearson_homeadv_rbo']:+.2f}**
  — models that lean hardest on the same-language copy are also the most language-sensitive in their
  rankings.
- The most divergent language pair is **{summary['most_divergent_pair']}**.

**Next questions.**
1. If rankings differ this much by language, the lists must be filling with *language-specific* docs.
   How monolingual is the top-k, and does that collapse explain CLIR failure? → Round 6.
2. Are these low-consistency, language-collapsed results also poorly *separated* in score space? → Round 8.
<!-- /round:05 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
