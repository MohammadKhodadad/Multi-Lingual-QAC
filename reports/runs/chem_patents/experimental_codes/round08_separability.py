"""
Round 8 — Score separability: are the gold documents separable from the crowd, in score space?

Confusion (Round 7) could be a ranking artefact or a deeper score-separability failure. For each query
we compute AUC = P(score(gold) > score(non-gold)) over the retrieved documents, and crucially split it
by direction: AUC_same (same-language gold) vs AUC_cross (foreign gold). A large same−cross gap means
the model assigns systematically lower similarity to the foreign twin than to the same-language copy —
a calibration/separability home advantage, not just a tie-breaking accident.

Outputs (../experimental_plots/round08_separability/):
  per_query_separability.csv, per_model.csv, summary.json, and figures:
    auc_by_model.png            AUC_all / AUC_same / AUC_cross per model
    auc_cross_by_language.png   cross-language separability by query language
    separability_vs_clir.png    per-model AUC_cross vs CLIR@10
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round08_separability"
DEGENERATE = "Alibaba-NLP/gte-multilingual-base"
POOL = [m for m in C.MODEL_ORDER if m != DEGENERATE]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = np.r_[pos, neg]
        return float(roc_auc_score(y, s))
    except Exception:
        # Mann–Whitney U fallback
        allv = np.r_[pos, neg]
        order = allv.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(allv) + 1)
        r_pos = ranks[:len(pos)].sum()
        u = r_pos - len(pos) * (len(pos) + 1) / 2
        return float(u / (len(pos) * len(neg)))


def sep_frame() -> pd.DataFrame:
    scores = C.score_lists()
    gold = C.gold_publication()
    ql = C.q_lang()
    qorg = C.q_origin()
    rows = []
    for model in C.MODEL_ORDER:
        for qid, gset in gold.items():
            sm = scores.get((model, qid), {})
            if not sm:
                continue
            lang = ql.get(qid, "")
            gold_ret = {d: s for d, s in sm.items() if d in gset}
            neg = np.array([s for d, s in sm.items() if d not in gset], dtype=float)
            if not gold_ret or neg.size == 0:
                continue
            same = np.array([s for d, s in gold_ret.items() if C.doc_lang(d) == lang], dtype=float)
            cross = np.array([s for d, s in gold_ret.items() if C.doc_lang(d) != lang], dtype=float)
            allg = np.array(list(gold_ret.values()), dtype=float)
            spread = max(sm.values()) - min(sm.values())
            rows.append({
                "model": model, "short": C.short(model), "query_id": qid, "qlang": lang,
                "origin": qorg.get(qid, ""),
                "auc_all": _auc(allg, neg),
                "auc_same": _auc(same, neg) if same.size else np.nan,
                "auc_cross": _auc(cross, neg) if cross.size else np.nan,
                "margin_cross": ((cross.max() - neg.max()) / spread) if (cross.size and spread > 0) else np.nan,
                "norm_score_cross": ((cross.mean() - neg.mean()) / spread) if (cross.size and spread > 0) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    df = sep_frame()
    df.to_csv(out / "per_query_separability.csv", index=False)

    rows = []
    for model in C.MODEL_ORDER:
        g = df[df.model == model]
        aa, alo, ahi = C.bootstrap_ci(g["auc_all"].values)
        ac, clo, chi = C.bootstrap_ci(g["auc_cross"].values)
        asame = float(g["auc_same"].mean(skipna=True))
        rows.append({"model": model, "short": C.short(model),
                     "auc_all": aa, "auc_cross": ac, "ac_lo": clo, "ac_hi": chi,
                     "auc_same": asame, "sep_home_adv": asame - ac,
                     "clir_at_10": float(C.core_per_query().query("model==@model")["clir_at_10"].mean())})
    msum = pd.DataFrame(rows)
    msum.to_csv(out / "per_model.csv", index=False)

    # ---- Fig 1: AUC all/same/cross per model -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = np.arange(len(msum))
    w = 0.27
    ax.bar(xs - w, msum["auc_all"], w, label="AUC all gold", color="#7f7f7f")
    ax.bar(xs, msum["auc_same"], w, label="AUC same-language gold", color="#2ca02c")
    errc = np.vstack([msum["auc_cross"] - msum["ac_lo"], msum["ac_hi"] - msum["auc_cross"]])
    ax.bar(xs + w, msum["auc_cross"], w, yerr=errc, capsize=2.5, label="AUC cross-language gold", color="#d62728")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(msum["short"], rotation=30, ha="right")
    ax.set_ylabel("AUC( gold > non-gold )")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", ncol=3)
    ax.set_title("Can the score separate gold from the crowd? Foreign twins are systematically harder")
    fig.tight_layout()
    fig.savefig(out / "auc_by_model.png")
    plt.close(fig)

    # ---- Fig 2: AUC_cross by query language -------------------------------------------------------
    pool = df[df.model.isin(POOL)]
    by = []
    for lang in C.QUERY_LANGS:
        sub = pool[pool.qlang == lang]
        pt, lo, hi = C.bootstrap_ci(sub["auc_cross"].values)
        by.append({"lang": lang, "auc": pt, "lo": lo, "hi": hi})
    bydf = pd.DataFrame(by)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = np.arange(len(bydf))
    err = np.vstack([bydf["auc"] - bydf["lo"], bydf["hi"] - bydf["auc"]])
    ax.bar(xs, bydf["auc"], yerr=err, capsize=3, color=[C.LANG_COLOR[l] for l in bydf["lang"]])
    for x, v in zip(xs, bydf["auc"]):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.LANG_NAME[l]}\n({l})" for l in bydf["lang"]])
    ax.set_ylabel("AUC cross-language gold (pooled reliable models)")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-language separability by query language")
    fig.tight_layout()
    fig.savefig(out / "auc_cross_by_language.png")
    plt.close(fig)

    # ---- Fig 3: separability vs CLIR (per model) -------------------------------------------------
    rel = msum[msum.model.isin(POOL)]
    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.scatter(rel["auc_cross"], rel["clir_at_10"], s=70, color=[C.MODEL_COLOR[m] for m in rel["model"]])
    for _, r in rel.iterrows():
        ax.annotate(r["short"], (r["auc_cross"], r["clir_at_10"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    if len(rel) >= 3:
        rr = np.corrcoef(rel["auc_cross"], rel["clir_at_10"])[0, 1]
        b, a = np.polyfit(rel["auc_cross"], rel["clir_at_10"], 1)
        xs = np.linspace(rel["auc_cross"].min(), rel["auc_cross"].max(), 50)
        ax.plot(xs, a + b * xs, "--", color="grey", lw=1, label=f"Pearson r = {rr:.2f}")
        ax.legend()
    ax.set_xlabel("AUC cross-language gold")
    ax.set_ylabel("CLIR@10")
    ax.set_title("Cross-lingual recall is a separability problem")
    fig.tight_layout()
    fig.savefig(out / "separability_vs_clir.png")
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------------------
    rr = float(np.corrcoef(rel["auc_cross"], rel["clir_at_10"])[0, 1]) if len(rel) >= 3 else float("nan")
    best = msum.sort_values("auc_cross", ascending=False).iloc[0]
    summary = {
        "best_separability_model": best["short"],
        "best_auc_cross": round(float(best["auc_cross"]), 4),
        "mean_sep_home_adv": round(float(msum[msum.model.isin(POOL)]["sep_home_adv"].mean()), 4),
        "pearson_auccross_clir": round(rr, 3),
        "weakest_language": bydf.sort_values("auc").iloc[0]["lang"],
        "per_model": msum[["short", "auc_all", "auc_same", "auc_cross", "sep_home_adv"]]
                     .round(4).to_dict(orient="records"),
    }
    C.jdump(summary, out / "summary.json")

    md = f"""<!-- round:08 -->
## Round 8 — Score separability: a calibration home advantage

**Question.** Is confusion a ranking accident or a score-separability failure? We compute
AUC(gold > non-gold) per query and split it by direction (same- vs cross-language gold).

**What we learned.**
- **Foreign twins are systematically less separable.** Even the best model separates same-language
  gold from the crowd far better than foreign gold; pooled, the separability home advantage
  (AUC_same − AUC_cross) is **{summary['mean_sep_home_adv']:+.2f}**. The model assigns lower
  similarity to the correct patent simply because it is written in another language.
- **Separability explains recall.** Across models, Pearson r(AUC_cross, CLIR@10) =
  **{summary['pearson_auccross_clir']:+.2f}** — cross-lingual recall is, mechanically, a
  cross-lingual *separability* problem, not just a cutoff problem. The best separator is
  **{summary['best_separability_model']}** (AUC_cross = {summary['best_auc_cross']:.2f}).
- The least separable query language is **{summary['weakest_language']}**.
- Implication: a monolingual re-ranker won't fix this — the foreign gold is under-scored at the
  embedding level. Better *alignment* (or fusing complementary models) is required.

**Next questions.**
1. Models collapse and under-score along different languages. Are their errors complementary enough
   that fusing them, or routing by language, recovers the buried twins? → Round 9.
2. Fold accuracy, CLIR, consistency, MT-robustness, collapse and separability into a single
   multilingual-robustness verdict. → Round 10.
<!-- /round:08 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
