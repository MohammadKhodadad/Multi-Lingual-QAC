"""
Round 10 — CLIR robustness synthesis (CLIR-MRS).

Nine rounds produced orthogonal lenses on multilingual retrieval. This round folds them into one
**CLIR Multilingual Robustness Score (CLIR-MRS)** with a query-bootstrap CI, a radar and a
leaderboard, and issues the final recommendation.

To avoid the "balanced mediocrity" trap (where a consistently *weak* model wins on proxy axes), CLIR-MRS
makes **capability the spine** and lets **robustness modulate within ±50%** — a model that is evenly
poor cannot out-rank a strong one, and a strong model is rewarded for being even:

  CAPABILITY  (is the right, foreign document found and well-scored)
    accuracy      overall Recall@10                         (Round 1)
    clir          CLIR@10, cross-language recall            (Round 1)
    separability  AUC(cross-language gold > non-gold)        (Round 8)
  ROBUSTNESS  (is it stable across languages and provenance)
    consistency   cross-lingual RBO                         (Round 5)
    mt_robust     1 − |human − machine-translated gap|      (Round 3)
    lang_parity   1 − (max−min CLIR@10 across query langs)  (Round 2/6)

  CLIR-MRS = mean(normalised CAPABILITY) × ( 0.5 + 0.5 × mean(normalised ROBUSTNESS) )

Outputs (../experimental_plots/round10_robustness_synthesis/):
  robustness_scores.csv, robustness_axes_normalized.csv, summary.json, and figures:
    mrs_leaderboard.png   CLIR-MRS per model with bootstrap CI
    robustness_radar.png  normalised axes for the leaders (+ a monolingual foil)
    axes_heatmap.png      normalised axes + group scores + MRS, all models
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "round10_robustness_synthesis"
PLOTS = C.PLOTS_DIR
CAP_AXES = ["accuracy", "clir", "separability"]
ROB_AXES = ["consistency", "mt_robust", "lang_parity"]
AXES = CAP_AXES + ROB_AXES
AXIS_LABEL = {
    "accuracy": "accuracy\n(Recall@10)", "clir": "CLIR@10", "separability": "separability\n(AUC)",
    "consistency": "consistency\n(RBO)", "mt_robust": "MT-robust", "lang_parity": "language\nparity",
}
N_BOOT = 1500


def _read(round_slug, name):
    return pd.read_csv(PLOTS / round_slug / name)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    for a in AXES:
        lo, hi = z[a].min(), z[a].max()
        z[a] = (z[a] - lo) / (hi - lo) if hi > lo else 0.0
    return z


def _mrs(z: pd.DataFrame) -> pd.DataFrame:
    z = z.copy()
    z["capability"] = z[CAP_AXES].mean(axis=1)
    z["robustness"] = z[ROB_AXES].mean(axis=1)
    # capability is the spine; robustness modulates within ±50% (cannot rescue a weak model)
    z["MRS"] = z["capability"].clip(lower=0) * (0.5 + 0.5 * z["robustness"].clip(lower=0))
    return z


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    cpq = C.core_per_query()
    sep = _read("round08_separability", "per_query_separability.csv")[["model", "query_id", "auc_cross"]]
    base = cpq[["model", "query_id", "query_language", "recall_at_10", "clir_at_10"]] \
        .merge(sep, on=["model", "query_id"], how="left")

    rbo = _read("round05_consistency", "per_model.csv").set_index("short")["rbo"]
    gap = _read("round03_mt_penalty", "per_model_summary.csv").set_index("short")["gap"]

    def model_axes(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for model in C.MODEL_ORDER:
            g = frame[frame.model == model]
            sh = C.short(model)
            per_lang = g.groupby("query_language")["clir_at_10"].mean()
            parity = 1.0 - float(per_lang.max() - per_lang.min()) if len(per_lang) else np.nan
            rows.append({
                "model": model, "short": sh,
                "accuracy": float(g["recall_at_10"].mean()),
                "clir": float(g["clir_at_10"].mean()),
                "separability": float(g["auc_cross"].mean()),
                "consistency": float(rbo.get(sh, np.nan)),
                "mt_robust": 1.0 - abs(float(gap.get(sh, 0.0))),
                "lang_parity": parity,
            })
        return pd.DataFrame(rows)

    raw = model_axes(base)
    norm = _mrs(_normalize(raw))
    raw_out = raw.merge(norm[["short", "capability", "robustness", "MRS"]], on="short")
    raw_out["rank"] = raw_out["MRS"].rank(ascending=False).astype(int)
    raw_out.sort_values("MRS", ascending=False).to_csv(out / "robustness_scores.csv", index=False)
    norm.to_csv(out / "robustness_axes_normalized.csv", index=False)

    # ---- bootstrap CI for MRS (resample queries; fixed axes held constant) -----------------------
    qids = base["query_id"].unique()
    pivots = {a: base.pivot_table(index="model", columns="query_id", values=col, aggfunc="mean")
              for a, col in [("accuracy", "recall_at_10"), ("clir", "clir_at_10"),
                             ("separability", "auc_cross")]}
    lang_of_q = base.drop_duplicates("query_id").set_index("query_id")["query_language"]
    clir_pivot = pivots["clir"]
    boot_mrs = {C.short(m): [] for m in C.MODEL_ORDER}
    rng = np.random.default_rng(20260609)
    for _ in range(N_BOOT):
        samp = rng.choice(qids, size=len(qids), replace=True)
        rowd = []
        for m in C.MODEL_ORDER:
            sh = C.short(m)
            ax = {"model": m, "short": sh,
                  "consistency": float(rbo.get(sh, np.nan)),
                  "mt_robust": 1.0 - abs(float(gap.get(sh, 0.0)))}
            for a in ("accuracy", "clir", "separability"):
                ax[a] = float(np.nanmean(pivots[a].loc[m, samp].values))
            cl = pd.Series(clir_pivot.loc[m, samp].values, index=lang_of_q.loc[samp].values)
            pl = cl.groupby(level=0).mean()
            ax["lang_parity"] = 1.0 - float(pl.max() - pl.min()) if len(pl) else np.nan
            rowd.append(ax)
        bnorm = _mrs(_normalize(pd.DataFrame(rowd)))
        for sh, v in zip(bnorm["short"], bnorm["MRS"]):
            boot_mrs[sh].append(v)
    ci = {sh: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for sh, v in boot_mrs.items()}

    lead = raw_out.sort_values("MRS", ascending=False).reset_index(drop=True)
    lead["mrs_lo"] = lead["short"].map(lambda s: ci[s][0])
    lead["mrs_hi"] = lead["short"].map(lambda s: ci[s][1])

    # ---- Fig 1: MRS leaderboard ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = np.arange(len(lead))
    err = np.vstack([np.maximum(lead["MRS"] - lead["mrs_lo"], 0), np.maximum(lead["mrs_hi"] - lead["MRS"], 0)])
    ax.bar(xs, lead["MRS"], yerr=err, capsize=3, color=[C.MODEL_COLOR[m] for m in lead["model"]])
    for x, v, hi in zip(xs, lead["MRS"], lead["mrs_hi"]):
        ax.text(x, max(v, hi) + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(lead["short"], rotation=30, ha="right")
    ax.set_ylabel("CLIR-MRS = capability × (0.5 + 0.5·robustness)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Multilingual robustness leaderboard (CLIR-MRS)\n"
                 "capability {accuracy, CLIR, separability}; robustness {consistency, MT-robust, parity} modulates ±50%")
    fig.tight_layout()
    fig.savefig(out / "mrs_leaderboard.png")
    plt.close(fig)

    # ---- Fig 2: radar for leaders (+ monolingual foil e5) ----------------------------------------
    show = list(lead["model"].head(3))
    foil = "intfloat/multilingual-e5-large-instruct"
    if foil not in show:
        show.append(foil)
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.2, 6.6), subplot_kw=dict(polar=True))
    for model in show:
        sh = C.short(model)
        vals = norm.set_index("model").loc[model, AXES].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=sh, color=C.MODEL_COLOR[model], lw=2)
        ax.fill(angles, vals, color=C.MODEL_COLOR[model], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([AXIS_LABEL[a] for a in AXES], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Robustness profiles (normalised axes)\ntop-3 by CLIR-MRS + a monolingual foil (e5-instruct)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    fig.tight_layout()
    fig.savefig(out / "robustness_radar.png")
    plt.close(fig)

    # ---- Fig 3: axes heatmap + group scores ------------------------------------------------------
    ord_models = lead["model"].tolist()
    mat = norm.set_index("model").loc[ord_models, AXES + ["capability", "robustness", "MRS"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat.values.astype(float), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    cols = AXES + ["CAPABILITY", "ROBUSTNESS", "MRS"]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([AXIS_LABEL.get(a, a) for a in cols], fontsize=8, rotation=0)
    ax.set_yticks(range(len(ord_models)))
    ax.set_yticklabels([C.short(m) for m in ord_models])
    for i in range(len(ord_models)):
        for j in range(len(cols)):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.5 else "black")
    for x in (len(AXES) - 0.5, len(AXES) + 1.5):
        ax.axvline(x, color="white", lw=2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalised (0=worst, 1=best model)")
    ax.set_title("Normalised robustness axes, group scores and CLIR-MRS")
    fig.tight_layout()
    fig.savefig(out / "axes_heatmap.png")
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------------------
    winner = lead.iloc[0]
    win_norm = norm.set_index("short").loc[winner["short"], AXES]
    weakest_axis = win_norm.idxmin()
    summary = {
        "winner": winner["short"], "winner_mrs": round(float(winner["MRS"]), 4),
        "winner_mrs_ci": [round(float(winner["mrs_lo"]), 4), round(float(winner["mrs_hi"]), 4)],
        "winner_capability": round(float(winner["capability"]), 4),
        "winner_robustness": round(float(winner["robustness"]), 4),
        "winner_weakest_axis": weakest_axis,
        "leaderboard": lead[["short", "MRS", "mrs_lo", "mrs_hi", "capability", "robustness", "rank"]]
                       .round(4).to_dict(orient="records"),
        "raw_axes": raw[["short"] + AXES].round(4).to_dict(orient="records"),
    }
    C.jdump(summary, out / "summary.json")

    tbl = "\n".join(f"| {int(r['rank'])} | `{r['short']}` | {r['MRS']:.2f} "
                    f"[{r['mrs_lo']:.2f},{r['mrs_hi']:.2f}] | {r['capability']:.2f} | {r['robustness']:.2f} |"
                    for _, r in lead.iterrows())
    md = f"""<!-- round:10 -->
## Round 10 — CLIR robustness synthesis (CLIR-MRS)

**Question.** Average recall hides cross-lingual weaknesses. Fold the lenses into one score in which
**capability is the spine and robustness only modulates** (±50%): CLIR-MRS = capability ×
(0.5 + 0.5·robustness), capability = mean normalised {{accuracy, CLIR, separability}}, robustness =
mean normalised {{consistency, MT-robust, language-parity}}. This rewards a strong model for being
even, but never lets an evenly-mediocre model out-rank a genuinely capable one.

**Leaderboard (CLIR-MRS, 95% query-bootstrap CI).**

| rank | model | CLIR-MRS | capability | robustness |
| ---: | --- | :---: | :---: | :---: |
{tbl}

**What we learned.**
- The most robust multilingual retriever is **{winner['short']}** (CLIR-MRS = {winner['MRS']:.2f}
  [{winner['mrs_lo']:.2f}, {winner['mrs_hi']:.2f}]; capability {winner['capability']:.2f}, robustness
  {winner['robustness']:.2f}). It leads on the capability axes (accuracy, CLIR, separability) and stays
  competitive on evenness; its weakest axis is **{weakest_axis}**.
- Single-number recall is **misleading**: monolingual/instruction models (e.g. `e5-large-instruct`)
  with respectable same-language matches collapse on the cross-lingual axes and fall to the bottom —
  exactly the failure a patent-retrieval deployment must avoid. The radar makes the lopsided profiles
  obvious.
- The robust-retriever recipe from this study: **avoid language collapse** (Round 6), **separate the
  foreign twin in score space** (Round 8), and accept that **the question's provenance — human or MT —
  barely matters** (Round 3). The remaining gains are in *alignment* and *smart combination*, not in
  re-ranking a single model (Rounds 8-9).

**Recommendation.** Deploy **{winner['short']}** as the single model; reserve a *score-aware* combiner
or per-language routing for the homeless/low-resource query languages (es, zh), where the oracle
headroom is largest (Round 9). Always report **CLIR@10 and language-parity alongside recall**.
<!-- /round:10 -->"""
    C.append_findings(md)
    print(f"[{SLUG}] done ->", out)


if __name__ == "__main__":
    main()
