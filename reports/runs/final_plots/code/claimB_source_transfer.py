"""
Claim B: the failure pattern transfers across patent offices (Google Patents vs EPO).

Two things must transfer for the benchmark to be more than one dataset's quirk:
  (i)  the model ranking is stable across offices, and
  (ii) the semantic > technical penalty itself replicates per model.

Candidates
  B1  paired scatter: language-balanced Recall@10 on GP vs EPO, per model, with the identity line,
      Spearman & Pearson, corpus sizes on the axes, cross-degenerate models drawn hollow.
  B2  slope chart GP -> EPO on raw Recall@10 (rank-stability: do lines cross?).
  B3  transfer of the penalty itself: per-model (semantic - technical) gap on GP vs EPO.

Normalization
  * language-balanced Recall@10 (macro over languages) for both offices, so the 5-vs-3 language
    difference and unequal per-language counts don't drive the comparison.
  * GP uses human-original queries only.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import fp_common as fp


def _overall(df: pd.DataFrame) -> pd.Series:
    """model short -> language-balanced overall Recall@10."""
    out = {}
    for model in fp.MODEL_ORDER:
        sub = df[df["model"] == model]
        if not sub.empty:
            out[fp.short(model)] = fp.lang_balanced_mean(sub, "recall_at_10")
    return pd.Series(out)


def _gp_all():
    return fp.cp_per_query()


def _per_model_gap(df: pd.DataFrame) -> pd.Series:
    """model short -> language-balanced [R@10(semantic) - R@10(technical)]."""
    out = {}
    for model in fp.MODEL_ORDER:
        sem = df[(df["model"] == model) & (df["mode"] == "semantic")]
        tec = df[(df["model"] == model) & (df["mode"] == "technical")]
        if sem.empty or tec.empty:
            continue
        out[fp.short(model)] = (fp.lang_balanced_mean(sem, "recall_at_10")
                                - fp.lang_balanced_mean(tec, "recall_at_10"))
    return pd.Series(out)


# --------------------------------------------------------------------------- B1
def b1_scatter():
    gp, epo = _overall(_gp_all()), _overall(fp.epo_per_query())
    common = [s for s in gp.index if s in epo.index]
    x = gp[common].to_numpy(); y = epo[common].to_numpy()
    rho = spearmanr(x, y).statistic
    r = pearsonr(x, y).statistic

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    lim = (0, max(x.max(), y.max()) * 1.18)
    ax.plot(lim, lim, ls="--", color="#888", lw=1.2, zorder=1, label="y = x (identical)")
    for s in common:
        model = [m for m in fp.MODEL_ORDER if fp.short(m) == s][0]
        degen = model in fp.DEGENERATE_CROSS
        ax.scatter(gp[s], epo[s], s=130, color=fp.MODEL_COLOR[model], zorder=3,
                   edgecolor="k", linewidth=0.8,
                   facecolor="none" if degen else fp.MODEL_COLOR[model])
        ax.annotate(s, (gp[s], epo[s]), textcoords="offset points", xytext=(8, 4), fontsize=8.5)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Google Patents — Recall@10  (23,787-doc haystack)")
    ax.set_ylabel("EPO — Recall@10  (11,315-doc haystack)")
    ax.set_title("Model standing transfers across patent offices")
    ax.text(0.04, 0.94, f"Spearman ρ = {rho:.2f}\nPearson r = {r:.2f}",
            transform=ax.transAxes, fontsize=10.5, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.legend(loc="lower right")
    fp.save(fig, "claimB_B1_scatter")
    fp.dump_data(pd.DataFrame({"short": common, "gp_recall10": gp[common].to_numpy(),
                               "epo_recall10": epo[common].to_numpy()}), "claimB_B1_scatter")


# --------------------------------------------------------------------------- B2
def b2_slope():
    gp, epo = _overall(_gp_all()), _overall(fp.epo_per_query())
    common = [s for s in gp.index if s in epo.index]
    order = sorted(common, key=lambda s: gp[s], reverse=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for s in order:
        model = [m for m in fp.MODEL_ORDER if fp.short(m) == s][0]
        ax.plot([0, 1], [gp[s], epo[s]], color=fp.MODEL_COLOR[model], lw=2.0, marker="o", ms=7)
        ax.annotate(s, (0, gp[s]), textcoords="offset points", xytext=(-6, 0),
                    ha="right", va="center", fontsize=8.5)
        ax.annotate(f"{epo[s]:.2f}", (1, epo[s]), textcoords="offset points", xytext=(6, 0),
                    ha="left", va="center", fontsize=8.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Google Patents", "EPO"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("Recall@10  (language-balanced)")
    ax.set_title("Few rank crossings between offices")
    ax.grid(axis="x", visible=False)
    fp.save(fig, "claimB_B2_slope")
    fp.dump_data(pd.DataFrame({"short": order, "gp_recall10": [gp[s] for s in order],
                               "epo_recall10": [epo[s] for s in order]}), "claimB_B2_slope")


# --------------------------------------------------------------------------- B3
def b3_gap_transfer():
    gp_gap = _per_model_gap(_gp_all())
    epo_gap = _per_model_gap(fp.epo_per_query())
    common = [s for s in gp_gap.index if s in epo_gap.index]
    x = gp_gap[common].to_numpy(); y = epo_gap[common].to_numpy()
    rho = spearmanr(x, y).statistic

    n_pos = int(((x > 0) & (y > 0)).sum())
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    lo = min(0, x.min(), y.min()) - 0.03
    hi = max(x.max(), y.max()) * 1.18
    ax.axhline(0, color="#bbb", lw=1); ax.axvline(0, color="#bbb", lw=1)
    ax.plot([lo, hi], [lo, hi], ls="--", color="#888", lw=1.1, label="y = x")
    # shade only the positive–positive quadrant (semantic beats technical on BOTH offices)
    xmin_frac = (0 - lo) / (hi - lo)
    ax.axhspan(0, hi, xmin=xmin_frac, xmax=1, color="#e8f3e8", zorder=0)
    for s in common:
        model = [m for m in fp.MODEL_ORDER if fp.short(m) == s][0]
        ax.scatter(gp_gap[s], epo_gap[s], s=130, color=fp.MODEL_COLOR[model],
                   edgecolor="k", linewidth=0.8, zorder=3)
        ax.annotate(s, (gp_gap[s], epo_gap[s]), textcoords="offset points", xytext=(8, 4), fontsize=8.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Google Patents — R@10 gap (semantic − technical)")
    ax.set_ylabel("EPO — R@10 gap (semantic − technical)")
    ax.set_title("The technical penalty itself transfers across offices")
    ax.text(0.04, 0.94,
            f"Spearman ρ = {rho:.2f}\n{n_pos}/{len(common)} models in the\npositive–positive quadrant",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.legend(loc="lower right")
    fp.save(fig, "claimB_B3_gap_transfer")
    fp.dump_data(pd.DataFrame({"short": common, "gp_gap": x, "epo_gap": y}), "claimB_B3_gap_transfer")


def main():
    fp.set_style()
    b1_scatter()
    b2_slope()
    b3_gap_transfer()
    print("claim B: B1-B3 written to candidates/")


if __name__ == "__main__":
    main()
