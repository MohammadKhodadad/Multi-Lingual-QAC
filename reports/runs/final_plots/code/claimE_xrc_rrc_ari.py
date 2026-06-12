"""
Claim E: the deployment-cost wrap-up — XRC, RRC@K, ARI in one place.

Three deployment-cost instruments (all precomputed under chem_patents/experimental_plots):
  XRC50   reading-cost multiplier: how much deeper you must read to reach a foreign twin than a
          same-language copy (D50_cross / D50_same).  lower = cheaper.
  RRC@K   re-ranker recoverability: P[first foreign twin within top-K].  1 - RRC@K of the cross-lingual
          shortfall is provably un-recoverable by any top-K re-ranker.
  ARI@100 alignment-recoverability index: L_inf / (1 - RRC@100) = the share of the gap left after a
          top-100 re-rank that ONLY better alignment can move.  higher = re-ranking helps less.

Degeneracy gate: a model with CLIR@10 < 0.10 (e5-large-instruct; gte already excluded) retrieves too
little for these instruments to mean anything — drawn hollow / annotated, never on the frontier.

Three main-figure designs (user picks one in curation); two appendix redesigns.
  E1  single scatter: CLIR@10 vs XRC50 (log), colour = ARI@100, Pareto frontier.
  E2  compact 3-panel: XRC50 bars | RRC@K curves (knee K*, L_inf floor) | ARI stacked bars.
  E3  frontier scatter + RRC@K inset for the top-3 models.
  E4  (appendix) full RRC@K budget curves.
  E5  (appendix) ARI stacked decomposition.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fp_common as fp
import claimE_metrics


def _load():
    # XRC/RRC/ARI recomputed on the full 524-query gold (the shipped extra_*.csv are stale 137-query).
    if not (fp.DATA / "E_cost_frontier.csv").exists():
        claimE_metrics.compute()
    cf = pd.read_csv(fp.DATA / "E_cost_frontier.csv")
    ari = pd.read_csv(fp.DATA / "E_ari_decomposition.csv")
    knee = pd.read_csv(fp.DATA / "E_rrc_knee.csv")
    curve = pd.read_csv(fp.DATA / "E_rrc_curve.csv")
    m = cf.merge(ari[["short", "RRC_at_100", "deep_100_to_1000", "L_inf", "ARI_at_100"]],
                 on="short", how="left")
    rrc10 = curve[curve["K"] == 10][["short", "RRC"]].rename(columns={"RRC": "RRC_at_10"})
    m = m.merge(rrc10, on="short", how="left")
    order = [fp.short(x) for x in fp.MODEL_ORDER if fp.short(x) in set(m["short"])]
    m["short"] = pd.Categorical(m["short"], categories=order, ordered=True)
    m = m.sort_values("short").reset_index(drop=True)
    return m, knee, curve


def _model_color(s):
    return fp.MODEL_COLOR[[mm for mm in fp.MODEL_ORDER if fp.short(mm) == s][0]]


def _plain_log(ax, axis="x", ticks=None):
    """Show a log axis with plain integer ticks (1, 2, 10, 100 …) instead of 10^0 / 2×10^0 notation."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    if ticks is not None:
        (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------- E1
def e1_scatter():
    m, _, _ = _load()
    good = m[~m["degenerate_clir"]]
    bad = m[m["degenerate_clir"]]
    import matplotlib.pyplot as _plt
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=good["ARI_at_100"].min(), vmax=good["ARI_at_100"].max())
    cmap = _plt.get_cmap("RdYlGn_r")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    # Pareto frontier line (on_frontier points, sorted by clir)
    fr = good[good["on_frontier"]].sort_values("clir_at_10")
    ax.plot(fr["clir_at_10"], fr["XRC50"], color="#444", lw=1.6, zorder=1,
            label="Pareto frontier (max CLIR@10, min XRC50)")
    sc = ax.scatter(good["clir_at_10"], good["XRC50"], c=good["ARI_at_100"], cmap=cmap, norm=norm,
                    s=230, edgecolor="k", linewidth=1.0, zorder=3)
    # explicit per-model label offsets to declutter the mid-cluster (dx, dy, ha)
    lab = {"embeddinggemma": (-9, 10, "right"), "bge-m3": (9, -12, "left"),
           "granite-278m": (-9, -12, "right"), "qwen3-0.6B": (-9, 9, "right"),
           "nomic-v2-moe": (9, 6, "left"), "LaBSE": (9, -3, "left"), "SapBERT": (9, 4, "left")}
    for _, r in good.iterrows():
        dx, dy, ha = lab.get(r["short"], (9, 5, "left"))
        ax.annotate(r["short"], (r["clir_at_10"], r["XRC50"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9)
    # degenerate model(s): hollow, annotated
    for _, r in bad.iterrows():
        ax.scatter(r["clir_at_10"], r["XRC50"], facecolor="none", edgecolor="#999",
                   s=230, linewidth=1.4, zorder=3, hatch="xx")
        ax.annotate(f"{r['short']}\n(degenerate: CLIR@10<0.10)", (r["clir_at_10"], r["XRC50"]),
                    textcoords="offset points", xytext=(10, -4), fontsize=8.5, color="#777")
    ax.axvline(0.10, color="#c44e52", ls=":", lw=1.3)
    ax.text(0.105, ax.get_ylim()[1], "degeneracy gate", color="#c44e52", fontsize=8.5,
            va="top", rotation=90)
    ax.set_yscale("log")
    _plain_log(ax, "y", ticks=[1, 2, 5, 10, 100])
    ax.set_xlim(0.03, float(m["clir_at_10"].max()) * 1.12)
    ax.set_xlabel("CLIR@10  (cross-lingual recall, higher = better)")
    ax.set_ylabel("XRC50  reading-cost multiplier  (log, lower = better)")
    ax.set_title("Cost vs capability: cheap-to-read and accurate models sit bottom-right")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ARI@100  (gap share only\nalignment can fix; higher = re-ranking helps less)")
    ax.legend(loc="upper right")
    fp.save(fig, "claimE_E1_scatter")
    fp.dump_data(m[["short", "clir_at_10", "XRC50", "ARI_at_100", "on_frontier", "degenerate_clir"]],
                 "claimE_E1_scatter")


# --------------------------------------------------------------------------- E2
def _panel_xrc(ax, m):
    good = m[~m["degenerate_clir"]].sort_values("XRC50")
    ys = np.arange(len(good))[::-1]
    ax.barh(ys, good["XRC50"], color=[_model_color(s) for s in good["short"]])
    for y, v in zip(ys, good["XRC50"]):
        ax.text(v * 1.03, y, f"{v:.1f}×", va="center", fontsize=8.5)
    ax.set_yticks(ys); ax.set_yticklabels(good["short"])
    ax.axvline(1.0, color="#888", ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlim(1, 7.5)
    _plain_log(ax, "x", ticks=[1, 2, 3, 4, 5, 6, 7])
    ax.set_xlabel("XRC50  (× deeper to read, log)")
    ax.set_title("(a) reading cost")


def _panel_rrc(ax, m, curve, knee):
    for s in m[~m["degenerate_clir"]]["short"]:
        c = curve[curve["short"] == s].sort_values("K")
        ax.plot(c["K"], c["RRC"], color=_model_color(s), lw=1.8, label=s)
        k = knee[knee["short"] == s]
        if len(k):
            kk, rr = int(k["K_star"].iloc[0]), float(k["RRC_at_Kstar"].iloc[0])
            ax.scatter([kk], [rr], color=_model_color(s), s=28, zorder=5,
                       edgecolor="k", linewidth=0.5)
    ax.set_xscale("log")
    _plain_log(ax, "x", ticks=[1, 10, 100, 1000])
    ax.set_xlabel("re-rank depth K")
    ax.set_ylabel("RRC@K  (first foreign twin within top-K)")
    ax.set_title("(b) re-ranker recoverability (• = knee K*)")
    ax.legend(fontsize=6.5, loc="lower right", ncol=2)


def _panel_ari(ax, m):
    # decomposition split at K=10 and K=100: re-rankable in a top-10 rerank, recoverable only with a
    # deeper top-100 pool, and the alignment-only floor that lives beyond top-100.
    good = m[~m["degenerate_clir"]].copy()
    good["floor"] = 1.0 - good["RRC_at_100"]
    good = good.sort_values("floor")
    ys = np.arange(len(good))[::-1]
    cheap = good["RRC_at_10"].to_numpy()
    deep = (good["RRC_at_100"] - good["RRC_at_10"]).to_numpy()
    floor = good["floor"].to_numpy()
    ax.barh(ys, cheap, color="#2a924a", label="re-rankable in top-10")
    ax.barh(ys, deep, left=cheap, color="#9bd49b", label="needs deeper pool (top-100)")
    ax.barh(ys, floor, left=cheap + deep, color="#c44e52", label="alignment-only floor (beyond top-100)")
    for y, fl in zip(ys, floor):
        ax.text(1.01, y, f"floor {fl:.2f}", va="center", fontsize=8)
    ax.set_yticks(ys); ax.set_yticklabels(good["short"])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("share of the cross-lingual shortfall")
    ax.set_title("(c) where the gap lives")
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)


def e2_triptych():
    m, knee, curve = _load()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    _panel_xrc(axes[0], m)
    _panel_rrc(axes[1], m, curve, knee)
    _panel_ari(axes[2], m)
    fig.suptitle("Deployment cost of cross-lingual retrieval: how deep to read, "
                 "how much a re-ranker recovers, and what only alignment can fix",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fp.save(fig, "claimE_E2_triptych")


# --------------------------------------------------------------------------- E3
def e3_scatter_inset():
    m, knee, curve = _load()
    good = m[~m["degenerate_clir"]]
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    fr = good[good["on_frontier"]].sort_values("clir_at_10")
    ax.plot(fr["clir_at_10"], fr["XRC50"], color="#444", lw=1.6, zorder=1, label="Pareto frontier")
    for _, r in good.iterrows():
        right = r["clir_at_10"] > 0.42
        ax.scatter(r["clir_at_10"], r["XRC50"], color=_model_color(r["short"]), s=210,
                   edgecolor="k", linewidth=0.9, zorder=3)
        ax.annotate(r["short"], (r["clir_at_10"], r["XRC50"]), textcoords="offset points",
                    xytext=(-9, 8) if right else (9, 5), ha="right" if right else "left", fontsize=9)
    bad = m[m["degenerate_clir"]]
    for _, r in bad.iterrows():
        ax.scatter(r["clir_at_10"], r["XRC50"], facecolor="none", edgecolor="#999",
                   s=210, linewidth=1.4, hatch="xx", zorder=3)
        ax.annotate(f"{r['short']} (degenerate)", (r["clir_at_10"], r["XRC50"]),
                    textcoords="offset points", xytext=(10, -2), fontsize=8.5, color="#777")
    ax.set_yscale("log")
    _plain_log(ax, "y", ticks=[1, 2, 5, 10, 100])
    ax.set_xlim(0.03, float(m["clir_at_10"].max()) * 1.12)
    ax.set_ylim(1.1, 220)  # headroom so the inset clears every data point
    ax.set_xlabel("CLIR@10  (higher = better)")
    ax.set_ylabel("XRC50 reading-cost multiplier (log, lower = better)")
    ax.set_title("Cost-vs-capability frontier, with the re-ranker budget curve inset")
    ax.legend(loc="upper left")

    top3 = ["embeddinggemma", "bge-m3", "qwen3-0.6B"]
    axin = ax.inset_axes([0.6, 0.63, 0.38, 0.33])
    for s in top3:
        c = curve[curve["short"] == s].sort_values("K")
        axin.plot(c["K"], c["RRC"], color=_model_color(s), lw=1.6, label=s)
    axin.set_xscale("log"); _plain_log(axin, "x", ticks=[1, 10, 100, 1000])
    axin.set_title("RRC@K (top-3)", fontsize=8)
    axin.tick_params(labelsize=6.5); axin.set_xlabel("K", fontsize=7)
    axin.legend(fontsize=6, loc="lower right")
    fp.save(fig, "claimE_E3_scatter_inset")


# --------------------------------------------------------------------------- E4 (appendix)
def e4_rrc_curves():
    m, knee, curve = _load()
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for s in m["short"]:
        degen = bool(m[m["short"] == s]["degenerate_clir"].iloc[0])
        c = curve[curve["short"] == s].sort_values("K")
        ax.plot(c["K"], c["RRC"], color=_model_color(s), lw=1.9 if not degen else 1.2,
                ls="-" if not degen else ":", label=s + (" (degenerate)" if degen else ""),
                alpha=0.95 if not degen else 0.6)
        k = knee[knee["short"] == s]
        if len(k) and not degen:
            ax.scatter([int(k["K_star"].iloc[0])], [float(k["RRC_at_Kstar"].iloc[0])],
                       color=_model_color(s), s=34, edgecolor="k", linewidth=0.5, zorder=5)
    ax.axvline(100, color="#888", ls="--", lw=1.0)
    ax.text(105, 0.05, "top-100 re-rank", fontsize=8, color="#666")
    ax.set_xscale("log")
    _plain_log(ax, "x", ticks=[1, 10, 100, 1000])
    ax.set_xlabel("re-rank depth K")
    ax.set_ylabel("RRC@K  (cumulative first-foreign-twin hit rate)")
    ax.set_title("Re-ranker recoverability curves (• = knee K*; 1 − RRC@1000 = un-rerankable floor)")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fp.save(fig, "claimE_E4_rrc_curves")
    fp.dump_data(knee, "claimE_E4_rrc_curves")


# --------------------------------------------------------------------------- E5 (appendix)
def e5_ari_stack():
    m, _, _ = _load()
    good = m[~m["degenerate_clir"]].copy()
    good["cheap10"] = good["RRC_at_10"]
    good["deep10_100"] = good["RRC_at_100"] - good["RRC_at_10"]
    good["floor100"] = 1.0 - good["RRC_at_100"]
    good = good.sort_values("floor100", ascending=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ys = np.arange(len(good))[::-1]
    cheap = good["cheap10"].to_numpy()
    deep = good["deep10_100"].to_numpy()
    floor = good["floor100"].to_numpy()
    ax.barh(ys, cheap, color="#2a924a", label="re-rankable in top-10")
    ax.barh(ys, deep, left=cheap, color="#9bd49b", label="needs deeper pool (top-100)")
    ax.barh(ys, floor, left=cheap + deep, color="#c44e52", label="alignment-only floor (beyond top-100)")
    for i, y in enumerate(ys):
        ax.text(cheap[i] / 2, y, f"{cheap[i]:.0%}", va="center", ha="center", fontsize=7.5, color="white")
        ax.text(1.005, y, f"floor = {floor[i]:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(ys); ax.set_yticklabels(good["short"])
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("share of the cross-lingual shortfall (top-10 + top-100 + floor = 1)")
    ax.set_title("Re-ranker budget: most of the cross-lingual gap is reachable in a top-100 pool, "
                 "a floor beyond it is not")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fp.save(fig, "claimE_E5_ari_stack")
    fp.dump_data(good[["short", "RRC_at_10", "RRC_at_100", "cheap10", "deep10_100", "floor100"]],
                 "claimE_E5_ari_stack")


def main():
    fp.set_style()
    e1_scatter()
    e2_triptych()
    e3_scatter_inset()
    e4_rrc_curves()
    e5_ari_stack()
    print("claim E: E1-E5 written to candidates/")


if __name__ == "__main__":
    main()
