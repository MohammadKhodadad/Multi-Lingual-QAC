"""
EXTRA (round-2 additive analysis) — the cross-lingual COST FRONTIER (DO-NOW-1).

Replaces the false N2 superlative ("lowest non-degenerate reading cost = embeddinggemma") with a
*true Pareto frontier* claim in the deployment plane:

    x = CLIR@10  (cross-lingual recall@10, HIGHER is better)
    y = XRC50    (median cross-lingual reading-cost multiplier, LOWER is better)

A model m is on the Pareto frontier iff no other FINITE-XRC model dominates it: i.e. no other model
has CLIR@10 >= m.CLIR@10 AND XRC50 <= m.XRC50 with at least one strict. The honest headline claim is
embeddinggemma's PARETO MEMBERSHIP (it is the unique max-CLIR@10 corner, undominated) — NOT a
"cheapest deployable reader" superlative (that was the N2 bug: at any sensible CLIR threshold the
min-XRC admitted model is bge-m3, NOT embeddinggemma).

The W2 "trap" number: Spearman rho(XRC50, CLIR@10) over the NON-DEGENERATE set (DEG gate
clir_at_10 < 0.10 excludes {gte-base, e5-large-instruct}). If it is POSITIVE, the trap framing
("the cheapest reader can be the worst retriever") is supported; the script gates the prose on the
actual computed sign and records it in summary.json.

Reads ONLY the on-disk CSV emitted by extra_xrc_reading_cost (no parquet load, no API, no eval):
    reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/xrc_per_model.csv

Writes to a NEW dir, never touching the round-1 outputs:
    reports/runs/chem_patents/experimental_plots/extra_cost_frontier/
Outputs: cost_frontier.csv, summary.json, cost_frontier.png
Paper figure target name: cp_fig18_cost_frontier.png

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/extra_cost_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_cost_frontier"
SEED = 20260610
TAU = 0.40  # stated (not tuned) CLIR@10 threshold for the deployment read-off
DEG_CLIR_CUTOFF = 0.10  # DEG gate: clir_at_10 < this == degenerate (flags exactly {gte, e5})


def pareto_frontier(df: pd.DataFrame) -> set[str]:
    """Models on the 2-obj non-dominated set over FINITE-XRC rows.
    Objective: maximize clir_at_10, minimize XRC50. m is dominated iff some other model has
    clir >= m.clir AND xrc <= m.xrc with at least one strict inequality."""
    rows = df[["short", "clir_at_10", "XRC50f"]].to_dict("records")
    front = set()
    for a in rows:
        dominated = False
        for b in rows:
            if b["short"] == a["short"]:
                continue
            ge = b["clir_at_10"] >= a["clir_at_10"] and b["XRC50f"] <= a["XRC50f"]
            strict = b["clir_at_10"] > a["clir_at_10"] or b["XRC50f"] < a["XRC50f"]
            if ge and strict:
                dominated = True
                break
        if not dominated:
            front.add(a["short"])
    return front


def dominators(df: pd.DataFrame, target: str) -> list[str]:
    a = df[df["short"] == target].iloc[0]
    out = []
    for b in df.itertuples():
        if b.short == target:
            continue
        ge = b.clir_at_10 >= a.clir_at_10 and b.XRC50f <= a.XRC50f
        strict = b.clir_at_10 > a.clir_at_10 or b.XRC50f < a.XRC50f
        if ge and strict:
            out.append(b.short)
    return out


def main() -> None:
    from scipy.stats import spearmanr

    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    src = C.PLOTS_DIR / "extra_xrc_reading_cost" / "xrc_per_model.csv"
    if not src.is_file():
        raise FileNotFoundError(f"missing input CSV: {src} (run extra_xrc_reading_cost.py first)")
    xrc = pd.read_csv(src)

    # Coerce XRC50 to float; blank (gte) -> NaN. Censored-XRC models with blank XRC50 are degenerate
    # / off-frontier (cannot be placed on the finite plane).
    xrc["XRC50f"] = pd.to_numeric(xrc["XRC50"], errors="coerce")
    xrc["XRC50_censored"] = xrc["XRC50_censored"].astype(bool)

    finite = xrc[np.isfinite(xrc["XRC50f"])].copy()       # 8 models (gte excluded: blank XRC50)
    degenerate_off_plane = xrc[~np.isfinite(xrc["XRC50f"])]["short"].tolist()  # ['gte-base']

    # ---- Pareto frontier over finite-XRC models ----
    front = pareto_frontier(finite)

    # ---- DEG gate (clir_at_10 < 0.10) -> non-degenerate set for the W2 trap correlation ----
    nondeg = xrc[xrc["clir_at_10"] >= DEG_CLIR_CUTOFF].copy()
    deg = xrc[xrc["clir_at_10"] < DEG_CLIR_CUTOFF]["short"].tolist()

    # W2 trap: Spearman rho(XRC50, CLIR@10) over non-degenerate FINITE-XRC models. All non-degenerate
    # models have finite XRC50 here (gte/e5 are the two degenerates), so n == len(nondeg).
    nd_fin = nondeg[np.isfinite(nondeg["XRC50f"])].copy()
    rho, pval = spearmanr(nd_fin["XRC50f"].to_numpy(), nd_fin["clir_at_10"].to_numpy())
    trap_supported = bool(rho > 0)

    # ---- deployment read-off at tau (stated threshold, NOT tuned) ----
    admitted = xrc[xrc["clir_at_10"] >= TAU].sort_values("XRC50f")
    if len(admitted):
        min_xrc_row = admitted.iloc[0]
        min_xrc_model = str(min_xrc_row["short"])
        min_xrc_val = float(min_xrc_row["XRC50f"])
    else:
        min_xrc_model, min_xrc_val = "", float("nan")
    # the unique max-CLIR@10 model (the top-right Pareto corner) — the honest egemma claim
    top_clir_row = xrc.sort_values("clir_at_10", ascending=False).iloc[0]
    top_clir_model = str(top_clir_row["short"])

    # ---- per-model frontier table ----
    rows = []
    for r in xrc.itertuples():
        finite_xrc = bool(np.isfinite(r.XRC50f))
        on_front = (r.short in front) if finite_xrc else False
        doms = dominators(finite, r.short) if (finite_xrc and r.short not in front) else []
        rows.append({
            "model": r.model, "short": r.short,
            "clir_at_10": round(float(r.clir_at_10), 4),
            "XRC50": float(r.XRC50f) if finite_xrc else "",
            "XRC50_censored": bool(r.XRC50_censored),
            "finite_xrc": finite_xrc,
            "degenerate_clir": bool(r.clir_at_10 < DEG_CLIR_CUTOFF),
            "on_frontier": on_front,
            "dominated_by": ";".join(doms),
        })
    front_df = pd.DataFrame(rows)
    front_df.to_csv(out / "cost_frontier.csv", index=False)

    # ---- figure: scatter with the frontier line ----
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    # frontier polyline: non-dominated finite models sorted by CLIR@10
    fdf = finite[finite["short"].isin(front)].sort_values("clir_at_10")
    ax.plot(fdf["clir_at_10"], fdf["XRC50f"], color="#444444", lw=1.4, ls="-", zorder=2,
            label="Pareto frontier (max CLIR@10, min XRC50)")
    for r in xrc.itertuples():
        col = C.MODEL_COLOR[r.model]
        if not np.isfinite(r.XRC50f):
            continue  # plotted separately below
        on_front = r.short in front
        is_deg = r.clir_at_10 < DEG_CLIR_CUTOFF
        mk = "o" if on_front else "X"
        edge = "#d62728" if is_deg else "black"
        ax.scatter(r.clir_at_10, r.XRC50f, s=140 if on_front else 90, color=col, marker=mk,
                   zorder=4, edgecolor=edge, linewidth=1.4 if is_deg else 0.5)
        ax.annotate(r.short, (r.clir_at_10, r.XRC50f), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    # degenerate off-plane models (blank XRC50, e.g. gte): mark at the top with a censored marker
    ymax_finite = float(np.nanmax(finite["XRC50f"]))
    for r in xrc.itertuples():
        if np.isfinite(r.XRC50f):
            continue
        col = C.MODEL_COLOR[r.model]
        ax.scatter(r.clir_at_10, ymax_finite * 1.35, s=110, color=col, marker="v",
                   zorder=4, edgecolor="#d62728", linewidth=1.4)
        ax.annotate(f"{r.short}\n(XRC50 censored,\noff-frontier)", (r.clir_at_10, ymax_finite * 1.35),
                    fontsize=7, xytext=(6, -2), textcoords="offset points", color="#d62728")
    ax.axvline(TAU, color="grey", ls=":", lw=0.9)
    ax.text(TAU, ax.get_ylim()[1] if False else ymax_finite * 1.45, f"  CLIR@10 = {TAU}",
            fontsize=7, color="grey", va="top")
    ax.axhline(1.0, color="grey", ls="--", lw=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("CLIR@10  (cross-lingual recall@10, higher = better)")
    ax.set_ylabel("XRC50  =  D50(cross) / D50(same)\n(median reading-cost multiplier, log, lower = better)")
    ax.set_title("Cross-lingual cost frontier: capability (CLIR@10) vs reading cost (XRC50)\n"
                 "(o = Pareto-optimal; X = dominated; red edge = degenerate CLIR@10 < 0.10)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "cost_frontier.png"); plt.close(fig)

    # ---- summary ----
    summary = {
        "method": "Pareto over finite-XRC models: max clir_at_10, min XRC50. DEG gate clir_at_10<0.10.",
        "tau": TAU,
        "deg_cutoff": DEG_CLIR_CUTOFF,
        "frontier_members": sorted(front),
        "dominated_finite_models": sorted(set(finite["short"]) - front),
        "degenerate_off_plane_models": degenerate_off_plane,
        "degenerate_by_clir_gate": deg,
        "embeddinggemma_on_frontier": "embeddinggemma" in front,
        "granite_on_frontier": "granite-278m" in front,
        "unique_top_clir_model": top_clir_model,
        "tau_admitted_set": admitted["short"].tolist(),
        "tau_admitted_min_xrc_model": min_xrc_model,
        "tau_admitted_min_xrc_value": min_xrc_val,
        "HONEST_CLAIM": (
            f"embeddinggemma is the unique max-CLIR@10 corner of the cost frontier and is "
            f"Pareto-optimal; the cheaper-XRC admitted model at tau={TAU} is "
            f"'{min_xrc_model}' (XRC50 {min_xrc_val}), so do NOT call egemma 'cheapest deployable'."),
        "W2_trap_spearman_rho_XRC50_vs_CLIR10_nondeg": round(float(rho), 4),
        "W2_trap_spearman_p": round(float(pval), 4),
        "W2_trap_n_nondeg": int(len(nd_fin)),
        "W2_trap_supported_positive_rho": trap_supported,
        "W2_trap_prose_guidance": (
            "POSITIVE rho => trap sentence valid ('cheapest reader can be the worst retriever')"
            if trap_supported else
            "NON-POSITIVE rho => DO NOT use the trap sentence; fall back to minimal frontier wording"),
        "verify_egemma_xrc50_3.5": float(xrc[xrc.short == "embeddinggemma"]["XRC50f"].iloc[0]),
        "verify_egemma_clir_0.5024": float(xrc[xrc.short == "embeddinggemma"]["clir_at_10"].iloc[0]),
        "verify_granite_xrc50_1.25": float(xrc[xrc.short == "granite-278m"]["XRC50f"].iloc[0]),
        "verify_granite_clir_0.3285": float(xrc[xrc.short == "granite-278m"]["clir_at_10"].iloc[0]),
    }
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] cost-frontier per-model:")
    print(front_df[["short", "clir_at_10", "XRC50", "finite_xrc", "degenerate_clir",
                    "on_frontier", "dominated_by"]].to_string(index=False))
    print(f"\n[{SLUG}] Pareto frontier members : {sorted(front)}")
    print(f"[{SLUG}] degenerate (clir<{DEG_CLIR_CUTOFF}) : {deg}")
    print(f"[{SLUG}] off-plane (blank XRC50)    : {degenerate_off_plane}")
    print(f"[{SLUG}] tau={TAU} admitted set      : {admitted['short'].tolist()}  "
          f"-> min-XRC50 = {min_xrc_model} ({min_xrc_val})")
    print(f"[{SLUG}] unique max-CLIR@10 corner   : {top_clir_model}")
    print(f"[{SLUG}] W2 trap Spearman rho(XRC50, CLIR@10) over non-deg n={len(nd_fin)} : "
          f"rho={rho:.4f} p={pval:.4f}  -> trap {'SUPPORTED' if trap_supported else 'NOT supported'}")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
