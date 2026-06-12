"""
EXTRA (round-3 additive analysis) — A1: the PER-ROUTE (per query-language) COST FRONTIER
                                    + W1: the per-route DECISION-FLIP count.

Converts the GLOBAL cost frontier (cp_fig18: CLIR@10 vs XRC50, pooled over all queries) into a
PER-ROUTE deployment map. A "route" here is a QUERY LANGUAGE ell in {en, de, es, fr, zh} (NOT a
directed language pair — the 5x4 directed XRC matrix would be far too thin to read). For each route:

  * x = CLIR@10_ell  = mean per-query clir_at_10 over the cross-gold queries asked in language ell.
                       ROBUST axis: n_cross per route = {de 27, en 27, es 34, fr 27, zh 22}.
  * y = XRC50_ell    = pct_depth(cross_ell, 50) / pct_depth(same_ell, 50)  (the MEDIAN, D50, NOT D95).
                       INDICATIVE axis: the same-language denominator is sparse per route
                       {en 21, fr 27, de 7, zh 2, es 0}. We carry n_same_ell + a censored flag with
                       EVERY value and frame the y-axis as indicative.
       - es route: n_same = 0 -> XRC UNDEFINED. es carries its CLIR coordinate ONLY (no y-value).
       - zh (n_same=2), de (n_same=7): thin_denominator=True -> XRC50_ell is indicative.

Per route we compute the 2-objective Pareto frontier (max CLIR@10_ell, min XRC50_ell) over the
finite-XRC models, and report frontier membership by route. For es (no XRC axis) the "frontier" is
the CLIR-only ordering (we report the max-CLIR corner; a 1-D frontier is just the argmax).

W1 (decision-flip): per route, the model a RECALL-ONLY dashboard picks (argmax recall@10_ell over ALL
gold) vs the model the CLIR/frontier picks (argmax CLIR@10_ell, which is the max-CLIR corner = always
on that route's frontier). Count the routes where the two disagree. This is the paper's thesis
("the recall dashboard is wrong") turned into a counted per-route fact.

Inputs (all on-disk, no API/network/embedding; HF_HUB_OFFLINE=1 via common):
  * common.core_per_query() — per-(model,query) frame with query_language, clir_at_10, recall_at_10,
    first_cross_rank, n_gold_cross (loads the 9 parquet rankings; lru_cached).
  * same-language reading depths rebuilt the extra_xrc_reading_cost way (first_gold_rank over the
    same-language gold, grouped by query_language).

Writes to a NEW dir, never touching existing outputs:
    reports/runs/chem_patents/experimental_plots/extra_per_route_frontier/
Outputs: per_route_frontier.csv, frontier_membership_by_route.csv, decision_flip_by_route.csv,
         summary.json, per_route_frontier.png
Paper figure target name: cp_fig23_per_route_frontier.png

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/extra_per_route_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
# reuse the exact machinery already written, do not re-implement
from extra_cost_frontier import pareto_frontier, dominators  # noqa: E402
from extra_xrc_reading_cost import pct_depth  # noqa: E402

SLUG = "extra_per_route_frontier"
SEED = 20260610
ROUTES = ["en", "de", "es", "fr", "zh"]  # query languages = routes
DEG_CLIR_CUTOFF = 0.10  # DEG gate (same as global frontier): clir_at_10 < this == degenerate
# (per-route cross/same-gold census is derived from the data and checked for consistency below;
#  the old 137-query release hardcoded EXPECTED_N_CROSS/EXPECTED_N_SAME, now stale on 524 queries.)


def main() -> None:
    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    cpq = C.core_per_query()
    gold = C.gold_publication()
    ql = C.q_lang()
    lists = C.ranked_lists()

    # ---- per-(model, query_language) SAME-language reading depths (the extra_xrc way) ----
    # same_lang_depth[model][lang] = list of first_gold_rank over the same-language gold set.
    same_lang_depth: dict[str, dict[str, list]] = {m: {} for m in C.MODEL_ORDER}
    for m in C.MODEL_ORDER:
        for qid, g in gold.items():
            lang = ql.get(qid, "")
            same, _ = C.split_gold(g, lang)
            if same:
                r = C.first_gold_rank(lists.get((m, qid), []), same)
                same_lang_depth[m].setdefault(lang, []).append(r)

    # ---- per-route, per-model CLIR@10_ell, XRC50_ell, recall@10_ell ----
    rows = []
    check_fails = []
    for m in C.MODEL_ORDER:
        s = C.short(m)
        msub = cpq[cpq.model == m]
        cross = msub[msub.n_gold_cross > 0]
        for lang in ROUTES:
            cl = cross[cross.query_language == lang]
            n_cross = int(len(cl))
            if n_cross == 0:
                continue
            clir10 = float(cl["clir_at_10"].mean())
            # recall@10 over the route's cross-gold queries (the recall-dashboard view, same domain
            # so the comparison is apples-to-apples per route)
            recall10 = float(cl["recall_at_10"].mean())

            cross_depths = cl["first_cross_rank"].to_numpy(dtype=float)
            same_depths = np.asarray(same_lang_depth[m].get(lang, []), dtype=float)
            n_same = int(same_depths.size)

            d50_cross, c50c = pct_depth(cross_depths, 50)
            if n_same == 0:
                # es route: no same-language gold -> XRC undefined. Do NOT impute.
                xrc50 = float("nan")
                xrc_censored = True
            else:
                d50_same, c50s = pct_depth(same_depths, 50)
                if not np.isfinite(d50_same) or d50_same == 0:
                    xrc50 = float("nan")
                    xrc_censored = True
                elif not np.isfinite(d50_cross):
                    # numerator censored: lower-bound XRC with MAXRANK floor on the numerator
                    xrc50 = 1000.0 / d50_same
                    xrc_censored = True
                else:
                    xrc50 = d50_cross / d50_same
                    xrc_censored = bool(c50c or c50s)

            thin = (0 < n_same <= 10)  # flag fragile (small) same-language denominators
            rows.append({
                "route": lang, "model": m, "short": s,
                "clir_at_10": round(clir10, 4),
                "recall_at_10": round(recall10, 4),
                "XRC50": round(float(xrc50), 3) if np.isfinite(xrc50) else float("nan"),
                "n_cross": n_cross, "n_same": n_same,
                "xrc_censored": bool(xrc_censored),
                "thin_denominator": bool(thin),
                "degenerate_clir": bool(clir10 < DEG_CLIR_CUTOFF),
            })

    rt_df = pd.DataFrame(rows)

    # ---- per-route Pareto frontier (max CLIR@10_ell, min XRC50_ell) over finite-XRC models ----
    # For es (XRC undefined for every model) the frontier degenerates to the CLIR argmax (1-D).
    on_front_col = {}
    dom_col = {}
    for lang in ROUTES:
        sub = rt_df[rt_df.route == lang].copy()
        finite = sub[np.isfinite(sub["XRC50"])].copy()
        if len(finite) >= 1 and finite["XRC50"].notna().any():
            # pareto_frontier expects a column named XRC50f
            fin2 = finite.rename(columns={"XRC50": "XRC50f"})
            front = pareto_frontier(fin2)
            for r in sub.itertuples():
                key = (lang, r.short)
                if r.short in finite["short"].values:
                    on_front_col[key] = (r.short in front)
                    if r.short in front:
                        dom_col[key] = ""
                    else:
                        dom_col[key] = ";".join(dominators(fin2, r.short))
                else:
                    on_front_col[key] = False  # non-finite XRC -> off the finite plane
                    dom_col[key] = ""
        else:
            # es: no finite XRC at all -> 1-D CLIR frontier = the single max-CLIR model
            cmax = sub.sort_values("clir_at_10", ascending=False).iloc[0]["short"]
            for r in sub.itertuples():
                key = (lang, r.short)
                on_front_col[key] = (r.short == cmax)
                dom_col[key] = ""

    rt_df["on_frontier"] = [on_front_col[(r.route, r.short)] for r in rt_df.itertuples()]
    rt_df["dominated_by"] = [dom_col[(r.route, r.short)] for r in rt_df.itertuples()]
    rt_df.to_csv(out / "per_route_frontier.csv", index=False)

    # ---- frontier membership by route (the headline table) ----
    memb_rows = []
    for lang in ROUTES:
        sub = rt_df[rt_df.route == lang]
        members = sub[sub.on_frontier]["short"].tolist()
        cmax_model = sub.sort_values("clir_at_10", ascending=False).iloc[0]["short"]
        cmax_val = float(sub["clir_at_10"].max())
        memb_rows.append({
            "route": lang,
            "n_cross": int(sub["n_cross"].iloc[0]) if len(sub) else 0,
            "n_same": int(sub["n_same"].iloc[0]) if len(sub) else 0,
            "xrc_axis": ("undefined (no same-lang gold)" if lang == "es"
                         else ("thin" if (sub["thin_denominator"].any()) else "ok")),
            "frontier_members": ";".join(members),
            "max_clir_corner": cmax_model,
            "max_clir_value": round(cmax_val, 4),
        })
    memb_df = pd.DataFrame(memb_rows)
    memb_df.to_csv(out / "frontier_membership_by_route.csv", index=False)

    # ---- W1: decision-flip per route (recall-only pick vs CLIR/frontier pick) ----
    flip_rows = []
    for lang in ROUTES:
        sub = rt_df[rt_df.route == lang]
        recall_pick = sub.sort_values("recall_at_10", ascending=False).iloc[0]["short"]
        clir_pick = sub.sort_values("clir_at_10", ascending=False).iloc[0]["short"]  # max-CLIR corner
        flip_rows.append({
            "route": lang,
            "recall_only_pick": recall_pick,
            "frontier_pick": clir_pick,
            "flipped": bool(recall_pick != clir_pick),
        })
    flip_df = pd.DataFrame(flip_rows)
    flip_df.to_csv(out / "decision_flip_by_route.csv", index=False)
    n_routes_flipped = int(flip_df["flipped"].sum())

    # ---- VERIFY: pooling routes reproduces the global numbers ----
    # (a) pooled CLIR@10 per model == cost_frontier.csv clir_at_10 (egemma 0.5024)
    global_front = pd.read_csv(C.PLOTS_DIR / "extra_cost_frontier" / "cost_frontier.csv")
    gmap = {r.short: float(r.clir_at_10) for r in global_front.itertuples()}
    for m in C.MODEL_ORDER:
        s = C.short(m)
        pooled_clir = float(cpq[(cpq.model == m) & (cpq.n_gold_cross > 0)]["clir_at_10"].mean())
        if abs(pooled_clir - gmap[s]) > 1e-3:
            check_fails.append(f"{s}: pooled CLIR@10 {pooled_clir:.4f} != global {gmap[s]:.4f}")
    # (b) per-route cross/same-gold counts are model-invariant (constant across models for a route)
    for lang in ROUTES:
        sub = rt_df[rt_df.route == lang]
        if sub.empty:
            continue
        if sub["n_cross"].nunique() > 1 or sub["n_same"].nunique() > 1:
            check_fails.append(f"route {lang}: per-model n_cross/n_same not constant")
    # (c) a route's XRC50 is defined only when it has same-language gold (n_same > 0)
    for lang in ROUTES:
        sub = rt_df[rt_df.route == lang]
        if sub.empty:
            continue
        if int(sub["n_same"].iloc[0]) == 0 and not sub["XRC50"].isna().all():
            check_fails.append(f"route {lang}: XRC50 should be NaN when n_same==0")

    # ---- does the max-CLIR (capability) corner move across routes? (the novelty hook) ----
    corner_by_route = {lang: memb_df[memb_df.route == lang]["max_clir_corner"].iloc[0] for lang in ROUTES}
    corner_models = set(corner_by_route.values())
    corner_moves = len(corner_models) > 1

    # ---- figure: small-multiples 5-panel grid (one panel per route) ----
    name2model = {C.short(m): m for m in C.MODEL_ORDER}
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.4), sharey=False)
    for ax, lang in zip(axes, ROUTES):
        sub = rt_df[rt_df.route == lang]
        es_route = (lang == "es")
        finite = sub[np.isfinite(sub["XRC50"])]
        # frontier polyline (finite models on frontier, sorted by CLIR)
        if not es_route and len(finite):
            fdf = finite[finite["on_frontier"]].sort_values("clir_at_10")
            if len(fdf) >= 2:
                ax.plot(fdf["clir_at_10"], fdf["XRC50"], color="#444444", lw=1.3, zorder=2)
        for r in sub.itertuples():
            col = C.MODEL_COLOR[name2model[r.short]]
            is_deg = r.degenerate_clir
            on_f = r.on_frontier
            if es_route or not np.isfinite(r.XRC50):
                # CLIR-only: place along a baseline, mark XRC-undefined
                yb = 1.0
                mk = "o" if on_f else "X"
                ax.scatter(r.clir_at_10, yb, s=110 if on_f else 70, color=col, marker=mk,
                           zorder=4, edgecolor=("#d62728" if is_deg else "black"),
                           linewidth=1.3 if is_deg else 0.5)
                ax.annotate(r.short, (r.clir_at_10, yb), fontsize=6.3,
                            xytext=(3, 4), textcoords="offset points")
            else:
                mk = "o" if on_f else "X"
                ax.scatter(r.clir_at_10, r.XRC50, s=120 if on_f else 70, color=col, marker=mk,
                           zorder=4, edgecolor=("#d62728" if is_deg else "black"),
                           linewidth=1.3 if is_deg else 0.5)
                ax.annotate(r.short, (r.clir_at_10, r.XRC50), fontsize=6.3,
                            xytext=(3, 4), textcoords="offset points")
        n_cross = int(sub["n_cross"].iloc[0]); n_same = int(sub["n_same"].iloc[0])
        if es_route:
            ax.set_yscale("linear"); ax.set_ylim(0.7, 1.3)
            ax.set_yticks([1.0]); ax.set_yticklabels(["(no y-axis)"])
            ax.text(0.5, 0.02, "no same-language gold\n-> reading-cost undefined\n(CLIR-only route)",
                    transform=ax.transAxes, fontsize=7, ha="center", va="bottom", color="#d62728")
        else:
            if len(finite):
                ax.set_yscale("log")
                ax.axhline(1.0, color="grey", ls="--", lw=0.6)
            thin = bool(sub["thin_denominator"].any())
            if thin:
                ax.text(0.5, 0.97, f"XRC indicative (n_same={n_same}, thin)",
                        transform=ax.transAxes, fontsize=6.8, ha="center", va="top", color="#cc6677")
        cmax = corner_by_route[lang]
        ax.set_title(f"route {lang}  (n_cross={n_cross}, n_same={n_same})\n"
                     f"max-CLIR corner: {cmax}", fontsize=9)
        ax.set_xlabel("CLIR@10_ell (higher = better)", fontsize=8)
        if lang == ROUTES[0]:
            ax.set_ylabel("XRC50_ell = D50(cross)/D50(same)\n(log; INDICATIVE, lower = better)", fontsize=8)
    fig.suptitle(
        "Per-route cost frontier (route = query language). x = CLIR@10_ell (ROBUST, n=22-34); "
        "y = XRC50_ell (INDICATIVE: thin/undefined same-lang denominator). "
        f"Max-CLIR corner moves across routes: {corner_moves}; decision-flips: {n_routes_flipped}/5.",
        fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "per_route_frontier.png")
    plt.close(fig)

    # ---- summary ----
    summary = {
        "method": (
            "Routes = query languages {en,de,es,fr,zh}. x=CLIR@10_ell=mean per-query clir_at_10 over "
            "route's cross-gold queries (robust). y=XRC50_ell=pct_depth(cross,50)/pct_depth(same,50) "
            "with the MEDIAN (D50), carrying n_same + censored/thin flags (indicative). es: no "
            "same-lang gold -> XRC undefined (not imputed). Per-route Pareto over finite-XRC models; "
            "es frontier = CLIR argmax. Decision-flip = argmax(recall@10_ell) vs argmax(CLIR@10_ell)."),
        "routes": ROUTES,
        "deg_cutoff": DEG_CLIR_CUTOFF,
        "n_cross_by_route": {lang: int(rt_df[rt_df.route == lang]["n_cross"].iloc[0]) for lang in ROUTES},
        "n_same_by_route": {lang: int(rt_df[rt_df.route == lang]["n_same"].iloc[0]) for lang in ROUTES},
        "xrc_axis_status": {
            "es": "undefined (n_same=0)",
            "zh": "thin (n_same=2, indicative)",
            "de": "thin (n_same=7, indicative)",
            "en": "credible (n_same=21)",
            "fr": "credible (n_same=27)",
        },
        "frontier_members_by_route": {
            lang: memb_df[memb_df.route == lang]["frontier_members"].iloc[0] for lang in ROUTES},
        "max_clir_corner_by_route": corner_by_route,
        "capability_corner_moves_across_routes": bool(corner_moves),
        "n_distinct_max_clir_corners": int(len(corner_models)),
        "decision_flip_by_route": {r.route: bool(r.flipped) for r in flip_df.itertuples()},
        "decision_flip_detail": {
            r.route: {"recall_pick": r.recall_only_pick, "frontier_pick": r.frontier_pick,
                      "flipped": bool(r.flipped)} for r in flip_df.itertuples()},
        "n_routes_flipped": n_routes_flipped,
        "global_pareto_set_reference": sorted(
            global_front[global_front.on_frontier]["short"].tolist()),
        "verify_egemma_global_clir_0.5024": float(gmap.get("embeddinggemma", float("nan"))),
        "checks_passed": len(check_fails) == 0,
        "check_failures": check_fails,
    }
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] per-route frontier (route, short, CLIR@10, XRC50, n_same, on_frontier):")
    print(rt_df[["route", "short", "clir_at_10", "recall_at_10", "XRC50", "n_same",
                 "xrc_censored", "thin_denominator", "on_frontier"]].to_string(index=False))
    print(f"\n[{SLUG}] frontier membership by route:")
    print(memb_df.to_string(index=False))
    print(f"\n[{SLUG}] decision-flip by route:")
    print(flip_df.to_string(index=False))
    print(f"\n[{SLUG}] max-CLIR corner by route: {corner_by_route}")
    print(f"[{SLUG}] capability corner moves across routes: {corner_moves} "
          f"({len(corner_models)} distinct corners: {sorted(corner_models)})")
    print(f"[{SLUG}] decision-flips: {n_routes_flipped}/5 routes")
    if check_fails:
        print(f"\n[{SLUG}] !!! CHECK FAILURES:")
        for f in check_fails:
            print("   ", f)
    else:
        print(f"[{SLUG}] all verify checks PASSED "
              f"(pooled CLIR==global; per-route census; es XRC undefined).")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
