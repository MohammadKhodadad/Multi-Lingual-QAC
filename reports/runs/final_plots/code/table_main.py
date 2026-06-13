"""
Main results table: every model's retrieval performance on both benchmarks (Google Patents, 524q/5
langs; EPO, 198q/3 langs), with four metrics chosen to capture IR power and cross-linguality together.

Metrics (per benchmark)
  nDCG@10   ranking quality (IR power)            — from mteb_tables/model_comparison.csv
  R@10      Recall@10, language-balanced (IR power)
  CLIR@10   cross-language Recall@10 (cross-linguality), language-balanced
  LT%       Language Transfer = CLIR@10 / MoLIR@10 (the share of its own same-language ceiling a
            model keeps cross-lingually; low LT = language-siloing)

Both benchmarks reconstruct gold from the source patent's language versions in the haystack and are
validated to the official MTEB Recall@10 to 4 decimals. Degenerate gte-multilingual-base is excluded;
e5-large-instruct is kept but daggered (fails the CLIR@10 < 0.10 cross-lingual degeneracy gate on GP).

Outputs: data/main_table.csv, tables/main_table.tex (booktabs), and a printed preview.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import claimE_metrics
import fp_common as fp

TABLES = fp.FINAL / "tables"
DEG_CLIR = 0.10  # cross-lingual degeneracy gate
TAU = 0.80       # joint bilingual coverage fraction
MAXRANK = 1000


def _joint_bilingual() -> dict:
    """Joint bilingual *gold* coverage (the caveat-free, foreign-GOLD formulation; a language-faceted
    instance of subtopic recall, Zhai et al. SIGIR 2003). Per query the joint depth is
    max(first same-language-gold rank, first cross-language-gold rank). Domain = the GP+EPO queries
    that have both gold types (n=459). Returns short -> (JBC@100, k*80):
      JBC@100 = fraction of queries with both golds inside the top-100  (S-recall@100, higher better)
      k*80    = smallest depth at which both golds are present for >= TAU (=0.80) of queries
                (nearest-rank TAU-percentile of the joint depth; right-censored at >1000)."""
    gp = claimE_metrics._first_ranks()[["model", "query_id", "first_same_rank", "first_cross_rank",
                                        "n_gold_same", "n_gold_cross"]]
    epo = fp.epo_first_ranks()[["model", "query_id", "first_same_rank", "first_cross_rank",
                               "n_gold_same", "n_gold_cross"]]
    dom = pd.concat([gp, epo], ignore_index=True)
    dom = dom[(dom["n_gold_same"] > 0) & (dom["n_gold_cross"] > 0)]
    def _pct(arr) -> int:
        a = np.sort(np.asarray(arr, float))
        v = a[min(int(np.ceil(TAU * len(a))) - 1, len(a) - 1)]
        return MAXRANK + 1 if not np.isfinite(v) else int(v)  # MAXRANK+1 sentinel = ">1000"

    out = {}
    for model in fp.MODEL_ORDER:
        g = dom[dom["model"] == model]
        joint = np.maximum(g["first_same_rank"], g["first_cross_rank"]).to_numpy(float)
        out[fp.short(model)] = {
            "jbc100": float(np.mean(joint <= 100)),
            "kjoint": _pct(joint),                       # depth for BOTH (max of the two)
            "ksame": _pct(g["first_same_rank"]),         # depth for the same-language gold
            "kcross": _pct(g["first_cross_rank"]),       # depth for the cross-language gold (foreign twin)
        }
    return out


def _balanced(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-model, macro-average over query languages (mean of per-language means)."""
    per = df.groupby(["short", "query_language"])[col].mean().reset_index()
    return per.groupby("short")[col].mean()


def compute() -> pd.DataFrame:
    gp = fp.cp_per_query()
    epo = fp.epo_per_query_clir()
    gp_ndcg = fp.model_comparison("chem_patents").set_index("short")["ndcg_at_10"]
    epo_ndcg = fp.model_comparison("epo").set_index("short")["ndcg_at_10"]

    gp_b = {c: _balanced(gp, c) for c in ("recall_at_10", "clir_at_10", "molir_at_10")}
    epo_b = {c: _balanced(epo, c) for c in ("recall_at_10", "clir_at_10", "molir_at_10")}

    jb = _joint_bilingual()
    rows = []
    for m in fp.MODEL_ORDER:
        s = fp.short(m)
        b = jb[s]
        rows.append({
            "model": s,
            "gp_ndcg": gp_ndcg[s], "gp_recall": gp_b["recall_at_10"][s], "gp_clir": gp_b["clir_at_10"][s],
            "gp_lt": 100 * gp_b["clir_at_10"][s] / gp_b["molir_at_10"][s],
            "epo_ndcg": epo_ndcg[s], "epo_recall": epo_b["recall_at_10"][s], "epo_clir": epo_b["clir_at_10"][s],
            "epo_lt": 100 * epo_b["clir_at_10"][s] / epo_b["molir_at_10"][s],
            "jbc100": b["jbc100"],     # joint bilingual gold coverage @100 (higher better), combined GP+EPO
            "kstar80": b["kjoint"],    # joint bilingual coverage depth @ tau=0.80 (both golds; lower better)
            "k_same": b["ksame"],      # same-language gold coverage depth @ tau=0.80 (lower better)
            "k_cross": b["kcross"],    # cross-language gold coverage depth @ tau=0.80 (lower better)
            "degenerate": gp_b["clir_at_10"][s] < DEG_CLIR,
        })
    df = pd.DataFrame(rows).sort_values("gp_ndcg", ascending=False).reset_index(drop=True)
    return df


def _fmt_kstar(v: int, best: bool) -> str:
    s = r"$>$1000" if v > MAXRANK else f"{v}"
    return f"\\textbf{{{s}}}" if best else s


# ---- LaTeX rendering -------------------------------------------------------
_COLS = ["gp_ndcg", "gp_recall", "gp_clir", "gp_lt", "epo_ndcg", "epo_recall", "epo_clir", "epo_lt"]
_HIGHER_BETTER = {c: True for c in _COLS}


def _fmt(v: float, col: str, best: bool) -> str:
    s = f"{v:.0f}" if col.endswith("_lt") else f"{v:.2f}"
    return f"\\textbf{{{s}}}" if best else s


def to_latex(df: pd.DataFrame) -> str:
    best = {c: df[c].max() for c in _COLS}
    best_jbc = df["jbc100"].max()       # higher better
    best_kj = df["kstar80"].min()       # lower better
    best_ks = df["k_same"].min()
    best_kc = df["k_cross"].min()
    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{l cccc cccc cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Google Patents} (524q, en/de/es/fr/zh)}"
        r" & \multicolumn{4}{c}{\textbf{EPO} (198q, en/de/fr)}"
        r" & \multicolumn{4}{c}{\textbf{Bilingual coverage} (both, $n{=}459$)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        r"Model & nDCG & R@10 & CLIR & LT\% & nDCG & R@10 & CLIR & LT\%"
        r" & JBC@100 & $k^\star_{80}$ & $k_{\mathrm{same}}$ & $k_{\mathrm{cross}}$ \\",
        r"\midrule",
    ]
    for r in df.itertuples():
        name = r.model + (r"$^\dagger$" if r.degenerate else "")
        cells = [_fmt(getattr(r, c), c, abs(getattr(r, c) - best[c]) < 1e-9) for c in _COLS]
        cells.append(_fmt(r.jbc100, "jbc", abs(r.jbc100 - best_jbc) < 1e-9))
        cells.append(_fmt_kstar(r.kstar80, r.kstar80 == best_kj))
        cells.append(_fmt_kstar(r.k_same, r.k_same == best_ks))
        cells.append(_fmt_kstar(r.k_cross, r.k_cross == best_kc))
        lines.append(f"\\texttt{{{name}}} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Retrieval performance on both patent benchmarks. \textbf{nDCG}@10 and \textbf{R}@10 "
        r"(Recall@10) measure IR power; \textbf{CLIR}@10 is Recall@10 over cross-language gold and "
        r"\textbf{LT}\% $=$ CLIR@10$/$MoLIR@10 is the share of same-language recall retained "
        r"cross-lingually (cross-linguality). R@10/CLIR/LT are language-balanced (mean of per-language "
        r"means). The two \textbf{Bilingual} columns treat language as a retrieval facet "
        r"(subtopic recall, \citealp{zhai2003subtopic}) and require a relevant document in \emph{both} "
        r"the query language and another language, over the GP+EPO queries that have both gold types "
        r"($n{=}459$): \textbf{JBC@100} (higher better) is the fraction of queries whose top-100 holds "
        r"both a same-language and a cross-language gold, and $\boldsymbol{k^\star_{80}}$ (lower better) "
        r"is the smallest depth at which both are present for $\geq$80\% of queries ($>$1000 $=$ never "
        r"reaches 80\% within the retrieved list). $k_{\mathrm{same}}$ and $k_{\mathrm{cross}}$ are the "
        r"same decomposed into the same-language and cross-language halves (the 80\%-coverage depth of "
        r"each gold alone): a small $k_{\mathrm{same}}$ with a censored $k_{\mathrm{cross}}$ marks "
        r"language-siloing (the model finds the home copy fast but never the foreign twin), whereas both "
        r"large marks a generally weak retriever. The degenerate \texttt{gte-multilingual-base} "
        r"(a loading artifact) is excluded; $^\dagger$\texttt{e5-large-instruct} fails the "
        r"CLIR@10$<$0.10 cross-lingual degeneracy gate. Best per column in bold.}",
        r"\label{tab:main}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def main():
    df = compute()
    TABLES.mkdir(parents=True, exist_ok=True)
    fp.dump_data(df, "main_table")
    tex = to_latex(df)
    (TABLES / "main_table.tex").write_text(tex + "\n")

    pd.set_option("display.width", 185)
    show = df.copy()
    for c in ("gp_lt", "epo_lt"):
        show[c] = show[c].round(0)
    for c in ("kstar80", "k_same", "k_cross"):
        show[c] = show[c].map(lambda v: ">1000" if v > MAXRANK else str(int(v)))
    print(show.round(3).to_string(index=False))
    rho = df["gp_ndcg"].corr(df["epo_ndcg"], method="spearman")
    print(f"\nSpearman(nDCG@10) GP vs EPO = {rho:.3f}")
    print(f"mean cross-lingual tax 1-LT: GP {100 - df['gp_lt'].mean():.0f}%  EPO {100 - df['epo_lt'].mean():.0f}%")
    print(f"wrote {TABLES / 'main_table.tex'}")


if __name__ == "__main__":
    main()
