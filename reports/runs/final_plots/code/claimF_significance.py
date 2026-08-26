"""
Claim F: confidence intervals + significance tests for the primary Recall@10 comparisons,
and per-route counts.  This module exists to answer the reviewer:

  "The test set is small and the paper reports no confidence intervals or significance tests for the
   primary Recall@10 comparisons. Per-route counts are also absent, despite substantial imbalance in
   language availability."

Everything is a re-aggregation of the cached per-(model, query) arrays already produced by the eval
pipeline (fp_common.cp_per_query / epo_per_query_clir).  No embedding model is ever loaded; no network
is touched; the only new dependency is scipy (already installed).

Statistical design (locked by an independent methodology panel; see reports/runs/final_plots/README):
  * Sampling unit  = query.  Headline estimator = language-balanced macro mean theta = (1/L) sum_l mean_l.
  * Cluster        = source patent family (GP: _patent_of(corpus_id); EPO: publication_number).  Two
                     queries in one family share their entire gold set -> they are resampled together.
                     Verified sizes: GP 412 families over 524 queries {1:376,2:9,3:1,4:5,5:19,6:2};
                     EPO 166 families over 198 {1:150,3:16} (language twins).  Clustering is load-bearing.
  * CIs            = language-stratified, family-clustered bootstrap of the macro mean; BCa (percentile
                     cross-check), B=10000.  Single proportions (per-language MoLIR, per-route recall)
                     use the Wilson score interval, never Wald, never a bootstrap of a ~13-Bernoulli mean.
  * Cross-lingual tax (MoLIR@10 vs CLIR@10): paired on the BOTH-GOLD domain only (GP 261 / EPO 198).
                     Effect = balanced-mean gap + cluster-bootstrap CI; p = family-level sign-flip
                     permutation test (Smucker et al. 2007), Holm across the 8 models.  Effect sizes:
                     raw gap+CI, win/tie/loss, matched-pairs rank-biserial.  Wilcoxon + sign test as
                     concordance cross-checks.
  * Leaderboard    : Friedman omnibus (blocked by query) gate -> paired family sign-flip permutation +
                     cluster-bootstrap CI for winner-vs-rest and adjacent-rank families, Holm within
                     each 7-test family; full 28-pair matrix (Benjamini-Hochberg) in the appendix csv.
  * Per-route      : route = (query_lang -> doc_lang); n_queries (== n_gold_pairs, <=1 version/lang) and
                     route Recall@10 + Wilson CI, diagonal = same-language, off-diagonal = cross-lingual.
  * Global tax     : cluster-robust (family) binomial GLM on gold-pair hits, hit ~ route_type + language,
                     odds ratio for same-vs-cross-language gold retrieval.  Hand-rolled IRLS + CR1 sandwich (no
                     statsmodels dependency); validated against the closed-form no-covariate odds ratio.
  * p-value policy : p-values only for (a) the per-model tax test and (b) full-benchmark model
                     comparisons.  Every per-language / per-route breakdown is descriptive (N + CI, no p).
                     Numeric backstop: no p when paired n<30 or #families<10 or #non-ties<10.

Outputs: data/F_*.csv (every plotted/tabulated number) and tables/F_*.tex (booktabs) + a printed summary.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import fp_common as fp

TABLES = fp.FINAL / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- config
B_BOOT = 10_000
B_PERM = 10_000
ALPHA = 0.05
SEED_ROOT = 20260724
# numeric p-value backstop (methodology pitfall #7): below any of these -> descriptive only
MIN_PAIRED_N = 30
MIN_FAMILIES = 10
MIN_NONTIE = 10
DEGENERATE = fp.DEGENERATE_CROSS  # e5-large-instruct: its "tax" is degenerate CLIR, not the real gap


# =========================================================================== per-query assembly
def _gp_pq() -> pd.DataFrame:
    """GP per-(model, query): +family (source patent), +has_same/has_cross masks."""
    pq = fp.cp_per_query().copy()
    fam = fp.gp_query_table().set_index("query_id")["corpus_id"].map(fp._patent_of)
    pq["family"] = pq["query_id"].map(fam)
    pq["has_same"] = pq["n_gold_same"] > 0
    pq["has_cross"] = pq["n_gold_cross"] > 0
    return pq


def _epo_pq() -> pd.DataFrame:
    """EPO per-(model, query): +family (publication_number), +has_same/has_cross masks."""
    pq = fp.epo_per_query_clir().copy()
    _, _, pub_of = fp._epo_corpus_family()
    stem = pq["query_id"].map(lambda q: re.sub(r"_q_[a-z]{2}$", "", q))
    pq["family"] = stem.map(lambda c: pub_of.get(c))
    gold = fp.epo_gold()
    pq["has_same"] = pq["query_id"].map(lambda q: len(gold[q][0]) > 0)
    pq["has_cross"] = pq["query_id"].map(lambda q: len(gold[q][1]) > 0)
    return pq


BENCH = {"GP": (_gp_pq, fp.CP_LANGS), "EPO": (_epo_pq, fp.EPO_LANGS)}


def _domain_queries(pq: pd.DataFrame, domain: str) -> pd.DataFrame:
    """One row per query in a domain: query_id, query_language, family (sorted, deterministic).
    domain: 'recall' (any gold), 'clir' (has cross gold), 'both' (has same AND cross gold)."""
    q = pq.drop_duplicates("query_id")[
        ["query_id", "query_language", "family", "has_same", "has_cross"]].copy()
    if domain == "clir":
        q = q[q["has_cross"]]
    elif domain == "both":
        q = q[q["has_same"] & q["has_cross"]]
    # 'recall' keeps all (every query in cp_per_query already has >=1 gold)
    return q.sort_values("query_id").reset_index(drop=True)


# =========================================================================== clustered resampler
class ClusterResampler:
    """Language-stratified, family-clustered bootstrap + family sign-flip permutation over a fixed set
    of queries (a benchmark x domain).  Draws are generated ONCE and reused across every model, metric
    and paired difference, so every comparison is paired by construction (methodology pitfall #4)."""

    def __init__(self, domain_q: pd.DataFrame, langs, seed: int):
        self.q = domain_q.reset_index(drop=True)
        self.langs = [l for l in langs if l in set(self.q["query_language"])]
        self.qid_order = self.q["query_id"].to_numpy()
        self.rng = np.random.default_rng(seed)
        # global family list (for jackknife + sign flips)
        self.fam_global = self.q["family"].to_numpy()
        self.fam_ids = pd.unique(self.fam_global)
        self.fam_pos = {f: i for i, f in enumerate(self.fam_ids)}
        self.n_fam = len(self.fam_ids)
        # per-language: family list (positions into fam_ids) and row indices grouped by family
        self.lang_fams = {}       # lang -> array of global family positions present in lang
        self.lang_fam_rows = {}   # lang -> list (aligned) of row-index arrays for each family
        for l in self.langs:
            rows = np.where(self.q["query_language"].to_numpy() == l)[0]
            byfam = defaultdict(list)
            for r in rows:
                byfam[self.fam_global[r]].append(r)
            fams = list(byfam)
            self.lang_fams[l] = np.array([self.fam_pos[f] for f in fams])
            self.lang_fam_rows[l] = [np.array(byfam[f]) for f in fams]
        # bootstrap draws: per lang, (B, n_fam_l) family-index positions (into that lang's family list)
        self.boot_draws = {l: self.rng.integers(0, len(self.lang_fams[l]), size=(B_BOOT, len(self.lang_fams[l])))
                           for l in self.langs}
        # sign-flip matrix: (B_PERM, n_fam_global) in {+1,-1}, one sign per global family
        self.signs = self.rng.choice(np.array([-1.0, 1.0]), size=(B_PERM, self.n_fam))

    # ---- helpers to turn a per-query value Series into aligned array over this domain ----
    def align(self, series: pd.Series) -> np.ndarray:
        return series.reindex(self.qid_order).to_numpy(dtype=float)

    # ---- per-language family sums/counts for a value array (NaN dropped) ----
    def _lang_sc(self, vals: np.ndarray, lang: str):
        s = np.empty(len(self.lang_fam_rows[lang]))
        c = np.empty(len(self.lang_fam_rows[lang]))
        for i, rows in enumerate(self.lang_fam_rows[lang]):
            v = vals[rows]
            m = ~np.isnan(v)
            s[i] = v[m].sum()
            c[i] = m.sum()
        return s, c

    def point_macro(self, vals: np.ndarray) -> float:
        means = []
        for l in self.langs:
            s, c = self._lang_sc(vals, l)
            tot = c.sum()
            if tot > 0:
                means.append(s.sum() / tot)
        return float(np.mean(means)) if means else float("nan")

    def boot_dist(self, vals: np.ndarray) -> np.ndarray:
        """B-vector of bootstrap macro means (language-stratified family resampling)."""
        per_lang = []
        for l in self.langs:
            s, c = self._lang_sc(vals, l)
            d = self.boot_draws[l]                 # (B, n_fam_l)
            num = s[d].sum(axis=1)
            den = c[d].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_lang.append(np.where(den > 0, num / den, np.nan))
        M = np.vstack(per_lang)                    # (L, B)
        return np.nanmean(M, axis=0)

    def _jackknife_macro(self, vals: np.ndarray) -> np.ndarray:
        """Leave-one-family-out macro means (for BCa acceleration)."""
        # precompute per-lang family sums/counts and totals
        Sl, Cl, lang_of_fam = {}, {}, defaultdict(list)
        for l in self.langs:
            s, c = self._lang_sc(vals, l)
            Sl[l] = (s, c, s.sum(), c.sum())
            for i, gpos in enumerate(self.lang_fams[l]):
                lang_of_fam[gpos].append((l, i))
        base = {l: (Sl[l][2] / Sl[l][3] if Sl[l][3] > 0 else np.nan) for l in self.langs}
        out = np.empty(self.n_fam)
        for gpos in range(self.n_fam):
            means = []
            touched = dict(lang_of_fam.get(gpos, []))
            for l in self.langs:
                s, c, S, C = Sl[l]
                if l in touched:
                    i = touched[l]
                    S2, C2 = S - s[i], C - c[i]
                    means.append(S2 / C2 if C2 > 0 else np.nan)
                else:
                    means.append(base[l])
            out[gpos] = np.nanmean(means)
        return out

    def ci(self, vals: np.ndarray, method: str = "bca"):
        """(point, lo, hi, method_used) 95% CI of the macro mean."""
        point = self.point_macro(vals)
        boot = self.boot_dist(vals)
        boot = boot[np.isfinite(boot)]
        if boot.size < 100:
            return point, float("nan"), float("nan"), "insufficient"
        lo_p, hi_p = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        if method == "percentile":
            return point, float(lo_p), float(hi_p), "percentile"
        # BCa
        z0_frac = np.mean(boot < point)
        if z0_frac <= 0 or z0_frac >= 1:
            return point, float(lo_p), float(hi_p), "percentile(z0-degenerate)"
        z0 = stats.norm.ppf(z0_frac)
        jk = self._jackknife_macro(vals)
        jbar = np.nanmean(jk)
        num = np.nansum((jbar - jk) ** 3)
        den = 6.0 * (np.nansum((jbar - jk) ** 2) ** 1.5)
        a = num / den if den != 0 else 0.0
        if not np.isfinite(a):
            return point, float(lo_p), float(hi_p), "percentile(a-nonfinite)"
        zl, zu = stats.norm.ppf(ALPHA / 2), stats.norm.ppf(1 - ALPHA / 2)
        a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
        a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
        if not (np.isfinite(a1) and np.isfinite(a2)) or a1 <= 0 or a2 >= 1 or a1 >= a2:
            return point, float(lo_p), float(hi_p), "percentile(bca-fallback)"
        lo, hi = np.percentile(boot, [100 * a1, 100 * a2])
        return point, float(lo), float(hi), "bca"

    def perm_p(self, diff_vals: np.ndarray) -> tuple[float, float]:
        """Two-sided family sign-flip permutation p for macro-mean(diff)=0.  Returns (T_obs, p)."""
        # per-lang: family diff-sums (sd) and counts (C_l fixed); flipped mean = signs.sd / C_l
        T_obs = self.point_macro(diff_vals)
        lang_sd, lang_gpos, lang_C = [], [], []
        for l in self.langs:
            s, c = self._lang_sc(diff_vals, l)
            lang_sd.append(s)
            lang_gpos.append(self.lang_fams[l])
            lang_C.append(c.sum())
        Tstar = np.zeros(B_PERM)
        valid = 0
        for sd, gpos, C in zip(lang_sd, lang_gpos, lang_C):
            if C <= 0:
                continue
            valid += 1
            Tstar += (self.signs[:, gpos] * sd).sum(axis=1) / C
        Tstar /= max(valid, 1)
        p = (1 + np.sum(np.abs(Tstar) >= abs(T_obs) - 1e-12)) / (B_PERM + 1)
        return float(T_obs), float(p)


# =========================================================================== small helpers
def wilson_ci(k: int, n: int, alpha: float = ALPHA):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, float(center - half), float(center + half)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (order preserved to input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def rank_biserial(diff: np.ndarray):
    """Matched-pairs rank-biserial r = (W+ - W-)/(W+ + W-) among discordant pairs (Kerby 2014)."""
    w = int(np.sum(diff > 0)); l = int(np.sum(diff < 0)); t = int(np.sum(diff == 0))
    denom = w + l
    r = (w - l) / denom if denom else float("nan")
    return w, t, l, r


# =========================================================================== cluster-robust logit (T5)
def cr_logit(y, X, groups, ridge=1e-8):
    """Logistic regression via IRLS + CR1 cluster-robust (sandwich) covariance.
    Returns (beta, se_cluster).  No statsmodels dependency; validated against the 2x2 odds ratio."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        H = XtW @ X + ridge * np.eye(p)
        beta_new = np.linalg.solve(H, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new; break
        beta = beta_new
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    bread = np.linalg.inv((X.T * w) @ X + ridge * np.eye(p))
    u = X * (y - mu)[:, None]                    # score contributions
    meat = np.zeros((p, p))
    gidx = pd.factorize(groups)[0]
    G = gidx.max() + 1
    for g in range(G):
        ug = u[gidx == g].sum(axis=0)
        meat += np.outer(ug, ug)
    cr1 = (G / (G - 1)) * ((n - 1) / (n - p))     # CR1 small-sample scaling
    cov = bread @ (cr1 * meat) @ bread
    return beta, np.sqrt(np.diag(cov)), G


# =========================================================================== gold-pair (route) frame
def _route_frame(bench: str) -> pd.DataFrame:
    """One row per (model, query, doc_language) gold pair: hit in {0,1}, route_type, family, langs.
    Each patent has <=1 version per language so a (query, doc_language) cell holds exactly one gold doc,
    and route Recall@10 == hit."""
    if bench == "GP":
        clang, _ = fp._gp_corpus_family()
        tab = fp.gp_query_table()
        rk = fp.gp_rankings()
        top = rk[rk["rank"] <= 10].groupby(["model", "query_id"])["corpus_id"].apply(set).to_dict()
        langs = fp.CP_LANGS
        rows = []
        for model in fp.MODEL_ORDER:
            for q in tab.itertuples():
                t = top.get((model, q.query_id), set())
                by = defaultdict(set)
                for d in q.gold:
                    by[clang.get(d)].add(d)
                for dl, gset in by.items():
                    if dl in langs:
                        rows.append({"model": model, "short": fp.short(model), "query_id": q.query_id,
                                     "query_language": q.query_language, "doc_language": dl,
                                     "family": fp._patent_of(q.corpus_id),
                                     "hit": 1.0 if (t & gset) else 0.0})
        return pd.DataFrame(rows)
    else:
        clang, _, pub_of = fp._epo_corpus_family()
        gold = fp.epo_gold()
        rk = fp.epo_rankings()
        top = rk[rk["rank"] <= 10].groupby(["model", "query_id"])["corpus_id"].apply(set).to_dict()
        meta = rk.drop_duplicates("query_id")[["query_id", "query_language"]]
        langs = fp.EPO_LANGS
        rows = []
        for r in meta.itertuples():
            same, cross = gold[r.query_id]
            stem = re.sub(r"_q_[a-z]{2}$", "", r.query_id)
            by = defaultdict(set)
            for d in (same | cross):
                by[clang.get(d)].add(d)
            for model in fp.MODEL_ORDER:
                t = top.get((model, r.query_id), set())
                for dl, gset in by.items():
                    if dl in langs:
                        rows.append({"model": model, "short": fp.short(model), "query_id": r.query_id,
                                     "query_language": r.query_language, "doc_language": dl,
                                     "family": pub_of.get(stem),
                                     "hit": 1.0 if (t & gset) else 0.0})
        return pd.DataFrame(rows)


# =========================================================================== T1: headline CIs
METRICS = {"recall_at_10": "recall", "clir_at_10": "clir", "molir_at_10": "both"}  # col -> domain


def t1_headline_ci():
    rows = []
    for bench, (loader, langs) in BENCH.items():
        pq = loader()
        seeds = np.random.SeedSequence(SEED_ROOT).spawn(4)
        resamplers = {dom: ClusterResampler(_domain_queries(pq, dom), langs, int(seeds[i].generate_state(1)[0]))
                      for i, dom in enumerate(["recall", "clir", "both"])}
        for m in fp.MODEL_ORDER:
            sub = pq[pq["model"] == m]
            rec = {}
            for col, dom in METRICS.items():
                R = resamplers[dom]
                vals = R.align(sub.set_index("query_id")[col])
                point, lo, hi, meth = R.ci(vals)
                rec[col] = (point, lo, hi, meth)
            # LT% = 100*CLIR/MoLIR on the BOTH domain with paired draws
            Rb = resamplers["both"]
            clir_b = Rb.align(sub.set_index("query_id")["clir_at_10"])
            molir_b = Rb.align(sub.set_index("query_id")["molir_at_10"])
            lt_point = 100 * Rb.point_macro(clir_b) / Rb.point_macro(molir_b)
            bc = Rb.boot_dist(clir_b); bm = Rb.boot_dist(molir_b)
            with np.errstate(invalid="ignore", divide="ignore"):
                lt_boot = 100 * bc / bm
            lt_boot = lt_boot[np.isfinite(lt_boot)]
            lt_lo, lt_hi = np.percentile(lt_boot, [2.5, 97.5])
            rows.append({
                "benchmark": bench, "model": fp.short(m),
                "recall": rec["recall_at_10"][0], "recall_lo": rec["recall_at_10"][1], "recall_hi": rec["recall_at_10"][2],
                "clir": rec["clir_at_10"][0], "clir_lo": rec["clir_at_10"][1], "clir_hi": rec["clir_at_10"][2],
                "molir": rec["molir_at_10"][0], "molir_lo": rec["molir_at_10"][1], "molir_hi": rec["molir_at_10"][2],
                "lt_pct": lt_point, "lt_lo": lt_lo, "lt_hi": lt_hi,
                "ci_method": rec["recall_at_10"][3],
            })
    df = pd.DataFrame(rows)
    fp.dump_data(df, "F_t1_headline_ci")
    return df


# =========================================================================== T2: cross-lingual tax
def t2_tax():
    rows = []
    for bench, (loader, langs) in BENCH.items():
        pq = loader()
        seed = int(np.random.SeedSequence(SEED_ROOT + 1).spawn(1)[0].generate_state(1)[0])
        R = ClusterResampler(_domain_queries(pq, "both"), langs, seed)
        # assert paired-domain per-language counts (methodology pitfall #1)
        counts = R.q["query_language"].value_counts().to_dict()
        model_rows = []
        for m in fp.MODEL_ORDER:
            sub = pq[pq["model"] == m].set_index("query_id")
            molir = R.align(sub["molir_at_10"]); clir = R.align(sub["clir_at_10"])
            diff = molir - clir
            gap_point, lo, hi, meth = R.ci(diff)
            Tobs, p_perm = R.perm_p(diff)
            w, t, l, r = rank_biserial(diff[~np.isnan(diff)])
            n_pair = int(np.sum(~np.isnan(diff)))
            # concordance cross-checks
            dd = diff[~np.isnan(diff)]
            try:
                p_wil = stats.wilcoxon(dd, zero_method="pratt", alternative="two-sided").pvalue if (w + l) else np.nan
            except ValueError:
                p_wil = np.nan
            p_sign = stats.binomtest(w, w + l, 0.5).pvalue if (w + l) else np.nan
            suppress = (n_pair < MIN_PAIRED_N) or (R.n_fam < MIN_FAMILIES) or ((w + l) < MIN_NONTIE)
            model_rows.append({
                "benchmark": bench, "model": fp.short(m), "degenerate": m in DEGENERATE,
                "molir": R.point_macro(molir), "clir": R.point_macro(clir),
                "gap": gap_point, "gap_lo": lo, "gap_hi": hi, "ci_method": meth,
                "n_pair": n_pair, "n_fam": R.n_fam, "win": w, "tie": t, "loss": l, "rank_biserial": r,
                "p_perm_raw": p_perm if not suppress else np.nan,
                "p_wilcoxon": p_wil, "p_sign": p_sign, "p_suppressed": suppress,
            })
        # Holm across the (non-suppressed) models within this benchmark
        mr = pd.DataFrame(model_rows)
        mask = ~mr["p_perm_raw"].isna()
        mr.loc[mask, "p_perm_holm"] = holm(mr.loc[mask, "p_perm_raw"].to_numpy())
        mr["paired_lang_counts"] = str(counts)
        rows.append(mr)
    df = pd.concat(rows, ignore_index=True)
    fp.dump_data(df, "F_t2_tax")
    return df


# =========================================================================== T3: leaderboard significance
def _friedman(pq: pd.DataFrame, col: str, domain: str, langs) -> dict:
    q = _domain_queries(pq, domain)["query_id"].to_numpy()
    mat = (pq.pivot_table(index="query_id", columns="model", values=col)
             .reindex(index=q, columns=fp.MODEL_ORDER))
    mat = mat.dropna(axis=0, how="any")
    arrs = [mat[m].to_numpy() for m in fp.MODEL_ORDER]
    chi, p = stats.friedmanchisquare(*arrs)
    return {"col": col, "domain": domain, "n_block": len(mat), "chi2": chi, "df": len(fp.MODEL_ORDER) - 1, "p": p}


def t3_leaderboard():
    friedman_rows, pair_rows = [], []
    for bench, (loader, langs) in BENCH.items():
        pq = loader()
        for col, dom in [("recall_at_10", "recall"), ("clir_at_10", "clir")]:
            fr = _friedman(pq, col, dom, langs)
            fr["benchmark"] = bench
            friedman_rows.append(fr)
            seed = int(np.random.SeedSequence(SEED_ROOT + 2).spawn(1)[0].generate_state(1)[0])
            R = ClusterResampler(_domain_queries(pq, dom), langs, seed)
            # rank models by point macro on this metric.  NOTE: the winner is chosen by point estimate on
            # the same data, so winner-vs-rest is post-selection (winner's-curse) and its p-values are
            # anticonservative in principle; here it is immaterial (Friedman gate p<1e-60, winner leads by
            # >=0.06 with CIs far from 0). The adjacent-rank family below is the selection-free ordering test.
            pts = {m: R.point_macro(R.align(pq[pq["model"] == m].set_index("query_id")[col])) for m in fp.MODEL_ORDER}
            order = sorted(fp.MODEL_ORDER, key=lambda m: pts[m], reverse=True)
            winner = order[0]
            families = {"winner_vs_rest": [(winner, o) for o in order[1:]],
                        "adjacent": [(order[i], order[i + 1]) for i in range(len(order) - 1)]}
            for fam_name, pairs in families.items():
                raws, recs = [], []
                for A, Bm in pairs:
                    a = R.align(pq[pq["model"] == A].set_index("query_id")[col])
                    b = R.align(pq[pq["model"] == Bm].set_index("query_id")[col])
                    diff = a - b
                    d_point, lo, hi, meth = R.ci(diff)
                    _, p_perm = R.perm_p(diff)
                    w, t, l, rb = rank_biserial(diff[~np.isnan(diff)])
                    raws.append(p_perm)
                    recs.append({"benchmark": bench, "metric": col, "family": fam_name,
                                 "A": fp.short(A), "B": fp.short(Bm), "diff": d_point,
                                 "diff_lo": lo, "diff_hi": hi, "p_raw": p_perm,
                                 "win": w, "tie": t, "loss": l})
                adj = holm(raws)
                for rc, a in zip(recs, adj):
                    rc["p_holm"] = a
                pair_rows.extend(recs)
            # exploratory full 28-pair matrix (BH)
            allp, allrec = [], []
            for i in range(len(fp.MODEL_ORDER)):
                for j in range(i + 1, len(fp.MODEL_ORDER)):
                    A, Bm = fp.MODEL_ORDER[i], fp.MODEL_ORDER[j]
                    diff = R.align(pq[pq["model"] == A].set_index("query_id")[col]) - \
                           R.align(pq[pq["model"] == Bm].set_index("query_id")[col])
                    d_point, lo, hi, _ = R.ci(diff)
                    _, p_perm = R.perm_p(diff)
                    allp.append(p_perm)
                    allrec.append({"benchmark": bench, "metric": col, "family": "all_pairs_BH",
                                   "A": fp.short(A), "B": fp.short(Bm), "diff": d_point,
                                   "diff_lo": lo, "diff_hi": hi, "p_raw": p_perm})
            adj = benjamini_hochberg(allp)
            for rc, a in zip(allrec, adj):
                rc["p_bh"] = a
            pair_rows.extend(allrec)
    fdf = pd.DataFrame(friedman_rows)
    pdf = pd.DataFrame(pair_rows)
    fp.dump_data(fdf, "F_t3_friedman")
    fp.dump_data(pdf, "F_t3_pairwise")
    return fdf, pdf


# =========================================================================== T4: per-route counts
def t4_routes():
    rows = []
    for bench, (loader, langs) in BENCH.items():
        rf = _route_frame(bench)
        one = rf[rf["model"] == fp.MODEL_ORDER[0]]  # counts are model-independent
        for q in langs:
            for d in langs:
                cell = one[(one["query_language"] == q) & (one["doc_language"] == d)]
                n = len(cell)
                if n == 0:
                    continue
                rec = {"benchmark": bench, "query_language": q, "doc_language": d,
                       "kind": "same" if q == d else "cross", "n_queries": n, "n_gold_pairs": n}
                # route recall + Wilson CI for all-model mean, embeddinggemma, e5-large-instruct
                for label, m in [("mean", None), ("embeddinggemma", "google/embeddinggemma-300m"),
                                 ("e5", "intfloat/multilingual-e5-large-instruct")]:
                    if m is None:
                        sub = rf[(rf["query_language"] == q) & (rf["doc_language"] == d)]
                        rec[f"recall_{label}"] = float(sub["hit"].mean())
                    else:
                        sub = rf[(rf["model"] == m) & (rf["query_language"] == q) & (rf["doc_language"] == d)]
                        k = int(sub["hit"].sum())
                        p, lo, hi = wilson_ci(k, len(sub))
                        rec[f"recall_{label}"] = p
                        rec[f"recall_{label}_lo"] = lo
                        rec[f"recall_{label}_hi"] = hi
                rows.append(rec)
    df = pd.DataFrame(rows)
    fp.dump_data(df, "F_t4_routes")
    return df


# =========================================================================== T5: global tax GLM
def t5_global_tax():
    rows = []
    for bench, (loader, langs) in BENCH.items():
        rf = _route_frame(bench)
        rf["route_same"] = (rf["query_language"] == rf["doc_language"]).astype(float)  # 1=same,0=cross
        lang_dummies = pd.get_dummies(rf["query_language"], prefix="ql", drop_first=True).astype(float)
        for m in fp.MODEL_ORDER:
            sub = rf[rf["model"] == m].reset_index(drop=True)
            y = sub["hit"].to_numpy()
            Xd = lang_dummies.loc[sub.index]
            X = np.column_stack([np.ones(len(sub)), sub["route_same"].to_numpy(), Xd.to_numpy()])
            names = ["const", "route_same"] + list(Xd.columns)
            try:
                beta, se, G = cr_logit(y, X, sub["family"].to_numpy())
                bi = names.index("route_same")
                or_ = float(np.exp(beta[bi]))
                z = beta[bi] / se[bi]
                p = float(2 * stats.norm.sf(abs(z)))
                lo = float(np.exp(beta[bi] - 1.96 * se[bi])); hi = float(np.exp(beta[bi] + 1.96 * se[bi]))
            except Exception as e:  # noqa
                or_, lo, hi, p, G = float("nan"), float("nan"), float("nan"), float("nan"), 0
            rows.append({"benchmark": bench, "model": fp.short(m), "degenerate": m in DEGENERATE,
                         "OR_same_vs_cross": or_, "OR_lo": lo, "OR_hi": hi, "p_clustered": p, "n_clusters": G})
    df = pd.DataFrame(rows)
    fp.dump_data(df, "F_t5_global_tax")
    return df


# =========================================================================== LaTeX rendering
def _tex_ci(p, lo, hi, dec=2):
    return f"{p:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]"


def write_latex(t1, t2, t4):
    # ---- F1: headline table with CIs (GP) ----
    def headline_tex(bench):
        d = t1[t1["benchmark"] == bench].copy()
        lines = [r"\begin{tabular}{lccc}", r"\toprule",
                 r"Model & R@10 [95\% CI] & CLIR@10 [95\% CI] & LT\% [95\% CI] \\", r"\midrule"]
        for r in d.itertuples():
            lines.append(f"\\texttt{{{r.model}}} & {_tex_ci(r.recall, r.recall_lo, r.recall_hi)} & "
                         f"{_tex_ci(r.clir, r.clir_lo, r.clir_hi)} & "
                         f"{r.lt_pct:.0f} [{r.lt_lo:.0f}, {r.lt_hi:.0f}] \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    (TABLES / "F_headline_ci_GP.tex").write_text(headline_tex("GP") + "\n")
    (TABLES / "F_headline_ci_EPO.tex").write_text(headline_tex("EPO") + "\n")

    # ---- F2: cross-lingual tax ----
    def tax_tex(bench):
        d = t2[t2["benchmark"] == bench]
        lines = [r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Model & MoLIR & CLIR & Gap $\Delta$ [95\% CI] & W/T/L & $p$ (Holm) \\", r"\midrule"]
        for r in d.itertuples():
            star = ""
            if not r.p_suppressed and not np.isnan(getattr(r, "p_perm_holm", np.nan)):
                ph = r.p_perm_holm
                star = ("$<$0.001" if ph < 1e-3 else f"{ph:.3f}")
            else:
                star = "--"
            name = r.model + (r"$^\dagger$" if r.degenerate else "")
            lines.append(f"\\texttt{{{name}}} & {r.molir:.2f} & {r.clir:.2f} & "
                         f"{r.gap:+.3f} [{r.gap_lo:+.3f}, {r.gap_hi:+.3f}] & "
                         f"{int(r.win)}/{int(r.tie)}/{int(r.loss)} & {star} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    (TABLES / "F_tax_GP.tex").write_text(tax_tex("GP") + "\n")
    (TABLES / "F_tax_EPO.tex").write_text(tax_tex("EPO") + "\n")

    # ---- F4: per-route count matrix (GP) ----
    def route_counts_tex(bench, langs):
        d = t4[t4["benchmark"] == bench]
        piv = d.pivot_table(index="query_language", columns="doc_language", values="n_queries",
                            fill_value=0).reindex(index=langs, columns=langs, fill_value=0).astype(int)
        header = " & ".join([r"\textbf{q$\downarrow$ / d$\rightarrow$}"] + [l.upper() for l in langs])
        lines = [r"\begin{tabular}{l" + "c" * len(langs) + "}", r"\toprule", header + r" \\", r"\midrule"]
        for q in langs:
            cells = []
            for dl in langs:
                v = int(piv.loc[q, dl])
                cells.append((r"\cellcolor{gray!15}" if q == dl else "") + str(v))
            lines.append(f"{q.upper()} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    (TABLES / "F_route_counts_GP.tex").write_text(route_counts_tex("GP", fp.CP_LANGS) + "\n")
    (TABLES / "F_route_counts_EPO.tex").write_text(route_counts_tex("EPO", fp.EPO_LANGS) + "\n")


# =========================================================================== figures
def fig_tax_forest(t2):
    """Forest plot of the cross-lingual tax (same - cross Recall@10) per model, GP + EPO, with 95% CIs.
    A CI that crosses 0 (no significant tax) is drawn hollow; the degenerate e5 is greyed."""
    import matplotlib.pyplot as plt
    fp.set_style()
    order = [fp.short(m) for m in fp.MODEL_ORDER]
    bcol = {"GP": "#1f77b4", "EPO": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    yticks, ylabels = [], []
    for i, m in enumerate(order):
        y0 = len(order) - i
        for j, bench in enumerate(["GP", "EPO"]):
            r = t2[(t2["benchmark"] == bench) & (t2["model"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = y0 + (0.18 if bench == "GP" else -0.18)
            crosses0 = (r["gap_lo"] <= 0 <= r["gap_hi"])
            degen = bool(r["degenerate"])
            if degen:                       # degenerate CLIR: greyed, not a real "tax"
                face, edge = "#c9c9c9", "#9a9a9a"
            elif crosses0:                  # tax not significant: hollow marker
                face, edge = "white", bcol[bench]
            else:                           # significant tax: filled
                face, edge = bcol[bench], bcol[bench]
            ax.errorbar(r["gap"], y, xerr=[[r["gap"] - r["gap_lo"]], [r["gap_hi"] - r["gap"]]],
                        fmt="o", ms=6, mfc=face, mec=edge, ecolor=edge, elinewidth=1.6, capsize=3, zorder=3)
        yticks.append(y0)
        ylabels.append(m + ("  †" if m == "e5-large-instruct" else ""))
    ax.axvline(0, color="#c44e52", lw=1.4, ls="--", zorder=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.set_xlabel(r"Cross-lingual tax  $\Delta = $ MoLIR@10 $-$ CLIR@10  (same $-$ cross)")
    ax.set_title("Same-language recall exceeds cross-language recall for almost every model\n"
                 "(paired, both-gold domain; 95% cluster-bootstrap CI)", fontsize=10.5)
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=bcol[b], mec=bcol[b], label=b) for b in ["GP", "EPO"]]
    handles += [
        plt.Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", label="CI crosses 0 (n.s.)"),
        plt.Line2D([0], [0], marker="o", ls="", mfc="#c9c9c9", mec="#9a9a9a", label=r"degenerate CLIR ($\dagger$)"),
        plt.Line2D([0], [0], color="#c44e52", ls="--", label=r"no tax ($\Delta=0$)")]
    ax.legend(handles=handles, loc="center right", fontsize=8.2, ncol=1, framealpha=0.9,
              frameon=True, edgecolor="#dddddd")
    ax.margins(y=0.06)
    fp.save(fig, "claimF_F2_tax_forest")


def fig_route_counts(t4):
    """GP route (query->doc language) count heatmap: n_queries per route, diagonal (same-language) outlined."""
    import matplotlib.pyplot as plt
    fp.set_style()
    langs = fp.CP_LANGS
    d = t4[t4["benchmark"] == "GP"]
    piv = (d.pivot_table(index="query_language", columns="doc_language", values="n_queries", fill_value=0)
             .reindex(index=langs, columns=langs, fill_value=0).astype(int))
    arr = piv.to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(arr, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(langs))); ax.set_yticks(range(len(langs)))
    ax.set_xticklabels([l.upper() for l in langs]); ax.set_yticklabels([l.upper() for l in langs])
    ax.set_xlabel("document language (gold)"); ax.set_ylabel("query language")
    vmax = arr.max()
    for i in range(len(langs)):
        for j in range(len(langs)):
            v = arr[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                    color="white" if v > vmax * 0.6 else "black",
                    fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#c44e52", lw=2.2))
    ax.set_title("Per-route counts: # queries with a gold document on each\n"
                 "query→document language route (Google Patents; red = same-language)", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="n queries (= n gold pairs)")
    fp.save(fig, "claimF_F4_route_counts")


# =========================================================================== main
def main():
    print("=" * 100)
    print("Claim F — CIs + significance tests for the primary Recall@10 comparisons, and per-route counts")
    print("=" * 100)
    t1 = t1_headline_ci()
    t2 = t2_tax()
    fdf, pdf = t3_leaderboard()
    t4 = t4_routes()
    t5 = t5_global_tax()
    write_latex(t1, t2, t4)
    fig_tax_forest(t2)
    fig_route_counts(t4)

    pd.set_option("display.width", 200)
    print("\n--- T1  headline Recall@10 / CLIR@10 with 95% CIs (language-balanced, cluster bootstrap) ---")
    show1 = t1.copy()
    for c in ["recall", "clir", "molir"]:
        show1[c] = show1.apply(lambda r, c=c: f"{r[c]:.3f} [{r[c+'_lo']:.3f},{r[c+'_hi']:.3f}]", axis=1)
    show1["LT%[CI]"] = show1.apply(lambda r: f"{r['lt_pct']:.0f} [{r['lt_lo']:.0f},{r['lt_hi']:.0f}]", axis=1)
    print(show1[["benchmark", "model", "recall", "clir", "LT%[CI]", "ci_method"]].to_string(index=False))
    print("  (note: LT% here is CLIR/MoLIR with BOTH on the paired both-gold domain; the paper's main_table"
          "\n   LT divides full-524 CLIR by 261-domain MoLIR, so those numbers differ by design.)")

    print("\n--- T2  cross-lingual tax: same (MoLIR) vs cross (CLIR) Recall@10, paired on both-gold domain ---")
    print("paired-language counts:", t2["paired_lang_counts"].iloc[0])
    show2 = t2.copy()
    show2["gap[CI]"] = show2.apply(lambda r: f"{r.gap:+.3f}[{r.gap_lo:+.3f},{r.gap_hi:+.3f}]", axis=1)
    show2["WTL"] = show2.apply(lambda r: f"{int(r.win)}/{int(r.tie)}/{int(r.loss)}", axis=1)
    print(show2[["benchmark", "model", "molir", "clir", "gap[CI]", "n_pair", "WTL",
                 "rank_biserial", "p_perm_holm", "p_suppressed"]].round(4).to_string(index=False))

    print("\n--- T3  Friedman omnibus (models differ?) ---")
    print(fdf[["benchmark", "col", "n_block", "chi2", "df", "p"]].to_string(index=False))
    print("\n--- T3  winner-vs-rest (GP, Recall@10) ---")
    wv = pdf[(pdf["benchmark"] == "GP") & (pdf["metric"] == "recall_at_10") & (pdf["family"] == "winner_vs_rest")]
    print(wv[["A", "B", "diff", "diff_lo", "diff_hi", "p_raw", "p_holm"]].round(4).to_string(index=False))
    print("\n--- T3  adjacent-rank (GP, Recall@10) ---")
    aj = pdf[(pdf["benchmark"] == "GP") & (pdf["metric"] == "recall_at_10") & (pdf["family"] == "adjacent")]
    print(aj[["A", "B", "diff", "diff_lo", "diff_hi", "p_raw", "p_holm"]].round(4).to_string(index=False))

    print("\n--- T4  per-route counts (GP): n_queries per (query_lang -> doc_lang) ---")
    piv = (t4[t4["benchmark"] == "GP"].pivot_table(index="query_language", columns="doc_language",
           values="n_queries", fill_value=0).reindex(index=fp.CP_LANGS, columns=fp.CP_LANGS, fill_value=0).astype(int))
    print(piv.to_string())
    print("same-lang (diagonal) routes total pairs:", int(np.trace(piv.to_numpy())),
          "| cross-lang total pairs:", int(piv.to_numpy().sum() - np.trace(piv.to_numpy())))

    print("\n--- T5  global cross-lingual tax: cluster-robust binomial GLM (OR same-vs-cross-language gold) ---")
    print(t5.round(4).to_string(index=False))

    print("\nAll CSVs -> reports/runs/final_plots/data/F_*.csv ; LaTeX -> reports/runs/final_plots/tables/F_*.tex")


if __name__ == "__main__":
    main()
