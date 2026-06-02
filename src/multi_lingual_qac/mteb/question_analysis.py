"""Question-level analysis of MTEB retrieval predictions.

Given per-query rankings saved by ``run_mteb_evaluation(..., prediction_dir=...)``
(MTEB ``prediction_folder`` JSON) plus the dataset's queries/corpus/qrels, this
breaks Recall@K / MRR@K down by:

  1. query language
  2. query origin: original vs synthetic-translation queries
  3. retrieval mode: same-language vs cross-language relevant targets
  4. query-language x target-language pair matrix (best model)

It is dataset-agnostic: each breakdown is skipped gracefully when the dataset
lacks the needed column (e.g. no ``is_synthetic_translation`` column, or a single
language), so it keeps working for other question sets / datasets.
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.multi_lingual_qac.mteb.evaluation import (
    DEFAULT_MTEB_DATASET_REPO,
    DEFAULT_MTEB_VARIANT,
    _dataset_config_name,
    _infer_language,
    _normalize_dataset_variant,
    _query_language_column,
    _slugify,
)

PREDICTION_GLOB = "*_predictions.json"
DEFAULT_K = 10


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _id_column(columns: list[str], specific: str) -> str:
    if "_id" in columns:
        return "_id"
    if specific in columns:
        return specific
    return columns[0] if columns else specific


def _corpus_language_column(columns: list[str]) -> str | None:
    for name in ("corpus_language", "language"):
        if name in columns:
            return name
    return None


def _load_predictions(model_dir: Path) -> dict[str, dict[str, float]] | None:
    files = glob.glob(str(model_dir / PREDICTION_GLOB))
    if not files:
        return None
    payload = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    subsets = [key for key in payload if key != "mteb_model_meta"]
    if not subsets:
        return None
    merged: dict[str, dict[str, float]] = {}
    for subset in subsets:
        for split_preds in payload[subset].values():
            merged.update(split_preds)
    return merged


def _discover_models(
    predictions_dir: Path, model_names: list[str] | None
) -> list[tuple[str, str]]:
    """Return [(label, slug)] pairs that actually have a predictions file."""
    if not predictions_dir.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    if model_names:
        for name in model_names:
            slug = _slugify(name)
            if (predictions_dir / slug).is_dir():
                pairs.append((name, slug))
    else:
        for child in sorted(p for p in predictions_dir.iterdir() if p.is_dir()):
            pairs.append((child.name, child.name))
    return [(label, slug) for label, slug in pairs if glob.glob(str(predictions_dir / slug / PREDICTION_GLOB))]


def _per_query_metrics(preds, rel, query_lang, corpus_lang, k):
    out = {}
    for qid, rel_set in rel.items():
        if qid not in preds or not rel_set:
            continue
        ranking = [doc for doc, _ in sorted(preds[qid].items(), key=lambda kv: -kv[1])]
        top = set(ranking[:k])
        rr = 0.0
        for rank, doc in enumerate(ranking[:k], start=1):  # MRR@k: only the top k count
            if doc in rel_set:
                rr = 1.0 / rank
                break
        ql = query_lang.get(qid)
        same = {doc for doc in rel_set if corpus_lang.get(doc) == ql} if corpus_lang else set()
        cross = (rel_set - same) if corpus_lang else set()
        out[qid] = {
            "recall": len(rel_set & top) / len(rel_set),
            "rr": rr,
            "hit": 1.0 if (rel_set & top) else 0.0,
            "same_recall": (len(same & top) / len(same)) if same else None,
            "cross_recall": (len(cross & top) / len(cross)) if cross else None,
            "top": top,
            "rel": rel_set,
        }
    return out


def _short_label(name: str) -> str:
    base = name.split("/")[-1]
    return base.replace("paraphrase-multilingual-", "").replace("multilingual-", "")


def _load_summary_metrics(summary_path: Path) -> dict[str, dict]:
    """Map model-slug -> metrics from a run's summary.json (empty if missing/unreadable)."""
    try:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {_slugify(m.get("model_name", "")): m.get("metrics", {}) for m in payload.get("models", [])}


def _summary_model_names(summary_path: Path) -> list[str]:
    """Real model names from a run's summary.json (for nice labels in standalone mode)."""
    try:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    return [m["model_name"] for m in payload.get("models", []) if m.get("model_name")]


def _make_plots(
    output_dir: Path,
    *,
    pq_by_model: dict,
    labels: list[str],
    langs: list[str],
    query_lang: dict,
    corpus_lang: dict,
    query_synth: dict,
    best: str | None,
    summary_metrics: dict,
    k: int,
) -> Path | None:
    """Render PNG summaries into output_dir/plots. Skips gracefully without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plots skipped] matplotlib unavailable: {exc}")
        return None

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    short = [_short_label(lb) for lb in labels]
    n = len(labels)

    def grouped_bar(fname, group_labels, values_by_label, ylabel, title, ymax=1.0):
        if not group_labels:
            return
        fig, ax = plt.subplots(figsize=(1.7 * max(len(group_labels), 3) + 1.5, 4.3))
        width = 0.8 / max(n, 1)
        xs = list(range(len(group_labels)))
        for i in range(n):
            offs = [x + (i - (n - 1) / 2) * width for x in xs]
            ax.bar(offs, [v if v is not None else 0.0 for v in values_by_label[i]], width=width, label=short[i])
            for off, v in zip(offs, values_by_label[i]):
                if v is not None:
                    ax.text(off, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(xs)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=7, ncol=min(n, 4), loc="upper center", bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)

    def mean_where(label, predicate, key="recall"):
        vals = [pq[key] for qid, pq in pq_by_model[label].items()
                if predicate(qid, pq) and pq.get(key) is not None]
        return _mean(vals) if vals else None

    # 1) overall metrics
    if summary_metrics:
        keys = [("recall_at_10", "Recall@10"), ("ndcg_at_10", "nDCG@10"), ("map_at_10", "MAP@10")]
        groups = [lbl for _, lbl in keys]
        values = [[summary_metrics.get(_slugify(lb), {}).get(mk) for mk, _ in keys] for lb in labels]
        grouped_bar("overall_metrics.png", groups, values, "score", f"Overall retrieval metrics (k={k})")
    else:
        groups = [f"Recall@{k}", f"MRR@{k}", f"hit@{k}"]
        values = [[_mean(p["recall"] for p in pq_by_model[lb].values()),
                   _mean(p["rr"] for p in pq_by_model[lb].values()),
                   _mean(p["hit"] for p in pq_by_model[lb].values())] for lb in labels]
        grouped_bar("overall_metrics.png", groups, values, "score", f"Overall (k={k})")

    # 2) recall by query language
    if langs:
        values = [[mean_where(lb, lambda q, pq, lng=lng: query_lang.get(q) == lng) for lng in langs]
                  for lb in labels]
        grouped_bar("recall_by_language.png", langs, values, f"Recall@{k}", f"Recall@{k} by query language")

    # 3) retrieval mode: same vs cross-language targets
    def mode_mean(label, key):
        vals = [pq[key] for pq in pq_by_model[label].values() if pq.get(key) is not None]
        return _mean(vals) if vals else None

    if any(mode_mean(lb, "same_recall") is not None or mode_mean(lb, "cross_recall") is not None for lb in labels):
        values = [[mode_mean(lb, "same_recall"), mode_mean(lb, "cross_recall")] for lb in labels]
        grouped_bar("mode_same_vs_cross.png", ["same-language", "cross-language"], values,
                    f"Recall@{k}", f"Same- vs cross-language targets (Recall@{k})")

    # 4) strategy: original vs synthetic-translation
    if query_synth:
        values = [[mean_where(lb, lambda q, pq, want=want: q in query_synth and query_synth[q] is want)
                   for want in (False, True)] for lb in labels]
        grouped_bar("strategy_original_vs_translation.png", ["original", "synthetic-translation"], values,
                    f"Recall@{k}", f"Recall@{k} by query origin")

    # 5) language-pair heatmap for the best model
    if best and corpus_lang and langs:
        pair_hits: dict = defaultdict(list)
        for qid, pq in pq_by_model[best].items():
            ql = query_lang.get(qid)
            for doc in pq["rel"]:
                pair_hits[(ql, corpus_lang.get(doc, "?"))].append(1.0 if doc in pq["top"] else 0.0)
        cols = sorted({dl for (_, dl) in pair_hits})
        if cols:
            mat = [[(_mean(pair_hits[(ql, dl)]) if pair_hits.get((ql, dl)) else float("nan")) for dl in cols]
                   for ql in langs]
            fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.5, 1.0 * len(langs) + 2))
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(cols)), labels=cols)
            ax.set_yticks(range(len(langs)), labels=langs)
            ax.set_xlabel("relevant-doc language")
            ax.set_ylabel("query language")
            ax.set_title(f"{_short_label(best)}: Recall@{k} by query x doc language")
            for i in range(len(langs)):
                for j in range(len(cols)):
                    v = mat[i][j]
                    nn = len(pair_hits.get((langs[i], cols[j]), []))
                    txt = "-" if v != v else f"{v:.2f}\n(n={nn})"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if (v == v and v < 0.6) else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(plots_dir / f"language_pair_heatmap_{_slugify(best)}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

    # 6) same-language irrelevant share by language (diagnostic; needs summary metrics)
    if summary_metrics and langs:
        bias_langs = [lng for lng in langs
                      if any(summary_metrics.get(_slugify(lb), {}).get(f"same_language_irrelevant_share_at_100_lang_{lng}") is not None
                             for lb in labels)]
        if bias_langs:
            values = [[summary_metrics.get(_slugify(lb), {}).get(f"same_language_irrelevant_share_at_100_lang_{lng}")
                       for lng in bias_langs] for lb in labels]
            grouped_bar("same_language_bias_by_language.png", bias_langs, values, "same-lang share",
                        "Same-language irrelevant share @100 (lower = less language bias)")

    print(f"Plots written to {plots_dir}")
    return plots_dir


def run_question_analysis(
    predictions_dir: str | Path,
    *,
    output_dir: str | Path,
    dataset_repo: str = DEFAULT_MTEB_DATASET_REPO,
    dataset_variant: str = DEFAULT_MTEB_VARIANT,
    revision: str = "main",
    k: int = DEFAULT_K,
    model_names: list[str] | None = None,
    make_plots: bool = True,
) -> Path:
    """Write a question-level analysis report from saved per-query predictions.

    Returns the path to the markdown report.
    """
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = _normalize_dataset_variant(dataset_variant)

    if model_names is None:  # recover real names from the run's summary.json for nicer labels
        model_names = _summary_model_names(output_dir.parent / "summary.json") or None

    models = _discover_models(predictions_dir, model_names)
    if not models:
        raise ValueError(
            f"No per-query predictions found under {predictions_dir}. "
            "Run the benchmark with prediction saving enabled first."
        )

    # --- dataset metadata (variant-aware config names) ---
    queries = load_dataset(
        dataset_repo, _dataset_config_name(dataset_repo, revision, variant, "queries"),
        split="train", revision=revision,
    )
    corpus = load_dataset(
        dataset_repo, _dataset_config_name(dataset_repo, revision, variant, "corpus"),
        split="train", revision=revision,
    )
    qrels = load_dataset(
        dataset_repo, _dataset_config_name(dataset_repo, revision, variant, "qrels"),
        split="train", revision=revision,
    )

    q_id_col = _id_column(list(queries.column_names), "query_id")
    q_lang_col = _query_language_column(list(queries.column_names))
    synth_col = "is_synthetic_translation" if "is_synthetic_translation" in queries.column_names else None
    query_lang, query_synth = {}, {}
    for row in queries:
        qid = str(row[q_id_col])
        lang = str(row.get(q_lang_col) or "").strip().lower() if q_lang_col else ""
        lang = lang or _infer_language(qid)  # fall back to language encoded in the id
        if lang:
            query_lang[qid] = lang
        if synth_col is not None:
            query_synth[qid] = str(row.get(synth_col)).strip().lower() in {"true", "1", "yes"}

    c_id_col = _id_column(list(corpus.column_names), "corpus_id")
    c_lang_col = _corpus_language_column(list(corpus.column_names))
    corpus_lang = {}
    for row in corpus:
        cid = str(row[c_id_col])
        lang = str(row.get(c_lang_col) or "").strip().lower() if c_lang_col else ""
        lang = lang or _infer_language(cid)  # fall back to language encoded in the id
        if lang:
            corpus_lang[cid] = lang

    qr_cols = list(qrels.column_names)
    qid_col_qr = "query-id" if "query-id" in qr_cols else qr_cols[0]
    cid_col_qr = "corpus-id" if "corpus-id" in qr_cols else (qr_cols[1] if len(qr_cols) > 1 else qr_cols[0])
    score_col = "score" if "score" in qr_cols else (qr_cols[2] if len(qr_cols) > 2 else None)
    rel = defaultdict(set)
    for row in qrels:
        if score_col is None or float(row[score_col]) > 0:  # no score column => binary relevance
            rel[str(row[qid_col_qr])].add(str(row[cid_col_qr]))

    # --- per-query metrics per model ---
    pq_by_model: dict[str, dict] = {}
    for label, slug in models:
        preds = _load_predictions(predictions_dir / slug)
        if preds is None:
            continue
        pq_by_model[label] = _per_query_metrics(preds, rel, query_lang, corpus_lang, k)
    labels = list(pq_by_model)
    if not labels:
        raise ValueError(
            f"Found prediction folders under {predictions_dir} but none contained usable "
            "per-query rankings."
        )

    langs = sorted({v for v in query_lang.values() if v}) if query_lang else []
    best = max(labels, key=lambda lb: _mean(pq["recall"] for pq in pq_by_model[lb].values()))
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit(f"# Question-level analysis ({dataset_repo}, `{variant}`, Recall@{k} / MRR@{k})\n")

    # --- dataset structure ---
    qids = [q for q in rel if (not query_lang or q in query_lang)]
    emit("## Dataset structure")
    emit(f"- Queries with relevance judgements: {len(qids)}")
    if synth_col is not None:
        n_synth = sum(1 for q in qids if query_synth.get(q))
        emit(f"- Original: {len(qids) - n_synth}  |  synthetic-translation: {n_synth}")
    if query_lang:
        by_lang = defaultdict(int)
        for q in qids:
            by_lang[query_lang.get(q, "?")] += 1
        emit("- Queries by language: " + ", ".join(f"{lng}={by_lang[lng]}" for lng in langs if by_lang.get(lng)))
    pairs = sum(len(rel[q]) for q in qids)
    emit(f"- Relevant (query, doc) pairs: {pairs} (avg {pairs / max(len(qids), 1):.2f}/query)")
    emit("- Models analysed: " + ", ".join(labels))
    emit("")

    def grouped_table(title, group_of, key):
        emit(f"## {title}")
        groups = set()
        for label in labels:
            for qid, pq in pq_by_model[label].items():
                g = group_of(qid)
                if g is not None and pq.get(key) is not None:
                    groups.add(g)
        emit("| Group | n | " + " | ".join(labels) + " |")
        emit("|" + "---|" * (len(labels) + 2))
        for g in sorted(groups):
            ref = labels[0]
            n = sum(1 for qid, pq in pq_by_model[ref].items() if group_of(qid) == g and pq.get(key) is not None)
            cells = []
            for label in labels:
                vals = [pq[key] for qid, pq in pq_by_model[label].items()
                        if group_of(qid) == g and pq.get(key) is not None]
                cells.append(f"{_mean(vals):.3f}" if vals else " - ")
            emit(f"| {g} | {n} | " + " | ".join(cells) + " |")
        emit("")

    if query_lang and len(langs) > 1:
        grouped_table(f"1) Recall@{k} by query language", lambda q: query_lang.get(q), "recall")
        grouped_table(f"   MRR@{k} by query language", lambda q: query_lang.get(q), "rr")

    if synth_col is not None:
        grouped_table(
            f"2) Recall@{k} by query origin (strategy)",
            lambda q: ("synthetic-translation" if query_synth.get(q) else "original"),
            "recall",
        )

    if corpus_lang:
        emit(f"## 3) Retrieval mode: same- vs cross-language targets (mean Recall@{k})")
        emit("| Mode | " + " | ".join(labels) + " |")
        emit("|" + "---|" * (len(labels) + 1))
        for mode, key in [("same-language target", "same_recall"), ("cross-language target", "cross_recall")]:
            cells = []
            for label in labels:
                vals = [pq[key] for pq in pq_by_model[label].values() if pq.get(key) is not None]
                cells.append(f"{_mean(vals):.3f}" if vals else " - ")
            emit(f"| {mode} | " + " | ".join(cells) + " |")
        emit("")

    if query_lang and corpus_lang and len(langs) > 1:
        emit(f"## 4) Language-pair Recall@{k} matrix — {best} (best model)")
        emit("Rows = query language, Cols = relevant-doc language; cell = fraction of those")
        emit("relevant docs retrieved in the top %d (n = #relevant pairs)." % k)
        emit("")
        pair_hits = defaultdict(list)
        for qid, pq in pq_by_model[best].items():
            ql = query_lang.get(qid)
            for doc in pq["rel"]:
                pair_hits[(ql, corpus_lang.get(doc, "?"))].append(1.0 if doc in pq["top"] else 0.0)
        cols = sorted({dl for (_, dl) in pair_hits})
        emit("| q\\d | " + " | ".join(cols) + " |")
        emit("|" + "---|" * (len(cols) + 1))
        for ql in langs:
            cells = []
            for dl in cols:
                vals = pair_hits.get((ql, dl), [])
                cells.append(f"{_mean(vals):.2f} ({len(vals)})" if vals else " - ")
            emit(f"| **{ql}** | " + " | ".join(cells) + " |")
        emit("")

    report_path = output_dir / "question_level_analysis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- per-query CSV (for custom pivots on other questions) ---
    csv_path = output_dir / "question_level_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "query_id", "query_language", "is_synthetic_translation",
                         f"recall_at_{k}", f"rr_at_{k}", f"hit_at_{k}", "n_relevant"])
        for label in labels:
            for qid, pq in pq_by_model[label].items():
                writer.writerow([
                    label, qid, query_lang.get(qid, ""),
                    query_synth.get(qid, "") if synth_col is not None else "",
                    round(pq["recall"], 5), round(pq["rr"], 5), int(pq["hit"]), len(pq["rel"]),
                ])
    print(f"Question-level analysis written to {report_path} and {csv_path}")

    if make_plots:
        summary_metrics = _load_summary_metrics(Path(output_dir).parent / "summary.json")
        try:
            _make_plots(
                output_dir, pq_by_model=pq_by_model, labels=labels, langs=langs,
                query_lang=query_lang, corpus_lang=corpus_lang, query_synth=query_synth,
                best=best, summary_metrics=summary_metrics, k=k,
            )
        except Exception as exc:  # pragma: no cover - plotting must never break the analysis
            print(f"[plots skipped] {exc}")
    return report_path
