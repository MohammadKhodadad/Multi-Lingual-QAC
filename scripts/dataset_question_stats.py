"""Generate question and corpus statistics for released Hugging Face datasets.

Usage:
    python scripts/dataset_question_stats.py

The script reads the public Hugging Face releases, computes compact statistics
for questions and corpus text, and writes:
  - reports/dataset_question_stats.md
  - paper/sections/generated_dataset_stats_table.tex
  - paper/figures/dataset_question_stats.png
"""

from __future__ import annotations

import argparse
import importlib
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset


DATASETS = [
    ("Google Patents", "MehdiAstaraki/multi-lingual-qac-chem-patents"),
    ("EPO", "MehdiAstaraki/multi-lingual-qac-epo"),
]

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class DatasetStats:
    name: str
    hf_id: str
    corpus_rows: int
    query_rows: int
    qrel_rows: int
    cross_qrel_rows: int
    languages: str
    avg_question_tokens: float
    std_question_tokens: float
    question_vocab: int
    avg_corpus_tokens: float
    std_corpus_tokens: float
    corpus_vocab: int


def _tokens(text: str | None) -> list[str]:
    return TOKEN_RE.findall((text or "").casefold())


def _vocab_size(texts: Iterable[str | None]) -> int:
    vocab: set[str] = set()
    for text in texts:
        vocab.update(_tokens(text))
    return len(vocab)


def _mean(values: list[int]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[int]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _load(hf_id: str, config: str) -> Dataset:
    return load_dataset(hf_id, config, split="train")


def _language_summary(corpus: Dataset, queries: Dataset) -> str:
    corpus_langs = Counter(corpus["corpus_language"])
    query_langs = Counter(queries["query_language"])
    langs = sorted(set(corpus_langs) | set(query_langs))
    return ", ".join(langs)


def compute_stats(name: str, hf_id: str) -> DatasetStats:
    corpus = _load(hf_id, "corpus")
    queries = _load(hf_id, "queries")
    qrels = _load(hf_id, "qrels")
    cross_qrels = _load(hf_id, "cross_language-qrels")

    query_texts = list(queries["text"])
    corpus_texts = list(corpus["text"])
    query_lengths = [len(_tokens(text)) for text in query_texts]
    corpus_lengths = [len(_tokens(text)) for text in corpus_texts]

    return DatasetStats(
        name=name,
        hf_id=hf_id,
        corpus_rows=len(corpus),
        query_rows=len(queries),
        qrel_rows=len(qrels),
        cross_qrel_rows=len(cross_qrels),
        languages=_language_summary(corpus, queries),
        avg_question_tokens=_mean(query_lengths),
        std_question_tokens=_std(query_lengths),
        question_vocab=_vocab_size(query_texts),
        avg_corpus_tokens=_mean(corpus_lengths),
        std_corpus_tokens=_std(corpus_lengths),
        corpus_vocab=_vocab_size(corpus_texts),
    )


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_float(value: float) -> str:
    return f"{value:.1f}"


def render_markdown(stats: list[DatasetStats]) -> str:
    lines = [
        "# Dataset Question Statistics",
        "",
        "Statistics are computed from the public Hugging Face releases using the",
        "`corpus`, `queries`, `qrels`, and `cross_language-qrels` configs.",
        "",
        "| Dataset | Corpus rows | Queries | Qrels | Cross-lang qrels | Languages | Mean question tokens | Std. question tokens | Question vocab | Mean corpus tokens | Std. corpus tokens | Corpus vocab |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        lines.append(
            f"| {item.name} | {_fmt_int(item.corpus_rows)} | {_fmt_int(item.query_rows)} "
            f"| {_fmt_int(item.qrel_rows)} | {_fmt_int(item.cross_qrel_rows)} "
            f"| {item.languages} | {_fmt_float(item.avg_question_tokens)} "
            f"| {_fmt_float(item.std_question_tokens)} | {_fmt_int(item.question_vocab)} "
            f"| {_fmt_float(item.avg_corpus_tokens)} | {_fmt_float(item.std_corpus_tokens)} "
            f"| {_fmt_int(item.corpus_vocab)} |"
        )
    lines.extend(["", "Dataset URLs:", ""])
    for item in stats:
        lines.append(f"- {item.name}: https://huggingface.co/datasets/{item.hf_id}")
    lines.append("")
    return "\n".join(lines)


def render_latex(stats: list[DatasetStats]) -> str:
    rows = []
    for item in stats:
        rows.append(
            f"    {item.name} & {_fmt_int(item.corpus_rows)} & {_fmt_int(item.query_rows)} "
            f"& {_fmt_int(item.qrel_rows)} & {_fmt_int(item.cross_qrel_rows)} "
            f"& {_fmt_float(item.avg_question_tokens)} $\\pm$ {_fmt_float(item.std_question_tokens)} "
            f"& {_fmt_int(item.question_vocab)} & {_fmt_float(item.avg_corpus_tokens)} $\\pm$ {_fmt_float(item.std_corpus_tokens)} "
            f"& {_fmt_int(item.corpus_vocab)} \\\\"
        )

    return "\n".join(
        [
            "% Auto-generated by scripts/dataset_question_stats.py.",
            "% Do not edit by hand; rerun the script after dataset updates.",
            "\\begin{table*}[t]",
            "  \\centering",
            "  \\footnotesize",
            "  \\setlength{\\tabcolsep}{4.5pt}",
            "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrr@{}}",
            "    \\toprule",
            "    Source & Corpus & Queries & Qrels & Cross-lang qrels & Q tokens & Q vocab & C tokens & C vocab \\\\",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular*}",
            "  \\caption{Statistics for the two released patent-derived multilingual QAC retrieval benchmarks. Question and corpus token columns report mean $\\pm$ standard deviation using regex token counts. Vocabularies are unique case-folded tokens.}",
            "  \\label{tab:benchmark-stats}",
            "\\end{table*}",
            "",
        ]
    )


def write_figure(stats: list[DatasetStats], output: Path) -> None:
    plt = importlib.import_module("matplotlib.pyplot")

    names = [s.name for s in stats]
    queries = [s.query_rows for s in stats]
    corpus = [s.corpus_rows for s in stats]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].bar(names, queries, color="#4c78a8")
    axes[0].set_title("Queries")
    axes[0].set_ylabel("Rows")
    axes[1].bar(names, corpus, color="#f58518")
    axes[1].set_title("Corpus")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate released dataset statistics.")
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/dataset_question_stats.md"))
    parser.add_argument("--latex-output", type=Path, default=Path("paper/sections/generated_dataset_stats_table.tex"))
    parser.add_argument("--figure-output", type=Path, default=Path("paper/figures/dataset_question_stats.png"))
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    stats = [compute_stats(name, hf_id) for name, hf_id in DATASETS]

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(stats), encoding="utf-8")

    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.write_text(render_latex(stats), encoding="utf-8")

    if not args.skip_figure:
        write_figure(stats, args.figure_output)

    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.latex_output}")
    if not args.skip_figure:
        print(f"Wrote {args.figure_output}")


if __name__ == "__main__":
    main()
