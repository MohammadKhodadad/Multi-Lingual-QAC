"""
Compare several LLMs on the same English -> simplified-Chinese patent
translation task, using the exact same translation prompt as
``scripts/translate_to_chinese.py``.

The same N English documents (default 30) are sampled once from the
multilingual corpus and translated by every model. For each call we record
token usage (prompt / completion / reasoning / cached) and the dollar cost,
then estimate the cost of translating ``--project-to`` documents (default 100)
with each model.

This is a model-comparison experiment, not a dataset extension: nothing is
appended to the corpus. All output goes to ``--output-dir``
(default ``reports/translation_model_comparison``):

  - selected_documents.csv        the N shared source English docs
  - translations__<label>.csv     per model, one row per document
  - cost_summary.csv / .md        per model token totals + 100-doc cost estimate

Models (slugs + provider) are defined in ``MODELS`` below:
  - google/gemma-4-31b-it   (OpenRouter, thinking enabled)
  - qwen/qwen3.6-35b-a3b     (OpenRouter, thinking enabled)
  - qwen/qwen3.7-max         (OpenRouter, thinking enabled)
  - gpt-5.5                  (OpenAI, reasoning_effort=medium)

Usage:
    python scripts/translate_model_comparison.py --count 30

Requires OPENAI_API_KEY (for gpt-5.5) and OPENROUTER_API_KEY (for the other
three) in the environment or .env file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from tqdm import tqdm

# Reuse the exact prompt + field handling from the existing translator so the
# translation task is identical across this comparison.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_to_chinese import (  # noqa: E402
    SYSTEM_PROMPT,
    TRANSLATABLE_FIELDS,
    _build_context,
    _parse_json_response,
    load_corpus,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Each model is run over the same documents. ``reasoning`` => enable thinking on
# OpenRouter; ``reasoning_effort`` is the top-level OpenAI parameter.
MODELS: List[Dict[str, Any]] = [
    {"label": "gemma-4-31b", "slug": "google/gemma-4-31b-it", "provider": "openrouter", "reasoning": True},
    {"label": "qwen3.6-35b-a3b", "slug": "qwen/qwen3.6-35b-a3b", "provider": "openrouter", "reasoning": True},
    {"label": "qwen3.7-max", "slug": "qwen/qwen3.7-max", "provider": "openrouter", "reasoning": True},
    {"label": "gpt-5.5", "slug": "gpt-5.5", "provider": "openai", "reasoning_effort": "medium"},
]

# List price, USD per 1M tokens. Used to compute gpt-5.5 cost (OpenAI returns no
# cost) and as a promo-free cross-check projection for the OpenRouter models.
# Resolved from OpenRouter / OpenAI pricing pages (June 2026).
PRICING: Dict[str, Dict[str, float]] = {
    "google/gemma-4-31b-it": {"in": 0.12, "out": 0.35},
    "qwen/qwen3.6-35b-a3b": {"in": 0.14, "out": 1.00},
    "qwen/qwen3.7-max": {"in": 1.25, "out": 3.75},  # 50% OpenRouter promo applies to measured cost
    "gpt-5.5": {"in": 5.00, "out": 30.00, "cached_in": 0.50},
}

PER_DOC_FIELDS = [
    "publication_number",
    "source_title",
    "source_abstract",
    "zh_title",
    "zh_abstract",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "total_tokens",
    "cost_usd",
    "latency_s",
    "status",
    "error",
]


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env for the gpt-5.5 model.")
    return OpenAI(api_key=api_key)


def _get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in .env for the OpenRouter models.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


# --------------------------------------------------------------------------- #
# Document selection
# --------------------------------------------------------------------------- #
def select_english_documents(
    corpus_path: Path, *, count: int, seed: int
) -> List[Dict[str, str]]:
    """Sample ``count`` English rows (with a non-empty abstract) deterministically."""
    rows = load_corpus(corpus_path)
    english = [
        r
        for r in rows
        if r.get("language") == "en" and (r.get("abstract") or "").strip()
    ]
    if count > len(english):
        raise ValueError(
            f"Requested {count} English documents but only {len(english)} are available."
        )
    english.sort(key=lambda r: r["publication_number"])
    return random.Random(seed).sample(english, count)


# --------------------------------------------------------------------------- #
# Usage / cost extraction
# --------------------------------------------------------------------------- #
def _extract_usage(response: Any) -> Dict[str, Optional[float]]:
    """Pull token counts and (OpenRouter) cost out of a chat-completion response.

    Uses ``model_dump()`` so provider-specific extra fields (``cost``,
    ``reasoning_tokens``, ``cached_tokens``) survive regardless of SDK typing.
    """
    try:
        data = response.model_dump()
    except Exception:
        data = {}
    usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
    ctd = usage.get("completion_tokens_details") or {}
    ptd = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": ctd.get("reasoning_tokens"),
        "cached_tokens": ptd.get("cached_tokens"),
        "cost": usage.get("cost"),  # OpenRouter only (with usage.include)
    }


def _compute_cost_from_pricing(
    slug: str,
    prompt_tokens: Optional[float],
    completion_tokens: Optional[float],
    cached_tokens: Optional[float],
) -> Optional[float]:
    price = PRICING.get(slug)
    if price is None or prompt_tokens is None or completion_tokens is None:
        return None
    cached = cached_tokens or 0
    cached_rate = price.get("cached_in", price["in"])
    uncached = max(prompt_tokens - cached, 0)
    return (
        uncached / 1e6 * price["in"]
        + cached / 1e6 * cached_rate
        + completion_tokens / 1e6 * price["out"]
    )


# --------------------------------------------------------------------------- #
# Per-model translation calls
# --------------------------------------------------------------------------- #
def _call_openrouter(
    client: OpenAI, slug: str, messages: List[Dict[str, str]], max_tokens: int
) -> Any:
    """Call an OpenRouter model with thinking + usage accounting, falling back
    through reasoning-parameter variants if a provider rejects one."""
    reasoning_variants: List[Optional[Dict[str, Any]]] = [
        {"effort": "medium"},
        {"enabled": True},
        None,
    ]
    last_exc: Optional[Exception] = None
    for rv in reasoning_variants:
        extra_body: Dict[str, Any] = {"usage": {"include": True}}
        if rv is not None:
            extra_body["reasoning"] = rv
        try:
            return client.chat.completions.create(
                model=slug,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except BadRequestError as exc:
            # Likely an unsupported reasoning shape for this provider; try the next.
            last_exc = exc
            continue
    raise last_exc  # type: ignore[misc]


def _call_openai(
    client: OpenAI, slug: str, messages: List[Dict[str, str]], reasoning_effort: str
) -> Any:
    return client.chat.completions.create(
        model=slug,
        messages=messages,
        reasoning_effort=reasoning_effort,
    )


def translate_one(
    client: OpenAI,
    model: Dict[str, Any],
    source: Dict[str, str],
    *,
    max_tokens: int,
) -> Dict[str, Any]:
    """Translate one document with one model; returns a per-doc result row."""
    payload = {f: source.get(f, "") or "" for f in TRANSLATABLE_FIELDS}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    def _do_call() -> Any:
        if model["provider"] == "openrouter":
            return _call_openrouter(client, model["slug"], messages, max_tokens)
        return _call_openai(client, model["slug"], messages, model.get("reasoning_effort", "medium"))

    record: Dict[str, Any] = {
        "publication_number": source.get("publication_number", ""),
        "source_title": source.get("title", "") or "",
        "source_abstract": source.get("abstract", "") or "",
        "zh_title": "",
        "zh_abstract": "",
        "prompt_tokens": "",
        "completion_tokens": "",
        "reasoning_tokens": "",
        "cached_tokens": "",
        "total_tokens": "",
        "cost_usd": "",
        "latency_s": "",
        "status": "",
        "error": "",
    }

    start = time.perf_counter()
    response: Any = None
    translated: Optional[Dict[str, str]] = None
    error = ""
    # One retry on parse failure (model returned non-JSON).
    for attempt in range(2):
        try:
            response = _do_call()
            content = response.choices[0].message.content or ""
            translated = _parse_json_response(content)
            error = ""
            break
        except json.JSONDecodeError as exc:
            error = f"json_parse_error: {exc}"
            translated = None
            continue
        except Exception as exc:  # network / API error
            error = f"api_error: {exc}"
            response = None
            break
    record["latency_s"] = round(time.perf_counter() - start, 3)

    # Token usage from whatever response we last got (even on parse error).
    if response is not None:
        usage = _extract_usage(response)
        record["prompt_tokens"] = usage["prompt_tokens"]
        record["completion_tokens"] = usage["completion_tokens"]
        record["reasoning_tokens"] = usage["reasoning_tokens"]
        record["cached_tokens"] = usage["cached_tokens"]
        record["total_tokens"] = usage["total_tokens"]
        # Prefer the provider-reported cost (OpenRouter, promo-inclusive);
        # otherwise compute from the list-price table (gpt-5.5).
        cost = usage["cost"]
        if cost is None:
            cost = _compute_cost_from_pricing(
                model["slug"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["cached_tokens"],
            )
        record["cost_usd"] = cost

    if translated is not None:
        zh_title = translated.get("title", "") or ""
        zh_abstract = translated.get("abstract", "") or ""
        record["zh_title"] = zh_title
        record["zh_abstract"] = zh_abstract
        # Rebuilt for parity with the corpus, even though we don't store it as a row.
        _ = _build_context(zh_title, zh_abstract)
        record["status"] = "ok"
    else:
        record["status"] = "error"
        record["error"] = error
    return record


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _write_selected(out_dir: Path, docs: List[Dict[str, str]]) -> None:
    path = out_dir / "selected_documents.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["publication_number", "id", "title", "abstract"]
        )
        writer.writeheader()
        for d in docs:
            writer.writerow(
                {
                    "publication_number": d.get("publication_number", ""),
                    "id": d.get("id", ""),
                    "title": d.get("title", "") or "",
                    "abstract": d.get("abstract", "") or "",
                }
            )


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summarize(label: str, slug: str, rows: List[Dict[str, Any]], project_to: int) -> Dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    n_ok = len(ok)
    sum_prompt = sum(_num(r["prompt_tokens"]) for r in ok)
    sum_completion = sum(_num(r["completion_tokens"]) for r in ok)
    sum_reasoning = sum(_num(r["reasoning_tokens"]) for r in ok)
    sum_total = sum(_num(r["total_tokens"]) for r in ok)
    sum_cost = sum(_num(r["cost_usd"]) for r in ok)
    avg_in = sum_prompt / n_ok if n_ok else 0.0
    avg_out = sum_completion / n_ok if n_ok else 0.0
    measured_per_doc = sum_cost / n_ok if n_ok else 0.0
    # Promo-free cross-check from list price using observed avg tokens.
    price = PRICING.get(slug, {})
    list_per_doc = (
        avg_in / 1e6 * price.get("in", 0.0) + avg_out / 1e6 * price.get("out", 0.0)
    )
    return {
        "label": label,
        "slug": slug,
        "n_ok": n_ok,
        "n_total": len(rows),
        "sum_prompt_tokens": int(sum_prompt),
        "sum_completion_tokens": int(sum_completion),
        "sum_reasoning_tokens": int(sum_reasoning),
        "sum_total_tokens": int(sum_total),
        "avg_input_tokens": round(avg_in, 1),
        "avg_output_tokens": round(avg_out, 1),
        "measured_cost_sampled": round(sum_cost, 6),
        f"measured_cost_{project_to}docs": round(measured_per_doc * project_to, 4),
        f"listprice_cost_{project_to}docs": round(list_per_doc * project_to, 4),
    }


def _write_summary(out_dir: Path, summaries: List[Dict[str, Any]], project_to: int, n_docs: int) -> None:
    csv_path = out_dir / "cost_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    md_path = out_dir / "cost_summary.md"
    measured_col = f"measured_cost_{project_to}docs"
    list_col = f"listprice_cost_{project_to}docs"
    lines: List[str] = []
    lines.append(f"# Translation model comparison — cost & token usage\n")
    lines.append(
        f"Same {n_docs} English documents translated to simplified Chinese by each "
        f"model, using the identical translation prompt from `translate_to_chinese.py`.\n"
    )
    lines.append(
        "| Model | slug | ok | avg in tok | avg out tok | reasoning tok (total) | "
        f"measured cost ({n_docs} docs) | est. {project_to} docs (measured) | "
        f"est. {project_to} docs (list price) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['label']} | `{s['slug']}` | {s['n_ok']}/{s['n_total']} | "
            f"{s['avg_input_tokens']:.0f} | {s['avg_output_tokens']:.0f} | "
            f"{s['sum_reasoning_tokens']} | ${s['measured_cost_sampled']:.4f} | "
            f"${s[measured_col]:.4f} | ${s[list_col]:.4f} |"
        )
    lines.append("")
    lines.append(
        "- **measured cost** uses the provider-reported dollar cost for the OpenRouter "
        "models (promo-inclusive) and list-price × token counts for gpt-5.5.\n"
        f"- **est. {project_to} docs (measured)** = measured cost / ok docs × {project_to}.\n"
        "- **est. {0} docs (list price)** is a promo-free cross-check from the list "
        "rates in `PRICING` × observed average tokens.\n"
        "- `qwen/qwen3.7-max` measured cost reflects the current 50% OpenRouter promo; "
        "the list-price column shows the un-discounted estimate.".format(project_to)
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    *,
    corpus_path: Path,
    count: int,
    seed: int,
    output_dir: Path,
    model_labels: Optional[List[str]],
    project_to: int,
    max_tokens: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_models = MODELS
    if model_labels:
        wanted = set(model_labels)
        selected_models = [m for m in MODELS if m["label"] in wanted]
        missing = wanted - {m["label"] for m in selected_models}
        if missing:
            raise ValueError(f"Unknown model labels: {sorted(missing)}. Known: {[m['label'] for m in MODELS]}")

    docs = select_english_documents(corpus_path, count=count, seed=seed)
    _write_selected(output_dir, docs)
    print(f"Selected {len(docs)} English documents -> {output_dir / 'selected_documents.csv'}")

    # Lazily build only the clients we need.
    clients: Dict[str, OpenAI] = {}
    need_providers = {m["provider"] for m in selected_models}
    if "openai" in need_providers:
        clients["openai"] = _get_openai_client()
    if "openrouter" in need_providers:
        clients["openrouter"] = _get_openrouter_client()

    summaries: List[Dict[str, Any]] = []
    for model in selected_models:
        client = clients[model["provider"]]
        out_csv = output_dir / f"translations__{model['label']}.csv"
        rows: List[Dict[str, Any]] = []
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PER_DOC_FIELDS, extrasaction="ignore")
            writer.writeheader()
            f.flush()
            for source in tqdm(docs, desc=f"{model['label']:>16} -> zh", unit="doc"):
                record = translate_one(client, model, source, max_tokens=max_tokens)
                rows.append(record)
                writer.writerow(record)
                f.flush()  # incremental: keep partial results on failure
                if record["status"] != "ok":
                    tqdm.write(f"  {record['publication_number']}: {record['error']}")
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"  {model['label']}: {n_ok}/{len(rows)} ok -> {out_csv}")
        summaries.append(_summarize(model["label"], model["slug"], rows, project_to))

    _write_summary(output_dir, summaries, project_to, len(docs))
    print(f"\nWrote cost summary -> {output_dir / 'cost_summary.md'}")
    # Echo the headline numbers.
    measured_col = f"measured_cost_{project_to}docs"
    for s in summaries:
        print(
            f"  {s['label']:>16}: avg {s['avg_input_tokens']:.0f} in / "
            f"{s['avg_output_tokens']:.0f} out tok/doc, "
            f"~${s[measured_col]:.3f} for {project_to} docs"
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Translate the same English documents to Chinese with several models "
            "and report per-model token usage and cost estimates."
        )
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/google_patents/multilingual_corpus.csv"),
        help="Path to the multilingual corpus CSV (read-only).",
    )
    parser.add_argument("--count", type=int, default=30, help="Number of English documents to translate (default: 30).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for document sampling (default: 42).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/translation_model_comparison"),
        help="Directory for translations and the cost summary.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Optional subset of model labels to run. Known: {[m['label'] for m in MODELS]}",
    )
    parser.add_argument(
        "--project-to",
        type=int,
        default=100,
        help="Number of documents to project the cost estimate to (default: 100).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="max_tokens for OpenRouter calls (room for thinking + JSON answer).",
    )
    args = parser.parse_args()

    run(
        corpus_path=args.corpus,
        count=args.count,
        seed=args.seed,
        output_dir=args.output_dir,
        model_labels=args.models,
        project_to=args.project_to,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
