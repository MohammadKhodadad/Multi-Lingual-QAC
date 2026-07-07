"""
Compare several LLMs on the same chemistry-patent question-generation + verification
task, using the exact same generation and verifier prompts as the pipeline that
produced data/google_patents/qac/qac_chempatents.csv.

The same 30 documents are sampled once, each carrying a fixed
(mode, strategy -> query-language) combination taken from qac_chempatents_best.csv.
That combination is shared across every generator model (only the generator varies),
so for one document all models receive identical passages and generate questions in
the same language and mode. 30 docs x 5 generators = 150 generation calls.

Each generation (3 Q&A pairs) is then VERIFIED with two grader calls — faithfulness
and mode-specific quality — exactly like the chempatents pipeline. The best of the 3
pairs (by total faith+quality score) is kept as the representative query.

Verifier routing (avoids a model grading its own output):
  - default verifier               = anthropic/claude-sonnet-4.6  (the chempatents grader)
  - when the GENERATOR is sonnet-4.6 -> gpt-5.5 verifies instead

For each call we record token usage (prompt / completion / reasoning / cached) and the
dollar cost, split into generation vs verification, then estimate the cost of producing
--project-to queries (default 1000) with each generator under two assumptions:
  - keep-best-1-of-3:  1 usable query per batch  (1000 queries  = 1000 gen+verify batches)
  - use-all-3:         3 usable queries per batch (1000 queries ~= 334 batches)

This is a model-comparison experiment: nothing is written back to the qac files.
Output goes to --output-dir (default reports/qa_generation_model_comparison):
  - selected_documents.csv         the 30 shared (doc, mode, query-language) triples
  - generations__<label>.csv       per generator, one row per document
  - cost_summary.csv / .md         per generator token totals + 1000-query cost estimates

Generators (gpt-* -> OpenAI; others -> OpenRouter), all with medium reasoning:
  gpt-5-mini, gpt-5.4-mini, anthropic/claude-sonnet-4.6, x-ai/grok-4.3, google/gemini-3.5-flash

Usage:
    python scripts/qa_generation_model_comparison.py --count 30

Requires OPENAI_API_KEY (gpt-*) and OPENROUTER_API_KEY (others) in env or .env.
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
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# Reuse the exact generation + verifier prompts, passage assembly, parsing and
# score helpers from the pipeline so the task is identical across this comparison.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.multi_lingual_qac.qac_generation.multilingual_qa import (  # noqa: E402
    MODE_TECHNICAL,
    _build_all_passages_text,
    _compute_faith_overall,
    _compute_quality_overall,
    _load_faithfulness_prompt,
    _load_generation_prompt,
    _load_quality_prompt,
    _parse_json_response,
    _pick_context,
    _serialize_context_languages,
    load_multilingual_corpus,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GEN_REASONING_EFFORT = "medium"     # matches DEFAULT_GENERATION_REASONING_EFFORT
VERIFY_REASONING_EFFORT = "low"     # matches the pipeline grader (DEFAULT_REASONING_EFFORT)

SONNET_SLUG = "anthropic/claude-sonnet-4.6"
DEFAULT_VERIFIER = SONNET_SLUG
VERIFIER_WHEN_SONNET_GENERATES = "gpt-5.5"

# Generator models (gpt-* -> OpenAI, else OpenRouter). All generate with medium reasoning.
MODELS: List[Dict[str, Any]] = [
    {"label": "gpt-5-mini", "slug": "gpt-5-mini"},
    {"label": "gpt-5.4-mini", "slug": "gpt-5.4-mini"},
    {"label": "sonnet-4.6", "slug": SONNET_SLUG},
    {"label": "grok-4.3", "slug": "x-ai/grok-4.3"},
    {"label": "gemini-3.5-flash", "slug": "google/gemini-3.5-flash"},
    {"label": "qwen3.6-35b-a3b", "slug": "qwen/qwen3.6-35b-a3b"},
]

# List price, USD per 1M tokens. gpt-* cost is computed from this (OpenAI returns no
# cost); OpenRouter cost is provider-measured but the table is kept for cross-check.
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {"in": 0.13, "out": 1.00},
    "gpt-5.4-mini": {"in": 0.75, "out": 4.50},
    "gpt-5.5": {"in": 5.00, "out": 30.00, "cached_in": 0.50},
    "anthropic/claude-sonnet-4.6": {"in": 3.00, "out": 15.00},
    "x-ai/grok-4.3": {"in": 1.25, "out": 2.50},
    "google/gemini-3.5-flash": {"in": 1.50, "out": 9.00},
    "qwen/qwen3.6-35b-a3b": {"in": 0.14, "out": 1.00},
}

PER_DOC_FIELDS = [
    "publication_number", "mode", "strategy_name", "question_language",
    "generator_model", "verifier_model",
    "best_question", "best_answer", "category",
    "best_total_score", "best_faith_overall", "best_qual_overall",
    "n_questions_returned",
    "gen_prompt_tokens", "gen_completion_tokens", "gen_reasoning_tokens", "gen_cost_usd",
    "verify_prompt_tokens", "verify_completion_tokens", "verify_reasoning_tokens", "verify_cost_usd",
    "total_cost_usd", "latency_s", "status", "error",
]


def _provider_for(slug: str) -> str:
    return "openai" if slug.startswith("gpt-") else "openrouter"


def _verifier_for(generator_slug: str) -> str:
    return VERIFIER_WHEN_SONNET_GENERATES if generator_slug == SONNET_SLUG else DEFAULT_VERIFIER


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env for the gpt-* models.")
    return OpenAI(api_key=api_key)


def _get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in .env for the OpenRouter models.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


# --------------------------------------------------------------------------- #
# Document selection (shared across all models)
# --------------------------------------------------------------------------- #
def select_documents(
    best_csv: Path,
    corpus_groups: Dict[str, List[Dict[str, Any]]],
    *,
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample `count` distinct publications from qac_chempatents_best.csv, each with
    its recorded (mode, strategy, question_language), keeping only docs present in the
    corpus with non-empty passages + context for the target language."""
    with best_csv.open(encoding="utf-8", newline="") as f:
        best_rows = list(csv.DictReader(f))

    seen: set[str] = set()
    unique: List[Dict[str, str]] = []
    for row in sorted(best_rows, key=lambda r: (r["publication_number"], r.get("question_language", ""))):
        pub = row["publication_number"]
        if pub in seen:
            continue
        seen.add(pub)
        unique.append(row)

    rng = random.Random(seed)
    rng.shuffle(unique)

    selected: List[Dict[str, Any]] = []
    for row in unique:
        if len(selected) >= count:
            break
        pub = row["publication_number"]
        rows = corpus_groups.get(pub)
        if not rows:
            continue
        qlang = row["question_language"]
        all_passages = _build_all_passages_text(rows)
        if not all_passages.strip():
            continue
        _, context_text = _pick_context(rows, qlang)
        if not context_text.strip():
            continue
        selected.append(
            {
                "publication_number": pub,
                "mode": row["mode"],
                "strategy_name": row.get("strategy_name", ""),
                "question_language": qlang,
                "context_language": _serialize_context_languages(rows),
                "all_passages": all_passages,
                "n_passage_chars": len(all_passages),
            }
        )

    if len(selected) < count:
        raise ValueError(
            f"Only {len(selected)} usable documents found (requested {count}). "
            "Try a different --seed or --count."
        )
    return selected


# --------------------------------------------------------------------------- #
# Usage / cost extraction
# --------------------------------------------------------------------------- #
def _extract_usage(response: Any) -> Dict[str, Optional[float]]:
    try:
        data = response.model_dump()
    except Exception:
        data = {}
    usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
    ctd = usage.get("completion_tokens_details") or {}
    ptd = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "total_tokens": usage.get("total_tokens") or 0,
        "reasoning_tokens": ctd.get("reasoning_tokens") or 0,
        "cached_tokens": ptd.get("cached_tokens") or 0,
        "cost": usage.get("cost"),  # OpenRouter only (with usage.include)
    }


def _cost_for(slug: str, usage: Dict[str, Optional[float]]) -> float:
    """Provider-reported cost if present (OpenRouter), else compute from PRICING."""
    if usage.get("cost") is not None:
        return float(usage["cost"])
    price = PRICING.get(slug)
    if price is None:
        return 0.0
    prompt = usage["prompt_tokens"] or 0
    completion = usage["completion_tokens"] or 0
    cached = usage["cached_tokens"] or 0
    cached_rate = price.get("cached_in", price["in"])
    uncached = max(prompt - cached, 0)
    return uncached / 1e6 * price["in"] + cached / 1e6 * cached_rate + completion / 1e6 * price["out"]


# --------------------------------------------------------------------------- #
# Provider-aware chat call
# --------------------------------------------------------------------------- #
def _chat(
    client: OpenAI,
    slug: str,
    messages: List[Dict[str, str]],
    *,
    reasoning_effort: str,
    max_tokens: int,
) -> Any:
    if _provider_for(slug) == "openai":
        return client.chat.completions.create(
            model=slug, messages=messages, reasoning_effort=reasoning_effort
        )
    # OpenRouter: enable reasoning + usage accounting, with a fallback ladder.
    reasoning_variants: List[Optional[Dict[str, Any]]] = [
        {"effort": reasoning_effort},
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
                model=slug, messages=messages, max_tokens=max_tokens, extra_body=extra_body
            )
        except BadRequestError as exc:
            last_exc = exc
            continue
    raise last_exc  # type: ignore[misc]


def _candidates_block(qa_pairs: List[Dict[str, str]]) -> str:
    return "\n\n".join(
        f"Candidate {i}:\n  Question: {qa['question']}\n  Answer: {qa['answer']}"
        for i, qa in enumerate(qa_pairs)
    )


# --------------------------------------------------------------------------- #
# Generation + verification for one (document, generator)
# --------------------------------------------------------------------------- #
def process_doc(
    gen_client: OpenAI,
    verify_client: OpenAI,
    generator_slug: str,
    verifier_slug: str,
    doc: Dict[str, Any],
    *,
    max_tokens: int,
) -> Dict[str, Any]:
    mode = doc["mode"]
    target_lang = doc["question_language"]
    all_passages = doc["all_passages"]

    record: Dict[str, Any] = {k: "" for k in PER_DOC_FIELDS}
    record.update(
        {
            "publication_number": doc["publication_number"],
            "mode": mode,
            "strategy_name": doc.get("strategy_name", ""),
            "question_language": target_lang,
            "generator_model": generator_slug,
            "verifier_model": verifier_slug,
        }
    )

    start = time.perf_counter()

    # ---- 1) Generation (3 Q&A pairs) ------------------------------------- #
    gen_messages = [
        {"role": "system", "content": _load_generation_prompt(mode, target_lang)},
        {"role": "user", "content": all_passages},
    ]
    qa_pairs: List[Dict[str, str]] = []
    gen_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0, "total_tokens": 0, "cost": None}
    gen_cost = 0.0
    try:
        for _ in range(2):  # one retry on parse failure
            resp = _chat(gen_client, generator_slug, gen_messages, reasoning_effort=GEN_REASONING_EFFORT, max_tokens=max_tokens)
            gen_usage = _extract_usage(resp)
            gen_cost = _cost_for(generator_slug, gen_usage)
            try:
                data = _parse_json_response(resp.choices[0].message.content or "")
                if isinstance(data, dict):
                    data = [data]
                for item in list(data)[:3]:
                    qa_pairs.append(
                        {
                            "question": str(item.get("question", "")).strip(),
                            "answer": str(item.get("answer", "")).strip(),
                            "category": str(item.get("question_type" if mode == MODE_TECHNICAL else "framing", "")).strip(),
                        }
                    )
                qa_pairs = [q for q in qa_pairs if q["question"]]
                break
            except json.JSONDecodeError:
                qa_pairs = []
                continue
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"generation_error: {exc}"
        record["latency_s"] = round(time.perf_counter() - start, 3)
        record["gen_cost_usd"] = round(gen_cost, 6)
        return record

    _store_usage(record, "gen", gen_usage, gen_cost)
    record["n_questions_returned"] = len(qa_pairs)
    if not qa_pairs:
        record["status"] = "error"
        record["error"] = "no_questions_generated"
        record["latency_s"] = round(time.perf_counter() - start, 3)
        return record

    # ---- 2) Verification (faithfulness + quality, all pairs at once) ----- #
    verify_usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    verify_cost = 0.0
    verify_error = ""
    faith: List[Dict[str, Any]] = []
    qual: List[Dict[str, Any]] = []
    candidates = _candidates_block(qa_pairs)

    try:
        f_resp = _chat(
            verify_client, verifier_slug,
            [
                {"role": "system", "content": _load_faithfulness_prompt()},
                {"role": "user", "content": f"{all_passages}\n\n{candidates}"},
            ],
            reasoning_effort=VERIFY_REASONING_EFFORT, max_tokens=max_tokens,
        )
        fu = _extract_usage(f_resp)
        verify_cost += _cost_for(verifier_slug, fu)
        _accumulate(verify_usage_total, fu)
        faith = _normalise_faith(_parse_json_response(f_resp.choices[0].message.content or ""), len(qa_pairs))

        q_resp = _chat(
            verify_client, verifier_slug,
            [
                {"role": "system", "content": _load_quality_prompt(mode)},
                {"role": "user", "content": f"{all_passages}\n\n{candidates}"},
            ],
            reasoning_effort=VERIFY_REASONING_EFFORT, max_tokens=max_tokens,
        )
        qu = _extract_usage(q_resp)
        verify_cost += _cost_for(verifier_slug, qu)
        _accumulate(verify_usage_total, qu)
        qual = _normalise_quality(_parse_json_response(q_resp.choices[0].message.content or ""), mode, len(qa_pairs))
    except Exception as exc:
        verify_error = f"verification_error: {exc}"

    verify_usage_total["total_tokens"] = verify_usage_total["prompt_tokens"] + verify_usage_total["completion_tokens"]
    _store_usage(record, "verify", verify_usage_total, verify_cost)
    record["total_cost_usd"] = round(gen_cost + verify_cost, 6)
    record["latency_s"] = round(time.perf_counter() - start, 3)

    # ---- 3) Pick best of the 3 by total score --------------------------- #
    best_idx, best_total, best_faith, best_qual = _pick_best(qa_pairs, faith, qual)
    best = qa_pairs[best_idx]
    record["best_question"] = best["question"]
    record["best_answer"] = best["answer"]
    record["category"] = best["category"]
    record["best_total_score"] = best_total
    record["best_faith_overall"] = best_faith
    record["best_qual_overall"] = best_qual

    if verify_error:
        record["status"] = "ok_gen_verify_failed"
        record["error"] = verify_error
    else:
        record["status"] = "ok"
    return record


def _store_usage(record: Dict[str, Any], prefix: str, usage: Dict[str, Any], cost: float) -> None:
    record[f"{prefix}_prompt_tokens"] = usage.get("prompt_tokens", 0)
    record[f"{prefix}_completion_tokens"] = usage.get("completion_tokens", 0)
    record[f"{prefix}_reasoning_tokens"] = usage.get("reasoning_tokens", 0)
    record[f"{prefix}_cost_usd"] = round(cost, 6)


def _accumulate(acc: Dict[str, int], usage: Dict[str, Any]) -> None:
    for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"):
        acc[k] = acc.get(k, 0) + int(usage.get(k, 0) or 0)


def _normalise_faith(data: Any, n: int) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        data = [data]
    rows = sorted(list(data)[:n], key=lambda x: x.get("index", 0)) if data else []
    out: List[Dict[str, Any]] = []
    for item in rows:
        row = {
            "grounding": int(item.get("grounding", 1)),
            "precision": int(item.get("precision", 1)),
            "numerical_fidelity": int(item.get("numerical_fidelity", 1)),
        }
        row["overall"] = _compute_faith_overall(row)
        out.append(row)
    while len(out) < n:
        row = {"grounding": 1, "precision": 1, "numerical_fidelity": 1}
        row["overall"] = _compute_faith_overall(row)
        out.append(row)
    return out


def _normalise_quality(data: Any, mode: str, n: int) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        data = [data]
    rows = sorted(list(data)[:n], key=lambda x: x.get("index", 0)) if data else []
    if mode == MODE_TECHNICAL:
        keys = ("search_bar_realism", "specificity", "phrasing_economy", "focus", "linguistic_quality")
    else:
        keys = ("search_realism", "lexical_distance", "conceptual_framing", "retrievability", "linguistic_quality")
    out: List[Dict[str, Any]] = []
    for item in rows:
        row = {k: int(item.get(k, 1)) for k in keys}
        row["overall"] = _compute_quality_overall(row, mode)
        out.append(row)
    while len(out) < n:
        row = {k: 1 for k in keys}
        row["overall"] = _compute_quality_overall(row, mode)
        out.append(row)
    return out


def _pick_best(
    qa_pairs: List[Dict[str, str]], faith: List[Dict[str, Any]], qual: List[Dict[str, Any]]
) -> Tuple[int, int, int, int]:
    best_idx, best_total, best_f, best_q = 0, -1, 0, 0
    for i in range(len(qa_pairs)):
        f = int(faith[i]["overall"]) if i < len(faith) else 0
        q = int(qual[i]["overall"]) if i < len(qual) else 0
        total = f + q
        if total > best_total:
            best_idx, best_total, best_f, best_q = i, total, f, q
    return best_idx, max(best_total, 0), best_f, best_q


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _write_selected(out_dir: Path, docs: List[Dict[str, Any]]) -> None:
    fields = ["publication_number", "mode", "strategy_name", "question_language", "context_language", "n_passage_chars"]
    with (out_dir / "selected_documents.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(docs)


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summarize(model: Dict[str, Any], verifier_slug: str, rows: List[Dict[str, Any]], project_to: int) -> Dict[str, Any]:
    ok = [r for r in rows if str(r["status"]).startswith("ok")]
    n_ok = len(ok)

    def avg(col: str) -> float:
        return (sum(_num(r[col]) for r in ok) / n_ok) if n_ok else 0.0

    avg_gen_in = avg("gen_prompt_tokens")
    avg_gen_out = avg("gen_completion_tokens")
    avg_verify_in = avg("verify_prompt_tokens")
    avg_verify_out = avg("verify_completion_tokens")
    avg_gen_cost = avg("gen_cost_usd")
    avg_verify_cost = avg("verify_cost_usd")
    avg_total_cost = avg_gen_cost + avg_verify_cost
    sum_total_cost = sum(_num(r["total_cost_usd"]) for r in ok)
    return {
        "label": model["label"],
        "slug": model["slug"],
        "verifier": verifier_slug,
        "n_ok": n_ok,
        "n_total": len(rows),
        "avg_gen_in_tok": round(avg_gen_in, 1),
        "avg_gen_out_tok": round(avg_gen_out, 1),
        "avg_verify_in_tok": round(avg_verify_in, 1),
        "avg_verify_out_tok": round(avg_verify_out, 1),
        "avg_gen_cost": round(avg_gen_cost, 6),
        "avg_verify_cost": round(avg_verify_cost, 6),
        "avg_total_cost_per_batch": round(avg_total_cost, 6),
        "measured_total_cost_sampled": round(sum_total_cost, 6),
        f"est_{project_to}_keep_best_1": round(avg_total_cost * project_to, 4),
        f"est_{project_to}_use_all_3": round(avg_total_cost * project_to / 3.0, 4),
        f"est_{project_to}_gen_only": round(avg_gen_cost * project_to, 4),
        f"est_{project_to}_verify_only": round(avg_verify_cost * project_to, 4),
    }


def _write_summary(out_dir: Path, summaries: List[Dict[str, Any]], project_to: int, n_docs: int) -> None:
    with (out_dir / "cost_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    keep_col = f"est_{project_to}_keep_best_1"
    all_col = f"est_{project_to}_use_all_3"
    lines: List[str] = []
    lines.append("# QA-generation + verification model comparison — cost & token usage\n")
    lines.append(
        f"Same {n_docs} documents, each with a fixed (mode, strategy -> query-language) shared "
        "across all generators. Identical generation + verifier prompts from the `qac_chempatents` "
        "pipeline. Each generation produces 3 Q&A pairs, verified by faithfulness + quality graders; "
        "the best pair (faith+quality) is kept.\n"
    )
    lines.append(
        "Verifier = `anthropic/claude-sonnet-4.6` by default; `gpt-5.5` when the generator is "
        "sonnet-4.6 (no self-grading). Generators use medium reasoning; verifiers use low.\n"
    )
    lines.append(
        "| Generator | verifier | ok | avg gen tok (in/out) | avg verify tok (in/out) | "
        f"gen $/batch | verify $/batch | total $/batch | est. {project_to} (keep best 1) | est. {project_to} (use all 3) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['label']} | `{s['verifier']}` | {s['n_ok']}/{s['n_total']} | "
            f"{s['avg_gen_in_tok']:.0f} / {s['avg_gen_out_tok']:.0f} | "
            f"{s['avg_verify_in_tok']:.0f} / {s['avg_verify_out_tok']:.0f} | "
            f"${s['avg_gen_cost']:.5f} | ${s['avg_verify_cost']:.5f} | ${s['avg_total_cost_per_batch']:.5f} | "
            f"${s[keep_col]:.2f} | ${s[all_col]:.2f} |"
        )
    lines.append("")
    lines.append(
        f"- A **batch** = 1 generation call + 2 verifier calls (faithfulness + quality), producing 3 graded queries.\n"
        f"- **est. {project_to} (keep best 1)** = total $/batch × {project_to} — 1 kept query per batch "
        "(matches qac_chempatents_best: generate 3, keep the best).\n"
        f"- **est. {project_to} (use all 3)** = total $/batch × {project_to}/3 — all 3 queries kept "
        "(matches qac_chempatents, all candidates).\n"
        "- Generation cost: OpenRouter is provider-measured; gpt-* computed from list price. "
        "Verification cost is attributed to the generator it grades.\n"
        "- The `cost_summary.csv` also has `est_*_gen_only` / `est_*_verify_only` splits."
    )
    (out_dir / "cost_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    *,
    corpus_path: Path,
    best_csv: Path,
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

    # Build only the clients we need (generators + their verifiers).
    need_providers: set[str] = set()
    for m in selected_models:
        need_providers.add(_provider_for(m["slug"]))
        need_providers.add(_provider_for(_verifier_for(m["slug"])))
    clients: Dict[str, OpenAI] = {}
    if "openai" in need_providers:
        clients["openai"] = _get_openai_client()
    if "openrouter" in need_providers:
        clients["openrouter"] = _get_openrouter_client()

    print("Loading corpus...")
    corpus_groups = load_multilingual_corpus(corpus_path)
    docs = select_documents(best_csv, corpus_groups, count=count, seed=seed)
    _write_selected(output_dir, docs)
    print(f"Selected {len(docs)} documents -> {output_dir / 'selected_documents.csv'}")

    summaries: List[Dict[str, Any]] = []
    for model in selected_models:
        gen_slug = model["slug"]
        verifier_slug = _verifier_for(gen_slug)
        gen_client = clients[_provider_for(gen_slug)]
        verify_client = clients[_provider_for(verifier_slug)]
        out_csv = output_dir / f"generations__{model['label']}.csv"
        rows: List[Dict[str, Any]] = []
        desc = f"{model['label']:>16} (v:{verifier_slug.split('/')[-1]})"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PER_DOC_FIELDS, extrasaction="ignore")
            writer.writeheader()
            f.flush()
            for doc in tqdm(docs, desc=desc, unit="doc"):
                record = process_doc(gen_client, verify_client, gen_slug, verifier_slug, doc, max_tokens=max_tokens)
                rows.append(record)
                writer.writerow(record)
                f.flush()
                if not str(record["status"]).startswith("ok"):
                    tqdm.write(f"  {record['publication_number']} [{record['question_language']}/{record['mode']}]: {record['error']}")
        n_ok = sum(1 for r in rows if str(r["status"]).startswith("ok"))
        print(f"  {model['label']}: {n_ok}/{len(rows)} ok (verifier={verifier_slug}) -> {out_csv}")
        summaries.append(_summarize(model, verifier_slug, rows, project_to))

    _write_summary(output_dir, summaries, project_to, len(docs))
    print(f"\nWrote cost summary -> {output_dir / 'cost_summary.md'}")
    keep_col = f"est_{project_to}_keep_best_1"
    all_col = f"est_{project_to}_use_all_3"
    for s in summaries:
        print(
            f"  {s['label']:>16}: ${s['avg_gen_cost']:.5f} gen + ${s['avg_verify_cost']:.5f} verify "
            f"= ${s['avg_total_cost_per_batch']:.5f}/batch -> {project_to} queries: "
            f"${s[keep_col]:.2f} (keep 1) / ${s[all_col]:.2f} (use 3)"
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Compare LLMs on chemistry-patent question generation + verification: same documents, "
            "same prompt/mode/query-language, per-model token usage and cost estimates."
        )
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/google_patents/multilingual_corpus.csv"), help="Multilingual corpus CSV (read-only).")
    parser.add_argument("--best-csv", type=Path, default=Path("data/google_patents/qac/qac_chempatents_best.csv"), help="CSV providing (mode, strategy, question_language) per doc.")
    parser.add_argument("--count", type=int, default=30, help="Number of documents (default: 30).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42).")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qa_generation_model_comparison"), help="Output directory.")
    parser.add_argument("--models", nargs="*", default=None, help=f"Subset of model labels. Known: {[m['label'] for m in MODELS]}")
    parser.add_argument("--project-to", type=int, default=1000, help="Queries to project cost to (default: 1000).")
    parser.add_argument("--max-tokens", type=int, default=16000, help="max_tokens for OpenRouter calls.")
    args = parser.parse_args()

    run(
        corpus_path=args.corpus,
        best_csv=args.best_csv,
        count=args.count,
        seed=args.seed,
        output_dir=args.output_dir,
        model_labels=args.models,
        project_to=args.project_to,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
