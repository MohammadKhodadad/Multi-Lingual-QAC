"""
OpenAI-based Q&A generation.

Default pipeline (`same_language=False`, used for EPO and Wikidata):
generate question and answer in English from the corpus context, validate, then
translate to all target languages.

JRC pipeline (`same_language=True`): generate and validate Q&A in each row's
language, with optional synthetic translated query variants.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# Languages to translate into (exclude en; we generate in English)
DEFAULT_TARGET_LANGS = [
    "de", "fr", "es", "ja", "ko", "zh", "ru", "pt", "it", "nl", "ar", "fa", "tr", "pl", "hi",
]

LANG_NAMES = {
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "ar": "Arabic",
    "fa": "Farsi",
    "tr": "Turkish",
    "pl": "Polish",
    "hi": "Hindi",
    "en": "English",
}

DEFAULT_GENERATION_MODEL = "gpt-5-mini"
DEFAULT_QUALITY_MODEL = "gpt-5-mini"
DEFAULT_SUPPORT_MODEL = "gpt-5-mini"
DEFAULT_TRANSLATION_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_GENERATION_REASONING_EFFORT = "medium"
DEFAULT_TRANSLATION_REASONING_EFFORT = "medium"


def _normalize_domain_hint(domain_hint: Optional[str]) -> str:
    domain = (domain_hint or "").strip().lower().replace("-", "_")
    return domain or "generic"


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env for Q&A generation.")
    return OpenAI(api_key=api_key)


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """Load corpus CSV into list of dicts."""
    rows: List[Dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            if "_id" in parsed and "id" not in parsed:
                parsed["id"] = parsed["_id"]
            if "text" in parsed and "context" not in parsed:
                parsed["context"] = parsed["text"]
            rows.append(parsed)
    return rows


def sample_corpus(
    rows: List[Dict[str, Any]],
    sample_size: int,
    *,
    stratify_by_language: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample rows from corpus. If stratify_by_language, take proportionally from each language.
    """
    if seed is not None:
        random.seed(seed)
    if sample_size >= len(rows):
        return rows
    if not stratify_by_language:
        return random.sample(rows, sample_size)
    # Stratify: group by language
    by_lang: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        lang = row.get("language", "en")
        by_lang.setdefault(lang, []).append(row)
    # Sample proportionally
    total = len(rows)
    sampled: List[Dict[str, Any]] = []
    for lang, lang_rows in by_lang.items():
        n = max(1, round(sample_size * len(lang_rows) / total))
        n = min(n, len(lang_rows))
        sampled.extend(random.sample(lang_rows, n))
    # If we got more or fewer, trim or pad
    random.shuffle(sampled)
    return sampled[:sample_size]


def generate_qa_english(
    client: OpenAI,
    context: str,
    *,
    previous_question: Optional[str] = None,
    previous_answer: Optional[str] = None,
    previous_feedback: Optional[str] = None,
    model: str = DEFAULT_GENERATION_MODEL,
    domain_hint: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate one validated-target Q&A pair in English from the given context.
    Returns question, answer, and supporting_text.
    """
    prompt = """You are an expert at creating retrieval questions from patent, legal, regulatory, encyclopedia, or technical passages.

The source context may be in any language, but your output must be in English only.

Generate exactly ONE question-answer pair from the context.

Rules:
- Output must be in natural English only.
- Do not copy the source language unless a technical term, formula, identifier, or proper noun should remain unchanged.
- The question must read like a realistic retrieval query that a researcher, engineer, lawyer, policy reader, compliance reader, or technical reader might actually type into a search system.
- Prefer short, natural, user-like wording over patent-summary or clause-lookup wording.
- Prefer a specific question about one of these: purpose, application, composition, method step, property, technical advantage, operating condition, material relationship, mechanism, effect, functional role, obligation, requirement, exception, applicability, legal effect, threshold, or consequence.
- The question must be answerable from the text and specific enough to be useful for retrieval.
- Prefer semantically challenging questions that dense retrieval should handle better than simple keyword matching.
- For this task, semantic reformulation is more important than choosing the easiest extractive fact.
- First look for a question about rationale, role, effect, mechanism, interaction, implication, or process purpose tied to a specific step.
- For legal or regulatory passages, first look for a question about what a rule requires, permits, excludes, triggers, proves, or causes in practice.
- Only fall back to exact ranges, exact ratios, exact named lists, or exact composition tables when the context does not support a stronger semantic question.
- If both are possible, prefer the question that requires understanding what the detail does or why it matters, not the one that only asks for the raw value.
- If the passage is legal or regulatory, prefer the question that requires understanding what the operative provision means or does, not where it is written.
- Prefer a question about one core technical fact, not a bundled summary of multiple advantages or multiple properties, unless the source presents them as one inseparable claim.
- Ask about function, effect, mechanism, role, use condition, or technical implication when possible, not just surface wording.
- For legal or regulatory passages, ask about one operative condition, addressee, exception, consequence, applicability limit, evidentiary requirement, or legal effect.
- Vary the question form across examples. Do not default to "How does ..." if another natural opening fits the fact better.
- Match the question opening to the fact type. Use forms such as:
  - "Why is ..." for step rationale or process purpose tied to a specific step
  - "Which ..." for identified biomarker pairs, materials, components, or options
  - "What function does ..." for a component's role
  - "What condition ..." or "At what ..." for operating constraints or measured ranges
  - "What property allows ..." for enabling characteristics
  - "How does ..." for mechanism, effect, or interaction only when that is the most natural form
- If you choose a method-style question, name the actual step, material, condition, or operation from the context.
- Do not ask vague questions like "What is the purpose of the method?" or "What is something about the process?" when the method contains a specific named step that can be asked about directly.
- For legal or regulatory passages:
  - prefer asking what condition must be met, what happens if it is not met, who is bound or exempted, what evidence is required, what the rule applies to, or what effect follows
  - do not mention article numbers, paragraph numbers, recital numbers, annex labels, or phrases like "this article", "this regulation", "this directive", "this decision", or "under the provision" unless the reference itself is the thing being asked about
  - the question should still work as a good query if all article numbers and labels were removed from the document
- Avoid citation-led openings such as:
  - "According to Article ..."
  - "Under Article ..."
  - "Pursuant to Article ..."
  - "In this article ..."
  - "What does this article require ..."
- Avoid clause-lookup questions that mainly ask for the provision number, legal basis citation, title, or exact article wording.
- Avoid making the question easy for exact-match retrieval by simply lifting the most distinctive nouns from the source into a template question.
- Preserve technical terms only when they are necessary for faithfulness or the question would become unnatural or ambiguous without them.
- Prefer grounded paraphrase over direct lexical overlap.
- Avoid spec-sheet questions when a more semantic alternative exists, especially:
  - exact value or ratio lookups
  - exact range lookups
  - exact component inventory or long named-list lookups
  - exact "what does it contain/include" questions
- A numeric question is acceptable only when the number itself is the important retrieval target and the context does not support a better question about function, rationale, or effect.
- In legal or regulatory text, a numeric question is acceptable only when the number itself is the important retrieval target and there is no better question about threshold, obligation, exception, trigger, or consequence.
- Avoid broad fallback wording like:
  - "What is the purpose of ..."
  - "What advantages does ... offer ..."
  - "What benefits does ... provide ..."
  when you can instead ask what a step achieves, why it is done, what a component does, or what effect it has.
- Avoid turning classification/grouping text into a weak taxonomy question if the same text supports a more functional question.
- When the context contains both "what it is" and "what it does", prefer "what it does".
- Avoid generic questions such as:
  - "What is the main object of the invention?"
  - "What is the main feature of the invention?"
  - "What are the main components?"
  unless the text is too short for anything better.
- Avoid document-centered phrasing such as:
  - "described in the invention"
  - "mentioned in the invention"
  - "according to the invention"
  - "in the text"
  - "described in the text"
  - "described in the present disclosure"
  - "used in the invention"
  Rewrite those into natural user-style English instead.
- Also avoid legal-document-centered phrasing such as:
  - "according to Article ..."
  - "under this Directive ..."
  - "in this Regulation ..."
  - "what does this article say ..."
  Rewrite those into natural user-style English that asks about the substance of the rule.
- Do not begin the question with broad template wording such as:
  - "What is the application of ..."
  - "What are the advantages of ..."
  - "What is the benefit of ..."
  - "What is the main technical advantage of ..."
  - "What is the purpose of ..."
  - "What types of products ..."
  - "What is the role of ..." when it only asks for a broad use summary
- Avoid these patterns especially when a more specific question can be asked about:
  - one process step
  - one operating condition
  - one material property
  - one mechanism
  - one component interaction
- Do not turn the title into a question.
- Do not simply wrap a copied title phrase or copied noun phrase in a question template.
- If a title-like wording comes to mind first, rewrite it into a more natural and more semantically reformulated question.
- Do not copy a sentence from the context nearly verbatim.
- Do not just wrap a copied noun phrase in a question template.
- Do not keep unusually high word overlap with the opening sentence or title unless a few technical anchor terms must remain for clarity.
- Do not make the question artificially difficult or obscure just to reduce word overlap.
- Before finalizing, ask yourself:
  - Did I choose the deepest answerable fact rather than the easiest extractive fact?
  - Would this still look like a good query if the exact numbers or list items were hidden?
  - Does this require some semantic understanding rather than simple table lookup?
  - Did I accidentally ask for a broad purpose/advantage summary when a narrower technical question was available?
  - If this is legal or regulatory text, did I ask about the operative meaning or effect without pointing directly to article numbers or labels?
- If the question starts with "What is the purpose of" or "What advantages does", rewrite it unless no narrower question is possible.
- If the question mentions an article number, paragraph number, recital number, annex label, or phrases like "this article" or "under this regulation", rewrite it unless the reference itself is essential.
- If the answer to those checks is no, regenerate a better question.
- The answer must be concise (1-2 sentences) and strictly grounded in the context.
- Include a short supporting_text quote copied from the source context that justifies the answer.
- Include a question_type chosen from: purpose, application, composition, method, property, advantage, operating_condition, material_relationship, other.
- Good style examples:
  - "What does the shape deformation layer do when the artificial nail is pressed onto the natural nail?"
  - "Why is cold rolling performed after hot rolling or forging in this steel production process?"
  - "What function does the buffer layer serve in the device structure?"
  - "What condition must be satisfied before the import can be authorized?"
  - "What happens if the authority does not object within the stated period?"
  - "Who may submit a request to stop the infringement?"
  - "What evidence has to accompany the shipment at first marketing?"
- Bad style examples:
  - "What are the recommended application methods described in the invention?"
  - "What type of products can include the component mentioned in the invention?"
  - "What is the application of [named compound/component]?"
  - "What types of products can utilize the composition described in the text?"
  - "What is the role of the component in the process?" when the context supports a more specific function or effect question
  - "What is the main technical advantage of this method?"
  - "What is the purpose of the process?"
  - "How does the method work?" when the context supports a more specific `Why`, `Which`, `What function`, `What condition`, or `At what` question
  - "What exact ratios or ranges are specified?" when the context also supports a better question about why the values matter
  - "Which specific items are listed?" when the context also supports a better question about role, interaction, or functional grouping
  - "According to Article 6, what conditions apply?"
  - "Under Article 4(1), how must the court treat the list?"
  - "What does this article require for authorization?"
  - "Which article sets out the exception?"
- Output valid JSON only, no markdown:
  {"question": "...", "answer": "...", "supporting_text": "...", "question_type": "..."}
"""
    retry_note = ""
    if previous_feedback:
        retry_note = (
            "\n\nPrevious attempt issue to fix:\n"
            f"{previous_feedback}\n"
            "Regenerate the question and answer so they fix that issue while staying fully grounded in the context. "
            "For legal or regulatory text, ask about the operative meaning, condition, or consequence without explicitly naming article numbers or labels."
        )
    previous_attempt_note = ""
    if previous_question or previous_answer:
        previous_attempt_note = (
            "\n\nPrevious failed attempt to improve upon:\n"
            f"Previous question: {previous_question or ''}\n"
            f"Previous answer: {previous_answer or ''}\n"
            "Use this only as feedback about what to avoid or improve. Do not lightly edit it or reuse its wording as a template. Generate a fresh corrected question-answer pair."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n\n{context[:4000]}"
                    f"{retry_note}"
                    f"{previous_attempt_note}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_GENERATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    return {
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "supporting_text": str(data.get("supporting_text", "")).strip(),
        "question_type": str(data.get("question_type", "other")).strip(),
    }


def generate_qa_in_language(
    client: OpenAI,
    context: str,
    target_lang_code: str,
    *,
    previous_question: Optional[str] = None,
    previous_answer: Optional[str] = None,
    previous_feedback: Optional[str] = None,
    model: str = DEFAULT_GENERATION_MODEL,
) -> Dict[str, str]:
    """
    Generate one Q&A pair in the same language as the corpus row (target_lang_code).
    """
    lang_name = LANG_NAMES.get(target_lang_code.lower(), target_lang_code)
    prompt = f"""You are an expert at creating retrieval questions from multilingual legal and regulatory passages, especially EU legislation, decisions, directives, regulations, opinions, and procedural acts.

The source context is written mainly in {lang_name} (BCP-47 code: {target_lang_code}).
You MUST write the question, answer, and supporting_text entirely in {lang_name}.

Goal:
- Write one strong legal retrieval question.
- The best question asks about one decisive legal point: one consequence, one trigger, one exclusion, one permission, one obligation, one evidentiary function, one scope limit, or one entitlement boundary.
- Prefer the strongest semantic legal question the passage supports, not the safest extractive one.

Core rules:
- Natural, fluent {lang_name} only.
- The question must read like a realistic search query a lawyer, regulator, policy reader, compliance analyst, or legal researcher would type.
- The question must be answerable strictly from the passage and specific enough for retrieval benchmarking.
- The question should still work if article numbers, paragraph numbers, recital numbers, and annex labels were hidden.
- Do not let article, paragraph, recital, annex, or provision labels lead the query when the same issue can be asked in substance-first wording.
- Prefer grounded paraphrase over direct lexical overlap.
- Prefer one compact main clause over a long joined question.
- Prefer a question whose answer can usually be stated in one focused sentence.

Do ask about:
- what a rule requires, permits, excludes, proves, limits, changes, or causes;
- one decisive condition for authorization, refusal, eligibility, or entitlement;
- one legal consequence of non-compliance, objection, recognition, inclusion, exclusion, or expiry;
- one evidentiary function of a certificate, report, declaration, or document;
- one scope boundary, exception, or trigger.

Do not ask:
- for article numbers, provision labels, exact clause wording, titles, or legal basis citations;
- for a raw date, amount, threshold, code, period, percentage, or listed item when a better question can ask what that detail changes legally;
- what must be notified, reported, submitted, included, listed, or communicated when the answer would naturally become several items or steps;
- for a full inventory of conditions, exceptions, guarantees, documents, powers, remedies, categories, or measures;
- both the rule and its exception, both purpose and applicability, both condition and duration, both action and consequence, or any other two-part combination when one sharper legal point should be chosen;
- whether something is mandatory or voluntary and how it is characterized, defined, or formulated, when one stronger legal point should be chosen;
- a definition plus its included elements, a category plus its contents, or a label plus the legal effect around it;
- what status, label, or category is assigned to a person, product, authority, or act when the answer would mainly be the assigned label rather than the legal consequence of that status;
- what an analysis, report, application, declaration, or document must contain when the answer would mainly become a list of required contents rather than one legal effect, function, or consequence;
- a question whose best answer would naturally become a checklist, menu, definition inventory, semicolon-separated list, or several alternative branches.

Before finalizing, check:
- Does this ask about one decisive legal point?
- Would answering it require understanding the legal effect, not just copying words?
- Would the best answer become a list, menu, or two-part answer? If yes, rewrite.
- Am I asking mainly for a status, label, category, or mandatory/voluntary characterization instead of the legal effect of that classification?
- Am I asking what something must contain or include when I should ask what that content requirement does legally or why it matters?
- Can I remove a second clause and keep the stronger legal question?
- If the draft asks for a date, amount, threshold, code, or item, can I instead ask what that detail changes legally?

Answer requirements:
- The answer must be concise (1-2 sentences) and fully grounded in the context.
- Include supporting_text: a short tight quote copied from one relevant part of the source that directly justifies the answer.
- Include question_type: one of obligation, requirement, scope, exception, consequence, evidence, definition, procedure, other.

Good style examples:
- a question about what happens if the requirement is not met
- a question about what condition must be satisfied before authorization is possible
- a question about what a certificate or report is meant to prove
- a question about when an exclusion or derogation applies
- a question about what legal effect follows from a threshold or timing rule

Bad style examples:
- a question beginning with the equivalent of "According to Article ..."
- a question asking which article sets out the exception
- a question asking for all conditions, all documents, or all consequences at once
- a question asking both a restriction and its exception
- a question asking for a definition plus its list of included items
- a question asking only for the exact date, amount, threshold, code, or listed items

Output valid JSON only, no markdown:
{{"question": "...", "answer": "...", "supporting_text": "...", "question_type": "..."}}
"""
    retry_note = ""
    if previous_feedback:
        retry_note = (
            "\n\nPrevious attempt issue to fix:\n"
            f"{previous_feedback}\n"
            f"Regenerate so the issue is fixed; keep everything in {lang_name}. Ask about one decisive legal point, not a list, menu, definition inventory, or two-part question. Do not lead the query with article, paragraph, recital, annex, or provision labels when a substance-first wording is possible. If the last attempt asked both A and B, keep only the stronger legal point. If the last attempt asked mainly for a status, label, category, or mandatory/voluntary characterization, rewrite it to ask for the legal effect, obligation, exclusion, scope limit, or consequence instead. If the last attempt asked what an analysis, report, application, declaration, or document must contain, rewrite it to ask what that content requirement proves, enables, limits, or requires legally instead of listing contents. If the last attempt would be answered by several items, branches, or alternatives, rewrite it so the answer becomes one focused legal consequence, requirement, trigger, exception, scope limit, or evidentiary function. If the last attempt asked only for a code, value, threshold, amount, duration, or listed item, rewrite it to ask what that detail changes legally. If the last attempt bundled an action with a deadline or later procedural detail, keep only the stronger legal point. Prefer the shorter single-clause version when possible. Keep supporting_text as one tight directly supporting excerpt."
        )
    previous_attempt_note = ""
    if previous_question or previous_answer:
        previous_attempt_note = (
            "\n\nPrevious failed attempt to improve upon:\n"
            f"Previous question: {previous_question or ''}\n"
            f"Previous answer: {previous_answer or ''}\n"
            "Generate a fresh corrected pair; do not lightly paraphrase the failed attempt."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context ({lang_name}):\n\n{context[:4000]}"
                    f"{retry_note}"
                    f"{previous_attempt_note}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_GENERATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    return {
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "supporting_text": str(data.get("supporting_text", "")).strip(),
        "question_type": str(data.get("question_type", "other")).strip(),
    }


def check_english_language(
    client: OpenAI,
    question: str,
    answer: str,
    *,
    model: str = DEFAULT_SUPPORT_MODEL,
) -> Tuple[bool, str]:
    """
    Validate that the generated question and answer are written in English.
    Returns (approved, reason).
    """
    prompt = """You are a strict language checker.

Decide whether BOTH the question and answer are written mainly in English.

Approve only if:
- both are natural English,
- they are not primarily written in another language,
- they are not mixed-language outputs except for unavoidable technical terms, formulas, identifiers, or proper nouns.

Output valid JSON only:
{"approved": true, "reason": "..."}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer: {answer}",
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    return approved, reason


def check_language_match(
    client: OpenAI,
    question: str,
    answer: str,
    target_lang_code: str,
    *,
    model: str = DEFAULT_SUPPORT_MODEL,
) -> Tuple[bool, str]:
    """
    Validate that question and answer are written mainly in the expected language.
    """
    lang_name = LANG_NAMES.get(target_lang_code.lower(), target_lang_code)
    prompt = f"""You are a strict language checker.

Decide whether BOTH the question and answer are written mainly in {lang_name} (language code {target_lang_code}).

Approve only if:
- both are natural {lang_name},
- they are not primarily in a different language,
- technical terms, formulas, units, identifiers, and proper nouns may appear in Latin script or standard notation when normal for that field.

Output valid JSON only:
{{"approved": true, "reason": "..."}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer: {answer}",
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    return approved, reason


def check_faithfulness(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    supporting_text: str,
    *,
    model: str = DEFAULT_SUPPORT_MODEL,
    domain_hint: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate that the answer is supported by the source context.
    Returns (approved, reason).
    """
    domain = _normalize_domain_hint(domain_hint)
    prompt = """You are a strict faithfulness checker for question-answer pairs derived from a technical or encyclopedia source passage.

Goal:
- Approve only fully grounded QA pairs.

Approve only if:
- the question is answerable from the context;
- the answer is fully supported by the context;
- the answer does not add missing facts, implications, causes, conditions, or conclusions;
- the supporting_text is a relevant quote from the context that genuinely supports the answer.

Reject if:
- the answer is generic, speculative, overstated, partially unsupported, or drifts beyond the text;
- the answer leaves the impression that the text says more than it actually says;
- the supporting_text is irrelevant, too weak, or does not support the claimed point.

Output valid JSON only:
{"approved": true, "reason": "..."}
"""
    if domain == "legal":
        prompt = """You are a strict faithfulness checker for question-answer pairs derived from a legal or regulatory source passage.

Goal:
- Approve only fully grounded legal QA pairs.

Approve only if:
- the question is answerable from the context;
- the answer is fully supported by the context;
- the answer states only what the text supports about the rule, condition, exception, scope, consequence, procedure, or entitlement;
- faithful paraphrase is allowed when every substantive claim remains traceable to the context;
- the answer does not add unsupported legal implications, missing exceptions, extra conditions, hidden limits, or downstream consequences;
- the supporting_text is a relevant quote from the context that genuinely supports the answer.

Reject if:
- the answer overstates what the rule means or how far it extends;
- the answer adds an exception, consequence, condition, procedural effect, entitlement boundary, cross-article implication, or legal implication not supported by the text;
- the answer is generic, speculative, partly supported, or materially looser than the source;
- the supporting_text is irrelevant, too weak, or does not support the claimed legal point.

Borderline rule:
- If the answer could mislead a reader into believing the law says more than the passage actually says, reject.

Output valid JSON only:
{"approved": true, "reason": "..."}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}\n\n"
                    f"Supporting text: {supporting_text}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    return approved, reason


def check_question_quality(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    *,
    model: str = DEFAULT_QUALITY_MODEL,
    output_language_name: Optional[str] = None,
    domain_hint: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate that the question is retrieval-useful, specific, and not overly generic.
    Returns (approved, reason).
    """
    lang_clause = ""
    if output_language_name:
        lang_clause = (
            f"The question and answer should be written in {output_language_name} "
            f"and must sound natural for domain readers of {output_language_name}.\n\n"
        )
    domain = _normalize_domain_hint(domain_hint)
    prompt = lang_clause + """You are a strict quality checker for retrieval questions built from technical, legal, regulatory, or encyclopedia text.

Goal:
- Approve only strong retrieval questions.

Approve only if the question:
- sounds like a realistic user retrieval query;
- is specific enough to distinguish the document;
- asks about one concrete point from the context;
- is phrased semantically, not as an exact-match lookup;
- is natural rather than document-centered;
- is not nearly copied from the context;
- and is useful for retrieval benchmarking.

Reject if the question is mainly:
- `title-lift`: the title or opening phrase turned into a question;
- `high-overlap`: too close to the source wording and still easy by exact matching;
- `overly-extractive`: mostly a raw value, list, span, threshold, range, or table lookup when a better semantic question is available;
- `broad-summary`: a broad purpose/advantage/application/feature summary when a narrower question is available;
- `bundled-facts`: several loosely related facts joined together instead of one core information need;
- `weak-query-shape`: understandable, but not shaped like a strong realistic retrieval query;
- `legalistic-lookup`: for legal text, driven mainly by article/provision lookup rather than substantive meaning.

Guidance:
- Do not reject numeric questions automatically. Reject them only when the number, date, range, code, or listed item is a shallow lookup and the context supports a better question about effect, role, condition, threshold meaning, or consequence.
- Reject document-centered wording such as "according to the invention" or "in the text" when a more natural query is possible.
- Reject broad templates such as "What is the purpose of ...?" or "What are the advantages of ...?" when they lead to a broad summary instead of one sharper point.
- For legal text, reject questions led mainly by article labels, clause numbers, or provision references when the same issue can be asked without them.

Borderline rule:
- Approve borderline cases only if the question already looks like the strongest realistic retrieval query supported by the passage.

If you reject the question:
- set `failure_type` to exactly one of:
  - `title-lift`
  - `high-overlap`
  - `overly-extractive`
  - `broad-summary`
  - `bundled-facts`
  - `weak-query-shape`
  - `legalistic-lookup`
- keep `reason` short and concrete
- provide `better_direction` as ONE short actionable hint

If you approve the question:
- set `failure_type` to `none`
- set `better_direction` to an empty string

Output valid JSON only:
{"approved": true, "reason": "...", "failure_type": "none", "better_direction": ""}
"""
    if domain == "legal":
        prompt = lang_clause + """You are a strict quality checker for retrieval questions built from legal and regulatory text.

Goal:
- Approve only the strongest realistic legal retrieval questions.

Approve only if the question:
- sounds like a realistic legal or policy retrieval query;
- asks about one operative legal point from the context;
- is specific enough to distinguish the document;
- is semantic rather than clause-lookup driven;
- would still be strong if article, recital, paragraph, and annex labels were hidden;
- is not nearly copied from the source;
- and is useful for legal retrieval benchmarking.

Reject if:
- there is a clearly sharper, more single-focus legal query available from the same passage;
- the question is mainly driven by article labels, clause numbers, provision references, or citation wording;
- the question can be answered mainly by copying one span, code, date, number, threshold, or listed item instead of understanding the legal effect;
- the question mainly asks what status, label, category, or mandatory/voluntary characterization applies when the stronger retrieval need is the legal effect, obligation, exclusion, or consequence created by that classification;
- the question mainly asks what an analysis, report, application, declaration, or document must contain when the answer is mostly a contents list rather than one legal effect, evidentiary function, or consequence;
- the question combines a classification choice with a second ask such as how it is formulated, defined, or described;
- the question combines a legal action with its deadline, time limit, or later procedural detail when one sharper legal point should be chosen;
- the question is document-led, recital-like, or provision-like rather than a natural legal information need;
- the question reads like a recital/provision restatement instead of a natural legal information need.

Map the main failure to exactly one `failure_type`:
- `title-lift`: mostly the title or opening legal phrase turned into a question
- `high-overlap`: too close to the source wording
- `overly-extractive`: mainly asks for a raw value, code, date, threshold, list, duration, or label when a better legal-effect question is available
- `broad-summary`: asks for a broad purpose, role, or general summary instead of one sharper legal point
- `bundled-facts`: asks for multiple conditions, branches, or legal points at once
- `weak-query-shape`: understandable but not phrased like a strong realistic legal retrieval query
- `legalistic-lookup`: driven mainly by provision lookup or citation wording

Borderline rule:
- Approve borderline cases if the current wording already looks like a realistic legal retrieval query and no clearly better single-focus alternative is available.

If you reject the question:
- set `failure_type` to exactly one of:
  - `title-lift`
  - `high-overlap`
  - `overly-extractive`
  - `broad-summary`
  - `bundled-facts`
  - `weak-query-shape`
  - `legalistic-lookup`
- keep `reason` short and concrete
- provide `better_direction` as ONE short actionable hint

If you approve the question:
- set `failure_type` to `none`
- set `better_direction` to an empty string

Output valid JSON only:
{"approved": true, "reason": "...", "failure_type": "none", "better_direction": ""}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    failure_type = str(data.get("failure_type", "")).strip()
    better_direction = str(data.get("better_direction", "")).strip()
    if failure_type and failure_type != "none":
        reason = f"{failure_type}: {reason}" if reason else failure_type
    if better_direction:
        reason = f"{reason} Better direction: {better_direction}".strip()
    return approved, reason


def check_legal_question_shape(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    *,
    model: str = DEFAULT_QUALITY_MODEL,
    output_language_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate that a legal question has a sharp single-focus shape rather than a
    checklist, menu, definition inventory, or shallow lookup.
    Returns (approved, reason).
    """
    lang_clause = ""
    if output_language_name:
        lang_clause = (
            f"The question and answer are written in {output_language_name} "
            f"and should be judged as natural legal retrieval text in {output_language_name}.\n\n"
        )
    prompt = lang_clause + """You are a strict shape checker for legal/regulatory retrieval questions.

Your task is NOT to judge factual correctness or language. Judge only whether the question has the right legal retrieval shape.

Goal:
- Approve only questions that target one decisive legal point.

Approve only if the question:
- is shaped like a focused legal retrieval need;
- asks about one main trigger, exception, exclusion, consequence, scope boundary, entitlement boundary, obligation, or responsible authority;
- can be answered with one focused legal point rather than a list, menu, checklist, inventory, or set of alternative branches;
- and already looks like the sharpest realistic legal question supported by the passage.

Reject if the question is mainly:
- `condition-list`: asks for conditions, criteria, or applicability factors when one stronger trigger, exception, refusal ground, exclusion, entitlement boundary, or consequence should be chosen;
- `menu-of-measures`: asks which measures, methods, routes, remedies, or actions are available or required when the answer is mainly a menu;
- `definition-inventory`: asks who/what counts as something, what is included in a category, what status/label/category is assigned, or what a report/document/application must contain when the answer is mainly a membership list, included-items list, assigned label, or contents list;
- `date-value-lookup`: asks for a date, duration, amount, threshold, code, period, or listed item when the stronger question is what that detail changes legally;
- `multi-branch`: asks two legal points at once, contrasts two legal paths, or would naturally require several branches, alternatives, sub-rules, or both an action and its deadline/detail in the answer;
- `broad-legal-shape`: broad, diffuse, or general in a way that misses the sharper single legal point available in the passage.

Decision rule:
- If the best answer naturally becomes a checklist, menu, inventory, or branch structure, reject.
- If the question asks both A and B, reject and keep only the stronger legal point.
- If the question asks who/what is included, what counts as something, which measures are available, or what conditions apply, reject whenever the answer naturally becomes members, measures, or conditions rather than one decisive legal point.
- If the question asks what a report, analysis, declaration, application, or document must contain, reject whenever the answer mainly becomes contents rather than one decisive legal effect, evidentiary function, or consequence.
- If the question asks what status/label/category applies, or whether something is mandatory/voluntary and how it is characterized, reject whenever the answer mainly becomes a label plus description instead of one decisive legal consequence.
- If the answer naturally becomes "may do X, and must do so by Y" or a similar action-plus-deadline bundle, reject and keep only the stronger legal point.
- If a materially sharper single-issue reformulation exists, reject.
- If the question is already aimed at one decisive legal point, approve.

Examples to reject:
- "Which measures must the Member State take ... ?"
- "What conditions must be met for ... ?"
- "Who counts as a dependent child?"
- "What facilities are included in ... ?"
- "Until what date does the entitlement last?"
- "What amount/code/threshold applies ... ?" when the stronger question is the legal effect of that detail

Examples to approve:
- "What consequence follows if the requirement is not met?"
- "When must entry be refused?"
- "What exclusion applies in this case?"
- "What legal effect does this limitation produce?"
- "Which authority is responsible once X happens?" if that is the one decisive legal point

If you reject the question:
- set `failure_type` to exactly one of:
  - `condition-list`
  - `menu-of-measures`
  - `definition-inventory`
  - `date-value-lookup`
  - `multi-branch`
  - `broad-legal-shape`
- keep `reason` short and concrete
- provide `better_direction` as ONE short actionable hint

If you approve the question:
- set `failure_type` to `none`
- set `better_direction` to an empty string

Output valid JSON only:
{"approved": true, "reason": "...", "failure_type": "none", "better_direction": ""}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    failure_type = str(data.get("failure_type", "")).strip()
    better_direction = str(data.get("better_direction", "")).strip()
    if failure_type and failure_type != "none":
        reason = f"{failure_type}: {reason}" if reason else failure_type
    if better_direction:
        reason = f"{reason} Better direction: {better_direction}".strip()
    return approved, reason


def translate_qa(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    target_langs: List[str],
    *,
    source_lang: str = "en",
    previous_feedback: Optional[str] = None,
    previous_translated_question: Optional[str] = None,
    previous_translated_answer: Optional[str] = None,
    model: str = DEFAULT_TRANSLATION_MODEL,
    domain_hint: Optional[str] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Translate (question, answer) to target languages. Returns {lang: (q, a)}.
    """
    if not target_langs:
        return {}
    source_lang = (source_lang or "en").strip().lower()
    source_lang_name = LANG_NAMES.get(source_lang, source_lang)
    lang_list = ", ".join(LANG_NAMES.get(l, l) for l in target_langs)
    prompt = f"""Translate the following {source_lang_name} retrieval question and answer pair into these languages: {lang_list}.

Goal:
- preserve the exact information need, answer meaning, specificity, and retrieval difficulty of the source pair
- produce natural, native-sounding target-language retrieval text rather than literal translation

Requirements:
- Use the source context to resolve ambiguity and preserve the original information need exactly.
- Keep the same meaning, level of specificity, and technical terms where appropriate.
- Do not make the question more generic, broader, or more citation-led than the original.
- Preserve the semantic difficulty of the original question.
- Do not simplify the question into a keyword-heavy or literal surface-form restatement.
- Prefer natural target-language phrasing over word-for-word translation.
- Do not omit or alter numbers, units, ranges, formulas, identifiers, or named technical materials.
- Preserve technical terms, abbreviations, symbols, and patent-style identifiers when translating them would be incorrect or unnatural.
- Keep the answer faithful to the source answer and consistent with the source context.
- Do not add explanation, background, or extra claims not present in the source pair or source context.
- If the source question is technical and concise, keep the target-language question technical and concise too.

Avoid translation artifacts:
- choose one natural term, not slash-separated alternatives like `X/Y`
- do not leave editor-style repair traces or synonym bundles
- do not include unnecessary foreign-language glosses in parentheses
- avoid code-mixed verbs or phrasing when the target language has a normal technical equivalent
- rewrite into natural target-language syntax instead of following source-language word order too closely
- keep the text fully in the target language except for unavoidable technical terms, formulas, units, identifiers, abbreviations, or proper nouns
- do not leak words from unrelated languages or scripts into the translation
- if a technical term can stay in Latin script, integrate it naturally into an otherwise target-language sentence
- if the source answer contains multiple supported facts, preserve them cleanly without turning the translation into a glossary or note
- prefer one polished final phrasing, not an exploratory or half-edited wording
"""
    if _normalize_domain_hint(domain_hint) == "legal":
        prompt += """

This is legal/regulatory retrieval data.
- Preserve the exact legal information need from the source question.
- Keep the question focused on one operative legal point, not a bundled checklist.
- Do not introduce article numbers, recital numbers, annex labels, provision labels, or phrases like "this article" unless they are already essential in the source question.
- Do not turn the translation into a clause-lookup query or a broader policy summary.
- Prefer natural legal phrasing in the target language over literal source-language syntax.
"""
    prompt += f"""

Output valid JSON only:
{{"translations": {{"de": {{"question": "...", "answer": "..."}}, "fr": {{...}}, ...}}}}

Languages to include: {json.dumps(target_langs)}
"""
    retry_note = ""
    if previous_feedback:
        retry_note = (
            "\n\nPrevious attempt issue to fix:\n"
            f"{previous_feedback}\n"
            "Revise the translation to fix that issue while preserving meaning and technical details."
        )
    previous_attempt_note = ""
    if previous_translated_question or previous_translated_answer:
        previous_attempt_note = (
            "\n\nPrevious failed translation to improve upon:\n"
            f"Previous translated question: {previous_translated_question or ''}\n"
            f"Previous translated answer: {previous_translated_answer or ''}\n"
            "Use this only as repair context. Do not copy it mechanically. Rewrite it so it sounds more natural in the target language while preserving the exact information need, meaning, and technical details."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Source context:\n{context[:5000]}\n\n"
                    f"{source_lang_name} question: {question}\n\n"
                    f"{source_lang_name} answer: {answer}"
                    f"{retry_note}"
                    f"{previous_attempt_note}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_TRANSLATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    trans = data.get("translations", data)
    result: Dict[str, Tuple[str, str]] = {}
    for lang in target_langs:
        if lang in trans and isinstance(trans[lang], dict):
            q = trans[lang].get("question", "")
            a = trans[lang].get("answer", "")
            result[lang] = (str(q).strip(), str(a).strip())
    return result


def check_translation_quality(
    client: OpenAI,
    context: str,
    source_question: str,
    source_answer: str,
    translated_question: str,
    translated_answer: str,
    target_lang: str,
    *,
    source_lang: str = "en",
    model: str = DEFAULT_QUALITY_MODEL,
    domain_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate that a translated QA pair is fluent, faithful, and in the target language.
    Returns structured quality signals for approval and retry decisions.
    """
    target_lang_name = LANG_NAMES.get(target_lang, target_lang)
    source_lang = (source_lang or "en").strip().lower()
    source_lang_name = LANG_NAMES.get(source_lang, source_lang)
    prompt = f"""You are a strict but practical translation quality checker for multilingual retrieval data.

The source context may be in any language. The reference question and answer are in {source_lang_name}.
The candidate translation must be in {target_lang_name}.

Judge these dimensions separately:
- `language_ok`: the translated question and answer are clearly written in {target_lang_name}
- `meaning_ok`: the meaning matches the {source_lang_name} question and {source_lang_name} answer closely
- `technical_ok`: numbers, units, ranges, formulas, identifiers, and important technical terms are preserved
- `specificity_ok`: the translated question keeps the same information need and specificity and does not become more generic, broader, or more citation-led
- `terminology_ok`: the translation uses appropriate technical terminology and register for {target_lang_name}
- `artifact_ok`: the translation does not contain repair artifacts such as slash-separated alternatives, unnecessary English glosses, editor-style synonym bundles, or gratuitous code mixing
- `fluency_ok`: the translation sounds natural enough for a native technical reader and is not clearly word-for-word or grammatically broken
- `grammar_ok`: grammar, agreement, case, morphology, and local sentence form are acceptable for {target_lang_name}

Be especially strict about these artifact failures:
- slash alternatives like `X/Y` when one natural wording should be chosen
- parenthetical foreign-language glosses like `(oiling)` when they are not required for correctness
- mixed-language repair wording or unresolved synonym pairs
- foreign-script leakage from an unrelated language when the span is not just a formula, identifier, unit, abbreviation, or proper noun
- faithful but clearly literal syntax that still reads like source-language structure mapped into {target_lang_name}

Severity guidelines:
- `high`: wrong language, meaning drift, dropped or altered technical details, or much more generic wording
- `medium`: meaning is mostly correct but there are clear grammar problems or very awkward/literal phrasing
- `low`: minor stiffness or small fluency issues only

Choose exactly one `failure_type`:
- `none`
- `wrong-language`
- `meaning-error`
- `missing-technical-detail`
- `too-generic`
- `unnatural-phrasing`
- `grammar-morphology`
- `terminology-register`
- `translation-artifact`

If you reject:
- keep `reason` short and concrete
- provide `better_direction` as ONE short actionable repair hint
- examples:
  - `rewrite more naturally for native technical phrasing`
  - `fix grammar and agreement`
  - `restore the missing technical detail exactly`
  - `use the standard technical term in {target_lang_name}`
  - `keep the question as specific as the source original`
  - `choose one natural term instead of slash alternatives`
  - `remove the foreign-language gloss and use native technical wording`
  - `rewrite in natural {target_lang_name} syntax`

If you approve:
- set `failure_type` to `none`
- set `better_direction` to an empty string

Approval policy:
- Reject when `language_ok`, `meaning_ok`, `technical_ok`, or `specificity_ok` is false.
- Reject when `terminology_ok` is false.
- Reject when `artifact_ok` is false.
- Reject when `grammar_ok` is false and severity is `medium` or `high`.
- Reject when `fluency_ok` is false and severity is `medium` or `high`.
- Approve when the only issue is minor fluency stiffness with `severity = low`.
"""
    if _normalize_domain_hint(domain_hint) == "legal":
        prompt += """

Legal/regulatory translation rules:
- Reject if the translation introduces or removes legal scope, conditions, exceptions, consequences, addressees, evidentiary requirements, or authorities.
- Reject if the translated question turns into a provision-lookup question, introduces article/provision labels absent from the source question, or becomes a broader legal summary.
- Reject if one focused legal point in the source becomes a checklist, menu of measures, definition inventory, or multi-branch question in translation.
"""
    prompt += """

Output valid JSON only:
{"language_ok": true, "meaning_ok": true, "technical_ok": true, "specificity_ok": true, "terminology_ok": true, "artifact_ok": true, "fluency_ok": true, "grammar_ok": true, "severity": "low", "failure_type": "none", "better_direction": "", "reason": "..."}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"{source_lang_name} question: {source_question}\n\n"
                    f"{source_lang_name} answer: {source_answer}\n\n"
                    f"{target_lang_name} question: {translated_question}\n\n"
                    f"{target_lang_name} answer: {translated_answer}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    language_ok = bool(data.get("language_ok", False))
    meaning_ok = bool(data.get("meaning_ok", False))
    technical_ok = bool(data.get("technical_ok", False))
    specificity_ok = bool(data.get("specificity_ok", False))
    terminology_ok = bool(data.get("terminology_ok", True))
    artifact_ok = bool(data.get("artifact_ok", True))
    fluency_ok = bool(data.get("fluency_ok", False))
    grammar_ok = bool(data.get("grammar_ok", True))
    severity = str(data.get("severity", "high")).strip().lower() or "high"
    if severity not in {"low", "medium", "high"}:
        severity = "high"
    reason = str(data.get("reason", "")).strip()
    failure_type = str(data.get("failure_type", "none")).strip().lower() or "none"
    if failure_type not in {
        "none",
        "wrong-language",
        "meaning-error",
        "missing-technical-detail",
        "too-generic",
        "unnatural-phrasing",
        "grammar-morphology",
        "terminology-register",
        "translation-artifact",
    }:
        failure_type = "meaning-error"
    better_direction = str(data.get("better_direction", "")).strip()
    approved = (
        language_ok
        and meaning_ok
        and technical_ok
        and specificity_ok
        and terminology_ok
        and artifact_ok
    )
    if approved and not grammar_ok and severity in {"medium", "high"}:
        approved = False
    if approved and not fluency_ok and severity in {"medium", "high"}:
        approved = False

    retry_recommended = (
        (not language_ok)
        or (not meaning_ok)
        or (not technical_ok)
        or (not specificity_ok)
        or (not terminology_ok)
        or (not artifact_ok)
        or ((not grammar_ok) and severity in {"medium", "high"})
        or ((not fluency_ok) and severity in {"medium", "high"})
    )
    return {
        "approved": approved,
        "retry_recommended": retry_recommended,
        "reason": reason,
        "severity": severity,
        "failure_type": failure_type,
        "better_direction": better_direction,
        "language_ok": language_ok,
        "meaning_ok": meaning_ok,
        "technical_ok": technical_ok,
        "specificity_ok": specificity_ok,
        "terminology_ok": terminology_ok,
        "artifact_ok": artifact_ok,
        "fluency_ok": fluency_ok,
        "grammar_ok": grammar_ok,
    }


def _row_target_languages(row: Dict[str, Any], default_target_languages: List[str]) -> List[str]:
    raw = row.get("target_languages_json", "")
    if not raw:
        return list(default_target_languages)
    try:
        parsed = json.loads(raw)
    except Exception:
        return list(default_target_languages)
    if not isinstance(parsed, list):
        return list(default_target_languages)
    cleaned = []
    seen = set()
    for item in parsed:
        lang = str(item).strip().lower()
        if lang and lang not in seen:
            cleaned.append(lang)
            seen.add(lang)
    return cleaned


def _row_target_corpus_ids(row: Dict[str, Any]) -> Dict[str, str]:
    raw = row.get("target_corpus_ids_json", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: Dict[str, str] = {}
    for key, value in parsed.items():
        lang = str(key).strip().lower()
        corpus_id = str(value).strip()
        if lang and corpus_id:
            result[lang] = corpus_id
    return result


def _row_synthetic_target_languages(
    row: Dict[str, Any],
    default_target_languages: List[str],
) -> List[str]:
    raw = row.get("synthetic_target_languages_json", "")
    if not raw:
        return list(default_target_languages)
    try:
        parsed = json.loads(raw)
    except Exception:
        return list(default_target_languages)
    if not isinstance(parsed, list):
        return list(default_target_languages)
    cleaned = []
    seen = set()
    for item in parsed:
        lang = str(item).strip().lower()
        if lang and lang not in seen:
            cleaned.append(lang)
            seen.add(lang)
    return cleaned


def _row_output_metadata(row: Dict[str, Any]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for key in (
        "pair_id",
        "celex",
        "source_language",
        "source_corpus_id",
        "target_language",
        "target_corpus_id",
        "query_id_hint",
        "linked_corpus_ids_json",
    ):
        value = str(row.get(key, "")).strip()
        if value:
            metadata[key] = value
    return metadata


def _run_translation_pass(
    client: OpenAI,
    *,
    context: str,
    source_question: str,
    source_answer: str,
    source_lang: str,
    translation_targets: List[str],
    translation_model: str,
    quality_model: str,
    domain_hint: Optional[str] = None,
    max_attempts: int = 3,
) -> tuple[Dict[str, Tuple[str, str]], List[str]]:
    approved_translations: Dict[str, Tuple[str, str]] = {}
    failed_languages: List[str] = []
    source_lang = (source_lang or "en").strip().lower()
    source_lang_name = LANG_NAMES.get(source_lang, source_lang)
    deduped_targets: List[str] = []
    seen_targets: set[str] = set()
    for lang in translation_targets:
        normalized = str(lang).strip().lower()
        if not normalized or normalized == source_lang or normalized in seen_targets:
            continue
        deduped_targets.append(normalized)
        seen_targets.add(normalized)

    for lang in deduped_targets:
        lang_name = LANG_NAMES.get(lang, lang)
        lang_failure = "translation missing"
        retry_feedback: Optional[str] = None
        retry_q: Optional[str] = None
        retry_a: Optional[str] = None
        for _attempt in range(1, max_attempts + 1):
            trans = translate_qa(
                client,
                context,
                source_question,
                source_answer,
                [lang],
                source_lang=source_lang,
                previous_feedback=retry_feedback,
                previous_translated_question=retry_q,
                previous_translated_answer=retry_a,
                model=translation_model,
                domain_hint=domain_hint,
            )
            if lang not in trans:
                lang_failure = "translation missing"
                retry_feedback = (
                    "The previous translation attempt was missing or malformed. "
                    "Return valid JSON with a complete translated question and answer."
                )
                continue

            q, a = trans[lang]
            retry_q = q
            retry_a = a
            trans_check = check_translation_quality(
                client,
                context,
                source_question,
                source_answer,
                q,
                a,
                lang,
                source_lang=source_lang,
                model=quality_model,
                domain_hint=domain_hint,
            )
            if not trans_check["approved"]:
                reason = str(trans_check.get("reason", "")).strip()
                severity = str(trans_check.get("severity", "high")).strip()
                failure_type = str(trans_check.get("failure_type", "")).strip()
                better_direction = str(trans_check.get("better_direction", "")).strip()
                lang_failure = (
                    "translation quality failed: "
                    f"{reason or 'not fluent/faithful enough'}"
                    f" [severity={severity}]"
                )
                feedback_parts = []
                if failure_type and failure_type != "none":
                    feedback_parts.append(f"Failure type: {failure_type}.")
                if reason:
                    feedback_parts.append(f"Reason: {reason}.")
                if better_direction:
                    feedback_parts.append(f"Better direction: {better_direction}.")
                feedback_parts.append(
                    f"Revise the {lang_name} translation by preserving the exact meaning, specificity, numbers, units, "
                    f"and source-grounded details from the {source_lang_name} pair and source context."
                )
                if _normalize_domain_hint(domain_hint) == "legal":
                    feedback_parts.append(
                        "Keep one focused legal point. Do not add article numbers, provision labels, or phrases like "
                        "'this article' if they are absent from the source question."
                    )
                feedback_parts.append(
                    f"If the problem is fluency or grammar, rewrite more naturally in {lang_name} without changing the information need."
                )
                retry_feedback = " ".join(feedback_parts)
                continue

            translated_quality_ok, translated_quality_reason = check_question_quality(
                client,
                context,
                q,
                a,
                model=quality_model,
                output_language_name=lang_name,
                domain_hint=domain_hint,
            )
            if not translated_quality_ok:
                lang_failure = (
                    "translated question quality failed: "
                    f"{translated_quality_reason or 'question not useful enough'}"
                )
                retry_feedback = (
                    f"{lang_failure}. Keep the exact {source_lang_name} information need, but rewrite the {lang_name} "
                    "question into a stronger natural retrieval query without making it broader, more generic, or more literal."
                )
                continue

            if _normalize_domain_hint(domain_hint) == "legal":
                shape_ok, shape_reason = check_legal_question_shape(
                    client,
                    context,
                    q,
                    a,
                    model=quality_model,
                    output_language_name=lang_name,
                )
                if not shape_ok:
                    lang_failure = (
                        "translated legal shape failed: "
                        f"{shape_reason or 'question shape not useful enough'}"
                    )
                    retry_feedback = (
                        f"{lang_failure}. Rewrite the {lang_name} translation so the question asks for exactly one focused legal point "
                        "rather than a checklist, menu, definition inventory, or multi-branch formulation."
                    )
                    continue

            approved_translations[lang] = (q, a)
            lang_failure = ""
            break

        if lang_failure:
            failed_languages.append(f"{lang} ({lang_failure})")

    return approved_translations, failed_languages


def _process_sample_row(
    index: int,
    row: Dict[str, Any],
    *,
    target_languages: List[str],
    synthetic_translation_targets: List[str],
    generation_model: str,
    quality_model: str,
    support_model: str,
    translation_model: str,
    max_attempts: int,
    same_language: bool = False,
    domain_hint: Optional[str] = None,
) -> Dict[str, Any]:
    corpus_id = str(row.get("query_corpus_id") or row.get("id", "")).strip()
    context = (
        row.get("generation_context")
        or row.get("context", row.get("abstract", ""))
        or row.get("title", "")
    )
    effective_target_languages = _row_target_languages(row, target_languages)
    effective_synthetic_targets = _row_synthetic_target_languages(
        row,
        synthetic_translation_targets,
    )
    target_corpus_ids = _row_target_corpus_ids(row)
    if not context.strip():
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": [],
            "status": "skipped (empty context)",
        }

    try:
        client = _get_client()
        attempt_logs: List[str] = []

        if same_language:
            row_lang = (row.get("language") or "en").strip().lower()
            lang_name = LANG_NAMES.get(row_lang, row_lang)
            row_metadata = _row_output_metadata(row)
            approved_sl = False
            approved_attempt = 0
            q_loc = ""
            a_loc = ""
            supporting_text = ""
            question_type = ""
            last_failure = ""
            retry_feedback: Optional[str] = None
            retry_question: Optional[str] = None
            retry_answer: Optional[str] = None

            for _attempt in range(1, max_attempts + 1):
                generated = generate_qa_in_language(
                    client,
                    context,
                    row_lang,
                    previous_question=retry_question,
                    previous_answer=retry_answer,
                    previous_feedback=retry_feedback,
                    model=generation_model,
                )
                q_loc = generated["question"]
                a_loc = generated["answer"]
                supporting_text = generated["supporting_text"]
                question_type = generated["question_type"]
                retry_question = q_loc
                retry_answer = a_loc

                lang_ok, lang_reason = check_language_match(
                    client,
                    q_loc,
                    a_loc,
                    row_lang,
                    model=support_model,
                )
                if not lang_ok:
                    last_failure = f"language check failed: {lang_reason or 'wrong language'}"
                    attempt_logs.append(
                        f"attempt {_attempt}/{max_attempts}: language rejected - {lang_reason or 'wrong language'}"
                    )
                    retry_feedback = (
                        f"{last_failure}. Rewrite the question and answer entirely in {lang_name}."
                    )
                    continue

                faithful_ok, faithful_reason = check_faithfulness(
                    client,
                    context,
                    q_loc,
                    a_loc,
                    supporting_text,
                    model=support_model,
                    domain_hint=domain_hint,
                )
                if not faithful_ok:
                    last_failure = f"faithfulness check failed: {faithful_reason or 'not grounded enough'}"
                    attempt_logs.append(
                        f"attempt {_attempt}/{max_attempts}: faithfulness rejected - {faithful_reason or 'not grounded enough'}"
                    )
                    retry_feedback = (
                        f"{last_failure}. Keep the answer strictly grounded in the context."
                    )
                    continue

                quality_ok, quality_reason = check_question_quality(
                    client,
                    context,
                    q_loc,
                    a_loc,
                    model=quality_model,
                    output_language_name=lang_name,
                    domain_hint=domain_hint,
                )
                if not quality_ok:
                    last_failure = f"quality check failed: {quality_reason or 'question not useful enough'}"
                    attempt_logs.append(
                        f"attempt {_attempt}/{max_attempts}: quality rejected - {quality_reason or 'question not useful enough'}"
                    )
                    retry_feedback = (
                        f"{last_failure}. Use the better direction above if present. Regenerate one fresh question "
                        f"that is more retrieval-useful, more specific, less citation-led, and more semantic, still "
                        f"in {lang_name}. Prefer asking what the rule means, requires, permits, excludes, or causes "
                        f"rather than where it is written. Do not mention article numbers, labels, or phrases like "
                        f"'this article' unless essential. Prefer one narrower legal point over a multi-condition "
                        f"checklist. Avoid questions whose answer becomes an inventory of conditions, exceptions, "
                        f"documents, authorities, or consequences. Avoid deadline-only, amount-only, threshold-only, "
                        f"or list-only lookup questions when a better semantic legal question is available. Ask one "
                        f"legal question, not two joined together or one contrastive A-or-B question. Prefer the "
                        f"main obligation, trigger, or consequence over reporting-frequency or procedural-timing "
                        f"details, and drop secondary calculation or follow-up details when they weaken the query. "
                        f"If the previous attempt asked both purpose and applicability, both condition and duration, "
                        f"or both what something is and what it must do, keep only the stronger legal point. If the "
                        f"answer would mainly become a list of remedies, alternatives, or branches joined by 'or', "
                        f"ask for the main legal effect instead."
                    )
                    continue

                shape_ok, shape_reason = check_legal_question_shape(
                    client,
                    context,
                    q_loc,
                    a_loc,
                    model=quality_model,
                    output_language_name=lang_name,
                )
                if not shape_ok:
                    last_failure = f"legal shape check failed: {shape_reason or 'question shape not useful enough'}"
                    attempt_logs.append(
                        f"attempt {_attempt}/{max_attempts}: legal-shape rejected - {shape_reason or 'question shape not useful enough'}"
                    )
                    retry_feedback = (
                        f"{last_failure}. Regenerate one fresh legal retrieval question in {lang_name} that asks for exactly one "
                        f"focused legal point. Do not ask for a menu of measures, a bundle of conditions, a definition plus "
                        f"its listed members, or a date/value/code lookup when a sharper legal-effect question is available."
                    )
                    continue

                approved_sl = True
                approved_attempt = _attempt
                if _attempt > 1:
                    attempt_logs.append(
                        f"attempt {_attempt}/{max_attempts}: accepted after retries"
                    )
                break

            if not approved_sl:
                return {
                    "index": index,
                    "corpus_id": corpus_id,
                    "rows": [],
                    "attempt_logs": attempt_logs,
                    "status": f"skipped ({last_failure or 'validation failed'})",
                }

            qac_rows = [
                {
                    "corpus_id": corpus_id,
                    "language": row_lang,
                    "question": q_loc,
                    "answer": a_loc,
                    "is_synthetic_translation": "false",
                    **row_metadata,
                }
            ]
            approved_translations, failed_languages = _run_translation_pass(
                client,
                context=context,
                source_question=q_loc,
                source_answer=a_loc,
                source_lang=row_lang,
                translation_targets=effective_synthetic_targets,
                translation_model=translation_model,
                quality_model=quality_model,
                domain_hint=domain_hint,
                max_attempts=max_attempts,
            )
            for lang, (q, a) in approved_translations.items():
                qac_rows.append(
                    {
                        "corpus_id": corpus_id,
                        "language": lang,
                        "question": q,
                        "answer": a,
                        "is_synthetic_translation": "true",
                        **row_metadata,
                    }
                )
            translation_status = ""
            if effective_synthetic_targets:
                translation_status = f", + {len(approved_translations)} synthetic translations"
                if failed_languages:
                    translation_status += f", skipped {len(failed_languages)}: {', '.join(failed_languages)}"
            return {
                "index": index,
                "corpus_id": corpus_id,
                "attempt_logs": attempt_logs,
                "rows": qac_rows,
                "status": f"ok ({question_type or 'validated'} {row_lang}, same-language{translation_status}, attempt {approved_attempt}/{max_attempts})",
            }

        approved = False
        approved_attempt = 0
        q_en = ""
        a_en = ""
        supporting_text = ""
        question_type = ""
        last_failure = ""
        retry_feedback: Optional[str] = None
        retry_question: Optional[str] = None
        retry_answer: Optional[str] = None

        for _attempt in range(1, max_attempts + 1):
            generated = generate_qa_english(
                client,
                context,
                previous_question=retry_question,
                previous_answer=retry_answer,
                previous_feedback=retry_feedback,
                model=generation_model,
                domain_hint=domain_hint,
            )
            q_en = generated["question"]
            a_en = generated["answer"]
            supporting_text = generated["supporting_text"]
            question_type = generated["question_type"]
            retry_question = q_en
            retry_answer = a_en

            lang_ok, lang_reason = check_english_language(
                client,
                q_en,
                a_en,
                model=support_model,
            )
            if not lang_ok:
                last_failure = f"language check failed: {lang_reason or 'not English enough'}"
                attempt_logs.append(
                    f"attempt {_attempt}/{max_attempts}: language rejected - {lang_reason or 'not English enough'}"
                )
                retry_feedback = (
                    f"{last_failure}. The output must be natural English only."
                )
                continue

            faithful_ok, faithful_reason = check_faithfulness(
                client,
                context,
                q_en,
                a_en,
                supporting_text,
                model=support_model,
                domain_hint=domain_hint,
            )
            if not faithful_ok:
                last_failure = f"faithfulness check failed: {faithful_reason or 'not grounded enough'}"
                attempt_logs.append(
                    f"attempt {_attempt}/{max_attempts}: faithfulness rejected - {faithful_reason or 'not grounded enough'}"
                )
                retry_feedback = (
                    f"{last_failure}. Remove unsupported details and keep the answer strictly grounded in the context."
                )
                continue

            quality_ok, quality_reason = check_question_quality(
                client,
                context,
                q_en,
                a_en,
                model=quality_model,
                domain_hint=domain_hint,
            )
            if not quality_ok:
                last_failure = f"quality check failed: {quality_reason or 'question not useful enough'}"
                attempt_logs.append(
                    f"attempt {_attempt}/{max_attempts}: quality rejected - {quality_reason or 'question not useful enough'}"
                )
                retry_feedback = (
                    f"{last_failure}. Use the better direction above if present. Regenerate one fresh question "
                    f"that is more retrieval-useful, more specific, less generic, and less surface-aligned. "
                    f"Prefer one narrower technical fact over a broad summary or literal lookup."
                )
                continue

            if _normalize_domain_hint(domain_hint) == "legal":
                shape_ok, shape_reason = check_legal_question_shape(
                    client,
                    context,
                    q_en,
                    a_en,
                    model=quality_model,
                )
                if not shape_ok:
                    last_failure = f"legal shape check failed: {shape_reason or 'question shape not useful enough'}"
                    attempt_logs.append(
                        f"attempt {_attempt}/{max_attempts}: legal-shape rejected - {shape_reason or 'question shape not useful enough'}"
                    )
                    retry_feedback = (
                        f"{last_failure}. Regenerate one fresh legal retrieval question that asks for exactly one focused legal point. "
                        f"Do not ask for a menu of measures, a bundle of conditions, a definition plus its listed members, or a "
                        f"date/value/code lookup when a sharper legal-effect question is available."
                    )
                    continue

            approved = True
            approved_attempt = _attempt
            if _attempt > 1:
                attempt_logs.append(
                    f"attempt {_attempt}/{max_attempts}: accepted after retries"
                )
            break

        if not approved:
            return {
                "index": index,
                "corpus_id": corpus_id,
                "rows": [],
                "attempt_logs": attempt_logs,
                "status": f"skipped ({last_failure or 'validation failed'})",
            }

        english_corpus_id = target_corpus_ids.get("en", corpus_id)
        qac_rows = [{
            "corpus_id": english_corpus_id,
            "language": "en",
            "question": q_en,
            "answer": a_en,
            "is_synthetic_translation": "false",
        }]
        translation_targets = [lang for lang in effective_target_languages if lang != "en"]
        approved_translations, failed_languages = _run_translation_pass(
            client,
            context=context,
            source_question=q_en,
            source_answer=a_en,
            source_lang="en",
            translation_targets=translation_targets,
            translation_model=translation_model,
            quality_model=quality_model,
            domain_hint=domain_hint,
            max_attempts=max_attempts,
        )

        for lang, (q, a) in approved_translations.items():
            qac_rows.append({
                "corpus_id": target_corpus_ids.get(lang, corpus_id),
                "language": lang,
                "question": q,
                "answer": a,
                "is_synthetic_translation": "true",
            })
        translation_status = f"{len(approved_translations)} translations"
        if failed_languages:
            translation_status += f", skipped {len(failed_languages)}: {', '.join(failed_languages)}"
        return {
            "index": index,
            "corpus_id": corpus_id,
            "attempt_logs": attempt_logs,
            "rows": qac_rows,
            "status": f"ok ({question_type or 'validated'} en + {translation_status}, attempt {approved_attempt}/{max_attempts})",
        }
    except Exception as exc:
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": [],
            "attempt_logs": [],
            "status": f"error: {exc}",
        }


def run_qa_pipeline(
    corpus_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 50,
    target_languages: Optional[List[str]] = None,
    synthetic_translation_targets: Optional[List[str]] = None,
    model: Optional[str] = None,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    quality_model: str = DEFAULT_QUALITY_MODEL,
    support_model: str = DEFAULT_SUPPORT_MODEL,
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
    max_attempts: int = 3,
    batch_mode: bool = False,
    same_language: bool = False,
    domain_hint: Optional[str] = None,
) -> int:
    """
    Sample corpus and generate Q&A.

    If same_language is False (default): generate in English, translate to target_languages.
    If same_language is True: generate legal question and answer in each row's `language`
    field, with optional synthetic translated query variants.

    Writes qac.csv (corpus_id, language, question, answer) to output_dir.
    Returns number of QAC rows written.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGS
    if synthetic_translation_targets is None:
        synthetic_translation_targets = []
    if model is not None:
        generation_model = model
        quality_model = model
        support_model = model
        translation_model = model
    if same_language and _normalize_domain_hint(domain_hint) != "legal":
        raise ValueError(
            "same_language mode is only supported for legal/JRC generation in this branch."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus(corpus_path)
    sampled = sample_corpus(rows, sample_size, stratify_by_language=True, seed=42)
    mode = (
        "same-language (per row language)"
        if same_language and not synthetic_translation_targets
        else f"same-language (per row language) + synthetic translations {synthetic_translation_targets}"
        if same_language
        else "English + translation"
    )
    print(f"Sampled {len(sampled)} documents from corpus ({len(rows)} total). Mode: {mode}.")
    qac_rows: List[Dict[str, str]] = []
    results: List[Dict[str, Any]] = []

    if batch_mode and sampled:
        available_cpus = os.cpu_count() or 1
        workers = max(1, min(len(sampled), max(1, available_cpus // 2), 5))
        print(
            f"Running batched Q&A generation with {workers} worker(s) "
            f"based on {available_cpus} available CPU(s)."
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_sample_row,
                    index,
                    row,
                    target_languages=target_languages,
                    synthetic_translation_targets=synthetic_translation_targets,
                    generation_model=generation_model,
                    quality_model=quality_model,
                    support_model=support_model,
                    translation_model=translation_model,
                    max_attempts=max_attempts,
                    same_language=same_language,
                    domain_hint=domain_hint,
                )
                for index, row in enumerate(sampled)
            ]
            progress = tqdm(as_completed(futures), total=len(futures), desc="Generate Q&A", unit="doc")
            for completed, future in enumerate(progress, start=1):
                result = future.result()
                results.append(result)
                for log_line in result.get("attempt_logs", []):
                    tqdm.write(f"     {result['corpus_id']}: {log_line}")
                tqdm.write(
                    f"  [{completed}/{len(sampled)}] {result['corpus_id']}... {result['status']}"
                )
    else:
        if sampled:
            print("Running Q&A generation in single-threaded mode.")
        progress = tqdm(sampled, total=len(sampled), desc="Generate Q&A", unit="doc")
        for index, row in enumerate(progress, start=1):
            result = _process_sample_row(
                index - 1,
                row,
                target_languages=target_languages,
                synthetic_translation_targets=synthetic_translation_targets,
                generation_model=generation_model,
                quality_model=quality_model,
                support_model=support_model,
                translation_model=translation_model,
                max_attempts=max_attempts,
                same_language=same_language,
                domain_hint=domain_hint,
            )
            results.append(result)
            for log_line in result.get("attempt_logs", []):
                tqdm.write(f"     {result['corpus_id']}: {log_line}")
            tqdm.write(f"  [{index}/{len(sampled)}] {result['corpus_id']}... {result['status']}")

    for result in sorted(results, key=lambda item: item["index"]):
        qac_rows.extend(result["rows"])

    out_csv = output_dir / "qac.csv"
    fieldnames = ["corpus_id", "language", "question", "answer"]
    for row in qac_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(qac_rows)

    print(f"Wrote {len(qac_rows)} QAC rows -> {out_csv}")
    return len(qac_rows)
