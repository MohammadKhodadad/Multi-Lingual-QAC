"""
OpenAI-based Q&A generation.

Default pipeline (`same_language=False`, including Wikidata via `pipeline.py`):
generate question and answer in English from the corpus context, validate, then
translate to all target languages.

Optional `same_language=True`: generate and validate Q&A in each row's
`language` field (no translation step); for experiments or direct API use.
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
    domain = _normalize_domain_hint(domain_hint)
    prompt = """You are an expert at creating retrieval questions from chemistry, patent, legal, regulatory, encyclopedia, or technical passages.

The source context may be in any language, but your output must be in English only.

Generate exactly ONE question-answer pair from the context.

Rules:
- Output must be in natural English only.
- Do not copy the source language unless a chemical name, formula, identifier, or proper noun should remain unchanged.
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
  - exact wt% or mol-ratio lookups
  - exact temperature, density, time, or concentration range lookups
  - exact component inventory or long named-list lookups
  - exact "what does the composition contain" questions
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
  - "How does the treatment improve hair growth when heat is applied afterward?"
  - "Where would these microcapsules be used in fragranced consumer goods?"
  - "What does the shape deformation layer do when the artificial nail is pressed onto the natural nail?"
  - "Why is cold rolling performed after hot rolling or forging in this steel production process?"
  - "What function does sodium bicarbonate serve in the enteric coating composition?"
  - "Which biomarker pairs are measured to assess early-onset preeclampsia risk?"
  - "At what density is the mixed solution evaporated before filtration?"
  - "What property of the glass substrate supports fine pattern formation?"
  - "What condition must be satisfied before the import can be authorized?"
  - "What happens if the authority does not object within the stated period?"
  - "Who may submit a request to stop the infringement?"
  - "What evidence has to accompany the shipment at first marketing?"
- Bad style examples:
  - "What are the recommended application methods for the preparation described in the invention?"
  - "What type of products can include the microcapsules mentioned in the invention?"
  - "What is the application of 8-(4-trifluoromethoxy)benzyloamino-2'-deoxyadenosine?"
  - "What types of products can utilize the hair dye composition described in the text?"
  - "What is the role of the benzoxazole derivatives in detecting GHB in beverages?"
  - "What is the main technical advantage of this method?"
  - "What is the purpose of the process?"
  - "How does the method work?" when the context supports a more specific `Why`, `Which`, `What function`, `What condition`, or `At what` question
  - "What SiO2/Li2O and SiO2/Al2O3 mol ratios does the glass composition require?" when the context also supports a better question about why the composition enables the target property
  - "What are the specified weight percent ranges for silicon and manganese?" when the context also supports a better question about the role or effect of the composition
  - "Which specific fungicides are named as component (2)?" when the context also supports a better question about selection logic, interaction, or functional grouping
  - "What is the purpose of flowing a portion of metal-rich produced water to an evaporation area ...?" when the better question is about what this step causes or why it enables metal recovery
  - "What advantages does the DNA oligonucleotide ... offer ...?" when the better question is about the concrete storage or biosafety property
  - "According to Article 6, what conditions apply?"
  - "Under Article 4(1), how must the court treat the list?"
  - "What does this article require for authorization?"
  - "Which article sets out the exception?"
- Output valid JSON only, no markdown:
  {"question": "...", "answer": "...", "supporting_text": "...", "question_type": "..."}
"""
    if domain == "legal":
        prompt = """You are an expert at creating retrieval questions from multilingual legal and regulatory passages, especially EU legislation, decisions, directives, regulations, opinions, and procedural acts.

The source context may be in any language, but your output must be in English only.

Generate exactly ONE question-answer pair from the context.

Rules:
- Output must be natural English only.
- The question must read like a realistic retrieval query that a lawyer, policy reader, regulator, compliance analyst, legal researcher, or informed practitioner might type.
- Prefer one substantive legal information need, not a clause lookup.
- Ask about one operative point such as a condition, consequence, addressee, exception, scope limit, evidence requirement, procedural effect, legal threshold, or what the rule practically requires, permits, excludes, or causes.
- The question must be answerable from the text and specific enough to be useful for retrieval.
- Prefer semantically challenging questions that require understanding the operative meaning of the provision rather than surface keyword lookup.
- The question should still work as a strong query if article numbers, paragraph numbers, recital numbers, and annex labels were hidden.
- Do not mention article numbers, paragraph numbers, recital numbers, annex labels, or phrases like "this article", "this regulation", "this directive", "this decision", or "under Article ..." unless the reference itself is essential.
- Do not ask for the provision number, legal basis citation, exact title, or exact clause wording.
- Do not simply restate the text in question form.
- Do not ask a broad summary question when a narrower operative question is available.
- Do not bundle multiple loosely related legal conditions into one long checklist question unless the source presents them as one inseparable rule.
- Avoid questions that only ask for a raw date, percentage, ratio, or listed items when the text supports a better question about threshold, obligation, exception, trigger, or consequence.
- Prefer grounded paraphrase over direct lexical overlap.
- Before finalizing, check:
  - Does this ask about what the rule means, requires, permits, excludes, proves, or causes?
  - Would this still be a good query without article labels?
  - Would dense retrieval need semantic understanding, not just BM25-style token overlap?
- The answer must be concise (1-2 sentences) and strictly grounded in the context.
- Include a short supporting_text quote copied from the source context that justifies the answer.
- Include a question_type chosen from: obligation, requirement, scope, exception, consequence, evidence, definition, procedure, other.
- Good style examples:
  - "What condition must be satisfied before the import can be authorized?"
  - "What happens if the authority does not object within the stated period?"
  - "Who may submit a request to stop the infringement?"
  - "What evidence has to accompany the shipment at first marketing?"
- Bad style examples:
  - "According to Article 6, what conditions apply?"
  - "Under Article 4(1), how must the court treat the list?"
  - "What does this article require for authorization?"
  - "Which article sets out the exception?"

Output valid JSON only, no markdown:
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
    domain_hint: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate one Q&A pair in the same language as the corpus row (target_lang_code).
    """
    lang_name = LANG_NAMES.get(target_lang_code.lower(), target_lang_code)
    domain = _normalize_domain_hint(domain_hint)
    prompt = f"""You are an expert at creating retrieval questions from legal, regulatory, encyclopedia, or technical passages.

The source context is written mainly in {lang_name} (BCP-47 code: {target_lang_code}).
You MUST write the question, answer, and supporting_text entirely in {lang_name}.

Rules:
- Natural, fluent {lang_name} only. Keep chemical names, formulas, identifiers, units, and proper nouns as appropriate (they may stay in Latin script or standard notation).
- The question must read like a realistic search query a researcher, lawyer, policy reader, compliance reader, or informed user would type.
- Prefer a specific question about one concrete point the passage supports best, such as: rule, requirement, scope, definition, obligation, exception, applicability, procedure, method, property, use, safety, history, comparison, legal effect, consequence, or evidence threshold.
- The question must be answerable strictly from the passage and be specific enough for retrieval benchmarking.
- Prefer semantically challenging questions over trivial keyword overlap with the first sentence.
- Do not copy a long span from the source verbatim into the question.
- Do not turn the title alone into the question.
- For legal or regulatory passages, prefer questions that require understanding what a provision means, when it applies, what it allows or forbids, who it binds, what condition triggers a consequence, what evidence or requirement must be satisfied, or what practical effect follows.
- For legal or regulatory passages, prefer one operative clause, condition, exception, addressee, definition, effect, threshold, or consequence rather than broad document-summary wording.
- When possible, ask about:
  - what condition must be met
  - what happens if a condition is or is not met
  - who is allowed, required, exempted, or affected
  - what a rule applies to or excludes
  - what evidence, certificate, or showing is required
  - what legal or procedural effect follows
- Prefer the substance of the rule over the citation label. Do not make the question mainly about locating "Article X" or "paragraph Y" if the same point can be asked as a natural information need.
- Do not mention article numbers, paragraph numbers, recital numbers, annex labels, or phrases like "this article", "this regulation", "this directive", or close equivalents in {lang_name} unless the reference itself is essential.
- Avoid citation-led openings such as:
  - "According to Article ..."
  - "Under Article ..."
  - "Per Article ..."
  - close equivalents in {lang_name}
  unless the article number itself is essential to the information need.
- Avoid clause-lookup questions that mainly ask for an exact provision number, exact title, exact legal basis citation, or exact article-reference wording when the same passage supports a better question about meaning, obligation, condition, exception, or effect.
- The question should still make sense as a strong query if the legal labels were hidden from the reader.
- Avoid questions that only ask for a raw date, raw percentage, raw ratio, or raw named list when the passage supports a better question about what that detail controls, permits, changes, or requires.
- Avoid bundling multiple loosely related legal conditions into one long checklist question unless the passage presents them as one inseparable rule.
- Prefer shorter, natural query wording over long recital-like phrasing copied from the text.
- A good legal retrieval question should still make sense if article numbers are hidden.
- The answer must be concise (1-2 sentences) and fully grounded in the context.
- Include supporting_text: a short quote copied from the source that justifies the answer.
- Include question_type: one of purpose, application, composition, method, property, safety, history, obligation, scope, requirement, other.

Output valid JSON only, no markdown:
{{"question": "...", "answer": "...", "supporting_text": "...", "question_type": "..."}}
"""
    if domain == "legal":
        prompt = f"""You are an expert at creating retrieval questions from multilingual legal and regulatory passages, especially EU legislation, decisions, directives, regulations, opinions, and procedural acts.

The source context is written mainly in {lang_name} (BCP-47 code: {target_lang_code}).
You MUST write the question, answer, and supporting_text entirely in {lang_name}.

Rules:
- Natural, fluent {lang_name} only.
- The question must read like a realistic search query a lawyer, policy reader, regulator, compliance analyst, legal researcher, or informed practitioner would type.
- Prefer one substantive legal information need, not a clause lookup.
- Ask about one operative point such as a condition, consequence, addressee, exception, scope limit, evidence requirement, procedural effect, legal threshold, or what the rule practically requires, permits, excludes, or causes.
- The question must be answerable strictly from the passage and be specific enough for retrieval benchmarking.
- Prefer semantically challenging questions that require understanding the operative meaning of the provision rather than surface keyword lookup.
- The question should still work as a strong query if article numbers, paragraph numbers, recital numbers, and annex labels were hidden.
- Do not mention article numbers, paragraph numbers, recital numbers, annex labels, or phrases like "this article", "this regulation", "this directive", or close equivalents in {lang_name} unless the reference itself is essential.
- Do not ask for the provision number, legal basis citation, exact title, or exact clause wording.
- Do not simply restate the passage in question form.
- Do not ask a broad summary question when a narrower operative question is available.
- Do not bundle multiple loosely related legal conditions into one long checklist question unless the passage presents them as one inseparable rule.
- Avoid questions that only ask for a raw date, percentage, ratio, or listed items when the passage supports a better question about threshold, obligation, exception, trigger, or consequence.
- Prefer grounded paraphrase over direct lexical overlap.
- The answer must be concise (1-2 sentences) and fully grounded in the context.
- Include supporting_text: a short quote copied from the source that justifies the answer.
- Include question_type: one of obligation, requirement, scope, exception, consequence, evidence, definition, procedure, other.

Output valid JSON only, no markdown:
{{"question": "...", "answer": "...", "supporting_text": "...", "question_type": "..."}}
"""
    retry_note = ""
    if previous_feedback:
        retry_note = (
            "\n\nPrevious attempt issue to fix:\n"
            f"{previous_feedback}\n"
            f"Regenerate so the issue is fixed; keep everything in {lang_name}. For legal or regulatory text, ask about the operative rule without explicitly pointing to article numbers or labels."
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
- they are not mixed-language outputs except for unavoidable chemical names, formulas, identifiers, or proper nouns.

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
- chemical names, formulas, units, identifiers, and proper nouns may appear in Latin script or standard notation when normal for that field.

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

Approve only if:
- the question is answerable from the context,
- the answer is fully supported by the context,
- the answer does not add unsupported details,
- the supporting_text is relevant evidence from the context.

Reject if the answer is generic, speculative, partially unsupported, or not clearly grounded.

Output valid JSON only:
{"approved": true, "reason": "..."}
"""
    if domain == "legal":
        prompt = """You are a strict faithfulness checker for question-answer pairs derived from a legal or regulatory source passage.

Approve only if:
- the question is answerable from the context,
- the answer is fully supported by the context,
- the answer does not add unsupported legal implications or extra conditions,
- the supporting_text is relevant evidence from the context.

Reject if the answer overstates what the rule means, adds missing exceptions or consequences, is generic, speculative, partially unsupported, or not clearly grounded.

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

Approve only if the question:
- sounds like a realistic search or retrieval query,
- is specific enough to distinguish the document,
- asks about a concrete point from the context,
- uses natural user-like wording rather than document-summary or clause-lookup wording,
- is phrased semantically rather than as an obvious exact-match template,
- would be easier for a strong semantic retriever than for naive keyword matching,
- is not too generic,
- is not nearly copied from the context verbatim,
- and is useful for retrieval benchmarking.

Reject questions that are broad or repetitive patterns such as:
- "What is the main object of the invention?"
- "What is the main feature?"
- "What are the main components?"
- "What are the applications ...?" when a more specific application question is possible
- "What is the composition ...?" when a more targeted material or component question is possible
- "What is the main technical advantage ...?" when a narrower effect, property, operating condition, or mechanism question is possible
- "What is the advantage ...?" when the answer would bundle multiple benefits instead of one fact
- "What is the purpose ...?" when the question does not name a specific step, component, material, or operation
- "What advantages does ... offer ...?" when the answer mainly bundles several benefits that could be asked about more concretely
unless the context is too short for a better question.

For legal or regulatory passages, also reject questions that:
- are led mainly by a citation such as "According to Article X" or "Under Article Y" when the same issue can be asked without the citation
- mainly ask the user to look up a clause number, legal basis citation, or provision label
- explicitly mention article numbers, paragraph numbers, recital numbers, annex labels, "this article", "this regulation", "this directive", or similar label-based wording when the same question can be asked without them
- ask only for an exact date, percentage, ratio, or listed items when the passage supports a better question about consequence, applicability, threshold, obligation, exception, or legal effect
- ask for multiple conditions or compliance criteria at once when one narrower condition or consequence would make a better retrieval query
- sound like a recital or provision restatement rather than a natural legal information need
- could not survive removal of article labels because the query depends on those labels more than on the operative content

Also reject questions that:
- mostly reuse a title phrase or a distinctive noun phrase from the source with only light reformatting,
- depend mainly on exact keyword overlap rather than semantic understanding,
- ask directly for the name, application, or advantage of a named entity when a more functional or effect-based question is possible.
- ask vaguely about "the method" or "the process" without identifying what part of it is being asked about, even though the context contains a more specific step or condition.
- are primarily spec-sheet or table-lookup questions when the same context supports a better semantic question about rationale, role, effect, mechanism, implication, or process purpose.
- ask only for raw values, ranges, ratios, or enumerated lists even though the document provides enough context to ask what those details enable, affect, control, or explain.
- in legal text, ask only for the literal cited clause instead of what the clause means, requires, permits, excludes, or causes.
- in legal text, point directly to article/provision labels instead of asking about the operative condition, exception, threshold, or consequence.

Treat these as common extractive failure modes:
- exact wt% / mol-ratio lookup
- exact temperature / density / time / concentration range lookup
- exact ingredient inventory or long named-list lookup
- direct "what does the composition contain" lookup
- direct "what values are specified for X and Y" lookup

Do NOT reject all numeric questions automatically.
Approve them only when the number or range itself is genuinely the most retrieval-worthy fact in the context and no clearly better semantic question is available.

Also reject document-centered wording such as:
- "described in the invention"
- "mentioned in the invention"
- "according to the invention"
- "in the text"

Also reject broad template openings such as:
- "What is the application of ..."
- "What are the advantages of ..."
- "What advantages does ..."
- "What is the benefit of ..."
- "What is the main technical advantage of ..."
- "What is the purpose of ..."
- "What types of products ..."
- "What is the role of ..."
when they lead to a broad summary question instead of a sharper technical query.

Be especially strict about these two failure modes:
1. title-lift: the question is basically the title or first source phrase converted into a question
2. high-overlap paraphrase: the question keeps too much surface wording from the source and would still be easy for exact-match retrieval
3. overly-extractive: the question is safe and specific but mainly asks for a literal value/list/span rather than a semantic technical point that the same context supports
4. broad-summary: the question asks for a broad purpose/advantage/benefit summary instead of one narrower technical fact
5. bundled-facts: the question asks for multiple loosely related facts at once instead of one core information need
6. weak-query-shape: the question is understandable but not phrased like a strong retrieval query a user would naturally type
7. legalistic-lookup: in legal or regulatory text, the question is driven mainly by article/provision lookup rather than substantive understanding of the rule

Approve borderline cases only if the question is clearly more natural, more specific, and less surface-aligned than those failure modes.

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
- provide `better_direction` as ONE short actionable hint for regeneration, for example:
  - `ask about why the step is used`
  - `ask about the component's role`
  - `ask about the effect, not the raw value`
  - `ask what the condition enables`
  - `focus on one narrower technical fact`
  - `ask what the rule requires or causes, not where it is written`
  - `remove the article reference and ask about the operative condition or effect`

If you approve the question:
- set `failure_type` to `none`
- set `better_direction` to an empty string

Output valid JSON only:
{"approved": true, "reason": "...", "failure_type": "none", "better_direction": ""}
"""
    if domain == "legal":
        prompt = lang_clause + """You are a strict quality checker for retrieval questions built from legal and regulatory text.

Approve only if the question:
- sounds like a realistic legal or policy retrieval query,
- is specific enough to distinguish the document,
- asks about one operative point from the context,
- uses natural user-like wording rather than clause-lookup wording,
- is phrased semantically rather than as an exact-match template,
- would require meaningful semantic understanding rather than simple BM25-style overlap,
- is not too generic,
- is not nearly copied from the context verbatim,
- and is useful for legal retrieval benchmarking.

Reject questions that:
- are led by article, paragraph, recital, annex, or provision labels,
- explicitly mention article numbers or phrases like "this article", "this regulation", or "under Article ..." when the same issue can be asked without them,
- mainly ask the user to locate a clause instead of understand what the rule requires, permits, excludes, proves, or causes,
- ask only for a raw date, percentage, ratio, or listed items when the text supports a better question about threshold, obligation, exception, trigger, consequence, or scope,
- bundle multiple loosely related legal conditions into one checklist question,
- sound like a recital or provision restatement rather than a natural legal information need.

Be especially strict about these failure modes:
1. title-lift
2. high-overlap
3. overly-extractive
4. broad-summary
5. bundled-facts
6. weak-query-shape
7. legalistic-lookup

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
- examples:
  - `ask what the rule requires or causes, not where it is written`
  - `remove the article reference and ask about the operative condition or effect`
  - `focus on one narrower legal consequence`
  - `ask about the threshold or exception, not the label`

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
    domain = _normalize_domain_hint(domain_hint)
    lang_list = ", ".join(LANG_NAMES.get(l, l) for l in target_langs)
    prompt = f"""Translate the following English retrieval question and answer pair into these languages: {lang_list}.

For each language, produce a natural, native-sounding, retrieval-style translation.
Use the source context to resolve ambiguity and preserve the original information need exactly.
Keep the same meaning, level of specificity, and technical terms where appropriate.
Do not make the question more generic than the original.
Preserve the semantic difficulty of the original question.
Do not simplify the question into a keyword-heavy or literal surface-form restatement.
Prefer natural target-language phrasing over word-for-word translation.
Do not omit or alter numbers, units, ranges, formulas, identifiers, or named technical materials.
Preserve chemical names, abbreviations, symbols, and patent-style identifiers when translating them would be incorrect or unnatural.
Keep the answer faithful to the English answer and consistent with the source context.
Do not add explanation, background, or extra claims not present in the English pair or source context.
If the English question is technical and concise, keep the target-language question technical and concise too.
Avoid translation artifacts:
- choose one natural term, not slash-separated alternatives like `X/Y`
- do not leave editor-style repair traces or synonym bundles
- do not include unnecessary English glosses in parentheses
- avoid code-mixed verbs or phrasing when the target language has a normal technical equivalent
- rewrite into natural target-language syntax instead of following English word order too closely
- keep the text fully in the target language except for unavoidable chemical names, formulas, units, identifiers, abbreviations, or proper nouns
- do not leak words from unrelated languages or scripts into the translation
- if a technical term can stay in Latin script, integrate it naturally into an otherwise target-language sentence
- if the English answer contains multiple supported facts, preserve them cleanly without turning the translation into a glossary or note
- prefer one polished final phrasing, not an exploratory or half-edited wording

Output valid JSON only:
{{"translations": {{"de": {{"question": "...", "answer": "..."}}, "fr": {{...}}, ...}}}}

Languages to include: {json.dumps(target_langs)}
"""
    if domain == "legal":
        prompt = f"""Translate the following English legal/regulatory retrieval question and answer pair into these languages: {lang_list}.

For each language, produce a natural, native-sounding legal retrieval-style translation.
Use the source context to resolve ambiguity and preserve the original information need exactly.
Keep the same meaning, level of specificity, legal effect, and operative focus as the English question.
Do not make the question more generic than the original.
Do not simplify the question into a keyword-heavy or literal surface-form restatement.
Do not introduce article numbers, paragraph numbers, recital numbers, annex labels, or phrases like "this article" or "under Article ..." if they are not present in the English question.
Preserve dates, numbers, thresholds, legal conditions, exceptions, scope limits, entities, and required evidence exactly when they matter.
Keep the answer faithful to the English answer and consistent with the source context.
Do not add explanation, background, or extra legal implications not present in the English pair or source context.
Prefer natural target-language legal phrasing over word-for-word translation.
Avoid translation artifacts:
- choose one natural term, not slash-separated alternatives like `X/Y`
- do not leave editor-style repair traces or synonym bundles
- do not include unnecessary English glosses in parentheses
- rewrite into natural target-language syntax instead of following English word order too closely
- keep the text fully in the target language except for unavoidable identifiers, abbreviations, citations, or proper nouns
- prefer one polished final phrasing, not an exploratory or half-edited wording

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
                    f"English question: {question}\n\n"
                    f"English answer: {answer}"
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
    english_question: str,
    english_answer: str,
    translated_question: str,
    translated_answer: str,
    target_lang: str,
    *,
    model: str = DEFAULT_QUALITY_MODEL,
    domain_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate that a translated QA pair is fluent, faithful, and in the target language.
    Returns structured quality signals for approval and retry decisions.
    """
    target_lang_name = LANG_NAMES.get(target_lang, target_lang)
    domain = _normalize_domain_hint(domain_hint)
    prompt = f"""You are a strict but practical translation quality checker for multilingual patent retrieval data.

The source context may be in any language. The reference question and answer are in English.
The candidate translation must be in {target_lang_name}.

Judge these dimensions separately:
- `language_ok`: the translated question and answer are clearly written in {target_lang_name}
- `meaning_ok`: the meaning matches the English question and English answer closely
- `technical_ok`: numbers, units, ranges, formulas, identifiers, and important technical terms are preserved
- `specificity_ok`: the translated question keeps the same information need and specificity and does not become more generic
- `terminology_ok`: the translation uses appropriate technical terminology and register for {target_lang_name}
- `artifact_ok`: the translation does not contain repair artifacts such as slash-separated alternatives, unnecessary English glosses, editor-style synonym bundles, or gratuitous code mixing
- `fluency_ok`: the translation sounds natural enough for a native technical reader and is not clearly word-for-word or grammatically broken
- `grammar_ok`: grammar, agreement, case, morphology, and local sentence form are acceptable for {target_lang_name}

Be especially strict about these artifact failures:
- slash alternatives like `X/Y` when one natural wording should be chosen
- parenthetical English glosses like `(oiling)` when they are not required for correctness
- mixed-language repair wording or unresolved synonym pairs
- foreign-script leakage from an unrelated language when the span is not just a formula, identifier, unit, abbreviation, or proper noun
- faithful but clearly literal syntax that still reads like English structure mapped into {target_lang_name}

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
  - `keep the question as specific as the English original`
  - `choose one natural term instead of slash alternatives`
  - `remove the English gloss and use native technical wording`
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

Output valid JSON only:
{{"language_ok": true, "meaning_ok": true, "technical_ok": true, "specificity_ok": true, "terminology_ok": true, "artifact_ok": true, "fluency_ok": true, "grammar_ok": true, "severity": "low", "failure_type": "none", "better_direction": "", "reason": "..."}}
"""
    if domain == "legal":
        prompt = f"""You are a strict but practical translation quality checker for multilingual legal/regulatory retrieval data.

The source context may be in any language. The reference question and answer are in English.
The candidate translation must be in {target_lang_name}.

Judge these dimensions separately:
- `language_ok`: the translated question and answer are clearly written in {target_lang_name}
- `meaning_ok`: the meaning matches the English question and English answer closely
- `technical_ok`: dates, numbers, thresholds, scope conditions, exceptions, entities, and required evidence are preserved
- `specificity_ok`: the translated question keeps the same information need and specificity and does not become more generic
- `terminology_ok`: the translation uses appropriate legal/regulatory terminology and register for {target_lang_name}
- `artifact_ok`: the translation does not contain repair artifacts such as slash-separated alternatives, unnecessary English glosses, editor-style synonym bundles, or gratuitous code mixing
- `fluency_ok`: the translation sounds natural enough for a native legal/policy reader and is not clearly word-for-word or grammatically broken
- `grammar_ok`: grammar, agreement, case, morphology, and local sentence form are acceptable for {target_lang_name}
- `label_reference_ok`: the translation does not introduce article/provision labels or phrases like "this article" or "under Article ..." if they are absent from the English question

Severity guidelines:
- `high`: wrong language, meaning drift, introduced label references, dropped or altered legal details, or much more generic wording
- `medium`: meaning is mostly correct but there are clear grammar problems or awkward/literal phrasing
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
- `introduced-label-reference`

If you reject:
- keep `reason` short and concrete
- provide `better_direction` as ONE short actionable repair hint
- examples:
  - `rewrite more naturally for legal phrasing`
  - `restore the missing legal detail exactly`
  - `keep the question as specific as the English original`
  - `remove the added article reference`
  - `use natural legal register in {target_lang_name}`

Approval policy:
- Reject when `language_ok`, `meaning_ok`, `technical_ok`, `specificity_ok`, or `label_reference_ok` is false.
- Reject when `terminology_ok` is false.
- Reject when `artifact_ok` is false.
- Reject when `grammar_ok` is false and severity is `medium` or `high`.
- Reject when `fluency_ok` is false and severity is `medium` or `high`.
- Approve when the only issue is minor fluency stiffness with `severity = low`.

Output valid JSON only:
{{"language_ok": true, "meaning_ok": true, "technical_ok": true, "specificity_ok": true, "terminology_ok": true, "artifact_ok": true, "fluency_ok": true, "grammar_ok": true, "label_reference_ok": true, "severity": "low", "failure_type": "none", "better_direction": "", "reason": "..."}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"English question: {english_question}\n\n"
                    f"English answer: {english_answer}\n\n"
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
    label_reference_ok = bool(data.get("label_reference_ok", True))
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
        "introduced-label-reference",
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
        and label_reference_ok
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
        or (not label_reference_ok)
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
        "label_reference_ok": label_reference_ok,
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


def _process_sample_row(
    index: int,
    row: Dict[str, Any],
    *,
    target_languages: List[str],
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

        if same_language:
            row_lang = (row.get("language") or "en").strip().lower()
            lang_name = LANG_NAMES.get(row_lang, row_lang)
            approved_sl = False
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
                    domain_hint=domain_hint,
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
                    retry_feedback = (
                        f"{last_failure}. Use the better direction above if present. Regenerate one fresh question "
                        f"that is more retrieval-useful, more specific, less citation-led, and more semantic, still "
                        f"in {lang_name}. Prefer asking what the rule means, requires, permits, excludes, or causes "
                        f"rather than where it is written. Do not mention article numbers, labels, or phrases like "
                        f"'this article' unless essential."
                    )
                    continue

                approved_sl = True
                break

            if not approved_sl:
                return {
                    "index": index,
                    "corpus_id": corpus_id,
                    "rows": [],
                    "status": f"skipped ({last_failure or 'validation failed'})",
                }

            return {
                "index": index,
                "corpus_id": corpus_id,
                "rows": [
                    {
                        "corpus_id": corpus_id,
                        "language": row_lang,
                        "question": q_loc,
                        "answer": a_loc,
                    }
                ],
                "status": f"ok ({question_type or 'validated'} {row_lang}, same-language)",
            }

        approved = False
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
                if _normalize_domain_hint(domain_hint) == "legal":
                    retry_feedback = (
                        f"{last_failure}. Use the better direction above if present. Regenerate one fresh question "
                        f"that is more retrieval-useful, more specific, less label-driven, and less surface-aligned. "
                        f"Ask about the operative condition, exception, threshold, evidence requirement, or "
                        f"consequence without mentioning article numbers or labels."
                    )
                else:
                    retry_feedback = (
                        f"{last_failure}. Use the better direction above if present. Regenerate one fresh question "
                        f"that is more retrieval-useful, more specific, less generic, and less surface-aligned. "
                        f"Prefer one narrower technical fact over a broad summary or literal lookup."
                    )
                continue

            approved = True
            break

        if not approved:
            return {
                "index": index,
                "corpus_id": corpus_id,
                "rows": [],
                "status": f"skipped ({last_failure or 'validation failed'})",
            }

        english_corpus_id = target_corpus_ids.get("en", corpus_id)
        qac_rows = [{
            "corpus_id": english_corpus_id,
            "language": "en",
            "question": q_en,
            "answer": a_en,
        }]
        approved_translations: Dict[str, Tuple[str, str]] = {}
        failed_languages: List[str] = []
        translation_targets = [lang for lang in effective_target_languages if lang != "en"]
        for lang in translation_targets:
            lang_failure = "translation missing"
            retry_feedback: Optional[str] = None
            retry_q: Optional[str] = None
            retry_a: Optional[str] = None
            for _attempt in range(1, max_attempts + 1):
                trans = translate_qa(
                    client,
                    context,
                    q_en,
                    a_en,
                    [lang],
                    previous_feedback=retry_feedback,
                    previous_translated_question=retry_q,
                    previous_translated_answer=retry_a,
                    model=translation_model,
                    domain_hint=domain_hint,
                )
                if lang not in trans:
                    lang_failure = "translation missing"
                    retry_feedback = "The previous translation attempt was missing or malformed. Return valid JSON with a complete translated question and answer."
                    continue

                q, a = trans[lang]
                retry_q = q
                retry_a = a
                trans_check = check_translation_quality(
                    client,
                    context,
                    q_en,
                    a_en,
                    q,
                    a,
                    lang,
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
                        "Revise the translation by preserving the exact meaning, specificity, numbers, units, and source-grounded details from the English pair and source context."
                    )
                    if _normalize_domain_hint(domain_hint) == "legal":
                        feedback_parts.append(
                            "Do not introduce article numbers, provision labels, or phrases like 'this article' if they are absent from the English question."
                        )
                    feedback_parts.append(
                        "If the problem is fluency or grammar, rewrite more naturally in the target language without changing the information need."
                    )
                    retry_feedback = " ".join(feedback_parts)
                    continue

                approved_translations[lang] = (q, a)
                lang_failure = ""
                break

            if lang_failure:
                failed_languages.append(f"{lang} ({lang_failure})")

        for lang, (q, a) in approved_translations.items():
            qac_rows.append({
                "corpus_id": target_corpus_ids.get(lang, corpus_id),
                "language": lang,
                "question": q,
                "answer": a,
            })
        translation_status = f"{len(approved_translations)} translations"
        if failed_languages:
            translation_status += f", skipped {len(failed_languages)}: {', '.join(failed_languages)}"
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": qac_rows,
            "status": f"ok ({question_type or 'validated'} en + {translation_status})",
        }
    except Exception as exc:
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": [],
            "status": f"error: {exc}",
        }


def run_qa_pipeline(
    corpus_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 50,
    target_languages: Optional[List[str]] = None,
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
    If same_language is True: generate question and answer in each row's `language` field;
    one output row per sampled document (no translation).

    Writes qac.csv (corpus_id, language, question, answer) to output_dir.
    Returns number of QAC rows written.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGS
    if model is not None:
        generation_model = model
        quality_model = model
        support_model = model
        translation_model = model
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus(corpus_path)
    sampled = sample_corpus(rows, sample_size, stratify_by_language=True, seed=42)
    mode = "same-language (per row language)" if same_language else "English + translation"
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
                generation_model=generation_model,
                quality_model=quality_model,
                support_model=support_model,
                translation_model=translation_model,
                max_attempts=max_attempts,
                same_language=same_language,
                domain_hint=domain_hint,
            )
            results.append(result)
            tqdm.write(f"  [{index}/{len(sampled)}] {result['corpus_id']}... {result['status']}")

    for result in sorted(results, key=lambda item: item["index"]):
        qac_rows.extend(result["rows"])

    out_csv = output_dir / "qac.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus_id", "language", "question", "answer"])
        w.writeheader()
        w.writerows(qac_rows)

    print(f"Wrote {len(qac_rows)} QAC rows -> {out_csv}")
    return len(qac_rows)
