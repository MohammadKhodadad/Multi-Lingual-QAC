"""
Google Patents Public Datasets (BigQuery) loader for chemistry-related patents.

Uses the official BigQuery public datasets:
- patents-public-data.patents.publications
- patents-public-data.google_patents_research.publications (optional)
- patents-public-data.ebi_surechembl.match (optional, chemistry-specific)

Output: NDJSON with multilingual title_localized, abstract_localized, etc.

Requires: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth, and a Google Cloud project.
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


def clean_text(s: str) -> str:
    """Decode HTML entities and normalize whitespace."""
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Fields that BigQuery returns as repeated RECORDs; need JSON serialization
_RECORD_KEYS = [
    "title_localized",
    "abstract_localized",
    "claims_localized",
    "description_localized",
    "cpc",
    "ipc",
    "inventor_harmonized",
    "assignee_harmonized",
    "citation",
    "priority_claim",
    "research_cpc",
    "languages_present",
]

DEFAULT_CPC_PREFIXES = ["C", "A61K", "A61P"]
DEFAULT_IPC_PREFIXES = ["C", "A61K", "A61P"]
DEFAULT_LANGS = [
    "en", "de", "fr", "es", "ja", "ko", "zh",
    "ru", "pt", "it", "nl",
    "ar", "fa", "tr", "pl", "hi",
]
MIN_ABSTRACT_WORDS = 50
PER_LANGUAGE_OVERFETCH_FACTOR = 1.25
PER_LANGUAGE_OVERFETCH_MIN = 10


def sql_list(values: List[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def word_count(text: str) -> int:
    """Count whitespace-delimited words in cleaned text."""
    return len((text or "").split())


def build_query(
    *,
    languages: Optional[List[str]] = None,
    cpc_prefixes: Optional[List[str]] = None,
    ipc_prefixes: Optional[List[str]] = None,
    use_surechembl: bool = True,
    use_classification: bool = True,
    require_multilingual: bool = False,
    min_language_count: int = 2,
    limit: Optional[int] = None,
    primary_lang: Optional[str] = None,
    min_primary_abstract_words: Optional[int] = None,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    country_codes: Optional[List[str]] = None,
) -> str:
    """Build BigQuery SQL for chemistry-related multilingual patents."""
    languages = languages or DEFAULT_LANGS
    cpc_prefixes = cpc_prefixes or DEFAULT_CPC_PREFIXES
    ipc_prefixes = ipc_prefixes or DEFAULT_IPC_PREFIXES

    if not use_surechembl and not use_classification:
        raise ValueError("At least one of use_surechembl or use_classification must be True.")

    lang_sql = sql_list(languages)

    date_filter = ""
    if start_date is not None:
        date_filter += f"\n  AND p.publication_date >= {start_date}"
    if end_date is not None:
        date_filter += f"\n  AND p.publication_date <= {end_date}"

    country_filter = ""
    if country_codes:
        cc_sql = sql_list(country_codes)
        country_filter = f"\n  AND p.country_code IN ({cc_sql})"

    chemistry_predicates: List[str] = []

    if use_classification:
        cpc_preds = " OR ".join(f"STARTS_WITH(c.code, {p!r})" for p in cpc_prefixes)
        ipc_preds = " OR ".join(f"STARTS_WITH(i.code, {p!r})" for p in ipc_prefixes)
        chemistry_predicates.append(
            f"""
            EXISTS (
              SELECT 1
              FROM UNNEST(IFNULL(p.cpc, [])) AS c
              WHERE {cpc_preds}
            )
            """
        )
        chemistry_predicates.append(
            f"""
            EXISTS (
              SELECT 1
              FROM UNNEST(IFNULL(p.ipc, [])) AS i
              WHERE {ipc_preds}
            )
            """
        )

    surechembl_join = ""
    surechembl_flag = "FALSE AS has_surechembl_match"
    if use_surechembl:
        surechembl_join = """
        LEFT JOIN (
          SELECT DISTINCT publication_number
          FROM `patents-public-data.ebi_surechembl.match`
        ) sc
        ON p.publication_number = sc.publication_number
        """
        surechembl_flag = "sc.publication_number IS NOT NULL AS has_surechembl_match"
        chemistry_predicates.append("sc.publication_number IS NOT NULL")

    chemistry_where = " OR ".join(f"({p.strip()})" for p in chemistry_predicates)

    # Filter to patents that have a usable localized abstract for primary_lang
    # before limiting. This lets per-language extraction target documents that
    # are likely to survive later preprocessing.
    primary_lang_filter = ""
    if primary_lang:
        pl = primary_lang.strip().lower()
        min_words_filter = ""
        if min_primary_abstract_words:
            min_words_filter = f"""
              AND ARRAY_LENGTH(
                SPLIT(REGEXP_REPLACE(TRIM(a.text), r'\\s+', ' '), ' ')
              ) >= {int(min_primary_abstract_words)}
            """
        primary_lang_filter = f"""
        AND EXISTS (
          SELECT 1
          FROM UNNEST(IFNULL(p.abstract_localized, [])) a
          WHERE LOWER(COALESCE(a.language, '')) = {pl!r}
            AND a.text IS NOT NULL
            AND LENGTH(TRIM(a.text)) > 0
            {min_words_filter}
        )
        """

    multilingual_having = ""
    if require_multilingual:
        multilingual_having = f"HAVING ARRAY_LENGTH(languages_present) >= {min_language_count}"

    limit_clause = f"\nLIMIT {limit}" if limit else ""

    query = f"""
    WITH base AS (
      SELECT
        p.publication_number,
        p.application_number,
        p.country_code,
        p.kind_code,
        p.application_kind,
        p.application_number_formatted,
        p.pct_number,
        p.family_id,
        p.publication_date,
        p.filing_date,
        p.grant_date,
        p.priority_date,

        p.title_localized,
        p.abstract_localized,
        p.claims_localized,
        p.description_localized,

        p.cpc,
        p.ipc,
        p.inventor_harmonized,
        p.assignee_harmonized,
        p.citation,
        p.priority_claim,

        {surechembl_flag},

        gr.title AS english_title_research,
        gr.title_translated AS english_title_machine_translated,
        gr.abstract AS english_abstract_research,
        gr.abstract_translated AS english_abstract_machine_translated,
        gr.cpc AS research_cpc,

        ARRAY(
          SELECT DISTINCT t.language
          FROM UNNEST(IFNULL(p.title_localized, [])) t
          WHERE t.language IN ({lang_sql}) AND t.text IS NOT NULL
          UNION DISTINCT
          SELECT DISTINCT a.language
          FROM UNNEST(IFNULL(p.abstract_localized, [])) a
          WHERE a.language IN ({lang_sql}) AND a.text IS NOT NULL
        ) AS languages_present
      FROM `patents-public-data.patents.publications` p
      LEFT JOIN `patents-public-data.google_patents_research.publications` gr
        ON p.publication_number = gr.publication_number
      {surechembl_join}
      WHERE
        ({chemistry_where})
        {primary_lang_filter}
        {date_filter}
        {country_filter}
    )
    SELECT *
    FROM base
    {multilingual_having}
    ORDER BY publication_date DESC
    {limit_clause}
    """
    return query


def _serialize_record(obj: Any) -> Any:
    """Convert BigQuery row values to JSON-serializable Python types."""
    return json.loads(json.dumps(obj, default=str))


def _run_query_iter(
    project_id: str,
    query: str,
    *,
    page_size: int = 1000,
):
    """Run BigQuery query and yield serialized record dicts."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
    query_job = client.query(query, job_config=job_config)
    result = query_job.result(page_size=page_size)

    for row in result:
        record: Dict[str, Any] = dict(row.items())
        for key in _RECORD_KEYS:
            if key in record and record[key] is not None:
                record[key] = _serialize_record(record[key])
        yield record


def run_query(
    project_id: str,
    query: str,
    output_path: Path,
    *,
    page_size: int = 1000,
) -> int:
    """
    Run BigQuery query and write results as NDJSON.
    Returns the number of rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in _run_query_iter(project_id, query, page_size=page_size):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"Wrote {count:,} rows...")
    print(f"Done. Wrote {count:,} rows to: {output_path}")
    return count


def extract_chemistry_patents(
    project_id: str,
    output_path: Path,
    *,
    languages: Optional[List[str]] = None,
    cpc_prefixes: Optional[List[str]] = None,
    ipc_prefixes: Optional[List[str]] = None,
    use_surechembl: bool = True,
    use_classification: bool = True,
    require_multilingual: bool = False,
    min_language_count: int = 2,
    limit: Optional[int] = None,
    primary_lang: Optional[str] = None,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    country_codes: Optional[List[str]] = None,
) -> int:
    """
    Extract chemistry-related multilingual patents from Google Patents BigQuery.
    Writes NDJSON to output_path.
    Returns number of rows written.
    """
    query = build_query(
        languages=languages,
        cpc_prefixes=cpc_prefixes,
        ipc_prefixes=ipc_prefixes,
        use_surechembl=use_surechembl,
        use_classification=use_classification,
        require_multilingual=require_multilingual,
        min_language_count=min_language_count,
        limit=limit,
        primary_lang=primary_lang,
        start_date=start_date,
        end_date=end_date,
        country_codes=country_codes,
    )
    return run_query(
        project_id=project_id,
        query=query,
        output_path=Path(output_path),
    )


def extract_chemistry_patents_per_language(
    project_id: str,
    output_path: Path,
    *,
    languages: Optional[List[str]] = None,
    limit_per_lang: int = 100,
    cpc_prefixes: Optional[List[str]] = None,
    ipc_prefixes: Optional[List[str]] = None,
    use_surechembl: bool = True,
    use_classification: bool = True,
) -> int:
    """
    For each language, pull a slightly overfetched set of patents that have a
    usable abstract in that language, then append all to one NDJSON.

    The overfetch helps preserve up to limit_per_lang usable rows per language
    after downstream cleaning and safety checks.
    """
    languages = languages or DEFAULT_LANGS
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for lang in tqdm(languages, desc="Languages", unit="lang"):
            fetch_limit = max(
                limit_per_lang + PER_LANGUAGE_OVERFETCH_MIN,
                int(limit_per_lang * PER_LANGUAGE_OVERFETCH_FACTOR),
            )
            query = build_query(
                cpc_prefixes=cpc_prefixes or DEFAULT_CPC_PREFIXES,
                ipc_prefixes=ipc_prefixes or DEFAULT_IPC_PREFIXES,
                use_surechembl=use_surechembl,
                use_classification=use_classification,
                limit=fetch_limit,
                primary_lang=lang,
                min_primary_abstract_words=MIN_ABSTRACT_WORDS,
            )
            tqdm.write(
                f"  {lang}: pulling up to {fetch_limit} candidates "
                f"(target {limit_per_lang} usable docs)..."
            )
            n = 0
            for record in _run_query_iter(project_id, query):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
                total += 1
            tqdm.write(f"  {lang}: {n} rows")
    tqdm.write(f"Done. Wrote {total:,} rows to: {output_path}")
    return total


def _get_localized_text(
    items: Optional[List[Dict[str, Any]]],
    lang: str,
) -> Optional[str]:
    """Extract text for a given language from title_localized or abstract_localized."""
    if not items:
        return None
    for item in items:
        if isinstance(item, dict):
            lang_val = (item.get("language") or "").lower()
            if lang_val == lang.lower():
                text = item.get("text")
                if text and text.strip():
                    return text.strip()
    return None


def preprocess_ndjson_to_csv(
    ndjson_path: Path,
    output_dir: Path,
    *,
    languages: Optional[List[str]] = None,
    per_lang_limit: Optional[int] = None,
    min_abstract_words: int = MIN_ABSTRACT_WORDS,
) -> Dict[str, int]:
    """
    Preprocess NDJSON patent data into per-language CSVs for QAC generation.

    For each language, extracts records that have title/abstract in that language,
    dedupes by publication_number, optionally caps at per_lang_limit rows, writes CSV.

    CSV columns: id, language, title, abstract, context, publication_number,
    country_code, publication_date, source

    Returns dict mapping language -> row count.
    """
    languages = languages or DEFAULT_LANGS
    ndjson_path = Path(ndjson_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all records once
    records: List[Dict[str, Any]] = []
    with ndjson_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    counts: Dict[str, int] = {}
    fieldnames = [
        "id",
        "language",
        "title",
        "abstract",
        "context",
        "publication_number",
        "country_code",
        "publication_date",
        "source",
    ]

    for lang in tqdm(languages, desc="Preprocess languages", unit="lang"):
        rows: List[Dict[str, Any]] = []
        seen_pub: set = set()
        skipped_short = 0
        for rec in records:
            title = _get_localized_text(rec.get("title_localized"), lang)
            abstract = _get_localized_text(rec.get("abstract_localized"), lang)

            if not abstract and not title:
                continue

            pub_num = rec.get("publication_number") or ""
            if pub_num in seen_pub:
                continue
            seen_pub.add(pub_num)

            title = clean_text(title or "")
            abstract = clean_text(abstract or "")
            if word_count(abstract) < min_abstract_words:
                skipped_short += 1
                continue
            context = f"{title}\n\n{abstract}".strip() if title else abstract

            rows.append({
                "id": f"{pub_num}_{lang}",
                "language": lang,
                "title": title,
                "abstract": abstract,
                "context": context,
                "publication_number": pub_num,
                "country_code": rec.get("country_code") or "",
                "publication_date": rec.get("publication_date") or "",
                "source": "google_patents",
            })

            if per_lang_limit and len(rows) >= per_lang_limit:
                break

        out_path = output_dir / f"{lang}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        counts[lang] = len(rows)
        tqdm.write(
            f"  {lang}: {len(rows):,} rows -> {out_path}"
            f" (skipped {skipped_short:,} short/title-only records)"
        )

    return counts


def merge_corpus_csv(
    preprocessed_dir: Path,
    output_path: Path,
    *,
    languages: Optional[List[str]] = None,
    min_abstract_words: int = MIN_ABSTRACT_WORDS,
) -> int:
    """
    Merge all per-language CSVs into one corpus CSV. Applies clean_text.
    Corpus = documents for retrieval; queries/answers come from QAC generation.
    """
    preprocessed_dir = Path(preprocessed_dir)
    output_path = Path(output_path)
    languages = languages or DEFAULT_LANGS

    rows: List[Dict[str, Any]] = []
    for lang in languages:
        p = preprocessed_dir / f"{lang}.csv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                row["title"] = clean_text(row.get("title", ""))
                row["abstract"] = clean_text(row.get("abstract", ""))
                row["context"] = clean_text(row.get("context", ""))
                if word_count(row["abstract"]) < min_abstract_words:
                    continue
                if not row["context"]:
                    row["context"] = f"{row['title']}\n\n{row['abstract']}".strip()
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "language", "title", "abstract", "context",
        "publication_number", "country_code", "publication_date", "source",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Merged {len(rows):,} rows -> {output_path}")
    return len(rows)
