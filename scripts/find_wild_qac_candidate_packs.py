"""Find QAC generation groups where the selected query was good but siblings vary.

The annotator workbook contains the selected/generated QAC rows with human
scores. The candidate CSV contains the three generated candidates per generation
group. This script matches good annotated questions back to the candidate CSV,
retrieves the other two candidates from the same source/generation group, and
exports packs of three questions with both generated and annotator scores.
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


SHEET_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
GROUP_COLUMNS = ("corpus_id", "mode", "strategy", "question_language", "context_language")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def column_number(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter.upper()) - 64
    return number


def normalize_target(target: str) -> str:
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read an .xlsx worksheet using only the Python standard library."""
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", SHEET_NS):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", SHEET_NS))
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {
            rel.attrib["Id"]: normalize_target(rel.attrib["Target"])
            for rel in rels.findall("rel:Relationship", REL_NS)
        }

        sheet_path = None
        for sheet in workbook.findall("a:sheets/a:sheet", SHEET_NS):
            if sheet.attrib["name"] == sheet_name:
                rel_id = sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                sheet_path = relmap[rel_id]
                break
        if sheet_path is None:
            raise ValueError(f"Sheet {sheet_name!r} not found in {path}")

        root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[str]] = []
        for row in root.findall("a:sheetData/a:row", SHEET_NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", SHEET_NS):
                index = column_number(cell.attrib.get("r", ""))
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", SHEET_NS)
                inline_node = cell.find("a:is", SHEET_NS)
                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text or 0)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(
                        text.text or "" for text in inline_node.findall(".//a:t", SHEET_NS)
                    )
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""
                values[index] = value
            if values:
                rows.append([values.get(i, "") for i in range(1, max(values) + 1)])

    if not rows:
        return []
    header = rows[0]
    return [dict(zip(header, row + [""] * (len(header) - len(row)))) for row in rows[1:]]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(normalize_text(row.get(column, "")) for column in GROUP_COLUMNS)


def question_key(row: dict[str, str]) -> tuple[tuple[str, ...], str]:
    return group_key(row), normalize_text(row.get("question", ""))


def find_packs(
    annotated_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    min_annotator_score: float,
    min_generated_spread: float,
    low_generated_score: float,
    include_all: bool,
) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    by_question: dict[tuple[tuple[str, ...], str], dict[str, str]] = {}
    fallback_by_question: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in candidate_rows:
        groups[group_key(row)].append(row)
        by_question[question_key(row)] = row
        fallback_by_question[
            (normalize_text(row.get("corpus_id", "")), normalize_text(row.get("question", "")))
        ].append(row)

    output_rows: list[dict[str, str]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for annotated in annotated_rows:
        annotator_score = to_float(annotated.get("total_score", ""))
        if annotator_score is None or annotator_score < min_annotator_score:
            continue

        selected = by_question.get(question_key(annotated))
        if selected is None:
            fallback_matches = fallback_by_question.get(
                (
                    normalize_text(annotated.get("corpus_id", "")),
                    normalize_text(annotated.get("question", "")),
                ),
                [],
            )
            selected = fallback_matches[0] if len(fallback_matches) == 1 else None
        if selected is None:
            continue

        key = group_key(selected)
        if key in seen_groups:
            continue
        candidates = groups[key]
        if len(candidates) != 3:
            continue

        generated_scores = [to_float(row.get("total_score", "")) for row in candidates]
        if any(score is None for score in generated_scores):
            continue
        numeric_scores = [score for score in generated_scores if score is not None]
        score_min = min(numeric_scores)
        score_max = max(numeric_scores)
        score_spread = score_max - score_min
        is_wild = score_spread >= min_generated_spread or score_min <= low_generated_score
        if not include_all and not is_wild:
            continue

        sorted_candidates = sorted(
            candidates,
            key=lambda row: (to_float(row.get("total_score", "")) or -1),
            reverse=True,
        )
        selected_question = normalize_text(selected.get("question", ""))
        selected_rank = next(
            index + 1
            for index, row in enumerate(sorted_candidates)
            if normalize_text(row.get("question", "")) == selected_question
        )
        seen_groups.add(key)

        for index, candidate in enumerate(sorted_candidates, start=1):
            is_selected = normalize_text(candidate.get("question", "")) == selected_question
            output_rows.append(
                {
                    "pack_id": "|".join(key),
                    "candidate_rank_by_generated_score": str(index),
                    "is_selected_for_annotation": "yes" if is_selected else "no",
                    "selected_rank_by_generated_score": str(selected_rank),
                    "is_wild_pack": "yes" if is_wild else "no",
                    "generated_score_spread": f"{score_spread:g}",
                    "generated_score_min": f"{score_min:g}",
                    "generated_score_max": f"{score_max:g}",
                    "annotator_total_score_selected": f"{annotator_score:g}",
                    "annotator_failure_type_selected": annotated.get("qual_failure_type", ""),
                    "selected_source_passages": annotated.get("passages", ""),
                    "corpus_id": candidate.get("corpus_id", ""),
                    "publication_number": candidate.get("publication_number", ""),
                    "mode": candidate.get("mode", ""),
                    "strategy": candidate.get("strategy", ""),
                    "question_language": candidate.get("question_language", ""),
                    "context_language": candidate.get("context_language", ""),
                    "generated_total_score": candidate.get("total_score", ""),
                    "generated_faith_overall": candidate.get("faith_overall", ""),
                    "generated_quality_overall": candidate.get("qual_overall", ""),
                    "generated_failure_type": candidate.get("qual_failure_type", ""),
                    "question": candidate.get("question", ""),
                    "answer": candidate.get("answer", ""),
                }
            )

    output_rows.sort(
        key=lambda row: (
            -float(row["generated_score_spread"]),
            row["pack_id"],
            int(row["candidate_rank_by_generated_score"]),
        )
    )
    return output_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], max_packs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        packs[row["pack_id"]].append(row)

    lines = ["# Wild QAC Candidate Packs", ""]
    lines.append(f"Showing {min(max_packs, len(packs))} of {len(packs)} matched packs.")
    lines.append("")
    for pack_index, (pack_id, pack_rows) in enumerate(packs.items(), start=1):
        if pack_index > max_packs:
            break
        selected = next(row for row in pack_rows if row["is_selected_for_annotation"] == "yes")
        lines.extend(
            [
                f"## Pack {pack_index}: `{pack_id}`",
                "",
                f"- Generated score range: {selected['generated_score_min']}--{selected['generated_score_max']} "
                f"(spread {selected['generated_score_spread']})",
                f"- Selected candidate rank by generated score: {selected['selected_rank_by_generated_score']}",
                f"- Annotator score for selected candidate: {selected['annotator_total_score_selected']}",
                f"- Annotator failure type: `{selected['annotator_failure_type_selected']}`",
                "",
                "### Corpus / Source Passages",
                "",
                "```json",
                selected.get("selected_source_passages", "")[:2500],
                "```",
                "",
            ]
        )
        for row in pack_rows:
            marker = "SELECTED" if row["is_selected_for_annotation"] == "yes" else "rejected"
            lines.extend(
                [
                    f"### Candidate {row['candidate_rank_by_generated_score']} ({marker})",
                    "",
                    f"- Generated total score: {row['generated_total_score']}",
                    f"- Generated failure type: `{row['generated_failure_type']}`",
                    f"- Question: {row['question']}",
                    f"- Answer: {row['answer']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated-xlsx", default="data/Evaluated data_by_annotator.xlsx")
    parser.add_argument("--candidate-csv", default="data/qac_chempatents.csv")
    parser.add_argument("--sheet", default="qac_with_modes")
    parser.add_argument("--min-annotator-score", type=float, default=8.0)
    parser.add_argument("--min-generated-spread", type=float, default=6.0)
    parser.add_argument("--low-generated-score", type=float, default=25.0)
    parser.add_argument("--include-all", action="store_true")
    parser.add_argument("--output-csv", default="reports/wild_qac_candidate_packs.csv")
    parser.add_argument("--output-md", default="reports/wild_qac_candidate_packs.md")
    parser.add_argument("--max-md-packs", type=int, default=10)
    args = parser.parse_args()

    annotated_rows = read_xlsx_sheet(Path(args.annotated_xlsx), args.sheet)
    candidate_rows = read_csv_rows(Path(args.candidate_csv))
    packs = find_packs(
        annotated_rows=annotated_rows,
        candidate_rows=candidate_rows,
        min_annotator_score=args.min_annotator_score,
        min_generated_spread=args.min_generated_spread,
        low_generated_score=args.low_generated_score,
        include_all=args.include_all,
    )
    write_csv(Path(args.output_csv), packs)
    write_markdown(Path(args.output_md), packs, args.max_md_packs)

    pack_count = len({row["pack_id"] for row in packs})
    selected_count = sum(1 for row in packs if row["is_selected_for_annotation"] == "yes")
    print(f"Annotated rows read: {len(annotated_rows)}")
    print(f"Candidate rows read: {len(candidate_rows)}")
    print(f"Matched packs written: {pack_count}")
    print(f"Selected annotated candidates in packs: {selected_count}")
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
