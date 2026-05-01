from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

SYSTEM_PROMPT = (
    "You are an expert fitness and nutrition coach built into a workout planning app. "
    "Give evidence-based, concise advice in 3 to 5 sentences. "
    "Always reference the user's level, goal, and current plan when relevant. "
    "Only mention urgent medical escalation for genuine red-flag scenarios. "
    "Use a direct, practical coaching tone."
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "Nutrition-and-Physical-Fitness-1660622249._print.md"
OUT_DIR = ROOT / "artifacts" / "datasets"
OUT_JSONL = OUT_DIR / "nutrition_textbook_full_qa.jsonl"
OUT_COMBINED = OUT_DIR / "full_llm_training_data.jsonl"

NOISE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^\d+\s*\|\s*"),
    re.compile(r"^Figure\s+\d+(?:\.\d+)*"),
    re.compile(r"^Table\s+\d+(?:\.\d+)*"),
    re.compile(r"^Media Attributions?", re.IGNORECASE),
    re.compile(r"^Chapter Attributions?", re.IGNORECASE),
    re.compile(r"^References$", re.IGNORECASE),
    re.compile(r"^About the Contributors$", re.IGNORECASE),
    re.compile(r"^PART\s+[IVXLC]+", re.IGNORECASE),
]

HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+"),
    re.compile(r"^\d+\.\d+(?:\.\d+)?\s+.+"),
    re.compile(r"^Chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^Introduction$", re.IGNORECASE),
]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if len(s) <= 3 and s.isdigit():
        return True
    for pat in NOISE_PATTERNS:
        if pat.search(s):
            return True
    return False


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 110:
        return False
    for pat in HEADING_PATTERNS:
        if pat.match(s):
            return True
    return False


def clean_heading(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^#{1,6}\s+", "", s)
    s = re.sub(r"\s*\|\s*\d+\s*$", "", s)
    s = re.sub(r"\s+\d+\s*$", "", s)
    s = normalize_spaces(s)
    return s


def clean_section_text(text: str) -> str:
    t = text
    t = t.replace("```", " ")
    t = t.replace("•", " ")
    t = re.sub(r"\|\s*\d+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def sentence_split(text: str) -> list[str]:
    text = clean_section_text(text)
    text = normalize_spaces(text)
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for s in raw:
        s2 = normalize_spaces(s)
        if not s2:
            continue
        if len(s2) < 25:
            continue
        if re.match(r"^[0-9\s\-|]+$", s2):
            continue
        if re.search(r"(licensed under|Adobe Stock|CC BY|All Rights Reserved)", s2, flags=re.IGNORECASE):
            continue
        out.append(s2)
    return out


def section_to_answer(section_text: str, max_sentences: int = 4) -> str:
    sents = sentence_split(section_text)
    if not sents:
        return "This section emphasizes evidence-based nutrition and fitness principles and how to apply them consistently in real life."

    selected: list[str] = []

    for sent in sents:
        if len(selected) >= max_sentences:
            break
        selected.append(sent)

    answer = " ".join(selected)
    return normalize_spaces(answer)


def get_main_body(md_text: str) -> str:
    """Drop early table-of-contents/front-matter noise and keep the actual textbook body."""
    marker = "This open access textbook was created for NFSC 303"
    idx = md_text.find(marker)
    if idx == -1:
        return md_text
    # Keep a little context before the marker so the opening heading is retained.
    start = max(0, idx - 200)
    return md_text[start:]


def iter_sections(md_text: str) -> Iterable[tuple[str, str]]:
    lines = md_text.splitlines()

    current_heading = "Course Overview"
    buffer: list[str] = []
    allow_heading_continuation = False

    continuation_tail_words = {
        "in", "of", "and", "to", "for", "other", "the", "a", "an", "with", "from"
    }

    for line in lines:
        if is_heading(line):
            if buffer:
                body = "\n".join(buffer).strip()
                if body:
                    yield current_heading, body
            current_heading = clean_heading(line)
            buffer = []

            last_word = current_heading.split()[-1].lower() if current_heading.split() else ""
            allow_heading_continuation = last_word in continuation_tail_words
            continue

        if is_noise_line(line):
            continue

        stripped = line.strip()

        # Merge wrapped heading lines like "1.1 State of Health in" + "the US".
        if allow_heading_continuation and not is_heading(stripped):
            if len(stripped) <= 60 and not re.search(r"[.!?]$", stripped):
                current_heading = normalize_spaces(f"{current_heading} {stripped}")
                allow_heading_continuation = False
                continue
            allow_heading_continuation = False

        # Keep bullets and narrative text.
        buffer.append(stripped)

    if buffer:
        body = "\n".join(buffer).strip()
        if body:
            yield current_heading, body


def make_question_variants(heading: str) -> list[str]:
    base = heading.strip()
    return [
        f"What are the key concepts in {base}?",
        f"How should I apply the guidance from {base} in practice?",
    ]


def build_records(md_text: str) -> list[dict]:
    records: list[dict] = []
    seen_questions: set[str] = set()

    md_text = get_main_body(md_text)

    for heading, section_text in iter_sections(md_text):
        # Skip headings that are clearly table/figure/attribution leftovers.
        if re.search(r"(Attributions?|Figure|Table|Logo|Contents|References)", heading, flags=re.IGNORECASE):
            continue

        answer = section_to_answer(section_text)

        # Filter out low-information/generic fallback sections.
        if "This section emphasizes evidence-based nutrition and fitness principles" in answer:
            continue
        if len(answer) < 150:
            continue
        if len(sentence_split(answer)) < 2:
            continue

        for q in make_question_variants(heading):
            q_norm = normalize_spaces(q)
            if q_norm in seen_questions:
                continue
            seen_questions.add(q_norm)

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q_norm},
                    {"role": "assistant", "content": answer},
                ]
            }
            records.append(record)

    return records


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_records(records: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for r in records:
        msgs = r.get("messages", [])
        user = ""
        assistant = ""
        for m in msgs:
            role = m.get("role")
            if role == "user":
                user = normalize_spaces(m.get("content", ""))
            elif role == "assistant":
                assistant = normalize_spaces(m.get("content", ""))
        key = (user, assistant)
        if not user or not assistant:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def main() -> None:
    if not SOURCE_MD.exists():
        raise FileNotFoundError(f"Missing source markdown file: {SOURCE_MD}")

    md_text = SOURCE_MD.read_text(encoding="utf-8", errors="ignore")
    new_records = build_records(md_text)
    write_jsonl(OUT_JSONL, new_records)

    seed_paths = [
        OUT_DIR / "fitness_training_data.jsonl",
        OUT_DIR / "nutrition_textbook_data.jsonl",
        OUT_JSONL,
    ]

    combined: list[dict] = []
    for p in seed_paths:
        combined.extend(read_jsonl(p))

    deduped = unique_records(combined)
    write_jsonl(OUT_COMBINED, deduped)

    summary = {
        "source_markdown": str(SOURCE_MD.name),
        "generated_dataset": str(OUT_JSONL.relative_to(ROOT).as_posix()),
        "generated_samples": len(new_records),
        "combined_dataset": str(OUT_COMBINED.relative_to(ROOT).as_posix()),
        "combined_samples": len(deduped),
    }
    summary_path = ROOT / "data" / "dataset_summary_full.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Generated {len(new_records)} textbook Q&A samples -> {OUT_JSONL}")
    print(f"Combined dataset has {len(deduped)} samples -> {OUT_COMBINED}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
