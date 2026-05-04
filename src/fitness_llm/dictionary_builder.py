from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ProjectPaths
from .prompts import SYSTEM_PROMPT


@dataclass(frozen=True)
class DictionaryPaths:
    domain_jsonl: Path
    user_schema_json: Path
    user_profiles_jsonl: Path
    instruction_jsonl: Path
    finetune_jsonl: Path
    summary_json: Path


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "item"


def _normalize(value: str) -> str:
    value = value.strip().lower()
    return re.sub(r"\s+", " ", value)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return "; ".join(parts)
    return str(value).strip()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _infer_concept_type(source: str) -> str:
    src = source.lower()
    if "nutrition" in src:
        return "nutrition_rule"
    if "injury" in src or "recover" in src:
        return "safety_guideline"
    if "program" in src:
        return "programming_principle"
    return "concept"


def _title_from_text(text: str, limit: int = 90) -> str:
    first = text.split(".")[0].strip()
    if not first:
        first = text.strip()
    return first[:limit].rstrip()


def build_domain_dictionary(root: Path) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()

    def add_item(item: dict) -> None:
        key = "|".join(
            [
                _normalize(item.get("type", "")),
                _normalize(item.get("title", "")),
                _normalize(item.get("body", "")[:240]),
            ]
        )
        if not key or key in seen:
            return
        seen.add(key)
        items.append(item)

    knowledge_seed_path = root / "data" / "knowledge_seed.json"
    if knowledge_seed_path.exists():
        for idx, row in enumerate(_load_json(knowledge_seed_path), start=1):
            source = _to_text(row.get("source") or "knowledge_seed")
            body = _to_text(row.get("text"))
            if not body:
                continue
            entry_type = _infer_concept_type(source)
            add_item(
                {
                    "id": f"concept_{idx:06d}",
                    "type": entry_type,
                    "title": _title_from_text(body),
                    "body": body,
                    "tags": sorted({_slugify(source), _slugify(entry_type)}),
                    "source": source,
                    "metadata": {"origin": "knowledge_seed"},
                }
            )

    examples_path = root / "data" / "full_training_examples.json"
    if examples_path.exists():
        for idx, row in enumerate(_load_json(examples_path), start=1):
            user = _to_text(row.get("user"))
            answer = _to_text(row.get("assistant"))
            category = _to_text(row.get("category") or "faq")
            if not user or not answer:
                continue
            add_item(
                {
                    "id": f"faq_seed_{idx:06d}",
                    "type": "faq_answer",
                    "title": user,
                    "body": answer,
                    "tags": sorted({_slugify(category), "faq_answer"}),
                    "source": "data/full_training_examples.json",
                    "metadata": {"category": category},
                }
            )

    full_dataset_path = root / "artifacts" / "datasets" / "fitness_training_data_full.jsonl"
    for idx, row in enumerate(_read_jsonl(full_dataset_path), start=1):
        msgs = row.get("messages", [])
        user = _to_text(next((m.get("content") for m in msgs if m.get("role") == "user"), ""))
        answer = _to_text(next((m.get("content") for m in msgs if m.get("role") == "assistant"), ""))
        if not user or not answer:
            continue
        add_item(
            {
                "id": f"faq_full_{idx:06d}",
                "type": "faq_answer",
                "title": user,
                "body": answer,
                "tags": ["faq_answer", "full_dataset"],
                "source": "artifacts/datasets/fitness_training_data_full.jsonl",
                "metadata": {"origin": "full_dataset"},
            }
        )

    exercisedb_path = root / "exercisedb" / "exercisedb_v1_sample" / "exercises.json"
    if exercisedb_path.exists():
        for row in _load_json(exercisedb_path):
            name = _to_text(row.get("name"))
            ex_id = _to_text(row.get("exerciseId"))
            if not name or not ex_id:
                continue
            body_parts = row.get("bodyParts", []) or []
            targets = row.get("targetMuscles", []) or []
            secondary = row.get("secondaryMuscles", []) or []
            equipment = row.get("equipments", []) or []
            instructions = [
                _to_text(s).replace("Step:", "") for s in (row.get("instructions", []) or []) if _to_text(s)
            ]
            body = (
                f"{name} targets {_to_text(targets) or 'multiple muscles'}. "
                f"Body parts: {_to_text(body_parts) or 'unspecified'}. "
                f"Secondary muscles: {_to_text(secondary) or 'none listed'}. "
                f"Equipment: {_to_text(equipment) or 'none listed'}. "
                f"Execution cues: {' '.join(instructions[:6])}".strip()
            )
            tags = [_slugify(t) for t in [*targets, *body_parts, *equipment] if _to_text(t)]
            add_item(
                {
                    "id": f"exercise_exdb_{_slugify(ex_id)}",
                    "type": "exercise",
                    "title": name,
                    "body": body,
                    "tags": sorted(set(tags[:20] + ["exercise"])),
                    "source": "exercisedb/exercisedb_v1_sample/exercises.json",
                    "metadata": {
                        "difficulty": "unknown",
                        "target_muscles": targets,
                        "secondary_muscles": secondary,
                        "equipment": equipment,
                    },
                }
            )

    for file_name, source_name in [
        ("strength.json", "strength"),
        ("flexibility.json", "flexibility"),
        ("cardio.json", "cardio"),
    ]:
        longhaul_path = root / "longhaul_fitness" / file_name
        if not longhaul_path.exists():
            continue
        for row in _load_json(longhaul_path):
            name = _to_text(row.get("name"))
            slug = _to_text(row.get("slug"))
            if not name:
                continue
            primary = row.get("primaryMuscles", []) or []
            secondary = row.get("secondaryMuscles", []) or []
            steps = row.get("steps", []) or []
            notes = _to_text(row.get("notes"))
            body = (
                f"{name}. Primary muscles: {_to_text(primary) or 'unspecified'}. "
                f"Secondary muscles: {_to_text(secondary) or 'unspecified'}. "
                f"Steps: {' '.join(_to_text(s) for s in steps[:6])}. "
                f"Notes: {notes}"
            ).strip()
            tags = [_slugify(t) for t in [*primary, *secondary, source_name] if _to_text(t)]
            add_item(
                {
                    "id": f"exercise_longhaul_{_slugify(slug or name)}",
                    "type": "exercise",
                    "title": name,
                    "body": body,
                    "tags": sorted(set(tags[:20] + ["exercise"])),
                    "source": f"longhaul_fitness/{file_name}",
                    "metadata": {
                        "difficulty": "unknown",
                        "primary_muscles": primary,
                        "secondary_muscles": secondary,
                    },
                }
            )

    megagym_path = root / "megaGymDataset" / "megaGymDataset.csv"
    if megagym_path.exists():
        with megagym_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                title = _to_text(row.get("Title"))
                desc = _to_text(row.get("Desc"))
                if not title or not desc:
                    continue
                body_part = _to_text(row.get("BodyPart"))
                equipment = _to_text(row.get("Equipment"))
                level = _to_text(row.get("Level"))
                ex_type = _to_text(row.get("Type"))
                add_item(
                    {
                        "id": f"exercise_megagym_{_slugify(title)}",
                        "type": "exercise",
                        "title": title,
                        "body": desc,
                        "tags": sorted(
                            set(
                                [
                                    "exercise",
                                    _slugify(body_part),
                                    _slugify(equipment),
                                    _slugify(level),
                                    _slugify(ex_type),
                                ]
                            )
                        ),
                        "source": "megaGymDataset/megaGymDataset.csv",
                        "metadata": {
                            "difficulty": level or "unknown",
                            "body_part": body_part,
                            "equipment": equipment,
                            "exercise_type": ex_type,
                        },
                    }
                )

    return items


def build_user_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "FitnessLLMUserDictionary",
        "type": "object",
        "required": ["user_id", "profile", "program", "state"],
        "properties": {
            "user_id": {"type": "string"},
            "profile": {
                "type": "object",
                "required": ["level", "goal", "constraints", "preferences"],
                "properties": {
                    "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                    "training_age_years": {"type": "number"},
                    "goal": {"type": "string"},
                    "primary_focus": {"type": "array", "items": {"type": "string"}},
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "injuries": {"type": "array", "items": {"type": "string"}},
                            "equipment": {"type": "array", "items": {"type": "string"}},
                            "schedule": {"type": "string"},
                        },
                    },
                    "preferences": {
                        "type": "object",
                        "properties": {
                            "disliked_exercises": {"type": "array", "items": {"type": "string"}},
                            "preferred_rep_ranges": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
            "program": {
                "type": "object",
                "properties": {
                    "split": {"type": "string"},
                    "days_per_week": {"type": "integer"},
                    "volume_summary": {"type": "object"},
                    "microcycle": {"type": "array", "items": {"type": "object"}},
                },
            },
            "state": {
                "type": "object",
                "properties": {
                    "perceived_fatigue": {"type": "string"},
                    "adherence_last_7_days": {"type": "number"},
                    "recent_issues": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }


def build_user_profiles() -> list[dict]:
    levels = ["beginner", "intermediate", "advanced"]
    goals = ["hypertrophy", "strength", "fat_loss", "muscle_gain", "recomp", "general_fitness"]
    focuses = [
        ["chest", "shoulders"],
        ["back", "hamstrings"],
        ["quads", "glutes"],
        ["general_conditioning"],
    ]
    schedules = [
        (3, ["Mon", "Wed", "Fri"], "full_body"),
        (4, ["Mon", "Tue", "Thu", "Sat"], "upper_lower"),
        (5, ["Mon", "Tue", "Thu", "Fri", "Sat"], "push_pull_legs_plus"),
    ]
    equipment_options = [
        ["home_dumbbells", "pullup_bar"],
        ["commercial_gym"],
        ["bands", "bodyweight"],
    ]
    injury_sets = [[], ["left_knee_pain"], ["right_shoulder_irritation"], ["lower_back_sensitivity"]]

    profiles: list[dict] = []
    idx = 1
    for level in levels:
        for goal in goals:
            for days, day_names, split in schedules:
                focus = focuses[(idx - 1) % len(focuses)]
                equipment = equipment_options[(idx - 1) % len(equipment_options)]
                injuries = injury_sets[(idx - 1) % len(injury_sets)]
                profile = {
                    "user_id": f"u_{idx:04d}",
                    "profile": {
                        "level": level,
                        "training_age_years": 1 if level == "beginner" else (3 if level == "intermediate" else 6),
                        "goal": goal,
                        "primary_focus": focus,
                        "constraints": {
                            "injuries": injuries,
                            "equipment": equipment,
                            "schedule": f"{days} days per week ({', '.join(day_names)})",
                        },
                        "preferences": {
                            "disliked_exercises": ["barbell_back_squat"] if "knee" in " ".join(injuries) else [],
                            "preferred_rep_ranges": ["6-8", "8-12"] if level != "advanced" else ["4-6", "6-10"],
                        },
                    },
                    "program": {
                        "split": split,
                        "days_per_week": days,
                        "volume_summary": {
                            "chest_sets_per_week": 12 if goal in {"hypertrophy", "muscle_gain"} else 8,
                            "back_sets_per_week": 14 if goal in {"hypertrophy", "strength", "muscle_gain"} else 10,
                            "legs_sets_per_week": 12 if goal != "general_fitness" else 8,
                        },
                        "microcycle": [
                            {
                                "day_name": day_names[0],
                                "targets": focus,
                                "exercises": [
                                    {
                                        "exercise_id": "exercise_bench_press",
                                        "sets": 3,
                                        "reps": "6-10",
                                        "rir": 2,
                                    }
                                ],
                            }
                        ],
                    },
                    "state": {
                        "perceived_fatigue": "medium",
                        "adherence_last_7_days": 0.75,
                        "recent_issues": ["missed_one_session"] if idx % 3 == 0 else [],
                    },
                }
                profiles.append(profile)
                idx += 1

    return profiles


def _format_user_context(profile: dict) -> str:
    p = profile["profile"]
    program = profile["program"]
    constraints = p["constraints"]
    return (
        f"Level: {p['level']}; Goal: {p['goal']}; Split: {program['split']}; "
        f"Days/week: {program['days_per_week']}; Injuries: {', '.join(constraints.get('injuries') or ['none'])}; "
        f"Equipment: {', '.join(constraints.get('equipment') or ['unspecified'])}; "
        f"Preferred reps: {', '.join(p['preferences'].get('preferred_rep_ranges') or ['8-12'])}."
    )


def build_instruction_dictionary(domain_items: list[dict], user_profiles: list[dict]) -> list[dict]:
    instructions: list[dict] = []

    faq_items = [d for d in domain_items if d.get("type") == "faq_answer"]
    for idx, item in enumerate(faq_items, start=1):
        instructions.append(
            {
                "id": f"instr_faq_{idx:06d}",
                "instruction": item["title"],
                "response": item["body"],
                "domain_context_ids": [item["id"]],
                "user_context": None,
                "source": item["source"],
                "intent": "faq_answer",
            }
        )

    profile_templates = [
        (
            "Explain why my plan trains each muscle around twice per week.",
            "Twice-weekly exposure usually balances quality volume and recovery better than crushing each muscle once per week. "
            "For your profile, that frequency improves skill practice and lets you distribute hard sets without excessive single-session fatigue. "
            "Keep effort high, leave 1 to 3 reps in reserve on most work, and adjust weekly sets based on performance and recovery.",
            "program_rationale",
        ),
        (
            "I missed two workouts this week. How should I adjust next week?",
            "Do not try to cram missed sessions into one day. Resume your normal split, prioritize compound lifts, and trim accessory volume by about 20 percent for one week so fatigue does not spike. "
            "If adherence keeps slipping, reduce weekly days temporarily and rebuild consistency before adding volume back.",
            "adherence_recovery",
        ),
        (
            "My knee is irritated. Give me safer lower-body substitutions.",
            "Use pain-tolerable patterns such as split squats, controlled leg press ranges, and hip hinges that do not aggravate symptoms. "
            "Keep tempo controlled, avoid painful depth, and maintain weekly lower-body volume with tolerated variations rather than stopping training entirely. "
            "If pain escalates or changes normal movement quality, pause the aggravating lift and seek clinical evaluation.",
            "injury_modification",
        ),
    ]

    start_idx = len(instructions) + 1
    for p_idx, profile in enumerate(user_profiles, start=1):
        user_context = _format_user_context(profile)
        for t_idx, (question, answer, intent) in enumerate(profile_templates, start=1):
            instructions.append(
                {
                    "id": f"instr_user_{start_idx + p_idx * 10 + t_idx:06d}",
                    "instruction": question,
                    "response": answer,
                    "domain_context_ids": [],
                    "user_context": user_context,
                    "source": "synthetic_user_template",
                    "intent": intent,
                    "user_id": profile["user_id"],
                }
            )

    return instructions


def build_finetune_chat_dataset(instructions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in instructions:
        user_text = item["instruction"].strip()
        if item.get("user_context"):
            user_text = f"User context: {item['user_context']}\n\nQuestion: {user_text}"
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": item["response"].strip()},
                ]
            }
        )
    return rows


def build_full_dictionary(output_dir: Path | None = None) -> DictionaryPaths:
    paths = ProjectPaths()
    root = paths.root

    dictionaries_dir = output_dir or (paths.artifacts_dir / "dictionaries")
    dictionaries_dir.mkdir(parents=True, exist_ok=True)

    out = DictionaryPaths(
        domain_jsonl=dictionaries_dir / "domain_dictionary.jsonl",
        user_schema_json=dictionaries_dir / "user_dictionary_schema.json",
        user_profiles_jsonl=dictionaries_dir / "user_dictionary_profiles.jsonl",
        instruction_jsonl=dictionaries_dir / "instruction_dictionary.jsonl",
        finetune_jsonl=paths.datasets_dir / "fitness_training_data_dictionary_full.jsonl",
        summary_json=dictionaries_dir / "dictionary_summary.json",
    )

    domain_items = build_domain_dictionary(root)
    user_schema = build_user_schema()
    user_profiles = build_user_profiles()
    instruction_items = build_instruction_dictionary(domain_items, user_profiles)
    finetune_rows = build_finetune_chat_dataset(instruction_items)

    _write_jsonl(out.domain_jsonl, domain_items)
    out.user_schema_json.write_text(json.dumps(user_schema, indent=2), encoding="utf-8")
    _write_jsonl(out.user_profiles_jsonl, user_profiles)
    _write_jsonl(out.instruction_jsonl, instruction_items)
    _write_jsonl(out.finetune_jsonl, finetune_rows)

    summary = {
        "domain_items": len(domain_items),
        "user_profiles": len(user_profiles),
        "instruction_items": len(instruction_items),
        "finetune_samples": len(finetune_rows),
        "outputs": {
            "domain_dictionary": str(out.domain_jsonl),
            "user_schema": str(out.user_schema_json),
            "user_profiles": str(out.user_profiles_jsonl),
            "instruction_dictionary": str(out.instruction_jsonl),
            "finetune_dataset": str(out.finetune_jsonl),
        },
    }
    out.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return out
