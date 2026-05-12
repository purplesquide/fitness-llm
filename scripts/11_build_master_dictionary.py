"""
Build the comprehensive master dictionary from ALL available data sources.

Reads:
  - exercisedb/exercisedb_v1_sample/exercises.json
  - fitness_exercises/exercises.csv
  - megaGymDataset/megaGymDataset.csv
  - wger_exercises.json
  - longhaul_fitness/strength.json + cardio.json + flexibility.json
  - ultimate_gym/Workout.csv
  - fitness_metrics/exercise_dataset.csv (calorie/biometric stats)
  - data/knowledge_seed.json
  - artifacts/datasets/nutrition_textbook_full_qa.jsonl
  - artifacts/dictionaries/domain_dictionary.jsonl  (carry-forward)

Writes to artifacts/dictionaries/:
  - master_exercise_dictionary.jsonl   — one entry per unique exercise
  - master_domain_dictionary.jsonl     — domain concepts (science, nutrition, safety, programming)
  - master_workout_plans.jsonl         — pre-built workout plan templates
  - master_user_schema.json            — enhanced user profile schema
  - master_summary.json                — stats
"""
from __future__ import annotations

import csv
import json
import re
import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = ROOT / "artifacts" / "dictionaries"
DICT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(v: Any) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v).strip().lower())


def _title(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().title()


def _list_norm(v: Any) -> list[str]:
    if isinstance(v, list):
        return [_norm(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        return [x.strip().lower() for x in re.split(r"[,;/|]", v) if x.strip()]
    return []


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Muscle / equipment normalisation maps
# ---------------------------------------------------------------------------

MUSCLE_ALIASES: dict[str, str] = {
    "pectorals": "chest", "pectoral": "chest", "chest": "chest",
    "lats": "lats", "lat": "lats", "latissimus dorsi": "lats",
    "quads": "quadriceps", "quad": "quadriceps", "quadriceps": "quadriceps",
    "hamstrings": "hamstrings", "hamstring": "hamstrings",
    "glutes": "glutes", "glute": "glutes", "gluteus": "glutes",
    "calves": "calves", "calf": "calves",
    "abs": "abs", "abdominals": "abs", "core": "core",
    "obliques": "obliques", "oblique": "obliques",
    "shoulders": "shoulders", "shoulder": "shoulders", "deltoids": "shoulders",
    "deltoid": "shoulders", "delt": "shoulders",
    "biceps": "biceps", "bicep": "biceps",
    "triceps": "triceps", "tricep": "triceps",
    "forearms": "forearms", "forearm": "forearms",
    "traps": "traps", "trap": "traps", "trapezius": "traps",
    "rhomboids": "rhomboids", "rhomboid": "rhomboids",
    "lower back": "lower back", "erectors": "lower back", "erector spinae": "lower back",
    "upper back": "upper back",
    "hip flexors": "hip flexors", "hip flexor": "hip flexors",
    "adductors": "adductors", "adductor": "adductors",
    "abductors": "abductors",
    "neck": "neck",
    "wrist": "wrist",
    "spine": "spine",
    "full body": "full body",
}

EQUIPMENT_ALIASES: dict[str, str] = {
    "body weight": "bodyweight", "bodyweight": "bodyweight", "none": "bodyweight",
    "barbell": "barbell", "bar": "barbell",
    "dumbbell": "dumbbell", "dumbbells": "dumbbell", "db": "dumbbell",
    "cable": "cable", "cables": "cable",
    "bands": "resistance band", "band": "resistance band", "resistance band": "resistance band",
    "machine": "machine", "leverage machine": "machine", "sled machine": "machine",
    "kettlebell": "kettlebell", "kb": "kettlebell",
    "ez-bar": "ez-bar", "ez bar": "ez-bar", "sz-bar": "ez-bar",
    "pull-up bar": "pull-up bar", "pull up bar": "pull-up bar",
    "bench": "bench",
    "roller": "foam roller", "foam roller": "foam roller",
    "assisted": "assisted machine", "leverage": "machine",
    "smith machine": "smith machine",
    "suspension": "trx", "trx": "trx",
    "rope": "rope",
    "sled": "sled machine",
    "stability ball": "stability ball", "exercise ball": "stability ball",
}


def _norm_muscles(raw: list[str] | str) -> list[str]:
    items = _list_norm(raw)
    result: list[str] = []
    for m in items:
        mapped = MUSCLE_ALIASES.get(m, m)
        if mapped and mapped not in result:
            result.append(mapped)
    return result


def _norm_equipment(raw: str | list) -> str:
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    key = _norm(raw)
    return EQUIPMENT_ALIASES.get(key, key or "bodyweight")


# ---------------------------------------------------------------------------
# MOVEMENT PATTERN mapping
# ---------------------------------------------------------------------------

def _infer_movement_pattern(name: str, primary_muscles: list[str]) -> str:
    n = name.lower()
    muscles = " ".join(primary_muscles)
    if any(x in n for x in ["squat", "leg press", "lunge", "step up", "goblet"]):
        return "squat_pattern"
    if any(x in n for x in ["deadlift", "romanian", "rdl", "hip thrust", "good morning", "hyperextension"]):
        return "hinge_pattern"
    if any(x in n for x in ["row", "pulldown", "pull-up", "pullup", "chin"]):
        return "pull_horizontal" if "row" in n else "pull_vertical"
    if any(x in n for x in ["press", "push up", "pushup", "dip", "fly", "flye", "pec"]):
        if any(x in n for x in ["overhead", "shoulder", "military", "ohp"]):
            return "push_vertical"
        return "push_horizontal"
    if any(x in n for x in ["curl", "bicep"]):
        return "elbow_flexion"
    if any(x in n for x in ["tricep", "skull", "pushdown", "extension"]):
        return "elbow_extension"
    if any(x in n for x in ["lateral raise", "front raise", "face pull", "upright row"]):
        return "shoulder_isolation"
    if any(x in n for x in ["calf", "calves"]):
        return "ankle_extension"
    if any(x in n for x in ["crunch", "sit-up", "situp", "ab", "plank", "twist", "russian"]):
        return "core_isolation"
    if any(x in n for x in ["run", "sprint", "jog", "cardio", "bike", "row", "swim", "elliptical", "jump"]):
        return "cardio"
    if any(x in n for x in ["stretch", "mobility", "foam", "yoga", "flexibility"]):
        return "mobility_flexibility"
    if "chest" in muscles:
        return "push_horizontal"
    if "lats" in muscles:
        return "pull_vertical"
    return "general"


# ---------------------------------------------------------------------------
# SOURCE 1: ExerciseDB JSON
# ---------------------------------------------------------------------------

def load_exercisedb_json() -> list[dict]:
    path = ROOT / "exercisedb" / "exercisedb_v1_sample" / "exercises.json"
    if not path.exists():
        return []
    raw_list = _load_json(path)
    results = []
    for ex in raw_list:
        name = _title(ex.get("name", ""))
        if not name:
            continue
        primary = _norm_muscles(ex.get("targetMuscles", []))
        secondary = _norm_muscles(ex.get("secondaryMuscles", []))
        equipment = _norm_equipment(ex.get("equipments", []))
        instructions = ex.get("instructions", [])
        if isinstance(instructions, list):
            instr_text = " ".join(str(s) for s in instructions)
        else:
            instr_text = str(instructions)

        results.append({
            "id": f"edb_{_hash(name)}",
            "source": "exercisedb",
            "name": name,
            "primary_muscles": primary,
            "secondary_muscles": secondary,
            "equipment": equipment,
            "body_part": _norm(ex.get("bodyParts", [""])[0] if ex.get("bodyParts") else ""),
            "movement_pattern": _infer_movement_pattern(name, primary),
            "difficulty": "intermediate",
            "instructions": instr_text.strip(),
            "cues": [],
            "contraindications": [],
            "tags": [],
        })
    return results


# ---------------------------------------------------------------------------
# SOURCE 2: fitness_exercises/exercises.csv (ExerciseDB v2 CSV)
# ---------------------------------------------------------------------------

def load_fitness_exercises_csv() -> list[dict]:
    path = ROOT / "fitness_exercises" / "exercises.csv"
    if not path.exists():
        return []
    results = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _title(row.get("name", ""))
            if not name:
                continue
            # Collect instruction columns
            instr_cols = sorted(k for k in row if k.startswith("instructions/"))
            instructions = " ".join(row[k].strip() for k in instr_cols if row[k].strip())
            sec_cols = sorted(k for k in row if k.startswith("secondaryMuscles/"))
            secondary_raw = [row[k].strip() for k in sec_cols if row[k].strip()]
            primary = _norm_muscles(row.get("target", ""))
            secondary = _norm_muscles(secondary_raw)
            equipment = _norm_equipment(row.get("equipment", ""))
            results.append({
                "id": f"csv_{_hash(name)}",
                "source": "fitness_exercises_csv",
                "name": name,
                "primary_muscles": primary,
                "secondary_muscles": secondary,
                "equipment": equipment,
                "body_part": _norm(row.get("bodyPart", "")),
                "movement_pattern": _infer_movement_pattern(name, primary),
                "difficulty": "intermediate",
                "instructions": instructions.strip(),
                "cues": [],
                "contraindications": [],
                "tags": [],
            })
    return results


# ---------------------------------------------------------------------------
# SOURCE 3: megaGymDataset.csv (bodybuilding.com exercises)
# ---------------------------------------------------------------------------

def load_mega_gym_csv() -> list[dict]:
    path = ROOT / "megaGymDataset" / "megaGymDataset.csv"
    if not path.exists():
        return []
    results = []
    difficulty_map = {"beginner": "beginner", "intermediate": "intermediate", "expert": "advanced"}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _title(row.get("Title", ""))
            desc = row.get("Desc", "").strip()
            if not name:
                continue
            body_part = _norm(row.get("BodyPart", ""))
            equip = _norm_equipment(row.get("Equipment", ""))
            level = difficulty_map.get(_norm(row.get("Level", "")), "intermediate")
            ex_type = _norm(row.get("Type", ""))
            # derive primary muscle from BodyPart field
            primary = _norm_muscles(body_part)
            results.append({
                "id": f"mega_{_hash(name)}",
                "source": "megagym",
                "name": name,
                "primary_muscles": primary,
                "secondary_muscles": [],
                "equipment": equip,
                "body_part": body_part,
                "movement_pattern": _infer_movement_pattern(name, primary),
                "exercise_type": ex_type,
                "difficulty": level,
                "instructions": desc,
                "cues": [],
                "contraindications": [],
                "tags": [ex_type] if ex_type else [],
            })
    return results


# ---------------------------------------------------------------------------
# SOURCE 4: wger_exercises.json
# ---------------------------------------------------------------------------

def load_wger_exercises() -> list[dict]:
    path = ROOT / "wger_exercises.json"
    if not path.exists():
        return []
    raw_list = _load_json(path)
    results = []
    for ex in raw_list:
        # Find English translation
        en_trans = None
        for t in ex.get("translations", []):
            if t.get("language") == 2 or t.get("language_code") == "en":
                en_trans = t
                break
        if not en_trans:
            # fallback to first translation
            if ex.get("translations"):
                en_trans = ex["translations"][0]
        if not en_trans:
            continue
        name = _title(en_trans.get("name", ""))
        if not name:
            continue

        # Extract description and steps
        description = en_trans.get("description", "") or ""
        description = re.sub(r"<[^>]+>", " ", description).strip()
        steps_raw = en_trans.get("steps", [])
        if isinstance(steps_raw, list):
            steps_text = " ".join(s.get("description", "") if isinstance(s, dict) else str(s) for s in steps_raw)
        else:
            steps_text = str(steps_raw)
        instructions = (description + " " + steps_text).strip()

        muscles_raw = ex.get("muscles", [])
        primary = _norm_muscles([m.get("name_en", m.get("name", "")) for m in muscles_raw if isinstance(m, dict)])
        secondary_raw = ex.get("muscles_secondary", [])
        secondary = _norm_muscles([m.get("name_en", m.get("name", "")) for m in secondary_raw if isinstance(m, dict)])
        equipment_raw = ex.get("equipment", [])
        equipment = _norm_equipment([e.get("name", "") for e in equipment_raw if isinstance(e, dict)])

        category_name = ""
        if isinstance(ex.get("category"), dict):
            category_name = ex["category"].get("name", "")

        results.append({
            "id": f"wger_{_hash(name)}",
            "source": "wger",
            "name": name,
            "primary_muscles": primary,
            "secondary_muscles": secondary,
            "equipment": equipment,
            "body_part": _norm(category_name),
            "movement_pattern": _infer_movement_pattern(name, primary),
            "difficulty": "intermediate",
            "instructions": instructions[:1000].strip(),
            "cues": [],
            "contraindications": [],
            "tags": [_norm(category_name)] if category_name else [],
        })
    return results


# ---------------------------------------------------------------------------
# SOURCE 5: longhaul_fitness (strength / cardio / flexibility)
# ---------------------------------------------------------------------------

def load_longhaul_exercises() -> list[dict]:
    results = []
    for fname, ex_type in [("strength.json", "strength"), ("cardio.json", "cardio"), ("flexibility.json", "flexibility")]:
        path = ROOT / "longhaul_fitness" / fname
        if not path.exists():
            continue
        raw_list = _load_json(path)
        for ex in raw_list:
            name = _title(ex.get("name", ""))
            if not name:
                continue
            steps = ex.get("steps", []) or ex.get("instructions", [])
            if isinstance(steps, list):
                instructions = " ".join(str(s) for s in steps)
            else:
                instructions = str(steps)
            notes = ex.get("notes", "")
            if notes:
                instructions = (instructions + " " + notes).strip()

            primary = _norm_muscles(ex.get("primaryMuscles", []))
            secondary = _norm_muscles(ex.get("secondaryMuscles", []))
            results.append({
                "id": f"lh_{_hash(name)}",
                "source": f"longhaul_{ex_type}",
                "name": name,
                "primary_muscles": primary,
                "secondary_muscles": secondary,
                "equipment": "varies",
                "body_part": primary[0] if primary else ex_type,
                "movement_pattern": _infer_movement_pattern(name, primary),
                "exercise_type": ex_type,
                "difficulty": "intermediate",
                "instructions": instructions.strip(),
                "cues": [],
                "contraindications": [],
                "tags": [ex_type],
            })
    return results


# ---------------------------------------------------------------------------
# SOURCE 6: ultimate_gym/Workout.csv (workout plan templates)
# ---------------------------------------------------------------------------

def load_ultimate_gym() -> list[dict]:
    """Returns workout plan template records."""
    path = ROOT / "ultimate_gym" / "Workout.csv"
    if not path.exists():
        return []
    templates: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            body_part = row.get("Body Part", "").strip()
            muscle_type = row.get("Type of Muscle", "").strip()
            workout = row.get("Workout", "").strip()
            sets_str = row.get("Sets", "").strip()
            reps_str = row.get("Reps per Set", "").strip()
            if not workout:
                continue
            key = body_part.lower()
            if key not in templates:
                templates[key] = {
                    "id": f"ugym_{_hash(key)}",
                    "source": "ultimate_gym",
                    "body_part": body_part,
                    "exercises": [],
                }
            templates[key]["exercises"].append({
                "name": workout,
                "muscle_type": muscle_type,
                "sets": sets_str,
                "reps": reps_str,
            })
    return list(templates.values())


# ---------------------------------------------------------------------------
# SOURCE 7: fitness_metrics/exercise_dataset.csv (calorie data)
# ---------------------------------------------------------------------------

def load_fitness_metrics() -> dict[str, dict]:
    """Returns per-exercise calorie burn statistics."""
    path = ROOT / "fitness_metrics" / "exercise_dataset.csv"
    if not path.exists():
        return {}
    stats: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Exercise", "").strip()
            try:
                cal = float(row.get("Calories Burn", 0))
            except (ValueError, TypeError):
                cal = 0.0
            try:
                dur = float(row.get("Duration", 0))
            except (ValueError, TypeError):
                dur = 0.0
            if name and cal > 0:
                key = _norm(name)
                if key not in stats:
                    stats[key] = []
                stats[key].append(cal / max(dur, 1) * 30)  # normalize to 30-min burn
    return {k: {"avg_cal_30min": round(sum(v) / len(v), 1)} for k, v in stats.items()}


# ---------------------------------------------------------------------------
# MERGE & DEDUPLICATE EXERCISES
# ---------------------------------------------------------------------------

def merge_exercises(sources: list[list[dict]]) -> list[dict]:
    """Merge all exercise lists, deduplicate by normalised name."""
    seen: dict[str, dict] = {}
    for source_list in sources:
        for ex in source_list:
            key = _norm(ex["name"])
            if key not in seen:
                seen[key] = ex
            else:
                existing = seen[key]
                # Merge fields from this record into the existing one
                for field in ("primary_muscles", "secondary_muscles", "tags"):
                    combined = list(dict.fromkeys(existing.get(field, []) + ex.get(field, [])))
                    existing[field] = combined
                # Prefer longer instructions
                if len(ex.get("instructions", "")) > len(existing.get("instructions", "")):
                    existing["instructions"] = ex["instructions"]
                # Prefer non-default difficulty
                if existing.get("difficulty") == "intermediate" and ex.get("difficulty") != "intermediate":
                    existing["difficulty"] = ex["difficulty"]
    result = list(seen.values())
    # Re-index
    for i, ex in enumerate(result, start=1):
        ex["index"] = i
    return result


# ---------------------------------------------------------------------------
# DOMAIN DICTIONARY (knowledge, nutrition, programming, safety)
# ---------------------------------------------------------------------------

def build_domain_dictionary() -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()

    def add(item: dict) -> None:
        key = _hash(item.get("body", "")[:200])
        if key not in seen and item.get("body", "").strip():
            seen.add(key)
            items.append(item)

    # --- knowledge_seed.json ---
    seed_path = ROOT / "data" / "knowledge_seed.json"
    if seed_path.exists():
        for idx, row in enumerate(_load_json(seed_path), 1):
            source = str(row.get("source", "knowledge"))
            body = str(row.get("text", "")).strip()
            if not body:
                continue
            concept_type = (
                "nutrition_rule" if "nutrition" in source else
                "safety_guideline" if any(x in source for x in ("injury", "recover", "safety")) else
                "programming_principle" if "program" in source else
                "sports_science"
            )
            add({
                "id": f"seed_{idx:05d}",
                "type": concept_type,
                "source": source,
                "title": body[:80].rstrip(),
                "body": body,
                "tags": [source.split("_")[0]],
            })

    # --- carry-forward existing domain_dictionary.jsonl ---
    existing_path = ROOT / "artifacts" / "dictionaries" / "domain_dictionary.jsonl"
    if existing_path.exists():
        for row in _read_jsonl(existing_path):
            body = row.get("body", row.get("text", "")).strip()
            if not body:
                continue
            add({
                "id": row.get("id", f"carry_{_hash(body)}"),
                "type": row.get("type", "concept"),
                "source": row.get("source", "existing"),
                "title": row.get("title", body[:80]),
                "body": body,
                "tags": row.get("tags", []),
            })

    # --- nutrition textbook Q&A ---
    nutr_path = ROOT / "artifacts" / "datasets" / "nutrition_textbook_full_qa.jsonl"
    if nutr_path.exists():
        for idx, row in enumerate(_read_jsonl(nutr_path), 1):
            msgs = row.get("messages", [])
            if len(msgs) >= 3:
                q = msgs[1].get("content", "").strip()
                a = msgs[2].get("content", "").strip()
                if q and a and len(a) > 30:
                    add({
                        "id": f"nutr_qa_{idx:05d}",
                        "type": "nutrition_rule",
                        "source": "nutrition_textbook",
                        "title": q[:80],
                        "body": f"Q: {q}\nA: {a}",
                        "tags": ["nutrition"],
                    })

    # --- Hard-coded programming templates ---
    programming_facts = [
        ("prog_001", "programming_principle", "Upper/Lower Split (4 days)",
         "A 4-day upper/lower split trains each muscle group twice per week. Upper days focus on chest, back, shoulders, and arms. Lower days focus on quads, hamstrings, glutes, and calves. Ideal for intermediate lifters targeting hypertrophy.",
         ["programming", "split", "intermediate"]),
        ("prog_002", "programming_principle", "Push/Pull/Legs Split (6 days)",
         "Push days: chest, shoulders, triceps. Pull days: back, biceps, rear delts. Leg days: quads, hamstrings, glutes, calves. 6-day PPL provides high frequency and volume for advanced lifters.",
         ["programming", "split", "advanced"]),
        ("prog_003", "programming_principle", "Full Body (2-3 days)",
         "Full-body training 2-3 times per week is optimal for beginners and deload phases. Each session hits all major muscle groups with compound movements. Allows maximum recovery between sessions.",
         ["programming", "split", "beginner"]),
        ("prog_004", "programming_principle", "Progressive Overload",
         "Progressive overload means increasing the training stimulus over time: more load, more reps, more sets, or better technique at the same load. It is the primary driver of long-term adaptation.",
         ["programming", "progressive_overload"]),
        ("prog_005", "programming_principle", "Deload Week",
         "A deload week reduces training volume by 40–60% while maintaining movement patterns. Used after 4–8 weeks of progressive training to allow full recovery and supercompensation.",
         ["programming", "recovery", "deload"]),
        ("prog_006", "programming_principle", "Rep Range Guidelines",
         "Strength: 1–5 reps at 85–100% 1RM. Hypertrophy: 6–15 reps at 65–85% 1RM. Endurance: 15–25+ reps at 40–65% 1RM. All rep ranges can build muscle when taken close to failure.",
         ["programming", "rep_ranges"]),
        ("prog_007", "programming_principle", "Volume Landmarks",
         "Minimum effective volume (MEV) is the least sets needed to maintain or grow. Maximum recoverable volume (MRV) is the most sets you can recover from. Start near MEV and add sets progressively.",
         ["programming", "volume"]),
        ("prog_008", "safety_guideline", "Warm-Up Protocol",
         "Always perform a 5–10 minute general warm-up (light cardio) followed by exercise-specific warm-up sets (50%, 70%, 85% of working weight). Never skip warm-ups for heavy compound lifts.",
         ["safety", "warm_up"]),
        ("prog_009", "safety_guideline", "Injury Red Flags",
         "Sharp, acute pain during a lift, joint swelling, numbness, tingling, or pain that worsens with rest are red flags requiring medical evaluation. Do not train through sharp joint pain.",
         ["safety", "injury"]),
        ("prog_010", "safety_guideline", "RPE and RIR",
         "Rate of Perceived Exertion (RPE) 1–10 scale: RPE 8 = 2 reps left in tank (2 RIR). Training at RPE 7–9 (1–3 RIR) is optimal for hypertrophy. Beginners should stay at RPE 6–7.",
         ["programming", "rpe", "intensity"]),
        ("nutr_001", "nutrition_rule", "Protein for Muscle Gain",
         "Target 1.6–2.2 g of protein per kg of bodyweight per day for muscle gain. Higher end (2.0–2.4 g/kg) may benefit during a calorie deficit. Distribute across 3–5 meals for optimal MPS.",
         ["nutrition", "protein", "muscle_gain"]),
        ("nutr_002", "nutrition_rule", "Calorie Surplus for Bulking",
         "A lean bulk requires 200–400 kcal above maintenance. Faster weight gain does not accelerate muscle growth and increases fat accumulation. Aim for 0.25–0.5% bodyweight gain per week.",
         ["nutrition", "calories", "bulking"]),
        ("nutr_003", "nutrition_rule", "Calorie Deficit for Fat Loss",
         "A 300–500 kcal daily deficit creates sustainable fat loss of 0.5–1% bodyweight per week while preserving muscle mass. Protein must remain high (2.0–2.4 g/kg) during a cut.",
         ["nutrition", "calories", "fat_loss"]),
        ("nutr_004", "nutrition_rule", "Pre-Workout Nutrition",
         "Consume 0.3–0.5 g/kg carbohydrates and 20–40 g protein 1–2 hours before training. Avoid high-fat meals immediately before training as fat slows digestion.",
         ["nutrition", "pre_workout", "timing"]),
        ("nutr_005", "nutrition_rule", "Post-Workout Nutrition",
         "Consume 20–40 g protein and 0.5–0.8 g/kg carbohydrates within 2 hours post-workout. The anabolic window is longer than once believed — total daily intake matters most.",
         ["nutrition", "post_workout", "timing"]),
        ("nutr_006", "nutrition_rule", "Hydration Guidelines",
         "Drink 0.033 L per kg bodyweight daily at minimum. During exercise, aim for 400–800 mL per hour. Urine colour should be pale yellow. Dehydration of 2%+ body weight impairs performance.",
         ["nutrition", "hydration"]),
        ("nutr_007", "nutrition_rule", "Creatine Supplementation",
         "Creatine monohydrate is the most evidence-based supplement. 3–5 g/day improves strength, power, and lean mass. No loading phase necessary. Safe for long-term use in healthy individuals.",
         ["nutrition", "supplements", "creatine"]),
        ("sci_001", "sports_science", "Hypertrophy Mechanisms",
         "Muscle hypertrophy is driven by mechanical tension, metabolic stress, and muscle damage. Mechanical tension from progressive overload is the primary stimulus. Range of motion affects stretch-mediated hypertrophy.",
         ["science", "hypertrophy"]),
        ("sci_002", "sports_science", "Muscle Protein Synthesis",
         "Muscle protein synthesis (MPS) is elevated for 24–48 hours post-training. MPS peaks at 20–40 g protein per meal. Leucine threshold (~2–3 g leucine) is required to maximally stimulate MPS.",
         ["science", "protein_synthesis", "nutrition"]),
        ("sci_003", "sports_science", "EPOC and Fat Burning",
         "Excess post-exercise oxygen consumption (EPOC) is elevated after high-intensity training. Total calorie expenditure during and after exercise determines fat loss — not fat burning rate during exercise alone.",
         ["science", "cardio", "fat_loss"]),
        ("sci_004", "sports_science", "Compound vs Isolation Exercises",
         "Compound exercises (squat, deadlift, bench press, rows) recruit multiple muscle groups and should form the foundation of any program. Isolation exercises are used to add volume to lagging muscles.",
         ["science", "exercise_selection"]),
        ("sci_005", "sports_science", "Sleep and Recovery",
         "7–9 hours of sleep per night is required for optimal muscle recovery and hormonal balance. Sleep deprivation elevates cortisol, reduces testosterone, and impairs protein synthesis.",
         ["science", "recovery", "sleep"]),
    ]

    for entry in programming_facts:
        add({
            "id": entry[0],
            "type": entry[1],
            "source": "curated",
            "title": entry[2],
            "body": entry[3],
            "tags": entry[4],
        })

    return items


# ---------------------------------------------------------------------------
# WORKOUT PLAN TEMPLATES
# ---------------------------------------------------------------------------

def build_workout_plan_templates() -> list[dict]:
    """Pre-built program templates by goal/level/days."""
    plans = []

    def plan(pid, goal, level, days, split_name, description, microcycle):
        return {
            "id": pid,
            "goal": goal,
            "level": level,
            "days_per_week": days,
            "split": split_name,
            "description": description,
            "microcycle": microcycle,
        }

    plans.append(plan(
        "plan_001", "muscle_gain", "beginner", 3, "Full Body 3x",
        "3-day full body. Each session hits chest, back, legs, shoulders with compound lifts. Ideal for beginners building foundation.",
        [
            {"day": "Day 1", "focus": "Full Body A", "exercises": [
                {"name": "Squat", "sets": "3x8-10", "notes": "Barbell or goblet squat"},
                {"name": "Bench Press", "sets": "3x8-10"},
                {"name": "Bent-over Row", "sets": "3x8-10"},
                {"name": "Overhead Press", "sets": "3x8-10"},
                {"name": "Romanian Deadlift", "sets": "3x10-12"},
            ]},
            {"day": "Day 2", "focus": "Full Body B", "exercises": [
                {"name": "Deadlift", "sets": "3x5"},
                {"name": "Incline Dumbbell Press", "sets": "3x10-12"},
                {"name": "Pull-Up / Lat Pulldown", "sets": "3x8-10"},
                {"name": "Dumbbell Lunges", "sets": "3x10/leg"},
                {"name": "Dumbbell Lateral Raise", "sets": "3x12-15"},
            ]},
            {"day": "Day 3", "focus": "Full Body C", "exercises": [
                {"name": "Squat", "sets": "3x10"},
                {"name": "Cable Row", "sets": "3x10-12"},
                {"name": "Dumbbell Shoulder Press", "sets": "3x10"},
                {"name": "Hip Thrust", "sets": "3x10-12"},
                {"name": "Tricep Pushdown", "sets": "3x12"},
                {"name": "Bicep Curl", "sets": "3x12"},
            ]},
        ]
    ))

    plans.append(plan(
        "plan_002", "muscle_gain", "intermediate", 4, "Upper/Lower 4x",
        "4-day upper/lower split for intermediate hypertrophy. Twice-weekly muscle frequency with progressive overload.",
        [
            {"day": "Day 1", "focus": "Upper A (Heavy)", "exercises": [
                {"name": "Barbell Bench Press", "sets": "4x5-8"},
                {"name": "Barbell Row", "sets": "4x5-8"},
                {"name": "Overhead Press", "sets": "3x6-8"},
                {"name": "Pull-Up", "sets": "3x6-8"},
                {"name": "Incline Dumbbell Press", "sets": "3x10"},
                {"name": "Face Pull", "sets": "3x15"},
            ]},
            {"day": "Day 2", "focus": "Lower A (Heavy)", "exercises": [
                {"name": "Barbell Squat", "sets": "4x5-8"},
                {"name": "Romanian Deadlift", "sets": "3x8-10"},
                {"name": "Leg Press", "sets": "3x10-12"},
                {"name": "Leg Curl", "sets": "3x10-12"},
                {"name": "Calf Raise", "sets": "4x12-15"},
            ]},
            {"day": "Day 3", "focus": "Upper B (Hypertrophy)", "exercises": [
                {"name": "Incline Barbell Press", "sets": "4x8-12"},
                {"name": "Cable Row", "sets": "4x10-12"},
                {"name": "Dumbbell Lateral Raise", "sets": "4x12-15"},
                {"name": "Lat Pulldown", "sets": "3x10-12"},
                {"name": "Tricep Dip", "sets": "3x10-12"},
                {"name": "Barbell Curl", "sets": "3x10-12"},
            ]},
            {"day": "Day 4", "focus": "Lower B (Hypertrophy)", "exercises": [
                {"name": "Hack Squat / Leg Press", "sets": "4x10-12"},
                {"name": "Deadlift", "sets": "3x5"},
                {"name": "Walking Lunge", "sets": "3x12/leg"},
                {"name": "Leg Extension", "sets": "3x12-15"},
                {"name": "Hip Thrust", "sets": "3x12"},
                {"name": "Seated Calf Raise", "sets": "4x15"},
            ]},
        ]
    ))

    plans.append(plan(
        "plan_003", "strength", "intermediate", 4, "4-Day Powerbuilding",
        "Strength-focused 4-day program with compound prioritisation. Progressive loading on main lifts.",
        [
            {"day": "Day 1", "focus": "Squat", "exercises": [
                {"name": "Barbell Back Squat", "sets": "5x5", "notes": "Main lift — add 2.5kg when all 5x5 done"},
                {"name": "Romanian Deadlift", "sets": "3x8"},
                {"name": "Leg Press", "sets": "3x10"},
                {"name": "Leg Curl", "sets": "3x10"},
            ]},
            {"day": "Day 2", "focus": "Bench Press", "exercises": [
                {"name": "Barbell Bench Press", "sets": "5x5"},
                {"name": "Incline Dumbbell Press", "sets": "3x8"},
                {"name": "Cable Row", "sets": "3x10"},
                {"name": "Tricep Pushdown", "sets": "3x12"},
            ]},
            {"day": "Day 3", "focus": "Deadlift", "exercises": [
                {"name": "Conventional Deadlift", "sets": "1x5, 4x4"},
                {"name": "Front Squat / Pause Squat", "sets": "3x5"},
                {"name": "Good Morning", "sets": "3x8"},
                {"name": "Back Extension", "sets": "3x10"},
            ]},
            {"day": "Day 4", "focus": "Overhead Press", "exercises": [
                {"name": "Barbell Overhead Press", "sets": "5x5"},
                {"name": "Pull-Up", "sets": "4x6-8"},
                {"name": "Dumbbell Row", "sets": "3x10"},
                {"name": "Lateral Raise", "sets": "3x12"},
                {"name": "Bicep Curl", "sets": "3x10"},
            ]},
        ]
    ))

    plans.append(plan(
        "plan_004", "weight_loss", "beginner", 3, "Fat Loss Circuit 3x",
        "3-day full body circuit for fat loss. Higher reps, shorter rest, maintains muscle mass.",
        [
            {"day": "Each Session (A/B/C rotate)", "focus": "Full Body Circuit", "exercises": [
                {"name": "Goblet Squat", "sets": "3x15"},
                {"name": "Push-Up", "sets": "3x12-15"},
                {"name": "Dumbbell Row", "sets": "3x12"},
                {"name": "Dumbbell Lunge", "sets": "3x12/leg"},
                {"name": "Shoulder Press", "sets": "3x12"},
                {"name": "Plank", "sets": "3x30-45s"},
                {"name": "Mountain Climbers", "sets": "3x20"},
            ]},
        ]
    ))

    plans.append(plan(
        "plan_005", "endurance", "intermediate", 5, "5-Day Endurance + Strength",
        "3 strength + 2 cardio sessions per week. Improves cardiovascular fitness while maintaining muscle mass.",
        [
            {"day": "Day 1", "focus": "Upper Body Strength", "exercises": [
                {"name": "Pull-Up", "sets": "4x8"},
                {"name": "Bench Press", "sets": "4x10"},
                {"name": "Row", "sets": "4x10"},
                {"name": "Overhead Press", "sets": "3x12"},
            ]},
            {"day": "Day 2", "focus": "Zone 2 Cardio", "exercises": [
                {"name": "Run / Bike / Row", "sets": "30-45 min at conversational pace"},
            ]},
            {"day": "Day 3", "focus": "Lower Body Strength", "exercises": [
                {"name": "Squat", "sets": "4x10"},
                {"name": "Romanian Deadlift", "sets": "3x10"},
                {"name": "Calf Raise", "sets": "3x15"},
                {"name": "Plank", "sets": "3x45s"},
            ]},
            {"day": "Day 4", "focus": "HIIT Cardio", "exercises": [
                {"name": "Interval Training", "sets": "20 min: 30s hard / 90s easy x 8"},
            ]},
            {"day": "Day 5", "focus": "Full Body", "exercises": [
                {"name": "Deadlift", "sets": "3x8"},
                {"name": "Push-Up", "sets": "3x15"},
                {"name": "Lat Pulldown", "sets": "3x12"},
                {"name": "Goblet Squat", "sets": "3x15"},
            ]},
        ]
    ))

    return plans


# ---------------------------------------------------------------------------
# ENHANCED USER SCHEMA
# ---------------------------------------------------------------------------

def build_enhanced_user_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "FitnessLLM_MasterUserSchema",
        "version": "2.0",
        "type": "object",
        "required": ["user_id", "profile", "program", "state"],
        "properties": {
            "user_id": {"type": "string", "description": "Unique user identifier"},
            "profile": {
                "type": "object",
                "required": ["level", "goal", "constraints", "preferences"],
                "properties": {
                    "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                    "training_age_years": {"type": "number", "minimum": 0, "maximum": 50},
                    "goal": {"type": "string", "enum": ["muscle_gain", "weight_loss", "strength", "endurance", "general_fitness", "recomposition", "sport_performance"]},
                    "primary_focus": {"type": "array", "items": {"type": "string"}},
                    "demographics": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer", "minimum": 13, "maximum": 100},
                            "sex": {"type": "string", "enum": ["male", "female", "other", "prefer_not_to_say"]},
                            "height_cm": {"type": "number"},
                            "weight_kg": {"type": "number"},
                            "bmi": {"type": "number"},
                        }
                    },
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "injuries": {"type": "array", "items": {"type": "string"}, "description": "User-reported injuries or limitations"},
                            "medical_conditions": {"type": "array", "items": {"type": "string"}},
                            "equipment": {"type": "array", "items": {"type": "string"}},
                            "schedule": {"type": "string", "description": "e.g. '4 days/week, weekdays only'"},
                            "time_per_session_min": {"type": "integer"},
                        }
                    },
                    "preferences": {
                        "type": "object",
                        "properties": {
                            "disliked_exercises": {"type": "array", "items": {"type": "string"}},
                            "liked_exercises": {"type": "array", "items": {"type": "string"}},
                            "preferred_training_style": {"type": "string", "enum": ["powerlifting", "bodybuilding", "functional", "calisthenics", "circuit", "any"]},
                            "preferred_rep_ranges": {"type": "array", "items": {"type": "string"}},
                            "diet_type": {"type": "string", "enum": ["standard", "vegetarian", "vegan", "keto", "paleo", "mediterranean", "other"]},
                        }
                    },
                }
            },
            "program": {
                "type": "object",
                "properties": {
                    "split": {"type": "string"},
                    "days_per_week": {"type": "integer", "minimum": 1, "maximum": 7},
                    "current_plan_id": {"type": "string"},
                    "volume_summary": {"type": "object", "description": "sets per muscle group per week"},
                    "microcycle": {"type": "array", "items": {"type": "object"}},
                    "program_start_date": {"type": "string", "format": "date"},
                    "current_week": {"type": "integer"},
                }
            },
            "state": {
                "type": "object",
                "properties": {
                    "last_workout_date": {"type": "string", "format": "date"},
                    "total_sessions_logged": {"type": "integer"},
                    "streak_days": {"type": "integer"},
                    "recent_workouts": {
                        "type": "array",
                        "maxItems": 10,
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string"},
                                "session_name": {"type": "string"},
                                "exercises": {"type": "array"},
                                "duration_min": {"type": "integer"},
                                "rpe": {"type": "number"},
                            }
                        }
                    },
                    "adherence_rate_percent": {"type": "number"},
                    "notes": {"type": "string"},
                }
            },
            "context": {
                "type": "object",
                "description": "Short-term conversation context",
                "properties": {
                    "session_id": {"type": "string"},
                    "last_message": {"type": "string"},
                    "last_intent": {"type": "string"},
                    "current_topic": {"type": "string"},
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading exercise sources...")
    ex_edb = load_exercisedb_json()
    ex_csv = load_fitness_exercises_csv()
    ex_mega = load_mega_gym_csv()
    ex_wger = load_wger_exercises()
    ex_lh = load_longhaul_exercises()

    print(f"  ExerciseDB JSON   : {len(ex_edb)}")
    print(f"  Fitness CSV       : {len(ex_csv)}")
    print(f"  MegaGym CSV       : {len(ex_mega)}")
    print(f"  Wger JSON         : {len(ex_wger)}")
    print(f"  Longhaul          : {len(ex_lh)}")

    print("Merging & deduplicating exercises...")
    all_exercises = merge_exercises([ex_edb, ex_csv, ex_mega, ex_wger, ex_lh])
    print(f"  Unique exercises  : {len(all_exercises)}")

    # Load workout plan templates (from ultimate_gym CSV)
    plan_templates_raw = load_ultimate_gym()
    # Load calorie stats
    calorie_stats = load_fitness_metrics()

    print("Building domain dictionary...")
    domain_items = build_domain_dictionary()
    print(f"  Domain items      : {len(domain_items)}")

    print("Building workout plan templates...")
    workout_plans = build_workout_plan_templates()

    print("Building user schema...")
    user_schema = build_enhanced_user_schema()

    # Write outputs
    ex_out = DICT_DIR / "master_exercise_dictionary.jsonl"
    _write_jsonl(ex_out, all_exercises)
    print(f"\nWrote: {ex_out}  ({len(all_exercises)} records)")

    domain_out = DICT_DIR / "master_domain_dictionary.jsonl"
    _write_jsonl(domain_out, domain_items)
    print(f"Wrote: {domain_out}  ({len(domain_items)} records)")

    plans_out = DICT_DIR / "master_workout_plans.jsonl"
    _write_jsonl(plans_out, workout_plans)
    print(f"Wrote: {plans_out}  ({len(workout_plans)} records)")

    schema_out = DICT_DIR / "master_user_schema.json"
    schema_out.write_text(json.dumps(user_schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {schema_out}")

    # Ultimate-gym body-part workout templates
    ugym_out = DICT_DIR / "ultimate_gym_templates.jsonl"
    _write_jsonl(ugym_out, plan_templates_raw)
    print(f"Wrote: {ugym_out}  ({len(plan_templates_raw)} body-part templates)")

    # Summary
    summary = {
        "total_exercises": len(all_exercises),
        "domain_items": len(domain_items),
        "workout_plan_templates": len(workout_plans),
        "ultimate_gym_bodypart_templates": len(plan_templates_raw),
        "sources": {
            "exercisedb_json": len(ex_edb),
            "fitness_exercises_csv": len(ex_csv),
            "megagym_csv": len(ex_mega),
            "wger_json": len(ex_wger),
            "longhaul_fitness": len(ex_lh),
        },
        "files": {
            "master_exercise_dictionary": str(ex_out),
            "master_domain_dictionary": str(domain_out),
            "master_workout_plans": str(plans_out),
            "master_user_schema": str(schema_out),
            "ultimate_gym_templates": str(ugym_out),
        }
    }
    summary_out = DICT_DIR / "master_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {summary_out}")
    print("\nMaster dictionary build complete.")


if __name__ == "__main__":
    main()
