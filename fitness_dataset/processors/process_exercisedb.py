"""
Process local ExerciseDB data (fitness_exercises/exercises.csv + exercisedb_v1_sample/exercises.json)
into exercise guidance and substitution training samples.

Sources:
  - fitness_exercises/exercises.csv  : 1,325 exercises (bodyPart, equipment, name, target, secondaryMuscles, instructions)
  - exercisedb/exercisedb_v1_sample/exercises.json : 30 exercises (same schema, JSON format)

Output:
  - output/exercisedb_guidance.jsonl      : ~400 guidance samples
  - output/exercisedb_substitutions.jsonl : ~250 substitution samples
"""

import csv
import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from constants import (
    SYSTEM_PROMPT, random_profile, REP_TABLE, GOAL_TEXT_MAP,
    EXERCISEDB_CSV, EXERCISEDB_JSON, OUTPUT_DIR,
)

# Equipment substitution map
EQUIP_SUBS = {
    "barbell":        ["dumbbell", "resistance band", "bodyweight"],
    "cable":          ["dumbbell", "resistance band"],
    "machine":        ["dumbbell", "bodyweight"],
    "ez barbell":     ["dumbbell", "barbell"],
    "smith machine":  ["barbell", "dumbbell"],
    "sled machine":   ["leg press machine", "dumbbell"],
    "leverage machine": ["dumbbell", "cable"],
    "kettlebell":     ["dumbbell", "bodyweight"],
    "medicine ball":  ["dumbbell", "bodyweight"],
    "band":           ["cable", "dumbbell"],
    "bosu ball":      ["bodyweight"],
    "roller":         ["bodyweight"],
    "rope":           ["cable"],
    "tire":           ["bodyweight"],
    "trap bar":       ["barbell", "dumbbell"],
    "weighted":       ["dumbbell", "barbell"],
}


def _load_csv_exercises() -> list:
    """Load exercises from fitness_exercises/exercises.csv."""
    exercises = []
    with open(EXERCISEDB_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Collect instructions from the wide columns
            instructions = []
            for i in range(11):
                col = f"instructions/{i}"
                val = row.get(col, "").strip()
                if val:
                    instructions.append(val)

            # Collect secondary muscles
            secondary = []
            for i in range(6):
                col = f"secondaryMuscles/{i}"
                val = row.get(col, "").strip()
                if val:
                    secondary.append(val)

            exercises.append({
                "name":             row.get("name", "").strip(),
                "bodyPart":         row.get("bodyPart", "").strip(),
                "equipment":        row.get("equipment", "body weight").strip(),
                "targetMuscles":    [row.get("target", "").strip()] if row.get("target") else [],
                "secondaryMuscles": secondary,
                "instructions":     instructions,
            })
    return exercises


def _load_json_exercises() -> list:
    """Load exercises from exercisedb_v1_sample/exercises.json."""
    raw = json.loads(Path(EXERCISEDB_JSON).read_text(encoding="utf-8"))
    exercises = []
    for ex in raw:
        exercises.append({
            "name":             ex.get("name", "").strip(),
            "bodyPart":         ", ".join(ex.get("bodyParts", [])),
            "equipment":        ex.get("equipments", ["bodyweight"])[0] if ex.get("equipments") else "bodyweight",
            "targetMuscles":    ex.get("targetMuscles", []),
            "secondaryMuscles": ex.get("secondaryMuscles", []),
            "instructions":     ex.get("instructions", []),
        })
    return exercises


def build_guidance_sample(exercise: dict, profile: dict) -> dict:
    """Turn one exercise record into a guidance Q&A training sample."""
    name   = exercise["name"].replace("-", " ").title()
    target = ", ".join(exercise["targetMuscles"]) or exercise.get("bodyPart", "target muscles")
    second = ", ".join(exercise["secondaryMuscles"])
    instr  = exercise["instructions"]
    equip  = exercise["equipment"]

    steps = " ".join(instr[:3]) if instr else f"Perform {name} with controlled form."
    level = profile["level"]
    goal  = profile["goal"]
    reps  = REP_TABLE.get(goal, "3 sets of 10-12 reps")
    goal_txt = GOAL_TEXT_MAP.get(goal, goal.replace("_", " "))

    question_templates = [
        f"I am {level} level targeting {goal_txt}. How do I do {name} correctly?",
        f"Can you explain the proper form for {name}? My goal is {goal_txt}.",
        f"What is the correct technique for {name}? I am {level} and training for {goal_txt}.",
        f"How should I perform {name} as a {level} with {goal_txt} as my goal?",
        f"Give me coaching cues for {name}. Level: {level.capitalize()}, Goal: {goal_txt}.",
        f"I want to learn {name} — what should I focus on? I'm a {level} training for {goal_txt}.",
    ]

    answer = (
        f"{name} primarily targets the {target}."
        + (f" Secondary movers include {second}." if second else "")
        + f" {steps}"
        + f" For {goal_txt}, use {reps} with {equip}."
        + (f" As a {level}, prioritize controlled form before adding load." if level == "beginner"
           else f" At {level} level, track progressive overload every session.")
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": random.choice(question_templates)},
            {"role": "assistant", "content": answer},
        ]
    }


def build_substitution_sample(exercise: dict, profile: dict) -> dict:
    """Generate an exercise substitution training sample."""
    name  = exercise["name"].replace("-", " ").title()
    equip = exercise["equipment"].lower()
    target = ", ".join(exercise["targetMuscles"]) or exercise.get("bodyPart", "the same muscles")
    goal  = profile["goal"]

    alt_options = EQUIP_SUBS.get(equip, ["dumbbell", "bodyweight"])
    chosen_alt  = random.choice(alt_options)

    reps = "3-4 sets of 8-12" if "muscle" in goal else "3 sets of 10-15"

    answer = (
        f"Without access to {equip} equipment, you can target {target} using {chosen_alt} alternatives. "
        f"Replicate the movement pattern as closely as possible — the loading tool matters less than the muscle action. "
        f"Use {reps} reps to match your {GOAL_TEXT_MAP.get(goal, goal.replace('_', ' '))} goal. "
        + ("Focus on slow eccentrics to compensate for lower load." if chosen_alt == "bodyweight"
           else f"Match the {chosen_alt} load to your usual effort level and track reps for progression.")
    )

    question_templates = [
        f"I don't have {equip} equipment. What replaces {name}?",
        f"Can I substitute {name} with something? I only have {chosen_alt}.",
        f"I have a {profile.get('equipment', 'home gym')} setup. What can I use instead of {name}?",
        f"My gym doesn't have the right equipment for {name}. Best substitution?",
        f"Home workout replacement for {name}? I'm {profile['level']} targeting {GOAL_TEXT_MAP.get(goal, goal)}.",
    ]

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": random.choice(question_templates)},
            {"role": "assistant", "content": answer},
        ]
    }


def process_exercisedb(
    guidance_count: int = 400,
    subs_count: int     = 250,
):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load and merge all exercise sources
    exercises = []
    if Path(EXERCISEDB_CSV).exists():
        csv_exs = _load_csv_exercises()
        exercises.extend(csv_exs)
        print(f"  Loaded {len(csv_exs)} exercises from CSV")
    if Path(EXERCISEDB_JSON).exists():
        json_exs = _load_json_exercises()
        exercises.extend(json_exs)
        print(f"  Loaded {len(json_exs)} exercises from JSON")

    print(f"  Total exercises available: {len(exercises)}")
    random.shuffle(exercises)

    # ── Guidance samples ──────────────────────────────────────────────────────
    guidance_samples = []
    for ex in exercises[:guidance_count]:
        profile = random_profile()
        sample = build_guidance_sample(ex, profile)
        if sample:
            guidance_samples.append(sample)

    guidance_out = os.path.join(OUTPUT_DIR, "exercisedb_guidance.jsonl")
    with open(guidance_out, "w", encoding="utf-8") as f:
        for s in guidance_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(guidance_samples)} guidance samples → {guidance_out}")

    # ── Substitution samples ──────────────────────────────────────────────────
    equip_exercises = [e for e in exercises if e.get("equipment", "body weight").lower() not in ("body weight", "bodyweight", "")]
    random.shuffle(equip_exercises)
    sub_samples = []
    for ex in equip_exercises[:subs_count]:
        profile = random_profile()
        sample = build_substitution_sample(ex, profile)
        if sample:
            sub_samples.append(sample)

    subs_out = os.path.join(OUTPUT_DIR, "exercisedb_substitutions.jsonl")
    with open(subs_out, "w", encoding="utf-8") as f:
        for s in sub_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(sub_samples)} substitution samples → {subs_out}")


if __name__ == "__main__":
    print("Processing ExerciseDB data...")
    process_exercisedb()
    print("Done.")
