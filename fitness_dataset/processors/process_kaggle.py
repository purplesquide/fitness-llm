"""
Process megaGymDataset/megaGymDataset.csv into program/split advice and exercise
description training samples.

megaGymDataset columns: Title, Desc, Type, BodyPart, Equipment, Level, Rating, RatingDesc

Output:
  - output/kaggle_samples.jsonl : ~400 program advice samples
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
    MEGA_GYM_CSV, OUTPUT_DIR,
)

LEVEL_MAP = {
    "beginner":     "beginner",
    "intermediate": "intermediate",
    "advanced":     "advanced",
    "expert":       "advanced",
    "": "intermediate",
}

SPLIT_RECS = {
    2: "Full Body (2x/week) — 2 days requires training everything each session to maximize frequency",
    3: "Full Body (3x/week) — optimal for beginners; every movement pattern trained every session",
    4: "Upper/Lower (4x/week) — each half of the body trained twice with 48h recovery between sessions",
    5: "Upper/Lower + Weak Point day, or Push/Pull/Legs/Upper/Lower rotation",
    6: "Push/Pull/Legs (2x rotation) — 6 days allows every muscle group twice per week",
}

VOLUME_RECS = {
    "muscle_gain":     "10-20 quality sets per muscle group per week with progressive overload every session",
    "strength":        "4-6 sets of 3-6 reps on main lifts; lower accessory volume keeps fatigue manageable",
    "weight_loss":     "moderate volume (10-15 sets/muscle), add metabolic circuits to maximize calorie burn",
    "endurance":       "higher reps (15-25), 45-60 second rest periods, circuit-style training",
    "general_fitness": "balanced 8-15 reps, full movement pattern coverage, 60-90 second rest periods",
}


def make_program_sample(row: dict, profile: dict) -> dict:
    """Convert one megaGym row into a split/program advice Q&A sample."""
    level    = LEVEL_MAP.get(str(row.get("Level", "")).lower().strip(), profile["level"])
    goal     = profile["goal"]
    days     = profile["days"]
    equip    = profile["equipment"]
    goal_txt = GOAL_TEXT_MAP.get(goal, goal.replace("_", " "))
    split    = SPLIT_RECS.get(days, "Upper/Lower")
    volume   = VOLUME_RECS.get(goal, "10-15 sets per muscle group per week")

    q_templates = [
        f"I am {level}, goal is {goal_txt}, training {days} days a week with {equip}. What split should I follow?",
        f"What is the best program structure for a {level} trying to achieve {goal_txt} training {days} days a week?",
        f"I have {days} days per week and access to {equip}. I am {level} focused on {goal_txt}. Which split?",
        f"Design a weekly structure for me: {level}, {goal_txt}, {days} training days, {equip}.",
        f"Best workout split for {days} days/week, {level} level, {goal_txt} goal?",
    ]

    answer = (
        f"For a {level} with {days} days per week targeting {goal_txt}, "
        f"the optimal structure is {split}. "
        f"Volume prescription: {volume}. "
        f"With {equip}, prioritize compound movements first, isolations last. "
        + ("Start conservative — 70% of estimated max — and add 2.5-5% weekly." if level == "beginner"
           else "Track progressive overload session to session — add at least one rep or 2.5kg weekly.")
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": random.choice(q_templates)},
            {"role": "assistant", "content": answer},
        ]
    }


def make_exercise_desc_sample(row: dict, profile: dict) -> dict:
    """Convert one megaGym row into an exercise explanation sample using Desc column."""
    title = str(row.get("Title", "")).strip()
    desc  = str(row.get("Desc", "")).strip()
    body_part = str(row.get("BodyPart", "")).strip()
    equip     = str(row.get("Equipment", "")).strip()
    level_raw = str(row.get("Level", "")).lower().strip()
    level     = LEVEL_MAP.get(level_raw, profile["level"])
    goal      = profile["goal"]
    goal_txt  = GOAL_TEXT_MAP.get(goal, goal.replace("_", " "))
    reps      = REP_TABLE.get(goal, "3 sets of 10-12 reps")

    if not title or not desc or len(desc) < 40:
        return None

    q_templates = [
        f"What is the {title} exercise and how do I do it? I'm {level} targeting {goal_txt}.",
        f"Explain the {title} — muscles worked, form tips, and rep range for {goal_txt}.",
        f"How do I program {title} for {goal_txt}? I'm {level} with {equip} access.",
        f"Give me a coaching breakdown of {title}. Level: {level}, Goal: {goal_txt}.",
    ]

    answer = (
        f"{title} is a {body_part.lower()} exercise using {equip.lower() if equip else 'bodyweight'}. "
        f"{desc[:300].rstrip('.')}. "
        f"For {goal_txt}, use {reps}. "
        + ("Focus on learning the movement pattern before increasing load." if level == "beginner"
           else f"As a {level}, apply progressive overload — add load or reps each session.")
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": random.choice(q_templates)},
            {"role": "assistant", "content": answer},
        ]
    }


def process_kaggle(sample_count: int = 400):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(MEGA_GYM_CSV).exists():
        print(f"  ✗ {MEGA_GYM_CSV} not found — skipping")
        return

    rows = []
    with open(MEGA_GYM_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"  Loaded {len(rows)} rows from megaGymDataset.csv")
    random.shuffle(rows)

    samples = []

    # Half program samples, half exercise description samples
    half = sample_count // 2
    for row in rows[:half]:
        profile = random_profile()
        s = make_program_sample(row, profile)
        if s:
            samples.append(s)

    for row in rows[half:half * 2]:
        profile = random_profile()
        s = make_exercise_desc_sample(row, profile)
        if s:
            samples.append(s)

    output_path = os.path.join(OUTPUT_DIR, "kaggle_samples.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples → {output_path}")


if __name__ == "__main__":
    print("Processing megaGymDataset (Kaggle)...")
    process_kaggle()
    print("Done.")
