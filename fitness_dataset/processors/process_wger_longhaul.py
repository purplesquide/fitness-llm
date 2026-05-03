"""
Process wger_exercises.json and longhaul_fitness/ JSON files into training samples.

wger schema   : {id, category: {name}, muscles, muscles_secondary, equipment: [{name}], translations: [{name, description, language}]}
longhaul/strength.json : [{name, primaryMuscles, secondaryMuscles, steps, notes}]
longhaul/flexibility.json : similar structure
longhaul/cardio.json  : [{name, instructions}]

Output:
  - output/wger_samples.jsonl     : ~200 samples
  - output/longhaul_samples.jsonl : ~200 samples
"""

import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from constants import (
    SYSTEM_PROMPT, random_profile, REP_TABLE, GOAL_TEXT_MAP,
    WGER_JSON, LONGHAUL_STRENGTH, LONGHAUL_CARDIO, LONGHAUL_FLEX, OUTPUT_DIR,
)

ENGLISH_LANG_ID = 2  # wger uses language id=2 for English


def _clean_html(text: str) -> str:
    """Strip basic HTML tags from wger description text."""
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split()).strip()


def process_wger(sample_count: int = 200):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(WGER_JSON).exists():
        print(f"  ✗ {WGER_JSON} not found — skipping wger")
        return

    exercises = json.loads(Path(WGER_JSON).read_text(encoding="utf-8"))
    print(f"  Loaded {len(exercises)} wger exercises")

    random.shuffle(exercises)
    samples = []

    for ex in exercises:
        if len(samples) >= sample_count:
            break

        # Get English translation
        translations = ex.get("translations", [])
        eng = next((t for t in translations if t.get("language") == ENGLISH_LANG_ID), None)
        if not eng:
            continue

        name = eng.get("name", "").strip()
        desc = _clean_html(eng.get("description", "")).strip()

        if not name or len(desc) < 30:
            continue

        category  = ex.get("category", {}).get("name", "General")
        equip_list = [e.get("name", "") for e in ex.get("equipment", []) if e.get("name")]
        equip_str  = ", ".join(equip_list) if equip_list else "bodyweight"

        muscles     = [m.get("name_en", m.get("name", "")) for m in ex.get("muscles", [])]
        muscles_sec = [m.get("name_en", m.get("name", "")) for m in ex.get("muscles_secondary", [])]
        target      = ", ".join(muscles) if muscles else category

        profile  = random_profile()
        goal     = profile["goal"]
        level    = profile["level"]
        goal_txt = GOAL_TEXT_MAP.get(goal, goal.replace("_", " "))
        reps     = REP_TABLE.get(goal, "3 sets of 10-12 reps")

        q_templates = [
            f"How do I do {name}? I'm {level} training for {goal_txt}.",
            f"Explain the {name} exercise — muscles, form, and reps for {goal_txt}.",
            f"What muscles does {name} work and how should I perform it as a {level}?",
            f"Coach me through {name}. Goal: {goal_txt}, Level: {level}.",
        ]

        answer = (
            f"{name} is a {category.lower()} exercise that targets {target}."
            + (f" Secondary muscles: {', '.join(muscles_sec)}." if muscles_sec else "")
            + f" {desc[:300].rstrip('.')}."
            + f" Use {equip_str}. For {goal_txt}, perform {reps}."
        )

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": random.choice(q_templates)},
                {"role": "assistant", "content": answer},
            ]
        })

    output_path = os.path.join(OUTPUT_DIR, "wger_samples.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} wger samples → {output_path}")


def _build_strength_sample(ex: dict, profile: dict) -> dict:
    """Build a guidance sample from a longhaul strength exercise."""
    name      = ex.get("name", "").strip()
    primary   = ex.get("primaryMuscles", [])
    secondary = ex.get("secondaryMuscles", [])
    steps     = ex.get("steps", [])
    notes     = ex.get("notes", "").strip()

    if not name or not steps:
        return None

    target   = ", ".join(primary) if primary else "target muscles"
    step_txt = " ".join(steps[:3]) if steps else f"Perform {name} with controlled form."
    goal     = profile["goal"]
    level    = profile["level"]
    goal_txt = GOAL_TEXT_MAP.get(goal, goal.replace("_", " "))
    reps     = REP_TABLE.get(goal, "3 sets of 10-12 reps")

    q_templates = [
        f"How do I perform {name} correctly? I'm {level} targeting {goal_txt}.",
        f"What are the form cues for {name}? Level: {level}, Goal: {goal_txt}.",
        f"Break down the {name} for a {level} working toward {goal_txt}.",
        f"Teach me {name} — I'm {level} and my goal is {goal_txt}.",
    ]

    answer = (
        f"{name} primarily works the {target}."
        + (f" Secondary movers: {', '.join(secondary)}." if secondary else "")
        + f" {step_txt}"
        + (f" Note: {notes}" if notes else "")
        + f" For {goal_txt}, use {reps}."
        + (" Focus on mastering the pattern before adding weight." if level == "beginner" else "")
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": random.choice(q_templates)},
            {"role": "assistant", "content": answer},
        ]
    }


def _build_flexibility_sample(ex: dict, profile: dict) -> dict:
    """Build a flexibility/mobility guidance sample."""
    name      = ex.get("name", "").strip()
    primary   = ex.get("primaryMuscles", [])
    steps     = ex.get("steps", [])
    notes     = ex.get("notes", "").strip()

    if not name or not steps:
        return None

    target   = ", ".join(primary) if primary else "the target muscles"
    step_txt = " ".join(steps[:3]) if steps else ""

    q_templates = [
        f"How do I do the {name} stretch/mobility drill?",
        f"Walk me through {name} for flexibility and recovery.",
        f"I need to improve {target} mobility — how do I do {name}?",
        f"What is the correct technique for {name}?",
    ]

    answer = (
        f"{name} targets {target} and is effective for improving flexibility and reducing injury risk. "
        f"{step_txt} "
        + (f"{notes} " if notes else "")
        + "Hold each stretch 20-30 seconds or perform 2-3 controlled reps. "
        "Incorporate this into your warm-up or cool-down routine for best results."
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": random.choice(q_templates)},
            {"role": "assistant", "content": answer},
        ]
    }


def _build_cardio_sample(ex: dict, profile: dict) -> dict:
    """Build a cardio guidance sample."""
    name  = ex.get("name", "").strip()
    instr = ex.get("instructions", [])

    if not name:
        return None

    desc_text = " ".join(instr[:1]) if instr else f"{name} is an effective cardiovascular exercise."
    goal     = profile["goal"]
    goal_txt = GOAL_TEXT_MAP.get(goal, goal.replace("_", " "))

    q_templates = [
        f"How should I incorporate {name} into my {goal_txt} routine?",
        f"What are the benefits of {name} for {goal_txt}?",
        f"Guide me on using {name} as cardio for {goal_txt}.",
        f"How much {name} should I do per week for {goal_txt}?",
    ]

    cardio_recs = {
        "weight_loss":     "3-5 sessions per week at moderate intensity (Zone 2-3, 30-45 minutes per session)",
        "muscle_gain":     "2-3 sessions per week of light Zone 2 work to support recovery without competing with strength adaptation",
        "strength":        "2-3 sessions per week of low-intensity Zone 2 cardio — prioritize strength work and keep cardio recovery-focused",
        "endurance":       "4-6 sessions per week progressively building from 20 to 60+ minutes; include one longer session weekly",
        "general_fitness": "3-4 sessions per week, mix of moderate steady-state and one higher-intensity session",
    }
    rec = cardio_recs.get(goal, "3-4 sessions per week at moderate intensity")

    answer = (
        f"{desc_text.strip()} "
        f"For {goal_txt}, aim for {rec}. "
        "Always warm up for 5-10 minutes before reaching your target intensity. "
        "Track heart rate — Zone 2 (can hold a conversation) is sustainable and maximizes fat oxidation."
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": random.choice(q_templates)},
            {"role": "assistant", "content": answer},
        ]
    }


def process_longhaul(sample_count: int = 200):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    samples = []

    # ── Strength exercises ────────────────────────────────────────────────────
    if Path(LONGHAUL_STRENGTH).exists():
        strength = json.loads(Path(LONGHAUL_STRENGTH).read_text(encoding="utf-8"))
        random.shuffle(strength)
        for ex in strength:
            profile = random_profile()
            s = _build_strength_sample(ex, profile)
            if s:
                samples.append(s)
        print(f"  Built {len(samples)} strength samples from longhaul")

    # ── Flexibility exercises ─────────────────────────────────────────────────
    if Path(LONGHAUL_FLEX).exists():
        flex = json.loads(Path(LONGHAUL_FLEX).read_text(encoding="utf-8"))
        flex_samples = []
        for ex in flex:
            profile = random_profile()
            s = _build_flexibility_sample(ex, profile)
            if s:
                flex_samples.append(s)
        samples.extend(flex_samples)
        print(f"  Built {len(flex_samples)} flexibility samples from longhaul")

    # ── Cardio exercises ──────────────────────────────────────────────────────
    if Path(LONGHAUL_CARDIO).exists():
        cardio = json.loads(Path(LONGHAUL_CARDIO).read_text(encoding="utf-8"))
        cardio_samples = []
        for ex in cardio:
            profile = random_profile()
            s = _build_cardio_sample(ex, profile)
            if s:
                cardio_samples.append(s)
        # Repeat cardio samples with different profiles to get more coverage
        for ex in cardio * 4:
            profile = random_profile()
            s = _build_cardio_sample(ex, profile)
            if s:
                cardio_samples.append(s)
        samples.extend(cardio_samples[:30])
        print(f"  Built {min(len(cardio_samples), 30)} cardio samples from longhaul")

    random.shuffle(samples)
    samples = samples[:sample_count]

    output_path = os.path.join(OUTPUT_DIR, "longhaul_samples.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} longhaul samples → {output_path}")


if __name__ == "__main__":
    print("Processing wger exercises...")
    process_wger()
    print("Processing longhaul fitness data...")
    process_longhaul()
    print("Done.")
