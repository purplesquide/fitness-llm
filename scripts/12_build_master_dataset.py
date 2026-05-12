"""
Build the master training dataset from ALL available data sources.

Generates diverse, high-quality fine-tuning examples across:
  1. Exercise guidance (technique, cues, muscles) — from all exercise sources
  2. Exercise substitutions (equipment-constrained alternatives)
  3. Workout programming (split design, volume, frequency)
  4. Nutrition guidance (protein, calories, macros, supplements)
  5. Sports science explanations (hypertrophy, recovery, progressive overload)
  6. Safety & injury guidance
  7. Carry-forward: existing curated datasets (filtered for quality)

Output:
  artifacts/datasets/master_training_data.jsonl  — complete merged dataset
  artifacts/datasets/master_training_data_clean.jsonl — filtered high-quality subset
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = ROOT / "artifacts" / "dictionaries"
OUT_DIR = ROOT / "artifacts" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert fitness coach built into a workout planning app.
You give evidence-based, concise advice in 3-5 sentences.
Always consider the user's fitness level, goal, and available equipment in your response.
You never recommend unsafe practices. Speak in a direct, motivating coach tone.
When relevant, cite the reason behind your recommendation (e.g., progressive overload, muscle recovery, movement pattern balance)."""

# ─────────────────────────────────────────────────────────────────────────────
# Profile pools
# ─────────────────────────────────────────────────────────────────────────────
LEVELS = ["beginner", "intermediate", "advanced"]
GOALS = ["muscle_gain", "weight_loss", "strength", "endurance", "general_fitness"]
EQUIPMENT_POOLS = [
    "bodyweight only",
    "home gym (dumbbells only)",
    "home gym (dumbbells + pull-up bar)",
    "minimal gym (barbell + bench + rack)",
    "full commercial gym",
]
FREQUENCIES = [2, 3, 4, 5, 6]
GOAL_TEXT = {
    "muscle_gain": "muscle building",
    "weight_loss": "fat loss",
    "strength": "strength development",
    "endurance": "endurance training",
    "general_fitness": "general fitness",
}
REP_TABLE = {
    "muscle_gain": "3-4 sets of 8-12 reps",
    "strength": "4-5 sets of 3-6 reps",
    "weight_loss": "3-4 sets of 12-20 reps",
    "endurance": "3-4 sets of 15-25 reps",
    "general_fitness": "3 sets of 10-15 reps",
}
LEVEL_NOTES = {
    "beginner": "As a beginner, prioritise controlled form before adding load. Keep 2-3 reps in reserve.",
    "intermediate": "At intermediate level, track progressive overload every session. Push to RPE 7-8.",
    "advanced": "At advanced level, use periodisation and track weekly volume per muscle group.",
}

def rp() -> dict:
    return {
        "level": random.choice(LEVELS),
        "goal": random.choice(GOALS),
        "equipment": random.choice(EQUIPMENT_POOLS),
        "days": random.choice(FREQUENCIES),
        "weight": random.randint(55, 110),
        "age": random.randint(17, 55),
    }

def _norm(v: Any) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v).strip().lower())

def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

def _load_jsonl(path: Path) -> list[dict]:
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

def _sample(path: Path) -> list[dict]:
    return _load_jsonl(path)

def chat(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXERCISE GUIDANCE from master_exercise_dictionary.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def gen_exercise_guidance(exercises: list[dict]) -> list[dict]:
    samples = []
    question_templates = [
        lambda ex, p: (
            f"I am {p['level']} level targeting {GOAL_TEXT[p['goal']]}. How do I do the {ex['name']} correctly?",
            _ex_guidance_answer(ex, p)
        ),
        lambda ex, p: (
            f"Coach me through {ex['name']}. Goal: {GOAL_TEXT[p['goal']]}, Level: {p['level']}.",
            _ex_guidance_answer(ex, p)
        ),
        lambda ex, p: (
            f"What muscles does {ex['name']} work and how should I perform it as a {p['level']}?",
            _ex_muscles_answer(ex, p)
        ),
        lambda ex, p: (
            f"Give me coaching cues for {ex['name']}. Level: {p['level'].title()}, Goal: {GOAL_TEXT[p['goal']]}.",
            _ex_cues_answer(ex, p)
        ),
        lambda ex, p: (
            f"Explain the {ex['name']} exercise — muscles, form, and reps for {GOAL_TEXT[p['goal']]}.",
            _ex_full_answer(ex, p)
        ),
    ]
    for ex in exercises:
        name = ex.get("name", "")
        instr = ex.get("instructions", "")
        primary = ex.get("primary_muscles", [])
        if not name or not instr or len(instr) < 40:
            continue
        p = rp()
        tmpl = random.choice(question_templates)
        try:
            q, a = tmpl(ex, p)
            if q and a and len(a) > 60:
                samples.append(chat(q, a))
        except Exception:
            pass
    return samples


def _ex_guidance_answer(ex: dict, p: dict) -> str:
    name = ex.get("name", "")
    primary = ", ".join(ex.get("primary_muscles", [])) or "the target muscles"
    secondary = ", ".join(ex.get("secondary_muscles", [])) or ""
    instr = ex.get("instructions", "")[:400].strip()
    reps = REP_TABLE.get(p["goal"], "3 sets of 10-12 reps")
    equip = ex.get("equipment", "appropriate equipment")
    level_note = LEVEL_NOTES.get(p["level"], "")
    parts = [f"{name} primarily targets {primary}."]
    if secondary:
        parts.append(f"Secondary muscles: {secondary}.")
    if instr:
        parts.append(instr)
    parts.append(f"For {GOAL_TEXT[p['goal']]}, use {reps} with {equip}.")
    parts.append(level_note)
    return " ".join(parts)


def _ex_muscles_answer(ex: dict, p: dict) -> str:
    name = ex.get("name", "")
    primary = ", ".join(ex.get("primary_muscles", [])) or "multiple muscles"
    secondary = ", ".join(ex.get("secondary_muscles", [])) or "supporting muscles"
    instr = ex.get("instructions", "")[:300].strip()
    reps = REP_TABLE.get(p["goal"], "3 sets of 10-12 reps")
    return (
        f"{name} primarily works {primary}. "
        f"Secondary movers include {secondary}. "
        f"{instr} "
        f"For {GOAL_TEXT[p['goal']]}, perform {reps}."
    )


def _ex_cues_answer(ex: dict, p: dict) -> str:
    name = ex.get("name", "")
    instr = ex.get("instructions", "")[:350].strip()
    primary = ", ".join(ex.get("primary_muscles", [])) or "target muscles"
    reps = REP_TABLE.get(p["goal"], "3 sets of 10-12 reps")
    level_note = LEVEL_NOTES.get(p["level"], "")
    return (
        f"{name} primarily targets the {primary}. "
        f"{instr} "
        f"For {GOAL_TEXT[p['goal']]}, use {reps}. "
        f"{level_note}"
    )


def _ex_full_answer(ex: dict, p: dict) -> str:
    name = ex.get("name", "")
    primary = ", ".join(ex.get("primary_muscles", [])) or "target muscles"
    secondary = ", ".join(ex.get("secondary_muscles", [])) or ""
    instr = ex.get("instructions", "")[:400].strip()
    equip = ex.get("equipment", "")
    reps = REP_TABLE.get(p["goal"], "3 sets of 10-12 reps")
    sec_part = f" Secondary muscles: {secondary}." if secondary else ""
    equip_part = f" Use {equip}." if equip and equip != "bodyweight" else ""
    return (
        f"{name} is a {ex.get('movement_pattern', 'strength')} exercise that targets {primary}.{sec_part} "
        f"{instr}{equip_part} For {GOAL_TEXT[p['goal']]}, perform {reps}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXERCISE SUBSTITUTIONS
# ─────────────────────────────────────────────────────────────────────────────

EQUIPMENT_SUBSTITUTIONS: dict[str, list[str]] = {
    "barbell": ["dumbbell", "bodyweight", "resistance band"],
    "dumbbell": ["barbell", "cable", "resistance band", "bodyweight"],
    "cable": ["resistance band", "dumbbell"],
    "machine": ["dumbbell", "cable", "bodyweight"],
    "sled machine": ["leg press", "barbell squat"],
    "smith machine": ["barbell", "dumbbell"],
    "leverage machine": ["dumbbell", "cable", "bodyweight"],
    "resistance band": ["cable", "dumbbell"],
    "bodyweight": ["dumbbell", "resistance band"],
}

MUSCLE_SUBSTITUTES: dict[str, list[str]] = {
    "chest": ["push-up", "dumbbell press", "cable fly", "dip"],
    "lats": ["pull-up", "lat pulldown", "dumbbell row", "resistance band pulldown"],
    "quadriceps": ["bodyweight squat", "split squat", "step-up", "leg press"],
    "hamstrings": ["stiff-leg deadlift", "Nordic curl", "lying leg curl", "Swiss ball curl"],
    "glutes": ["hip thrust", "glute bridge", "cable kickback", "step-up"],
    "shoulders": ["dumbbell press", "lateral raise", "face pull", "resistance band press"],
    "biceps": ["pull-up", "resistance band curl", "dumbbell curl", "cable curl"],
    "triceps": ["push-up", "tricep dip", "overhead dumbbell extension", "resistance band pushdown"],
    "calves": ["bodyweight calf raise", "seated calf raise", "step calf raise"],
    "abs": ["plank", "dead bug", "hollow hold", "bicycle crunch"],
}


def gen_substitutions(exercises: list[dict]) -> list[dict]:
    samples = []
    # Build a muscle → exercise map from our exercise list
    muscle_ex_map: dict[str, list[str]] = {}
    for ex in exercises:
        for m in ex.get("primary_muscles", []):
            muscle_ex_map.setdefault(m, []).append(ex["name"])

    sub_questions = [
        lambda ex, sub_equip: (
            f"I don't have {ex.get('equipment','the required')} equipment. What replaces {ex['name']}?",
            _sub_answer(ex, sub_equip, muscle_ex_map)
        ),
        lambda ex, sub_equip: (
            f"Home workout replacement for {ex['name']}? I'm {rp()['level']} targeting {GOAL_TEXT[rp()['goal']]}.",
            _sub_answer(ex, sub_equip, muscle_ex_map)
        ),
        lambda ex, sub_equip: (
            f"My gym doesn't have the right equipment for {ex['name']}. Best substitution?",
            _sub_answer(ex, sub_equip, muscle_ex_map)
        ),
    ]
    seen = set()
    for ex in exercises:
        equip = ex.get("equipment", "")
        subs = EQUIPMENT_SUBSTITUTIONS.get(equip, [])
        if not subs:
            continue
        sub_equip = random.choice(subs)
        primary = ex.get("primary_muscles", [])
        key = _hash(ex["name"] + sub_equip)
        if key in seen:
            continue
        seen.add(key)
        tmpl = random.choice(sub_questions)
        try:
            q, a = tmpl(ex, sub_equip)
            if q and a:
                p = rp()
                samples.append(chat(q, a))
        except Exception:
            pass
    return samples


def _sub_answer(ex: dict, sub_equip: str, muscle_ex_map: dict) -> str:
    primary_list = ex.get("primary_muscles", [])
    primary_str = ", ".join(primary_list) or "target muscles"
    p = rp()
    reps = REP_TABLE.get(p["goal"], "3 sets of 10-12 reps")

    # Find alternatives from the muscle map
    alts = []
    for m in primary_list[:2]:
        candidates = [n for n in muscle_ex_map.get(m, []) if _norm(n) != _norm(ex["name"])]
        alts.extend(candidates[:3])
    if not alts:
        alts = MUSCLE_SUBSTITUTES.get(primary_list[0] if primary_list else "chest", ["bodyweight alternative"])

    # Pick top 2-3 alternatives
    shown_alts = list(dict.fromkeys(alts))[:3]
    alt_str = ", ".join(shown_alts) if shown_alts else "a bodyweight alternative targeting the same muscles"

    return (
        f"Without access to {ex.get('equipment', 'that equipment')}, you can target {primary_str} "
        f"using {sub_equip} alternatives. Good options that match the movement pattern: {alt_str}. "
        f"Replicate the muscle action as closely as possible — the loading tool matters less than the stimulus. "
        f"Use {reps} and match your usual effort level for progression."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. WORKOUT PROGRAMMING Q&A
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_RECOMMENDATIONS = {
    (2, "any"): ("Full Body 2x", "2 days requires full body each session to maximise frequency. Cover all major groups each session."),
    (3, "any"): ("Full Body 3x", "3-day full body provides the best beginner and intermediate stimulus. Hit compounds each session."),
    (4, "muscle_gain"): ("Upper/Lower 4x", "Upper/lower provides twice-weekly frequency and manageable session lengths."),
    (4, "strength"): ("Upper/Lower 4x", "Upper A/B + Lower A/B with heavy and volume days for each."),
    (4, "weight_loss"): ("Upper/Lower 4x", "Upper/lower with higher rep ranges and shorter rest periods to increase calorie burn."),
    (5, "muscle_gain"): ("Push/Pull/Legs + 2 extras", "5-day PPL variant with an extra upper and lower day."),
    (6, "muscle_gain"): ("Push/Pull/Legs 6x", "Classic PPL with 3x weekly push, pull, and leg frequency."),
    (6, "strength"): ("Daily Undulating Periodization", "6-day program alternating heavy and volume days for strength + hypertrophy."),
}


def gen_programming_qa() -> list[dict]:
    samples = []
    for p in [rp() for _ in range(180)]:
        goal = p["goal"]
        level = p["level"]
        days = p["days"]
        equip = p["equipment"]
        split_key = (days, goal) if (days, goal) in SPLIT_RECOMMENDATIONS else (days, "any")
        if split_key not in SPLIT_RECOMMENDATIONS and (days, "any") not in SPLIT_RECOMMENDATIONS:
            continue
        split_name, split_note = SPLIT_RECOMMENDATIONS.get(split_key, SPLIT_RECOMMENDATIONS.get((days, "any"), ("Full Body", "Train all muscle groups each session.")))

        q = random.choice([
            f"Design a weekly structure for me: {level}, {GOAL_TEXT[goal]}, {days} training days, {equip}.",
            f"What split should I use? Level: {level}, goal: {GOAL_TEXT[goal]}, {days} days/week, {equip}.",
            f"I'm a {level} training {days} days per week for {GOAL_TEXT[goal]}. What program structure?",
            f"Recommend a training structure for a {level} with {days} days available targeting {GOAL_TEXT[goal]}.",
        ])

        reps = REP_TABLE.get(goal, "3 sets of 10-12 reps")
        answer = (
            f"For a {level} with {days} days per week targeting {GOAL_TEXT[goal]}, the optimal structure is {split_name}. "
            f"{split_note} "
            f"Volume prescription: {reps}, appropriate rest periods for {GOAL_TEXT[goal]}. "
            f"With {equip}, prioritise compound movements first, isolation work last. "
            f"Track progressive overload session to session — add at least one rep or 2.5 kg weekly."
        )
        samples.append(chat(q, answer))

    # Progressive overload Q&A
    for _ in range(40):
        exercise = random.choice(["squat", "bench press", "deadlift", "pull-up", "overhead press", "row", "hip thrust"])
        rep_target = random.choice(["8 to 10", "5 to 8", "10 to 12", "6 to 8"])
        p = rp()
        q = f"When should I increase the weight on my {exercise} if my target is {rep_target} reps?"
        answer = (
            f"Increase the load once you can hit the top of the {rep_target} range across all prescribed "
            f"working sets with clean technique and at least 1-2 reps still in reserve. "
            f"If form degrades or the last set falls short, keep the same load until you own those reps. "
            f"For the {exercise}, small increments (2.5 kg on barbell, 1-2 kg on dumbbells) keep progress "
            f"moving without turning one good session into several bad ones."
        )
        samples.append(chat(q, answer))

    # Plateau Q&A
    for _ in range(40):
        exercise = random.choice(["squat", "pull-up", "bench press", "deadlift", "overhead press"])
        weeks = random.choice([2, 3, 4, 5, 6])
        p = rp()
        q = f"My {exercise} has not improved for {weeks} weeks and I am {p['level']}. What should I change?"
        answer = (
            f"A {weeks}-week stall on the {exercise} usually means one of: insufficient calories or protein, "
            f"inadequate sleep (under 7 h), technique breakdown limiting load, or accumulated fatigue. "
            f"Audit sleep, nutrition, and form first. If those are sound, take a 3-5 day deload (reduce volume "
            f"by 40-50%), then return with a slightly different rep target or a new intensity technique. "
            f"At {p['level']} level, progress is not always linear — trends across 4+ sessions matter more than any single workout."
        )
        samples.append(chat(q, answer))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. NUTRITION Q&A
# ─────────────────────────────────────────────────────────────────────────────

def gen_nutrition_qa() -> list[dict]:
    samples = []
    for _ in range(100):
        weight = random.randint(55, 110)
        p = rp()
        goal = p["goal"]
        # Protein Q
        prot_min = round(weight * 1.6)
        prot_max = round(weight * 2.2)
        prot_cut = round(weight * 2.2)
        q = random.choice([
            f"I weigh {weight} kg and my goal is {GOAL_TEXT[goal]}. How much protein should I eat?",
            f"As a {p['level']} aiming for {GOAL_TEXT[goal]}, how much protein per day?",
            f"Protein target for {weight} kg bodyweight, goal: {GOAL_TEXT[goal]}?",
        ])
        if goal in ("muscle_gain", "general_fitness"):
            answer = (
                f"Target {prot_min}–{prot_max} g of protein per day at {weight} kg. "
                f"The 1.6–2.2 g/kg range is well-supported by current sports nutrition research for muscle retention and growth. "
                f"Distribute across 3–5 meals (20–40 g per meal) to maximise muscle protein synthesis. "
                f"Consistent daily total matters more than precise timing. "
                f"Common sources: 100 g chicken breast = ~31 g, 3 whole eggs = ~18 g, 200 g Greek yogurt = ~20 g."
            )
        elif goal == "weight_loss":
            answer = (
                f"At {weight} kg during a fat-loss phase, aim for {prot_cut}–{round(weight*2.4)} g of protein daily. "
                f"Higher protein (2.0–2.4 g/kg) is especially important in a deficit to preserve lean mass. "
                f"Protein also increases satiety, making the calorie deficit easier to maintain. "
                f"Distribute over 3–5 meals; each meal should contain at least 20 g protein."
            )
        elif goal == "strength":
            answer = (
                f"For strength development at {weight} kg, 1.6–2.2 g/kg means {prot_min}–{prot_max} g/day. "
                f"Strength training relies heavily on neural adaptation, but protein remains essential for tissue repair and recovery. "
                f"Hit the daily total consistently — meal timing matters less than total intake. "
                f"Prioritise complete protein sources: meat, eggs, dairy, or high-leucine plant blends."
            )
        else:
            answer = (
                f"For {GOAL_TEXT[goal]} at {weight} kg, target {prot_min}–{prot_max} g protein daily (1.6–2.2 g/kg). "
                f"Adequate protein supports muscle maintenance, recovery, and body composition. "
                f"Spread intake across 3–5 meals. Adjust based on total calorie intake and progress."
            )
        samples.append(chat(q, answer))

    # Calorie Q&A
    for _ in range(60):
        weight = random.randint(55, 110)
        p = rp()
        goal = p["goal"]
        maint = weight * random.randint(30, 36)
        q = random.choice([
            f"How many calories should I eat per day for {GOAL_TEXT[goal]} at {weight} kg?",
            f"Calorie target for a {p['level']} weighing {weight} kg aiming for {GOAL_TEXT[goal]}?",
            f"I'm {weight} kg and want to {GOAL_TEXT[goal]}. How many kcal per day?",
        ])
        if goal == "muscle_gain":
            cal_target = maint + random.randint(250, 400)
            answer = (
                f"For lean muscle gain at {weight} kg, start around {cal_target} kcal/day — "
                f"roughly 200–400 kcal above your estimated maintenance (~{maint} kcal). "
                f"A modest surplus prevents excessive fat gain while supporting muscle growth. "
                f"Aim for 0.25–0.5% bodyweight gain per week. Adjust after 2 weeks based on scale and performance."
            )
        elif goal == "weight_loss":
            cal_target = maint - random.randint(400, 600)
            answer = (
                f"For fat loss at {weight} kg, target ~{cal_target} kcal/day — "
                f"a 400–600 kcal deficit below maintenance (~{maint} kcal). "
                f"This supports 0.5–1% bodyweight loss per week while preserving muscle. "
                f"Keep protein at 2.0–2.4 g/kg. Do not reduce below 1,400 kcal for women or 1,600 kcal for men."
            )
        else:
            answer = (
                f"For {GOAL_TEXT[goal]} at {weight} kg, maintenance calories are approximately {maint} kcal/day. "
                f"Adjust ±200–400 kcal depending on weekly weight trend and energy levels. "
                f"Track calories for 2 weeks to establish your true maintenance, then calibrate from there."
            )
        samples.append(chat(q, answer))

    # Supplement Q&A
    suppl_qa = [
        ("Is creatine worth taking?",
         "Creatine monohydrate is the most evidence-backed supplement in sports nutrition. 3–5 g/day improves strength output, lean mass, and high-intensity performance. No loading phase is needed. It is safe for long-term use in healthy individuals. Take it consistently at any time of day — timing is irrelevant."),
        ("Should I use a pre-workout supplement?",
         "Pre-workout supplements primarily deliver caffeine (150–300 mg), which acutely improves focus, endurance, and strength. The other ingredients (beta-alanine, citrulline, tyrosine) have modest or training-specific benefits. If sleep, nutrition, and hydration are solid, pre-workout can offer a small edge. Avoid caffeine within 6 hours of sleep to protect recovery quality."),
        ("Do I need BCAAs?",
         "BCAAs are unnecessary if you already consume sufficient total protein (1.6–2.2 g/kg daily). The leucine, isoleucine, and valine in BCAAs are present in any complete protein source. Save the money and invest it in whole food protein or creatine, which has much stronger evidence. BCAAs may be useful for fasted training, but this scenario rarely applies to most people."),
        ("What is the best protein powder?",
         "Whey isolate or concentrate are highly effective, fast-digesting, and cost-efficient. Casein is useful before bed for slow-release protein. For plant-based options, pea protein combined with rice protein delivers a complete amino acid profile. Choose based on tolerance, preference, and budget — the differences between quality proteins are small if total intake is adequate."),
        ("Should I take omega-3?",
         "Omega-3 fatty acids (EPA + DHA) reduce inflammation, support joint health, and have cardiovascular benefits. 1–3 g of combined EPA + DHA per day is a reasonable dose from fish oil or algae-based supplements. If you eat fatty fish 2–3 times per week, supplementation may not be necessary."),
    ]
    for q, a in suppl_qa:
        samples.append(chat(q, a))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 5. SPORTS SCIENCE EDUCATION
# ─────────────────────────────────────────────────────────────────────────────

def gen_science_qa() -> list[dict]:
    qa_pairs = [
        ("What is progressive overload and why does it matter?",
         "Progressive overload means systematically increasing the training stimulus over time — more load, more reps, more sets, or improved technique. It is the single most important principle driving long-term strength and muscle gains. Without it, the body has no reason to adapt beyond its current capacity. Track at least one metric (weight or reps) each session to ensure you are progressing."),
        ("How many sets per muscle group per week for hypertrophy?",
         "Research supports 10–20 sets per muscle group per week as an effective hypertrophy range for most lifters. Beginners can make excellent progress at 8–12 sets. Intermediates typically benefit from 12–16 sets. Advanced lifters may push to 20+ sets for lagging groups. Start at the low end and add sets only when performance and recovery support it."),
        ("What is the best rep range for muscle growth?",
         "Hypertrophy can occur across a wide rep range (5–30+) when sets are taken close to failure. Moderate reps (8–15) are practical for most exercises. Heavy sets (3–5 reps) build strength that supports hypertrophy over time. High reps (15–30) are effective for isolation exercises and joint-friendly loading. The key variable is effort, not the rep range itself."),
        ("How often should I train each muscle group?",
         "Training a muscle group 2x per week produces more hypertrophy than 1x in most studies, primarily due to more frequent protein synthesis spikes. 3x per week provides a further small benefit. Higher frequencies require lower per-session volume to allow recovery. For most people, twice-weekly frequency (upper/lower or PPL) is the practical optimum."),
        ("What is mind-muscle connection and does it matter?",
         "Mind-muscle connection refers to consciously focusing attention on the target muscle during contraction. Research shows it increases EMG activation in isolated muscles. It matters most for isolation movements (bicep curls, lateral raises, cable flyes) and less for compound lifts where external load and technique drive adaptation. Develop it with lighter loads, slow eccentrics, and peak contraction holds."),
        ("How long should I rest between sets?",
         "For strength (1–5 reps): rest 3–5 minutes to allow full ATP recovery. For hypertrophy (6–15 reps): 60–120 seconds balances metabolic stress and performance. For endurance circuits (15+ reps): 30–60 seconds. Shorter rest increases metabolic stress but may compromise performance on subsequent sets. Prioritise performance over time efficiency for strength and hypertrophy goals."),
        ("What is RPE and how do I use it?",
         "RPE (Rate of Perceived Exertion) is a 1–10 scale measuring effort: RPE 10 = absolute maximum with no reps left. Reps in reserve (RIR) is the inverse: RIR 2 means 2 reps left. Training at RPE 7–9 (1–3 RIR) is optimal for hypertrophy and strength development. Beginners should stay at RPE 6–7 to learn technique. RPE autoregulates load to daily readiness."),
        ("Does the order of exercises in a session matter?",
         "Yes — place compound multi-joint exercises first when energy and neural drive are highest (e.g., squat before leg extension). This allows maximum load and quality on the most impactful movements. Isolation exercises can be performed later with less performance cost. Exceptions: pre-exhaust techniques intentionally place isolation before compound."),
        ("Is cardio bad for muscle gains?",
         "Cardio does not inherently impair muscle growth if programmed correctly. Interference effects are real but modest when cardio volume is controlled and recovery is adequate. 2–3 sessions of 20–40 min low-to-moderate intensity cardio per week supports cardiovascular health without meaningfully blunting hypertrophy. Avoid high-volume leg cardio before heavy leg sessions."),
        ("What is a deload week and when should I take one?",
         "A deload is a planned reduction in training volume (40–60%) while maintaining movement patterns. It allows dissipation of accumulated fatigue, so the underlying fitness adaptations can express themselves — this is called supercompensation. Signs you need a deload: stagnating performance, elevated soreness, disrupted sleep, reduced motivation, and declining technique quality. Typical timing: every 4–8 weeks of progressive training."),
        ("Why is sleep important for muscle growth?",
         "The majority of growth hormone secretion occurs during deep sleep. Muscle protein synthesis is elevated during sleep. Cortisol (a catabolic hormone) rises with sleep deprivation, blunting anabolic signalling. Even one night of poor sleep measurably reduces strength and training performance. 7–9 hours per night is the minimum target for optimal recovery and gains."),
        ("What is the difference between training volume, intensity, and frequency?",
         "Volume: total workload (sets × reps × load). Intensity: the load relative to maximum (% 1RM or RPE). Frequency: how often a muscle is trained per week. All three interact — you cannot simultaneously maximise all three. Increasing volume requires either reducing intensity or frequency. Effective program design balances these three variables based on training experience and recovery capacity."),
        ("How do I know if I am overtraining?",
         "True overtraining syndrome is rare. More common is functional overreaching — short-term performance decline from excessive load without enough recovery. Symptoms: persistent performance decline over 2+ weeks, excessive soreness, elevated resting heart rate, sleep disruption, mood changes, and decreased motivation. Address it with a deload or 1-week active recovery before resuming progressive training."),
    ]
    for q, a in qa_pairs:
        samples = []
        p = rp()
        prefix = random.choice([
            f"[{p['level'].title()}, {GOAL_TEXT[p['goal']]}] ",
            f"",
            f"",
        ])
        samples.append(chat(prefix + q, a))
    return [chat(prefix + q, a) for q, a in qa_pairs for prefix in [random.choice(["", f"[{rp()['level']}, {GOAL_TEXT[rp()['goal']]}] "])]]


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAFETY & INJURY GUIDANCE
# ─────────────────────────────────────────────────────────────────────────────

def gen_safety_qa() -> list[dict]:
    samples = []
    safety_pairs = [
        ("I feel sharp knee pain when squatting. What should I do?",
         "Stop squatting immediately and assess when the pain occurs — at depth, during the ascent, or with certain foot positions. Sharp joint pain during a lift is a red flag; unlike muscle soreness, it should not be trained through. Common causes include excessive forward knee travel, poor hip mobility, or patellar tracking issues. If pain persists for more than a few days or worsens with rest, see a sports medicine physician or physiotherapist. In the meantime, switch to pain-free alternatives like leg press, goblet squat with limited depth, or step-ups."),
        ("I have lower back pain after deadlifts. Is something wrong?",
         "Mild lower back fatigue after deadlifts is common and usually reflects the erectors working under load. However, sharp, acute pain, pain that radiates down the leg, or pain that worsens over 48 hours requires medical evaluation. Review your technique: common causes include lumbar flexion under load, hyperextension, bar drift, or excessive load relative to your capacity. Take a full week off heavy loading, address technique with lighter weights, and consider working with a coach. Never push through neurological symptoms like tingling or numbness."),
        ("Is it safe to train with a sore muscle?",
         "Delayed onset muscle soreness (DOMS) is not an injury — it reflects normal microscopic muscle damage from training. You can train a sore muscle if you can move through full range of motion without sharp pain. However, training extremely sore muscles at high intensity can reduce performance and increase injury risk. Reduce load by 20–30%, use technique-focused sets, and allow the muscle to recover if soreness is severe (3+ days post-session)."),
        ("I have shoulder impingement. Which exercises should I avoid?",
         "With shoulder impingement, avoid exercises that provoke pain in the impingement arc — typically 70–120° of shoulder abduction. Common culprits: behind-the-neck press, wide-grip upright row, overhead pressing with a narrow grip, and lateral raises above shoulder height. Safer alternatives include neutral-grip pressing, landmine press, cable external rotation, face pulls, and serratus anterior work. Work with a physiotherapist to address the underlying cause (rotator cuff weakness, scapular dyskinesis, or mobility restrictions)."),
        ("Can I still train while sick?",
         "For mild above-the-neck symptoms (runny nose, mild sore throat, no fever), low-intensity training is generally safe and may not prolong illness. For below-the-neck symptoms (fever, chest congestion, body aches, gastrointestinal illness), complete rest is strongly recommended — training with these symptoms can worsen illness and risks serious complications including myocarditis. When in doubt, rest. A missed session has no meaningful impact on long-term progress; a serious illness does."),
        ("What is the safest way to warm up before heavy lifting?",
         "Start with 5–10 minutes of general cardiovascular warm-up (light bike, brisk walk) to elevate core temperature and increase joint lubrication. Follow with mobility work for the joints you will load. Then perform exercise-specific warm-up sets: 1 set at 50% of working weight, 1 set at 70%, 1 set at 85%, then your working sets. For compound lifts over 80% 1RM, never skip warm-up sets — they prime the nervous system and reduce injury risk significantly."),
        ("I'm a beginner — how do I avoid injury in the gym?",
         "The most important rules for beginners: learn technique before adding load, progress weight gradually (2.5–5 kg per session for beginners is appropriate), always warm up, avoid training to complete failure on compound lifts, get at least 7 hours sleep, and allow 48 hours between sessions targeting the same muscle group. Most gym injuries in beginners result from ego lifting, skipping warm-ups, or ignoring early warning signs. Technique sessions with a coach for the main lifts (squat, deadlift, bench press) are an excellent investment."),
    ]
    for q, a in safety_pairs:
        samples.append(chat(q, a))

    # Injury-specific modifications
    injury_qa = [
        ("bad knees", "squat", "goblet squat with limited depth, leg press with feet high on platform, step-ups, hip thrust. Avoid deep knee flexion under load until you are pain-free."),
        ("lower back issues", "deadlift", "trap bar deadlift, Romanian deadlift with lighter load, back extension machine, hip thrust. Focus on hip hinge mechanics and spinal neutrality."),
        ("shoulder impingement", "bench press", "neutral-grip dumbbell press, landmine press, push-up with serratus emphasis. Avoid wide grip and decline angle."),
        ("wrist pain", "barbell movements", "use wrist wraps, switch to dumbbells or neutral-grip bars, and address wrist mobility. Wrist pain often comes from lack of mobility or poor bar path."),
        ("elbow tendinopathy", "pulling exercises", "reduce load, use reverse-grip options, add wrist extensor stretching and eccentric exercises for the forearm. Do not train through sharp elbow pain."),
    ]
    for condition, movement, alternatives in injury_qa:
        p = rp()
        q = random.choice([
            f"I have {condition} — what should I do instead of {movement}?",
            f"With {condition}, can I still do {movement}? If not, what alternatives are safe?",
            f"Modifications for someone with {condition} who wants to train {movement}?",
        ])
        answer = (
            f"With {condition}, avoid loading the {movement} through painful ranges. "
            f"Safe alternatives that maintain the training stimulus: {alternatives} "
            f"Always work within a pain-free range of motion. If pain persists or worsens, consult a sports medicine professional or physiotherapist before continuing."
        )
        samples.append(chat(q, answer))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 7. CARDIO GUIDANCE from longhaul_fitness/cardio.json
# ─────────────────────────────────────────────────────────────────────────────

def gen_cardio_guidance() -> list[dict]:
    path = ROOT / "longhaul_fitness" / "cardio.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        cardio_list = json.load(f)
    samples = []
    for ex in cardio_list:
        name = str(ex.get("name", "")).strip()
        instructions = ""
        if isinstance(ex.get("instructions"), list):
            instructions = " ".join(str(i) for i in ex["instructions"])
        elif isinstance(ex.get("instructions"), str):
            instructions = ex["instructions"]
        if not name or not instructions:
            continue
        p = rp()
        goal = p["goal"]
        intensity_note = (
            "For fat loss, aim for Zone 2 (conversational pace) 3x/week plus 1–2 HIIT sessions."
            if goal == "weight_loss"
            else "For strength athletes, keep cardio at low-to-moderate intensity to avoid interfering with strength adaptations."
            if goal == "strength"
            else "For general fitness, 150 min/week of moderate cardio satisfies health guidelines."
        )
        q = random.choice([
            f"How should I incorporate {name} into my {GOAL_TEXT[goal]} routine?",
            f"Best way to use {name} as a {p['level']} for {GOAL_TEXT[goal]}?",
            f"Tell me about {name} — how to do it and how to program it.",
        ])
        answer = (
            f"{name} is a cardiovascular exercise: {instructions[:350].strip()} "
            f"{intensity_note} "
            f"Always warm up for 5 minutes at easy pace before reaching your target intensity. "
            f"Track duration and subjective effort to ensure progressive overload."
        )
        samples.append(chat(q, answer))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 8. FLEXIBILITY & MOBILITY from longhaul_fitness/flexibility.json
# ─────────────────────────────────────────────────────────────────────────────

def gen_flexibility_guidance() -> list[dict]:
    path = ROOT / "longhaul_fitness" / "flexibility.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        flex_list = json.load(f)
    samples = []
    for ex in flex_list:
        name = str(ex.get("name", "")).strip()
        steps = ex.get("steps", []) or []
        if isinstance(steps, list):
            instructions = " ".join(str(s) for s in steps)
        else:
            instructions = str(steps)
        primary = ex.get("primaryMuscles", [])
        if not name or not instructions:
            continue
        muscle_str = ", ".join(primary) if primary else "the target area"
        p = rp()
        q = random.choice([
            f"Walk me through the {name} stretch.",
            f"How do I perform {name} correctly?",
            f"Add {name} to my warm-up or cool-down? How?",
        ])
        answer = (
            f"{name} targets flexibility in {muscle_str}. {instructions[:400].strip()} "
            f"Hold each stretch for 20–30 seconds (static) or 30–60 seconds (deep tissue). "
            f"Perform after training or as a dedicated mobility session. "
            f"Never bounce in a static stretch — use controlled breathing to relax into the position."
        )
        samples.append(chat(q, answer))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 9. WORKOUT PLAN TEMPLATES GUIDANCE
# ─────────────────────────────────────────────────────────────────────────────

def gen_workout_plan_qa() -> list[dict]:
    samples = []
    plan_qa = [
        ("Give me a 3-day beginner full body program.",
         "Here is a 3-day full body beginner program (A/B/C rotation):\n\n"
         "Day A: Squat 3x8-10 | Bench Press 3x8-10 | Bent-over Row 3x8-10 | Overhead Press 3x8-10 | Romanian Deadlift 3x10-12\n"
         "Day B: Deadlift 3x5 | Incline Dumbbell Press 3x10-12 | Pull-Up or Lat Pulldown 3x8-10 | Lunges 3x10/leg | Lateral Raise 3x12-15\n"
         "Day C: Squat 3x10 | Cable Row 3x10-12 | Dumbbell Shoulder Press 3x10 | Hip Thrust 3x10-12 | Tricep Pushdown 3x12 | Bicep Curl 3x12\n\n"
         "Rest 60–90 seconds between sets. Add 2.5 kg per week on main lifts when all sets are completed with good form."),
        ("Give me a 4-day upper/lower intermediate hypertrophy program.",
         "4-Day Upper/Lower Split (Intermediate Hypertrophy):\n\n"
         "Day 1 – Upper A (Heavy): Bench Press 4x5-8 | Barbell Row 4x5-8 | OHP 3x6-8 | Pull-Up 3x6-8 | Incline DB Press 3x10 | Face Pull 3x15\n"
         "Day 2 – Lower A (Heavy): Squat 4x5-8 | Romanian Deadlift 3x8-10 | Leg Press 3x10-12 | Leg Curl 3x10-12 | Calf Raise 4x12-15\n"
         "Day 3 – Upper B (Hypertrophy): Incline Barbell Press 4x8-12 | Cable Row 4x10-12 | Lateral Raise 4x12-15 | Lat Pulldown 3x10-12 | Tricep Dip 3x10-12 | Barbell Curl 3x10-12\n"
         "Day 4 – Lower B (Hypertrophy): Hack Squat 4x10-12 | Deadlift 3x5 | Walking Lunge 3x12/leg | Leg Extension 3x12-15 | Hip Thrust 3x12 | Seated Calf Raise 4x15\n\n"
         "Rest 2 min on heavy sets, 60–90 sec on hypertrophy sets. Progress by adding reps first, then load."),
        ("I only have dumbbells at home. Give me a 3-day full body program.",
         "3-Day Dumbbell-Only Full Body Program:\n\n"
         "Day 1 – Push/Legs: Goblet Squat 4x10-12 | DB Bench Press 4x10-12 | DB Shoulder Press 3x10-12 | DB Lateral Raise 3x15 | DB Tricep Kickback 3x12 | Calf Raise 3x15\n"
         "Day 2 – Pull/Hinge: DB Romanian Deadlift 4x10-12 | DB Row 4x10-12 | DB Rear Delt Fly 3x15 | DB Bicep Curl 3x12 | DB Hammer Curl 3x12 | Plank 3x30-45s\n"
         "Day 3 – Full Body: DB Lunge 3x10/leg | DB Incline Press 3x10-12 | DB Single-Arm Row 3x10 | DB Hip Thrust 3x12 | DB Arnold Press 3x12 | DB Curl to Press 3x12\n\n"
         "Progressive overload: increase by 1 rep per session; when you hit the top of the range, move to heavier dumbbells."),
        ("What is the best program for fat loss?",
         "The best fat loss program is one that preserves muscle mass while creating a consistent calorie deficit. Use resistance training 3x/week full body or 4x upper/lower with compound movements at moderate-high rep ranges (10–15). Add 2–3 cardio sessions (20–30 min Zone 2 or HIIT). Set calories at a 400–500 kcal deficit. Keep protein at 2.0–2.4 g/kg. Avoid crash dieting — slow, progressive fat loss (0.5–1% bodyweight/week) preserves the most muscle."),
        ("How do I progress from a beginner program to an intermediate one?",
         "Transition when linear progression (adding weight every session) becomes unreliable — usually 3–6 months for most lifters. Signs you're ready: you can no longer add weight each session on main lifts, recovery takes more than 48 hours, and your technique on compound movements is solid. Move from full-body 3x to upper/lower 4x or PPL 6x. Introduce wave loading (e.g., week 1: 4x8, week 2: 4x9, week 3: 4x10, week 4: 4x8 with more weight) instead of simple linear progression."),
    ]
    for q, a in plan_qa:
        samples.append(chat(q, a))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 10. LOAD EXISTING HIGH-QUALITY DATASETS (filtered)
# ─────────────────────────────────────────────────────────────────────────────

FITNESS_KEYWORDS = re.compile(
    r"(exercise|workout|muscle|strength|hypertrophy|protein|calorie|rep|set|squat|deadlift|"
    r"bench|pull|push|cardio|fat|weight|fitness|training|coach|lunge|curl|press|row|"
    r"nutrition|recovery|sleep|deload|progressive|overload|mobility|flexibility|injury|"
    r"supplement|creatine|sprint|endurance|glute|quad|hamstring|chest|back|shoulder|bicep|"
    r"tricep|calf|core|abs|hinge|compound|isolation|program|split|volume|frequency)",
    re.IGNORECASE
)


def is_fitness_sample(record: dict) -> bool:
    """Return True if this sample is clearly fitness-domain."""
    msgs = record.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) != 3:
        return False
    roles = [m.get("role") for m in msgs]
    if roles != ["system", "user", "assistant"]:
        return False
    user_text = msgs[1].get("content", "")
    assistant_text = msgs[2].get("content", "")
    if len(assistant_text.strip()) < 40:
        return False
    combined = user_text + " " + assistant_text
    return bool(FITNESS_KEYWORDS.search(combined))


def load_existing_datasets() -> list[dict]:
    existing: list[dict] = []
    sources = [
        ROOT / "artifacts" / "datasets" / "fitness_training_data_full.jsonl",
        ROOT / "artifacts" / "datasets" / "full_llm_training_data.jsonl",
        ROOT / "artifacts" / "datasets" / "fitness_training_data.jsonl",
        ROOT / "artifacts" / "datasets" / "nutrition_textbook_full_qa.jsonl",
        ROOT / "artifacts" / "datasets" / "nutrition_textbook_data.jsonl",
        ROOT / "artifacts" / "datasets" / "combined_training_data.jsonl",
        ROOT / "fitness_dataset" / "output" / "fitness_training_data_full.jsonl",
        ROOT / "fitness_dataset" / "output" / "hf_fitness_qa.jsonl",
        ROOT / "fitness_dataset" / "output" / "hf_fitness_qa2.jsonl",
        ROOT / "fitness_dataset" / "output" / "hf_generate_workouts.jsonl",
        ROOT / "fitness_dataset" / "output" / "hf_fine_corpus.jsonl",
        ROOT / "fitness_dataset" / "output" / "kaggle_samples.jsonl",
        ROOT / "fitness_dataset" / "output" / "longhaul_samples.jsonl",
        ROOT / "fitness_dataset" / "output" / "manual_samples.jsonl",
        ROOT / "fitness_dataset" / "output" / "wger_samples.jsonl",
        ROOT / "fitness_dataset" / "output" / "exercisedb_guidance.jsonl",
        ROOT / "fitness_dataset" / "output" / "exercisedb_substitutions.jsonl",
    ]
    seen = set()
    total_loaded = 0
    total_filtered = 0
    for src in sources:
        records = _load_jsonl(src)
        total_loaded += len(records)
        for rec in records:
            if not is_fitness_sample(rec):
                total_filtered += 1
                continue
            msgs = rec["messages"]
            key = _hash(msgs[1].get("content", "")[:100] + msgs[2].get("content", "")[:100])
            if key not in seen:
                seen.add(key)
                existing.append(rec)
    print(f"  Loaded {total_loaded} existing samples, kept {len(existing)} unique fitness samples (filtered {total_filtered} off-topic)")
    return existing


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading master exercise dictionary...")
    ex_dict_path = DICT_DIR / "master_exercise_dictionary.jsonl"
    if not ex_dict_path.exists():
        print("  master_exercise_dictionary.jsonl not found — run script 11 first.")
        exercises = []
    else:
        exercises = []
        for line in ex_dict_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    exercises.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        print(f"  Loaded {len(exercises)} exercises from master dictionary")

    all_samples: list[dict] = []

    print("Generating exercise guidance samples...")
    guidance = gen_exercise_guidance(exercises)
    all_samples.extend(guidance)
    print(f"  Exercise guidance: {len(guidance)}")

    print("Generating substitution samples...")
    subs = gen_substitutions(exercises)
    all_samples.extend(subs)
    print(f"  Substitutions    : {len(subs)}")

    print("Generating programming Q&A...")
    prog = gen_programming_qa()
    all_samples.extend(prog)
    print(f"  Programming Q&A  : {len(prog)}")

    print("Generating nutrition Q&A...")
    nutr = gen_nutrition_qa()
    all_samples.extend(nutr)
    print(f"  Nutrition Q&A    : {len(nutr)}")

    print("Generating sports science Q&A...")
    sci = gen_science_qa()
    all_samples.extend(sci)
    print(f"  Science Q&A      : {len(sci)}")

    print("Generating safety & injury Q&A...")
    safe = gen_safety_qa()
    all_samples.extend(safe)
    print(f"  Safety Q&A       : {len(safe)}")

    print("Generating cardio guidance...")
    cardio = gen_cardio_guidance()
    all_samples.extend(cardio)
    print(f"  Cardio guidance  : {len(cardio)}")

    print("Generating flexibility guidance...")
    flex = gen_flexibility_guidance()
    all_samples.extend(flex)
    print(f"  Flexibility      : {len(flex)}")

    print("Generating workout plan Q&A...")
    plan_qa = gen_workout_plan_qa()
    all_samples.extend(plan_qa)
    print(f"  Workout plan Q&A : {len(plan_qa)}")

    print("Loading existing high-quality datasets (filtered)...")
    existing = load_existing_datasets()
    all_samples.extend(existing)

    print(f"\nTotal before dedup: {len(all_samples)}")

    # Deduplicate by user+assistant hash
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in all_samples:
        msgs = rec.get("messages", [])
        if len(msgs) < 3:
            continue
        key = _hash(msgs[1].get("content", "")[:120] + msgs[2].get("content", "")[:120])
        if key not in seen:
            seen.add(key)
            deduped.append(rec)

    # Shuffle
    random.shuffle(deduped)

    print(f"Total after dedup : {len(deduped)}")

    # Write full dataset
    full_out = OUT_DIR / "master_training_data.jsonl"
    with full_out.open("w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote: {full_out}  ({len(deduped)} samples)")

    # Write clean subset (first 20k if very large, else all)
    clean = deduped[:20000]
    clean_out = OUT_DIR / "master_training_data_clean.jsonl"
    with clean_out.open("w", encoding="utf-8") as f:
        for rec in clean:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote: {clean_out}  ({len(clean)} samples)")

    # Summary
    summary = {
        "total_samples": len(deduped),
        "clean_samples": len(clean),
        "breakdown": {
            "exercise_guidance": len(guidance),
            "substitutions": len(subs),
            "programming_qa": len(prog),
            "nutrition_qa": len(nutr),
            "science_qa": len(sci),
            "safety_qa": len(safe),
            "cardio_guidance": len(cardio),
            "flexibility_guidance": len(flex),
            "workout_plan_qa": len(plan_qa),
            "existing_datasets": len(existing),
        }
    }
    (OUT_DIR / "master_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Master dataset build complete.")


if __name__ == "__main__":
    main()
