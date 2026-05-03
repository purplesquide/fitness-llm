"""
Generate high-quality manual training samples covering:
  - Nutrition basics (protein, calories, timing, meal planning)
  - Progress & plateaus
  - Injury & recovery
  - App-specific concepts (RPE, splits, deloads, cardio)
  - Motivation & habit formation

Output:
  - output/manual_samples.jsonl : ~600 samples
"""

import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from constants import (
    SYSTEM_PROMPT, random_profile, GOAL_TEXT_MAP, REP_TABLE, OUTPUT_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# NUTRITION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

PROTEIN_TEMPLATES = [
    ("How much protein should I eat daily? I am {level}, {goal_txt}, weigh {weight}kg.",
     "For {goal_txt} at {weight}kg, target {pmin}–{pmax}g of protein per day ({pperkg}g/kg bodyweight). "
     "Spread it across 3-5 meals — each meal should contain 30-50g to maximally stimulate muscle protein synthesis. "
     "Prioritize whole food sources: chicken, eggs, fish, Greek yogurt, legumes. "
     "Consistency in hitting your daily total matters more than perfect meal timing."),

    ("I'm {weight}kg and training for {goal_txt}. What's my protein target?",
     "A strong daily protein target for {goal_txt} at {weight}kg is {pmin}–{pmax}g ({pperkg}g/kg). "
     "Research consistently shows this range maximizes muscle retention and growth. "
     "If you struggle to hit it through whole foods, a whey or plant protein shake is a practical supplement. "
     "Split intake across at least 3 meals for best results."),

    ("As a {level} aiming for {goal_txt}, how much protein per day?",
     "Target {pmin}–{pmax}g of protein per day at {weight}kg bodyweight. "
     "At {level} level, your muscles are highly responsive to training — adequate protein is critical to recover and grow. "
     "Aim for at least 3 protein-containing meals, with 30-50g per meal. "
     "Common high-protein foods: 100g chicken breast = 31g, 3 eggs = 18g, 200g Greek yogurt = 20g."),
]

PRE_POST_WORKOUT_TEMPLATES = [
    ("Should I eat before my workout? I train at {time_of_day}.",
     "Eating before training improves performance, especially for sessions over 45 minutes. "
     "Aim for 30-50g of carbohydrates and 20-30g of protein 60-90 minutes pre-workout. "
     "If training early morning and a full meal is impractical, a banana and protein shake 30 minutes before is enough. "
     "Never train in a fully fasted state for strength sessions — glycogen availability directly affects output."),

    ("What should I eat after my workout? Goal: {goal_txt}.",
     "Post-workout nutrition targets two things: replenish glycogen (carbohydrates) and initiate repair (protein). "
     "For {goal_txt}, aim for 30-50g protein and 50-80g carbohydrates within 90 minutes of finishing. "
     "Simple options: rice and chicken, or a protein shake with fruit. "
     "The anabolic window is longer than gym mythology suggests — any quality meal within 2 hours post-workout is fine."),

    ("Is it bad to train fasted in the morning?",
     "Training fasted is fine for low-intensity cardio or very short sessions, but it hurts heavy strength work. "
     "Muscle glycogen drops during sleep, and fasted strength sessions produce lower output and higher cortisol. "
     "A small pre-workout snack (30-40g carbs + 15-20g protein) takes 10 minutes to prepare and significantly improves performance. "
     "If fat loss is the goal, fasted cardio has minimal advantage over fed-state cardio when total daily intake is matched."),
]

CALORIE_TEMPLATES = [
    ("How many calories should I eat for {goal_txt}?",
     "Calorie targets depend on your TDEE (Total Daily Energy Expenditure), which varies by weight, age, and activity level. "
     "For {goal_txt}: {calorie_rec}. "
     "Track your intake for two weeks — if weight changes faster or slower than expected, adjust by 100-200 calories. "
     "Avoid aggressive deficits over 750 calories per day, as they accelerate muscle loss alongside fat."),

    ("I've been eating at maintenance but not losing weight. What should I do?",
     "If body weight isn't moving after 2 consistent weeks, your estimated maintenance is higher than your actual TDEE, or tracking has gaps. "
     "The most common cause is underestimating portion sizes — cooking oils, sauces, and snacks are frequently missed. "
     "Drop intake by 200 calories for 2 weeks and reassess. "
     "A food scale is the single most effective tool for accurate tracking."),

    ("What macros should I follow for {goal_txt}?",
     "A practical macro breakdown for {goal_txt}: protein 30-35% of calories, carbohydrates 40-50%, fats 20-30%. "
     "Protein is the highest priority — hit {pmin}-{pmax}g/day first, then fill remaining calories with carbs and fats based on preference. "
     "High-carbohydrate diets support training performance; higher-fat diets are sustainable for some individuals. "
     "Both work — adherence is more important than the exact split."),
]

def build_nutrition_samples(count: int = 200) -> list:
    samples = []
    all_templates = PROTEIN_TEMPLATES + PRE_POST_WORKOUT_TEMPLATES + CALORIE_TEMPLATES

    while len(samples) < count:
        template = random.choice(all_templates)
        profile  = random_profile()
        goal     = profile["goal"]
        goal_txt = GOAL_TEXT_MAP.get(goal, goal.replace("_", " "))
        weight   = profile["weight"]
        level    = profile["level"]

        # Protein calc
        pperkg = 2.0 if goal == "muscle_gain" else 1.8 if goal == "strength" else 1.6
        pmin   = round(weight * (pperkg - 0.2))
        pmax   = round(weight * pperkg)

        # Calorie rec
        calorie_recs = {
            "weight_loss":     "eat at a deficit of 300-500 calories below your TDEE",
            "muscle_gain":     "eat at a surplus of 200-350 calories above your TDEE",
            "strength":        "eat at maintenance or a slight surplus of 150-250 calories above TDEE",
            "endurance":       "eat at maintenance with carbohydrate focus (55-65% of calories)",
            "general_fitness": "eat at maintenance — focus on food quality over calorie manipulation",
        }
        calorie_rec = calorie_recs.get(goal, "eat at maintenance")

        time_of_day = random.choice(["6am", "7am", "noon", "5pm", "6pm", "8pm"])

        try:
            q = template[0].format(
                level=level, goal=goal, goal_txt=goal_txt, weight=weight,
                pmin=pmin, pmax=pmax, pperkg=round(pperkg, 1),
                time_of_day=time_of_day, calorie_rec=calorie_rec,
            )
            a = template[1].format(
                level=level, goal=goal, goal_txt=goal_txt, weight=weight,
                pmin=pmin, pmax=pmax, pperkg=round(pperkg, 1),
                time_of_day=time_of_day, calorie_rec=calorie_rec,
            )
        except KeyError:
            continue

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    return samples[:count]


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS & PLATEAU SAMPLES
# ─────────────────────────────────────────────────────────────────────────────

PROGRESS_QA = [
    (
        "I have been stuck at the same bench press weight for 3 weeks. What should I do?",
        "A 3-week plateau is the body's normal signal to vary the stimulus. "
        "First audit your sleep, nutrition, and stress — these account for 80% of plateaus before programming is the issue. "
        "If recovery is solid, take a deload week at 50-60% of usual weight, then return. "
        "Alternatively, shift rep ranges: if you have been doing 3x8, try 5x5 or 4x12 for 3 weeks — strength gains transfer back."
    ),
    (
        "When should I increase the weight on an exercise?",
        "Add weight when you complete all prescribed reps with 2+ reps in reserve (RPE 7-8) for two consecutive sessions. "
        "Standard jumps: 2.5kg for upper body lifts, 5kg for lower body. "
        "If weight jumps cause form breakdown, use rep progression instead — add one rep per set weekly until you hit the top of your range, then add weight and reset reps."
    ),
    (
        "I'm not seeing results after 2 months. What is wrong?",
        "Two months is often too early for visible changes if you started at a normal body composition. "
        "The most common issues are inconsistent protein intake, not tracking progressive overload, and sleep under 7 hours. "
        "Check your training log — if your lift numbers aren't going up session to session, the stimulus isn't there. "
        "Visible physique change typically becomes noticeable at 3-4 months of consistent, progressive training."
    ),
    (
        "What is a deload week and when do I need one?",
        "A deload week is a planned period of reduced volume and intensity (40-60% of normal) to allow the nervous system and connective tissue to recover. "
        "Beginners: deload after every 8-12 weeks. Intermediate/Advanced: every 4-6 weeks, or when you notice persistent fatigue, stalled strength, motivation drops, and disturbed sleep together. "
        "Do not skip deloads — accumulated fatigue masks fitness, and the week after a deload is often when you set personal records."
    ),
    (
        "My squat numbers went down this week. Should I worry?",
        "Single-session dips in strength are almost always from recovery factors, not a training problem. "
        "Common causes: poor sleep the previous night, incomplete glycogen replenishment, or accumulated fatigue from the past week. "
        "If your numbers are generally trending upward over months, one off day is not a concern. "
        "If performance has been flat or declining for 2-3 weeks consistently, reassess your recovery and program structure."
    ),
    (
        "How do I know if I'm overtraining?",
        "True overtraining syndrome is rare — what most people experience is under-recovery. "
        "Warning signs: persistent performance decline over 2+ weeks, elevated resting heart rate, sleep disturbances, mood changes, and chronic joint soreness. "
        "Before cutting volume, first assess sleep quality, calorie intake, and stress levels outside the gym. "
        "Most 'overtraining' resolves with a deload week and improved recovery habits."
    ),
    (
        "Should I train to failure every set?",
        "Training to failure occasionally is useful as a test, but not optimal as a daily practice. "
        "The highest quality reps are those done before failure — sets ending at RPE 7-8 (2 reps in reserve) maximize stimulus while minimizing cumulative fatigue. "
        "For compound movements like squats and deadlifts, stopping 2-3 reps short of failure protects form and reduces injury risk. "
        "Reserve true failure for isolation exercises at the end of a session."
    ),
    (
        "How long does it take to see muscle gain?",
        "Visible muscle gain typically takes 3-4 months of consistent, progressive resistance training with adequate protein. "
        "Strength gains appear faster — often within 2-4 weeks — due to neural adaptations, not just muscle growth. "
        "In the first year, beginners can realistically gain 1-2kg of muscle per month under optimal conditions. "
        "Intermediates and advanced lifters progress significantly slower — 500g to 1kg per month is a reasonable expectation."
    ),
]

def build_progress_samples(count: int = 120) -> list:
    samples = []
    pool = PROGRESS_QA * (count // len(PROGRESS_QA) + 2)
    for qa in pool[:count]:
        profile  = random_profile()
        level    = profile["level"]
        goal     = profile["goal"]
        goal_txt = GOAL_TEXT_MAP.get(goal, goal.replace("_", " "))
        user_txt = f"[{level.capitalize()}, {goal_txt}, {profile['days']} days/week] {qa[0]}"
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_txt},
                {"role": "assistant", "content": qa[1]},
            ]
        })
    return samples[:count]


# ─────────────────────────────────────────────────────────────────────────────
# INJURY & RECOVERY SAMPLES
# ─────────────────────────────────────────────────────────────────────────────

INJURY_QA = [
    (
        "I have a shoulder injury. What replaces overhead press?",
        "Remove painful overhead pressing until the movement is pain-free. "
        "A landmine press is the best replacement — the angle is more shoulder-friendly while still training delts and triceps. "
        "Light lateral raises can maintain shoulder stimulus without heavy compression. "
        "If the pain is sharp or persists beyond a week of rest, stop loading the joint and get it evaluated."
    ),
    (
        "My lower back hurts after deadlifts. What should I do?",
        "Acute low back discomfort after deadlifts is usually muscular fatigue, not injury — but sharp, shooting, or one-sided pain warrants medical evaluation before returning to loading. "
        "For muscular fatigue: rest 48-72 hours, then return with 60-70% of previous weight and emphasize bracing and hip hinge mechanics. "
        "Romanian deadlifts and trap bar deadlifts are often more lower-back friendly than conventional pulls. "
        "A weak core or faulty hip hinge pattern is the root cause in most cases — address it directly."
    ),
    (
        "I have knee pain during squats. Should I stop squatting?",
        "It depends on the pain type and location. Anterior knee pain (below the kneecap) during squats often means the knee is traveling too far forward or load is too high. "
        "Try a box squat or goblet squat with intentionally slower descent and reduced depth to see if it resolves. "
        "Lateral or medial pain warrants evaluation before continuing. "
        "Never squat through sharp, locking, or swelling-causing pain — reduce or remove the load and address the cause first."
    ),
    (
        "How long should I rest after a muscle strain?",
        "Minor muscle strains (Grade 1) typically resolve in 3-7 days with light activity and adequate protein. "
        "Grade 2 strains (partial tear) require 2-6 weeks of modified training, avoiding full range of motion until pain-free. "
        "Grade 3 (complete tear) requires medical evaluation and potentially surgical intervention. "
        "In all cases: maintain movement on unaffected muscle groups, prioritize nutrition, and return to loading only when the strain is pain-free through full range."
    ),
    (
        "Can I train with DOMS (delayed onset muscle soreness)?",
        "Training with DOMS is generally safe — soreness is not an injury. "
        "Light activity and lower-intensity work on sore muscles actually speeds recovery by increasing blood flow. "
        "Avoid high-intensity or maximum-effort work on a muscle group until soreness is below a 4/10 discomfort level. "
        "If DOMS is consistently severe (5/10+) after every session, reduce total volume by 20-30% and let adaptation catch up."
    ),
    (
        "I have tight hip flexors. What should I do?",
        "Tight hip flexors are extremely common with prolonged sitting and respond well to a combination of stretching and strengthening. "
        "Daily hip flexor stretches (kneeling lunge hold, 60 seconds per side) and hip circles help maintain mobility. "
        "Strengthen the antagonists — glutes and core — because weakness there forces hip flexors to compensate. "
        "Long-term, the most effective fix is breaking up prolonged sitting with standing breaks every 45-60 minutes."
    ),
]

def build_injury_samples(count: int = 100) -> list:
    samples = []
    pool = INJURY_QA * (count // len(INJURY_QA) + 2)
    for qa in pool[:count]:
        profile  = random_profile()
        user_txt = f"[{profile['level'].capitalize()}] {qa[0]}"
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_txt},
                {"role": "assistant", "content": qa[1]},
            ]
        })
    return samples[:count]


# ─────────────────────────────────────────────────────────────────────────────
# APP-SPECIFIC & CONCEPT SAMPLES
# ─────────────────────────────────────────────────────────────────────────────

APP_QA = [
    (
        "What does RPE mean in my workout plan?",
        "RPE stands for Rate of Perceived Exertion — a 1-10 scale where 10 is maximum effort (cannot do one more rep). "
        "RPE 6: very easy, 4+ reps in reserve. RPE 7: moderate, 3 reps in reserve. "
        "RPE 8: challenging but manageable, 2 reps in reserve — this is the target for most working sets. "
        "RPE 9: 1 rep in reserve. RPE 10: absolute maximum effort — use only for testing, not regular training."
    ),
    (
        "What is the difference between compound and isolation exercises?",
        "Compound exercises involve multiple joints and muscle groups simultaneously: squats, deadlifts, bench press, rows, overhead press. "
        "Isolation exercises target one joint and primary muscle: bicep curls, leg extensions, lateral raises, calf raises. "
        "Compounds provide the highest training stimulus per unit of time and should form the foundation of every program. "
        "Isolations add volume to lagging muscles. Standard ratio: 60-70% compound volume, 30-40% isolation."
    ),
    (
        "Should I do cardio on rest days?",
        "Light cardio on rest days — walking, cycling at Zone 2 (can hold a conversation), or swimming — actively aids recovery. "
        "20-40 minutes of Zone 2 cardio 2-3x per week improves cardiovascular base and speeds metabolic recovery. "
        "Avoid high-intensity cardio on rest days from strength work — it adds cumulative fatigue and competes with strength adaptation. "
        "Zone 2 cardio is the optimal rest-day choice: enough stimulus to improve aerobic capacity without impairing recovery."
    ),
    (
        "How do I know if my form is correct?",
        "The fastest method: record yourself from the side and front, then compare to a qualified reference. "
        "Key checkpoints: neutral spine throughout the range of motion, joints tracking over feet or wrists as appropriate, controlled tempo (especially the lowering phase), and full intended range of motion. "
        "Pain during a movement — not muscular effort, but joint pain — is always a form or load issue. "
        "Drop the weight 20-30% and rebuild the pattern before increasing load again."
    ),
    (
        "What is progressive overload and why does it matter?",
        "Progressive overload is the principle of gradually increasing the training stimulus to continually challenge your muscles to adapt. "
        "The most common application is adding weight to the bar, but you can also increase reps, sets, decrease rest time, or improve range of motion. "
        "Without progressive overload, the body has no reason to add muscle or strength — it only adapts to demands that exceed previous ones. "
        "Track your lifts session by session — if the numbers aren't moving, neither is your fitness."
    ),
    (
        "How important is sleep for muscle building?",
        "Sleep is when the majority of muscle repair and growth occurs through growth hormone release and protein synthesis. "
        "Studies consistently show that 7-9 hours of sleep is associated with better strength gains, lower cortisol, and better body composition. "
        "Under 6 hours of sleep dramatically increases muscle catabolism and reduces anabolic hormone output. "
        "If training and nutrition are consistent but results are poor, sleep quality is the first variable to audit."
    ),
    (
        "What is a warm-up and do I really need one?",
        "A warm-up serves two purposes: elevating core temperature to improve muscle elasticity, and rehearsing the movement patterns you are about to load. "
        "A 5-10 minute general warm-up (light cardio or dynamic mobility) followed by progressive warm-up sets at 50-70% of your working weight is the minimum standard. "
        "Skipping warm-ups increases injury risk and reduces peak performance, especially in cold environments. "
        "The more complex or heavy the movement (squats, deadlifts), the more important the warm-up becomes."
    ),
    (
        "How long should I rest between sets?",
        "Rest periods depend on the training goal. "
        "For strength (3-6 rep range): 3-5 minutes to allow full phosphocreatine system recovery. "
        "For hypertrophy (8-12 reps): 60-120 seconds — enough to maintain performance without fully dissipating the metabolic stress. "
        "For endurance (15-25 reps or circuits): 30-60 seconds. "
        "Most people under-rest, which reduces the quality of subsequent sets and limits progressive overload potential."
    ),
    (
        "Is it better to do full body or split workouts?",
        "Neither is universally better — the best program is the one you can execute consistently with progressive overload. "
        "Full body training (3x/week) maximizes training frequency per muscle group, which is optimal for beginners and those with limited training days. "
        "Splits (Upper/Lower, Push/Pull/Legs) allow more total volume per session and per muscle group, making them better for intermediate and advanced lifters with 4+ training days. "
        "A full body 3x/week is generally the strongest starting point for most people."
    ),
    (
        "What is mind-muscle connection and does it matter?",
        "Mind-muscle connection refers to consciously focusing on the target muscle contracting during an exercise. "
        "Research shows it increases EMG activation in the targeted muscle, particularly for isolation exercises. "
        "It matters most for isolation work (bicep curls, lateral raises, cable flyes) and less so for compound movements where load and technique are the primary drivers. "
        "Developing it requires lighter loads and deliberate practice — slow eccentrics and squeezing at the peak contraction help significantly."
    ),
    (
        "What supplements are actually worth taking?",
        "The evidence-based short list: creatine monohydrate (3-5g daily, one of the most researched supplements in sports science), protein powder (only if you cannot hit protein targets through food), and caffeine (3-6mg/kg pre-workout for performance). "
        "Vitamin D3 is worth supplementing if you live in a low-sunlight environment or work indoors. "
        "Everything else — pre-workouts, BCAAs, fat burners — has weak evidence or is redundant if protein targets are met. "
        "No supplement replaces consistent training, adequate protein, and quality sleep."
    ),
    (
        "How do I stay consistent with my training?",
        "Consistency comes from lowering the activation energy of training, not motivation. "
        "Set a fixed training schedule and treat it as non-negotiable as a work meeting. "
        "Have your gym bag packed the night before, use the same gym time daily, and prioritize 'good enough' sessions over skipping when time is tight. "
        "Track your workouts — seeing a streak of completed sessions and improving numbers is a powerful intrinsic motivator that compounds over time."
    ),
    (
        "Should I do abs every day?",
        "The abdominals are muscles and recover like any other — training them every day does not allow adequate recovery. "
        "2-3 dedicated core sessions per week is optimal. "
        "Additionally, the core is heavily recruited in heavy compound movements (squats, deadlifts, overhead press), so your abs are being trained indirectly on every strength day. "
        "Focus on quality progressive core work (deadbugs, pallof press, ab wheel rollouts) over high-rep crunches."
    ),
]

def build_app_samples(count: int = 130) -> list:
    samples = []
    pool = APP_QA * (count // len(APP_QA) + 2)
    for qa in pool[:count]:
        profile  = random_profile()
        level    = profile["level"]
        goal_txt = GOAL_TEXT_MAP.get(profile["goal"], profile["goal"].replace("_", " "))
        user_txt = f"[{level.capitalize()}, {goal_txt}] {qa[0]}"
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_txt},
                {"role": "assistant", "content": qa[1]},
            ]
        })
    return samples[:count]


# ─────────────────────────────────────────────────────────────────────────────
# MOTIVATION & HABIT SAMPLES
# ─────────────────────────────────────────────────────────────────────────────

MOTIVATION_QA = [
    (
        "I keep missing workouts. How do I build the habit?",
        "Missing workouts is a scheduling problem, not a motivation problem. "
        "Set a fixed time 3-4 days per week and protect it like a work commitment. "
        "Start with the minimum: 30 minutes is enough to build the habit, even if it's not the ideal session. "
        "Track completions with a simple calendar mark — the visual streak becomes a motivator in itself after 2-3 weeks."
    ),
    (
        "I feel embarrassed at the gym as a beginner. What should I do?",
        "Everyone at the gym was a beginner once — most experienced lifters have genuine respect for people just starting. "
        "The most effective mindset shift: focus entirely on your own session, your form, and your numbers. "
        "Going at off-peak hours (early morning or mid-afternoon) reduces gym traffic and makes the environment less intimidating. "
        "Within 4-6 weeks of consistent training, the gym will feel like your space."
    ),
    (
        "How do I stay motivated on days I don't feel like training?",
        "Motivation is unreliable — discipline and routine are what drive long-term consistency. "
        "Show up anyway and commit to just starting: put on your gear, do your warm-up. The hardest part is always starting. "
        "A reduced session (half the volume, same movements) is infinitely better than no session. "
        "Track your progress — seeing your numbers improve over months provides intrinsic motivation that outlasts any initial excitement."
    ),
    (
        "Is it normal to feel sore after every workout?",
        "Some soreness in the first 1-2 weeks of a new program is normal and expected — it is called DOMS. "
        "After 3-4 weeks of consistent training with the same movements, soreness typically diminishes significantly as the body adapts. "
        "Absence of soreness does not mean the workout was not effective — progressive overload matters more than soreness. "
        "Severe, persistent soreness is a sign to reduce volume and allow more recovery time."
    ),
    (
        "How do I track my progress effectively?",
        "Track three things: lift performance (weight and reps), body measurements (scale weight, waist, key measurements), and photos every 4 weeks. "
        "Scale weight fluctuates daily from water, food volume, and sleep — use a weekly average, not a daily number. "
        "Your training log is the most reliable progress indicator — if you lifted more this week than 4 weeks ago, you are progressing regardless of what the scale says. "
        "Progress photos taken in consistent lighting and poses provide the clearest visual evidence of body composition change."
    ),
]

def build_motivation_samples(count: int = 70) -> list:
    samples = []
    pool = MOTIVATION_QA * (count // len(MOTIVATION_QA) + 2)
    for qa in pool[:count]:
        profile = random_profile()
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": qa[0]},
                {"role": "assistant", "content": qa[1]},
            ]
        })
    return samples[:count]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_manual_samples(output_path: str = None, count: int = 620):
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "manual_samples.jsonl")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    samples = []
    samples.extend(build_nutrition_samples(200))
    samples.extend(build_progress_samples(120))
    samples.extend(build_injury_samples(100))
    samples.extend(build_app_samples(130))
    samples.extend(build_motivation_samples(70))

    random.shuffle(samples)
    samples = samples[:count]

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} manual samples → {output_path}")
    return len(samples)


if __name__ == "__main__":
    print("Generating manual training samples...")
    build_manual_samples()
    print("Done.")
