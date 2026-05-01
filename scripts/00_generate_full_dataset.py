from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = """You are an expert fitness coach built into a workout planning app.
You give evidence-based, concise advice in 3 to 5 sentences.
Always reference the user's level, goal, and current plan when relevant.
Only mention urgent medical escalation for genuine red-flag scenarios.
Use a direct, practical coaching tone."""


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts" / "datasets"


def add(records: list[dict], seen: set[tuple[str, str]], category: str, user: str, assistant: str) -> None:
    key = (category.strip(), user.strip())
    if key in seen:
        return
    seen.add(key)
    records.append({"category": category, "user": user.strip(), "assistant": assistant.strip()})


def build_records() -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    exercise_guidance_data = [
        ("Romanian deadlift", "hamstrings and glutes", "keep the bar close and hinge through the hips", "3 to 4 sets of 8 to 12 reps"),
        ("barbell squat", "quads and glutes", "brace hard, sit between the hips, and control the descent", "3 to 5 sets of 5 to 10 reps"),
        ("bench press", "chest, front delts, and triceps", "keep the upper back tight and touch a consistent bar path", "3 to 5 sets of 5 to 10 reps"),
        ("deadlift", "glutes, hamstrings, and spinal erectors", "push the floor away and keep the bar over the midfoot", "3 to 4 sets of 3 to 6 reps"),
        ("overhead press", "front delts and triceps", "squeeze glutes, keep ribs down, and press in a straight line", "3 to 4 sets of 6 to 10 reps"),
        ("pull-up", "lats and upper back", "start from a dead hang and drive elbows toward the ribs", "3 to 4 sets close to technical failure"),
        ("barbell row", "lats, middle back, and rear delts", "hinge first and row without turning it into a shrug", "3 to 4 sets of 6 to 12 reps"),
        ("leg press", "quads and glutes", "control the eccentric and avoid bouncing at the bottom", "3 to 4 sets of 10 to 15 reps"),
        ("hip thrust", "glutes", "pause at lockout and keep the ribcage stacked over the pelvis", "3 to 4 sets of 8 to 12 reps"),
        ("lat pulldown", "lats and upper back", "pull elbows down instead of yanking with the hands", "3 to 4 sets of 8 to 12 reps"),
        ("incline dumbbell press", "upper chest, front delts, and triceps", "use a stable arch and lower the dumbbells with control", "3 to 4 sets of 8 to 12 reps"),
        ("Bulgarian split squat", "quads and glutes", "stay balanced, use full depth, and keep tension on the lead leg", "3 to 4 sets of 8 to 12 reps per leg"),
        ("cable row", "middle back and lats", "keep the torso steady and finish by driving elbows back", "3 to 4 sets of 8 to 15 reps"),
        ("lateral raise", "side delts", "lead with the elbows and stop before upper traps take over", "3 to 5 sets of 12 to 20 reps"),
        ("leg curl", "hamstrings", "control both directions and avoid swinging the stack", "3 to 4 sets of 10 to 15 reps"),
    ]
    profiles = [
        ("beginner", "general fitness"),
        ("beginner", "fat loss"),
        ("intermediate", "hypertrophy"),
        ("intermediate", "strength"),
        ("advanced", "hypertrophy"),
        ("advanced", "strength"),
        ("intermediate", "muscle gain"),
        ("beginner", "confidence and technique"),
        ("intermediate", "lean bulk"),
        ("advanced", "performance"),
    ]
    for exercise, muscles, cue, dose in exercise_guidance_data:
        for level, goal in profiles:
            add(
                records,
                seen,
                "exercise_guidance",
                f"I am {level} level with a {goal} goal. How should I do the {exercise}?",
                f"The {exercise} mainly trains the {muscles}. Focus on {cue} so the target muscles stay loaded and the movement stays repeatable. For a {level} lifter chasing {goal}, start with {dose} and keep 1 to 3 reps in reserve on most working sets. If technique breaks before the target reps, reduce the load and keep the pattern clean before progressing.",
            )

    program_topics = [
        ("4 days per week", "hypertrophy", "upper lower split", "it gives each muscle roughly twice-weekly exposure with manageable fatigue"),
        ("3 days per week", "general fitness", "full body split", "it lets you practice the basics often without spreading volume too thin"),
        ("5 days per week", "hypertrophy", "push pull legs with an extra upper day", "it raises specialization while keeping overlap manageable"),
        ("6 days per week", "muscle gain", "push pull legs repeated", "it allows high weekly volume if recovery is excellent"),
        ("4 days per week", "strength", "upper lower with heavy top sets", "it balances recovery and frequent exposure to the main lifts"),
        ("2 days per week", "fat loss", "full body training", "it covers the big patterns and leaves room for walking or cardio"),
        ("5 days per week", "strength", "strength split with squat, bench, and pull emphasis", "it keeps the main lifts high priority while controlling assistance volume"),
        ("3 days per week", "hypertrophy", "full body hypertrophy split", "it is the most efficient way to accumulate quality weekly volume on a limited schedule"),
        ("4 days per week", "lean bulk", "upper lower split", "it is predictable, recoverable, and easy to progress"),
        ("5 days per week", "performance", "upper lower plus specialization day", "it builds skill and extra volume without making every session too long"),
    ]
    concerns = [
        "What split should I use?",
        "How should I organize my training week?",
        "What structure makes the most sense for progress?",
        "How do I set up a sustainable weekly plan?",
        "What weekly setup would fit me best?",
    ]
    levels = ["beginner", "intermediate", "advanced"]
    for frequency, goal, split_name, rationale in program_topics:
        for idx, concern in enumerate(concerns):
            level = levels[idx % len(levels)]
            add(
                records,
                seen,
                "program_advice",
                f"I am {level}, training {frequency} for {goal}. {concern}",
                f"Use a {split_name}. For your schedule and {goal} goal, {rationale}. Keep each day centered on compound lifts first, then add isolation work only after the primary volume is covered. Track performance for 4 to 6 weeks before making big changes, because a good plan needs enough time to show whether it is working.",
            )
    for plateau_lift in ["bench press", "squat", "deadlift", "overhead press", "pull-up"]:
        for stall_weeks in ["2", "3", "4", "5", "6"]:
            for level in ["beginner", "intermediate", "advanced"]:
                add(
                    records,
                    seen,
                    "program_advice",
                    f"My {plateau_lift} has not improved for {stall_weeks} weeks and I am {level}. What should I change?",
                    f"A {stall_weeks}-week stall on the {plateau_lift} means you should first audit sleep, calories, protein, and technique before blaming the program. If recovery is fine, reduce fatigue with a short deload, then return with either a slightly lower volume target or a new rep range. At {level} level, progress is rarely linear, so use objective performance trends across several sessions instead of reacting to one bad workout.",
                )
    volume_targets = [
        ("chest", "10 to 20"),
        ("back", "10 to 20"),
        ("quads", "10 to 18"),
        ("hamstrings", "8 to 16"),
        ("shoulders", "8 to 18"),
    ]
    volume_contexts = [
        "How many sets per week should I do?",
        "What is a smart weekly volume target?",
        "How much weekly work is enough to grow?",
        "What set range should I start with?",
        "How much volume should I recover from before adding more?",
    ]
    for muscle, target in volume_targets:
        for question in volume_contexts:
            add(
                records,
                seen,
                "program_advice",
                f"For hypertrophy, how many weekly sets should I do for {muscle}? {question}",
                f"A practical weekly target for {muscle} is about {target} quality sets, starting nearer the low end if recovery is inconsistent. Add volume only if performance, recovery, and technique stay solid over several weeks. The best target is the most productive amount you can repeat, not the highest number you can survive once.",
            )

    substitutions = [
        ("barbell squat", "home with dumbbells", "goblet squats and Bulgarian split squats", "you can still train quads and glutes hard without a rack"),
        ("bench press", "home with dumbbells", "dumbbell bench press or weighted push-ups", "both keep a strong horizontal press pattern with easier setup"),
        ("overhead press", "shoulder irritation", "landmine press and lateral raises", "they reduce overhead stress while still training the delts"),
        ("deadlift", "lower back sensitivity", "Romanian deadlifts with lighter load or machine hip hinges", "they keep the hip hinge pattern with a lower recovery cost"),
        ("pull-up", "no pull-up bar", "one-arm dumbbell rows and chest-supported rows", "they are the best available way to keep lat volume high"),
        ("leg press", "no machines", "front-foot-elevated split squats", "they load the quads well with minimal equipment"),
        ("barbell row", "equipment-limited home gym", "dumbbell rows", "they are easy to progress and easier on the lower back"),
        ("lat pulldown", "no cable station", "band pulldowns and pull-up regressions", "they preserve a vertical pull pattern when machines are unavailable"),
        ("hip thrust", "no bench", "glute bridges and frog pumps", "they still bias the glutes with simple setup"),
        ("leg curl", "no hamstring machine", "slider leg curls and Romanian deadlifts", "they train knee flexion and hip extension without a dedicated machine"),
    ]
    constraints = [
        "I train at home with only dumbbells.",
        "My gym does not have the machine.",
        "I need a shoulder-friendly option.",
        "I need a lower-back-friendly substitute.",
        "I only have bands and bodyweight.",
        "I am traveling and have limited equipment.",
        "I need something easier to learn.",
        "I want a version I can do in a crowded gym.",
        "I need a unilateral alternative.",
        "I want the closest substitute for hypertrophy.",
    ]
    for original, scenario, replacement, rationale in substitutions:
        for constraint in constraints:
            add(
                records,
                seen,
                "exercise_substitution",
                f"{constraint} What replaces {original} if I am dealing with {scenario}?",
                f"Replace {original} with {replacement}. That substitution works because {rationale}. Match the movement pattern and keep the effort hard enough that the target muscle is still challenged. Use similar weekly set volume to the original exercise so the substitution does not quietly reduce your total training stimulus.",
            )

    bodyweights = [55, 60, 65, 70, 75, 80, 85, 90]
    goals = ["muscle gain", "hypertrophy", "fat loss", "lean bulk", "recomp"]
    for bw in bodyweights:
        for goal in goals:
            protein_low = round(bw * 1.6)
            protein_high = round(bw * 2.2)
            add(
                records,
                seen,
                "nutrition",
                f"I weigh {bw} kg and my goal is {goal}. How much protein should I eat?",
                f"Aim for about {protein_low} to {protein_high} grams of protein per day. That range covers what most lifters need for {goal}, assuming total calories and training are aligned. Split it across 3 to 5 meals so each meal gives you a meaningful dose. Hit the daily total consistently before worrying about perfect meal timing.",
            )
    for bw in bodyweights:
        for user in [
            f"I weigh {bw} kg. How big should my calorie surplus be for a lean bulk?",
            f"At {bw} kg, should I bulk aggressively or use a small surplus?",
            f"How many extra calories should I eat to gain muscle at {bw} kg?",
            f"What is a smart calorie target for slow muscle gain if I weigh {bw} kg?",
            f"I am {bw} kg and want muscle without too much fat gain. What surplus should I use?",
        ]:
            add(
                records,
                seen,
                "nutrition",
                user,
                "Use a modest surplus of roughly 200 to 400 calories above maintenance and adjust from weekly scale trends. If bodyweight is climbing too quickly, the surplus is probably too high and you are just buying extra fat gain. Muscle is built by productive training plus enough food to recover, not by eating far beyond what you can use.",
            )

    lifts = ["bench press", "squat", "deadlift", "overhead press", "row", "pull-up", "leg press", "hip thrust"]
    rep_ranges = ["5 to 8", "6 to 8", "8 to 10", "10 to 12", "12 to 15"]
    for lift in lifts:
        for rep_range in rep_ranges:
            add(
                records,
                seen,
                "progress_plateaus",
                f"When should I increase the weight on my {lift} if my target is {rep_range} reps?",
                f"Increase the load once you can hit the top of the {rep_range} range across all prescribed working sets with clean technique and a small buffer left. If your form degrades or the last set falls far below target, keep the same load and own the reps first. For the {lift}, small load jumps keep progress moving without turning one good session into several bad ones.",
            )
    plateau_symptoms = [
        "persistent soreness",
        "motivation drop",
        "joint aches",
        "performance decline",
        "poor sleep",
        "low appetite",
        "high fatigue",
        "lack of bar speed",
    ]
    plateau_contexts = [
        "after 5 hard weeks",
        "after 6 hard weeks",
        "during a plateau",
        "while cutting calories",
        "during high stress",
    ]
    for symptom in plateau_symptoms:
        for context in plateau_contexts:
            add(
                records,
                seen,
                "progress_plateaus",
                f"I have {symptom} {context}. Do I need a deload?",
                f"You probably need at least a short fatigue reduction phase if {symptom} is lasting and training quality is falling. The simplest deload is to keep the same exercises, reduce volume by about 40 to 60 percent, and keep intensity moderate instead of grinding. If you come out of that week feeling sharper and performance rebounds, the issue was accumulated fatigue rather than a bad program.",
            )

    issues = [
        ("shoulder pain on pressing", "pull back from painful overhead and deep pressing angles, then use landmine press or neutral-grip dumbbells"),
        ("knee pain in squats", "reduce load, control the eccentric, and work in a pain-free range while using split squats or leg press if tolerated"),
        ("lower back tightness after hinges", "drop the load, clean up the brace and hinge, and use a friendlier posterior-chain variation temporarily"),
        ("elbow irritation from curls", "change grip, reduce junk volume, and use a pain-free curl pattern with slower tempo"),
        ("wrist pain on bench press", "use a stacked wrist position and consider dumbbells or a slightly different grip width"),
        ("hip pinching in deep squats", "adjust stance, toe angle, and depth so you can train hard without forcing the painful position"),
        ("ankle stiffness limiting squats", "add calf and ankle mobility work and elevate the heels if that improves positioning"),
        ("neck tightness during upper-body days", "reduce shrugging compensation and keep the ribcage and shoulder blades under control"),
        ("patellar tendon irritation", "manage jump volume, use slower eccentrics, and stay in ranges you can tolerate"),
        ("hamstring strain history", "rebuild with controlled hinges and curls before returning to maximal loading"),
    ]
    injury_questions = [
        "Can I keep training?",
        "What should I change in my plan?",
        "What is the safest training adjustment?",
        "How do I modify training without losing progress?",
        "What should I avoid right now?",
        "What is a smart return-to-training approach?",
    ]
    for issue, advice in issues:
        for question in injury_questions:
            add(
                records,
                seen,
                "injury_recovery",
                f"I have {issue}. {question}",
                f"Yes, you can usually keep training if symptoms stay manageable and the movement choice respects the issue. First, {advice}. Keep effort high on pain-free patterns so you still create a useful training stimulus. If pain is sharp, worsening, or changes normal movement quality, stop forcing the aggravating exercise and reassess instead of trying to push through it.",
            )

    motivation_scenarios = [
        "I keep missing workouts after work.",
        "I start strong for two weeks and then fall off.",
        "I lose motivation when progress is slow.",
        "I struggle to stay consistent on weekends.",
        "I skip sessions when the gym is crowded.",
        "I keep overplanning and undertraining.",
        "I miss sessions when work gets stressful.",
        "I do not feel disciplined enough to train regularly.",
    ]
    motivation_angles = [
        "How do I stay consistent?",
        "What should I change in my routine?",
        "How do I make training automatic?",
        "What is the most practical fix?",
        "How do I stop relying on motivation?",
    ]
    for scenario in motivation_scenarios:
        for angle in motivation_angles:
            add(
                records,
                seen,
                "motivation_habit",
                f"{scenario} {angle}",
                "Reduce the friction around training instead of waiting to feel motivated. Use a fixed schedule, shorten the minimum session target, and make the first exercise automatic so starting requires less decision-making. Track completed sessions and trend consistency over a month, because habits are built by repetition, not by chasing perfect energy every day.",
            )

    app_questions = [
        ("What does RPE 7 mean in my workout plan?", "RPE 7 means the set should feel challenging but still leave roughly 3 reps in reserve."),
        ("What does RPE 8 mean in my workout plan?", "RPE 8 means you should finish the set with about 2 good reps still available."),
        ("What does RPE 9 mean in my workout plan?", "RPE 9 means you are very close to failure and usually only have about 1 rep left."),
        ("My plan says 3 sets of 8 to 10. How do I progress that?", "Add reps within the range first, then increase the load once all sets reach the top end with clean form."),
        ("My plan says 4 sets of 10 to 12. How do I use that rep range?", "Use the range as a progression window rather than trying to hit the top number immediately on day one."),
        ("Why does my plan repeat some movement patterns every week?", "Repeated movement patterns make progression measurable and improve skill on the lifts that matter most."),
        ("Why are there lighter and heavier days in my plan?", "Different loading days help you manage fatigue while still practicing the main movement patterns."),
        ("How should I rest between sets on compounds?", "Most compound lifts need about 2 to 3 minutes of rest so performance stays high across sets."),
        ("How should I rest between isolation sets?", "Isolation work usually does well with roughly 60 to 90 seconds of rest, sometimes up to 2 minutes if performance drops hard."),
        ("What should I do if I miss one workout from the plan?", "Do not cram missed sessions on top of the week. Resume the plan in order and keep the weekly structure stable."),
    ]
    app_variants = [
        "Explain it simply.",
        "Give me the practical version.",
        "What should I actually do in the gym?",
        "Keep it short.",
    ]
    for question, answer in app_questions:
        for variant in app_variants:
            add(
                records,
                seen,
                "app_specific",
                f"{question} {variant}",
                f"{answer} In practice, use that instruction to keep effort and progression consistent instead of guessing from session to session. If execution quality drops badly, adjust the load or rest time before you assume the program itself is wrong.",
            )

    return records


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    counts = Counter(item["category"] for item in records)

    expected = {
        "exercise_guidance": 150,
        "program_advice": 150,
        "exercise_substitution": 100,
        "nutrition": 80,
        "progress_plateaus": 80,
        "injury_recovery": 60,
        "motivation_habit": 40,
        "app_specific": 40,
    }
    if dict(counts) != expected:
        raise ValueError(f"Dataset counts do not match expected counts: {dict(counts)}")

    json_path = DATA_DIR / "full_training_examples.json"
    jsonl_path = ARTIFACTS_DIR / "fitness_training_data_full.jsonl"
    summary_path = DATA_DIR / "dataset_summary.json"

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": record["user"]},
                    {"role": "assistant", "content": record["assistant"]},
                ]
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps({"total_samples": len(records), "category_counts": dict(counts)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"total_samples": len(records), "category_counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()