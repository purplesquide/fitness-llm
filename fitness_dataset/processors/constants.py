"""
Shared constants for all dataset processors.
"""
import random
import os

# Absolute path to the workspace root (two levels up from this file)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Paths to local data sources
EXERCISEDB_CSV     = os.path.join(WORKSPACE_ROOT, "fitness_exercises", "exercises.csv")
EXERCISEDB_JSON    = os.path.join(WORKSPACE_ROOT, "exercisedb", "exercisedb_v1_sample", "exercises.json")
MEGA_GYM_CSV       = os.path.join(WORKSPACE_ROOT, "megaGymDataset", "megaGymDataset.csv")
WGER_JSON          = os.path.join(WORKSPACE_ROOT, "wger_exercises.json")
LONGHAUL_STRENGTH  = os.path.join(WORKSPACE_ROOT, "longhaul_fitness", "strength.json")
LONGHAUL_CARDIO    = os.path.join(WORKSPACE_ROOT, "longhaul_fitness", "cardio.json")
LONGHAUL_FLEX      = os.path.join(WORKSPACE_ROOT, "longhaul_fitness", "flexibility.json")
EXISTING_DATASET   = os.path.join(WORKSPACE_ROOT, "artifacts", "datasets", "fitness_training_data_full.jsonl")
EXISTING_NUTRITION = os.path.join(WORKSPACE_ROOT, "artifacts", "datasets", "nutrition_textbook_full_qa.jsonl")
EXISTING_LLM       = os.path.join(WORKSPACE_ROOT, "artifacts", "datasets", "full_llm_training_data.jsonl")

# Output directory (inside fitness_dataset/)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

# ─────────────────────────────────────────────
# SYSTEM PROMPT — used in every training sample
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert fitness coach built into a workout planning app.
You give evidence-based, concise advice in 3-5 sentences.
Always consider the user's fitness level, goal, and available equipment in your response.
You never recommend unsafe practices. Speak in a direct, motivating coach tone.
When relevant, cite the reason behind your recommendation (e.g., progressive overload,
muscle recovery, movement pattern balance)."""

# ─────────────────────────────────────────────
# PROFILE VARIATION POOLS
# ─────────────────────────────────────────────
LEVELS      = ["beginner", "intermediate", "advanced"]
GOALS       = ["muscle_gain", "weight_loss", "strength", "endurance", "general_fitness"]
EQUIPMENT   = [
    "bodyweight",
    "home (dumbbells only)",
    "home (dumbbells + pull-up bar)",
    "minimal gym (barbell + bench)",
    "full gym",
]
FREQUENCIES = [2, 3, 4, 5, 6]

GOAL_TEXT_MAP = {
    "muscle_gain":     "muscle building",
    "weight_loss":     "weight loss",
    "strength":        "strength development",
    "endurance":       "endurance training",
    "general_fitness": "general fitness",
}

REP_TABLE = {
    "muscle_gain":     "3-4 sets of 8-12 reps",
    "strength":        "4-5 sets of 3-6 reps",
    "weight_loss":     "3-4 sets of 12-20 reps",
    "endurance":       "3-4 sets of 15-25 reps",
    "general_fitness": "3 sets of 10-15 reps",
}


def random_profile() -> dict:
    """Generate a random but realistic user profile for sample variation."""
    return {
        "level":     random.choice(LEVELS),
        "goal":      random.choice(GOALS),
        "equipment": random.choice(EQUIPMENT),
        "days":      random.choice(FREQUENCIES),
        "age":       random.randint(16, 55),
        "weight":    random.randint(50, 110),
    }
