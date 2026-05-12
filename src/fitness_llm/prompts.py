from __future__ import annotations
from typing import Any

SYSTEM_PROMPT = """You are an expert fitness coach built into a workout planning app.
You give evidence-based, concise advice in 3-5 sentences.
Always reference the user's level, goal, and current plan when relevant.
You never recommend unsafe practices or make medical diagnoses.
For red-flag symptoms (chest pain, severe joint pain, numbness, dizziness), always advise consulting a healthcare professional.
Use a direct, practical coaching tone and explain your reasoning briefly (e.g., progressive overload, muscle recovery, movement pattern balance)."""


def build_system_prompt(
    user_profile: dict | None = None,
    current_plan: dict | None = None,
    retrieved_context: str | None = None,
    user_context_section: str | None = None,
) -> str:
    """
    Build the complete system prompt with user context and RAG knowledge injected.

    Priority: user_context_section (pre-built from UserContextManager) > user_profile dict.
    """
    sections = [SYSTEM_PROMPT]

    # User profile section
    if user_context_section and user_context_section.strip():
        sections.append("USER PROFILE:\n" + user_context_section.strip())
    elif user_profile:
        profile_lines = [
            f"- Experience level: {user_profile.get('level', 'intermediate')}",
            f"- Goal: {user_profile.get('goal', 'general_fitness')}",
            f"- Training days: {user_profile.get('days', 4)} per week",
        ]
        if user_profile.get("weight"):
            profile_lines.append(f"- Weight: {user_profile['weight']} kg")
        if user_profile.get("injuries"):
            profile_lines.append(f"- Injuries/limitations: {user_profile['injuries']}")
        if user_profile.get("equipment"):
            profile_lines.append(f"- Equipment: {user_profile['equipment']}")
        if user_profile.get("bmi"):
            profile_lines.append(f"- BMI: {user_profile['bmi']}")
        sections.append("USER PROFILE:\n" + "\n".join(profile_lines))

    # Current program section
    if current_plan:
        if isinstance(current_plan, dict):
            plan_text = json.dumps(current_plan, indent=2)
        else:
            plan_text = str(current_plan)
        sections.append(f"CURRENT PROGRAM:\n{plan_text[:600]}")

    # RAG knowledge section
    if retrieved_context and retrieved_context.strip():
        sections.append(
            "RELEVANT KNOWLEDGE (use this to ground your answer):\n" + retrieved_context.strip()
        )

    sections.append(
        "INSTRUCTIONS:\n"
        "- Answer specifically for this user based on their profile.\n"
        "- If suggesting exercise replacements, preserve the muscle stimulus and movement pattern.\n"
        "- Explain your reasoning briefly and scientifically.\n"
        "- If the question concerns pain or injury, emphasise safety and suggest consulting a healthcare professional.\n"
        "- Do not fabricate exercises, contraindications, or medical diagnoses."
    )

    return "\n\n".join(sections)


def build_intent_detection_prompt(message: str) -> str:
    """Prompt for quick intent classification."""
    return f"""Classify this fitness chatbot message into ONE of these intents:
EXERCISE_GUIDANCE - technique, form, cues, how to perform an exercise
EXERCISE_SUBSTITUTION - replacing an exercise, equipment alternatives
PROGRAM_MODIFICATION - changing a workout plan, frequency, split
NUTRITION - protein, calories, diet, supplements, food
SCIENCE_EDUCATION - explaining training concepts, hypertrophy, recovery
INJURY_SAFETY - pain, injury, red flags, modifications for injuries
WORKOUT_PLANNING - designing a program, weekly structure
MOTIVATION - adherence, habit, staying consistent
GENERAL - does not fit the above

Message: "{message}"

Respond with ONLY the intent label (e.g. EXERCISE_GUIDANCE)."""


import json  # noqa: E402 (keep at module bottom to avoid circular issues)
