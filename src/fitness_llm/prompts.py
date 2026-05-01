from __future__ import annotations


SYSTEM_PROMPT = """You are an expert fitness coach built into a workout planning app.
You give evidence-based, concise advice in 3 to 5 sentences.
Always reference the user's level, goal, and current plan when relevant.
Only mention urgent medical escalation for genuine red-flag scenarios.
Use a direct, practical coaching tone."""


def build_system_prompt(
    user_profile: dict | None = None,
    current_plan: dict | None = None,
    retrieved_context: str | None = None,
) -> str:
    sections = [SYSTEM_PROMPT]

    if user_profile:
        sections.append(
            "User profile:\n"
            f"- Level: {user_profile.get('level', 'intermediate')}\n"
            f"- Goal: {user_profile.get('goal', 'hypertrophy')}\n"
            f"- Training days: {user_profile.get('days', 4)} per week\n"
            f"- Weight: {user_profile.get('weight', 'unknown')} kg\n"
            f"- BMI: {user_profile.get('bmi', 'unknown')}"
        )

    if current_plan:
        sections.append(f"Current plan context:\n{current_plan}")

    if retrieved_context:
        sections.append(f"Relevant knowledge:\n{retrieved_context}")

    return "\n\n".join(sections)
