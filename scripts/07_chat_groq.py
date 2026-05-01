from __future__ import annotations

import argparse

from fitness_llm.llm_engine import FitnessLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one message through the Groq-backed coach.")
    parser.add_argument("--message", type=str, required=True)
    parser.add_argument("--level", type=str, default="intermediate")
    parser.add_argument("--goal", type=str, default="hypertrophy")
    parser.add_argument("--days", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = FitnessLLM()
    try:
        engine.load_rag()
    except Exception:
        pass
    reply = engine.chat_groq(
        message=args.message,
        user_profile={"level": args.level, "goal": args.goal, "days": args.days},
    )
    print(reply)


if __name__ == "__main__":
    main()
