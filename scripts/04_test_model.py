from __future__ import annotations

import argparse
from pathlib import Path

from fitness_llm.llm_engine import FitnessLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local test prompt against the fine-tuned model.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--level", type=str, default="intermediate")
    parser.add_argument("--goal", type=str, default="hypertrophy")
    parser.add_argument("--days", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = FitnessLLM()
    engine.load_local_model(adapter_dir=args.adapter_dir)
    reply = engine.chat_local(
        message=args.question,
        user_profile={"level": args.level, "goal": args.goal, "days": args.days},
    )
    print(reply)


if __name__ == "__main__":
    main()
