from __future__ import annotations

import argparse
from pathlib import Path

from fitness_llm.rag import build_knowledge_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FAISS RAG index.")
    parser.add_argument("--exercise-dir", type=Path, default=None, help="Optional path to exercise JSON files.")
    parser.add_argument("--seed-file", type=Path, default=None, help="Optional path to knowledge seed JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_knowledge_base(exercise_dir=args.exercise_dir, seed_file=args.seed_file)
    print(result)


if __name__ == "__main__":
    main()
