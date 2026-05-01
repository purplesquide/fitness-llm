from __future__ import annotations

import argparse
from pathlib import Path

from fitness_llm.config import ProjectPaths
from fitness_llm.dataset_builder import validate_jsonl_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a fine-tuning JSONL dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ProjectPaths().dataset_path,
        help="Path to the JSONL dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_jsonl_dataset(args.dataset)
    print(result)


if __name__ == "__main__":
    main()
