from __future__ import annotations

import argparse
from pathlib import Path

from fitness_llm.dataset_builder import build_jsonl_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the JSONL fine-tuning dataset.")
    parser.add_argument("--seed-file", type=Path, default=None, help="Optional path to a JSON seed examples file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = build_jsonl_dataset(output_path=args.output, seed_file=args.seed_file)
    print(f"Dataset created: {dataset_path}")


if __name__ == "__main__":
    main()
