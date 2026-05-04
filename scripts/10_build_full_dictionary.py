from __future__ import annotations

import argparse
from pathlib import Path

from fitness_llm.dictionary_builder import build_full_dictionary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full domain + user dictionaries and a fine-tuning dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for dictionary artifacts. Defaults to artifacts/dictionaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_full_dictionary(output_dir=args.output_dir)
    print("Dictionary build complete:")
    print(f"  Domain dictionary : {result.domain_jsonl}")
    print(f"  User schema       : {result.user_schema_json}")
    print(f"  User profiles     : {result.user_profiles_jsonl}")
    print(f"  Instructions      : {result.instruction_jsonl}")
    print(f"  Fine-tune dataset : {result.finetune_jsonl}")
    print(f"  Summary           : {result.summary_json}")


if __name__ == "__main__":
    main()
