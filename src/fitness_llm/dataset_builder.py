from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import ProjectPaths
from .prompts import SYSTEM_PROMPT


def _load_seed_examples(seed_file: Path) -> list[dict]:
    with seed_file.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Seed examples file must contain a JSON list.")
    return records


def _to_chat_record(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["user"].strip()},
            {"role": "assistant", "content": example["assistant"].strip()},
        ]
    }


def build_jsonl_dataset(
    output_path: Path | None = None,
    seed_file: Path | None = None,
    extra_examples: Iterable[dict] | None = None,
) -> Path:
    paths = ProjectPaths()
    paths.ensure()
    seed_path = seed_file or (paths.data_dir / "seed_training_examples.json")
    target_path = output_path or paths.dataset_path

    examples = _load_seed_examples(seed_path)
    if extra_examples:
        examples.extend(extra_examples)

    samples = [_to_chat_record(example) for example in examples]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return target_path


def validate_jsonl_dataset(dataset_path: Path) -> dict:
    total = 0
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        total += 1
        record = json.loads(raw_line)
        messages = record.get("messages", [])
        if len(messages) != 3:
            raise ValueError(f"Line {line_number}: expected exactly 3 messages.")
        if [msg.get("role") for msg in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Line {line_number}: roles must be system, user, assistant.")
        if len(messages[2].get("content", "").strip()) < 20:
            raise ValueError(f"Line {line_number}: assistant answer is too short.")

    if total == 0:
        raise ValueError("Dataset is empty.")

    return {"status": "ok", "samples": total, "dataset": str(dataset_path)}
