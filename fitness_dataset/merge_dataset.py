"""
Merge all processed JSONL output files into a single training dataset.
Deduplicates by assistant response hash, shuffles, and writes the final file.

Output:
  - output/fitness_training_data_full.jsonl
  Also copies to:
  - ../../artifacts/datasets/fitness_training_data_full.jsonl  (the Colab notebook location)
"""

import json
import random
import hashlib
import shutil
import os
from pathlib import Path

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
FINAL_FILE  = os.path.join(OUTPUT_DIR, "fitness_training_data_full.jsonl")
COLAB_FILE  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "artifacts", "datasets", "fitness_training_data_full.jsonl"))

SOURCE_FILES = [
    "exercisedb_guidance.jsonl",
    "exercisedb_substitutions.jsonl",
    "kaggle_samples.jsonl",
    "wger_samples.jsonl",
    "longhaul_samples.jsonl",
    "manual_samples.jsonl",
    "existing_datasets.jsonl",
    # HF files — included if present
    "hf_fitness_qa.jsonl",
    "hf_fitness_qa2.jsonl",
    "hf_generate_workouts.jsonl",
    "hf_fine_corpus.jsonl",
]


def load_jsonl(path: str) -> list:
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def is_valid_sample(obj: dict) -> bool:
    """Check that a sample has the correct structure."""
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        return False
    roles = [m.get("role") for m in msgs]
    if roles != ["system", "user", "assistant"]:
        return False
    for m in msgs:
        content = m.get("content", "")
        if not isinstance(content, str) or len(content.strip()) < 5:
            return False
    assistant_len = len(msgs[2].get("content", ""))
    if assistant_len < 30:
        return False
    return True


def dedup_by_assistant(samples: list) -> list:
    """Remove duplicate samples that have the same assistant response."""
    seen = set()
    unique = []
    for s in samples:
        key = hashlib.md5(s["messages"][2]["content"].encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def merge():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    all_samples = []

    for filename in SOURCE_FILES:
        path = os.path.join(OUTPUT_DIR, filename)
        if not Path(path).exists():
            print(f"  ⚠ Missing: {filename} — skipping")
            continue
        records = load_jsonl(path)
        valid   = [r for r in records if is_valid_sample(r)]
        print(f"  ✓ {filename}: {len(valid)} valid samples")
        all_samples.extend(valid)

    print(f"\n  Total before dedup: {len(all_samples)}")
    all_samples = dedup_by_assistant(all_samples)
    print(f"  Total after dedup:  {len(all_samples)}")

    # Shuffle to prevent source-ordering bias during training
    random.shuffle(all_samples)

    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'='*55}")
    print(f"  Final dataset: {len(all_samples)} samples")
    print(f"  Saved to:      {FINAL_FILE}")

    # Copy to Colab/artifacts location
    colab_dir = os.path.dirname(COLAB_FILE)
    if Path(colab_dir).exists():
        shutil.copy2(FINAL_FILE, COLAB_FILE)
        print(f"  Copied to:     {COLAB_FILE}")
    else:
        print(f"  ⚠ Colab artifacts dir not found at {colab_dir} — skipping copy")

    return len(all_samples)


if __name__ == "__main__":
    print("Merging all dataset sources...")
    count = merge()
    if count < 500:
        print(f"\n  ⚠ WARNING: Only {count} samples. Target is 2,000+.")
    elif count < 1000:
        print(f"\n  ⚠ Dataset functional but adding more samples will improve results.")
    else:
        print(f"\n  ✓ Dataset ready for validation and fine-tuning.")
