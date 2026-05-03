"""
Download and process HuggingFace fitness datasets into training samples.
Also incorporates existing artifacts/datasets/ JSONL files already in the workspace.

HF datasets attempted:
  - hammamwahab/fitness-qa            → raw/huggingface/fitness_qa.jsonl
  - its-myrto/fitness-question-answers → raw/huggingface/fitness_qa2.jsonl
  - Kolibri753/generate-workouts       → raw/huggingface/generate_workouts.jsonl
  - padilfm/FineCorpus-WorkoutExercise → raw/huggingface/fine_corpus.jsonl

Existing workspace datasets incorporated:
  - artifacts/datasets/fitness_training_data_full.jsonl
  - artifacts/datasets/nutrition_textbook_full_qa.jsonl
  - artifacts/datasets/full_llm_training_data.jsonl

Output:
  - output/hf_fitness_qa.jsonl        : ~200 samples (or 0 if unavailable)
  - output/hf_generate_workouts.jsonl : ~200 samples (or 0 if unavailable)
  - output/existing_datasets.jsonl    : samples from artifacts/
"""

import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from constants import (
    SYSTEM_PROMPT, random_profile, GOAL_TEXT_MAP,
    EXISTING_DATASET, EXISTING_NUTRITION, EXISTING_LLM, OUTPUT_DIR,
)

RAW_HF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw", "huggingface"))


def download_hf_datasets():
    """Attempt to download HuggingFace datasets. Skips gracefully if unavailable."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' library not installed — skipping HF download.")
        print("  Install with: pip install datasets")
        return

    Path(RAW_HF_DIR).mkdir(parents=True, exist_ok=True)

    hf_sources = {
        "fitness_qa":        "hammamwahab/fitness-qa",
        "fitness_qa2":       "its-myrto/fitness-question-answers",
        "generate_workouts": "Kolibri753/generate-workouts",
        "fine_corpus":       "padilfm/FineCorpus-WorkoutExercise",
    }

    for name, hf_id in hf_sources.items():
        out_path = os.path.join(RAW_HF_DIR, f"{name}.jsonl")
        if Path(out_path).exists():
            print(f"  ✓ {name}: already downloaded")
            continue
        try:
            ds = load_dataset(hf_id, split="train", trust_remote_code=True)
            ds.to_json(out_path)
            print(f"  ✓ {name}: {len(ds)} rows saved → {out_path}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")


def _load_jsonl(path: str) -> list:
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


# Phrases that indicate a non-answer in the fitness_qa dataset
_SKIP_PHRASES = [
    "i don't have data", "i do not have data", "no data", "not applicable",
    "n/a", "none", "nothing",
]


def process_fitness_qa(path: str, output_name: str, count: int = 200):
    """
    Process Q&A schema. Handles multiple field name conventions:
      {question/answer}, {Question/Answer}, {Q/A}, {context/question/answer}
    """
    if not Path(path).exists():
        print(f"  ✗ {path} not found — skipping")
        return

    records = _load_jsonl(path)
    random.shuffle(records)
    samples = []

    for rec in records:
        # Support multiple field name conventions
        question = (
            rec.get("question") or rec.get("Question") or
            rec.get("Q") or rec.get("input") or ""
        ).strip()
        answer = (
            rec.get("answer") or rec.get("Answer") or
            rec.get("A") or rec.get("output") or ""
        ).strip()

        if not question or not answer:
            continue
        if len(answer) < 30 or len(answer) > 1200:
            continue

        # Skip non-answers from the hammamwahab fitness-qa dataset
        if any(skip in answer.lower() for skip in _SKIP_PHRASES):
            continue

        profile = random_profile()
        enriched_q = (
            f"[{profile['level'].capitalize()}, {GOAL_TEXT_MAP.get(profile['goal'], profile['goal'])}, "
            f"{profile['days']} days/week] {question}"
        )

        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": enriched_q},
                {"role": "assistant", "content": answer},
            ]
        })

        if len(samples) >= count:
            break

    out = os.path.join(OUTPUT_DIR, output_name)
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples → {out}")


def _parse_inst_text(text: str):
    """
    Parse Llama-style [INST]...[/INST]... format into (question, answer).
    Also handles plain text with newline separators.
    """
    import re
    # Llama/Mistral [INST] format
    m = re.search(r"\[INST\](.*?)\[\/INST\](.*)", text, re.DOTALL)
    if m:
        question = m.group(1).strip().lstrip("Generate workout for this purpose:").strip().strip('"').strip()
        answer   = m.group(2).strip().rstrip("</s>").strip()
        return question, answer
    return None, None


def process_generate_workouts(path: str, count: int = 100):
    """
    Process workout generation data. Handles:
      - {instruction, input, output} schema
      - {text} schema with [INST]...[/INST]... format
    """
    if not Path(path).exists():
        print(f"  ✗ {path} not found — skipping")
        return

    records = _load_jsonl(path)
    random.shuffle(records)
    samples = []

    for rec in records:
        # Standard instruction-following format
        instruction = (rec.get("instruction") or "").strip()
        inp         = (rec.get("input") or "").strip()
        output_text = (rec.get("output") or "").strip()

        if instruction and output_text and len(output_text) >= 50:
            user_text = instruction
            if inp:
                user_text = f"{instruction}\n\nContext: {inp}"
            samples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_text.strip()},
                    {"role": "assistant", "content": output_text.strip()},
                ]
            })
            if len(samples) >= count:
                break
            continue

        # [INST] text format
        text = (rec.get("text") or "").strip()
        if text:
            question, answer = _parse_inst_text(text)
            if question and answer and len(answer) >= 50:
                samples.append({
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": question},
                        {"role": "assistant", "content": answer[:1500]},
                    ]
                })

        if len(samples) >= count:
            break

    out = os.path.join(OUTPUT_DIR, "hf_generate_workouts.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} generate-workouts samples → {out}")


def process_fine_corpus(path: str, count: int = 150):
    """Process FineCorpus — handles messages, text, or content schemas."""
    if not Path(path).exists():
        print(f"  ✗ {path} not found — skipping")
        return

    records = _load_jsonl(path)
    random.shuffle(records)
    samples = []

    for rec in records:
        # Already in messages format
        if "messages" in rec and isinstance(rec["messages"], list):
            msgs = rec["messages"]
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = SYSTEM_PROMPT
            else:
                msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            if len(msgs) >= 3:
                samples.append({"messages": msgs[:3]})
            continue

        # Text/content split
        text = (rec.get("text") or rec.get("content") or "").strip()
        if len(text) < 80:
            continue

        for delim in ["\nA:", "\nAnswer:", "Answer:", "\nResponse:"]:
            if delim in text:
                parts = text.split(delim, 1)
                question = parts[0].replace("Q:", "").replace("Question:", "").strip()
                answer   = parts[1].strip()
                if len(question) > 10 and len(answer) > 30:
                    samples.append({
                        "messages": [
                            {"role": "system",    "content": SYSTEM_PROMPT},
                            {"role": "user",      "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    })
                    break

        if len(samples) >= count:
            break

    out = os.path.join(OUTPUT_DIR, "hf_fine_corpus.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} FineCorpus samples → {out}")


def process_existing_datasets():
    """
    Re-package existing workspace datasets with the canonical system prompt.
    This ensures all existing samples use the same system message as the new ones.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    all_existing = []

    for src_path in [EXISTING_DATASET, EXISTING_NUTRITION, EXISTING_LLM]:
        if not Path(src_path).exists():
            print(f"  ✗ {src_path} not found — skipping")
            continue
        records = _load_jsonl(src_path)
        for rec in records:
            msgs = rec.get("messages", [])
            if not isinstance(msgs, list) or len(msgs) < 3:
                continue
            # Normalize to exactly [system, user, assistant]
            roles = [m.get("role") for m in msgs[:3]]
            if roles != ["system", "user", "assistant"]:
                continue
            # Replace system prompt with canonical one
            msgs[0]["content"] = SYSTEM_PROMPT
            all_existing.append({"messages": msgs[:3]})
        print(f"  Loaded {len(records)} samples from {os.path.basename(src_path)}")

    out = os.path.join(OUTPUT_DIR, "existing_datasets.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for s in all_existing:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(all_existing)} existing samples → {out}")


def process_all_hf():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Try to download from HuggingFace
    download_hf_datasets()

    # Process whatever is available
    process_fitness_qa(
        path=os.path.join(RAW_HF_DIR, "fitness_qa.jsonl"),
        output_name="hf_fitness_qa.jsonl",
        count=200,
    )
    process_fitness_qa(
        path=os.path.join(RAW_HF_DIR, "fitness_qa2.jsonl"),
        output_name="hf_fitness_qa2.jsonl",
        count=150,
    )
    process_generate_workouts(
        path=os.path.join(RAW_HF_DIR, "generate_workouts.jsonl"),
        count=200,
    )
    process_fine_corpus(
        path=os.path.join(RAW_HF_DIR, "fine_corpus.jsonl"),
        count=150,
    )

    # Always process existing workspace datasets
    process_existing_datasets()


if __name__ == "__main__":
    print("Processing HuggingFace + existing workspace datasets...")
    process_all_hf()
    print("Done.")
