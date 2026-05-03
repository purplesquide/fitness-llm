"""
Validate the final merged training dataset.
Checks structure, role order, content length, and prints a distribution report.
"""

import json
import os
from pathlib import Path
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(SCRIPT_DIR, "output", "fitness_training_data_full.jsonl")


def validate(path: str = DEFAULT_PATH):
    if not Path(path).exists():
        print(f"✗ File not found: {path}")
        return 0, ["File not found"]

    lines   = Path(path).read_text(encoding="utf-8").strip().split("\n")
    errors  = []
    valid   = 0
    lengths = []
    user_lengths = []
    role_counts  = Counter()

    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: JSON parse error — {e}")
            continue

        msgs = obj.get("messages")

        if not isinstance(msgs, list):
            errors.append(f"Line {i}: 'messages' must be a list")
            continue

        if len(msgs) != 3:
            errors.append(f"Line {i}: expected 3 messages, got {len(msgs)}")
            continue

        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"Line {i}: wrong role order {roles}")
            continue

        content_ok = True
        for m in msgs:
            content = m.get("content", "")
            if not isinstance(content, str) or len(content.strip()) < 5:
                errors.append(f"Line {i}: empty/short content in role '{m.get('role')}'")
                content_ok = False

        if not content_ok:
            continue

        assistant_len = len(msgs[2]["content"])
        user_len      = len(msgs[1]["content"])

        if assistant_len < 30:
            errors.append(f"Line {i}: assistant response too short ({assistant_len} chars)")
            continue
        if assistant_len > 2000:
            errors.append(f"Line {i}: WARNING — very long assistant response ({assistant_len} chars)")

        lengths.append(assistant_len)
        user_lengths.append(user_len)
        for m in msgs:
            role_counts[m["role"]] += 1
        valid += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FILE: {path}")
    print(f"{'='*55}")
    print(f"  Lines processed : {len(lines)}")
    print(f"  Valid samples   : {valid}")
    print(f"  Errors found    : {len(errors)}")

    if lengths:
        print(f"\n  Assistant response length:")
        print(f"    Min : {min(lengths)} chars")
        print(f"    Max : {max(lengths)} chars")
        print(f"    Avg : {sum(lengths) // len(lengths)} chars")

    if user_lengths:
        print(f"\n  User message length:")
        print(f"    Min : {min(user_lengths)} chars")
        print(f"    Max : {max(user_lengths)} chars")
        print(f"    Avg : {sum(user_lengths) // len(user_lengths)} chars")

    # Count by profile tier (beginner/intermediate/advanced found in user messages)
    level_counts = Counter()
    goal_counts  = Counter()
    lines_reloaded = Path(path).read_text(encoding="utf-8").strip().split("\n")
    for line in lines_reloaded:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            user_msg = obj["messages"][1]["content"].lower()
            for lvl in ["beginner", "intermediate", "advanced"]:
                if lvl in user_msg:
                    level_counts[lvl] += 1
                    break
            for goal in ["muscle gain", "muscle building", "weight loss", "strength", "endurance", "general fitness"]:
                if goal in user_msg:
                    goal_counts[goal] += 1
                    break
        except Exception:
            pass

    if level_counts:
        print(f"\n  Level distribution in user messages:")
        for k, v in sorted(level_counts.items(), key=lambda x: -x[1]):
            print(f"    {k:<15}: {v}")

    if goal_counts:
        print(f"\n  Goal distribution in user messages:")
        for k, v in sorted(goal_counts.items(), key=lambda x: -x[1]):
            print(f"    {k:<20}: {v}")

    if errors:
        print(f"\n  --- First 15 errors ---")
        for e in errors[:15]:
            print(f"    ✗ {e}")

    print(f"\n{'='*55}")
    if valid < 500:
        print(f"  ⚠ WARNING: Only {valid} valid samples. Target is 2,000+.")
    elif valid < 1000:
        print(f"  ⚠ Dataset functional — adding more samples will improve fine-tuning.")
    elif valid < 2000:
        print(f"  ✓ Good dataset ({valid} samples). Consider adding more for best results.")
    else:
        print(f"  ✓ EXCELLENT — {valid} samples. Ready for fine-tuning.")

    return valid, errors


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    valid, errors = validate(path)
    sys.exit(0 if not errors else 1)
