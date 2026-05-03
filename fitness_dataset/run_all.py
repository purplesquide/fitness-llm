"""
Master script — runs the entire dataset building pipeline in order.

Usage:
    cd fitness_dataset
    python run_all.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "processors"))

def step(name: str, fn):
    print(f"\n{'─'*55}")
    print(f"  STEP: {name}")
    print(f"{'─'*55}")
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 55)
    print("  FITNESS LLM — Dataset Building Pipeline")
    print("=" * 55)

    from process_exercisedb import process_exercisedb
    from process_kaggle import process_kaggle
    from process_wger_longhaul import process_wger, process_longhaul
    from process_manual import build_manual_samples
    from process_huggingface import process_all_hf

    step("ExerciseDB (guidance + substitutions)", process_exercisedb)
    step("MegaGym / Kaggle (programs + descriptions)", process_kaggle)
    step("wger exercises", process_wger)
    step("Longhaul fitness (strength/flexibility/cardio)", process_longhaul)
    step("Manual high-quality samples", build_manual_samples)
    step("HuggingFace datasets + existing workspace data", process_all_hf)

    # Merge
    print(f"\n{'─'*55}")
    print("  STEP: Merge all sources")
    print(f"{'─'*55}")
    sys.path.insert(0, os.path.dirname(__file__))
    from merge_dataset import merge
    count = merge()

    # Validate
    print(f"\n{'─'*55}")
    print("  STEP: Validate final dataset")
    print(f"{'─'*55}")
    from validate_dataset import validate
    valid, errors = validate()

    print(f"\n{'='*55}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total samples: {valid}")
    if errors:
        print(f"  Issues found:  {len(errors)} (check output above)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
