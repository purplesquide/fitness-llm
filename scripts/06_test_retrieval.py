from __future__ import annotations

import argparse

from fitness_llm.rag import retrieve_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect retrieval results from the FAISS index.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve_chunks(query=args.query, top_k=args.top_k)
    for idx, item in enumerate(results, start=1):
        print(f"[{idx}] source={item.get('source')} score={item.get('score'):.3f}")
        print(item.get("text", ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
