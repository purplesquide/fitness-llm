from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import ProjectPaths, Settings


def _load_seed_knowledge(seed_file: Path) -> list[dict]:
    with seed_file.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("Knowledge seed file must contain a JSON list.")
    return items


def _exercise_to_chunk(payload: dict) -> dict:
    name = payload.get("name", "Unknown Exercise")
    primary = ", ".join(payload.get("primaryMuscles", []) or [])
    secondary = ", ".join(payload.get("secondaryMuscles", []) or [])
    equipment = payload.get("equipment", "bodyweight")
    category = payload.get("category", "strength")
    level = payload.get("level", "intermediate")
    instructions = " ".join(payload.get("instructions", []) or [])[:500]
    chunk = (
        f"Exercise: {name}\n"
        f"Primary muscles: {primary or 'unknown'}\n"
        f"Secondary muscles: {secondary or 'unknown'}\n"
        f"Equipment: {equipment}\n"
        f"Category: {category}\n"
        f"Level: {level}\n"
        f"Instructions: {instructions or 'No instructions provided.'}"
    )
    return {"source": "exercise_db", "exercise_name": name, "text": chunk}


def _load_exercise_chunks(exercise_dir: Path | None) -> list[dict]:
    if not exercise_dir or not exercise_dir.exists():
        return []

    chunks: list[dict] = []
    for exercise_file in sorted(exercise_dir.glob("*.json")):
        try:
            payload = json.loads(exercise_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                chunks.append(_exercise_to_chunk(payload))
        except Exception:
            continue
    return chunks


def build_knowledge_base(
    exercise_dir: Path | None = None,
    seed_file: Path | None = None,
    index_path: Path | None = None,
    chunks_path: Path | None = None,
    embedding_model: str | None = None,
) -> dict:
    paths = ProjectPaths()
    settings = Settings()
    paths.ensure()

    seed_path = seed_file or (paths.data_dir / "knowledge_seed.json")
    index_target = index_path or paths.rag_index_path
    chunks_target = chunks_path or paths.rag_chunks_path
    model_name = embedding_model or settings.embedding_model

    knowledge_chunks = _load_seed_knowledge(seed_path)
    knowledge_chunks.extend(_load_exercise_chunks(exercise_dir))

    if not knowledge_chunks:
        raise ValueError("No knowledge chunks were loaded.")

    embedder = SentenceTransformer(model_name)
    texts = [item["text"] for item in knowledge_chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_target.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_target))

    with chunks_target.open("wb") as handle:
        pickle.dump(knowledge_chunks, handle)

    return {
        "status": "ok",
        "chunks": len(knowledge_chunks),
        "index_path": str(index_target),
        "chunks_path": str(chunks_target),
    }


def retrieve_chunks(
    query: str,
    index_path: Path | None = None,
    chunks_path: Path | None = None,
    top_k: int = 3,
    embedding_model: str | None = None,
) -> list[dict]:
    paths = ProjectPaths()
    settings = Settings()

    index_file = index_path or paths.rag_index_path
    chunks_file = chunks_path or paths.rag_chunks_path
    model_name = embedding_model or settings.embedding_model

    index = faiss.read_index(str(index_file))
    with chunks_file.open("rb") as handle:
        chunks = pickle.load(handle)

    embedder = SentenceTransformer(model_name)
    query_embedding = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = dict(chunks[idx])
        item["score"] = float(score)
        results.append(item)
    return results
