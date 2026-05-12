"""
ChromaDB-based RAG pipeline for the fitness coaching chatbot.

Follows the implementation guide recommendations:
  - BGE-large-en-v1.5 embeddings (falls back to all-MiniLM-L6-v2 if unavailable)
  - Three ChromaDB collections: exercises, knowledge, qa_pairs
  - Hybrid search with metadata filtering by topic/muscle/equipment
  - Re-ranking by score threshold

Usage:
    from fitness_llm.chroma_rag import ChromaRAG
    rag = ChromaRAG()
    rag.build()           # indexes all sources
    results = rag.search("how to squat for hypertrophy", top_k=5)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import ProjectPaths, Settings

# Lazy imports — installed at runtime
def _import_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError as exc:
        raise ImportError("chromadb is not installed. Run: pip install chromadb") from exc

def _import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Embedding function wrapper for ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

class _BGEEmbeddingFunction:
    """ChromaDB-compatible embedding function using sentence-transformers BGE model."""

    BGE_INSTRUCTION = "Represent this fitness coaching passage for retrieval: "

    def __init__(self, model_name: str) -> None:
        SentenceTransformer = _import_sentence_transformers()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    # ChromaDB >= 0.6 requires an EmbeddingFunction protocol with .name()
    def name(self) -> str:
        return f"bge-embedding-{self.model_name.replace('/', '-')}"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        use_instruction = "bge" in self.model_name.lower()
        if use_instruction:
            texts = [self.BGE_INSTRUCTION + t for t in input]
        else:
            texts = input
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# ChromaRAG
# ─────────────────────────────────────────────────────────────────────────────

class ChromaRAG:
    """
    Three-collection ChromaDB RAG system for fitness coaching.

    Collections:
        exercises   — exercise descriptions, technique, muscles
        knowledge   — sports science, nutrition, programming, safety
        qa_pairs    — curated Q&A pairs for direct retrieval
    """

    COLLECTION_EXERCISES = "fitness_exercises"
    COLLECTION_KNOWLEDGE = "fitness_knowledge"
    COLLECTION_QA = "fitness_qa_pairs"

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.paths = ProjectPaths()
        self.settings = Settings()
        self.persist_dir = Path(persist_dir or (self.paths.rag_dir / "chroma"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Prefer BGE-large; fall back to MiniLM if BGE is not available
        self.embedding_model_name = embedding_model or self.settings.bge_embedding_model
        self._embed_fn: _BGEEmbeddingFunction | None = None
        self._client = None
        self._col_exercises = None
        self._col_knowledge = None
        self._col_qa = None

    # ── Initialisation ───────────────────────────────────────────────────────

    def _get_embed_fn(self) -> _BGEEmbeddingFunction:
        if self._embed_fn is None:
            try:
                self._embed_fn = _BGEEmbeddingFunction(self.embedding_model_name)
            except Exception:
                # Fallback
                fallback = "sentence-transformers/all-MiniLM-L6-v2"
                self._embed_fn = _BGEEmbeddingFunction(fallback)
        return self._embed_fn

    def _get_client(self):
        if self._client is None:
            chromadb = _import_chromadb()
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def _get_collections(self):
        client = self._get_client()
        embed_fn = self._get_embed_fn()
        if self._col_exercises is None:
            self._col_exercises = client.get_or_create_collection(
                name=self.COLLECTION_EXERCISES,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        if self._col_knowledge is None:
            self._col_knowledge = client.get_or_create_collection(
                name=self.COLLECTION_KNOWLEDGE,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        if self._col_qa is None:
            self._col_qa = client.get_or_create_collection(
                name=self.COLLECTION_QA,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._col_exercises, self._col_knowledge, self._col_qa

    # ── Build / Index ─────────────────────────────────────────────────────────

    def build(
        self,
        exercise_dict_path: Path | None = None,
        domain_dict_path: Path | None = None,
        training_data_path: Path | None = None,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        """Index all sources into ChromaDB collections."""
        ex_path = exercise_dict_path or (self.paths.artifacts_dir / "dictionaries" / "master_exercise_dictionary.jsonl")
        dom_path = domain_dict_path or (self.paths.artifacts_dir / "dictionaries" / "master_domain_dictionary.jsonl")
        qa_path = training_data_path or (self.paths.datasets_dir / "master_training_data_clean.jsonl")

        col_ex, col_know, col_qa = self._get_collections()

        results: dict[str, Any] = {}

        # Exercise collection
        ex_count = col_ex.count()
        if force_rebuild or ex_count == 0:
            exercises = self._load_exercise_chunks(ex_path)
            if exercises:
                col_ex.upsert(
                    ids=[e["id"] for e in exercises],
                    documents=[e["text"] for e in exercises],
                    metadatas=[e["metadata"] for e in exercises],
                )
            results["exercises_indexed"] = len(exercises)
        else:
            results["exercises_indexed"] = ex_count

        # Knowledge collection
        know_count = col_know.count()
        if force_rebuild or know_count == 0:
            knowledge = self._load_knowledge_chunks(dom_path)
            if knowledge:
                col_know.upsert(
                    ids=[k["id"] for k in knowledge],
                    documents=[k["text"] for k in knowledge],
                    metadatas=[k["metadata"] for k in knowledge],
                )
            results["knowledge_indexed"] = len(knowledge)
        else:
            results["knowledge_indexed"] = know_count

        # Q&A collection
        qa_count = col_qa.count()
        if force_rebuild or qa_count == 0:
            qa_chunks = self._load_qa_chunks(qa_path)
            if qa_chunks:
                col_qa.upsert(
                    ids=[q["id"] for q in qa_chunks],
                    documents=[q["text"] for q in qa_chunks],
                    metadatas=[q["metadata"] for q in qa_chunks],
                )
            results["qa_indexed"] = len(qa_chunks)
        else:
            results["qa_indexed"] = qa_count

        results["status"] = "ok"
        results["persist_dir"] = str(self.persist_dir)
        return results

    # ── Chunk loaders ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_exercise_chunks(path: Path) -> list[dict]:
        if not path.exists():
            return []
        chunks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = ex.get("name", "")
            if not name:
                continue
            primary = ", ".join(ex.get("primary_muscles", [])) or "various muscles"
            secondary = ", ".join(ex.get("secondary_muscles", [])) or ""
            equipment = ex.get("equipment", "bodyweight")
            difficulty = ex.get("difficulty", "intermediate")
            instructions = ex.get("instructions", "")[:600].strip()
            pattern = ex.get("movement_pattern", "")
            text = (
                f"Exercise: {name}\n"
                f"Primary muscles: {primary}\n"
                f"Secondary muscles: {secondary or 'none'}\n"
                f"Equipment: {equipment}\n"
                f"Difficulty: {difficulty}\n"
                f"Movement pattern: {pattern}\n"
                f"Instructions: {instructions or 'See exercise database.'}"
            )
            tags = ex.get("tags", [])
            chunks.append({
                "id": ex.get("id", f"ex_{len(chunks)}"),
                "text": text,
                "metadata": {
                    "name": name,
                    "primary_muscles": ",".join(ex.get("primary_muscles", [])),
                    "equipment": equipment,
                    "difficulty": difficulty,
                    "movement_pattern": pattern,
                    "source": ex.get("source", ""),
                    "topic": "exercise",
                    "tags": ",".join(tags),
                },
            })
        return chunks

    @staticmethod
    def _load_knowledge_chunks(path: Path) -> list[dict]:
        if not path.exists():
            return []
        chunks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = item.get("body", item.get("text", "")).strip()
            if not body or len(body) < 30:
                continue
            item_type = item.get("type", "concept")
            title = item.get("title", body[:60])
            tags = item.get("tags", [])
            topic_map = {
                "nutrition_rule": "nutrition",
                "safety_guideline": "safety",
                "programming_principle": "programming",
                "sports_science": "science",
            }
            topic = topic_map.get(item_type, "general")
            text = f"Topic: {title}\n{body}"
            chunks.append({
                "id": item.get("id", f"know_{len(chunks)}"),
                "text": text,
                "metadata": {
                    "title": title,
                    "type": item_type,
                    "topic": topic,
                    "tags": ",".join(tags),
                    "source": item.get("source", ""),
                },
            })
        return chunks

    @staticmethod
    def _load_qa_chunks(path: Path, max_samples: int = 5000) -> list[dict]:
        if not path.exists():
            return []
        chunks = []
        import hashlib
        for line in path.read_text(encoding="utf-8").splitlines():
            if len(chunks) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages", [])
            if len(msgs) < 3:
                continue
            q = msgs[1].get("content", "").strip()
            a = msgs[2].get("content", "").strip()
            if not q or len(a) < 40:
                continue
            uid = hashlib.md5((q[:80] + a[:80]).encode()).hexdigest()[:16]
            topic = _infer_topic(q + " " + a)
            text = f"Q: {q}\nA: {a}"
            chunks.append({
                "id": f"qa_{uid}",
                "text": text,
                "metadata": {
                    "question": q[:120],
                    "topic": topic,
                    "source": "training_data",
                },
            })
        return chunks

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        intent: str | None = None,
        filters: dict | None = None,
        include_qa: bool = True,
    ) -> list[dict]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query:      The user question / retrieval query.
            top_k:      Number of results to return per collection.
            intent:     Optional intent tag to bias collection selection.
            filters:    Optional ChromaDB where-clause metadata filter.
            include_qa: Whether to search the Q&A collection.

        Returns:
            List of dicts with keys: text, metadata, score, collection.
        """
        col_ex, col_know, col_qa = self._get_collections()
        results: list[dict] = []

        # Determine which collections to search based on intent
        search_exercises = True
        search_knowledge = True
        search_qa = include_qa

        if intent:
            if intent in ("EXERCISE_GUIDANCE", "EXERCISE_SUBSTITUTION", "TECHNIQUE"):
                search_knowledge = False
            elif intent in ("SCIENCE_EDUCATION", "NUTRITION", "PROGRAMMING"):
                search_exercises = False

        def _query_collection(col, where: dict | None = None) -> list[dict]:
            try:
                kwargs: dict[str, Any] = {"query_texts": [query], "n_results": top_k}
                if where:
                    kwargs["where"] = where
                res = col.query(**kwargs)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                distances = res.get("distances", [[]])[0]
                out = []
                for doc, meta, dist in zip(docs, metas, distances):
                    # ChromaDB cosine distance: 0=identical, 2=opposite
                    # Convert to similarity score (0–1)
                    score = 1.0 - (dist / 2.0) if dist is not None else 0.0
                    out.append({"text": doc, "metadata": meta, "score": round(score, 4), "collection": col.name})
                return out
            except Exception:
                return []

        if search_exercises:
            ex_filter = filters.get("exercises") if filters else None
            results.extend(_query_collection(col_ex, ex_filter))

        if search_knowledge:
            know_filter = filters.get("knowledge") if filters else None
            results.extend(_query_collection(col_know, know_filter))

        if search_qa:
            qa_filter = filters.get("qa") if filters else None
            results.extend(_query_collection(col_qa, qa_filter))

        # Sort by score descending, deduplicate by text prefix
        results.sort(key=lambda x: x["score"], reverse=True)
        seen_prefixes: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            prefix = r["text"][:80]
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                deduped.append(r)

        return deduped[:top_k * 2]  # return up to 2×top_k for caller to further filter

    def format_context(
        self,
        results: list[dict],
        threshold: float = 0.25,
        max_chars: int = 2000,
    ) -> str:
        """Format search results into a context string for the LLM prompt."""
        selected = [r for r in results if r["score"] >= threshold]
        if not selected:
            return ""
        parts = []
        total = 0
        for r in selected:
            text = r["text"]
            if total + len(text) > max_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n\n---\n\n".join(parts)

    # ── Convenience ───────────────────────────────────────────────────────────

    def search_exercises(
        self,
        query: str,
        muscle: str | None = None,
        equipment: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Search only the exercises collection with optional metadata filters."""
        col_ex, _, _ = self._get_collections()
        where: dict | None = None
        conditions = []
        if muscle:
            conditions.append({"primary_muscles": {"$contains": muscle.lower()}})
        if equipment:
            conditions.append({"equipment": {"$eq": equipment.lower()}})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}
        try:
            res = col_ex.query(query_texts=[query], n_results=top_k, where=where)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            return [
                {"text": d, "metadata": m, "score": round(1.0 - (dist / 2.0), 4)}
                for d, m, dist in zip(docs, metas, dists)
            ]
        except Exception:
            return []

    def is_built(self) -> bool:
        """Returns True if all collections have been indexed."""
        try:
            col_ex, col_know, _ = self._get_collections()
            return col_ex.count() > 0 and col_know.count() > 0
        except Exception:
            return False

    def stats(self) -> dict[str, int]:
        """Return item counts for each collection."""
        try:
            col_ex, col_know, col_qa = self._get_collections()
            return {
                "exercises": col_ex.count(),
                "knowledge": col_know.count(),
                "qa_pairs": col_qa.count(),
            }
        except Exception:
            return {"exercises": 0, "knowledge": 0, "qa_pairs": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _infer_topic(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ("protein", "calorie", "macro", "nutrient", "supplement", "creatine", "diet", "food", "eat", "carb", "fat intake")):
        return "nutrition"
    if any(x in t for x in ("injury", "pain", "safe", "warm up", "red flag", "medical", "impingement", "tendon")):
        return "safety"
    if any(x in t for x in ("split", "program", "volume", "frequency", "deload", "sets per week", "periodization", "rpe")):
        return "programming"
    if any(x in t for x in ("hypertrophy", "mps", "epoc", "muscle protein", "progressive overload", "science")):
        return "science"
    if any(x in t for x in ("exercise", "muscle", "form", "technique", "cue", "rep", "set", "squat", "deadlift", "press", "row")):
        return "exercise"
    return "general"
