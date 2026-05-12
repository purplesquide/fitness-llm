from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    datasets_dir: Path = ROOT_DIR / "artifacts" / "datasets"
    rag_dir: Path = ROOT_DIR / "artifacts" / "rag"
    models_dir: Path = ROOT_DIR / "artifacts" / "models"

    @property
    def dataset_path(self) -> Path:
        return self.datasets_dir / "fitness_training_data.jsonl"

    @property
    def rag_index_path(self) -> Path:
        return self.rag_dir / "fitness_knowledge.index"

    @property
    def rag_chunks_path(self) -> Path:
        return self.rag_dir / "fitness_knowledge_chunks.pkl"

    @property
    def default_adapter_dir(self) -> Path:
        return self.models_dir / "fitness-mistral-qlora-final"

    def ensure(self) -> None:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    base_model_id: str = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    hf_token: str | None = os.getenv("HF_TOKEN")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    # FAISS RAG embedding (lightweight fallback)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # BGE embedding for ChromaDB RAG (higher quality, recommended by guide)
    bge_embedding_model: str = os.getenv("BGE_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    retrieval_threshold: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.25"))
    chroma_persist_dir: str | None = os.getenv("CHROMA_PERSIST_DIR")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "400"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
