from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from groq import Groq
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ProjectPaths, Settings
from .prompts import build_system_prompt


class FitnessLLM:
    def __init__(self) -> None:
        self.settings = Settings()
        self.paths = ProjectPaths()
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.index = None
        self.chunks = None
        self.groq_client = Groq(api_key=self.settings.groq_api_key) if self.settings.groq_api_key else None

    def load_rag(self, index_path: Path | None = None, chunks_path: Path | None = None) -> None:
        index_file = index_path or self.paths.rag_index_path
        chunks_file = chunks_path or self.paths.rag_chunks_path
        self.embedder = SentenceTransformer(self.settings.embedding_model)
        self.index = faiss.read_index(str(index_file))
        with chunks_file.open("rb") as handle:
            self.chunks = pickle.load(handle)

    def load_local_model(self, adapter_dir: Path | None = None, base_model_id: str | None = None) -> None:
        adapter_path = Path(adapter_dir or self.paths.default_adapter_dir)
        model_id = base_model_id or self.settings.base_model_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path) if adapter_path.exists() else model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if adapter_path.exists():
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            self.model = base_model
        self.model.eval()

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        if self.index is None or self.chunks is None or self.embedder is None:
            return ""

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)

        selected = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.settings.retrieval_threshold:
                selected.append(self.chunks[idx]["text"])
        return "\n\n".join(selected)

    def chat_local(
        self,
        message: str,
        user_profile: dict[str, Any] | None = None,
        current_plan: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local model is not loaded.")

        system_prompt = build_system_prompt(
            user_profile=user_profile,
            current_plan=current_plan,
            retrieved_context=self.retrieve_context(message),
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in (history or [])[-4:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["bot"]})
        messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def chat_groq(
        self,
        message: str,
        user_profile: dict[str, Any] | None = None,
        current_plan: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if self.groq_client is None:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        system_prompt = build_system_prompt(
            user_profile=user_profile,
            current_plan=current_plan,
            retrieved_context=self.retrieve_context(message),
        )
        messages = [{"role": "system", "content": system_prompt}]
        for turn in (history or [])[-4:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["bot"]})
        messages.append({"role": "user", "content": message})

        response = self.groq_client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
