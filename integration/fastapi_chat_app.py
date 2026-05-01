from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fitness_llm.llm_engine import FitnessLLM


engine = FitnessLLM()


class ChatMessage(BaseModel):
    user_id: int = Field(..., ge=1)
    message: str = Field(..., min_length=1)
    user_profile: dict[str, Any] | None = None
    current_plan: dict[str, Any] | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: str = Field(default="auto", pattern="^(auto|local|groq)$")


class ChatResponse(BaseModel):
    reply: str
    mode_used: str
    context_used: bool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        engine.load_rag()
    except Exception:
        pass
    try:
        engine.load_local_model()
    except Exception:
        pass
    yield


app = FastAPI(title="Fitness LLM Chat API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "rag_loaded": engine.index is not None,
        "local_model_loaded": engine.model is not None,
        "groq_available": engine.groq_client is not None,
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatMessage) -> ChatResponse:
    try:
        if payload.mode == "local":
            reply = engine.chat_local(payload.message, payload.user_profile, payload.current_plan, payload.history)
            mode_used = "local"
        elif payload.mode == "groq":
            reply = engine.chat_groq(payload.message, payload.user_profile, payload.current_plan, payload.history)
            mode_used = "groq"
        else:
            if engine.model is not None:
                reply = engine.chat_local(payload.message, payload.user_profile, payload.current_plan, payload.history)
                mode_used = "local"
            elif engine.groq_client is not None:
                reply = engine.chat_groq(payload.message, payload.user_profile, payload.current_plan, payload.history)
                mode_used = "groq"
            else:
                raise RuntimeError("Neither a local model nor Groq client is available.")

        context_used = bool(engine.retrieve_context(payload.message))
        return ChatResponse(reply=reply, mode_used=mode_used, context_used=context_used)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
