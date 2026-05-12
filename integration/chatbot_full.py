"""
Complete FastAPI chatbot for the Personalized AI Fitness Coaching Assistant.

Implements the full architecture from the implementation guide:
  - User context engine (profiles, goals, injuries, workout history)
  - ChromaDB RAG retrieval (exercises + knowledge + Q&A)
  - Intent detection for targeted retrieval
  - LLM response engine (Groq cloud OR local fine-tuned model)
  - Conversation history management

Endpoints:
  POST /api/v1/chat               — main chat
  POST /api/v1/user               — create / upsert user profile
  GET  /api/v1/user/{user_id}     — get user profile
  PATCH /api/v1/user/{user_id}    — update profile fields
  GET  /api/v1/program/{user_id}  — get current workout program
  POST /api/v1/program/{user_id}  — save / update workout program
  POST /api/v1/workout/{user_id}  — log a completed workout session
  GET  /api/v1/workout/{user_id}  — get workout history
  GET  /api/v1/rag/stats          — vector DB stats
  POST /api/v1/rag/build          — (re-)build the RAG index
  GET  /health                    — health check

Usage:
    uvicorn integration.chatbot_full:app --reload --port 8000
"""
from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Internal imports
from fitness_llm.chroma_rag import ChromaRAG
from fitness_llm.config import ProjectPaths, Settings
from fitness_llm.llm_engine import FitnessLLM
from fitness_llm.prompts import build_intent_detection_prompt, build_system_prompt
from fitness_llm.user_context import UserContextManager

# ─────────────────────────────────────────────────────────────────────────────
# App globals
# ─────────────────────────────────────────────────────────────────────────────

_settings = Settings()
_paths = ProjectPaths()

_rag = ChromaRAG()
_llm = FitnessLLM()
_ucm = UserContextManager()

# ─────────────────────────────────────────────────────────────────────────────
# Intent categories and retrieval routing
# ─────────────────────────────────────────────────────────────────────────────

INTENT_LABELS = {
    "EXERCISE_GUIDANCE", "EXERCISE_SUBSTITUTION", "PROGRAM_MODIFICATION",
    "NUTRITION", "SCIENCE_EDUCATION", "INJURY_SAFETY", "WORKOUT_PLANNING",
    "MOTIVATION", "GENERAL",
}

INTENT_EXERCISE_FOCUS = {"EXERCISE_GUIDANCE", "EXERCISE_SUBSTITUTION"}
INTENT_KNOWLEDGE_FOCUS = {"NUTRITION", "SCIENCE_EDUCATION", "INJURY_SAFETY", "PROGRAMMING", "PROGRAM_MODIFICATION"}

# Regex-based fast intent detection (fallback before LLM call)
_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(how\s+do\s+i|technique|form|cue|perform|correct|step[s]?)\b.*(exercise|squat|deadlift|bench|press|row|curl|extension|pull|push|lunge|dip)", re.I), "EXERCISE_GUIDANCE"),
    (re.compile(r"\b(replace|substitut|instead|alternative|without|don.?t have|no\s+access)\b", re.I), "EXERCISE_SUBSTITUTION"),
    (re.compile(r"\b(protein|calori|macro|supplement|creatine|diet|eat|food|carb|nutrient|hydrat)\b", re.I), "NUTRITION"),
    (re.compile(r"\b(pain|injur|hurt|sore|impingement|tendon|ligament|red.?flag|safe|avoid|warning)\b", re.I), "INJURY_SAFETY"),
    (re.compile(r"\b(hypertrophy|progressive overload|volume|frequency|deload|rpe|rir|muscle protein|science|research)\b", re.I), "SCIENCE_EDUCATION"),
    (re.compile(r"\b(program|split|workout plan|schedule|design|structure|week|days per week)\b", re.I), "WORKOUT_PLANNING"),
    (re.compile(r"\b(increase|plateau|stall|progress|overload|when\s+should\s+i\s+add)\b", re.I), "PROGRAM_MODIFICATION"),
    (re.compile(r"\b(motivat|habit|consistenc|adherence|stay|keep\s+going|unmotivated)\b", re.I), "MOTIVATION"),
]


def detect_intent(message: str) -> str:
    """Fast regex-based intent detection."""
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(message):
            return intent
    return "GENERAL"


def build_retrieval_query(
    message: str,
    intent: str,
    user_profile: dict | None = None,
) -> str:
    """Enrich the retrieval query with user context."""
    parts = [message]
    if user_profile:
        level = user_profile.get("level", "")
        goal = user_profile.get("goal", "")
        if level:
            parts.append(f"user level: {level}")
        if goal:
            parts.append(f"user goal: {goal}")
    if intent == "EXERCISE_GUIDANCE":
        parts.append("exercise technique form instructions muscles")
    elif intent == "NUTRITION":
        parts.append("nutrition protein calories diet recommendations")
    elif intent == "SCIENCE_EDUCATION":
        parts.append("sports science hypertrophy training principles")
    elif intent == "INJURY_SAFETY":
        parts.append("injury safety pain modification contraindications")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: load RAG + LLM on startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Try to load ChromaDB RAG
    try:
        if not _rag.is_built():
            print("[startup] ChromaDB index not found — run POST /api/v1/rag/build to index.")
        else:
            print(f"[startup] ChromaDB RAG loaded: {_rag.stats()}")
    except Exception as e:
        print(f"[startup] ChromaDB warning: {e}")

    # Try to load FAISS RAG (legacy fallback)
    try:
        _llm.load_rag()
        print("[startup] FAISS RAG index loaded (legacy fallback).")
    except Exception:
        print("[startup] FAISS RAG index not available.")

    # Try to load local model (optional — Groq is primary)
    try:
        _llm.load_local_model()
        print("[startup] Local model loaded.")
    except Exception:
        print("[startup] No local model — using Groq API.")

    yield


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fitness AI Coach API",
    description="Personalized AI fitness coaching chatbot with RAG + LoRA",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str | None = Field(default=None, description="Optional user ID for personalised context")
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")
    message: str = Field(..., min_length=1, max_length=2000)
    user_profile: dict[str, Any] | None = Field(default=None, description="Inline user profile (overrides stored)")
    current_plan: dict[str, Any] | None = None
    mode: str = Field(default="auto", pattern="^(auto|groq|local)$")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    mode_used: str
    context_used: bool


class UserProfileRequest(BaseModel):
    user_id: str | None = None
    level: str = Field(default="intermediate", pattern="^(beginner|intermediate|advanced)$")
    goal: str = Field(default="general_fitness")
    days_per_week: int = Field(default=4, ge=1, le=7)
    equipment: list[str] = Field(default_factory=list)
    injuries: list[str] = Field(default_factory=list)
    age: int | None = Field(default=None, ge=13, le=100)
    weight_kg: float | None = Field(default=None, gt=0)
    diet_type: str = Field(default="standard")
    preferred_style: str = Field(default="any")


class ProgramRequest(BaseModel):
    split: str = ""
    days_per_week: int = Field(default=4, ge=1, le=7)
    microcycle: list[dict] = Field(default_factory=list)
    notes: str = ""


class WorkoutLogRequest(BaseModel):
    session_name: str = "Workout"
    exercises: list[dict] = Field(default_factory=list)
    duration_min: int | None = None
    rpe: float | None = Field(default=None, ge=1, le=10)
    notes: str = ""
    session_date: str | None = None


class RagBuildRequest(BaseModel):
    force_rebuild: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    rag_stats = {}
    try:
        rag_stats = _rag.stats()
    except Exception:
        pass
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "chroma_rag": rag_stats,
        "faiss_rag_loaded": _llm.index is not None,
        "local_model_loaded": _llm.model is not None,
        "groq_available": _llm.groq_client is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Flow:
      1. Resolve user profile (stored DB → inline payload → anonymous)
      2. Detect intent
      3. Build enriched retrieval query
      4. Retrieve context from ChromaDB (falls back to FAISS)
      5. Build full system prompt
      6. Generate response via Groq or local model
      7. Persist conversation turn
    """
    # ── 1. Resolve user context ───────────────────────────────────────────
    user_data: dict | None = None
    user_profile: dict | None = payload.user_profile
    user_context_section: str = ""

    if payload.user_id:
        user_data = _ucm.get_user(payload.user_id)
        if user_data:
            # Stored profile takes precedence unless inline override provided
            if not user_profile:
                user_profile = {
                    "level": user_data["profile"].get("level", "intermediate"),
                    "goal": user_data["profile"].get("goal", "general_fitness"),
                    "days": user_data["program"].get("days_per_week", 4),
                    "injuries": ", ".join(user_data["profile"].get("constraints", {}).get("injuries", [])),
                    "equipment": ", ".join(user_data["profile"].get("constraints", {}).get("equipment", [])),
                }
            user_context_section = _ucm.build_context_prompt_section(payload.user_id)

    # ── 2. Detect intent ─────────────────────────────────────────────────
    intent = detect_intent(payload.message)

    # ── 3. Build retrieval query ──────────────────────────────────────────
    retrieval_query = build_retrieval_query(payload.message, intent, user_profile)

    # ── 4. Retrieve context ───────────────────────────────────────────────
    retrieved_context = ""
    context_used = False

    # Try ChromaDB first
    try:
        if _rag.is_built():
            results = _rag.search(
                retrieval_query,
                top_k=4,
                intent=intent,
                include_qa=(intent not in INTENT_EXERCISE_FOCUS),
            )
            retrieved_context = _rag.format_context(
                results,
                threshold=_settings.retrieval_threshold,
                max_chars=1800,
            )
            context_used = bool(retrieved_context)
    except Exception:
        pass

    # Fall back to FAISS legacy RAG
    if not context_used:
        try:
            retrieved_context = _llm.retrieve_context(retrieval_query, top_k=3)
            context_used = bool(retrieved_context)
        except Exception:
            pass

    # ── 5. Build prompt ───────────────────────────────────────────────────
    system_prompt = build_system_prompt(
        user_profile=user_profile,
        current_plan=payload.current_plan,
        retrieved_context=retrieved_context or None,
        user_context_section=user_context_section or None,
    )

    # ── 6. Resolve session history ────────────────────────────────────────
    history: list[dict] = []
    session_id = payload.session_id or ""
    if payload.user_id:
        session_id = _ucm.get_or_create_session(payload.user_id, session_id or None)
        raw_history = _ucm.get_session_history(session_id, last_n=6)
        # Convert to {user, bot} format expected by LLM engine
        i = 0
        while i < len(raw_history) - 1:
            if raw_history[i]["role"] == "user" and raw_history[i + 1]["role"] == "assistant":
                history.append({
                    "user": raw_history[i]["content"],
                    "bot": raw_history[i + 1]["content"],
                })
                i += 2
            else:
                i += 1

    # ── 7. Generate reply ─────────────────────────────────────────────────
    reply = ""
    mode_used = payload.mode

    if payload.mode == "local":
        try:
            reply = _llm.chat_local(
                payload.message,
                user_profile=user_profile,
                current_plan=payload.current_plan,
                history=history,
            )
            mode_used = "local"
        except RuntimeError:
            payload = payload.model_copy(update={"mode": "groq"})

    if not reply:
        try:
            reply = _chat_groq_with_prompt(
                message=payload.message,
                system_prompt=system_prompt,
                history=history,
            )
            mode_used = "groq"
        except Exception as exc:
            # Last resort: try local model
            try:
                reply = _llm.chat_local(
                    payload.message,
                    user_profile=user_profile,
                    current_plan=payload.current_plan,
                    history=history,
                )
                mode_used = "local"
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"No LLM backend available: {exc}",
                ) from exc

    # ── 8. Persist turn ───────────────────────────────────────────────────
    if payload.user_id and session_id:
        try:
            _ucm.append_session_turn(
                session_id,
                payload.message,
                reply,
                intent=intent,
                topic=intent.lower(),
            )
        except Exception:
            pass

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        intent=intent,
        mode_used=mode_used,
        context_used=context_used,
    )


def _chat_groq_with_prompt(
    message: str,
    system_prompt: str,
    history: list[dict],
) -> str:
    """Call Groq API with the pre-built system prompt and conversation history."""
    if _llm.groq_client is None:
        raise RuntimeError("Groq client not initialised — set GROQ_API_KEY.")

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in history[-4:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": message})

    response = _llm.groq_client.chat.completions.create(
        model=_settings.groq_model,
        messages=messages,
        max_tokens=_settings.max_new_tokens,
        temperature=_settings.temperature,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# User profile endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/user", status_code=status.HTTP_201_CREATED)
def create_user(payload: UserProfileRequest) -> dict[str, Any]:
    """Create or upsert a user profile."""
    profile: dict[str, Any] = {
        "level": payload.level,
        "goal": payload.goal,
        "training_age_years": None,
        "constraints": {
            "injuries": payload.injuries,
            "equipment": payload.equipment,
            "schedule": f"{payload.days_per_week} days/week",
        },
        "preferences": {
            "diet_type": payload.diet_type,
            "preferred_training_style": payload.preferred_style,
        },
        "demographics": {},
    }
    if payload.age:
        profile["demographics"]["age"] = payload.age
    if payload.weight_kg:
        profile["demographics"]["weight_kg"] = payload.weight_kg

    if payload.user_id:
        user_id = _ucm.upsert_user(payload.user_id, profile)
        action = "updated"
    else:
        user_id = _ucm.create_user(profile)
        action = "created"

    return {"user_id": user_id, "action": action, "profile": profile}


@app.get("/api/v1/user/{user_id}")
def get_user(user_id: str) -> dict[str, Any]:
    """Retrieve a user's full profile and state."""
    user = _ucm.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return user


@app.patch("/api/v1/user/{user_id}")
def update_user(user_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Partially update a user's profile fields."""
    ok = _ucm.update_profile(user_id, updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return {"user_id": user_id, "updated": True}


# ─────────────────────────────────────────────────────────────────────────────
# Program endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/program/{user_id}")
def get_program(user_id: str) -> dict[str, Any]:
    """Get the current workout program for a user."""
    user = _ucm.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return {"user_id": user_id, "program": user["program"]}


@app.post("/api/v1/program/{user_id}")
def save_program(user_id: str, payload: ProgramRequest) -> dict[str, Any]:
    """Save or update a user's workout program."""
    program: dict[str, Any] = {
        "split": payload.split,
        "days_per_week": payload.days_per_week,
        "microcycle": payload.microcycle,
        "notes": payload.notes,
        "program_start_date": datetime.utcnow().date().isoformat(),
        "current_week": 1,
    }
    ok = _ucm.update_program(user_id, program)
    if not ok:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return {"user_id": user_id, "program": program}


# ─────────────────────────────────────────────────────────────────────────────
# Workout log endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/workout/{user_id}", status_code=status.HTTP_201_CREATED)
def log_workout(user_id: str, payload: WorkoutLogRequest) -> dict[str, Any]:
    """Log a completed workout session."""
    user = _ucm.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    log_id = _ucm.log_workout(
        user_id=user_id,
        session_name=payload.session_name,
        exercises=payload.exercises,
        duration_min=payload.duration_min,
        rpe=payload.rpe,
        notes=payload.notes,
        session_date=payload.session_date,
    )
    return {"log_id": log_id, "user_id": user_id, "status": "logged"}


@app.get("/api/v1/workout/{user_id}")
def get_workout_history(user_id: str, limit: int = 10) -> dict[str, Any]:
    """Retrieve workout history for a user."""
    user = _ucm.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    history = _ucm.get_workout_history(user_id, limit=limit)
    return {"user_id": user_id, "workouts": history, "total": len(history)}


# ─────────────────────────────────────────────────────────────────────────────
# RAG management endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/rag/stats")
def rag_stats() -> dict[str, Any]:
    """Return ChromaDB collection stats."""
    try:
        stats = _rag.stats()
        built = _rag.is_built()
    except Exception as exc:
        return {"error": str(exc), "built": False}
    return {"built": built, "collections": stats}


@app.post("/api/v1/rag/build")
def rag_build(payload: RagBuildRequest) -> dict[str, Any]:
    """Build or rebuild the ChromaDB RAG index from all dictionary sources."""
    try:
        result = _rag.build(force_rebuild=payload.force_rebuild)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"RAG build failed: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Exercise search endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/exercises/search")
def search_exercises(q: str, muscle: str | None = None, equipment: str | None = None, top_k: int = 5) -> dict[str, Any]:
    """Semantic exercise search with optional muscle/equipment filters."""
    try:
        results = _rag.search_exercises(q, muscle=muscle, equipment=equipment, top_k=top_k)
        return {"query": q, "results": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
