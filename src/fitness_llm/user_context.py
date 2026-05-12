"""
User context manager for the fitness coaching chatbot.

Manages:
  - User profiles (goals, level, constraints, preferences)
  - Workout history (session logs)
  - Conversation state (short-term memory)

Storage: SQLite via Python's built-in sqlite3 (no ORM dependency).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator

from .config import ProjectPaths


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    profile     TEXT NOT NULL,   -- JSON
    program     TEXT NOT NULL,   -- JSON
    state       TEXT NOT NULL    -- JSON
);

CREATE TABLE IF NOT EXISTS workout_logs (
    log_id          TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    session_name    TEXT,
    exercises       TEXT NOT NULL,  -- JSON array
    duration_min    INTEGER,
    rpe             REAL,
    notes           TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    history         TEXT NOT NULL,  -- JSON array of {role, content}
    last_intent     TEXT,
    current_topic   TEXT
);

CREATE INDEX IF NOT EXISTS idx_workout_logs_user ON workout_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


# ─────────────────────────────────────────────────────────────────────────────
# User context manager
# ─────────────────────────────────────────────────────────────────────────────

class UserContextManager:
    """
    Manages user profiles, workout logs, and session state in SQLite.

    Thread-safety: Each call opens and closes its own connection. Use a single
    instance per process; do not share across processes.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        paths = ProjectPaths()
        self.db_path = db_path or (paths.artifacts_dir / "users.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ── Profile CRUD ─────────────────────────────────────────────────────────

    def create_user(self, profile: dict[str, Any], program: dict | None = None) -> str:
        """Create a new user and return their user_id."""
        user_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        empty_state: dict[str, Any] = {
            "last_workout_date": None,
            "total_sessions_logged": 0,
            "streak_days": 0,
            "recent_workouts": [],
            "adherence_rate_percent": 0.0,
            "notes": "",
        }
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?)",
                (
                    user_id, now, now,
                    json.dumps(profile),
                    json.dumps(program or {}),
                    json.dumps(empty_state),
                ),
            )
        return user_id

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return None
        return {
            "user_id": row["user_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "profile": json.loads(row["profile"]),
            "program": json.loads(row["program"]),
            "state": json.loads(row["state"]),
        }

    def update_profile(self, user_id: str, profile_update: dict[str, Any]) -> bool:
        user = self.get_user(user_id)
        if not user:
            return False
        merged = {**user["profile"], **profile_update}
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET profile=?, updated_at=? WHERE user_id=?",
                (json.dumps(merged), now, user_id),
            )
        return True

    def update_program(self, user_id: str, program: dict[str, Any]) -> bool:
        user = self.get_user(user_id)
        if not user:
            return False
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET program=?, updated_at=? WHERE user_id=?",
                (json.dumps(program), now, user_id),
            )
        return True

    def upsert_user(
        self,
        user_id: str,
        profile: dict[str, Any],
        program: dict | None = None,
    ) -> str:
        """Create user if not exists, otherwise update profile."""
        existing = self.get_user(user_id)
        if existing:
            self.update_profile(user_id, profile)
            if program:
                self.update_program(user_id, program)
            return user_id
        # Insert with provided user_id
        now = datetime.utcnow().isoformat()
        empty_state: dict[str, Any] = {
            "last_workout_date": None,
            "total_sessions_logged": 0,
            "streak_days": 0,
            "recent_workouts": [],
            "adherence_rate_percent": 0.0,
            "notes": "",
        }
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO users VALUES (?,?,?,?,?,?)",
                (
                    user_id, now, now,
                    json.dumps(profile),
                    json.dumps(program or {}),
                    json.dumps(empty_state),
                ),
            )
        return user_id

    # ── Workout logs ─────────────────────────────────────────────────────────

    def log_workout(
        self,
        user_id: str,
        session_name: str,
        exercises: list[dict],
        duration_min: int | None = None,
        rpe: float | None = None,
        notes: str = "",
        session_date: str | None = None,
    ) -> str:
        """Log a completed workout session. Returns log_id."""
        log_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        s_date = session_date or date.today().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO workout_logs VALUES (?,?,?,?,?,?,?,?,?)",
                (log_id, user_id, s_date, session_name, json.dumps(exercises),
                 duration_min, rpe, notes, now),
            )
        # Update user state
        self._refresh_user_state(user_id)
        return log_id

    def get_workout_history(self, user_id: str, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM workout_logs WHERE user_id=? ORDER BY session_date DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "log_id": r["log_id"],
                "session_date": r["session_date"],
                "session_name": r["session_name"],
                "exercises": json.loads(r["exercises"]),
                "duration_min": r["duration_min"],
                "rpe": r["rpe"],
                "notes": r["notes"],
            }
            for r in rows
        ]

    def _refresh_user_state(self, user_id: str) -> None:
        history = self.get_workout_history(user_id, limit=10)
        total = self._count_workouts(user_id)
        last_date = history[0]["session_date"] if history else None

        # Compute simple streak (consecutive days with logged workouts)
        if history:
            dates = sorted({h["session_date"] for h in history}, reverse=True)
            streak = 1
            for i in range(1, len(dates)):
                d1 = date.fromisoformat(dates[i - 1])
                d2 = date.fromisoformat(dates[i])
                if (d1 - d2).days <= 2:  # allow 1-day gap
                    streak += 1
                else:
                    break
        else:
            streak = 0

        state: dict[str, Any] = {
            "last_workout_date": last_date,
            "total_sessions_logged": total,
            "streak_days": streak,
            "recent_workouts": history[:5],
            "adherence_rate_percent": 0.0,
            "notes": "",
        }
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET state=?, updated_at=? WHERE user_id=?",
                (json.dumps(state), now, user_id),
            )

    def _count_workouts(self, user_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM workout_logs WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["cnt"] if row else 0

    # ── Session / conversation state ─────────────────────────────────────────

    def get_or_create_session(self, user_id: str, session_id: str | None = None) -> str:
        """Return an existing session_id or create a new one."""
        if session_id:
            with self._conn() as conn:
                row = conn.execute("SELECT session_id FROM sessions WHERE session_id=? AND user_id=?",
                                   (session_id, user_id)).fetchone()
            if row:
                return session_id

        new_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                (new_id, user_id, now, now, json.dumps([]), None, None),
            )
        return new_id

    def get_session_history(self, session_id: str, last_n: int = 6) -> list[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT history FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return []
        history: list[dict] = json.loads(row["history"])
        return history[-last_n:]

    def append_session_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        intent: str | None = None,
        topic: str | None = None,
    ) -> None:
        with self._conn() as conn:
            row = conn.execute("SELECT history FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                return
            history: list[dict] = json.loads(row["history"])
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})
            # Keep only last 20 turns (10 exchanges)
            history = history[-20:]
            now = datetime.utcnow().isoformat()
            conn.execute(
                "UPDATE sessions SET history=?, updated_at=?, last_intent=?, current_topic=? WHERE session_id=?",
                (json.dumps(history), now, intent, topic, session_id),
            )

    def build_context_prompt_section(self, user_id: str) -> str:
        """Build the USER PROFILE section of the RAG prompt."""
        user = self.get_user(user_id)
        if not user:
            return ""
        profile = user["profile"]
        program = user["program"]
        state = user["state"]

        lines = [
            f"- Experience level: {profile.get('level', 'intermediate')}",
            f"- Goal: {profile.get('goal', 'general_fitness')}",
        ]
        demo = profile.get("demographics", {})
        if demo.get("age"):
            lines.append(f"- Age: {demo['age']}")
        if demo.get("weight_kg"):
            lines.append(f"- Weight: {demo['weight_kg']} kg")
        constraints = profile.get("constraints", {})
        if constraints.get("injuries"):
            lines.append(f"- Injuries/limitations: {', '.join(constraints['injuries'])}")
        if constraints.get("equipment"):
            lines.append(f"- Equipment: {', '.join(constraints['equipment'])}")
        if constraints.get("schedule"):
            lines.append(f"- Schedule: {constraints['schedule']}")
        if program.get("split"):
            lines.append(f"- Current split: {program['split']}")
        recent = state.get("recent_workouts", [])
        if recent:
            last = recent[0]
            lines.append(f"- Last session: {last.get('session_name', 'workout')} on {last.get('session_date', 'recently')}")
        if state.get("streak_days"):
            lines.append(f"- Training streak: {state['streak_days']} consecutive days")

        return "\n".join(lines)
