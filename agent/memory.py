import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

DB_PATH = Path("db/agent_memory.db")


def init_db() -> None:
    """Initialize SQLite schema for agent state persistence."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            mandate_raw  TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            status       TEXT DEFAULT 'running'
        );

        CREATE TABLE IF NOT EXISTS search_results (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     TEXT NOT NULL,
            query          TEXT NOT NULL,
            url            TEXT NOT NULL,
            title          TEXT,
            classification TEXT,
            confidence     REAL,
            reasoning      TEXT,
            created_at     TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS agent_decisions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            node_name   TEXT NOT NULL,
            decision    TEXT NOT NULL,
            reasoning   TEXT,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
    """)
    conn.commit()
    conn.close()


def save_session(session_id: str, mandate_raw: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)",
        (session_id, mandate_raw, datetime.now().isoformat(), "running")
    )
    conn.commit()
    conn.close()


def update_session_status(session_id: str, status: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE sessions SET status = ? WHERE session_id = ?",
        (status, session_id)
    )
    conn.commit()
    conn.close()


def save_result(
    session_id: str, query: str, url: str, title: str,
    classification: str, confidence: float, reasoning: str
) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO search_results VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, query, url, title, classification,
         confidence, reasoning, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def log_decision(session_id: str, node_name: str, decision: str, reasoning: str = "") -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO agent_decisions VALUES (NULL, ?, ?, ?, ?, ?)",
        (session_id, node_name, decision, reasoning, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_session_results(session_id: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    results = conn.execute(
        "SELECT * FROM search_results WHERE session_id = ? ORDER BY created_at",
        (session_id,)
    ).fetchall()
    conn.close()
    return results
