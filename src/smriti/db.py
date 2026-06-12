"""SQLite storage layer with audit log. Zero external dependencies."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    importance REAL NOT NULL DEFAULT 0.5,
    embedding TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    emotion TEXT NOT NULL DEFAULT '',
    arousal REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS emotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    emotion TEXT NOT NULL,
    intensity REAL NOT NULL,
    cause TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    task TEXT NOT NULL,
    success INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from REAL NOT NULL,
    valid_to REAL,                -- NULL = currently valid
    source TEXT,
    embedding TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    markdown TEXT NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL,
    embedding TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_sp ON facts(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Upgrade v0.1 databases in place (additive only)."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(events)")]
        if "emotion" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN emotion TEXT NOT NULL DEFAULT ''")
        if "arousal" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN arousal REAL NOT NULL DEFAULT 0")

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def now() -> float:
        return time.time()

    def audit(self, action: str, target: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO audit(ts, action, target, detail) VALUES (?,?,?,?)",
            (self.now(), action, target, detail),
        )
        self.conn.commit()

    @staticmethod
    def dump_vec(vec: list[float]) -> str:
        return json.dumps([round(v, 6) for v in vec])

    @staticmethod
    def load_vec(s: str) -> list[float]:
        return json.loads(s)

    def close(self) -> None:
        self.conn.close()
