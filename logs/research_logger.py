"""
Research Logger
================
SQLite-backed logger for tracking all experiment activity automatically:
  - Experiment runs and results
  - Model training events
  - File changes (via git diff snapshots)
  - Custom key-value events
  - AI agent prompts / responses (call log_agent_interaction)

Usage:
    from logs.research_logger import log_event, log_agent_interaction, query_log

    log_event("eval_result", {"method": "static_25", "ppl": 35.2})
    log_agent_interaction(prompt="...", response="...", context="critic_train")
    rows = query_log(event_type="eval_result")
"""

import sqlite3
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# DB stored in the repo root so it travels with the code
_DB_PATH = Path(__file__).parent / "research.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema():
    with _get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            event_type  TEXT    NOT NULL,
            payload     TEXT    NOT NULL,
            git_commit  TEXT,
            run_id      TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_interactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            context     TEXT,
            prompt      TEXT    NOT NULL,
            response    TEXT    NOT NULL,
            model       TEXT,
            run_id      TEXT
        );

        CREATE TABLE IF NOT EXISTS file_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            filename    TEXT    NOT NULL,
            diff        TEXT,
            git_commit  TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_event_ts   ON events(ts);
        """)


_ensure_schema()


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).parent.parent),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _run_id() -> str:
    """Stable run id per process (PID + startup time)."""
    return os.environ.get("RESEARCH_RUN_ID", f"pid{os.getpid()}")


def log_event(event_type: str, payload: dict):
    """
    Log a structured research event.

    Args:
        event_type: category string, e.g. "eval_result", "run_start", "model_saved"
        payload: dict of arbitrary key-value pairs
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO events (ts, event_type, payload, git_commit, run_id) VALUES (?,?,?,?,?)",
                (ts, event_type, json.dumps(payload, default=str), _git_head(), _run_id()),
            )
    except Exception:
        pass  # logging must never crash the main process


def log_agent_interaction(prompt: str, response: str, context: str = "", model: str = ""):
    """
    Record an AI agent prompt + response pair.

    Args:
        prompt: the prompt sent to the agent
        response: the agent's response
        context: human-readable context label, e.g. "critic_train_debug"
        model: model identifier string, e.g. "claude-sonnet-4.6"
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO agent_interactions (ts, context, prompt, response, model, run_id) VALUES (?,?,?,?,?,?)",
                (ts, context, prompt, response, model, _run_id()),
            )
    except Exception:
        pass


def log_file_snapshot(filename: str):
    """
    Capture and store the current git diff for a file.
    Call this whenever you modify a file programmatically.

    Args:
        filename: relative path from repo root
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", filename],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode()
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO file_snapshots (ts, filename, diff, git_commit) VALUES (?,?,?,?)",
                (ts, filename, diff, _git_head()),
            )
    except Exception:
        pass


def query_log(event_type: str = None, limit: int = 50) -> list[dict]:
    """
    Query logged events.

    Args:
        event_type: filter by event type; None returns all
        limit: maximum rows to return
    Returns:
        list of dicts with keys: id, ts, event_type, payload (parsed), git_commit, run_id
    """
    with _get_conn() as conn:
        if event_type:
            rows = conn.execute(
                "SELECT * FROM events WHERE event_type=? ORDER BY ts DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        result.append(d)
    return result


def query_agent_log(context: str = None, limit: int = 20) -> list[dict]:
    """Query agent interactions."""
    with _get_conn() as conn:
        if context:
            rows = conn.execute(
                "SELECT * FROM agent_interactions WHERE context=? ORDER BY ts DESC LIMIT ?",
                (context, limit),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM agent_interactions ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def print_recent(n: int = 20):
    """Print the n most recent events in a readable format."""
    rows = query_log(limit=n)
    for r in rows:
        print(f"[{r['ts']}] {r['event_type']:20s} | {r['git_commit']} | {r['payload']}")
