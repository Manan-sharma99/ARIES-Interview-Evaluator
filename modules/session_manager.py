"""
session_manager.py
Handles all session persistence for ARIES Interview System.

On-disk format (NEW):
  { "Candidate Name": { "sessions": [ { type, status, timestamp, average_score, questions: [...] } ] } }

Legacy flat-list format is auto-migrated on read via normalize_sessions().
"""

import json
import os
from datetime import datetime

# Absolute path so this works regardless of CWD
_MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR     = os.path.normpath(os.path.join(_MODULE_DIR, ".."))
SESSIONS_FILE = os.path.join(_BASE_DIR, "models", "sessions.json")


# ── Public API ───────────────────────────────────────────────────────────────

def load_sessions():
    """Load sessions from disk, normalized to flat-list format for UI consumption."""
    return normalize_sessions(_load_raw())


def save_sessions(flat_sessions: dict):
    """
    Accept the flat-list format { candidate: [records] } that the UI works with
    and persist it. Existing grouped sessions are preserved; flat records are
    wrapped in a single 'imported' session block.
    """
    raw = _load_raw()
    for candidate, records in flat_sessions.items():
        if not records:
            continue
        _ensure_grouped(raw, candidate)
        avg = round(sum(r.get("scores", {}).get("final", 0) for r in records) / len(records), 1)
        # Replace last 'imported' session or append new one
        sessions = raw[candidate]["sessions"]
        # Find existing imported block, if any, and update it
        for s in sessions:
            if s.get("type") == "imported":
                s["questions"] = records
                s["average_score"] = avg
                break
        else:
            sessions.append({
                "type": "imported",
                "status": "completed",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "average_score": avg,
                "questions": records,
            })
    _save_raw(raw)


def save_session_block(candidate: str, session_type: str, records: list, status: str = "completed"):
    """
    Save a completed session as a proper grouped block.
    Called after a live/practice session finishes with ≥1 answer.
    """
    raw = _load_raw()
    _ensure_grouped(raw, candidate)
    avg = round(sum(r.get("scores", {}).get("final", 0) for r in records) / len(records), 1) if records else 0
    raw[candidate]["sessions"].append({
        "type":          session_type,
        "status":        status,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "average_score": avg,
        "questions":     records,
    })
    _save_raw(raw)


def save_abandoned_session(candidate: str, session_type: str = "live"):
    """Save a session that ended with 0 answers (status = 'abandoned')."""
    raw = _load_raw()
    _ensure_grouped(raw, candidate)
    raw[candidate]["sessions"].append({
        "type":          session_type,
        "status":        "abandoned",
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "average_score": 0,
        "questions":     [],
    })
    _save_raw(raw)


def build_student_summary(student_name: str, records: list, calculate_esi) -> dict:
    """Build one row for the interviewer comparison table."""
    scores   = [r["scores"]["final"] for r in records if r.get("scores")]
    emotions = [r.get("emotion", "neutral") for r in records]
    avg      = round(sum(scores) / len(scores), 1) if scores else 0.0
    best     = round(max(scores), 1) if scores else 0.0
    last     = round(scores[-1], 1) if scores else 0.0
    trend    = round(scores[-1] - scores[0], 1) if len(scores) >= 2 else 0.0
    esi      = calculate_esi(emotions) if emotions else 0.0
    return {
        "Student":    student_name,
        "Sessions":   len(records),
        "Avg Score":  avg,
        "Last Score": last,
        "Best Score": best,
        "Trend":      f"+{trend}" if trend >= 0 else str(trend),
        "ESI":        esi,
        "Last Seen":  records[-1].get("date", "-") if records else "-",
    }


def normalize_sessions(data) -> dict:
    """
    Convert any on-disk format to the canonical flat-list format:
      { candidate_name: [ record_dict, ... ] }

    Handles three formats:
      1. Top-level list (very old export format)
      2. { candidate: [records] }  (previous flat-dict format)
      3. { candidate: { "sessions": [{type, status, questions:[...]}] } }  (new format)
    """
    if isinstance(data, list):
        return _normalize_from_list(data)
    if not isinstance(data, dict):
        return {}

    normalized = {}
    for student, value in data.items():
        normalized[student] = []
        # ── New grouped format ────────────────────────────────────────────
        if isinstance(value, dict) and "sessions" in value:
            for session in value["sessions"]:
                for q in session.get("questions", []):
                    if not isinstance(q, dict):
                        continue
                    rec = dict(q)
                    # Back-fill missing keys
                    rec.setdefault("date",         session.get("timestamp", "-"))
                    rec.setdefault("category",     session.get("type", "General"))
                    rec.setdefault("emotion",      "neutral")
                    rec.setdefault("mode",         session.get("type", "practice"))
                    rec.setdefault("session_status", session.get("status", "completed"))
                    scores = rec.setdefault("scores", {})
                    scores.setdefault("final",      0)
                    scores.setdefault("relevance",  0)
                    scores.setdefault("fluency",    0)
                    scores.setdefault("confidence", 70)
                    scores.setdefault("sentiment",  0)
                    normalized[student].append(rec)
        # ── Legacy flat-list format ───────────────────────────────────────
        elif isinstance(value, list):
            for rec in value:
                if not isinstance(rec, dict):
                    continue
                scores = rec.setdefault("scores", {})
                scores.setdefault("confidence", 70)
                normalized[student].append(rec)
    return normalized


# ── Private helpers ──────────────────────────────────────────────────────────

def _load_raw() -> dict:
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_raw(data: dict):
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _ensure_grouped(raw: dict, candidate: str):
    """Make sure raw[candidate] uses the new grouped structure."""
    if candidate not in raw:
        raw[candidate] = {"sessions": []}
    elif isinstance(raw[candidate], list):
        # Migrate old flat list → grouped
        raw[candidate] = {
            "sessions": [{
                "type":          "imported",
                "status":        "completed",
                "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
                "average_score": 0,
                "questions":     raw[candidate],
            }]
        }


def _normalize_from_list(data: list) -> dict:
    """Handle very old top-level-list export format."""
    normalized = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        student = entry.get("student") or entry.get("name") or "Unknown Student"
        normalized.setdefault(student, [])
        for q in entry.get("questions", []):
            if isinstance(q, dict):
                normalized[student].append({
                    "date":     entry.get("timestamp", "-"),
                    "question": q.get("question", ""),
                    "category": entry.get("category", "Imported"),
                    "answer":   q.get("answer", ""),
                    "emotion":  q.get("emotion", "neutral"),
                    "scores": {
                        "final":      q.get("final_score", entry.get("average_score", 0)),
                        "relevance":  q.get("relevance", 0),
                        "fluency":    q.get("fluency", 0),
                        "confidence": 70,
                        "sentiment":  q.get("sentiment", 0),
                    },
                    "grade":       q.get("grade", "N/A"),
                    "grade_label": entry.get("grade_label", ""),
                    "mode":        entry.get("mode", "practice"),
                })
    return normalized
