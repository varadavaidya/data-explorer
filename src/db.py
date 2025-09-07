from __future__ import annotations
import sqlite3, json, time
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path("artifacts/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  created_at INTEGER NOT NULL,
  title TEXT,
  meta_json TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
  type TEXT NOT NULL CHECK(type IN ('text','table','image')),
  content TEXT,              -- for type='text'
  payload_json TEXT,         -- for type='table'/'image'
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS plots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  file_path TEXT NOT NULL,
  spec_json TEXT,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
"""

class DB:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init(self):
        con = self.connect()
        con.executescript(SCHEMA_SQL)
        con.commit()

    # ---- sessions
    def ensure_session(self, session_id: str, title: Optional[str] = None, meta: Optional[dict] = None):
        now = int(time.time())
        self.connect().execute(
            "INSERT OR IGNORE INTO sessions(id, created_at, title, meta_json) VALUES (?,?,?,?)",
            (session_id, now, title, json.dumps(meta or {})),
        )
        self.connect().commit()

    # ---- messages
    def add_text(self, session_id: str, role: str, content: str):
        self._add_msg(session_id, role, "text", content=content)

    def add_table(self, session_id: str, df_columns: list[str], df_records: list[dict], caption: Optional[str] = None):
        payload = {"columns": df_columns, "data": df_records, "caption": caption}
        self._add_msg(session_id, "assistant", "table", payload_json=payload)

    def add_image(self, session_id: str, path: str, caption: Optional[str] = None):
        payload = {"path": path, "caption": caption}
        self._add_msg(session_id, "assistant", "image", payload_json=payload)

    def _add_msg(self, session_id: str, role: str, mtype: str, content: Optional[str] = None, payload_json: Optional[dict] = None):
        now = int(time.time())
        self.connect().execute(
            "INSERT INTO messages(session_id, ts, role, type, content, payload_json) VALUES (?,?,?,?,?,?)",
            (session_id, now, role, mtype, content, json.dumps(payload_json) if payload_json else None),
        )
        self.connect().commit()

    def load_messages(self, session_id: str) -> list[dict]:
        cur = self.connect().execute(
            "SELECT role, type, content, payload_json FROM messages WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        )
        out = []
        for row in cur.fetchall():
            m = {"role": row["role"], "type": row["type"]}
            if row["type"] == "text":
                m["content"] = row["content"]
            else:
                m.update(json.loads(row["payload_json"] or "{}"))
            out.append(m)
        return out

    # ---- plots
    def add_plot(self, session_id: str, file_path: str, spec: Optional[dict]):
        now = int(time.time())
        self.connect().execute(
            "INSERT INTO plots(session_id, file_path, spec_json, created_at) VALUES (?,?,?,?)",
            (session_id, file_path, json.dumps(spec or {}), now),
        )
        self.connect().commit()
    
        # ---- maintenance
    def clear_session(self, session_id: str):
        """Delete all rows for a single session (messages, plots, and the session)."""
        con = self.connect()
        con.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        con.execute("DELETE FROM plots WHERE session_id=?", (session_id,))
        con.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        con.commit()

    def get_plot_paths(self, session_id: str) -> list[str]:
        """Return all plot file paths for a session."""
        cur = self.connect().execute(
            "SELECT file_path FROM plots WHERE session_id=?",
            (session_id,),
        )
        return [r["file_path"] for r in cur.fetchall()]

    def wipe_all(self):
        """Delete ALL rows from all tables (use with care!)."""
        con = self.connect()
        con.execute("DELETE FROM messages")
        con.execute("DELETE FROM plots")
        con.execute("DELETE FROM sessions")
        con.commit()






# singletons / helpers
db = DB()

def init_db():
    db.init()
    return db
