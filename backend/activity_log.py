"""SQLite logger for /convert calls. Never raises — logging failures must
not break the user's request.
"""

import base64
import io
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROWS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp       TEXT    NOT NULL,
  source          TEXT    NOT NULL,
  recognized_latex TEXT   NOT NULL,
  confidence      REAL,
  num_components  INTEGER,
  operation       TEXT,
  primary_solver  TEXT,
  primary_solution TEXT,
  crosscheck_solution TEXT,
  agreement       TEXT,
  duration_ms     INTEGER,
  thumbnail_b64   TEXT
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_source ON requests(source);
"""


def _migrate_legacy_columns(conn: sqlite3.Connection) -> None:
    """Rename pre-rebrand columns in-place if they're still present.

    Older databases stored the displayed answer in `sympy_solution` and the
    cross-checker output in `ryacas_solution`. Now Ryacas drives the displayed
    answer and Gemini is the cross-checker, so the columns are renamed to
    `primary_solution` / `crosscheck_solution` and a `primary_solver` column
    is added. SQLite ≥3.25 supports RENAME COLUMN, which preserves data.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)").fetchall()}
    if not cols:
        return
    if "sympy_solution" in cols and "primary_solution" not in cols:
        conn.execute("ALTER TABLE requests RENAME COLUMN sympy_solution TO primary_solution")
    if "ryacas_solution" in cols and "crosscheck_solution" not in cols:
        conn.execute("ALTER TABLE requests RENAME COLUMN ryacas_solution TO crosscheck_solution")
    if "primary_solver" not in cols:
        conn.execute("ALTER TABLE requests ADD COLUMN primary_solver TEXT")


def _make_thumbnail_b64(image_bytes: bytes, size: int = 64) -> Optional[str]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        ox = (size - img.width) // 2
        oy = (size - img.height) // 2
        canvas.paste(img, (ox, oy))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("thumbnail failed: %s", exc)
        return None


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open + verify schema. On corruption, raises sqlite3.DatabaseError after
    closing the bad connection so the caller can safely unlink the file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        _migrate_legacy_columns(conn)
        conn.executescript(_SCHEMA)
    except sqlite3.DatabaseError:
        conn.close()
        raise
    return conn


def _try_recover_corrupt(db_path: Path) -> sqlite3.Connection:
    try:
        db_path.unlink()
    except FileNotFoundError:
        pass
    return _connect(db_path)


def log_request(
    db_path: Path,
    *,
    source: str,
    recognized_latex: str,
    confidence: Optional[float],
    num_components: Optional[int],
    operation: Optional[str],
    primary_solution: Optional[str],
    crosscheck_solution: Optional[str],
    primary_solver: Optional[str] = None,
    agreement: Optional[str],
    duration_ms: int,
    image_bytes: bytes,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> None:
    """Append one row to activity.db. Swallows all exceptions."""
    try:
        try:
            conn = _connect(db_path)
        except sqlite3.DatabaseError:
            conn = _try_recover_corrupt(db_path)
        with conn:
            conn.execute(
                """INSERT INTO requests (
                    timestamp, source, recognized_latex, confidence, num_components,
                    operation, primary_solver, primary_solution, crosscheck_solution,
                    agreement, duration_ms, thumbnail_b64
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    source,
                    recognized_latex,
                    confidence,
                    num_components,
                    operation,
                    primary_solver,
                    primary_solution,
                    crosscheck_solution,
                    agreement,
                    duration_ms,
                    _make_thumbnail_b64(image_bytes),
                ),
            )
            conn.execute(
                """DELETE FROM requests WHERE id NOT IN
                   (SELECT id FROM requests ORDER BY id DESC LIMIT ?)""",
                (max_rows,),
            )
        conn.close()
    except Exception as exc:
        print(f"[activity_log] failed: {exc}", file=sys.stderr, flush=True)
