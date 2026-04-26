import io
import sqlite3
from pathlib import Path

from PIL import Image

import activity_log


def _png_bytes(size=(40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def _open(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def test_creates_schema_on_first_use(tmp_path):
    db = tmp_path / "a.db"
    activity_log.log_request(
        db_path=db,
        source="local", recognized_latex="x", confidence=0.9, num_components=1,
        operation="simplify", sympy_solution="x", ryacas_solution="x",
        agreement="match", duration_ms=42, image_bytes=_png_bytes(),
    )
    assert db.exists()
    with _open(db) as c:
        row = c.execute("SELECT * FROM requests").fetchone()
    assert row["source"] == "local"
    assert row["confidence"] == 0.9
    assert row["agreement"] == "match"


def test_log_then_read_roundtrip(tmp_path):
    db = tmp_path / "a.db"
    activity_log.log_request(
        db_path=db,
        source="gemini", recognized_latex="x+5=12", confidence=None, num_components=4,
        operation="solve", sympy_solution="7", ryacas_solution="7",
        agreement="match", duration_ms=812, image_bytes=_png_bytes(),
    )
    with _open(db) as c:
        row = c.execute("SELECT * FROM requests").fetchone()
    assert row["recognized_latex"] == "x+5=12"
    assert row["confidence"] is None
    assert row["operation"] == "solve"
    assert row["thumbnail_b64"] is not None


def test_thumbnail_is_64x64(tmp_path):
    import base64
    db = tmp_path / "a.db"
    activity_log.log_request(
        db_path=db,
        source="local", recognized_latex="x", confidence=0.9, num_components=1,
        operation="simplify", sympy_solution=None, ryacas_solution=None,
        agreement="match", duration_ms=42, image_bytes=_png_bytes(size=(200, 200)),
    )
    with _open(db) as c:
        b64 = c.execute("SELECT thumbnail_b64 FROM requests").fetchone()[0]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (64, 64)


def test_cap_enforced(tmp_path):
    db = tmp_path / "a.db"
    for i in range(15):
        activity_log.log_request(
            db_path=db, source="local", recognized_latex=str(i), confidence=0.9,
            num_components=1, operation="simplify", sympy_solution=str(i),
            ryacas_solution=str(i), agreement="match", duration_ms=10,
            image_bytes=_png_bytes(), max_rows=10,
        )
    with _open(db) as c:
        n = c.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        last = c.execute("SELECT recognized_latex FROM requests ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert n == 10
    assert last == "14"


def test_recovers_from_corrupt_db(tmp_path):
    db = tmp_path / "a.db"
    db.write_bytes(b"not a sqlite database at all")
    activity_log.log_request(
        db_path=db, source="local", recognized_latex="x", confidence=0.9,
        num_components=1, operation="simplify", sympy_solution="x",
        ryacas_solution="x", agreement="match", duration_ms=10, image_bytes=_png_bytes(),
    )
    with _open(db) as c:
        n = c.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    assert n == 1


def test_never_raises_on_failure(tmp_path):
    """If we can't write, log_request swallows the error."""
    activity_log.log_request(
        db_path=tmp_path / "definitely" / "missing" / "subdir" / "a.db",
        source="local", recognized_latex="x", confidence=0.9, num_components=1,
        operation="simplify", sympy_solution="x", ryacas_solution="x",
        agreement="match", duration_ms=10, image_bytes=_png_bytes(),
    )
