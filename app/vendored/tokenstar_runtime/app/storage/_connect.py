"""SQLite connection helpers shared by storage and API code."""

import sqlite3

SQLITE_BUSY_TIMEOUT_MS = 60000


def connect_sqlite(
    db_path: str,
    *,
    row_factory=None,
    foreign_keys: bool = True,
    busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys=ON")
    if row_factory is not None:
        conn.row_factory = row_factory
    return conn
