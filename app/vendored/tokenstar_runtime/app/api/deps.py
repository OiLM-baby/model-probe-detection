"""API 依赖注入：提供数据库连接。"""

import os
import sqlite3
from collections.abc import Generator

from fastapi import Request

from app.storage._connect import connect_sqlite

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "history.db",
)


def use_db_path(app, db_path: str) -> None:
    """将 db_path 注册到 app.state，供所有路由依赖读取。"""
    app.state.db_path = db_path


def get_db(request: Request) -> Generator[sqlite3.Connection]:
    path = getattr(request.app.state, "db_path", DEFAULT_DB_PATH)
    conn = connect_sqlite(path, row_factory=sqlite3.Row, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()
