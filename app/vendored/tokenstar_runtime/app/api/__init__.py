"""TokenStar API — FastAPI 应用工厂。"""

import sqlite3
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import DEFAULT_DB_PATH, use_db_path
from app.core.models import AppConfig
from app.storage._connect import connect_sqlite


def create_app(db_path: str = "", cors_origins: list[str] | None = None, app_config: AppConfig | None = None) -> FastAPI:
    app = FastAPI(title="TokenStar API", version="1.0.0")

    if cors_origins is None:
        cors_origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    use_db_path(app, db_path or DEFAULT_DB_PATH)
    app.state.app_config = app_config

    # 统一错误响应格式
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.status_code, "message": exc.detail, "data": None},
        )

    from app.api.routes import alerts, probe, providers, reports, runs, summary

    app.include_router(reports.router)
    app.include_router(providers.router)
    app.include_router(probe.router)
    app.include_router(runs.router)
    app.include_router(summary.router)
    app.include_router(alerts.router)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        resolved = getattr(request.app.state, "db_path", DEFAULT_DB_PATH)
        result: dict[str, Any] = {"db_connected": False, "schema_ready": False, "missing_tables": [], "db_rows": {}}

        conn = None
        try:
            conn = connect_sqlite(resolved)
            result["db_connected"] = True

            required_tables = ("reports", "provider_snapshot", "test_results", "alert_log")
            for t in required_tables:
                try:
                    row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                    result["db_rows"][t] = row[0] if row else 0
                except sqlite3.OperationalError:
                    result["missing_tables"].append(t)
                    result["db_rows"][t] = 0

            result["schema_ready"] = len(result["missing_tables"]) == 0
            result["status"] = "degraded" if not result["schema_ready"] else "ok"
            if result["status"] == "degraded":
                return JSONResponse(
                    status_code=503,
                    content={"code": 503, "message": "Database schema incomplete", "data": result},
                )
            return {"code": 0, "data": result}
        except Exception as exc:
            result["status"] = "error"
            return JSONResponse(
                status_code=503,
                content={"code": 503, "message": str(exc), "data": result},
            )
        finally:
            if conn is not None:
                conn.close()

    return app
