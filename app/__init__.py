"""Standalone model probe and detection app."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Model Probe Detection", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.routes import detection, probe

    app.include_router(probe.router)
    app.include_router(detection.router)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def root():
        index = STATIC_DIR / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {"ok": True, "message": "frontend is not built"}

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(404, "Not Found")
        path = STATIC_DIR / full_path
        if path.is_file():
            return FileResponse(path)
        index = STATIC_DIR / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {"ok": True, "message": "frontend is not built"}

    return app
