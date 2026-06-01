"""Standalone startup entry."""

import os
import subprocess
from pathlib import Path

import uvicorn

from app import create_app


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
HOST = os.getenv("MODEL_TOOL_HOST", "0.0.0.0")
PORT = int(os.getenv("MODEL_TOOL_PORT", "8090"))


def build_frontend() -> None:
    if os.getenv("MODEL_TOOL_SKIP_FRONTEND_BUILD") == "1":
        print("Skip frontend build: MODEL_TOOL_SKIP_FRONTEND_BUILD=1")
        return
    if not (FRONTEND_DIR / "node_modules").is_dir():
        print("Skip frontend build: frontend/node_modules not found, run npm install first")
        return
    print("Building frontend...")
    subprocess.run(["npm", "run", "build"], cwd=str(FRONTEND_DIR), check=True)


if __name__ == "__main__":
    build_frontend()
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")
