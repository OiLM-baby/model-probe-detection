"""Tiny JSON-file storage for standalone mode."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CONFIGS_FILE = DATA_DIR / "probe_configs.json"
PROBE_RUNS_FILE = DATA_DIR / "probe_runs.json"
DETECTION_RUNS_FILE = DATA_DIR / "detection_runs.json"

_LOCK = threading.RLock()


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_id(length: int = 12) -> str:
    return uuid.uuid4().hex[:length]


def list_configs() -> list[dict[str, Any]]:
    return _read_list(CONFIGS_FILE)


def create_config(payload: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": new_id(),
        "label": payload.get("label") or "",
        "base_url": payload.get("base_url") or "",
        "api_key": payload.get("api_key") or "",
        "api_format": payload.get("api_format") or "openai",
        "created_at": now_text(),
    }
    with _LOCK:
        rows = _read_list(CONFIGS_FILE)
        rows.insert(0, item)
        _write_list(CONFIGS_FILE, rows)
    return item


def delete_config(config_id: str) -> bool:
    with _LOCK:
        rows = _read_list(CONFIGS_FILE)
        kept = [row for row in rows if row.get("id") != config_id]
        if len(kept) == len(rows):
            return False
        _write_list(CONFIGS_FILE, kept)
        return True


def get_config(config_id: str) -> dict[str, Any] | None:
    for item in list_configs():
        if item.get("id") == config_id:
            return item
    return None


def list_probe_runs(config_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
    rows = _read_list(PROBE_RUNS_FILE)
    if config_id:
        rows = [row for row in rows if row.get("config_id") == config_id]
    return rows[:limit]


def create_probe_run(config_id: str, base_url: str, api_format: str, total: int) -> dict[str, Any]:
    run = {
        "id": new_id(),
        "config_id": config_id or "",
        "base_url": base_url,
        "api_format": api_format,
        "total": total,
        "passed": 0,
        "created_at": now_text(),
        "results": [],
    }
    with _LOCK:
        rows = _read_list(PROBE_RUNS_FILE)
        rows.insert(0, run)
        _write_list(PROBE_RUNS_FILE, rows)
    return run


def save_probe_result(run_id: str, result: dict[str, Any], passed: int | None = None) -> None:
    with _LOCK:
        rows = _read_list(PROBE_RUNS_FILE)
        for run in rows:
            if run.get("id") != run_id:
                continue
            results = [row for row in run.get("results", []) if row.get("model") != result.get("model")]
            results.append(result)
            run["results"] = results
            if passed is not None:
                run["passed"] = passed
            break
        _write_list(PROBE_RUNS_FILE, rows)


def update_probe_run(run_id: str, passed: int) -> None:
    with _LOCK:
        rows = _read_list(PROBE_RUNS_FILE)
        for run in rows:
            if run.get("id") == run_id:
                run["passed"] = passed
                break
        _write_list(PROBE_RUNS_FILE, rows)


def get_probe_run(run_id: str) -> dict[str, Any] | None:
    for run in _read_list(PROBE_RUNS_FILE):
        if str(run.get("id")) == str(run_id):
            return run
    return None


def list_detection_runs(limit: int = 50) -> list[dict[str, Any]]:
    return _read_list(DETECTION_RUNS_FILE)[:limit]


def create_detection_run(config_id: str, suite: str, models: list[str], tests: list[str]) -> dict[str, Any]:
    run = {
        "id": new_id(),
        "config_id": config_id,
        "suite_key": suite,
        "status": "running",
        "summary": {"planned_models": models, "planned_tests": tests},
        "raw_payload": None,
        "results": [],
        "error": "",
        "started_at": now_text(),
        "finished_at": "",
    }
    with _LOCK:
        rows = _read_list(DETECTION_RUNS_FILE)
        rows.insert(0, run)
        _write_list(DETECTION_RUNS_FILE, rows)
    return run


def update_detection_run(run_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    with _LOCK:
        rows = _read_list(DETECTION_RUNS_FILE)
        updated = None
        for run in rows:
            if run.get("id") == run_id:
                run.update(updates)
                updated = run
                break
        _write_list(DETECTION_RUNS_FILE, rows)
        return updated


def get_detection_run(run_id: str) -> dict[str, Any] | None:
    for run in _read_list(DETECTION_RUNS_FILE):
        if str(run.get("id")) == str(run_id):
            return run
    return None


def _read_list(path: Path) -> list[dict[str, Any]]:
    with _LOCK:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []


def _write_list(path: Path, rows: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
