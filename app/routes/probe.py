"""Provider model probe routes without database."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.error_category import classify_error
from app.core.multimodal_probe import (
    default_models_for,
    is_likely_non_chat_model,
    probe_multimodal,
    skipped_chat_result,
)
from app.storage import json_store
from app.vendored.llm_client import LLMClient


router = APIRouter(tags=["probe"])

PROBE_PROMPT = "hi, reply hello"
PROBE_MAX_TOKENS = 128
PROBE_MAX_WORKERS = 6


class ProbeCreate(BaseModel):
    label: str = ""
    base_url: str = ""
    api_key: str = ""
    api_format: str = "openai"


class ProbeRunRequest(BaseModel):
    config_id: str = ""
    base_url: str
    api_key: str = ""
    api_format: str = "openai"
    probe_type: str = "chat"
    models: list[str] = []


@router.get("/api/probe/configs")
def list_configs() -> list[dict[str, Any]]:
    return json_store.list_configs()


@router.post("/api/probe/configs")
def create_config(body: ProbeCreate) -> dict[str, Any]:
    if not (body.base_url or "").strip():
        raise HTTPException(400, "Base URL 不能为空")
    item = json_store.create_config(body.model_dump())
    return {"ok": True, "id": item["id"]}


@router.delete("/api/probe/configs/{config_id}")
def delete_config(config_id: str) -> dict[str, Any]:
    if not json_store.delete_config(config_id):
        raise HTTPException(404, "配置不存在")
    return {"ok": True}


@router.get("/api/probe/runs")
def list_runs(config_id: str = "", limit: int = 20, probe_type: str = "") -> list[dict[str, Any]]:
    rows = json_store.list_probe_runs(config_id=config_id, limit=limit, probe_type=probe_type)
    return [
        {
            "id": row["id"],
            "config_id": row.get("config_id") or "",
            "base_url": row.get("base_url") or "",
            "api_format": row.get("api_format") or "openai",
            "probe_type": row.get("probe_type") or "chat",
            "total": row.get("total") or 0,
            "passed": row.get("passed") or 0,
            "created_at": row.get("created_at") or "",
        }
        for row in rows
    ]


@router.get("/api/probe/runs/{run_id}/results")
def get_run_results(run_id: str) -> list[dict[str, Any]]:
    run = json_store.get_probe_run(run_id)
    if not run:
        raise HTTPException(404, "探测记录不存在")
    return [
        {
            **row,
            "ok": bool(row.get("ok")),
            "error_category": row.get("error_category") or classify_error(row.get("error") or ""),
        }
        for row in run.get("results", [])
    ]


def _probe_one(base_url: str, api_key: str, api_format: str, model: str) -> dict[str, Any]:
    if is_likely_non_chat_model(model):
        return skipped_chat_result(model)

    client = LLMClient(base_url=base_url, api_key=api_key, model=model, api_format=api_format, timeout=30)
    try:
        text, latency_ms, first_token_ms, chars_per_second, error = client.chat_stream_metrics(
            [{"role": "user", "content": PROBE_PROMPT}],
            max_tokens=PROBE_MAX_TOKENS,
        )
    finally:
        client.close()

    ok = bool(first_token_ms is not None and not error)
    if not error and first_token_ms is None:
        error = "未收到首 Token（不可用）"
    return {
        "model": model,
        "ok": ok,
        "skipped": False,
        "probe_type": "chat",
        "modality": "text",
        "endpoint": "chat",
        "latency_ms": latency_ms,
        "first_token_ms": first_token_ms,
        "chars_per_second": chars_per_second,
        "response_preview": text[:200] if text else "",
        "error": error,
        "error_category": classify_error(error),
    }


@router.post("/api/probe/run")
def run_probe(body: ProbeRunRequest):
    base_url = (body.base_url or "").strip()
    if not base_url:
        raise HTTPException(400, "base_url 不能为空")

    api_format = (body.api_format or "openai").strip() or "openai"
    probe_type = (body.probe_type or "chat").strip() or "chat"

    async def generate():
        run_id = ""
        passed = 0
        models = list(body.models)
        list_error = ""
        list_latency_ms = 0

        if not models and probe_type != "chat":
            models = default_models_for(probe_type)

        if not models and probe_type == "chat":
            tmp = LLMClient(base_url=base_url, api_key=body.api_key, model="", api_format=api_format)
            try:
                models, list_latency_ms, list_error = tmp.list_models()
            finally:
                tmp.close()

        if not models:
            message = list_error or "未发现任何模型"
            yield f"data: {json.dumps({'type': 'error', 'message': message, 'error_category': classify_error(message)}, ensure_ascii=False)}\n\n"
            return

        yield f"data: {json.dumps({'type': 'models', 'models': models, 'total': len(models), 'list_latency_ms': list_latency_ms}, ensure_ascii=False)}\n\n"

        run = json_store.create_probe_run(body.config_id, base_url, api_format, len(models), probe_type=probe_type)
        run_id = run["id"]
        max_workers = PROBE_MAX_WORKERS if probe_type == "chat" else 3
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            if probe_type == "chat":
                futures = {executor.submit(_probe_one, base_url, body.api_key, api_format, model): model for model in models}
            else:
                futures = {
                    executor.submit(probe_multimodal, base_url, body.api_key, probe_type, model): model
                    for model in models
                }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "model": futures[future],
                        "ok": False,
                        "skipped": False,
                        "probe_type": probe_type,
                        "latency_ms": 0,
                        "first_token_ms": None,
                        "chars_per_second": 0,
                        "response_preview": "",
                        "error": str(exc),
                        "error_category": classify_error(exc),
                    }
                if result["ok"]:
                    passed += 1
                json_store.save_probe_result(run_id, result, passed=passed)
                yield f"data: {json.dumps({'type': 'result', 'run_id': run_id, **result}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if run_id:
                json_store.update_probe_run(run_id, passed)

        yield f"data: {json.dumps({'type': 'done', 'run_id': run_id, 'total': len(models), 'passed': passed}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
