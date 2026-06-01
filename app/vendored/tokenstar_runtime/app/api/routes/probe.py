"""实时模型探测接口。"""

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_db
from app.api.filters import VISIBLE_GROUP
from app.core.llm_client import LLMClient
from app.core.models import ProviderConfig
from app.storage.sqlite_store import SqliteReportStore
from app.tests.availability import FIRST_TOKEN_CONNECTIVITY_PROMPT
from app.utils.error_category import classify_error

router = APIRouter(tags=["probe"])
_probe_semaphore = asyncio.Semaphore(8)


class ModelProbeRequest(BaseModel):
    group: str = Field(VISIBLE_GROUP, description="中转组，当前只支持 clawos")
    model: str = Field(..., description="模型名称，精确匹配")
    prompt: str = Field(FIRST_TOKEN_CONNECTIVITY_PROMPT, description="探测 prompt，落库时截断到 500 字")
    max_tokens: int = Field(64, ge=1, le=512)
    timeout_seconds: int = Field(30, ge=1, le=180)


class ModelProbeHistoryRequest(BaseModel):
    group: str = Field(VISIBLE_GROUP, description="中转组，当前只支持 clawos")
    model: str = Field(..., description="模型名称，精确匹配")
    start: str = Field(..., description="开始时间，YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS")
    end: str = Field(..., description="结束时间，YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS")
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=200)
    ok: bool | None = Field(None, description="可选，按成功/失败过滤")
    error_category: str = Field("", description="可选，按错误分类过滤")
    min_first_token_ms: int | None = Field(None, ge=0, description="可选，筛选 TTFT 不低于该值的记录")


def _normalize_range(start: str, end: str) -> tuple[str, str]:
    start = start.strip()
    end = end.strip()
    if len(end) == 10:
        end = f"{end} 23:59:59"
    return start, end


def _providers_from_request(request: Request) -> list[ProviderConfig]:
    config = getattr(request.app.state, "app_config", None)
    return list(getattr(config, "providers", []) or [])


def _find_provider(request: Request, group: str, model: str) -> ProviderConfig | None:
    for provider in _providers_from_request(request):
        if provider.group == group and provider.model == model:
            return provider
    return None


def _do_probe(provider: ProviderConfig, prompt: str, max_tokens: int) -> dict[str, Any]:
    client = LLMClient(provider)
    try:
        text, latency_ms, first_token_ms, chars_per_second, error = client.chat_stream_metrics(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            test_name="api_probe",
        )
    finally:
        client.session.close()
    ok = bool(first_token_ms is not None and text and not error)
    effective_error = error
    if not effective_error and not text:
        effective_error = "流式响应为空"
    category, detail = classify_error(
        message=effective_error,
        test_name="first_token_connectivity",
        status="成功" if ok else "失败",
    )
    return {
        "group": provider.group,
        "provider": provider.name,
        "model": provider.model,
        "ok": ok,
        "latency_ms": latency_ms,
        "first_token_ms": first_token_ms,
        "chars_per_second": chars_per_second,
        "char_count": len(text),
        "response_preview": text[:200],
        "error": effective_error,
        "error_category": category,
        "error_detail": detail,
    }


@router.post("/probes")
async def probe_model(body: ModelProbeRequest, request: Request) -> dict[str, Any]:
    group = (body.group or VISIBLE_GROUP).strip() or VISIBLE_GROUP
    model = body.model.strip()
    if group != VISIBLE_GROUP:
        return {"code": 400, "msg": "当前 API 只支持 clawos 中转组", "data": {}}
    if not model:
        return {"code": 400, "msg": "model 不能为空", "data": {}}
    provider = _find_provider(request, group, model)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"未找到模型配置: {group}/{model}")

    prompt = (body.prompt or FIRST_TOKEN_CONNECTIVITY_PROMPT)[:500]
    async with _probe_semaphore:
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(_do_probe, provider, prompt, body.max_tokens),
                timeout=body.timeout_seconds,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="实时探测超时") from exc

    store = SqliteReportStore(getattr(request.app.state, "db_path"))
    probe_id = await run_in_threadpool(
        store.write_model_probe_log,
        group_name=result["group"],
        provider=result["provider"],
        model=result["model"],
        prompt=prompt,
        ok=result["ok"],
        latency_ms=result["latency_ms"],
        first_token_ms=result["first_token_ms"],
        chars_per_second=result["chars_per_second"],
        char_count=result["char_count"],
        response_preview=result["response_preview"],
        error=result["error"],
        error_category=result["error_category"],
        error_detail=result["error_detail"],
    )
    result["id"] = probe_id
    return {"code": 0, "msg": "ok", "data": result}


@router.post("/probes/history")
def list_model_probe_history(
    body: ModelProbeHistoryRequest,
    request: Request,
    conn = Depends(get_db),
) -> dict[str, Any]:
    group = (body.group or VISIBLE_GROUP).strip() or VISIBLE_GROUP
    model = body.model.strip()
    start, end = _normalize_range(body.start, body.end)
    if group != VISIBLE_GROUP:
        return {"code": 400, "msg": "当前 API 只支持 clawos 中转组", "data": {"items": [], "total": 0}}
    if not model or not start or not end:
        return {"code": 400, "msg": "model、start 和 end 不能为空", "data": {"items": [], "total": 0}}

    where = ["group_name=?", "model=?", "tested_at>=?", "tested_at<=?"]
    params: list[Any] = [group, model, start, end]
    if body.ok is not None:
        where.append("ok=?")
        params.append(1 if body.ok else 0)
    if body.error_category:
        where.append("error_category=?")
        params.append(body.error_category)
    if body.min_first_token_ms is not None:
        where.append("first_token_ms>=?")
        params.append(body.min_first_token_ms)
    where_clause = " AND ".join(where)
    total_row = conn.execute(f"SELECT COUNT(*) FROM model_probe_log WHERE {where_clause}", params).fetchone()
    total = total_row[0] if total_row else 0
    offset = (body.page - 1) * body.page_size
    rows = conn.execute(
        f"""SELECT * FROM model_probe_log
            WHERE {where_clause}
            ORDER BY tested_at DESC, id DESC
            LIMIT ? OFFSET ?""",
        params + [body.page_size, offset],
    ).fetchall()
    items = [dict(row) for row in rows]
    for item in items:
        item["ok"] = bool(item.get("ok"))
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "items": items,
            "total": total,
            "page": body.page,
            "page_size": body.page_size,
        },
    }
