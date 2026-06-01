"""手动轻量巡检接口。"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_db
from app.api.filters import VISIBLE_GROUP
from app.core.runner import run_provider
from app.events.bus import EventBus
from app.events.types import (
    PROVIDER_CRASHED,
    PROVIDER_DONE,
    PROVIDER_STARTED,
    REPORT_COMPLETED,
    REPORT_CRASHED,
    REPORT_STARTED,
    ProgressEvent,
)
from app.report.generator import build_payload
from app.storage.sqlite_store import SqliteReportStore, suite_type_for, template_name_for
from app.subscribers.alert_subscriber import AlertSubscriber
from app.subscribers.store_subscriber import StoreSubscriber
from app.utils.tags import auto_tags
from app.utils.timezone import beijing_from_timestamp, beijing_now_str

router = APIRouter(tags=["runs"])

ALLOWED_TRIGGER_SUITES = {"first_token_connectivity"}


class RunTriggerRequest(BaseModel):
    suite: str = Field("first_token_connectivity", description="当前只允许 first_token_connectivity")
    group: str = Field(VISIBLE_GROUP, description="可选，当前只支持 clawos")
    models: list[str] = Field(default_factory=list, description="可选，只跑指定模型名")
    workers: int = Field(16, description="并发数，后端会限制在 1-32")


def _providers_from_request(request: Request):
    config = getattr(request.app.state, "app_config", None)
    return list(getattr(config, "providers", []) or [])


def _acquire_lock(lock_path: str):
    lock_fd = open(lock_path, "w")
    fcntl = __import__("fcntl")
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except (BlockingIOError, OSError):
        lock_fd.close()
        return None


def _release_lock(lock_fd) -> None:
    if lock_fd is None:
        return
    fcntl = __import__("fcntl")
    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
    lock_fd.close()


def _publish_provider_events(bus, provider, summary=None, stage="started", error="") -> None:
    tags_json = json.dumps(auto_tags(provider.name, provider.model, provider.group), ensure_ascii=False)
    if stage == "started":
        bus.publish(ProgressEvent(kind=PROVIDER_STARTED, payload={
            "provider": provider.name,
            "model": provider.model,
            "group_name": provider.group,
            "provider_format": provider.format,
            "base_url": provider.base_url,
            "tags_json": tags_json,
        }))
    elif stage == "done" and summary is not None:
        bus.publish(ProgressEvent(
            kind=PROVIDER_DONE,
            payload={"summary": asdict(summary), "results": [asdict(r) for r in summary.results]},
        ))
    elif stage == "crashed":
        bus.publish(ProgressEvent(kind=PROVIDER_CRASHED, payload={
            "provider": provider.name,
            "model": provider.model,
            "group_name": provider.group,
            "error": error,
        }))


def _run_trigger_job(
    db_path: str,
    app_config,
    providers,
    suite: str,
    run_id: str,
    workers: int,
    lock_fd,
    started_event: threading.Event | None = None,
) -> None:
    store = SqliteReportStore(db_path)
    store.init()
    bus = EventBus()
    store_sub = StoreSubscriber(store)
    store_sub.attach(bus)
    AlertSubscriber(store).attach(bus)
    suite_tests = [suite]
    start_time = time.time()
    summaries = []
    try:
        bus.publish(ProgressEvent(kind=REPORT_STARTED, payload={
            "env": app_config.runtime.env,
            "suite": suite,
            "run_id": run_id,
            "planned": len(providers),
            "suite_type": suite_type_for(suite),
            "template_name": template_name_for(suite_type_for(suite)),
            "triggered_by": "api",
        }))
        if started_event is not None:
            started_event.set()
        if workers > 1:
            for provider in providers:
                _publish_provider_events(bus, provider, None, stage="started")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(
                        run_provider,
                        provider,
                        app_config.thresholds,
                        suite_tests,
                        pricing_config=app_config.pricing,
                        judge_config=app_config.judge,
                        quota_config=app_config.quota,
                        bus=bus,
                    ): provider for provider in providers
                }
                ordered = {}
                for future in as_completed(future_map):
                    provider = future_map[future]
                    try:
                        summary = future.result()
                    except Exception as exc:
                        _publish_provider_events(bus, provider, None, stage="crashed", error=str(exc))
                        summary = None
                    ordered[provider.name] = summary
                    if summary is not None:
                        _publish_provider_events(bus, provider, summary, stage="done")
                summaries.extend(summary for provider in providers if (summary := ordered.get(provider.name)) is not None)
        else:
            for provider in providers:
                _publish_provider_events(bus, provider, None, stage="started")
                try:
                    summary = run_provider(
                        provider,
                        app_config.thresholds,
                        suite_tests,
                        pricing_config=app_config.pricing,
                        judge_config=app_config.judge,
                        quota_config=app_config.quota,
                        bus=bus,
                    )
                    _publish_provider_events(bus, provider, summary, stage="done")
                    summaries.append(summary)
                except Exception as exc:
                    _publish_provider_events(bus, provider, None, stage="crashed", error=str(exc))
        payload = build_payload(
            summaries,
            start_time,
            time.time(),
            env=app_config.runtime.env,
            run_id=run_id,
            suite=suite,
        )
        bus.publish(ProgressEvent(
            kind=REPORT_COMPLETED,
            payload={"run_id": run_id, "payload_json": json.dumps(payload, ensure_ascii=False)},
        ))
    except Exception as exc:
        try:
            bus.publish(ProgressEvent(kind=REPORT_CRASHED, payload={
                "run_id": run_id,
                "reason": str(exc),
                "payload_json": "",
            }))
        finally:
            raise
    finally:
        if started_event is not None:
            started_event.set()
        _release_lock(lock_fd)


def _run_overview(conn, report_id: int) -> dict[str, Any]:
    row = conn.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN tr.status IN ('成功','pass') THEN 1 ELSE 0 END) AS passed
           FROM provider_snapshot ps
           LEFT JOIN test_results tr
             ON tr.provider_snapshot_id=ps.id AND tr.test_name='first_token_connectivity'
           WHERE ps.report_id=?""",
        (report_id,),
    ).fetchone()
    total = row["total"] if row else 0
    passed = row["passed"] if row and row["passed"] is not None else 0
    pass_rate = round(passed / total * 100, 2) if total else None
    return {
        "first_token_passed": passed,
        "first_token_total": total,
        "model_full_pass": passed,
        "model_pass_rate": pass_rate,
    }


@router.post("/runs")
def trigger_run(body: RunTriggerRequest, request: Request) -> dict[str, Any]:
    config = getattr(request.app.state, "app_config", None)
    if config is None:
        raise HTTPException(status_code=503, detail="API 未加载 provider 配置，无法触发巡检")
    suite = body.suite.strip() or "first_token_connectivity"
    if suite not in ALLOWED_TRIGGER_SUITES:
        raise HTTPException(status_code=400, detail="不支持的 suite")
    group = (body.group or VISIBLE_GROUP).strip() or VISIBLE_GROUP
    if group != VISIBLE_GROUP:
        return {"code": 400, "msg": "当前 API 只支持 clawos 中转组", "data": {}}
    wanted_models = {m.strip() for m in body.models if m.strip()}
    providers = [p for p in _providers_from_request(request) if p.group == group]
    if wanted_models:
        providers = [p for p in providers if p.model in wanted_models]
    if not providers:
        raise HTTPException(status_code=404, detail="没有匹配的模型配置")

    db_path = getattr(request.app.state, "db_path")
    lock_fd = _acquire_lock(f"{db_path}.lock")
    if lock_fd is None:
        raise HTTPException(status_code=409, detail="另一个巡检正在运行，请稍后重试")

    workers = min(max(int(body.workers or 16), 1), 32)
    run_id = beijing_from_timestamp(time.time(), "%Y%m%d_%H%M%S")
    started_event = threading.Event()
    thread = threading.Thread(
        target=_run_trigger_job,
        args=(db_path, config, providers, suite, run_id, workers, lock_fd, started_event),
        name=f"tokenstar-trigger-{run_id}",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        _release_lock(lock_fd)
        raise
    started_event.wait(timeout=2)
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "run_id": run_id,
            "status": "running",
            "planned": len(providers),
            "workers": workers,
            "started_at": beijing_now_str(),
        },
    }


@router.get("/runs/{run_id}/status")
def get_run_status(run_id: str, conn = Depends(get_db)) -> dict[str, Any]:
    report = conn.execute("SELECT * FROM reports WHERE run_id=?", (run_id,)).fetchone()
    if not report:
        raise HTTPException(status_code=404, detail=f"巡检 {run_id} 不存在")
    report_id = report["id"]
    planned = report["provider_count_planned"] or 0
    completed_row = conn.execute(
        "SELECT COUNT(*) FROM provider_snapshot WHERE report_id=? AND status IN ('completed','crashed')",
        (report_id,),
    ).fetchone()
    completed = completed_row[0] if completed_row else 0
    status = report["status"] or "running"
    progress_pct = round(completed / planned * 100, 2) if planned else 0
    overview = _run_overview(conn, report_id) if status == "completed" else None
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "run_id": run_id,
            "status": status,
            "planned": planned,
            "completed": completed,
            "progress_pct": progress_pct,
            "started_at": report["started_at"],
            "finished_at": report["finished_at"],
            "duration_seconds": report["duration_seconds"],
            "triggered_by": report["triggered_by"] if "triggered_by" in report.keys() else "cron",
            "overview": overview,
        },
    }
