"""Provider 快照相关路由。"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_db
from app.api.filters import VISIBLE_GROUP, visible_group_where
from app.utils.error_category import classify_error
from app.utils.timezone import BEIJING_TIME_FORMAT, BEIJING_TZ

router = APIRouter(tags=["providers"])
MAX_METRIC_BUCKETS = 500


class ModelMetricsRequest(BaseModel):
    start: str = Field("", description="开始时间，YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS；不传默认最近两天")
    end: str = Field("", description="结束时间，YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS；不传默认当前北京时间")
    bucket: str = Field("hour", description="统计维度：hour/day/month/year")
    model: str = Field("", description="模型名称模糊搜索")
    page: int = Field(1, ge=1)
    page_size: int = Field(200, ge=1, le=500)


class ModelHistoryRequest(BaseModel):
    model: str = Field(..., description="模型名称，精确匹配")
    start: str = Field(..., description="开始时间，YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS")
    end: str = Field(..., description="结束时间，YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS")
    group: str = Field(VISIBLE_GROUP, description="中转组，当前默认 clawos")
    timeline_bucket: str = Field("run", description="run/hour/day")
    recent_limit: int = Field(10, ge=0, le=50)


def _ps_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["tags"] = json.loads(d.pop("tags_json", "[]") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["tags"] = []
    d["price_matched"] = bool(d.get("price_matched"))
    ht = d.get("hard_total") or 0
    hp = d.get("hard_passed") or 0
    d["hard_pass_rate"] = round(hp / ht * 100, 2) if ht else None
    return d


def _json_detail(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _test_row(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {"status": "", "latency_ms": None, "message": "", "detail": {}}
    return {
        "status": row["status"],
        "latency_ms": row["latency_ms"],
        "message": row["message"] or "",
        "detail": _json_detail(row["detail_json"]),
    }


def _normalize_range(start: str, end: str) -> tuple[str, str]:
    start = start.strip()
    end = end.strip()
    if len(end) == 10:
        end = f"{end} 23:59:59"
    return start, end


def _normalize_metrics_range(start: str, end: str) -> tuple[str, str]:
    start = start.strip()
    end = end.strip()
    if not end:
        end_dt = datetime.now(BEIJING_TZ)
        end = end_dt.strftime(BEIJING_TIME_FORMAT)
    elif len(end) == 10:
        end = f"{end} 23:59:59"
        end_dt = datetime.strptime(end, BEIJING_TIME_FORMAT).replace(tzinfo=BEIJING_TZ)
    else:
        end_dt = datetime.strptime(end, BEIJING_TIME_FORMAT).replace(tzinfo=BEIJING_TZ)
    if not start:
        start = (end_dt - timedelta(days=2)).strftime(BEIJING_TIME_FORMAT)
    elif len(start) == 10:
        start = f"{start} 00:00:00"
    return start, end


def _first_token_ms(latency: dict[str, Any]) -> Any:
    detail = latency.get("detail") or {}
    value = detail.get("avg_first_token_ms")
    if value is None or value == "":
        value = detail.get("first_token_ms")
    return value


def _status_pass(status: str) -> bool:
    return status in {"成功", "pass"}


def _rate(passed: int, total: int) -> float | None:
    return round(passed / total * 100, 2) if total else None


def _bucket_key(ts: str, bucket: str) -> str:
    if bucket == "year":
        return ts[:4]
    if bucket == "month":
        return ts[:7]
    if bucket == "day":
        return ts[:10]
    if bucket == "hour":
        return ts[:13]
    return ts


def _parse_beijing_time(value: str) -> datetime:
    return datetime.strptime(value, BEIJING_TIME_FORMAT).replace(tzinfo=BEIJING_TZ)


def _bucket_floor(dt: datetime, bucket: str) -> datetime:
    if bucket == "year":
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    if bucket == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if bucket == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)


def _add_bucket(dt: datetime, bucket: str) -> datetime:
    if bucket == "year":
        return dt.replace(year=dt.year + 1)
    if bucket == "month":
        year = dt.year + 1 if dt.month == 12 else dt.year
        month = 1 if dt.month == 12 else dt.month + 1
        return dt.replace(year=year, month=month)
    if bucket == "day":
        return dt + timedelta(days=1)
    return dt + timedelta(hours=1)


def _bucket_label(dt: datetime, bucket: str) -> str:
    return _bucket_key(dt.strftime(BEIJING_TIME_FORMAT), bucket)


def _metric_buckets(start: str, end: str, bucket: str) -> tuple[list[str], bool]:
    start_dt = _bucket_floor(_parse_beijing_time(start), bucket)
    end_dt = _bucket_floor(_parse_beijing_time(end), bucket)
    labels = []
    current = start_dt
    while current <= end_dt:
        if len(labels) >= MAX_METRIC_BUCKETS:
            return labels, True
        labels.append(_bucket_label(current, bucket))
        current = _add_bucket(current, bucket)
    return labels, False


def _running_metric_buckets(conn, start: str, end: str, bucket: str) -> set[str]:
    rows = conn.execute(
        """SELECT started_at FROM reports
           WHERE started_at>=? AND started_at<=? AND status='running'""",
        (start, end),
    ).fetchall()
    return {_bucket_key(row["started_at"], bucket) for row in rows}


def _avg(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None and v != ""]
    return round(sum(nums) / len(nums), 1) if nums else None


def _metric_first_token(tests: dict[str, sqlite3.Row]) -> Any:
    first_token_test = tests.get("first_token_connectivity")
    if first_token_test is not None:
        detail = _json_detail(first_token_test["detail_json"])
        value = detail.get("first_token_ms")
        if value is not None and value != "":
            return value
    latency = _test_row(tests.get("daily_latency") or tests.get("latency"))
    return _first_token_ms(latency)


def _metric_connected(tests: dict[str, sqlite3.Row]) -> bool:
    first_token_test = tests.get("first_token_connectivity")
    if first_token_test is not None:
        return _status_pass(first_token_test["status"])
    connectivity = tests.get("connectivity")
    if connectivity is not None:
        return _status_pass(connectivity["status"])
    latency = tests.get("daily_latency") or tests.get("latency")
    if latency is not None:
        return _status_pass(latency["status"])
    return False


def _model_metric_item(row: sqlite3.Row, tests: dict[str, sqlite3.Row]) -> dict[str, Any]:
    connectivity = _test_row(tests.get("connectivity"))
    latency = _test_row(tests.get("daily_latency") or tests.get("latency"))
    first_token = _first_token_ms(latency)
    latency_block = {
        **latency,
        "avg_latency_ms": row["avg_latency_ms"],
        "p95_latency_ms": row["p95_latency_ms"],
        "first_token_ms": first_token,
    }
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "env": row["env"],
        "suite": row["suite"],
        "report_started_at": row["report_started_at"],
        "report_finished_at": row["report_finished_at"],
        "provider": row["provider"],
        "group": row["group_name"],
        "model": row["model"],
        "status": row["status"],
        "connectivity": connectivity,
        "latency": latency_block,
        "first_token_ms": first_token,
    }


def _load_metric_tests(conn, snapshot_ids: list[int]) -> dict[int, dict[str, sqlite3.Row]]:
    tests_by_snapshot: dict[int, dict[str, sqlite3.Row]] = {}
    if not snapshot_ids:
        return tests_by_snapshot
    placeholders = ",".join("?" for _ in snapshot_ids)
    test_rows = conn.execute(
        f"""SELECT provider_snapshot_id, test_name, status, latency_ms, message, detail_json
            FROM test_results
            WHERE provider_snapshot_id IN ({placeholders})
              AND test_name IN ('connectivity','daily_latency','latency','first_token_connectivity')
            ORDER BY id""",
        snapshot_ids,
    ).fetchall()
    for test in test_rows:
        tests_by_snapshot.setdefault(test["provider_snapshot_id"], {})[test["test_name"]] = test
    return tests_by_snapshot


@router.post("/metrics")
def list_model_metrics(
    body: ModelMetricsRequest,
    conn = Depends(get_db),
) -> dict[str, Any]:
    """按时间桶返回前端需要的模型连通性和首 Token 指标。"""
    bucket = body.bucket.strip() or "hour"
    if bucket not in {"hour", "day", "month", "year"}:
        return {"code": 400, "msg": "bucket 仅支持 hour/day/month/year", "data": {"models": []}}
    start, end = _normalize_metrics_range(body.start, body.end)
    if not start or not end:
        return {"code": 400, "msg": "start 和 end 不能为空", "data": {"models": []}}
    bucket_list, too_many_buckets = _metric_buckets(start, end, bucket)
    if too_many_buckets:
        return {
            "code": 400,
            "msg": "时间范围过大，请缩小范围或使用更大的 bucket 粒度",
            "data": {
                "bucket": bucket,
                "models": [],
                "total": 0,
                "page": body.page,
                "page_size": body.page_size,
                "start": start,
                "end": end,
                "max_buckets": MAX_METRIC_BUCKETS,
            },
        }

    where = ["r.started_at>=?", "r.started_at<=?", "ps.group_name=?"]
    params: list[Any] = [start, end, VISIBLE_GROUP]
    if body.model:
        where.append("ps.model LIKE ?")
        params.append(f"%{body.model}%")
    where_clause = " AND ".join(where)

    rows = conn.execute(
        f"""SELECT ps.*, r.run_id, r.env, r.suite, r.status AS report_status,
                   r.started_at AS report_started_at,
                   r.finished_at AS report_finished_at
            FROM provider_snapshot ps
            JOIN reports r ON r.id=ps.report_id
            WHERE {where_clause}
            ORDER BY ps.model ASC, r.started_at ASC, ps.id ASC""",
        params,
    ).fetchall()
    if not rows:
        return {
            "code": 0,
            "msg": "ok",
            "data": {
                "bucket": bucket,
                "models": [],
                "total": 0,
                "page": body.page,
                "page_size": body.page_size,
                "start": start,
                "end": end,
            },
        }

    snapshot_ids = [row["id"] for row in rows]
    tests_by_snapshot = _load_metric_tests(conn, snapshot_ids)
    running_buckets = _running_metric_buckets(conn, start, end, bucket)

    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
    model_meta: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        tests = tests_by_snapshot.get(row["id"], {})
        model_key = (row["provider"], row["group_name"], row["model"])
        bucket_time = _bucket_key(row["report_started_at"], bucket)
        model_meta[model_key] = {
            "provider": row["group_name"],
            "group": row["group_name"],
            "model": row["model"],
        }
        grouped.setdefault(model_key, {}).setdefault(bucket_time, []).append({
            "report_started_at": row["report_started_at"],
            "report_status": row["report_status"],
            "snapshot_id": row["id"],
            "connected": _metric_connected(tests),
            "first_token_ms": _metric_first_token(tests),
        })

    models = []
    for model_key in sorted(grouped, key=lambda key: (key[2], key[0])):
        bucket_map = grouped[model_key]
        report_list = []
        for bucket_time in sorted(bucket_list, reverse=True):
            items = bucket_map.get(bucket_time) or []
            if items:
                latest = max(
                    items,
                    key=lambda item: (item["report_started_at"], item["snapshot_id"]),
                )
                report_list.append({
                    "bucket_time": bucket_time,
                    "connected": latest["connected"],
                    "first_token_ms": latest["first_token_ms"],
                    "data_status": "complete",
                })
            else:
                status = "running" if bucket_time in running_buckets else "missing"
                report_list.append({
                    "bucket_time": bucket_time,
                    "connected": None,
                    "first_token_ms": None,
                    "data_status": status,
                })
        models.append({**model_meta[model_key], "report_list": report_list})

    total = len(models)
    offset = (body.page - 1) * body.page_size
    page_models = models[offset:offset + body.page_size]

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "bucket": bucket,
            "models": page_models,
            "total": total,
            "page": body.page,
            "page_size": body.page_size,
            "start": start,
            "end": end,
        },
    }


@router.post("/history")
def get_model_history(
    body: ModelHistoryRequest,
    conn = Depends(get_db),
) -> dict[str, Any]:
    """单模型下钻历史。"""
    start, end = _normalize_range(body.start, body.end)
    model = body.model.strip()
    group = (body.group or VISIBLE_GROUP).strip() or VISIBLE_GROUP
    bucket = body.timeline_bucket if body.timeline_bucket in {"run", "hour", "day"} else "run"
    if not model or not start or not end:
        return {"code": 400, "msg": "model、start 和 end 不能为空", "data": {}}
    if group != VISIBLE_GROUP:
        return {"code": 400, "msg": "当前 API 只支持 clawos 中转组", "data": {}}

    rows = conn.execute(
        """SELECT ps.*, r.run_id, r.env, r.suite, r.started_at AS report_started_at,
                  r.finished_at AS report_finished_at
           FROM provider_snapshot ps
           JOIN reports r ON r.id=ps.report_id
           WHERE r.started_at>=? AND r.started_at<=?
             AND ps.group_name=? AND ps.model=?
           ORDER BY r.started_at ASC, ps.id ASC""",
        (start, end, group, model),
    ).fetchall()
    snapshot_ids = [row["id"] for row in rows]
    tests_by_snapshot = _load_metric_tests(conn, snapshot_ids)

    items = [_model_metric_item(row, tests_by_snapshot.get(row["id"], {})) for row in rows]
    connectivity_total = sum(1 for item in items if item["connectivity"]["status"])
    connectivity_passed = sum(1 for item in items if _status_pass(item["connectivity"]["status"]))
    latency_total = sum(1 for item in items if item["latency"]["status"])
    latency_passed = sum(1 for item in items if _status_pass(item["latency"]["status"]))
    overview = {
        "run_count": len(items),
        "connectivity_passed": connectivity_passed,
        "connectivity_total": connectivity_total,
        "connectivity_pass_rate": _rate(connectivity_passed, connectivity_total),
        "latency_passed": latency_passed,
        "latency_total": latency_total,
        "latency_pass_rate": _rate(latency_passed, latency_total),
        "avg_latency_ms": _avg([item["latency"]["avg_latency_ms"] for item in items]),
        "p95_latency_ms": _avg([item["latency"]["p95_latency_ms"] for item in items]),
        "avg_first_token_ms": _avg([item["latency"]["first_token_ms"] for item in items]),
        "first_seen": items[0]["report_started_at"] if items else None,
        "last_seen": items[-1]["report_started_at"] if items else None,
    }

    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = item["run_id"] if bucket == "run" else _bucket_key(item["report_started_at"], bucket)
        buckets.setdefault(key, []).append(item)
    timeline = []
    for key, bucket_items in buckets.items():
        conn_total = sum(1 for item in bucket_items if item["connectivity"]["status"])
        conn_passed = sum(1 for item in bucket_items if _status_pass(item["connectivity"]["status"]))
        lat_total = sum(1 for item in bucket_items if item["latency"]["status"])
        lat_passed = sum(1 for item in bucket_items if _status_pass(item["latency"]["status"]))
        first = bucket_items[0]
        timeline.append({
            "bucket": key,
            "run_id": first["run_id"] if bucket == "run" else "",
            "time": first["report_started_at"] if bucket == "run" else key,
            "connectivity_pass_rate": _rate(conn_passed, conn_total),
            "latency_pass_rate": _rate(lat_passed, lat_total),
            "avg_latency_ms": _avg([item["latency"]["avg_latency_ms"] for item in bucket_items]),
            "p95_latency_ms": _avg([item["latency"]["p95_latency_ms"] for item in bucket_items]),
            "avg_first_token_ms": _avg([item["latency"]["first_token_ms"] for item in bucket_items]),
        })

    recent = []
    error_counts: dict[str, dict[str, Any]] = {}
    for item in reversed(items):
        compact = {
            "run_id": item["run_id"],
            "time": item["report_started_at"],
            "connectivity": _compact_test_for_history(item["connectivity"], "connectivity"),
            "latency": {
                **_compact_test_for_history(item["latency"], "latency"),
                "avg_latency_ms": item["latency"]["avg_latency_ms"],
                "p95_latency_ms": item["latency"]["p95_latency_ms"],
                "first_token_ms": item["latency"]["first_token_ms"],
            },
        }
        if len(recent) < body.recent_limit:
            recent.append(compact)
        for test_name in ("connectivity", "latency"):
            block = compact[test_name]
            category = block["error_category"]
            if category and category != "无错误":
                slot = error_counts.setdefault(category, {"error_category": category, "count": 0, "last_seen": item["report_started_at"]})
                slot["count"] += 1
                slot["last_seen"] = max(slot["last_seen"], item["report_started_at"])

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "model": model,
            "group": group,
            "time_range": {"start": body.start, "end": body.end},
            "overview": overview,
            "timeline": timeline,
            "recent": recent,
            "error_distribution": sorted(error_counts.values(), key=lambda item: item["count"], reverse=True),
        },
    }


def _compact_test_for_history(test: dict[str, Any], test_name: str) -> dict[str, Any]:
    category, detail = classify_error(
        message=test.get("message") or "",
        detail=test.get("detail") or {},
        test_name=test_name,
        status=test.get("status") or "",
    )
    return {
        "status": test.get("status") or "",
        "latency_ms": test.get("latency_ms"),
        "message": test.get("message") or "",
        "error_category": category,
        "error_detail": detail,
    }


@router.get("/reports/{run_id}/providers")
def list_providers(
    run_id: str,
    model_family: str = Query("", description="模型家族"),
    group_name: str = Query("", description="中转组"),
    status: str = Query("", description="状态"),
    model: str = Query("", description="模型名称模糊搜索"),
    sort_by: str = Query("avg_latency_ms", description="排序字段"),
    order: str = Query("asc", description="asc/desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    conn = Depends(get_db),
) -> dict[str, Any]:
    allowed_sorts = {"avg_latency_ms", "p95_latency_ms", "hard_pass_rate"}
    if sort_by not in allowed_sorts:
        sort_by = "avg_latency_ms"
    order_dir = "DESC" if order.lower() == "desc" else "ASC"

    rep = conn.execute("SELECT id FROM reports WHERE run_id=?", (run_id,)).fetchone()
    if not rep:
        raise HTTPException(status_code=404, detail=f"报告 {run_id} 不存在")

    report_id = rep[0]
    where = ["report_id=?", visible_group_where()]
    params: list[Any] = [report_id, VISIBLE_GROUP]
    if model_family:
        where.append("model_family=?")
        params.append(model_family)
    if group_name:
        where.append("group_name=?")
        params.append(group_name)
    if status:
        where.append("status=?")
        params.append(status)
    if model:
        where.append("model LIKE ?")
        params.append(f"%{model}%")

    where_clause = "WHERE " + " AND ".join(where)
    count_row = conn.execute(
        f"SELECT COUNT(*) FROM provider_snapshot {where_clause}", params
    ).fetchone()
    total = count_row[0] if count_row else 0

    offset = (page - 1) * page_size
    if sort_by == "hard_pass_rate":
        sort_expr = "CAST(COALESCE(hard_passed,0) AS REAL) / MAX(COALESCE(hard_total,0), 1)"
    else:
        sort_expr = sort_by
    rows = conn.execute(
        f"SELECT * FROM provider_snapshot {where_clause} ORDER BY {sort_expr} {order_dir} LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()

    return {
        "code": 0,
        "data": {
            "items": [_ps_row(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    }


@router.get("/providers/{provider_id}")
def get_provider(provider_id: int, conn = Depends(get_db)) -> dict[str, Any]:
    row = conn.execute(
        f"SELECT * FROM provider_snapshot WHERE id=? AND {visible_group_where()}", (provider_id, VISIBLE_GROUP)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} 不存在")

    ps = _ps_row(row)

    tr_rows = conn.execute(
        """SELECT test_name, status, score, latency_ms,
                  message, evaluation_type, detail_json
           FROM test_results WHERE provider_snapshot_id=? ORDER BY id""",
        (provider_id,),
    ).fetchall()

    tests = []
    for tr in tr_rows:
        t = dict(tr)
        detail_raw = t.pop("detail_json", None)
        try:
            t["detail"] = json.loads(detail_raw) if detail_raw else {}
        except (json.JSONDecodeError, TypeError):
            t["detail"] = {}
        tests.append(t)

    ps["tests"] = tests
    return {"code": 0, "data": ps}
