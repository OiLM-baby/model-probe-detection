"""报告相关路由。"""

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_db
from app.api.filters import VISIBLE_GROUP

router = APIRouter(tags=["reports"])


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["tags"] = json.loads(d.pop("tags_json", "[]") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["tags"] = []
    d.pop("payload_json", None)
    return d


@router.get("/reports")
def list_reports(
    env: str = Query("", description="环境过滤"),
    suite: str = Query("", description="套件过滤"),
    suite_type: str = Query("", description="套件类型"),
    status: str = Query("", description="状态"),
    start_date: str = Query("", description="开始日期 YYYY-MM-DD"),
    end_date: str = Query("", description="结束日期 YYYY-MM-DD"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    conn = Depends(get_db),
) -> dict[str, Any]:
    where = []
    params: list[Any] = []
    if env:
        where.append("env=?")
        params.append(env)
    if suite:
        where.append("suite=?")
        params.append(suite)
    if suite_type:
        where.append("suite_type=?")
        params.append(suite_type)
    if status:
        where.append("status=?")
        params.append(status)
    if start_date:
        where.append("started_at>=?")
        params.append(start_date)
    if end_date:
        where.append("started_at<=?")
        params.append(end_date + " 23:59:59")

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    count_row = conn.execute(
        f"SELECT COUNT(*) FROM reports {where_clause}", params
    ).fetchone()
    total = count_row[0] if count_row else 0

    offset = (page - 1) * page_size
    rows = conn.execute(
        f"SELECT * FROM reports {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()

    return {
        "code": 0,
        "data": {
            "items": [_row_to_dict(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    }


@router.get("/reports/{run_id}")
def get_report(run_id: str, conn = Depends(get_db)) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM reports WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"报告 {run_id} 不存在")

    report = _row_to_dict(row)
    report_id = report["id"]

    # 按模型家族汇总
    family_rows = conn.execute(
        """SELECT model_family,
                  COUNT(*) AS cnt,
                  COALESCE(SUM(passed),0) AS passed,
                  COALESCE(SUM(failed),0) AS failed,
                  ROUND(AVG(avg_latency_ms),1) AS avg_lat,
                  ROUND(AVG(p95_latency_ms),1) AS avg_p95
           FROM provider_snapshot WHERE report_id=? AND group_name=?
           GROUP BY model_family ORDER BY cnt DESC""",
        (report_id, VISIBLE_GROUP),
    ).fetchall()

    # 按中转组汇总
    group_rows = conn.execute(
        """SELECT group_name,
                  COUNT(*) AS cnt,
                  COALESCE(SUM(passed),0) AS passed,
                  COALESCE(SUM(failed),0) AS failed,
                  ROUND(AVG(avg_latency_ms),1) AS avg_lat
           FROM provider_snapshot WHERE report_id=? AND group_name=?
           GROUP BY group_name ORDER BY cnt DESC""",
        (report_id, VISIBLE_GROUP),
    ).fetchall()

    report["provider_summary"] = {
        "by_family": {
            r["model_family"]: {
                "total": r["cnt"], "passed": r["passed"], "failed": r["failed"],
                "avg_latency_ms": r["avg_lat"], "avg_p95_latency_ms": r["avg_p95"],
            }
            for r in family_rows
        },
        "by_group": {
            r["group_name"]: {
                "total": r["cnt"], "passed": r["passed"], "failed": r["failed"],
                "avg_latency_ms": r["avg_lat"],
            }
            for r in group_rows
        },
    }

    return {"code": 0, "data": report}
