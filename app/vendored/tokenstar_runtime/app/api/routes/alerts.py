"""告警日志路由。"""

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_db
from app.api.filters import VISIBLE_GROUP

router = APIRouter(tags=["alerts"])


@router.get("/alerts")
def list_alerts(
    report_id: int = Query(None, description="报告 ID"),
    provider: str = Query("", description="中转名称"),
    error_kind: str = Query("", description="错误类型"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    conn = Depends(get_db),
) -> dict[str, Any]:
    where = ["group_name=?"]
    params: list[Any] = [VISIBLE_GROUP]
    if report_id:
        where.append("report_id=?")
        params.append(report_id)
    if provider:
        where.append("provider LIKE ?")
        params.append(f"%{provider}%")
    if error_kind:
        where.append("error_kind=?")
        params.append(error_kind)

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    count_row = conn.execute(
        f"SELECT COUNT(*) FROM alert_log {where_clause}", params
    ).fetchone()
    total = count_row[0] if count_row else 0

    offset = (page - 1) * page_size
    rows = conn.execute(
        f"SELECT * FROM alert_log {where_clause} ORDER BY occurred_at DESC LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        d["notified_wechat"] = bool(d.get("notified_wechat"))
        d["notified_email"] = bool(d.get("notified_email"))
        items.append(d)

    return {
        "code": 0,
        "data": {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    }
