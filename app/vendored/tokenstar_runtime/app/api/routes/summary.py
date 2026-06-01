"""统计汇总路由。"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_db
from app.api.filters import VISIBLE_GROUP

router = APIRouter(tags=["summary"])


@router.get("/reports/{run_id}/summaries/family")
def summary_by_family(run_id: str, conn = Depends(get_db)) -> dict[str, Any]:
    rep = conn.execute("SELECT id FROM reports WHERE run_id=?", (run_id,)).fetchone()
    if not rep:
        raise HTTPException(status_code=404, detail=f"报告 {run_id} 不存在")

    rows = conn.execute(
        """SELECT model_family,
                  COUNT(*) AS provider_count,
                  ROUND(AVG(avg_latency_ms),1) AS avg_latency_ms,
                  ROUND(AVG(p95_latency_ms),1) AS avg_p95_latency_ms,
                  ROUND(COALESCE(SUM(hard_passed),0) * 1.0 / MAX(COALESCE(SUM(hard_total),0), 1) * 100, 2) AS hard_pass_rate,
                  ROUND(AVG(soft_avg_score),1) AS soft_health,
                  COALESCE(SUM(passed),0) AS total_passed,
                  COALESCE(SUM(failed),0) AS total_failed
           FROM provider_snapshot WHERE report_id=? AND group_name=?
           GROUP BY model_family ORDER BY provider_count DESC""",
        (rep[0], VISIBLE_GROUP),
    ).fetchall()

    return {"code": 0, "data": [dict(r) for r in rows]}


@router.get("/reports/{run_id}/summaries/group")
def summary_by_group(run_id: str, conn = Depends(get_db)) -> dict[str, Any]:
    rep = conn.execute("SELECT id FROM reports WHERE run_id=?", (run_id,)).fetchone()
    if not rep:
        raise HTTPException(status_code=404, detail=f"报告 {run_id} 不存在")

    rows = conn.execute(
        """SELECT group_name,
                  COUNT(*) AS provider_count,
                  ROUND(AVG(avg_latency_ms),1) AS avg_latency_ms,
                  ROUND(COALESCE(SUM(hard_passed),0) * 1.0 / MAX(COALESCE(SUM(hard_total),0), 1) * 100, 2) AS hard_pass_rate,
                  COALESCE(SUM(passed),0) AS total_passed,
                  COALESCE(SUM(failed),0) AS total_failed
           FROM provider_snapshot WHERE report_id=? AND group_name=?
           GROUP BY group_name ORDER BY provider_count DESC""",
        (rep[0], VISIBLE_GROUP),
    ).fetchall()

    return {"code": 0, "data": [dict(r) for r in rows]}
