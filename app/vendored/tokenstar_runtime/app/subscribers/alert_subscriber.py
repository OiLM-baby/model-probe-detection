"""告警订阅者：把请求失败事件写入 alert_log，供报告/API 查询。"""

import logging

from app.events.bus import EventBus
from app.events.types import (
    REPORT_STARTED,
    REQUEST_FAILED,
    ProgressEvent,
)

logger = logging.getLogger("tokenstar")


class AlertSubscriber:
    """监听请求失败事件，写入 alert_log 持久化。

    当前主流程只发送日报/邮件汇总，不再自动聚合发送 alert_log。
    PROVIDER_CRASHED 的告警写入由 StoreSubscriber 负责，此处不再重复订阅。
    """

    def __init__(self, store):
        self._store = store
        self._report_id: int | None = None

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(REPORT_STARTED, self._on_report_started)
        bus.subscribe(REQUEST_FAILED, self._on_request_failed)

    # ── handlers ──────────────────────────────────────────

    def _on_report_started(self, event: ProgressEvent) -> None:
        self._report_id = event.payload.get("report_id")

    def _on_request_failed(self, event: ProgressEvent) -> None:
        p = event.payload
        sid = p.get("snapshot_id")
        self._store.write_alert(
            snapshot_id=sid if sid and sid > 0 else None,
            error_kind=p.get("error_kind", "unknown"),
            error_msg=p.get("error_msg", ""),
            test_name=p.get("test_name", ""),
            report_id=self._report_id,
            provider=p.get("provider", ""),
            model=p.get("model", ""),
            group_name=p.get("group_name", ""),
        )
