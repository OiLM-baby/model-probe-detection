"""存储订阅者：把 ProgressEvent 写入 ReportStore。"""

import logging
import threading

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

logger = logging.getLogger("tokenstar")


class StoreSubscriber:
    """监听进度事件，写入 SQLite 存储。"""

    def __init__(self, store):
        self._store = store
        self._report_id: int | None = None
        self._snapshot_ids: dict[str, int] = {}
        self._lock = threading.Lock()

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(REPORT_STARTED, self._on_report_started)
        bus.subscribe(PROVIDER_STARTED, self._on_provider_started)
        bus.subscribe(PROVIDER_DONE, self._on_provider_done)
        bus.subscribe(PROVIDER_CRASHED, self._on_provider_crashed)
        bus.subscribe(REPORT_COMPLETED, self._on_report_completed)
        bus.subscribe(REPORT_CRASHED, self._on_report_crashed)

    @property
    def report_id(self) -> int | None:
        return self._report_id

    # ── handlers ──────────────────────────────────────────

    def _on_report_started(self, event: ProgressEvent) -> None:
        p = event.payload
        self._report_id = self._store.start_report(
            env=p.get("env", ""),
            suite=p.get("suite", ""),
            run_id=p.get("run_id", ""),
            planned=p.get("planned", 0),
            suite_type=p.get("suite_type", ""),
            template_name=p.get("template_name", ""),
            triggered_by=p.get("triggered_by", "cron"),
        )
        event.payload["report_id"] = self._report_id
        logger.info("report started: id=%s", self._report_id)

    def _on_provider_started(self, event: ProgressEvent) -> None:
        if self._report_id is None:
            return
        p = event.payload
        sid = self._store.mark_provider_started(
            report_id=self._report_id,
            provider=p.get("provider", ""),
            model=p.get("model", ""),
            group_name=p.get("group_name", ""),
            model_family=p.get("model_family", ""),
            provider_format=p.get("provider_format", ""),
            base_url=p.get("base_url", ""),
            tags_json=p.get("tags_json", "[]"),
        )
        key = f"{p.get('provider','')}__{p.get('model','')}"
        with self._lock:
            self._snapshot_ids[key] = sid

    def _on_provider_done(self, event: ProgressEvent) -> None:
        from app.core.models import ProviderSummary, TestResult

        p = event.payload
        summary_dict = p.get("summary")
        results_raw = p.get("results") or []
        summary = ProviderSummary(
            **{k: v for k, v in summary_dict.items() if k != "results"},
            results=[],
        ) if summary_dict else None
        results = [TestResult(**r) for r in results_raw]
        key = f"{summary.provider}__{summary.model}" if summary else ""
        with self._lock:
            sid = self._snapshot_ids.get(key)
        if sid is None:
            return
        self._store.mark_provider_done(sid, summary, results)

    def _on_provider_crashed(self, event: ProgressEvent) -> None:
        p = event.payload
        key = f"{p.get('provider','')}__{p.get('model','')}"
        with self._lock:
            sid = self._snapshot_ids.get(key)
        if sid is None:
            return
        error = p.get("error", "unknown")
        self._store.mark_provider_crashed(sid, error)
        self._store.write_alert(
            snapshot_id=sid,
            error_kind="crashed",
            error_msg=error,
            provider=p.get("provider", ""),
            model=p.get("model", ""),
            group_name=p.get("group_name", ""),
        )

    def _on_report_completed(self, event: ProgressEvent) -> None:
        if self._report_id is None:
            return
        self._store.mark_report_completed(self._report_id, "")

    def _on_report_crashed(self, event: ProgressEvent) -> None:
        if self._report_id is None:
            return
        reason = event.payload.get("reason", "unknown")
        self._store.mark_report_crashed(self._report_id, reason, "")
