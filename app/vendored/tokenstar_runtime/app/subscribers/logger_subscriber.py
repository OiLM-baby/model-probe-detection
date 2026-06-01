"""日志订阅者：把关键事件写入应用日志。"""

import logging

from app.events.bus import EventBus
from app.events.types import (
    PROVIDER_CRASHED,
    PROVIDER_DONE,
    PROVIDER_STARTED,
    REPORT_COMPLETED,
    REPORT_CRASHED,
    REPORT_STARTED,
    REQUEST_FAILED,
    ProgressEvent,
)

logger = logging.getLogger("tokenstar")


class LoggerSubscriber:
    """监听关键事件，写结构化日志。"""

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(REPORT_STARTED, self._on_report_started)
        bus.subscribe(PROVIDER_STARTED, self._on_provider_started)
        bus.subscribe(PROVIDER_DONE, self._on_provider_done)
        bus.subscribe(PROVIDER_CRASHED, self._on_provider_crashed)
        bus.subscribe(REQUEST_FAILED, self._on_request_failed)
        bus.subscribe(REPORT_COMPLETED, self._on_report_completed)
        bus.subscribe(REPORT_CRASHED, self._on_report_crashed)

    def _on_report_started(self, event: ProgressEvent) -> None:
        p = event.payload
        logger.info("report_started: env=%s suite=%s run_id=%s planned=%s",
                    p.get("env"), p.get("suite"), p.get("run_id"), p.get("planned"))

    def _on_provider_started(self, event: ProgressEvent) -> None:
        p = event.payload
        logger.info("provider_started: %s/%s", p.get("provider"), p.get("model"))

    def _on_provider_done(self, event: ProgressEvent) -> None:
        s = event.payload.get("summary")
        if s:
            logger.info("provider_done: %s/%s passed=%s failed=%s",
                        s["provider"], s["model"], s["passed"], s["failed"])

    def _on_provider_crashed(self, event: ProgressEvent) -> None:
        p = event.payload
        logger.error("provider_crashed: %s/%s error=%s",
                     p.get("provider"), p.get("model"), p.get("error", ""))

    def _on_request_failed(self, event: ProgressEvent) -> None:
        p = event.payload
        logger.warning("request_failed: %s/%s kind=%s test=%s",
                       p.get("provider"), p.get("model"),
                       p.get("error_kind"), p.get("test_name"))

    def _on_report_completed(self, event: ProgressEvent) -> None:
        p = event.payload
        logger.info("report_completed: run_id=%s", p.get("run_id", ""))

    def _on_report_crashed(self, event: ProgressEvent) -> None:
        p = event.payload
        logger.error("report_crashed: run_id=%s reason=%s",
                     p.get("run_id"), p.get("reason", ""))
