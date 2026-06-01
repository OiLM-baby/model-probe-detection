"""事件类型定义。runner 通过 EventBus 发布这些事件，subscriber 订阅处理。"""

import time
from dataclasses import dataclass, field
from typing import Any


# ── Event dataclass ─────────────────────────────────────────


@dataclass
class ProgressEvent:
    """一次运行中的进度事件。kind 决定 payload 结构。"""

    kind: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)


# ── 事件 kind 常量 ──────────────────────────────────────────

REPORT_STARTED = "report_started"
PROVIDER_STARTED = "provider_started"
PROVIDER_DONE = "provider_done"
PROVIDER_CRASHED = "provider_crashed"
REQUEST_FAILED = "request_failed"
REPORT_COMPLETED = "report_completed"
REPORT_CRASHED = "report_crashed"
