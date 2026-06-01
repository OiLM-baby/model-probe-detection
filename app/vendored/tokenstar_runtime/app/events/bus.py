"""轻量同步事件总线。不引第三方依赖，30 行搞定。"""

import logging
from collections.abc import Callable

from app.events.types import ProgressEvent

logger = logging.getLogger("tokenstar")


class EventBus:
    """同步发布-订阅总线。注册 handler 按事件 kind 分发。"""

    def __init__(self):
        self._subscribers: dict[str, list[Callable[[ProgressEvent], None]]] = {}
        self._fail_counts: dict[tuple[int, str], int] = {}

    def subscribe(self, event_kind: str, handler: Callable[[ProgressEvent], None]) -> None:
        self._subscribers.setdefault(event_kind, []).append(handler)

    def publish(self, event: ProgressEvent) -> None:
        for handler in self._subscribers.get(event.kind, ()):
            try:
                handler(event)
                self._fail_counts.pop((id(handler), event.kind), None)
            except Exception:
                key = (id(handler), event.kind)
                count = self._fail_counts.get(key, 0) + 1
                self._fail_counts[key] = count
                if count <= 3 or count % 50 == 0:
                    logger.exception(
                        "subscriber failed on event %s (#%d): handler=%s",
                        event.kind, count, getattr(handler, '__qualname__', handler),
                    )
