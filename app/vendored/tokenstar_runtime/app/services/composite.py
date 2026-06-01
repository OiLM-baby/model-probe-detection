"""多渠道聚合通知。同时推送到所有已注册的 Notifier。"""

import logging

logger = logging.getLogger("tokenstar")


class CompositeNotifier:
    """组合通知器：把一条消息广播到多个渠道。"""

    def __init__(self, notifiers: list | None = None):
        self._notifiers: list = notifiers or []

    def add(self, notifier) -> None:
        self._notifiers.append(notifier)

    def __bool__(self) -> bool:
        return bool(self._notifiers)

    def is_empty(self) -> bool:
        return not self._notifiers

    def send_summary(self, content: str,
                     attachment_path: str | None = None) -> bool:
        if not self._notifiers:
            return False
        ok = True
        for n in self._notifiers:
            try:
                if not n.send_summary(content, attachment_path):
                    ok = False
            except Exception:
                logger.exception("通知渠道 %s 发送失败", type(n).__name__)
                ok = False
        return ok
