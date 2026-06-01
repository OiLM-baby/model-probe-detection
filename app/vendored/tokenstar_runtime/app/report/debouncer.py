"""Debounced 渲染器：接收渲染请求后在 debounce 窗口内只执行最后一次。

避免 provider 密集完成时高频重写 HTML 文件。
"""

import threading
import logging

logger = logging.getLogger("tokenstar")


class DebouncedRenderer:
    """包装一个 Renderer，在 debounce_secs 秒内多次 schedule 只执行一次 render。

    flush() 强制取消等待、立即执行尚未执行的渲染。
    """

    def __init__(self, renderer, debounce_secs: float = 10.0):
        self._renderer = renderer
        self._debounce_secs = debounce_secs
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._pending_payload: dict | None = None
        self._pending_path: str = ""

    def schedule(self, payload: dict, output_path: str) -> None:
        """安排一次渲染。如果 debounce 窗口内有新的 schedule，旧的取消。"""
        with self._lock:
            self._pending_payload = payload
            self._pending_path = output_path
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_secs, self._on_timer)
            self._timer.start()

    def flush(self) -> None:
        """强制立即执行当前排队的渲染（如有）。"""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            payload = self._pending_payload
            path = self._pending_path
            self._pending_payload = None
            self._pending_path = ""
        if payload and path:
            try:
                self._renderer.render(payload, path)
            except Exception:
                logger.exception("debounced render failed: path=%s", path)

    def _on_timer(self) -> None:
        with self._lock:
            self._timer = None
            payload = self._pending_payload
            path = self._pending_path
            self._pending_payload = None
            self._pending_path = ""
        if payload and path:
            self._renderer.render(payload, path)
