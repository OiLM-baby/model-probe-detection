"""渲染订阅者：监听 provider 完成事件，执行 debounced 中间报告渲染。"""

import logging
import os
import time

from app.events.bus import EventBus
from app.events.types import (
    PROVIDER_DONE,
    REPORT_COMPLETED,
    REPORT_CRASHED,
    REPORT_STARTED,
    ProgressEvent,
)
from app.report.generator import build_payload
from app.utils.baseline import load_baselines

logger = logging.getLogger("tokenstar")


class RenderSubscriber:
    """监听 provider_done 事件，增量构建 payload 并 debounced 渲染 HTML。

    report_completed / report_crashed 时 flush 强制最终渲染。
    """

    def __init__(self, renderer, report_dir: str = "reports", baseline_config_path: str = ""):
        self._renderer = renderer
        self._report_dir = report_dir
        self._baseline_config_path = baseline_config_path or _default_baseline_config_path()
        self._summaries: list = []
        self._start_time: float = 0
        self._env: str = ""
        self._run_id: str = ""
        self._suite: str = ""
        self._output_path: str = ""
        self._baseline_config: dict | None = None

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(REPORT_STARTED, self._on_report_started)
        bus.subscribe(PROVIDER_DONE, self._on_provider_done)
        bus.subscribe(REPORT_COMPLETED, self._on_report_completed)
        bus.subscribe(REPORT_CRASHED, self._on_report_crashed)

    def _on_report_started(self, event: ProgressEvent) -> None:
        p = event.payload
        self._start_time = event.timestamp
        self._env = p.get("env", "")
        self._run_id = p.get("run_id", "")
        self._suite = p.get("suite", "")
        self._baseline_config = _load_baseline_config(self._suite, self._baseline_config_path)
        self._output_path = self._output_path or _make_output_path(
            self._report_dir, self._env, self._suite, self._run_id
        )

    def _on_provider_done(self, event: ProgressEvent) -> None:
        from app.core.models import ProviderSummary, TestResult

        summary_dict = event.payload.get("summary")
        if summary_dict is None:
            return
        results = [
            TestResult(**item)
            for item in (event.payload.get("results") or summary_dict.get("results") or [])
        ]
        summary = ProviderSummary(
            **{k: v for k, v in summary_dict.items() if k != "results"},
            results=results,
        )
        self._summaries.append(summary)
        payload = build_payload(
            self._summaries,
            self._start_time,
            time.time(),
            env=self._env,
            run_id=self._run_id,
            suite=self._suite,
            baseline_config=self._baseline_config,
        )
        self._renderer.schedule(payload, self._output_path)

    def _on_report_completed(self, event: ProgressEvent) -> None:
        # 使用最终 payload（含 comparison/baseline）重新渲染
        payload_json = event.payload.get("payload_json", "")
        if payload_json:
            import json as _json

            try:
                final_payload = _json.loads(payload_json)
                self._renderer.schedule(final_payload, self._output_path)
            except Exception:
                logger.warning("最终 payload 解析失败，回退 flush")
        self._renderer.flush()

    def _on_report_crashed(self, event: ProgressEvent) -> None:
        self._renderer.flush()


def _make_output_path(report_dir: str, env: str, suite: str, run_id: str) -> str:
    archive_dir = os.path.join(report_dir, env, suite, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    return os.path.join(archive_dir, f"tokenstar_report_{run_id}.html")


def _default_baseline_config_path() -> str:
    env_path = os.environ.get("TOKENSTAR_BASELINES_CONFIG")
    if env_path:
        return env_path
    return os.path.join(_project_root(), "config", "baselines.yaml")


def _load_baseline_config(suite: str, config_path: str = "") -> dict | None:
    config_path = config_path or _default_baseline_config_path()
    if not os.path.exists(config_path):
        return None
    baseline = load_baselines(config_path, suite)
    source = os.path.relpath(config_path, _project_root())
    return baseline.to_dict(source=source)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
