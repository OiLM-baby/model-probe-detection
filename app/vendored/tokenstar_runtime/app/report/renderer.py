"""报告渲染模块。把 payload 渲染成 HTML 文件。

Render 逻辑从 generator.py 抽出，实现 Renderer Protocol。
"""

import json
import logging
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger("tokenstar")

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_DEFAULT_TEMPLATE = "live_board_template.html"

_SUITE_TEMPLATE_MAP = {
    "availability": "availability_template.html",
    "daily": "availability_template.html",
    "daily_full": "availability_template.html",
    "connectivity_matrix": "availability_template.html",
    "model_audit": "model_audit_light_template.html",
    "protocol_audit": "model_audit_light_template.html",
    "capability_probe": "model_audit_light_template.html",
    "audit": "model_audit_deep_template.html",
    "cache_audit": "model_audit_deep_template.html",
    "concurrency_audit": "model_audit_deep_template.html",
    "political_sensitivity": "model_audit_deep_template.html",
    "all": _DEFAULT_TEMPLATE,
    "all_no_political": _DEFAULT_TEMPLATE,
}


@runtime_checkable
class Renderer(Protocol):
    """报告渲染接口。"""

    def render(self, payload: dict, output_path: str) -> None: ...


def _template_name_for_suite(suite: str | None) -> str:
    return _SUITE_TEMPLATE_MAP.get(suite or "", _DEFAULT_TEMPLATE)


class HtmlRenderer:
    """HTML 报告渲染器。"""

    def __init__(self, template_dir: str | None = None):
        self.template_dir = template_dir or _TEMPLATE_DIR

    def render(self, payload: dict, output_path: str) -> None:
        html_content = self._render(payload)
        if html_content is None:
            html_content = self._fallback_html(payload)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html_content)

    @staticmethod
    def _fallback_html(payload: dict) -> str:
        summary = payload.get("summary", {})
        html = "<html><head><meta charset='utf-8'><title>TokenStar Report</title></head><body>"
        html += "<h1>TokenStar Report</h1><pre>"
        for k, v in sorted(summary.items()):
            html += f"{k}: {v}\n"
        html += f"providers: {len(payload.get('providers', []))}\n"
        html += "</pre></body></html>"
        return html

    def _render(self, payload: dict) -> str | None:
        """使用模板渲染 HTML，失败返回 None 由调用方回退。"""
        suite = payload.get("suite", "")
        template_name = _template_name_for_suite(suite)
        template_path = os.path.join(self.template_dir, template_name)
        try:
            with open(template_path, encoding="utf-8") as handle:
                template = handle.read()
            payload_json = (
                json.dumps(payload, ensure_ascii=False)
                .replace("<", "\\u003c")
                .replace(">", "\\u003e")
                .replace("&", "\\u0026")
                .replace(" ", "\\u2028")
                .replace(" ", "\\u2029")
            )
            return template.replace("__TOKENSTAR_REPORT_JSON__", payload_json, 1)
        except Exception:
            logger.exception("HTML template render failed: template=%s", template_path)
            return None
