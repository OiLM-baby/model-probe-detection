"""错误原因提取和分类。"""

from __future__ import annotations

import re
from typing import Any

ERROR_KEYS = ("error", "error_message", "error_msg", "proxy_error", "first_error", "second_error")
REQUEST_ID_RE = re.compile(r"\s*\(?request\s*id[:：]\s*[^)\s]+?\)?", re.IGNORECASE)
HTTP_RE = re.compile(r"\bHTTP\s+\d{3}\b", re.IGNORECASE)
TAG_RE = re.compile(r"\[[a-z_]+\]", re.IGNORECASE)
CONTENT_POLICY_KEYWORDS = (
    "sensitive",
    "prohibited",
    "blocked",
    "content policy",
    "safety",
    "敏感",
    "内容安全",
    "不合规",
    "风控",
    "被拦截",
)

CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("拒绝访问", ("http 403", "forbidden", "access denied", "permission denied", "已被弃用", "拒绝访问", "无权限", "权限不足", "禁止访问")),
    ("鉴权失败", ("[auth]", "http 401", "unauthorized", "invalid key", "invalid api key", "api key", "鉴权失败", "认证失败", "密钥无效", "令牌无效")),
    ("额度不足", ("[payment]", "credit quota exceeded", "quota exceeded", "insufficient balance", "balance", "额度不足", "余额不足", "预扣费额度失败", "预扣费失败")),
    ("限流", ("[rate_limit]", "http 429", "too many requests", "rate limit", "ratelimit", "限流", "频率限制", "请求过多")),
    ("模型不可用", ("http 404", "no endpoints found", "model not found", "model_not_found", "模型不存在", "模型不可用")),
    ("参数不兼容", ("http 400", "enable_thinking", "invalid parameter", "unsupported parameter", "parameter.", "参数错误", "参数不兼容")),
    ("上游异常", ("[server]", "http 500", "http 502", "http 503", "http 504", "no upstream route", "upstream", "provider is unavailable", "无可用渠道", "未配置渠道能力", "distributor", "渠道不可用", "上游异常", "服务不可用")),
    ("超时", ("[timeout]", "timeout", "timed out", "deadline", "read timeout", "connect timeout", "超时", "请求超时", "连接超时", "响应超过", "已断开连接")),
    ("网络不通", ("[network]", "connection refused", "connection reset", "dns", "tls", "ssl", "网络错误", "网络不通", "连接失败", "连接被重置")),
    ("响应格式错误", ("[non_json]", "invalid json", "empty response", "parse error", "空响应", "解析失败", "响应格式错误", "非 json")),
)


def classify_error(
    message: str = "",
    detail: dict[str, Any] | None = None,
    test_name: str = "",
    status: str = "",
) -> tuple[str, str]:
    """返回 (失败原因总结, 失败原因原文)。"""
    status_text = str(status or "").lower()
    if status_text in {"成功", "pass"}:
        return "无错误", ""

    raw = clean_error(_extract_error(detail) or message)
    if not raw:
        return "未知错误", ""

    text = raw.lower()
    for label, keywords in CATEGORY_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            return label, raw

    if test_name in {"daily_latency", "latency"}:
        return "延迟不达标", raw

    if test_name == "connectivity" and _looks_like_connectivity_summary(raw):
        return "连通不通过", raw

    if _looks_like_content_failure(raw, test_name):
        return "内容不合规", raw

    return "未知错误", raw


def clean_error(value: Any) -> str:
    """清理错误文本中的 request id 等噪音。"""
    text = str(value or "").strip()
    if not text:
        return ""
    text = REQUEST_ID_RE.sub("", text).strip()
    return text


def _extract_error(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            found = _extract_error(item)
            if found:
                return found
        return ""
    if isinstance(value, dict):
        for key in ERROR_KEYS:
            found = _extract_error(value.get(key))
            if found:
                return found
        for key in ("errors", "attempts"):
            found = _extract_error(value.get(key))
            if found:
                return found
    return ""


def _looks_like_content_failure(raw: str, test_name: str) -> bool:
    text = raw.strip()
    if test_name != "connectivity" or not text:
        return False
    if HTTP_RE.search(text) or TAG_RE.search(text):
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in CONTENT_POLICY_KEYWORDS) or _looks_like_short_model_response(text)


def _looks_like_connectivity_summary(raw: str) -> bool:
    return bool(re.search(r"连通\s*\d+/\d+", raw) or re.search(r"connectivity\s*\d+/\d+", raw, re.IGNORECASE))


def _looks_like_short_model_response(raw: str) -> bool:
    text = raw.strip()
    if not text or len(text) > 120:
        return False
    if re.search(r"[=:/\\]|\d", text):
        return False
    return True
