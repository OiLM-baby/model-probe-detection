"""错误原因分类。"""

from __future__ import annotations

from typing import Any


CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("拒绝访问", ("http 403", "forbidden", "access denied", "permission denied", "已被弃用", "拒绝访问", "无权限", "权限不足", "禁止访问")),
    ("鉴权失败", ("[auth]", "http 401", "unauthorized", "invalid key", "invalid api key", "api key", "鉴权失败", "认证失败", "密钥无效", "令牌无效")),
    ("额度不足", ("[payment]", "credit quota exceeded", "quota exceeded", "insufficient balance", "balance", "额度不足", "余额不足", "预扣费额度失败", "预扣费失败")),
    ("限流", ("[rate_limit]", "http 429", "too many request", "too many requests", "rate limit", "ratelimit", "限流", "频率限制", "请求过多")),
    ("模型不可用", ("http 404", "no endpoints found", "model not found", "model_not_found", "模型不存在", "模型不可用")),
    ("参数不兼容", ("http 400", "enable_thinking", "invalid parameter", "unsupported parameter", "parameter.", "参数错误", "参数不兼容")),
    ("上游异常", ("[server]", "http 500", "http 502", "http 503", "http 504", "no upstream route", "no available resources", "upstream", "provider is unavailable", "无可用渠道", "未配置渠道能力", "distributor", "渠道不可用", "上游异常", "服务不可用")),
    ("超时", ("[timeout]", "timeout", "timed out", "deadline", "read timeout", "connect timeout", "超时", "请求超时", "连接超时", "响应超过", "已断开连接")),
    ("网络不通", ("[network]", "connection refused", "connection reset", "dns", "tls", "ssl", "网络错误", "网络不通", "连接失败", "连接被重置")),
    ("响应格式错误", ("[non_json]", "invalid json", "empty response", "parse error", "空响应", "解析失败", "响应格式错误", "非 json")),
)


def classify_error(message: Any) -> str:
    text = str(message or "").strip().lower()
    if not text:
        return ""
    for label, keywords in CATEGORY_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            return label
    return "未知错误"
