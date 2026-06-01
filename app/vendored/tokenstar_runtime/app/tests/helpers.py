"""公共工具函数、状态常量和关键词列表。"""

import json
import statistics

from app.utils.percentile import percentile

STATUS_PASS = "成功"
STATUS_FAIL = "失败"
STATUS_WARN = "警告"
STATUS_INFO = "信息"


def _status_by_score(score: float, min_score: float = 60) -> str:
    return STATUS_PASS if score >= min_score else STATUS_FAIL


def _score_slowness(latencies, first_token_times, chars_per_second_values) -> dict:
    if len(latencies) < 3:
        return {"classification": "inconclusive", "reason": "样本不足"}

    p50 = percentile(latencies, 50) or 0
    p95 = percentile(latencies, 95) or 0
    max_latency = max(latencies)

    jittery_network = (
        p50 > 0 and p95 >= p50 * 2.5
    ) or (
        p50 > 0 and max_latency >= p50 * 3.0
    )
    stable_network = not jittery_network

    avg_ttft = statistics.mean(first_token_times) if first_token_times else 0
    avg_cps = statistics.mean(chars_per_second_values) if chars_per_second_values else None
    model_compute_slow = stable_network and (
        avg_ttft >= max(2500, p50 + 1500)
        or (avg_cps is not None and avg_cps < 15.0)
    )

    if jittery_network and model_compute_slow:
        return {
            "classification": "mixed",
            "reason": f"网络抖动(p50={p50:.0f}ms p95={p95:.0f}ms) + 模型计算慢(ttft={avg_ttft:.0f}ms tps={avg_cps})",
        }
    if jittery_network:
        return {
            "classification": "network_jitter_likely",
            "reason": f"延迟方差大 p50={p50:.0f}ms p95={p95:.0f}ms max={max_latency:.0f}ms",
        }
    if model_compute_slow:
        return {
            "classification": "model_compute_likely",
            "reason": f"网络稳定但流式慢 ttft={avg_ttft:.0f}ms tps={avg_cps}",
        }
    return {
        "classification": "inconclusive",
        "reason": f"无明显异常 p50={p50:.0f}ms p95={p95:.0f}ms ttft={avg_ttft:.0f}ms",
    }


def _extract_cache_usage(fmt: str, response: dict) -> dict:
    """从响应中提取缓存使用字段，支持 Anthropic、OpenAI Chat 和 Responses。"""
    usage = (response or {}).get("usage", {})
    if fmt == "anthropic":
        return {
            "cache_creation": usage.get("cache_creation_input_tokens"),
            "cache_read": usage.get("cache_read_input_tokens"),
            "cached_tokens": usage.get("cache_read_input_tokens"),
            "has_cache_fields": "cache_creation_input_tokens" in usage or "cache_read_input_tokens" in usage,
        }
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached_tokens = details.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = usage.get("cached_tokens")
    return {
        "cache_creation": None,
        "cache_read": None,
        "cached_tokens": cached_tokens,
        "has_cache_fields": (
            "prompt_tokens_details" in usage
            or "input_tokens_details" in usage
            or "cached_tokens" in usage
        ),
    }


_REFUSE_KEYWORDS = [
    "无法", "不能", "抱歉", "拒绝", "不合规", "不安全",
    "cannot", "unable", "sorry", "decline", "illegal",
    "违反", "违法", "不允许", "不可以",
    "不能帮助", "不能提供行动方案", "不能满足",
]

_UNCERTAIN_KEYWORDS = [
    "无法确认", "不确定", "没有可靠", "不能确定", "无法判断",
    "不知道", "没有信息", "没有相关", "无法核实", "不能确认",
    "cannot confirm", "not sure", "no reliable",
]

_REFUSE_KEYWORDS_LOWER = [kw.lower() for kw in _REFUSE_KEYWORDS]
_UNCERTAIN_KEYWORDS_LOWER = [kw.lower() for kw in _UNCERTAIN_KEYWORDS]


def _check_refuse(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _REFUSE_KEYWORDS_LOWER)


def _check_uncertainty(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _UNCERTAIN_KEYWORDS_LOWER)


def _dialogue_check_ok(check: dict, text: str) -> bool:
    compact = text.replace(" ", "")
    if "contains" in check:
        return check["contains"].replace(" ", "") in compact
    if "must_include" in check:
        return all(word in text for word in check["must_include"])
    if "must_include_any" in check:
        return any(word in text for word in check["must_include_any"])
    if "json_contains" in check:
        try:
            payload = json.loads(_strip_json_fence(text))
        except Exception:
            return False
        return all(payload.get(key) == value for key, value in check["json_contains"].items())
    return bool(text)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text
