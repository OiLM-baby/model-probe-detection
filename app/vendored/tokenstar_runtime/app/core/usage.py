"""统一 token 用量提取与成本估算。"""

import threading
from typing import Any

# 线程安全的用量收集器，LLMClient 自动推送，runner 读取后分发到 TestResult。
_collector = threading.local()


def start_usage_collection() -> None:
    _collector.records = []


def collect_usage(usage_dict: dict[str, int]) -> None:
    if hasattr(_collector, "records"):
        _collector.records.append(usage_dict)


def drain_usage_records() -> list[dict[str, int]]:
    records = getattr(_collector, "records", [])
    _collector.records = []
    return records


def extract_usage(provider_format: str, response: dict[str, Any] | None) -> dict[str, int]:
    """从三种格式的响应中统一提取 token 用量字段。"""
    empty = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "total_tokens": 0,
    }
    if not response:
        return empty
    usage = response.get("usage") or {}
    if not usage:
        return empty

    if provider_format == "anthropic":
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        cache_read = usage.get("cache_read_input_tokens") or 0
        cache_create = usage.get("cache_creation_input_tokens") or 0
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cache_read,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_create,
            "total_tokens": input_tokens + output_tokens,
        }

    # OpenAI / Responses 格式
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    if not cached:
        cached = usage.get("cached_tokens") or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached,
        "cache_read_tokens": cached,
        "cache_creation_tokens": 0,
        "total_tokens": usage.get("total_tokens") or (input_tokens + output_tokens),
    }


def lookup_price(pricing_config: dict | None, provider: str, model: str) -> dict:
    """按 provider+model 查找定价规则，四级优先级匹配。"""
    empty = {
        "input_per_1m": 0.0, "output_per_1m": 0.0,
        "cache_read_per_1m": 0.0, "cache_write_per_1m": 0.0,
        "matched": False,
    }
    if not pricing_config:
        return empty
    rules = pricing_config.get("rules") or []

    # 1. provider + model 精确匹配
    for rule in rules:
        p = rule.get("provider", "")
        m = rule.get("model", "")
        if p == provider and m == model:
            return _build_price(rule)

    # 2. provider + *
    for rule in rules:
        p = rule.get("provider", "")
        m = rule.get("model", "")
        if p == provider and m == "*":
            return _build_price(rule)

    # 3. * + model
    for rule in rules:
        p = rule.get("provider", "")
        m = rule.get("model", "")
        if p == "*" and m == model:
            return _build_price(rule)

    # 4. * + * 通用兜底
    for rule in rules:
        p = rule.get("provider", "")
        m = rule.get("model", "")
        if p == "*" and m == "*":
            return _build_price(rule)

    return empty


def _build_price(rule: dict) -> dict:
    return {
        "input_per_1m": float(rule.get("input_per_1m") or 0),
        "output_per_1m": float(rule.get("output_per_1m") or 0),
        "cache_read_per_1m": float(rule.get("cache_read_per_1m") or 0),
        "cache_write_per_1m": float(rule.get("cache_write_per_1m") or 0),
        "matched": True,
    }


def estimate_cost(usage: dict[str, int], price: dict) -> float | None:
    """根据 token 用量和单价估算成本，单位与 pricing.currency 一致。"""
    if not price.get("matched"):
        return None
    if not any(price.get(k) for k in ("input_per_1m", "output_per_1m", "cache_read_per_1m", "cache_write_per_1m")):
        return None  # 价格全为 0，视为未配置
    input_cost = (usage["input_tokens"] / 1_000_000) * price["input_per_1m"]
    output_cost = (usage["output_tokens"] / 1_000_000) * price["output_per_1m"]
    cache_read_cost = (usage["cache_read_tokens"] / 1_000_000) * price["cache_read_per_1m"]
    cache_create_cost = (usage["cache_creation_tokens"] / 1_000_000) * price["cache_write_per_1m"]
    return round(input_cost + output_cost + cache_read_cost + cache_create_cost, 6)
