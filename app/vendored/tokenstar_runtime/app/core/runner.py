"""测试执行器。

负责按 provider 配置中的 tests 列表调度具体测试项，
并把多条 TestResult 汇总成 ProviderSummary。
"""

import logging
import statistics
from typing import Any

from app.core.judge import apply_judge
from app.core.models import ProviderConfig, ProviderSummary, TestResult
from app.core.usage import drain_usage_records, estimate_cost, lookup_price, start_usage_collection
from app.tests.cases import EVALUATION_TYPES, TEST_REGISTRY
from app.utils.percentile import percentile

# 请求级故障（连接失败、5xx、额度不足等），检测到后停止该 provider 后续测试。
# 注意：不包含 ReadTimeout / "timed out" — 读超时可能是代理层按内容过滤，应跳过而非停止。
# 仅匹配 LLMClient 输出的结构化 [tag] 前缀，避免误判模型回复正文中的关键字。
REQUEST_FAILURE_PATTERNS = (
    "[auth]",
    "[payment]",
    "[rate_limit]",
    "[server]",
    "[network]",
    "[non_json]",
)

# 中转返回的明文错误关键字，仅在 result.message 上匹配（不递归到 detail 字符串里）。
REQUEST_FAILURE_KEYWORDS = (
    "Credit quota exceeded",
    "预扣费额度失败",
    "额度不足",
    "未配置渠道能力",
    "no upstream route",
    "requires verification",
    "account requires verification",
)


def run_provider(provider: ProviderConfig, thresholds: dict, tests: list[str] | None = None, *, pricing_config: dict[str, Any] | None = None, judge_config: dict[str, Any] | None = None, quota_config: dict[str, Any] | None = None, bus=None) -> ProviderSummary:
    """执行单个 provider/model 的测试项并汇总结果。

    tests 参数为空时使用 provider 自身配置的测试项；每个测试函数返回一组
    TestResult。执行过程中会补充 usage、评价类型，并在连通失败或请求级故障时
    停止当前 provider 后续测试，避免不可用链路继续打延迟、质量等请求。
    """
    logger = logging.getLogger("tokenstar")
    results: list[TestResult] = []
    logger.info("开始测试中转: %s model=%s", provider.name, provider.model)
    for test_name in tests or provider.tests:
        test_func = TEST_REGISTRY.get(test_name)
        if not test_func:
            results.append(
                TestResult(
                    provider.name,
                    provider.model,
                    test_name,
                    "警告",
                    message=f"未知测试项: {test_name}",
                )
            )
            continue
        try:
            start_usage_collection()
            test_results = test_func(provider, thresholds)
            _enrich_usage(test_results)
            _inject_evaluation_type(test_name, test_results)
            results.extend(test_results)
            if _should_stop_after_test(test_name, test_results):
                logger.warning("检测到连通失败或请求失败，停止后续测试: provider=%s test=%s", provider.name, test_name)
                if bus:
                    _emit_request_failed(bus, provider, test_results, test_name)
                break
        except Exception as exc:
            logger.exception("测试项执行失败: provider=%s test=%s", provider.name, test_name)
            results.append(
                TestResult(
                    provider.name,
                    provider.model,
                    test_name,
                    "失败",
                    message=str(exc),
                )
            )
            # 不 break，跳过阻塞项继续跑后续测试

    # Judge 复核（在汇总前执行，可修改 results）
    if judge_config and judge_config.get("enabled"):
        judge_provider = judge_config.get("_provider")
        if judge_provider:
            try:
                results = apply_judge(results, judge_config, judge_provider)
            except Exception:
                logger.warning("Judge 复核执行失败，跳过", exc_info=True)

    return summarize_provider(provider, results, pricing_config, quota_config)


def _should_stop_after_test(test_name: str, results: list[TestResult]) -> bool:
    """判断当前测试项完成后是否停止该 provider 的后续测试。

    daily 的第一项是 connectivity。只要连通性没有成功，就直接钉死停止，
    不再进入 latency；其他测试项仍沿用请求级故障短路规则。
    """
    if test_name == "connectivity" and any(result.status not in {"成功", "pass"} for result in results):
        return True
    return _has_request_failure(results)


def _has_request_failure(results: list[TestResult]) -> bool:
    """检查一组结果中是否包含需要短路的请求级故障。"""
    for result in results:
        if _find_request_failure_text(result):
            return True
    return False


def _find_request_failure_text(result: TestResult) -> str:
    """提取 TestResult 中第一个请求级故障文本。"""
    if _message_has_request_failure(result.message):
        return result.message or ""
    return _find_detail_failure_text(result.detail)


def _find_detail_failure_text(value) -> str:
    """递归检查 detail 中显式错误字段，避免误扫模型回复正文。"""
    if isinstance(value, dict):
        for key in ("error", "error_message", "error_msg", "proxy_error", "first_error", "second_error"):
            err_val = value.get(key)
            if isinstance(err_val, str) and _message_has_request_failure(err_val):
                return err_val
        for sub in value.values():
            found = _find_detail_failure_text(sub)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_detail_failure_text(item)
            if found:
                return found
    return ""


def _detail_error_has_failure(value) -> bool:
    """兼容测试和调用方：判断 detail 中是否存在请求级故障。"""
    return bool(_find_detail_failure_text(value))


def _emit_request_failed(bus, provider, test_results: list, test_name: str) -> None:
    """向 EventBus 发布 REQUEST_FAILED 事件。

    只对能提取出请求级故障文本的失败结果发事件；普通连通失败会停止后续测试，
    但不会伪造成 REQUEST_FAILED 告警。
    """
    from app.events.types import REQUEST_FAILED, ProgressEvent

    for r in test_results:
        if r.status not in {"失败", "fail"}:
            continue
        error_msg = _find_request_failure_text(r)
        if not error_msg:
            continue
        bus.publish(ProgressEvent(
            kind=REQUEST_FAILED,
            payload={
                "provider": provider.name,
                "model": provider.model,
                "group_name": provider.group,
                "error_kind": _extract_error_kind(error_msg),
                "error_msg": error_msg,
                "test_name": test_name,
            },
        ))


def _extract_error_kind(message: str) -> str:
    """从错误消息中提取错误类型标记。"""
    for tag in REQUEST_FAILURE_PATTERNS:
        if tag in message:
            return tag.strip("[]")
    for kw in REQUEST_FAILURE_KEYWORDS:
        if kw in message:
            return "server"
    return "unknown"


def _message_has_request_failure(message: str) -> bool:
    """判断错误文本是否命中请求级故障标记或关键字。"""
    if not message:
        return False
    if any(tag in message for tag in REQUEST_FAILURE_PATTERNS):
        return True
    return any(kw in message for kw in REQUEST_FAILURE_KEYWORDS)


def _enrich_usage(results: list[TestResult]) -> None:
    """将收集器中的 usage 记录分发到各 TestResult.detail。

    保留完整的 usage_records 数组，同时生成合并后的 usage 供统计使用。
    """
    records = drain_usage_records()
    if not records:
        return

    _SUM_FIELDS = (
        "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens",
        "total_tokens", "cached_tokens", "cache_read_tokens", "cache_creation_tokens",
    )

    # 保存原始请求级 usage 记录
    if results:
        results[0].detail["usage_records"] = list(records)

    for r in results:
        if not r.detail.get("usage") and records:
            r.detail["usage"] = records.pop(0)
    # 剩余的记录（多轮调用等）追加到最后一个结果
    while records:
        if results:
            existing = dict(results[-1].detail.get("usage") or {})
            extra = records.pop(0)
            for field in _SUM_FIELDS:
                if field in existing or field in extra:
                    existing[field] = existing.get(field, 0) + extra.get(field, 0)
            for key, value in extra.items():
                if key not in _SUM_FIELDS and key not in existing:
                    existing[key] = value
            results[-1].detail["usage"] = existing


def _inject_evaluation_type(test_name: str, results: list[TestResult]) -> None:
    """根据 EVALUATION_TYPES 映射自动注入评价类型字段。

    优先按结果自身的 test_name 查找类型映射，
    找不到时回退到父测试名，保证聚合入口（如 political_sensitivity）
    返回的子结果各自保留正确的 INFO/CRITICAL/SOFT 类型。
    """
    for r in results:
        lookup_name = r.test_name if r.test_name in EVALUATION_TYPES else test_name
        mapping = EVALUATION_TYPES.get(lookup_name)
        if mapping is None:
            continue
        r.evaluation_type, r.counts_toward_pass_rate = mapping


def _summarize_eval(results: list[TestResult]) -> dict:
    """评价类型分组统计。"""
    scorable = [r for r in results if r.counts_toward_pass_rate]
    hard_results = [r for r in scorable if r.evaluation_type == "HARD"]
    soft_results = [r for r in scorable if r.evaluation_type == "SOFT"]
    critical_results = [r for r in scorable if r.evaluation_type == "CRITICAL"]
    mixed_results = [r for r in scorable if r.evaluation_type == "MIXED"]
    info_results = [r for r in results if r.evaluation_type == "INFO"]

    soft_scores = [r.score for r in soft_results if r.score is not None]
    critical_failed = len([r for r in critical_results if r.status in {"失败", "fail"}])
    critical_failed += sum(1 for r in mixed_results if (r.detail or {}).get("critical_fail"))

    return {
        "scorable_total": len(scorable),
        "scorable_passed": len([r for r in scorable if r.status in {"成功", "pass"}]),
        "scorable_failed": len([r for r in scorable if r.status in {"失败", "fail"}]),
        "scorable_warned": len([r for r in scorable if r.status in {"警告", "warn"}]),
        "hard_total": len(hard_results),
        "hard_passed": len([r for r in hard_results if r.status in {"成功", "pass"}]),
        "soft_avg_score": round(sum(soft_scores) / len(soft_scores), 1) if soft_scores else None,
        "critical_failed": critical_failed,
        "info_count": len(info_results),
    }


def _summarize_usage(results: list[TestResult], pricing_config: dict[str, Any] | None, provider_name: str, model: str) -> dict:
    """聚合 token 用量与成本。"""
    req_count = 0
    total_input = 0
    total_output = 0
    total_cached = 0
    for r in results:
        d = r.detail or {}
        usage = d.get("usage") or d
        if usage.get("input_tokens") or usage.get("prompt_tokens"):
            req_count += 1
        total_input += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        total_output += usage.get("output_tokens") or usage.get("completion_tokens") or 0
        total_cached += (
            usage.get("cached_tokens") or usage.get("cache_read_tokens")
            or usage.get("cache_read_input_tokens") or 0
        )

    price = lookup_price(pricing_config, provider_name, model)
    currency = (pricing_config or {}).get("currency", "CNY")
    usage_summary = {
        "input_tokens": total_input, "output_tokens": total_output,
        "cached_tokens": total_cached, "cache_read_tokens": total_cached,
        "cache_creation_tokens": 0, "total_tokens": total_input + total_output,
    }
    return {
        "request_count": req_count,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cached_tokens": total_cached,
        "estimated_cost": estimate_cost(usage_summary, price),
        "cost_currency": currency,
        "price_matched": price.get("matched", False),
    }


def _summarize_quota(quota_config: dict[str, Any] | None, provider_name: str) -> dict:
    """读取 provider 级 quota 配置，返回展示字段。"""
    if not quota_config:
        return {}
    providers_quota = quota_config.get("providers") or {}
    entry = providers_quota.get(provider_name) or {}
    balance = entry.get("balance")
    currency = entry.get("currency", "")
    if balance is None:
        return {}
    return {
        "quota_balance": float(balance),
        "quota_currency": str(currency),
        "estimated_remaining": float(balance),  # 第一版 manual 类型，剩余=余额
    }


def summarize_provider(provider: ProviderConfig, results: list[TestResult], pricing_config: dict[str, Any] | None = None, quota_config: dict[str, Any] | None = None) -> ProviderSummary:
    latencies = [item.latency_ms for item in results if item.latency_ms is not None]
    total = len(results)
    passed = len([item for item in results if item.status in {"成功", "pass"}])
    failed = len([item for item in results if item.status in {"失败", "fail"}])
    warned = len([item for item in results if item.status in {"警告", "warn"}])

    latency_results = [item for item in results if item.test_name in {"daily_latency", "latency"}]
    latency_detail = latency_results[-1].detail if latency_results else {}

    eval_stats = _summarize_eval(results)
    usage_stats = _summarize_usage(results, pricing_config, provider.group or provider.name, provider.model)
    quota_stats = _summarize_quota(quota_config, provider.group or provider.name)

    if latency_detail:
        avg_latency_ms = latency_detail.get("avg_latency_ms")
        p95_latency_ms = latency_detail.get("p95_latency_ms")
    elif latencies:
        avg_latency_ms = round(statistics.mean(latencies), 2)
        p95_latency_ms = percentile(latencies, 95)
    else:
        avg_latency_ms = None
        p95_latency_ms = None

    return ProviderSummary(
        provider=provider.name,
        model=provider.model,
        total=total, passed=passed, failed=failed, warned=warned,
        avg_latency_ms=avg_latency_ms,
        p95_latency_ms=p95_latency_ms,
        results=results,
        group=provider.group,
        **eval_stats,
        **usage_stats,
        **quota_stats,
    )
