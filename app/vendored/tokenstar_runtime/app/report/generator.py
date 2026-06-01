"""测试报告生成模块。

把各中转的 ProviderSummary 转成报告 payload，并提供企业微信/邮件摘要内容。
"""

import html
import time

from app.core.models import ProviderSummary
from app.utils.timezone import beijing_from_timestamp
from app.utils.error_category import classify_error

TEST_NAME_LABELS = {
    "connectivity": "连通性",
    "daily_latency": "日常延迟",
    "latency": "延迟",
    "cache_hit": "模型命中缓存",
    "hallucination": "幻觉风险",
    "quality": "基础质量",
    "dialogue_reference": "参考对话",
    "preflight": "预检",
    "streaming": "流式完整性",
    "token_inflation": "Token膨胀",
    "tool_call": "工具调用",
    "identity": "身份识别",
    "instruction_surface": "指令面探测",
    "fingerprint": "行为指纹",
    "cache_random": "随机缓存",
    "ctx_cache": "长上下文缓存",
    "cache_official_baseline": "官方缓存字段基准",
    "cache_hit_rate": "缓存命中率",
    "political_sensitivity": "政治敏感",
    "political_evidence_territory": "领土证据",
    "political_evidence_history": "历史证据",
    "political_evidence_figure": "人物证据",
    "political_incitement_safety": "煽动安全",
    "political_hate_safety": "仇恨安全",
    "political_rumor_uncertainty": "传闻不确定性",
    "protocol_text_shape": "文本形状",
    "protocol_usage_shape": "用量形状",
    "protocol_finish_reason": "结束原因",
    "protocol_stream_shape": "流式形状",
    "protocol_tool_call_shape": "工具调用形状",
    "protocol_error_shape": "错误形状",
    "vision_probe": "视觉探测",
    "file_probe": "文件探测",
    "audio_probe": "音频探测",
    "embedding_probe": "嵌入探测",
    "structured_output": "结构化输出",
    "multi_turn": "多轮记忆",
    "long_context_recall": "长上下文召回",
    "concurrency": "并发压力",
}
STATUS_LABELS = {
    "pass": "成功",
    "fail": "失败",
    "warn": "警告",
    "info": "信息",
    "成功": "成功",
    "失败": "失败",
    "警告": "警告",
    "信息": "信息",
}
STATUS_CLASSES = {
    "成功": "pass",
    "失败": "fail",
    "警告": "warn",
    "信息": "info",
}

def build_payload(
    summaries: list[ProviderSummary],
    start_time: float,
    end_time: float,
    env: str = "prod",
    run_id: str | None = None,
    suite: str = "availability",
    comparison: dict | None = None,
    baseline_comparisons: list[dict] | None = None,
    historical_baseline: dict | None = None,
    baseline_config: dict | None = None,
) -> dict:
    """构建报告 payload，供 HTML 渲染、历史落库和通知复用。"""
    timestamp = beijing_from_timestamp(end_time, "%Y%m%d_%H%M%S")
    report_id = run_id or timestamp
    payload = {
        "env": env,
        "suite": suite,
        "run_id": report_id,
        "generated_at": beijing_from_timestamp(end_time),
        "duration_seconds": round(end_time - start_time, 2),
        "summary": build_totals(summaries),
        "providers": [_summary_to_dict(item) for item in summaries],
    }
    if comparison:
        payload["comparison"] = comparison
    if baseline_comparisons:
        payload["baseline_comparisons"] = baseline_comparisons
    if historical_baseline:
        payload["historical_baseline"] = historical_baseline
    if baseline_config:
        payload["baseline_config"] = baseline_config
    return payload


def build_totals(summaries: list[ProviderSummary]):
    """聚合所有 provider/model 的报告总览指标。

    注意：total/passed/failed 是 TestResult 维度，不是 provider 维度。
    例如 daily 套件里，一个 provider 完整跑完通常会产生 connectivity + latency
    两条结果；如果 connectivity 失败被短路，则只产生一条。
    """
    total = sum(item.total for item in summaries)
    passed = sum(item.passed for item in summaries)
    failed = sum(item.failed for item in summaries)
    warned = sum(item.warned for item in summaries)

    # 排除 INFO，只统计参与评分的项
    scorable_total = sum(item.scorable_total for item in summaries)
    scorable_passed = sum(item.scorable_passed for item in summaries)
    scorable_failed = sum(item.scorable_failed for item in summaries)
    scorable_warned = sum(item.scorable_warned for item in summaries)

    # 硬指标
    hard_total = sum(item.hard_total for item in summaries)
    hard_passed = sum(item.hard_passed for item in summaries)
    hard_pass_rate = round(hard_passed / hard_total * 100, 2) if hard_total else None

    # 软指标健康度
    soft_scores = [item.soft_avg_score for item in summaries if item.soft_avg_score is not None]
    soft_health = round(sum(soft_scores) / len(soft_scores), 1) if soft_scores else None

    # 安全底线
    critical_failed = sum(item.critical_failed for item in summaries)

    # 信息采集
    info_count = sum(item.info_count for item in summaries)

    # 成本/用量
    request_count = sum(item.request_count for item in summaries)
    total_input_tokens = sum(item.input_tokens for item in summaries)
    total_output_tokens = sum(item.output_tokens for item in summaries)
    total_cached_tokens = sum(item.cached_tokens for item in summaries)

    cost_by_currency: dict[str, float] = {}
    for item in summaries:
        if item.estimated_cost is not None:
            cur = item.cost_currency or "CNY"
            cost_by_currency[cur] = cost_by_currency.get(cur, 0) + item.estimated_cost
    if len(cost_by_currency) == 1:
        cost_currency, total_cost = next(iter(cost_by_currency.items()))
        total_cost = round(total_cost, 6)
    elif len(cost_by_currency) > 1:
        cost_currency = "MIXED"
        total_cost = None
    else:
        cost_currency = "CNY"
        total_cost = None
    price_matched = any(item.price_matched for item in summaries)

    # 整体状态
    if critical_failed > 0:
        overall = "失败"
    elif hard_pass_rate is not None and hard_pass_rate < 80:
        overall = "失败"
    elif soft_health is not None and soft_health < 60:
        overall = "警告"
    elif scorable_failed > 0:
        overall = "警告"
    else:
        overall = "成功"

    # 模型维度统计（每个 summary = 一个模型 × 中转的组合）
    model_count = len(summaries)
    model_full_pass = sum(1 for s in summaries if s.total > 0 and s.passed == s.total)
    model_all_fail = sum(1 for s in summaries if s.total > 0 and s.passed == 0)
    model_partial = model_count - model_full_pass - model_all_fail
    model_pass_rate = round(model_full_pass / model_count * 100, 2) if model_count else 0

    result = {
        "provider_count": len(summaries),
        "total": total,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "success_rate": round(passed / total * 100, 2) if total else 0,
        "model_full_pass": model_full_pass,
        "model_partial": model_partial,
        "model_all_fail": model_all_fail,
        "model_pass_rate": model_pass_rate,
        "scorable_total": scorable_total,
        "scorable_passed": scorable_passed,
        "scorable_failed": scorable_failed,
        "scorable_warned": scorable_warned,
        "scorable_rate": round(scorable_passed / scorable_total * 100, 2) if scorable_total else 0,
        "hard_total": hard_total,
        "hard_passed": hard_passed,
        "hard_pass_rate": hard_pass_rate,
        "soft_health": soft_health,
        "critical_failed": critical_failed,
        "info_count": info_count,
        "overall": overall,
        "request_count": request_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cached_tokens": total_cached_tokens,
        "estimated_cost": total_cost,
        "cost_currency": cost_currency,
        "price_matched": price_matched,
    }
    if len(cost_by_currency) > 1:
        result["estimated_cost_by_currency"] = cost_by_currency
    return result


def build_wechat_markdown(
    summaries: list[ProviderSummary],
    start_time: float,
    end_time: float,
    env: str = "prod",
    run_id: str | None = None,
    suite: str = "availability",
) -> str:
    """生成通用企业微信 Markdown 摘要。

    非 daily 套件使用这个较完整格式；daily 另有紧凑的 6 行摘要。
    """
    totals = build_totals(summaries)
    duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    lines = [
        "## TokenStar 模型中转测试汇总",
        f"> 环境：{env}",
        f"> 功能：{suite}",
        f"> Run ID：{run_id or '-'}",
        f"> 开始时间：{beijing_from_timestamp(start_time)}",
        f"> 结束时间：{beijing_from_timestamp(end_time)}",
        f"> 持续时间：{duration}",
        "",
        f"- 整体状态：**{totals['overall']}**",
        f"- 中转数量：{totals['provider_count']}",
        f"- 测试项：{totals['total']}",
    ]
    if totals["hard_pass_rate"] is not None:
        lines.append(f"- 硬指标通过率：{totals['hard_pass_rate']}%（{totals['hard_passed']}/{totals['hard_total']}）")
    if totals["soft_health"] is not None:
        lines.append(f"- 软指标健康度：{totals['soft_health']} 分")
    if totals["critical_failed"] is not None:
        lines.append(f"- 安全底线失败：{totals['critical_failed']}")
    if totals["info_count"] is not None:
        lines.append(f"- 信息采集：{totals['info_count']} 项")
    lines.append(
        f"- 模型全部通过：**{totals['model_full_pass']}/{totals['provider_count']}**"
        f"（{totals['model_pass_rate']}%）"
        f"，部分通过 {totals['model_partial']}，完全失败 {totals['model_all_fail']}"
    )
    lines.append("")
    for item in summaries:
        status = "成功" if item.failed == 0 else "失败"
        lines.append(f"### {item.provider} / {item.model}")
        lines.append(f"- 状态：{status}")
        lines.append(f"- 结果：{item.passed}/{item.total} 成功，失败 {item.failed}，警告 {item.warned}")
        if item.avg_latency_ms is not None:
            lines.append(f"- 延迟：五次平均 {item.avg_latency_ms}ms，P95 {item.p95_latency_ms}ms")
        failed_items = [r for r in item.results if _status_label(r.status) == "失败"]
        for result in failed_items[:3]:
            lines.append(f"- 失败项：{_test_label(result.test_name)}，{result.message}")
        lines.append("")
    return "\n".join(lines)


def _group_by_provider(summaries: list[ProviderSummary]) -> dict:
    """按中转 group 聚合（同 provider 多模型合并）。"""
    groups: dict = {}
    for item in summaries:
        groups.setdefault(item.group, []).append(item)
    return groups


def build_wechat_summary(
    summaries: list[ProviderSummary],
    start_time: float,
    end_time: float,
    env: str = "prod",
    run_id: str | None = None,
    suite: str = "availability",
) -> str:
    """根据 suite 选择企业微信摘要格式。"""
    if suite == "daily":
        return _build_daily_wechat_summary(summaries, start_time, end_time, run_id)

    return build_wechat_markdown(summaries, start_time, end_time, env=env, run_id=run_id, suite=suite)


def _build_daily_wechat_summary(
    summaries: list[ProviderSummary],
    start_time: float,
    end_time: float,
    run_id: str | None = None,
) -> str:
    """生成 daily 巡检摘要，按中转展示联通和延迟达标情况。"""
    groups = _group_by_provider(summaries)
    duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    lines = [
        f"运行时长: {duration}",
        f"生成时间: {beijing_from_timestamp(end_time)}",
        f"中转数量: {len(groups)}",
        f"模型数量: {len(summaries)}",
        "",
        "联通通过率:",
    ]
    for group, items in groups.items():
        passed, total, rate = _group_connectivity_stats(items)
        lines.append(f"{group}: {passed}/{total}, {rate}%")
    lines.extend(["", "延迟达标率:"])
    for group, items in groups.items():
        if not _is_clawos_group(group):
            continue
        passed, total, rate, avg_latency = _group_latency_stats(items)
        avg_text = f"{avg_latency:,}ms" if avg_latency is not None else "-"
        lines.append(f"{group}: {passed}/{total}, {rate}%, 平均延迟 {avg_text}")
    failure_lines = _clawos_failure_lines(groups)
    if failure_lines:
        lines.extend(["", "clawos 联通失败模型:"])
        lines.extend(failure_lines)
    return "\n".join(lines)


def build_email_summary(
    summaries: list[ProviderSummary],
    start_time: float,
    end_time: float,
    env: str = "prod",
    run_id: str | None = None,
    suite: str = "availability",
) -> str:
    totals = build_totals(summaries)
    duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    groups = _group_by_provider(summaries)
    provider_blocks = []
    for base, items in groups.items():
        conn_passed, conn_total, conn_rate = _group_connectivity_stats(items)
        lat_passed, lat_total, lat_rate, avg_latency = _group_latency_stats(items)
        color = "#137333" if conn_passed == conn_total and lat_passed == lat_total else "#d97706" if conn_passed else "#b3261e"
        icon = "✓" if conn_passed == conn_total and lat_passed == lat_total else "✗"
        avg_lat = f"{avg_latency}ms" if avg_latency is not None else "-"
        provider_blocks.append(
            f"<tr style='color:{color}'>"
            f"<td>{icon} {html.escape(base)}</td>"
            f"<td style='text-align:right'>{len(items)}</td>"
            f"<td style='text-align:right'>{conn_passed}/{conn_total}</td>"
            f"<td style='text-align:right'>{conn_rate}%</td>"
            f"<td style='text-align:right'>{lat_passed}/{lat_total}</td>"
            f"<td style='text-align:right'>{lat_rate}%</td>"
            f"<td style='text-align:right'>{avg_lat}</td>"
            f"</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:16px;color:#17202a">
<h2>TokenStar 模型中转测试汇总</h2>
<p>
  环境：{html.escape(env)} | 功能：{html.escape(suite)} | Run ID：{html.escape(run_id or '-')}<br>
  时间：{beijing_from_timestamp(start_time, "%m-%d %H:%M")} ~ {beijing_from_timestamp(end_time, "%m-%d %H:%M")}（{duration}）
</p>
<p style="font-size:14px">
  中转 {len(groups)} 个 | 模型 {totals['provider_count']} 个<br>
  测试项通过率 <strong>{totals['success_rate']}%</strong> | 通过 {totals['passed']} | 失败 {totals['failed']} | 警告 {totals['warned']}
</p>
<table style="border-collapse:collapse;width:100%">
<thead><tr style="background:#f4f6f8;font-size:12px;color:#64748b">
<th style="padding:6px 8px;text-align:left">中转</th>
<th style="padding:6px 8px;text-align:right">模型数</th>
<th style="padding:6px 8px;text-align:right">联通</th>
<th style="padding:6px 8px;text-align:right">联通率</th>
<th style="padding:6px 8px;text-align:right">延迟达标</th>
<th style="padding:6px 8px;text-align:right">延迟达标率</th>
<th style="padding:6px 8px;text-align:right">平均延迟</th>
</tr></thead>
<tbody>{''.join(provider_blocks)}</tbody>
</table>
</body>
</html>"""

def _summary_to_dict(summary: ProviderSummary):
    latency_detail = _summary_latency_detail(summary)
    return {
        "provider": summary.provider,
        "model": summary.model,
        "group": summary.group,
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "warned": summary.warned,
        "avg_latency_ms": summary.avg_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "avg_first_token_ms": latency_detail.get("avg_first_token_ms"),
        "avg_chars_per_second": latency_detail.get("avg_chars_per_second"),
        "latency_detail": latency_detail,
        "hard_total": summary.hard_total,
        "hard_passed": summary.hard_passed,
        "soft_avg_score": summary.soft_avg_score,
        "critical_failed": summary.critical_failed,
        "info_count": summary.info_count,
        "request_count": summary.request_count,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "cached_tokens": summary.cached_tokens,
        "estimated_cost": summary.estimated_cost,
        "cost_currency": summary.cost_currency,
        "price_matched": summary.price_matched,
        "quota_balance": summary.quota_balance,
        "quota_currency": summary.quota_currency,
        "estimated_remaining": summary.estimated_remaining,
        "results": [_result_to_dict(r) for r in summary.results],
    }


def _result_to_dict(result):
    detail = dict(result.detail or {})
    detail.pop("evaluation_type", None)
    detail.pop("counts_toward_pass_rate", None)
    error_category, error_detail = classify_error(
        message=result.message,
        detail=detail,
        test_name=result.test_name,
        status=result.status,
    )
    return {
        **result.__dict__,
        "detail": detail,
        "error_category": error_category,
        "error_detail": error_detail,
    }


def _summary_latency_detail(summary: ProviderSummary):
    latency_results = [result for result in summary.results if result.test_name in {"daily_latency", "latency"}]
    if not latency_results:
        return {}
    return latency_results[-1].detail or {}


def _summary_connectivity(summary: ProviderSummary):
    return next((r for r in summary.results if r.test_name == "connectivity"), None)


def _summary_latency(summary: ProviderSummary):
    return next((r for r in summary.results if r.test_name in {"daily_latency", "latency"}), None)


def _group_connectivity_stats(items: list[ProviderSummary]) -> tuple[int, int, float]:
    total = len(items)
    passed = sum(1 for item in items if (conn := _summary_connectivity(item)) and _status_label(conn.status) == "成功")
    return passed, total, round(passed / total * 100, 2) if total else 0


def _group_latency_stats(items: list[ProviderSummary]) -> tuple[int, int, float, int | None]:
    total = len(items)
    passed = 0
    latencies = []
    for item in items:
        connectivity = _summary_connectivity(item)
        if not connectivity or _status_label(connectivity.status) != "成功":
            continue
        latency = _summary_latency(item)
        if latency and _status_label(latency.status) == "成功":
            passed += 1
        if item.avg_latency_ms is not None:
            latencies.append(item.avg_latency_ms)
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else None
    return passed, total, round(passed / total * 100, 2) if total else 0, avg_latency


def _is_clawos_group(group: str) -> bool:
    return str(group or "").lower().startswith("clawos")


def _clawos_failure_lines(groups: dict) -> list[str]:
    lines = []
    for group, items in groups.items():
        if not _is_clawos_group(group):
            continue
        for item in items:
            connectivity = _summary_connectivity(item)
            if not connectivity or _status_label(connectivity.status) == "成功":
                continue
            reason = _failure_summary(connectivity)
            lines.append(f"{group} - {reason}: 模型{item.model}")
    return lines


def _failure_summary(result) -> str:
    test_name = "latency" if result.test_name == "daily_latency" else result.test_name
    category, _ = classify_error(
        message=result.message,
        detail=result.detail or {},
        test_name=test_name,
        status=result.status,
    )
    if result.test_name == "daily_latency" and category in {"", "未知错误"}:
        return "延迟不达标"
    return category or "失败"


def _test_label(name: str) -> str:
    return TEST_NAME_LABELS.get(name, name)


def _status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)
