"""日常可用性监控测试（方向2）。"""
import statistics
import time

from app.core.llm_client import LLMClient, extract_text
from app.core.models import ProviderConfig, TestResult
from app.utils.error_category import classify_error
from app.tests.helpers import (
    STATUS_FAIL,
    STATUS_PASS,
    _check_refuse,
    _check_uncertainty,
    _dialogue_check_ok,
    _extract_cache_usage,
    _score_slowness,
    _status_by_score,
)
from app.utils.percentile import percentile
from app.utils.timezone import beijing_now_str


FIRST_TOKEN_CONNECTIVITY_PROMPT = "hi, 回复我 hello"


def run_connectivity(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """API 联通测试。

    最多做 3 次非流式 chat 请求，默认要求 3/3 成功才通过。
    daily 套件会先跑它；如果结果失败，runner 会停止该 provider 后续 latency。
    """
    client = LLMClient(provider)
    details = []
    latencies = []
    passed = 0
    for index in range(3):
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": FIRST_TOKEN_CONNECTIVITY_PROMPT}],
            max_tokens=64,
            test_name="connectivity",
        )
        text = extract_text(provider.format, response)
        ok = bool(response) and not error
        passed += 1 if ok else 0
        latencies.append(latency_ms)
        details.append(
            {
                "index": index + 1,
                "ok": ok,
                "latency_ms": latency_ms,
                "error": error,
                "response_text": text[:500],
            }
        )
        if error:
            break
    success_rate = passed / 3
    min_rate = float(thresholds.get("connectivity_success_rate", 1.0))
    score = round(success_rate * 100, 2)
    status = STATUS_PASS if success_rate >= min_rate else STATUS_FAIL
    avg_latency = int(statistics.mean(latencies)) if latencies else None
    message = f"连通 {passed}/3，平均延迟={avg_latency}ms"
    if passed < 3 and details:
        first_err = next((d["error"] for d in details if d.get("error")), None)
        if first_err:
            message += f"，{first_err[:150]}"
    return [
        TestResult(
            provider.name,
            provider.model,
            "connectivity",
            status,
            latency_ms=avg_latency,
            score=score,
            message=message,
            detail={"attempts": details, "avg_latency_ms": avg_latency, "min_rate": min_rate},
        )
    ]


def run_first_token_connectivity(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """流式连通探测：收到首个有效 token 即视为连通成功，并记录 TTFT。"""
    client = LLMClient(provider)
    started_at = beijing_now_str()
    text = ""
    latency_ms = 0
    first_token_ms = None
    chars_per_second = 0.0
    error = ""
    try:
        text, latency_ms, first_token_ms, chars_per_second, error = client.chat_stream_metrics(
            [{"role": "user", "content": FIRST_TOKEN_CONNECTIVITY_PROMPT}],
            max_tokens=int(thresholds.get("first_token_connectivity_max_tokens", 64)),
            temperature=0,
            test_name="first_token_connectivity",
        )
    finally:
        client.session.close()
    finished_at = beijing_now_str()
    ok = bool(first_token_ms is not None and text and not error)
    status = STATUS_PASS if ok else STATUS_FAIL
    message = f"首Token连通成功，TTFT={first_token_ms}ms，总耗时={latency_ms}ms" if ok else "首Token连通失败"
    if error:
        message += f"，{error[:150]}"
    elif not text:
        message += "，流式响应为空"
        error = "流式响应为空"
    category, detail_text = classify_error(
        message=error,
        test_name="first_token_connectivity",
        status=status,
    )
    detail = {
        "ok": ok,
        "started_at": started_at,
        "finished_at": finished_at,
        "first_token_ms": first_token_ms,
        "latency_ms": latency_ms,
        "chars_per_second": chars_per_second,
        "char_count": len(text),
        "response_preview": text[:200],
        "error_category": category,
        "error_detail": detail_text,
    }
    return [
        TestResult(
            provider.name,
            provider.model,
            "first_token_connectivity",
            status,
            latency_ms=latency_ms,
            score=100.0 if ok else 0.0,
            message=message,
            detail=detail,
            evaluation_type="HARD",
        )
    ]


def run_daily_latency(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """轻量日常延迟测试。

    daily 套件使用它替代原 latency 的 10 次流式采样：默认做 3 次流式请求，
    用平均延迟、P95 和首 Token 写入报告，减少定时巡检的请求量和运行时间。
    """
    client = LLMClient(provider)
    prompt = thresholds.get("daily_latency_prompt", "请用中文写一段 30 字左右的模型响应速度测试文本。")
    count = int(thresholds.get("daily_latency_requests", 3))
    limit = float(thresholds.get("daily_latency_avg_ms", thresholds.get("latency_p95_ms", 10000)))
    latencies = []
    first_token_times = []
    chars_per_second_values = []
    errors = []
    details = []

    for index in range(count):
        text, latency_ms, first_token_ms, chars_per_second, error = client.chat_stream_metrics(
            [{"role": "user", "content": prompt}],
            max_tokens=int(thresholds.get("daily_latency_max_tokens", 64)),
            temperature=0,
            test_name="daily_latency",
        )
        ok = bool(text) and not error
        latencies.append(latency_ms)
        if first_token_ms is None:
            first_token_ms = latency_ms
        first_token_times.append(first_token_ms)
        if chars_per_second:
            chars_per_second_values.append(chars_per_second)
        if error:
            errors.append(error)
        details.append(
            {
                "index": index + 1,
                "ok": ok,
                "latency_ms": latency_ms,
                "first_token_ms": first_token_ms,
                "chars_per_second": chars_per_second,
                "char_count": len(text),
                "error": error,
                "response_text": text[:500],
            }
        )
        if error:
            break
        time.sleep(0.2)

    avg_latency = statistics.mean(latencies) if latencies else 0
    p95 = percentile(latencies, 95) if latencies else 0
    avg_first_token_ms = statistics.mean(first_token_times) if first_token_times else None
    avg_chars_per_second = statistics.mean(chars_per_second_values) if chars_per_second_values else 0.0
    score = 0.0 if errors else round(max(0.0, min(100.0, (1 - avg_latency / max(limit, 1)) * 100)), 2)
    status = STATUS_PASS if not errors and avg_latency <= limit else STATUS_FAIL
    first_token_text = f"首Token={int(avg_first_token_ms)}ms" if avg_first_token_ms is not None else "首Token=未采集"
    message = f"{len(latencies)}次平均={int(avg_latency)}ms, P95={int(p95)}ms, {first_token_text}, 阈值={int(limit)}ms"
    if errors:
        message += f"，{errors[0][:150]}"

    return [
        TestResult(
            provider.name,
            provider.model,
            "daily_latency",
            status,
            latency_ms=int(p95),
            score=score,
            message=message,
            detail={
                "latencies_ms": latencies,
                "avg_latency_ms": round(avg_latency, 2),
                "p95_latency_ms": int(p95),
                "first_token_times_ms": first_token_times,
                "avg_first_token_ms": round(avg_first_token_ms, 2) if avg_first_token_ms is not None else None,
                "chars_per_second_values": chars_per_second_values,
                "avg_chars_per_second": round(avg_chars_per_second, 2),
                "limit_ms": int(limit),
                "attempts": details,
                "errors": errors,
            },
        )
    ]


def run_latency(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """延迟测试。

    默认做 10 次流式请求，采集总耗时、首 token、吐字速度，并计算 P50/P95。
    若某次请求报错会提前停止；若流式没有报错但空输出，会降级用非流式补采样。
    """
    client = LLMClient(provider)
    latencies = []
    first_token_times = []
    chars_per_second_values = []
    errors = []
    details = []
    count = int(thresholds.get("latency_requests", 10))
    for index in range(count):
        text, latency_ms, first_token_ms, chars_per_second, error = client.chat_stream_metrics(
            [{"role": "user", "content": "请用中文写一段 80 字左右的模型响应速度测试文本。"}],
            max_tokens=160,
            temperature=0,
            test_name="latency",
        )
        fallback_used = False
        first_token_source = "stream" if first_token_ms is not None else ""
        if not error and not text:
            response, latency_ms, fallback_error = client.chat(
                [{"role": "user", "content": "请用中文写一段 80 字左右的模型响应速度测试文本。"}],
                max_tokens=160,
                temperature=0,
                test_name="latency_non_stream_fallback",
            )
            fallback_text = extract_text(provider.format, response)
            if fallback_text:
                text = fallback_text
                first_token_ms = latency_ms
                chars_per_second = 0.0
                fallback_used = True
                first_token_source = "non_stream_fallback"
                error = ""
            elif fallback_error:
                error = fallback_error
        if first_token_ms is None:
            first_token_ms = latency_ms
            first_token_source = "error_latency_fallback" if error else "latency_fallback"
        latencies.append(latency_ms)
        if first_token_ms is not None:
            first_token_times.append(first_token_ms)
        if chars_per_second:
            chars_per_second_values.append(chars_per_second)
        if error:
            errors.append(error)
        details.append(
            {
                "index": index + 1,
                "latency_ms": latency_ms,
                "first_token_ms": first_token_ms,
                "chars_per_second": chars_per_second,
                "char_count": len(text),
                "error": error,
                "response_text": text[:500],
                "fallback_non_stream": fallback_used,
                "first_token_source": first_token_source,
            }
        )
        if error:
            break
        time.sleep(0.2)

    # 基础统计
    p50 = percentile(latencies, 50) if latencies else 0
    p95 = percentile(latencies, 95) if latencies else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    avg_first_token_ms = statistics.mean(first_token_times) if first_token_times else None
    avg_chars_per_second = statistics.mean(chars_per_second_values) if chars_per_second_values else 0.0

    # 冷启动检测：第 1 次 vs 后 4 次的 TTFT 差值
    cold_start_ratio = None
    cold_start_signal = "无"
    if len(first_token_times) >= 3:
        first_ttft = first_token_times[0]
        rest_avg = statistics.mean(first_token_times[1:]) if len(first_token_times) > 1 else first_ttft
        cold_start_threshold = float(thresholds.get("cold_start_ratio_threshold", 1.5))
        cold_start_severe = float(thresholds.get("cold_start_severe_ratio", 3.0))
        if rest_avg > 0 and first_ttft > rest_avg * cold_start_threshold:
            cold_start_ratio = round(first_ttft / rest_avg, 2)
            cold_start_signal = "疑似冷启动" if cold_start_ratio < cold_start_severe else "明显冷启动"

    # 慢速归因：网络抖动 vs 模型计算慢
    slowness = _score_slowness(latencies, first_token_times, chars_per_second_values)

    # 评分：综合 P95 和错误率
    limit = float(thresholds.get("latency_p95_ms", 10000))
    score = 0.0 if errors else round(max(0.0, min(100.0, (1 - p95 / max(limit, 1)) * 100)), 2)
    if slowness["classification"] in ("network_jitter_likely", "model_compute_likely", "mixed"):
        score = max(0.0, score - 15)

    actual_count = len(latencies)
    first_token_text = f"首Token={int(avg_first_token_ms)}ms" if avg_first_token_ms is not None else "首Token=未采集"
    msg = (
        f"P50={int(p50)}ms, P95={int(p95)}ms, {actual_count}次平均={int(avg_latency)}ms, "
        f"{first_token_text}，吐字={round(avg_chars_per_second, 2)}字/秒"
    )
    if cold_start_signal != "无":
        msg += f"，{cold_start_signal}(x{cold_start_ratio})"
    if slowness["classification"] != "inconclusive":
        msg += f"，{slowness['classification']}"

    return [
        TestResult(
            provider.name,
            provider.model,
            "latency",
            _status_by_score(score),
            latency_ms=int(p95),
            score=score,
            message=msg,
            detail={
                "latencies_ms": latencies,
                "p50_latency_ms": int(p50),
                "avg_latency_ms": round(avg_latency, 2),
                "p95_latency_ms": int(p95),
                "first_token_times_ms": first_token_times,
                "avg_first_token_ms": round(avg_first_token_ms, 2) if avg_first_token_ms is not None else None,
                "chars_per_second_values": chars_per_second_values,
                "avg_chars_per_second": round(avg_chars_per_second, 2),
                "cold_start_ratio": cold_start_ratio,
                "cold_start_signal": cold_start_signal,
                "slowness": slowness,
                "attempts": details,
                "errors": errors,
            },
        )
    ]


def run_cache_hit(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """缓存命中探测。

    对固定 prompt 请求两次，综合判断第二次是否更快、输出是否一致，以及 usage
    中是否出现缓存命中字段。该测试是参考指标，不等同于严格缓存验收。
    """
    client = LLMClient(provider)
    # 两种探针类型：纯文本 + JSON 结构化
    probes = [
        {
            "type": "text",
            "prompt": "缓存命中测试：请只输出固定字符串 TOKENSTAR_CACHE_PROBE",
            "expected": "TOKENSTAR_CACHE_PROBE",
        },
        {
            "type": "json",
            "prompt": '缓存命中测试：请只输出JSON {"status":"ok","value":"TOKENSTAR_CACHE_PROBE"}，不要其他任何内容',
            "expected": "TOKENSTAR_CACHE_PROBE",
        },
    ]
    speedup_ratio = float(thresholds.get("cache_hit_speedup_ratio", 0.85))
    probe_results = []
    cache_hits = 0
    cache_field_hits = 0

    for probe in probes:
        first_response, first_latency, first_error = client.chat(
            [{"role": "user", "content": probe["prompt"]}],
            max_tokens=64,
            temperature=0,
            test_name="cache_hit",
        )
        if first_error:
            probe_results.append({
                "type": probe["type"],
                "prompt": probe["prompt"],
                "first_error": first_error,
                "first_latency_ms": first_latency,
                "same_output": False,
                "faster": False,
                "field_hit": False,
            })
            continue

        first_text = extract_text(provider.format, first_response)
        first_cache = _extract_cache_usage(provider.format, first_response)

        time.sleep(0.5)

        second_response, second_latency, second_error = client.chat(
            [{"role": "user", "content": probe["prompt"]}],
            max_tokens=64,
            temperature=0,
            test_name="cache_hit",
        )
        second_text = extract_text(provider.format, second_response) if not second_error else ""
        if second_error:
            probe_results.append({
                "type": probe["type"],
                "prompt": probe["prompt"],
                "first_latency_ms": first_latency,
                "first_response_text": first_text[:500],
                "second_error": second_error,
                "first_cache": first_cache,
                "same_output": False,
                "faster": False,
                "field_hit": False,
            })
            continue

        second_cache = _extract_cache_usage(provider.format, second_response)

        same_output = bool(first_text) and bool(second_text) and first_text.strip() == second_text.strip()
        faster = second_latency <= int(first_latency * speedup_ratio) if first_latency and second_latency else False

        # 缓存字段命中：Anthropic cache_read 或 OpenAI cached_tokens > 0
        field_hit = False
        if second_cache:
            if (second_cache.get("cache_read") or 0) > 0:
                field_hit = True
            if (second_cache.get("cached_tokens") or 0) > 0:
                field_hit = True

        if same_output and faster:
            cache_hits += 1
        if field_hit:
            cache_field_hits += 1

        probe_results.append({
            "type": probe["type"],
            "prompt": probe["prompt"],
            "first_latency_ms": first_latency,
            "second_latency_ms": second_latency,
            "first_error": first_error,
            "second_error": second_error,
            "same_output": same_output,
            "faster": faster,
            "speedup_ratio_threshold": speedup_ratio,
            "first_cache": first_cache,
            "second_cache": second_cache or {},
            "field_hit": field_hit,
            "first_response_text": first_text[:500],
            "second_response_text": second_text[:500],
        })

    # 综合评分：输出一致 + 加速 + 缓存字段
    if cache_hits == len(probes):
        score = 100.0
    elif cache_hits > 0:
        score = 75.0
    elif cache_field_hits > 0:
        score = 60.0
    else:
        all_same = all(r.get("same_output") for r in probe_results)
        score = 40.0 if all_same else 0.0

    valid_second = [r["second_latency_ms"] for r in probe_results if r["second_latency_ms"]]
    avg_second_latency = int(sum(valid_second) / len(valid_second)) if valid_second else None
    message = f"cache_hits={cache_hits}/{len(probes)}"
    if cache_field_hits > 0:
        message += f", 缓存字段命中={cache_field_hits}/{len(probes)}"

    return [
        TestResult(
            provider.name, provider.model, "cache_hit",
            _status_by_score(score),
            latency_ms=avg_second_latency,
            score=score,
            message=message,
            detail={"probes": probe_results},
        )
    ]


def run_hallucination(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """幻觉风险探测。

    用几类虚构实体/论文/数据问题观察模型是否拒答或表达不确定。
    这是启发式评分，不做事实数据库校验。
    """
    client = LLMClient(provider)
    probes = [
        # 虚构公司/产品
        {
            "category": "fictitious_entity",
            "prompt": "请回答：不存在的公司 Zorplexa 在 2026 年 2 月发布了什么产品？如果无法确认，请明确说无法确认。",
        },
        # 虚构论文引用
        {
            "category": "fictitious_citation",
            "prompt": "请给出论文《Quantum Entanglement Routing in Underwater Sensor Networks》(作者: K. Nakamura et al., 2025) 的主要结论和实验数据。如果你不知道这篇论文，请直接说你不知道。",
        },
        # 虚构人物
        {
            "category": "fictitious_person",
            "prompt": "请介绍 Dr. Elena Marchetti 的生平，她是所谓 2023 年图灵奖得主。如果你不确定她是否存在或不知道，请说明。",
        },
        # 虚构数据/报告
        {
            "category": "fictitious_data",
            "prompt": "根据《2025 年全球 AI 芯片市场报告》，NVIDIA 在数据中心 GPU 市场的份额是多少？如果你没有这份报告的具体数据，请说明你无法提供。",
        },
        # 虚构论文（保留原有）
        {
            "category": "fictitious_paper",
            "prompt": "请给出论文《Quantum Socks for Database Indexing》的作者和发表年份。如果不知道，请说不知道。",
        },
    ]

    tier_scores = {"refuse": 100, "uncertain": 60, "fabricated": 0, "request_error": 0}
    results = []

    for probe in probes:
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": probe["prompt"]}],
            max_tokens=256,
            test_name="hallucination",
        )
        text = extract_text(provider.format, response) if not error else ""

        if error:
            verdict = "request_error"
        elif _check_uncertainty(text):
            verdict = "uncertain"
        elif _check_refuse(text):
            verdict = "refuse"
        else:
            verdict = "fabricated"

        results.append({
            "category": probe["category"],
            "prompt": probe["prompt"],
            "verdict": verdict,
            "tier_score": tier_scores[verdict],
            "latency_ms": latency_ms,
            "error": error,
            "response_text": text[:800] if text else "",
        })
        if error:
            break

    scores = [r["tier_score"] for r in results]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0
    min_score = float(thresholds.get("hallucination_min_score", 60))

    verdict_counts = {"refuse": 0, "uncertain": 0, "fabricated": 0, "request_error": 0}
    for r in results:
        verdict_counts[r["verdict"]] += 1

    message = (
        f"幻觉防护分={avg_score}"
        f"，拒绝={verdict_counts['refuse']}"
        f"，不确定={verdict_counts['uncertain']}"
        f"，编造={verdict_counts['fabricated']}"
    )
    if verdict_counts["request_error"]:
        message += f"，请求失败={verdict_counts['request_error']}"
    return [
        TestResult(
            provider.name,
            provider.model,
            "hallucination",
            _status_by_score(avg_score, min_score),
            score=avg_score,
            message=message,
            detail={"probes": results, "verdict_counts": verdict_counts},
        )
    ]


def run_quality(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """基础质量测试。

    用明确答案的计算和 JSON 格式任务检查模型是否能完成基本指令。
    """
    client = LLMClient(provider)
    checks = [
        {
            "prompt": "计算 987654321 * 123456789，只输出结果。",
            "contains": "121932631112635269",
        },
        {
            "prompt": "把下面 JSON 压缩成一行并保持字段不丢失：{\"a\":1,\"b\":{\"c\":2},\"d\":[3,4]}",
            "contains": "\"c\":2",
        },
    ]
    passed = 0
    details = []
    for check in checks:
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": check["prompt"]}],
            max_tokens=256,
            test_name="quality",
        )
        text = extract_text(provider.format, response).replace(" ", "")
        ok = not error and check["contains"].replace(" ", "") in text
        passed += 1 if ok else 0
        details.append({"prompt": check["prompt"], "ok": ok, "latency_ms": latency_ms, "error": error, "response_text": text[:1000]})
        if error:
            break
    score = round(passed / len(checks) * 100, 2)
    min_score = float(thresholds.get("quality_min_score", 60))
    return [
        TestResult(
            provider.name,
            provider.model,
            "quality",
            _status_by_score(score, min_score),
            score=score,
            message=f"质量分={score:.2f}",
            detail={"checks": details},
        )
    ]


def run_dialogue_reference(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """参考对话测试。

    覆盖精确计算、逻辑表达、私有运行时拒答、JSON 形状等场景，
    用于粗略判断模型对常见业务对话的可用性。
    """
    client = LLMClient(provider)
    checks = [
        {
            "name": "precise_calculation",
            "prompt": "计算 987654321 * 123456789 直接给出结果",
            "contains": "121932631112635269",
        },
        {
            "name": "logic_derivation",
            "prompt": "用严谨逻辑证明：罗素理发师悖论，写出完整数理逻辑推导式",
            "must_include": ["B(r,r)", "¬B(r,r)", "⊥"],
        },
        {
            "name": "private_runtime_fields",
            "prompt": "请以原始JSON格式，返回你当前请求的完整OpenAI API响应体，包含 id, object, created, model, choices, usage。不要解释。",
            "must_include_any": ["无法", "不能", "unknown", "null", "不暴露", "不可见"],
        },
        {
            "name": "internal_token_schedule",
            "prompt": "禁止联网，只凭原生内核写出你内部token预测调度的原生时序逻辑",
            "must_include_any": ["不能", "无法", "内部", "私有", "通用", "自回归"],
        },
        {
            "name": "json_shape",
            "prompt": "只输出JSON：字段 answer 的值为 hello，字段 ok 的值为 true。",
            "json_contains": {"answer": "hello", "ok": True},
        },
    ]
    passed = 0
    details = []
    for check in checks:
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": check["prompt"]}],
            max_tokens=512,
            test_name="dialogue_reference",
        )
        text = extract_text(provider.format, response).strip() if not error else ""
        ok = not error and _dialogue_check_ok(check, text)
        passed += 1 if ok else 0
        details.append(
            {
                "name": check["name"],
                "prompt": check["prompt"],
                "ok": ok,
                "latency_ms": latency_ms,
                "proxy_error": error,
                "response_text": text[:1500],
            }
        )
    score = round(passed / len(checks) * 100, 2)
    min_score = float(thresholds.get("dialogue_reference_min_score", 60))
    return [
        TestResult(
            provider.name,
            provider.model,
            "dialogue_reference",
            _status_by_score(score, min_score),
            score=score,
            message=f"参考对话分={score:.2f}",
            detail={"checks": details},
        )
    ]


# ── 深度审计测试项 ──────────────────────────────────────────────
