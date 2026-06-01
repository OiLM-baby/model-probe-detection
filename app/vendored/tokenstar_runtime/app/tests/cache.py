"""缓存相关测试（随机缓存、长上下文缓存、缓存命中率）。"""
import re
import time

from app.core.llm_client import LLMClient, extract_text
from app.core.models import ProviderConfig, TestResult
from app.tests.helpers import STATUS_FAIL, STATUS_INFO, STATUS_PASS, STATUS_WARN, _extract_cache_usage
from app.utils.percentile import percentile

_CACHE_PREFIX = """\
You are reviewing a comprehensive technical architecture document.

1. The platform spans three availability zones in active-active mode with full replication.
2. JWT access tokens expire every 15 minutes and refresh tokens rotate on use.
3. Kafka is used for asynchronous events with seven-day retention and replication factor three.
4. PostgreSQL handles transactional data; Redis manages session state and rate limits.
5. GitHub Actions drives CI/CD and ArgoCD handles deployments.
6. Prometheus, Thanos, OpenTelemetry, and Jaeger provide observability.
7. All data in transit uses TLS 1.3 and data at rest uses AES-256.
8. Production rollback triggers automatically if error rate exceeds 1% within five minutes.

Repeat the above document mentally and keep it available while answering the user.
""" * 60


# ── 缓存测试 ──────────────────────────────────────────────────

def run_cache_random(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    prompt = "Generate a random UUID v4 and return only the UUID."
    uuid_pattern = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
    responses = []
    extracted = []

    for _ in range(3):
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=50,
            test_name="cache_random",
        )
        text = extract_text(provider.format, response).strip() if not error else None
        extracted_value = None
        if text:
            match = uuid_pattern.search(text)
            extracted_value = match.group(0).lower() if match else text.lower()[:64]
            extracted.append(extracted_value)
        responses.append({
            "ok": not error,
            "latency_ms": latency_ms,
            "answer": text,
            "error": error,
            "extracted_value": extracted_value,
        })
        if error:
            break

    errors = [r["error"] for r in responses if r["error"]]
    if errors:
        status, score = STATUS_FAIL, 0.0
        msg = f"请求失败: {errors[0]}"
    else:
        unique = len(set(extracted))
        if unique == len(extracted):
            status, score = STATUS_PASS, 100.0
            msg = f"3 次随机输出各不相同 (unique={unique})"
        elif unique == 1:
            status, score = STATUS_FAIL, 0.0
            msg = "3 次 UUID 输出完全相同，可能存在不当缓存"
        else:
            status, score = STATUS_WARN, 50.0
            msg = f"仅 {unique}/{len(extracted)} 次输出不同"

    return [
        TestResult(
            provider.name, provider.model, "cache_random",
            status, score=score,
            message=msg,
            detail={"responses": responses, "extracted_values": extracted},
        )
    ]


def run_ctx_cache(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    questions = [
        "How many availability zones does the platform span? Reply in one sentence.",
        "What is the JWT token expiry time mentioned in the document? Reply in one sentence.",
    ]
    requests = []
    latencies = []

    for index, question in enumerate(questions):
        response, latency_ms, error = client.chat_with_cache_system(
            [{"role": "user", "content": question}],
            system_text=_CACHE_PREFIX,
            max_tokens=30,
            test_name="ctx_cache",
        )
        latencies.append(latency_ms)
        usage = (response or {}).get("usage", {}) if not error else {}
        if provider.format == "anthropic":
            cache_create = usage.get("cache_creation_input_tokens")
            cache_read = usage.get("cache_read_input_tokens")
            has_cache_fields = cache_create is not None or cache_read is not None
        else:
            details = usage.get("prompt_tokens_details") or {}
            cache_create = None
            cache_read = details.get("cached_tokens")
            has_cache_fields = "prompt_tokens_details" in usage
        requests.append({
            "index": index + 1,
            "question": question,
            "ok": not error,
            "latency_ms": latency_ms,
            "cache_create": cache_create,
            "cache_read": cache_read,
            "has_cache_fields": has_cache_fields,
            "error": error,
        })
        if error:
            break

    errors = [r["error"] for r in requests if r["error"]]
    if errors:
        status, score = STATUS_FAIL, 0.0
        msg = f"请求失败: {errors[0]}"
    elif len(requests) >= 2:
        second = requests[1]
        if second.get("cache_read") and (second["cache_read"] or 0) > 0:
            status, score = STATUS_PASS, 100.0
            msg = f"第 2 次请求命中缓存 (cache_read={second['cache_read']})"
        elif second.get("has_cache_fields"):
            status, score = STATUS_INFO, 70.0
            msg = "缓存字段可见但未直接命中"
        else:
            status, score = STATUS_WARN, 50.0
            msg = "接口未暴露 prompt cache 字段"
    else:
        status, score = STATUS_WARN, 0.0
        msg = "缓存探测证据不足"

    speedup = round(((latencies[0] - latencies[1]) / max(latencies[0], 1)) * 100, 2) if len(latencies) >= 2 and latencies[0] > 0 else None

    return [
        TestResult(
            provider.name, provider.model, "ctx_cache",
            status, score=score,
            message=msg,
            detail={
                "prefix_tokens_approx": len(_CACHE_PREFIX) // 4,
                "requests": requests,
                "speedup_pct": speedup,
            },
        )
    ]


# ── 预检 ────────────────────────────────────────────────────────


def run_cache_hit_rate(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """系统化缓存命中率测试：warmup + 多次测量请求，统计请求级/Token 级命中率。"""
    client = LLMClient(provider)
    measured_requests_target = int(thresholds.get("cache_hit_rate_requests", 10))
    warmup_count = int(thresholds.get("cache_hit_rate_warmup", 1))
    delay = float(thresholds.get("cache_hit_rate_delay", 0.2))
    prefix_repeat = int(thresholds.get("cache_hit_rate_prefix_repeat", 1))
    total = warmup_count + measured_requests_target

    prefix = _CACHE_PREFIX * prefix_repeat
    records = []
    cache_fields_observed = False

    for i in range(total):
        measured = i >= warmup_count
        prompt = f"Cache probe number {i + 1}. Reply with: CACHE_PROBE_{i + 1}."
        response, latency_ms, error = client.chat_with_cache_system(
            [{"role": "user", "content": prompt}],
            system_text=prefix,
            max_tokens=24,
            test_name="cache_hit_rate",
        )

        if error:
            records.append({
                "index": i + 1,
                "measured": measured,
                "ok": False,
                "latency_ms": latency_ms,
                "error": error,
            })
            break

        usage = (response or {}).get("usage", {})
        cache_info = _extract_cache_usage(provider.format, response)
        request_hit = (cache_info.get("cache_read") or 0) > 0 or (cache_info.get("cached_tokens") or 0) > 0
        if cache_info.get("has_cache_fields"):
            cache_fields_observed = True

        records.append({
            "index": i + 1,
            "measured": measured,
            "ok": True,
            "latency_ms": latency_ms,
            "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
            "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
            "cache_create_tokens": cache_info.get("cache_creation"),
            "cache_read_tokens": cache_info.get("cache_read"),
            "cache_hit_tokens": cache_info.get("cached_tokens"),
            "request_hit": request_hit,
            "usage": usage,
            "error": None,
        })

        if delay > 0 and i < total - 1:
            time.sleep(delay)

    measured = [r for r in records if r["measured"] and r["ok"]]
    measured_success = len(measured)
    hit_requests = sum(1 for r in measured if r.get("request_hit"))
    failed_requests = sum(1 for r in records if r["measured"] and not r["ok"])

    request_hit_rate = round(hit_requests / measured_success * 100, 2) if measured_success else 0.0
    total_input_denominator = sum(_cache_rate_denominator(provider.format, r) for r in measured)
    total_cache_hit = sum(r.get("cache_hit_tokens") or 0 for r in measured)
    token_hit_rate = round(total_cache_hit / total_input_denominator * 100, 2) if total_input_denominator else 0.0

    measured_latencies = [r["latency_ms"] for r in measured]
    p50 = percentile(measured_latencies, 50) if measured_latencies else None
    p95 = percentile(measured_latencies, 95) if measured_latencies else None
    if failed_requests > 0 and measured_success == 0:
        status, score = STATUS_FAIL, 0.0
        msg = f"全部 {measured_requests_target} 次测量请求失败"
    elif not cache_fields_observed:
        status, score = STATUS_WARN, 50.0
        msg = "接口未暴露缓存字段，无法直接计算命中率"
    elif hit_requests > 0:
        status, score = STATUS_PASS, 90.0
        msg = f"缓存命中: {hit_requests}/{measured_success} 请求, 请求命中率={request_hit_rate}%, Token命中率={token_hit_rate}%"
    else:
        status, score = STATUS_INFO, 60.0
        msg = f"未观测到缓存命中, 请求命中率={request_hit_rate}%"

    if p50 is not None:
        msg += f", P50={int(p50)}ms, P95={int(p95)}ms" if p95 else f", P50={int(p50)}ms"

    return [
        TestResult(
            provider.name, provider.model, "cache_hit_rate",
            status, score=score,
            message=msg,
            detail={
                "config": {
                    "total_requests": total,
                    "measured_requests": measured_requests_target,
                    "warmup": warmup_count,
                    "delay": delay,
                    "prefix_repeat": prefix_repeat,
                    "prefix_tokens_approx": len(prefix) // 4,
                },
                "summary": {
                    "measured_success": measured_success,
                    "failed_requests": failed_requests,
                    "hit_requests": hit_requests,
                    "request_hit_rate": request_hit_rate,
                    "token_hit_rate": token_hit_rate,
                    "total_cache_hit_tokens": total_cache_hit,
                    "total_input_tokens_for_rate": total_input_denominator,
                    "cache_fields_observed": cache_fields_observed,
                },
                "latency": {"p50": p50, "p95": p95},
                "records": records,
            },
        )
    ]


def run_cache_official_baseline(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """官方 usage 字段基准对照：检查缓存相关字段是否按协议暴露。"""
    client = LLMClient(provider)
    requests = int(thresholds.get("cache_official_baseline_requests", 3))
    delay = float(thresholds.get("cache_official_baseline_delay", 0.2))
    prefix_repeat = int(thresholds.get("cache_official_baseline_prefix_repeat", thresholds.get("cache_hit_rate_prefix_repeat", 1)))
    prefix = _CACHE_PREFIX * prefix_repeat
    records = []

    for i in range(requests):
        prompt = f"Official cache baseline probe {i + 1}. Reply with: CACHE_BASELINE_{i + 1}."
        response, latency_ms, error = client.chat_with_cache_system(
            [{"role": "user", "content": prompt}],
            system_text=prefix,
            max_tokens=24,
            test_name="cache_official_baseline",
        )
        if error:
            records.append({
                "index": i + 1,
                "ok": False,
                "latency_ms": latency_ms,
                "error": error,
            })
            break
        usage = (response or {}).get("usage", {})
        cache_info = _extract_cache_usage(provider.format, response)
        records.append({
            "index": i + 1,
            "ok": True,
            "latency_ms": latency_ms,
            "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
            "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
            "cache_create_tokens": cache_info.get("cache_creation"),
            "cache_read_tokens": cache_info.get("cache_read"),
            "cache_hit_tokens": cache_info.get("cached_tokens"),
            "usage": usage,
            "error": None,
        })
        if delay > 0 and i < requests - 1:
            time.sleep(delay)

    baseline = _cache_official_baseline(provider.format)
    field_audit = _build_cache_field_audit(provider.format, records, baseline)
    failed = sum(1 for record in records if not record.get("ok"))
    if failed == len(records):
        status, score = STATUS_FAIL, 0.0
        msg = f"官方字段基准对照失败: {records[0].get('error') if records else '无成功请求'}"
    elif field_audit["missing_field_rate"] == 0:
        status, score = STATUS_PASS, 100.0
        msg = "官方字段基准对照通过，usage 字段完整"
    else:
        status, score = STATUS_WARN, max(0.0, 100.0 - field_audit["missing_field_rate"])
        msg = f"官方字段基准对照发现缺失字段: {', '.join(field_audit['missing_fields'])}"

    return [
        TestResult(
            provider.name,
            provider.model,
            "cache_official_baseline",
            status,
            score=score,
            message=msg,
            detail={
                "config": {
                    "requests": requests,
                    "delay": delay,
                    "prefix_repeat": prefix_repeat,
                    "prefix_tokens_approx": len(prefix) // 4,
                },
                "official_baseline": baseline,
                "field_audit": field_audit,
                "records": records,
            },
        )
    ]


def _cache_rate_denominator(fmt: str, record: dict) -> int:
    if fmt == "anthropic":
        return sum(
            int(record.get(key) or 0)
            for key in ("input_tokens", "cache_create_tokens", "cache_read_tokens")
        )
    return int(record.get("input_tokens") or 0)


def _cache_official_baseline(fmt: str) -> dict:
    if fmt == "anthropic":
        return {
            "format": "anthropic",
            "usage_path": "usage",
            "expected_usage_fields": [
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ],
            "token_fields": {
                "input": "input_tokens",
                "output": "output_tokens",
                "cache_create": "cache_creation_input_tokens",
                "cache_read": "cache_read_input_tokens",
                "cache_hit": "cache_read_input_tokens",
            },
        }
    if fmt == "responses":
        return {
            "format": "responses",
            "usage_path": "usage",
            "expected_usage_fields": [
                "input_tokens",
                "output_tokens",
                "input_tokens_details.cached_tokens",
            ],
            "token_fields": {
                "input": "input_tokens",
                "output": "output_tokens",
                "cache_hit": "input_tokens_details.cached_tokens",
            },
        }
    return {
        "format": "openai",
        "usage_path": "usage",
        "expected_usage_fields": [
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details.cached_tokens",
        ],
        "token_fields": {
            "input": "prompt_tokens",
            "output": "completion_tokens",
            "cache_hit": "prompt_tokens_details.cached_tokens",
        },
    }


def _build_cache_field_audit(fmt: str, records: list[dict], baseline: dict) -> dict:
    successful = [record for record in records if record.get("ok")]
    expected_fields = baseline["expected_usage_fields"]
    missing_counts = {field: 0 for field in expected_fields}
    missing_by_request = []

    for record in successful:
        usage = record.get("usage") or {}
        missing = [field for field in expected_fields if _nested_get(usage, field) is None]
        for field in missing:
            missing_counts[field] += 1
        if missing:
            missing_by_request.append({"index": record.get("index"), "missing_fields": missing})

    checked = len(successful)
    total_expected = checked * len(expected_fields)
    total_missing = sum(missing_counts.values())
    return {
        "checked_requests": checked,
        "expected_fields": expected_fields,
        "missing_field_counts": missing_counts,
        "missing_fields": [field for field, count in missing_counts.items() if count > 0],
        "missing_by_request": missing_by_request,
        "requests_missing_any_field": len(missing_by_request),
        "missing_field_rate": round(total_missing / total_expected * 100, 2) if total_expected else 0.0,
        "token_totals": {
            "input_tokens": sum(record.get("input_tokens") or 0 for record in successful),
            "output_tokens": sum(record.get("output_tokens") or 0 for record in successful),
            "cache_create_tokens": sum(record.get("cache_create_tokens") or 0 for record in successful),
            "cache_read_tokens": sum(record.get("cache_read_tokens") or 0 for record in successful),
            "cache_hit_tokens": sum(record.get("cache_hit_tokens") or 0 for record in successful),
        },
    }


def _nested_get(value: dict, path: str):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current.get(part)
    return current
