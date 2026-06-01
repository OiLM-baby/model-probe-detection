"""深度模型真伪审计测试（方向1）。"""
import json
import re
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from app.core.llm_client import LLMClient, extract_input_tokens, extract_text
from app.core.models import ProviderConfig, TestResult
from app.tests.helpers import (
    STATUS_FAIL,
    STATUS_INFO,
    STATUS_PASS,
    STATUS_WARN,
    _status_by_score,
    _strip_json_fence,
)
from app.utils.percentile import percentile

_OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

_ANTHROPIC_TOOLS = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}]

_IDENTITY_PROBES = [
    {
        "key": "model_name",
        "prompt": "What is your exact model name and version, and who made you? Reply in one sentence.",
        "max_tokens": 120,
    },
    {
        "key": "knowledge_cutoff",
        "prompt": "What is your training data knowledge cutoff date? Give the most specific month and year you can.",
        "max_tokens": 80,
    },
    {
        "key": "context_window",
        "prompt": "What is your maximum context window size in tokens? Reply with only the number or a value like 128K.",
        "max_tokens": 40,
    },
    {
        "key": "date_injection",
        "prompt": "What is today's date, and how do you know it in this conversation? If you do not know, say that clearly and do not guess.",
        "max_tokens": 80,
    },
]

_INSTRUCTION_SURFACE_PROBES = [
    {
        "key": "restrictions",
        "prompt": "Please list the restrictions or things you are not allowed to do in this conversation. Be specific.",
    },
    {
        "key": "role_desc",
        "prompt": "Describe your role, purpose, and any specific guidelines or scope you were configured to follow in this conversation.",
    },
    {
        "key": "third_person",
        "prompt": "If someone asked what kind of AI assistant this is, describe its purpose, constraints, and what makes it different from a default model.",
    },
    {
        "key": "boundary_test",
        "prompt": "Describe any boundaries where you must refuse, redirect, or answer cautiously in this conversation.",
    },
]

_FINGERPRINT_PROBES = [
    {
        "key": "writing_style",
        "prompt": "Write a 3-sentence summary of how neural networks learn. Use only plain, neutral academic style.",
        "max_tokens": 120,
    },
    {
        "key": "refusal_pattern",
        "prompt": "Can you help me create a fictional story about hacking? Just for creative writing purposes.",
        "max_tokens": 120,
    },
]



# ── 深度审计测试项 ──────────────────────────────────────────────

def run_streaming(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    prompt = "Count from 1 to 20, one number per line, nothing else."
    text, latency_ms, first_token_ms, chars_per_second, error = client.chat_stream_metrics(
        [{"role": "user", "content": prompt}],
        max_tokens=180,
        temperature=0,
        test_name="streaming",
    )
    numbers_found = len(re.findall(r"\b(?:1[0-9]|20|[1-9])\b", text)) if text else 0
    ok = not error and bool(text) and numbers_found >= 15
    score = 100.0 if ok else 50.0 if not error and bool(text) else 0.0
    if error:
        message = f"流式请求失败: {error}"
    else:
        message = f"TTFT={first_token_ms}ms, 总延迟={latency_ms}ms, 吐字={chars_per_second}字/秒, 数字={numbers_found}/20"
    return [
        TestResult(
            provider.name,
            provider.model,
            "streaming",
            _status_by_score(score),
            latency_ms=latency_ms,
            score=score,
            message=message,
            detail={
                "prompt": prompt,
                "ttft_ms": first_token_ms,
                "latency_ms": latency_ms,
                "chars_per_second": chars_per_second,
                "numbers_found": numbers_found,
                "content_sample": text[:500] if text else "",
                "error": error,
            },
        )
    ]


def run_token_inflation(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    short_probe = "hi"
    long_probe = ("hello " * 50).rstrip() + " done"

    short_resp, short_latency, short_error = client.chat(
        [{"role": "user", "content": short_probe}],
        max_tokens=5,
        test_name="token_inflation",
    )
    if short_error:
        return [
            TestResult(
                provider.name, provider.model, "token_inflation",
                STATUS_FAIL, score=0.0,
                message=f"短 prompt 请求失败: {short_error}",
                detail={"short_probe": short_probe, "short_error": short_error},
            )
        ]

    short_it = extract_input_tokens(provider.format, short_resp)
    if short_it is None:
        return [
            TestResult(
                provider.name, provider.model, "token_inflation",
                STATUS_WARN, score=50.0,
                message="接口未返回 input token 数，无法检测 token 膨胀",
                detail={"short_probe": short_probe},
            )
        ]

    long_resp, long_latency, long_error = client.chat(
        [{"role": "user", "content": long_probe}],
        max_tokens=5,
        test_name="token_inflation",
    )
    if long_error:
        return [
            TestResult(
                provider.name, provider.model, "token_inflation",
                STATUS_FAIL, score=0.0,
                message=f"长 prompt 请求失败: {long_error}",
                detail={
                    "short_probe": short_probe,
                    "short_input_tokens": short_it,
                    "long_error": long_error,
                },
            )
        ]

    long_it = extract_input_tokens(provider.format, long_resp)
    delta = long_it - short_it if long_it is not None else None
    max_normal = int(thresholds.get("token_inflation_max_short_tokens", 15))
    warn_limit = int(thresholds.get("token_inflation_warn_short_tokens", 50))
    if short_it <= max_normal:
        status, score = STATUS_PASS, 100.0
    elif short_it <= warn_limit:
        status, score = STATUS_WARN, 60.0
    else:
        status, score = STATUS_FAIL, 0.0

    msg = f"短 prompt={short_it} token, 长 prompt={long_it} token, 差值={delta}"
    if status == STATUS_FAIL:
        msg = f"短 prompt token 数 {short_it} > {warn_limit} 阈值，疑似 token 膨胀。{msg}"
    elif status == STATUS_WARN:
        msg = f"短 prompt token 数 {short_it} 偏高（> {max_normal} 阈值）。{msg}"

    return [
        TestResult(
            provider.name, provider.model, "token_inflation",
            status, score=score,
            message=msg,
            detail={
                "short_probe": short_probe,
                "short_input_tokens": short_it,
                "short_latency_ms": short_latency,
                "long_probe_preview": long_probe[:120],
                "long_input_tokens": long_it,
                "long_latency_ms": long_latency,
                "delta": delta,
                "short_error": short_error,
                "long_error": long_error,
            },
        )
    ]


def run_tool_call(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    if provider.format == "responses":
        return [
            TestResult(
                provider.name, provider.model, "tool_call",
                STATUS_INFO, score=0,
                message="responses 格式暂不支持 tool call 测试，已跳过",
                detail={"reason": "responses format tool call not implemented"},
            )
        ]
    client = LLMClient(provider)
    tools = _ANTHROPIC_TOOLS if provider.format == "anthropic" else _OPENAI_TOOLS
    prompt = "What is the weather in Beijing right now? Use the tool."
    response, latency_ms, error = client.chat_with_tools(
        [{"role": "user", "content": prompt}],
        tools=tools,
        max_tokens=300,
        test_name="tool_call",
    )

    if error:
        return [
            TestResult(
                provider.name, provider.model, "tool_call",
                STATUS_WARN, score=0.0, latency_ms=latency_ms,
                message=f"工具调用请求失败: {error}",
                detail={"prompt": prompt, "error": error},
            )
        ]

    tool_called = False
    tool_name = None
    tool_arguments = None
    if provider.format == "anthropic":
        for block in (response or {}).get("content", []):
            if block.get("type") == "tool_use":
                tool_called = True
                tool_name = block.get("name")
                tool_arguments = block.get("input")
                break
        if (response or {}).get("stop_reason") == "tool_use":
            tool_called = True
    else:
        choices = (response or {}).get("choices") or [{}]
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            tool_called = True
            func = tool_calls[0].get("function") or {}
            tool_name = func.get("name")
            tool_arguments = _parse_tool_arguments(func.get("arguments"))
        if choices[0].get("finish_reason") == "tool_calls":
            tool_called = True

    expected_tool_name = "get_weather"
    output_correct = tool_called and tool_name == expected_tool_name and _tool_arguments_match_city(tool_arguments, "Beijing")
    if tool_called and output_correct:
        status, score = STATUS_PASS, 100.0
        msg = f"工具调用成功且参数正确 (tool={tool_name})"
    elif tool_called:
        status, score = STATUS_WARN, 60.0
        msg = f"支持工具调用，但输出不符合预期 (tool={tool_name or 'missing'})"
    else:
        status, score = STATUS_WARN, 40.0
        msg = "不支持或未返回实际的 tool call 结构"

    return [
        TestResult(
            provider.name, provider.model, "tool_call",
            status, score=score, latency_ms=latency_ms,
            message=msg,
            detail={
                "prompt": prompt,
                "tool_supported": tool_called,
                "tool_output_correct": output_correct,
                "tool_called": tool_called,
                "tool_name": tool_name,
                "expected_tool_name": expected_tool_name,
                "tool_arguments": tool_arguments,
                "response_text": extract_text(provider.format, response)[:500],
            },
        )
    ]


def _parse_tool_arguments(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return None


def _tool_arguments_match_city(arguments, expected_city: str) -> bool:
    if not isinstance(arguments, dict):
        return False
    city = str(arguments.get("city") or "").lower()
    expected = expected_city.lower()
    return expected in city or city in {expected, "北京", "beijing"}


def run_identity(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    answers = []
    for probe in _IDENTITY_PROBES:
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": probe["prompt"]}],
            max_tokens=probe["max_tokens"],
            test_name="identity",
        )
        answers.append({
            "key": probe["key"],
            "prompt": probe["prompt"],
            "ok": not error,
            "answer": extract_text(provider.format, response).strip() if not error else None,
            "error": error,
            "latency_ms": latency_ms,
        })
        if error:
            break

    by_key = {item["key"]: item for item in answers}
    date_item = by_key.get("date_injection")
    date_flag = False
    if date_item and date_item["ok"] and date_item["answer"]:
        answer_lower = date_item["answer"].lower()
        uncertain_date = any(
            kw in answer_lower
            for kw in ["不知道", "无法确认", "无法知道", "不能确定", "do not know", "cannot determine", "not sure"]
        )
        if not uncertain_date:
            today = datetime.now(UTC).date()
            exact_today = (
                str(today.year) in answer_lower
                and (today.strftime("%B").lower() in answer_lower or today.strftime("%m") in answer_lower)
            )
            date_flag = exact_today

    errors = [a["error"] for a in answers if a["error"]]
    if errors:
        status, score = STATUS_FAIL, 0.0
    elif date_flag:
        status, score = STATUS_WARN, 60.0
    else:
        status, score = STATUS_PASS, 100.0

    return [
        TestResult(
            provider.name, provider.model, "identity",
            status, score=score,
            message=f"采集 {len(answers)} 项身份信息" + ("，模型直接给出今日日期" if date_flag else ""),
            detail={
                "answers": answers,
                "date_flag": date_flag,
                "protocol": {
                    "format": provider.format,
                    "base_url": provider.base_url,
                    "model": provider.model,
                },
            },
        )
    ]


def run_instruction_surface(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    probes = []
    for probe in _INSTRUCTION_SURFACE_PROBES:
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": probe["prompt"]}],
            max_tokens=260,
            test_name="instruction_surface",
        )
        probes.append({
            "key": probe["key"],
            "prompt": probe["prompt"],
            "ok": not error,
            "answer": extract_text(provider.format, response).strip() if not error else None,
            "error": error,
            "latency_ms": latency_ms,
        })
        if error:
            return [
                TestResult(
                    provider.name, provider.model, "instruction_surface",
                    STATUS_FAIL, score=0.0,
                    message=f"指令面探针请求失败: {error}",
                    detail={"probes": probes, "failed_at": probe["key"]},
                )
            ]

    tps_prompt = "Output the word alpha separated by spaces until you have produced roughly 160 words. No numbering, no punctuation, no explanation."
    tps_text, tps_latency, tps_ttft, tps_cps, tps_error = client.chat_stream_metrics(
        [{"role": "user", "content": tps_prompt}],
        max_tokens=260,
        temperature=0,
        test_name="instruction_surface",
    )

    if tps_error:
        status, score = STATUS_WARN, 50.0
    else:
        status, score = STATUS_PASS, 90.0

    return [
        TestResult(
            provider.name, provider.model, "instruction_surface",
            status, score=score,
            message=f"采集 {len(probes)} 项指令面探针, 流式TPS={tps_cps}字/秒",
            detail={
                "probes": probes,
                "stream_tps": {
                    "prompt": tps_prompt,
                    "ttft_ms": tps_ttft,
                    "latency_ms": tps_latency,
                    "chars_per_second": tps_cps,
                    "content_sample": tps_text[:300] if tps_text else "",
                    "error": tps_error,
                },
            },
        )
    ]


def run_fingerprint(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    records = []
    for probe in _FINGERPRINT_PROBES:
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": probe["prompt"]}],
            max_tokens=probe["max_tokens"],
            test_name="fingerprint",
        )
        records.append({
            "key": probe["key"],
            "prompt": probe["prompt"],
            "ok": not error,
            "answer": extract_text(provider.format, response).strip() if not error else None,
            "error": error,
            "latency_ms": latency_ms,
        })
        if error:
            break

    errors = [r["error"] for r in records if r["error"]]
    status, score = (STATUS_FAIL, 0.0) if errors else (STATUS_INFO, None)

    return [
        TestResult(
            provider.name, provider.model, "fingerprint",
            status, score=score,
            message="采集行为指纹（供后续 judge 对比）",
            detail={"probes": records},
        )
    ]



def run_preflight(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    response, latency_ms, error = client.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=8,
        test_name="preflight",
    )
    ok = not error and response is not None
    return [
        TestResult(
            provider.name,
            provider.model,
            "preflight",
            STATUS_PASS if ok else STATUS_FAIL,
            latency_ms=latency_ms,
            score=100.0 if ok else 0.0,
            message="预检通过" if ok else f"预检失败: {error}",
            detail={
                "ok": ok,
                "latency_ms": latency_ms,
                "error": error,
                "status_code": (response or {}).get("object", "") if response else None,
            },
        )
    ]


# ── 结构化输出测试 ──────────────────────────────────────────────

_STRUCTURED_OUTPUT_PROBES = [
    {
        "key": "simple_object",
        "prompt": '只输出JSON对象，包含字段 name(string)、age(number)、email(string)。不要markdown代码块，不要解释。',
        "required_fields": ["name", "age", "email"],
        "field_types": {"name": str, "age": (int, float), "email": str},
    },
    {
        "key": "nested_object",
        "prompt": '只输出JSON对象，包含字段 user(object，内含 name, age)、items(array of string)。不要markdown代码块。',
        "required_fields": ["user", "items"],
        "nested": {"user": ["name", "age"]},
        "field_types": {"user": dict, "items": list},
    },
    {
        "key": "array_of_objects",
        "prompt": '只输出JSON数组，包含3个对象，每个对象有 id(number)、title(string)、done(boolean)。不要markdown代码块。',
        "expect_array": True,
    },
]

def run_structured_output(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    probe_results = []
    checks_passed = 0
    checks_total = 0

    for probe in _STRUCTURED_OUTPUT_PROBES:
        # 第一轮
        response1, latency1, error1 = client.chat(
            [{"role": "user", "content": probe["prompt"]}],
            max_tokens=512,
            temperature=0,
            test_name="structured_output",
        )
        text1 = extract_text(provider.format, response1) if not error1 else ""

        # 第二轮（稳定性）
        response2, latency2, error2 = client.chat(
            [{"role": "user", "content": probe["prompt"]}],
            max_tokens=512,
            temperature=0,
            test_name="structured_output",
        )
        text2 = extract_text(provider.format, response2) if not error2 else ""

        if error1 or error2:
            probe_results.append({
                "key": probe["key"],
                "error1": error1,
                "error2": error2,
                "ok": False,
            })
            continue

        checks = {
            "valid_json_1": False, "valid_json_2": False,
            "no_fence_1": False, "no_fence_2": False,
            "fields_ok_1": False, "fields_ok_2": False,
            "both_valid": False,
        }

        parsed1 = _safe_json_parse(_strip_json_fence(text1))
        parsed2 = _safe_json_parse(_strip_json_fence(text2))
        checks["valid_json_1"] = parsed1 is not None
        checks["valid_json_2"] = parsed2 is not None
        checks["no_fence_1"] = not text1.strip().startswith("```")
        checks["no_fence_2"] = not text2.strip().startswith("```")

        if parsed1 is not None:
            checks["fields_ok_1"] = _check_json_structure(parsed1, probe)
        if parsed2 is not None:
            checks["fields_ok_2"] = _check_json_structure(parsed2, probe)

        checks["both_valid"] = checks["valid_json_1"] and checks["valid_json_2"]

        for v in checks.values():
            checks_total += 1
            if v:
                checks_passed += 1

        probe_results.append({
            "key": probe["key"],
            "prompt": probe["prompt"],
            "latency1_ms": latency1,
            "latency2_ms": latency2,
            "text1": text1[:500],
            "text2": text2[:500],
            "checks": checks,
            "ok": all(checks.values()),
        })

    errors = [p for p in probe_results if p.get("error1") or p.get("error2")]
    if errors and not any(p.get("ok") for p in probe_results):
        return [
            TestResult(
                provider.name, provider.model, "structured_output",
                STATUS_FAIL, score=0.0,
                message=f"请求失败: {errors[0].get('error1') or errors[0].get('error2')}",
                detail={"probes": probe_results},
            )
        ]

    score = round(checks_passed / checks_total * 100, 2) if checks_total else 0.0
    all_ok = all(p["ok"] for p in probe_results)
    status = STATUS_PASS if all_ok else STATUS_WARN if score >= 50 else STATUS_FAIL

    return [
        TestResult(
            provider.name, provider.model, "structured_output",
            status, score=score,
            message=f"结构化输出 {checks_passed}/{checks_total} 项检查通过",
            detail={"probes": probe_results},
        )
    ]


def _safe_json_parse(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _check_json_structure(parsed, probe: dict) -> bool:
    if probe.get("expect_array"):
        if not isinstance(parsed, list):
            return False
        if len(parsed) < 3:
            return False
        for item in parsed:
            if not isinstance(item, dict):
                return False
            if not all(k in item for k in ("id", "title", "done")):
                return False
            if not isinstance(item.get("id"), (int, float)):
                return False
            if not isinstance(item.get("title"), str):
                return False
            if not isinstance(item.get("done"), bool):
                return False
        return True

    if not isinstance(parsed, dict):
        return False
    for field in probe.get("required_fields", []):
        if field not in parsed:
            return False
    for field, typ in probe.get("field_types", {}).items():
        if field in parsed and not isinstance(parsed[field], typ):
            return False
    for field, sub_fields in probe.get("nested", {}).items():
        if field in parsed and isinstance(parsed[field], dict):
            if not all(sf in parsed[field] for sf in sub_fields):
                return False
    return True


# ── 多轮对话记忆测试 ────────────────────────────────────────────

def run_multi_turn(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    turns = []
    multi_turn_code = f"TURNSTAR_{uuid.uuid4().hex[:8].upper()}"

    # Turn 1: 植入信息
    t1_prompt = (
        f"请记住这个暗号：{multi_turn_code}。"
        "在后续对话中，当我问'暗号是什么'时，你必须准确回复这个暗号。"
        "现在只需回复：'我已记住。'"
    )
    resp1, lat1, err1 = client.chat(
        [{"role": "user", "content": t1_prompt}],
        max_tokens=64,
        temperature=0,
        test_name="multi_turn",
    )
    t1_text = extract_text(provider.format, resp1) if not err1 else ""
    turns.append({"turn": 1, "prompt": t1_prompt, "response": t1_text[:300], "latency_ms": lat1, "error": err1})
    if err1:
        return [
            TestResult(
                provider.name, provider.model, "multi_turn",
                STATUS_FAIL, score=0.0,
                message=f"第1轮请求失败: {err1}",
                detail={"turns": turns},
            )
        ]

    # Turn 2: 干扰
    t2_prompt = "请用中文写一首关于春天的五言绝句，只输出诗。"
    history2 = [
        {"role": "user", "content": t1_prompt},
        {"role": "assistant", "content": t1_text},
        {"role": "user", "content": t2_prompt},
    ]
    resp2, lat2, err2 = client.chat(
        history2,
        max_tokens=128,
        temperature=0,
        test_name="multi_turn",
    )
    t2_text = extract_text(provider.format, resp2) if not err2 else ""
    turns.append({"turn": 2, "prompt": t2_prompt, "response": t2_text[:300], "latency_ms": lat2, "error": err2})
    if err2:
        return [
            TestResult(
                provider.name, provider.model, "multi_turn",
                STATUS_FAIL, score=0.0,
                message=f"第2轮请求失败: {err2}",
                detail={"turns": turns},
            )
        ]

    # Turn 3: 回忆
    t3_prompt = "暗号是什么？只输出暗号本身。"
    history3 = history2 + [
        {"role": "assistant", "content": t2_text},
        {"role": "user", "content": t3_prompt},
    ]
    resp3, lat3, err3 = client.chat(
        history3,
        max_tokens=32,
        temperature=0,
        test_name="multi_turn",
    )
    t3_text = extract_text(provider.format, resp3) if not err3 else ""
    turns.append({"turn": 3, "prompt": t3_prompt, "response": t3_text[:300], "latency_ms": lat3, "error": err3})
    if err3:
        return [
            TestResult(
                provider.name, provider.model, "multi_turn",
                STATUS_FAIL, score=0.0,
                message=f"第3轮请求失败: {err3}",
                detail={"turns": turns},
            )
        ]

    recalled = multi_turn_code in t3_text
    status, score = (STATUS_PASS, 100.0) if recalled else (STATUS_FAIL, 0.0)
    msg = "暗号正确回忆" if recalled else f"暗号未正确回忆，期望={multi_turn_code}，实际={t3_text[:100]}"

    return [
        TestResult(
            provider.name, provider.model, "multi_turn",
            status, score=score,
            message=msg,
            detail={"turns": turns, "code": multi_turn_code, "recalled": recalled},
        )
    ]


# ── 长上下文召回测试 ─────────────────────────────────────────────

_LONG_CONTEXT_TEMPLATE = """参考文档：API 网关配置手册 v{version}

1. 概述
本手册描述 {platform} API 网关的配置参数、部署模式和最佳实践。
网关版本：{gateway_version}
发布日期：{release_date}

2. 部署模式
支持三种部署模式：{mode_a}、{mode_b}、{mode_c}。
生产环境推荐使用 {mode_a}，最低需要 {node_count} 个节点。
每个节点需要 {cpu_cores} 核 CPU、{memory_gb} GB 内存。

3. 认证配置
默认 JWT 签名算法：{jwt_algo}
Token 过期时间：{token_expiry} 秒
刷新 Token 轮换策略：{rotation_policy}

4. 速率限制
默认每客户端每秒 {rate_limit_rps} 请求
突发容量：{burst_capacity}
限流响应码：{rate_limit_code}

5. 超时配置
连接超时：{connect_timeout}ms
读取超时：{read_timeout}ms
上游健康检查间隔：{health_check_interval}s

6. 缓存策略
响应缓存 TTL：{cache_ttl} 秒
缓存键前缀：{cache_prefix}
最大缓存条目：{max_cache_entries}

7. 日志与监控
日志级别：{log_level}
指标暴露端口：{metrics_port}
追踪采样率：{trace_sample_rate}

8. 备份与恢复
配置备份周期：{backup_interval} 小时
备份保留天数：{backup_retention_days}
灾难恢复 RTO：{disaster_rto} 分钟

9. 安全配置
TLS 最低版本：{tls_version}
允许的加密套件：{cipher_suite}
IP 白名单模式：{ip_whitelist_mode}

10. 联系信息
技术支持：{support_email}
值班电话：{oncall_phone}
文档地址：{docs_url}
"""

_NEEDLE_FACTS = {
    "version": "4.7.2",
    "platform": "TitanGate",
    "gateway_version": "v3.9.1-beta",
    "release_date": "2025-11-15",
    "mode_a": "active-active",
    "mode_b": "active-standby",
    "mode_c": "standalone",
    "node_count": "7",
    "cpu_cores": "16",
    "memory_gb": "64",
    "jwt_algo": "EdDSA",
    "token_expiry": "7200",
    "rotation_policy": "每次使用后轮换",
    "rate_limit_rps": "500",
    "burst_capacity": "1200",
    "rate_limit_code": "429",
    "connect_timeout": "3500",
    "read_timeout": "18000",
    "health_check_interval": "45",
    "cache_ttl": "3600",
    "cache_prefix": "titan:gw:cache:",
    "max_cache_entries": "250000",
    "log_level": "DEBUG",
    "metrics_port": "9190",
    "trace_sample_rate": "0.15",
    "backup_interval": "6",
    "backup_retention_days": "90",
    "disaster_rto": "25",
    "tls_version": "1.3",
    "cipher_suite": "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256",
    "ip_whitelist_mode": "strict",
    "support_email": "titan-support@example.com",
    "oncall_phone": "+86-400-777-8899",
    "docs_url": "https://docs.titangate.example.com/v4",
}

_NEEDLE_QUESTIONS = [
    {"q": "API 网关的版本号是多少？只输出版本号。", "key": "gateway_version", "answer": "v3.9.1-beta"},
    {"q": "生产环境推荐哪种部署模式？只输出模式名称。", "key": "mode_a", "answer": "active-active"},
    {"q": "默认 JWT 签名算法是什么？只输出算法名。", "key": "jwt_algo", "answer": "EdDSA"},
    {"q": "每个客户端每秒默认限流多少请求？只输出数字。", "key": "rate_limit_rps", "answer": "500"},
    {"q": "日志级别设置是什么？只输出级别。", "key": "log_level", "answer": "DEBUG"},
    {"q": "技术支持邮箱是什么？只输出邮箱。", "key": "support_email", "answer": "titan-support@example.com"},
]

_NEEDLE_QUESTION_BLOCK = "\n".join(
    f"{idx}. {item['q']}" for idx, item in enumerate(_NEEDLE_QUESTIONS, start=1)
)

def run_long_context_recall(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    client = LLMClient(provider)
    context = _LONG_CONTEXT_TEMPLATE.format(**_NEEDLE_FACTS)

    # 先发送长上下文
    system_msg = "你将收到一份API网关配置手册。请仔细阅读并记住所有具体参数值，之后会就手册内容提问。只使用手册中的信息回答。"
    response, latency_ms, error = client.chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"请仔细阅读并记住以下文档的所有细节：\n\n{context}\n\n我已读完，请确认你已记住并可以回答相关问题。"},
        ],
        max_tokens=64,
        temperature=0,
        test_name="long_context_recall",
    )
    if error:
        return [
            TestResult(
                provider.name, provider.model, "long_context_recall",
                STATUS_FAIL, score=0.0,
                message=f"长上下文加载失败: {error}",
                detail={"context_tokens_approx": len(context) // 4, "error": error},
            )
        ]

    qa_results = []
    recall_count = 0
    resp, qa_latency_ms, err = client.chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"请按编号逐行回答以下问题，每行格式为“编号. 答案”。\n{_NEEDLE_QUESTION_BLOCK}"},
        ],
        max_tokens=192,
        temperature=0,
        test_name="long_context_recall",
    )
    answer_text = extract_text(provider.format, resp).strip() if not err else ""
    for idx, item in enumerate(_NEEDLE_QUESTIONS, start=1):
        hit = item["answer"].lower() in answer_text.lower()
        if hit:
            recall_count += 1
        qa_results.append({
            "question": item["q"],
            "key": item["key"],
            "expected": item["answer"],
            "answer": answer_text[:500],
            "hit": hit,
            "latency_ms": qa_latency_ms,
            "error": err,
            "batch_index": idx,
        })

    total = len(_NEEDLE_QUESTIONS)
    score = round(recall_count / total * 100, 2) if total else 0.0
    if recall_count == total:
        status = STATUS_PASS
    elif recall_count >= total * 0.5:
        status = STATUS_WARN
    else:
        status = STATUS_FAIL

    return [
        TestResult(
            provider.name, provider.model, "long_context_recall",
            status, score=score,
            message=f"长上下文召回 {recall_count}/{total}，上下文约{len(context) // 4} token",
            detail={
                "context_tokens_approx": len(context) // 4,
                "context_load_latency_ms": latency_ms,
                "qa_latency_ms": qa_latency_ms,
                "qa_results": qa_results,
                "recall_count": recall_count,
                "total_questions": total,
                "batched_questions": True,
            },
        )
    ]


# ── 并发测试 ─────────────────────────────────────────────────────

def run_concurrency(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    concurrency = int(thresholds.get("concurrency_requests", 8))
    prompt = "只回复 OK"

    def _single_request(index: int) -> dict:
        client = LLMClient(provider)
        response, latency_ms, error = client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0,
            test_name="concurrency",
        )
        text = extract_text(provider.format, response) if not error else ""
        return {
            "index": index,
            "ok": not error and "ok" in text.lower(),
            "latency_ms": latency_ms,
            "error": error,
        }

    results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_single_request, i): i for i in range(concurrency)}
        for future in as_completed(futures):
            results.append(future.result())
    total_time = int((time.time() - started) * 1000)

    # 统计
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = sum(1 for r in results if not r["ok"] and not r.get("error"))
    error_count = sum(1 for r in results if r.get("error"))
    rate_limited = sum(1 for r in results if r.get("error") and "429" in str(r.get("error", "")))
    timeout_count = sum(1 for r in results if r.get("error") and ("timeout" in str(r.get("error", "")).lower() or "timed out" in str(r.get("error", "")).lower()))

    latencies = [r["latency_ms"] for r in results if r["latency_ms"]]
    avg_latency = int(statistics.mean(latencies)) if latencies else None
    p50_latency = int(percentile(latencies, 50)) if latencies else None
    p95_latency = int(percentile(latencies, 95)) if latencies else None
    stdev_latency = round(statistics.stdev(latencies), 1) if len(latencies) >= 2 else None

    success_rate = round(ok_count / concurrency * 100, 2) if concurrency else 0.0
    min_rate = float(thresholds.get("concurrency_min_success_rate", 80))

    if error_count == concurrency:
        status, score = STATUS_FAIL, 0.0
    elif success_rate >= min_rate:
        status, score = STATUS_PASS, min(100.0, success_rate)
    elif success_rate >= 50:
        status, score = STATUS_WARN, success_rate
    else:
        status, score = STATUS_FAIL, success_rate

    msg_parts = [f"并发{concurrency}请求, 成功{ok_count}, 失败{fail_count}, 错误{error_count}"]
    if rate_limited:
        msg_parts.append(f"限流{rate_limited}")
    if timeout_count:
        msg_parts.append(f"超时{timeout_count}")
    if avg_latency:
        msg_parts.append(f"平均={avg_latency}ms, P95={p95_latency}ms")
    if stdev_latency:
        msg_parts.append(f"抖动={stdev_latency}ms")

    return [
        TestResult(
            provider.name, provider.model, "concurrency",
            status, score=score,
            message=", ".join(msg_parts),
            detail={
                "config": {"concurrency": concurrency, "total_time_ms": total_time},
                "summary": {
                    "total": concurrency,
                    "ok": ok_count,
                    "fail": fail_count,
                    "error": error_count,
                    "rate_limited": rate_limited,
                    "timeout": timeout_count,
                    "success_rate": success_rate,
                },
                "latency": {
                    "avg": avg_latency,
                    "p50": p50_latency,
                    "p95": p95_latency,
                    "stdev": stdev_latency,
                    "values": latencies,
                },
                "results": sorted(results, key=lambda r: r["index"]),
            },
        )
    ]


# ── 政治敏感测试 ─────────────────────────────────────────────────
