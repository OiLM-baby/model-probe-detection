"""API 协议兼容矩阵测试。

验证中转返回的响应格式是否严格符合 OpenAI / Anthropic / Responses 协议规范。
所有测试项均为纯格式检查，不关心语义内容。
"""


from app.core.llm_client import LLMClient, extract_text
from app.core.models import ProviderConfig, TestResult
from app.tests.helpers import STATUS_FAIL, STATUS_INFO, STATUS_PASS, STATUS_WARN

_SIMPLE_PROMPT = [{"role": "user", "content": "说：你好"}]


def _check_keys(actual: dict, required: list, path: str) -> list[str]:
    """检查嵌套结构是否包含所有必需 key/index，返回缺失列表。

    required 元素可以是:
    - str: dict key
    - int: list index
    - tuple: 多个候选项（str 或 int），匹配第一个成功的
    """
    missing = []
    current = actual
    for key in required:
        segment = str(key) if not isinstance(key, tuple) else "/".join(str(k) for k in key)
        if isinstance(key, tuple):
            found = False
            for alt in key:
                nxt = _navigate(current, alt)
                if nxt is not None:
                    current = nxt
                    found = True
                    break
            if not found:
                missing.append(f"{path}.{segment}" if path else segment)
                return missing
        else:
            nxt = _navigate(current, key)
            if nxt is None:
                missing.append(f"{path}.{segment}" if path else segment)
                return missing
            current = nxt
    return missing


def _navigate(container, key):
    """安全地按 key (str → dict, int → list) 访问下一层，失败返回 None。"""
    if isinstance(key, int):
        if isinstance(container, list) and 0 <= key < len(container):
            return container[key]
        return None
    if isinstance(container, dict) and key in container:
        return container[key]
    return None


def run_protocol_text_shape(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """验证非流式响应文本字段形状符合协议。"""
    client = LLMClient(provider)
    response, latency_ms, error = client.chat(_SIMPLE_PROMPT, max_tokens=30, test_name="protocol_text_shape")

    if error:
        return [TestResult(provider.name, provider.model, "protocol_text_shape",
                           STATUS_FAIL, score=0.0, message=f"请求失败: {error}",
                           detail={"error": error})]

    text = extract_text(provider.format, response)
    if not text:
        return [TestResult(provider.name, provider.model, "protocol_text_shape",
                           STATUS_FAIL, score=0.0, message="响应中未找到文本内容",
                           detail={"response_keys": list(response.keys())})]

    fmt = provider.format
    checks = []
    if fmt == "openai":
        missing = _check_keys(response, ["choices", 0, "message", "content"], "root")
        checks.append(("choices[0].message.content", not missing, str(missing) if missing else ""))
    elif fmt == "anthropic":
        missing = _check_keys(response, ["content", 0, "text"], "root")
        checks.append(("content[0].text", not missing, str(missing) if missing else ""))
    elif fmt == "responses":
        has_output_text = bool(response.get("output_text"))
        has_output = bool(response.get("output"))
        checks.append(("output_text or output", has_output_text or has_output, ""))

    failed = [c for c in checks if not c[1]]
    return [TestResult(
        provider.name, provider.model, "protocol_text_shape",
        STATUS_FAIL if failed else STATUS_PASS,
        score=100.0 if not failed else 0.0,
        message=f"文本形状: {len(checks)}项检查, {len(failed)}失败" if failed else "文本形状符合协议",
        detail={"text": text[:200], "checks": checks, "format": fmt},
    )]


def run_protocol_usage_shape(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """验证 usage 字段存在且包含必要的 token 计数。"""
    client = LLMClient(provider)
    response, latency_ms, error = client.chat(_SIMPLE_PROMPT, max_tokens=30, test_name="protocol_usage_shape")

    if error:
        return [TestResult(provider.name, provider.model, "protocol_usage_shape",
                           STATUS_FAIL, score=0.0, message=f"请求失败: {error}",
                           detail={"error": error})]

    usage = response.get("usage") or {}
    fmt = provider.format
    checks = []

    if fmt == "anthropic":
        for key in ["input_tokens", "output_tokens"]:
            checks.append((f"usage.{key}", key in usage, str(usage.get(key, "MISSING"))))
    elif fmt == "responses":
        for key in ["input_tokens", "output_tokens"]:
            checks.append((f"usage.{key}", key in usage, str(usage.get(key, "MISSING"))))
    else:
        for key in ["prompt_tokens", "completion_tokens"]:
            checks.append((f"usage.{key}", key in usage, str(usage.get(key, "MISSING"))))

    failed = [c for c in checks if not c[1]]
    has_usage = len(checks) > 0 and len(failed) < len(checks)
    score = round((len(checks) - len(failed)) / len(checks) * 100, 1) if checks else 0

    return [TestResult(
        provider.name, provider.model, "protocol_usage_shape",
        STATUS_PASS if not failed else (STATUS_WARN if has_usage else STATUS_FAIL),
        score=score,
        message=f"Usage 形状: {len(checks)}项检查, {len(failed)}缺失" if failed else "Usage 形状符合协议",
        detail={"usage": usage, "checks": checks, "format": fmt},
    )]


def run_protocol_finish_reason(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """验证响应包含合理的 finish_reason / stop_reason。"""
    client = LLMClient(provider)
    response, latency_ms, error = client.chat(_SIMPLE_PROMPT, max_tokens=30, test_name="protocol_finish_reason")

    if error:
        return [TestResult(provider.name, provider.model, "protocol_finish_reason",
                           STATUS_FAIL, score=0.0, message=f"请求失败: {error}",
                           detail={"error": error})]

    fmt = provider.format
    reason = None
    key_path = ""

    if fmt == "anthropic":
        reason = response.get("stop_reason")
        key_path = "stop_reason"
    elif fmt == "responses":
        reason = response.get("status")
        key_path = "status"
    else:
        choices = response.get("choices") or []
        if choices:
            reason = choices[0].get("finish_reason")
            key_path = "choices[0].finish_reason"

    valid_reasons = {
        "openai": {"stop", "length", "content_filter", "tool_calls", "function_call"},
        "anthropic": {"end_turn", "max_tokens", "stop_sequence", "tool_use"},
        "responses": {"completed", "incomplete", "in_progress"},
    }

    valid = valid_reasons.get(fmt, set())
    is_valid = reason in valid if reason is not None else False

    return [TestResult(
        provider.name, provider.model, "protocol_finish_reason",
        STATUS_PASS if is_valid else STATUS_WARN,
        score=100.0 if is_valid else 50.0,
        message=f"{key_path}={reason}" + (" (有效)" if is_valid else " (未知/缺失)"),
        detail={"finish_reason": reason, "key_path": key_path, "valid_set": list(valid)},
    )]


def run_protocol_stream_shape(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """验证流式 SSE 事件形状符合协议。"""
    client = LLMClient(provider)
    text, latency_ms, events, error = client.chat_stream_probe(
        _SIMPLE_PROMPT, max_tokens=30, test_name="protocol_stream_shape",
    )

    if error:
        return [TestResult(provider.name, provider.model, "protocol_stream_shape",
                           STATUS_FAIL, score=0.0, message=f"流式请求失败: {error}",
                           detail={"error": error})]

    if not events:
        return [TestResult(provider.name, provider.model, "protocol_stream_shape",
                           STATUS_FAIL, score=0.0, message="未收到任何 SSE 事件",
                           detail={"text": text})]

    fmt = provider.format
    checks: list[tuple] = []

    if fmt == "openai":
        for i, ev in enumerate(events[:5]):
            has_choices = "choices" in ev
            has_object = ev.get("object") is not None
            checks.append((f"event[{i}].choices", has_choices))
            checks.append((f"event[{i}].object", has_object))
    elif fmt == "anthropic":
        for i, ev in enumerate(events[:5]):
            has_type = "type" in ev
            checks.append((f"event[{i}].type", has_type))
    elif fmt == "responses":
        for i, ev in enumerate(events[:5]):
            has_type = "type" in ev
            checks.append((f"event[{i}].type", has_type))

    failed = [c for c in checks if not c[1]]
    passed = len(checks) - len(failed)
    score = round(passed / len(checks) * 100, 1) if checks else 0

    return [TestResult(
        provider.name, provider.model, "protocol_stream_shape",
        STATUS_PASS if not failed else STATUS_FAIL,
        score=score,
        message=f"流式形状: {len(checks)}项检查, {len(failed)}失败, {len(events)}个事件" if failed else f"流式形状符合协议, {len(events)}个事件",
        detail={"event_count": len(events), "sample_events": events[:5], "checks": checks, "text": text[:200]},
    )]


def run_protocol_tool_call_shape(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """验证工具调用响应形状符合协议。"""
    client = LLMClient(provider)
    if provider.format == "anthropic":
        tools = [{
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }]
    else:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
    response, latency_ms, error = client.chat_with_tools(
        [{"role": "user", "content": "北京今天天气怎么样？"}],
        tools, max_tokens=100, test_name="protocol_tool_call_shape",
    )

    if error:
        return [TestResult(provider.name, provider.model, "protocol_tool_call_shape",
                           STATUS_FAIL, score=0.0, message=f"工具调用失败: {error}",
                           detail={"error": error})]

    fmt = provider.format
    checks: list[tuple] = []
    if fmt == "responses":
        output = response.get("output") or []
        tool_items = [o for o in output if o.get("type") == "function_call"]
        checks.append(("output 含 function_call", len(tool_items) > 0))
        if tool_items:
            tc = tool_items[0]
            checks.append(("function_call.name", bool(tc.get("name"))))
            checks.append(("function_call.arguments", isinstance(tc.get("arguments"), (dict, str)))
                           or bool(tc.get("arguments")))
            checks.append(("function_call.call_id", bool(tc.get("call_id"))))
    elif fmt == "anthropic":
        content = response.get("content") or []
        tool_blocks = [b for b in content if b.get("type") == "tool_use"]
        checks.append(("content 含 tool_use", len(tool_blocks) > 0))
        if tool_blocks:
            tb = tool_blocks[0]
            checks.append(("tool_use.name", bool(tb.get("name"))))
            checks.append(("tool_use.input", isinstance(tb.get("input"), dict)))
    else:
        choices = response.get("choices") or []
        if not choices:
            checks.append(("choices 非空", False))
        else:
            msg = choices[0].get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            checks.append(("message.tool_calls", len(tool_calls) > 0))
            if tool_calls:
                tc = tool_calls[0]
                checks.append(("tool_call.id", bool(tc.get("id"))))
                checks.append(("tool_call.type", tc.get("type") == "function"))
                func = tc.get("function") or {}
                checks.append(("tool_call.function.name", bool(func.get("name"))))
                checks.append(("tool_call.function.arguments", bool(func.get("arguments"))))

    failed = [c for c in checks if not c[1]]
    score = round((len(checks) - len(failed)) / len(checks) * 100, 1) if checks else 0

    return [TestResult(
        provider.name, provider.model, "protocol_tool_call_shape",
        STATUS_PASS if not failed else STATUS_WARN,
        score=score,
        message=f"工具调用形状: {len(checks)}项, {len(failed)}失败" if failed else "工具调用形状符合协议",
        detail={"checks": checks, "format": fmt, "response_keys": list(response.keys())},
    )]


def run_protocol_error_shape(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    """验证错误响应格式：故意发送缺少必填字段的请求，检查错误 JSON 结构。"""
    if not thresholds.get("protocol_error_probe_enabled", False):
        return [TestResult(
            provider.name, provider.model, "protocol_error_shape",
            STATUS_INFO, score=None,
            message="错误形状探测未启用（thresholds.protocol_error_probe_enabled=false）",
            detail={"skipped": True},
        )]
    client = LLMClient(provider)
    # 发送空 messages 触发 400 错误
    response, latency_ms, error = client.chat([], max_tokens=10, test_name="protocol_error_shape")

    if not error:
        return [TestResult(provider.name, provider.model, "protocol_error_shape",
                           STATUS_INFO, score=None,
                           message="预期应返回错误但请求成功（服务器可能过于宽容）",
                           detail={"note": "no error returned", "response_keys": list(response.keys()) if response else []})]

    checks: list[tuple] = []
    error_str = str(error).lower()

    # 检查是否包含 HTTP 状态码
    has_status = "http 4" in error_str or "http 5" in error_str
    checks.append(("含 HTTP 状态码", has_status))

    # 检查是否有错误标记
    has_tag = any(tag in error for tag in ("[auth]", "[rate_limit]", "[server]", "[non_json]", "[timeout]", "[network]"))
    checks.append(("含错误标记", has_tag))

    # 检查是否有实际错误消息体
    has_body = len(error) > 50
    checks.append(("含错误详情", has_body))

    failed = [c for c in checks if not c[1]]

    return [TestResult(
        provider.name, provider.model, "protocol_error_shape",
        STATUS_INFO, score=None,
        message=f"错误响应形状: {len(checks)}项, {len(failed)}不符合预期",
        detail={"error": error, "checks": checks},
    )]
