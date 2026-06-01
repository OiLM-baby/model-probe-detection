"""大模型接口客户端。

统一封装 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 三类接口，
测试用例只需要调用 chat()，不用关心不同中转的具体请求格式。
"""

import json
import logging
import os
import time
from typing import Any

import requests

from app.core.models import ProviderConfig
from app.core.usage import collect_usage, extract_usage

logger = logging.getLogger("tokenstar")
MAX_MODEL_TIMEOUT_SECONDS = 180
MODEL_TIMEOUT_MESSAGE = f"模型响应超过 {MAX_MODEL_TIMEOUT_SECONDS} 秒，已断开连接"
CHUNK_SIZE = 64 * 1024


def _normalize_base_url(raw: str) -> str:
    base = raw.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


class LLMClient:
    """统一的大模型请求客户端。

    ProviderConfig.format 决定使用 OpenAI Chat Completions、Responses 还是
    Anthropic Messages 协议。测试项只调用这里的 chat/chat_stream_metrics 等方法，
    不直接拼具体供应商的 HTTP 请求。
    """

    def __init__(self, provider: ProviderConfig):
        """创建 provider 专属会话，复用连接并统一规范 base_url。"""
        self.provider = provider
        self.base_url = _normalize_base_url(provider.base_url)
        self.session = requests.Session()

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0,
        stream: bool = False,
        test_name: str = "",
    ) -> tuple[dict[str, Any] | None, int, str]:
        """发送一次非流式对话请求。

        返回 (响应 JSON, 总耗时 ms, 错误文本)。成功时会抽取 usage 并放入全局
        usage 收集器，runner 稍后会把 usage 回填到对应 TestResult。
        """
        started = time.time()
        logger.info(
            "请求开始 provider=%s model=%s test=%s max_tokens=%s temperature=%s input=%s",
            self.provider.name,
            self.provider.model,
            test_name or "-",
            max_tokens,
            temperature,
            _compact(messages, 1200),
        )
        try:
            if self.provider.format == "anthropic":
                response = self._anthropic_chat(messages, max_tokens, temperature, stream)
            elif self.provider.format == "responses":
                response = self._responses_chat(messages, max_tokens, temperature, stream)
            else:
                response = self._openai_chat(messages, max_tokens, temperature, stream)
            if response:
                u = extract_usage(self.provider.format, response)
                response["_tokenstar_usage"] = u
                collect_usage(u)
            latency_ms = int((time.time() - started) * 1000)
            logger.info(
                "请求成功 provider=%s model=%s test=%s latency_ms=%s output=%s",
                self.provider.name,
                self.provider.model,
                test_name or "-",
                latency_ms,
                _compact_response(response),
            )
            return response, latency_ms, ""
        except Exception as exc:
            latency_ms = int((time.time() - started) * 1000)
            error_msg = _tag_error(exc)
            logger.warning(
                "请求失败 provider=%s model=%s test=%s latency_ms=%s error=%s",
                self.provider.name,
                self.provider.model,
                test_name or "-",
                latency_ms,
                error_msg,
            )
            return None, latency_ms, error_msg

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        max_tokens: int = 300,
        temperature: float = 0,
        test_name: str = "",
    ) -> tuple[dict[str, Any] | None, int, str]:
        """发送一次带 tools 的请求，用于 tool_call 测试项。"""
        started = time.time()
        logger.info(
            "工具调用请求 provider=%s model=%s test=%s input=%s",
            self.provider.name,
            self.provider.model,
            test_name or "-",
            _compact(messages, 800),
        )
        try:
            if self.provider.format == "anthropic":
                response = self._anthropic_tools(messages, tools, max_tokens, temperature)
            else:
                response = self._openai_tools(messages, tools, max_tokens, temperature)
            if response:
                u = extract_usage(self.provider.format, response)
                response["_tokenstar_usage"] = u
                collect_usage(u)
            latency_ms = int((time.time() - started) * 1000)
            logger.info(
                "工具调用成功 provider=%s model=%s test=%s latency_ms=%s",
                self.provider.name,
                self.provider.model,
                test_name or "-",
                latency_ms,
            )
            return response, latency_ms, ""
        except Exception as exc:
            latency_ms = int((time.time() - started) * 1000)
            logger.warning(
                "工具调用失败 provider=%s model=%s test=%s latency_ms=%s error=%s",
                self.provider.name,
                self.provider.model,
                test_name or "-",
                latency_ms,
                _tag_error(exc),
            )
            return None, latency_ms, _tag_error(exc)

    def chat_with_cache_system(
        self,
        messages: list[dict[str, str]],
        system_text: str,
        max_tokens: int = 30,
        temperature: float = 0,
        test_name: str = "",
    ) -> tuple[dict[str, Any] | None, int, str]:
        """发送缓存探测请求。

        不同协议的 prompt/cache-control 形态不同，测试项只关心返回 usage 中
        是否出现 cached/cache_read 相关字段。
        """
        started = time.time()
        logger.info(
            "缓存探测请求 provider=%s model=%s test=%s input=%s",
            self.provider.name,
            self.provider.model,
            test_name or "-",
            _compact(messages, 500),
        )
        try:
            if self.provider.format == "anthropic":
                response = self._anthropic_cache_probe(messages, system_text, max_tokens, temperature)
            elif self.provider.format == "responses":
                response = self._responses_cache_probe(messages, system_text, max_tokens, temperature)
            else:
                response = self._openai_cache_probe(messages, system_text, max_tokens, temperature)
            if response:
                u = extract_usage(self.provider.format, response)
                response["_tokenstar_usage"] = u
                collect_usage(u)
            latency_ms = int((time.time() - started) * 1000)
            logger.info(
                "缓存探测成功 provider=%s model=%s test=%s latency_ms=%s",
                self.provider.name,
                self.provider.model,
                test_name or "-",
                latency_ms,
            )
            return response, latency_ms, ""
        except Exception as exc:
            latency_ms = int((time.time() - started) * 1000)
            logger.warning(
                "缓存探测失败 provider=%s model=%s test=%s latency_ms=%s error=%s",
                self.provider.name,
                self.provider.model,
                test_name or "-",
                latency_ms,
                _tag_error(exc),
            )
            return None, latency_ms, _tag_error(exc)

    def chat_stream_metrics(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0,
        test_name: str = "",
    ) -> tuple[str, int, int | None, float, str]:
        """发送一次流式请求并采集性能指标。

        返回 (完整文本, 总耗时, 首 token 耗时, 吐字速度, 错误文本)。
        daily 的 latency 和 first_token_connectivity 都依赖这里。
        """
        started = time.time()
        logger.info(
            "流式请求开始 provider=%s model=%s test=%s max_tokens=%s temperature=%s input=%s",
            self.provider.name,
            self.provider.model,
            test_name or "-",
            max_tokens,
            temperature,
            _compact(messages, 1200),
        )
        try:
            if self.provider.format == "anthropic":
                text, latency_ms, first_token_ms = self._anthropic_stream(messages, max_tokens, temperature)
            elif self.provider.format == "responses":
                text, latency_ms, first_token_ms = self._responses_stream(messages, max_tokens, temperature)
            else:
                text, latency_ms, first_token_ms = self._openai_stream(messages, max_tokens, temperature)
            if text and first_token_ms is None:
                first_token_ms = latency_ms
            generation_ms = max(latency_ms - (first_token_ms or 0), 1)
            chars_per_second = round(len(text) / (generation_ms / 1000), 2) if text else 0.0
            logger.info(
                "流式请求成功 provider=%s model=%s test=%s latency_ms=%s first_token_ms=%s chars_per_second=%s output=%s",
                self.provider.name,
                self.provider.model,
                test_name or "-",
                latency_ms,
                first_token_ms,
                chars_per_second,
                _compact_text(text, 200),
            )
            return text, latency_ms, first_token_ms, chars_per_second, ""
        except Exception as exc:
            latency_ms = int((time.time() - started) * 1000)
            logger.warning(
                "流式请求失败 provider=%s model=%s test=%s latency_ms=%s error=%s",
                self.provider.name,
                self.provider.model,
                test_name or "-",
                latency_ms,
                _tag_error(exc),
            )
            return "", latency_ms, None, 0.0, _tag_error(exc)

    def chat_stream_probe(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 30,
        temperature: float = 0,
        test_name: str = "",
    ) -> tuple[str, int, list[dict[str, Any]], str]:
        """流式请求并采集原始 SSE 事件样本，供协议兼容矩阵检查。"""
        started = time.time()
        logger.info(
            "流式探针请求 provider=%s model=%s test=%s input=%s",
            self.provider.name, self.provider.model, test_name or "-", _compact(messages, 500),
        )
        try:
            if self.provider.format == "anthropic":
                text, latency_ms, events = self._anthropic_stream_probe(messages, max_tokens, temperature)
            elif self.provider.format == "responses":
                text, latency_ms, events = self._responses_stream_probe(messages, max_tokens, temperature)
            else:
                text, latency_ms, events = self._openai_stream_probe(messages, max_tokens, temperature)
            logger.info(
                "流式探针成功 provider=%s model=%s test=%s latency_ms=%s events=%s",
                self.provider.name, self.provider.model, test_name or "-", latency_ms, len(events),
            )
            return text, latency_ms, events, ""
        except Exception as exc:
            latency_ms = int((time.time() - started) * 1000)
            logger.warning(
                "流式探针失败 provider=%s model=%s test=%s latency_ms=%s error=%s",
                self.provider.name, self.provider.model, test_name or "-", latency_ms, _tag_error(exc),
            )
            return "", latency_ms, [], _tag_error(exc)

    def _headers(self) -> dict[str, str]:
        """生成基础请求头，并合并 provider.extra_headers。"""
        headers = {"Content-Type": "application/json"}
        headers.update(self.provider.extra_headers)
        return headers

    def _request_timeout(self) -> int:
        """限制单次模型请求最大超时时间，避免巡检任务被单个中转拖死。"""
        return min(int(self.provider.timeout or MAX_MODEL_TIMEOUT_SECONDS), MAX_MODEL_TIMEOUT_SECONDS)

    def _check_elapsed_timeout(self, started: float) -> None:
        """流式读取过程中主动检查总耗时，超过上限则中断。"""
        if time.time() - started > MAX_MODEL_TIMEOUT_SECONDS:
            raise TimeoutError(MODEL_TIMEOUT_MESSAGE)

    def _openai_chat(self, messages, max_tokens, temperature, stream):
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        payload = {
            "model": self.provider.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        return self._post_json_with_parameter_retry(url, headers, payload)

    def _responses_chat(self, messages, max_tokens, temperature, stream):
        url = f"{self.base_url}/v1/responses"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        input_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        payload = {
            "model": self.provider.model,
            "input": input_text,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        return self._post_json_with_parameter_retry(url, headers, payload)

    def _anthropic_chat(self, messages, max_tokens, temperature, stream):
        url = f"{self.base_url}/v1/messages"
        headers = self._headers()
        headers["x-api-key"] = self.provider.api_key
        headers.setdefault("anthropic-version", "2023-06-01")
        payload = {
            "model": self.provider.model,
            "messages": [m for m in messages if m["role"] != "system"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        if system_messages:
            payload["system"] = "\n".join(system_messages)
        return self._post_json_with_parameter_retry(url, headers, payload)

    def _anthropic_tools(self, messages, tools, max_tokens, temperature):
        url = f"{self.base_url}/v1/messages"
        headers = self._headers()
        headers["x-api-key"] = self.provider.api_key
        headers.setdefault("anthropic-version", "2023-06-01")
        payload = {
            "model": self.provider.model,
            "messages": [m for m in messages if m["role"] != "system"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
        }
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        if system_messages:
            payload["system"] = "\n".join(system_messages)
        return self._post_json_with_parameter_retry(url, headers, payload)

    def _openai_tools(self, messages, tools, max_tokens, temperature):
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        payload = {
            "model": self.provider.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
        }
        return self._post_json_with_parameter_retry(url, headers, payload)

    def _anthropic_cache_probe(self, messages, system_text, max_tokens, temperature):
        url = f"{self.base_url}/v1/messages"
        headers = self._headers()
        headers["x-api-key"] = self.provider.api_key
        headers.setdefault("anthropic-version", "2023-06-01")
        payload = {
            "model": self.provider.model,
            "messages": [m for m in messages if m["role"] != "system"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        }
        return self._post_json_with_parameter_retry(url, headers, payload)

    def _openai_cache_probe(self, messages, system_text, max_tokens, temperature):
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        payload = {
            "model": self.provider.model,
            "messages": [{"role": "system", "content": system_text}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._post_json_with_parameter_retry(url, headers, payload)

    def _responses_cache_probe(self, messages, system_text, max_tokens, temperature):
        url = f"{self.base_url}/v1/responses"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        payload = {
            "model": self.provider.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_text}],
                },
                *[
                    {
                        "role": msg["role"],
                        "content": [{"type": "input_text", "text": msg["content"]}],
                    }
                    for msg in messages
                    if msg.get("role") != "system"
                ],
            ],
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._post_json_with_parameter_retry(url, headers, payload)

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        with self.session.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self._request_timeout(),
            stream=True,
        ) as response:
            body_parts = []
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE, decode_unicode=True):
                if chunk:
                    body_parts.append(chunk)
                self._check_elapsed_timeout(started)
            response_text = "".join(body_parts)
            try:
                resp_data = json.loads(response_text) if response_text else {}
            except ValueError as err:
                if response.ok:
                    raw_preview = response_text[:300]
                    raise RuntimeError(f"HTTP {response.status_code} [non_json]: 非 JSON 响应，Content-Type={response.headers.get('Content-Type', 'unknown')}, body={raw_preview}") from err
                raise RuntimeError(f"HTTP {response.status_code} [{_error_kind(response.status_code)}]: {response_text[:300]}") from err
            if not response.ok:
                if isinstance(resp_data.get("error"), dict):
                    message = resp_data["error"].get("message", response_text[:300])
                else:
                    message = response_text[:300]
                raise RuntimeError(f"HTTP {response.status_code} [{_error_kind(response.status_code)}]: {message}")
            return resp_data

    def _openai_stream(self, messages, max_tokens, temperature):
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        payload = {
            "model": self.provider.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        return self._post_stream_with_parameter_retry(url, headers, payload, _extract_openai_stream_text)

    def _post_json_with_parameter_retry(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_with_parameter_retry(lambda current: self._post_json(url, headers, current), payload)

    def _post_stream_with_parameter_retry(self, url: str, headers: dict[str, str], payload: dict[str, Any], extractor):
        return self._post_with_parameter_retry(lambda current: self._post_stream(url, headers, current, extractor), payload)

    def _post_with_parameter_retry(self, sender, payload: dict[str, Any]):
        current_payload = dict(payload)
        for _ in range(3):
            try:
                return sender(current_payload)
            except RuntimeError as exc:
                retry_payload = _parameter_retry_payload(exc, current_payload)
                if retry_payload is None:
                    raise
                current_payload = retry_payload
        return sender(current_payload)

    def _responses_stream(self, messages, max_tokens, temperature):
        url = f"{self.base_url}/v1/responses"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        input_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        payload = {
            "model": self.provider.model,
            "input": input_text,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        return self._post_stream(url, headers, payload, _extract_responses_stream_text)

    def _anthropic_stream(self, messages, max_tokens, temperature):
        url = f"{self.base_url}/v1/messages"
        headers = self._headers()
        headers["x-api-key"] = self.provider.api_key
        headers.setdefault("anthropic-version", "2023-06-01")
        payload = {
            "model": self.provider.model,
            "messages": [m for m in messages if m["role"] != "system"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        if system_messages:
            payload["system"] = "\n".join(system_messages)
        return self._post_stream(url, headers, payload, _extract_anthropic_stream_text)

    def _post_stream(self, url: str, headers: dict[str, str], payload: dict[str, Any], extractor):
        started = time.time()
        text_parts = []
        first_token_ms = None
        with self.session.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self._request_timeout(),
            stream=True,
        ) as response:
            if not response.ok:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = {"raw_text": response.text[:1000]}
                message = error_payload.get("error", {}).get("message") if isinstance(error_payload.get("error"), dict) else response.text[:300]
                raise RuntimeError(f"HTTP {response.status_code} [{_error_kind(response.status_code)}]: {message}")
            for raw_line in response.iter_lines(decode_unicode=True):
                self._check_elapsed_timeout(started)
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                _raise_for_stream_error_event(event)
                text = extractor(event)
                if text:
                    if first_token_ms is None:
                        first_token_ms = int((time.time() - started) * 1000)
                    text_parts.append(text)
        latency_ms = int((time.time() - started) * 1000)
        return "".join(text_parts), latency_ms, first_token_ms


    def _openai_stream_probe(self, messages, max_tokens, temperature):
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        payload = {
            "model": self.provider.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        return self._post_stream_probe(url, headers, payload, _extract_openai_stream_text)

    def _responses_stream_probe(self, messages, max_tokens, temperature):
        url = f"{self.base_url}/v1/responses"
        headers = self._headers()
        headers["Authorization"] = f"Bearer {self.provider.api_key}"
        input_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        payload = {
            "model": self.provider.model,
            "input": input_text,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        return self._post_stream_probe(url, headers, payload, _extract_responses_stream_text)

    def _anthropic_stream_probe(self, messages, max_tokens, temperature):
        url = f"{self.base_url}/v1/messages"
        headers = self._headers()
        headers["x-api-key"] = self.provider.api_key
        headers.setdefault("anthropic-version", "2023-06-01")
        payload = {
            "model": self.provider.model,
            "messages": [m for m in messages if m["role"] != "system"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        if system_messages:
            payload["system"] = "\n".join(system_messages)
        return self._post_stream_probe(url, headers, payload, _extract_anthropic_stream_text)

    def _post_stream_probe(self, url, headers, payload, extractor):
        started = time.time()
        text_parts = []
        events: list[dict[str, Any]] = []
        with self.session.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self._request_timeout(),
            stream=True,
        ) as response:
            if not response.ok:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = {"raw_text": response.text[:1000]}
                message = error_payload.get("error", {}).get("message") if isinstance(error_payload.get("error"), dict) else response.text[:300]
                raise RuntimeError(f"HTTP {response.status_code} [{_error_kind(response.status_code)}]: {message}")
            for raw_line in response.iter_lines(decode_unicode=True):
                self._check_elapsed_timeout(started)
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                _raise_for_stream_error_event(event)
                events.append(event)
                text = extractor(event)
                if text:
                    text_parts.append(text)
        latency_ms = int((time.time() - started) * 1000)
        return "".join(text_parts), latency_ms, events


def _extract_openai_stream_text(event: dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


def _extract_responses_stream_text(event: dict[str, Any]) -> str:
    if event.get("type") == "response.output_text.delta":
        return event.get("delta") or ""
    if event.get("type") in {"response.content_part.delta", "response.output_item.delta"}:
        delta = event.get("delta") or {}
        if isinstance(delta, dict):
            return delta.get("text") or delta.get("content") or ""
    return ""


def _extract_anthropic_stream_text(event: dict[str, Any]) -> str:
    if event.get("type") == "content_block_delta":
        delta = event.get("delta") or {}
        return delta.get("text") or ""
    return ""


def _raise_for_stream_error_event(event: dict[str, Any]) -> None:
    """Some providers return HTTP 200 and then stream an error event."""
    error = event.get("error")
    if error:
        raise RuntimeError(f"HTTP 200 [unknown]: {_stream_error_message(error)}")

    event_type = str(event.get("type") or "").lower()
    if event_type in {"error", "response.failed", "response.error"}:
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        nested_error = response.get("error") or event.get("delta") or event.get("message") or event
        raise RuntimeError(f"HTTP 200 [unknown]: {_stream_error_message(nested_error)}")


def _stream_error_message(error: Any) -> str:
    if isinstance(error, dict):
        for key in ("message", "error", "type", "code"):
            value = error.get(key)
            if isinstance(value, str) and value:
                return value
        return json.dumps(error, ensure_ascii=False)[:300]
    return str(error)[:300]


def _should_retry_without_thinking(exc: RuntimeError, payload: dict[str, Any]) -> bool:
    if payload.get("enable_thinking") is False:
        return False
    msg = str(exc).lower()
    return "enable_thinking" in msg and "false" in msg


def _should_retry_without_temperature(exc: RuntimeError, payload: dict[str, Any]) -> bool:
    if "temperature" not in payload:
        return False
    msg = str(exc).lower()
    return "temperature" in msg and ("deprecated" in msg or "not supported" in msg or "unsupported" in msg)


def _parameter_retry_payload(exc: RuntimeError, payload: dict[str, Any]) -> dict[str, Any] | None:
    if _should_retry_without_thinking(exc, payload):
        retry_payload = dict(payload)
        retry_payload["enable_thinking"] = False
        return retry_payload
    if _should_retry_without_temperature(exc, payload):
        retry_payload = dict(payload)
        retry_payload.pop("temperature", None)
        return retry_payload
    return None


def extract_text(provider_format: str, response: dict[str, Any] | None) -> str:
    if not response:
        return ""
    if provider_format == "anthropic":
        parts = response.get("content") or []
        return "\n".join(part.get("text", "") for part in parts if part.get("type") == "text")
    if provider_format == "responses":
        if response.get("output_text"):
            return response.get("output_text") or ""
        texts = []
        for item in response.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") in {"output_text", "text"}:
                    texts.append(content.get("text", ""))
        return "\n".join(texts)
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
        return text.strip()
    return ""


def extract_input_tokens(provider_format: str, response: dict[str, Any] | None) -> int | None:
    if not response:
        return None
    usage = response.get("usage") or {}
    if provider_format == "anthropic":
        return usage.get("input_tokens")
    return usage.get("prompt_tokens")


def _error_kind(status_code: int) -> str:
    """把 HTTP 状态码归类为可匹配的错误标记。"""
    if status_code in (401, 403):
        return "auth"
    if status_code == 402:
        return "payment"
    if status_code == 429:
        return "rate_limit"
    if 500 <= status_code < 600:
        return "server"
    return "unknown"


def _tag_error(exc: Exception) -> str:
    """给异常消息加错误标记前缀，方便 runner 精确匹配。"""
    msg = str(exc)
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return f"[timeout]: 连接超时: {msg}"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return f"[timeout]: 响应读取超时: {msg}"
    if isinstance(exc, requests.exceptions.Timeout):
        return f"[timeout]: {msg}"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"[network]: {msg}"
    return msg  # RuntimeError 已含 [auth]/[rate_limit]/[server]/[non_json] 标记


def _compact(value: Any, limit: int) -> str:
    # 默认开启脱敏：日志只记录类型和长度，不记录请求/响应内容；
    # 默认开启脱敏；只有明确设置 0/false/off/no 才关闭，空字符串仍按默认开启。
    redact_setting = os.environ.get("TOKENSTAR_LOG_REDACT")
    redact_enabled = True if redact_setting is None else redact_setting.strip().lower() not in {"0", "false", "off", "no"}
    if redact_enabled:
        if isinstance(value, list):
            return f"<list len={len(value)}>"
        if isinstance(value, dict):
            return f"<dict keys={list(value.keys())[:10]}>"
        return f"<{type(value).__name__} len={len(str(value))}>"
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _compact_response(response: Any) -> str:
    """安全记录响应：只输出 keys + usage 摘要，不记录完整内容。"""
    if not isinstance(response, dict):
        return f"<response type={type(response).__name__}>"
    keys = list(response.keys())[:10]
    usage = response.get("usage") or {}
    summary = f"<response keys={keys}"
    if usage:
        filtered = {key: val for key, val in usage.items() if val}
        summary += f" usage={filtered}"
    return summary + ">"


def _compact_text(text: str, limit: int) -> str:
    """安全记录文本：只输出长度 + 前 N 字符。"""
    if not text:
        return "<text len=0>"
    preview = text[:limit]
    return f"<text len={len(text)}> {json.dumps(preview, ensure_ascii=False)}"
